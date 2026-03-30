#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Queue LongBench jobs with multi-GPU scheduling.")
    parser.add_argument("--jobs-file", required=True, type=str)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), type=str)
    parser.add_argument("--poll-seconds", default=20, type=int)
    parser.add_argument("--min-free-mib", default=22000, type=int)
    parser.add_argument("--max-total-workers", default=8, type=int)
    parser.add_argument("--gpu-ids", default="", type=str, help="Comma-separated GPU ids. Empty means all GPUs.")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        help="Rewrite path prefixes inside jobs as old_prefix=new_prefix. Can be passed multiple times.",
    )
    return parser.parse_args()


def parse_path_maps(path_maps):
    mappings = []
    for item in path_maps:
        if "=" not in item:
            raise ValueError(f"Invalid --path-map '{item}', expected old=new")
        old, new = item.split("=", 1)
        mappings.append((old, new))
    return mappings


def remap_value(value: str, mappings) -> str:
    for old, new in mappings:
        if value.startswith(old):
            return new + value[len(old) :]
    return value


def normalize_job(job, mappings):
    normalized = json.loads(json.dumps(job))
    normalized["log_path"] = remap_value(normalized["log_path"], mappings)
    normalized["cmd"] = [remap_value(token, mappings) if isinstance(token, str) else token for token in normalized["cmd"]]
    return normalized


def get_job_output_dir(job) -> Path | None:
    cmd = job["cmd"]
    if "--output-dir" not in cmd:
        return None
    idx = cmd.index("--output-dir")
    if idx + 1 >= len(cmd):
        return None
    return Path(cmd[idx + 1])


def job_is_completed(job) -> bool:
    output_dir = get_job_output_dir(job)
    return output_dir is not None and (output_dir / "result.json").exists()


def get_gpu_free_mem_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    free_mem = {}
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        gpu_index, memory_free = [part.strip() for part in line.split(",", 1)]
        free_mem[int(gpu_index)] = int(memory_free)
    return free_mem


def prune_active(active):
    remaining = []
    for job in active:
        if job["proc"].poll() is None:
            remaining.append(job)
            continue
        status = "ok" if job["proc"].returncode == 0 else f"failed({job['proc'].returncode})"
        print(f"[done] {job['name']} gpu={job['gpu_id']} -> {status}")
        job["log_file"].close()
    return remaining


def split_env_prefix(cmd):
    cmd = list(cmd)
    env_overrides = {}
    if cmd and cmd[0] == "env":
        cmd.pop(0)
        while cmd and "=" in cmd[0]:
            key, value = cmd.pop(0).split("=", 1)
            env_overrides[key] = value
    return env_overrides, cmd


def launch_job(job, repo_root: Path, gpu_slot: str):
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    env_overrides, cmd = split_env_prefix(job["cmd"])
    env = os.environ.copy()
    env.update(env_overrides)
    env["CUDA_VISIBLE_DEVICES"] = gpu_slot
    env.setdefault("LONG_BENCH_DATA_DIR", "/data/home/szm/backup_dataset/LongBench/data")
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    print(f"[launch] {job['name']} gpu={gpu_slot} pid={proc.pid}")
    return {"name": job["name"], "proc": proc, "log_file": log_file, "gpu_id": gpu_slot}


def parse_gpu_ids(raw_gpu_ids: str, detected_gpu_ids):
    if not raw_gpu_ids.strip():
        return [str(item) for item in detected_gpu_ids]
    if ";" in raw_gpu_ids:
        return [item.strip() for item in raw_gpu_ids.split(";") if item.strip()]
    return [item.strip() for item in raw_gpu_ids.split(",") if item.strip()]


def slot_gpu_ids(gpu_slot: str):
    return [int(item.strip()) for item in gpu_slot.split(",") if item.strip()]


def slot_free_mem(gpu_slot: str, free_mem_by_gpu):
    gpu_ids = slot_gpu_ids(gpu_slot)
    return min(free_mem_by_gpu.get(gpu_id, 0) for gpu_id in gpu_ids)


def slots_conflict(slot_a: str, slot_b: str):
    return bool(set(slot_gpu_ids(slot_a)) & set(slot_gpu_ids(slot_b)))


def pick_gpu(active, free_mem_by_gpu, allowed_gpu_ids, min_free_mib):
    active_slots = []
    for job in active:
        active_slots.append(job["gpu_id"])

    for gpu_id in sorted(allowed_gpu_ids, key=lambda item: slot_free_mem(item, free_mem_by_gpu), reverse=True):
        if any(slots_conflict(gpu_id, active_slot) for active_slot in active_slots):
            continue
        if slot_free_mem(gpu_id, free_mem_by_gpu) >= min_free_mib:
            return gpu_id
    return None


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    mappings = parse_path_maps(args.path_map)
    jobs = json.loads(Path(args.jobs_file).read_text(encoding="utf-8"))
    normalized_jobs = [normalize_job(job, mappings) for job in jobs]
    pending = [job for job in normalized_jobs if not job_is_completed(job)]
    active = []

    skipped = len(normalized_jobs) - len(pending)
    free_mem_by_gpu = get_gpu_free_mem_mib()
    allowed_gpu_ids = parse_gpu_ids(args.gpu_ids, sorted(free_mem_by_gpu))
    print(f"[queue] loaded {len(normalized_jobs)} jobs from {args.jobs_file}")
    print(f"[queue] skipped {skipped} completed jobs")
    print(f"[queue] allowed_gpus={allowed_gpu_ids}")
    while pending or active:
        active = prune_active(active)
        free_mem_by_gpu = get_gpu_free_mem_mib()

        while pending and len(active) < args.max_total_workers:
            gpu_id = pick_gpu(active, free_mem_by_gpu, allowed_gpu_ids, args.min_free_mib)
            if gpu_id is None:
                break
            job = pending.pop(0)
            active.append(launch_job(job, repo_root, gpu_id))
            time.sleep(2)
            free_mem_by_gpu = get_gpu_free_mem_mib()

        print(
            f"[status] pending={len(pending)} active={len(active)} "
            f"free_mem_by_gpu={{{', '.join(f'{gid}:{slot_free_mem(gid, free_mem_by_gpu)}' for gid in allowed_gpu_ids)}}}"
        )
        time.sleep(args.poll_seconds)

    for job in active:
        job["log_file"].close()
    print("[queue] all jobs finished")


if __name__ == "__main__":
    main()
