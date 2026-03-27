#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Queue LongBench jobs with simple GPU-aware scheduling.")
    parser.add_argument("--jobs-file", required=True, type=str)
    parser.add_argument("--repo-root", default="/mnt/home/zh/mustafar", type=str)
    parser.add_argument("--poll-seconds", default=20, type=int)
    parser.add_argument("--min-free-mib", default=28000, type=int)
    parser.add_argument("--max-total-workers", default=3, type=int)
    parser.add_argument("--process-match", default="RotateTileKV/run_longbench.py", type=str)
    return parser.parse_args()


def get_free_gpu_mem_mib() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    first_line = result.stdout.strip().splitlines()[0]
    return int(first_line)


def count_matching_workers(process_match: str) -> int:
    result = subprocess.run(
        ["pgrep", "-af", process_match],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return 0
    lines = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        if "queue_longbench_jobs.py" in line:
            continue
        if "pgrep -af" in line:
            continue
        lines.append(line)
    return len(lines)


def prune_active(active):
    remaining = []
    for job in active:
        if job["proc"].poll() is None:
            remaining.append(job)
            continue
        status = "ok" if job["proc"].returncode == 0 else f"failed({job['proc'].returncode})"
        print(f"[done] {job['name']} -> {status}")
    return remaining


def launch_job(job, repo_root: Path):
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        job["cmd"],
        cwd=repo_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    print(f"[launch] {job['name']} pid={proc.pid}")
    return {"name": job["name"], "proc": proc, "log_file": log_file}


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    jobs = json.loads(Path(args.jobs_file).read_text(encoding="utf-8"))
    pending = list(jobs)
    active = []

    print(f"[queue] loaded {len(pending)} jobs from {args.jobs_file}")
    while pending or active:
        active = prune_active(active)

        free_mem = get_free_gpu_mem_mib()
        total_workers = count_matching_workers(args.process_match)

        while pending and free_mem >= args.min_free_mib and total_workers < args.max_total_workers:
            job = pending.pop(0)
            active.append(launch_job(job, repo_root))
            time.sleep(2)
            free_mem = get_free_gpu_mem_mib()
            total_workers = count_matching_workers(args.process_match)

        print(
            f"[status] pending={len(pending)} active={len(active)} free_mem_mib={free_mem} "
            f"matching_workers={total_workers}"
        )
        time.sleep(args.poll_seconds)

    for job in active:
        job["log_file"].close()
    print("[queue] all jobs finished")


if __name__ == "__main__":
    main()
