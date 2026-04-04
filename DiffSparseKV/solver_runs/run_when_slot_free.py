#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time


ACTIVE_PATTERNS = [
    "eval_diff_sparse_kv_longbench.py --model_path /home/zh/model/Llama-2-7b-hf",
    "eval_diff_sparse_kv_longbench.py --model_path /home/zh/model/Llama-2-13b-hf",
    "eval_diff_sparse_kv_longbench.py --model_path /home/zh/model/Mistral-7B-Instruct-v0.1",
    "eval_diff_sparse_kv_longbench.py --model_path /home/zh/model/Mistral-7B-v0.1",
    "eval_diff_sparse_kv_longbench.py --model_path /home/zh/model/Meta-Llama-3-8B-Instruct",
]


def gpu_free_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
    return max(vals) if vals else 0.0


def active_job_count() -> int:
    out = subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    count = 0
    for line in out.splitlines():
        if any(p in line for p in ACTIVE_PATTERNS):
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_free_mb", type=int, default=18000)
    ap.add_argument("--max_active_jobs", type=int, default=2)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if not args.command:
        raise SystemExit("missing command")

    print(f"[watch] waiting: free>={args.min_free_mb}MB and active_jobs<={args.max_active_jobs}", flush=True)
    print(f"[watch] command: {' '.join(args.command)}", flush=True)
    while True:
        free_mb = gpu_free_mb()
        active = active_job_count()
        print(f"[watch] free={free_mb:.0f}MB active_jobs={active}", flush=True)
        if free_mb >= args.min_free_mb and active <= args.max_active_jobs:
            break
        time.sleep(60)

    subprocess.run(args.command, check=True)


if __name__ == "__main__":
    main()
