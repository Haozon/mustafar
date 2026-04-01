#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/zh/miniconda3/envs/mustar/bin/python"
LCC_DIFF_RESULT = (
    ROOT
    / "solver_runs"
    / "Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.50_meta50_full_lcc_bestdiff_full"
    / "result.json"
)


def main() -> None:
    print(f"[watch] waiting for {LCC_DIFF_RESULT}", flush=True)
    while not LCC_DIFF_RESULT.exists():
        time.sleep(60)

    print("[watch] meta50 lcc diff full finished; resuming mistral qasper wide search", flush=True)
    subprocess.run(
        [PY, "solver_runs/run_mistral_qasper_wide_search.py"],
        cwd=str(ROOT),
        check=True,
    )


if __name__ == "__main__":
    main()
