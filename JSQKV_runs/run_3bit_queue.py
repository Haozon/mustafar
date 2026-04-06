#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/mnt/nas/nas_192.168.7.2/zh/mustafar")
CONDA = Path("/home/zh/miniconda3/bin/conda")
SUMMARY = ROOT / "JSQKV_runs" / "FINAL_RESULTS_SUMMARY.md"
WORK_LOG = ROOT / "JSQKV_runs" / "WORK_LOG_2026-04-04.md"
LOG_DIR = ROOT / "JSQKV_runs" / "run_logs"
OUTPUT_ROOT = ROOT / "JSQKV_runs" / "threebit_4096"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class Task:
    tag: str
    model_label: str
    model_path: str
    budget_label: str
    target_distribution: str
    sparsity_levels: str
    quant_impl: str
    k_scheme: str
    v_scheme: str
    sink_keep: int
    enable_hadamard: bool

    @property
    def result_path(self) -> Path:
        return OUTPUT_ROOT / self.tag / "result.json"

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"{self.tag}.log"


def build_tasks() -> list[Task]:
    meta = "/home/zh/model/Meta-Llama-3-8B-Instruct"
    llama2 = "/home/zh/nas/nas_10g/models/llama-2-7b"
    mistral = "/home/zh/nas/nas_10g/models/Mistral-7B-v0.1"
    qwen = "/home/zh/nas/nas_10g/models/Qwen2.5-7B-instruct"

    tasks: list[Task] = []

    def add_model(tag_prefix: str, label: str, model_path: str, sink_keep: int):
        for budget_label, dist, levels in [
            ("50", "0.0,1.0,0.0", "0.0,0.5,1.0"),
            ("70", "0.0,1.0,0.0", "0.0,0.7,1.0"),
        ]:
            tasks.append(
                Task(
                    tag=f"{tag_prefix}{budget_label}_uniformkivi_3bit_selected6_full_4096",
                    model_label=label,
                    model_path=model_path,
                    budget_label=budget_label,
                    target_distribution=dist,
                    sparsity_levels=levels,
                    quant_impl="kivi",
                    k_scheme="kivi-channel",
                    v_scheme="per-token-head",
                    sink_keep=sink_keep,
                    enable_hadamard=False,
                )
            )

        for budget_label, dist, levels in [
            ("50", "0.0,0.833333,0.166667", "0.0,0.4,1.0"),
            ("70", "0.0,0.75,0.25", "0.0,0.6,1.0"),
        ]:
            tasks.append(
                Task(
                    tag=f"{tag_prefix}{budget_label}_jsqkv_3bit_tilehad_selected6_full_4096",
                    model_label=label,
                    model_path=model_path,
                    budget_label=budget_label,
                    target_distribution=dist,
                    sparsity_levels=levels,
                    quant_impl="default",
                    k_scheme="per-token-tile",
                    v_scheme="per-token-tile",
                    sink_keep=sink_keep,
                    enable_hadamard=True,
                )
            )

    add_model("meta", "Meta-Llama-3-8B-Instruct", meta, 4)
    add_model("llama2_", "Llama-2-7B", llama2, 2)
    add_model("mistral", "Mistral-7B-v0.1", mistral, 2)
    add_model("qwen", "Qwen2.5-7B-Instruct", qwen, 2)
    return tasks


TASKS = build_tasks()


def read_result(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def gpu_status():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    status = []
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        status.append((int(idx), int(mem), int(util)))
    return status


def pick_free_gpu(reserved: set[int]) -> int | None:
    for idx, mem, util in gpu_status():
        if idx in reserved:
            continue
        if mem < 2000 and util < 20:
            return idx
    return None


def launch(task: Task, gpu: int) -> subprocess.Popen:
    cmd = [
        str(CONDA),
        "run",
        "--no-capture-output",
        "-n",
        "mustafar",
        "python",
        "JSQKV/eval_jsqkv_longbench.py",
        "--model_path",
        task.model_path,
        "--max_length",
        "4096",
        "--datasets",
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "trec",
        "lcc",
        "--output_dir",
        str(OUTPUT_ROOT),
        "--output_tag",
        task.tag,
        "--target_distribution",
        task.target_distribution,
        "--sparsity_levels",
        task.sparsity_levels,
        "--importance_mode",
        "value_aware",
        "--head_aggregation_mode",
        "max",
        "--value_sink_keep",
        str(task.sink_keep),
        "--level_2_mode",
        "evict",
        "--k_bits",
        "3",
        "--v_bits",
        "3",
        "--quant_impl",
        task.quant_impl,
        "--k_quant_scheme",
        task.k_scheme,
        "--v_quant_scheme",
        task.v_scheme,
        "--group_size",
        "128",
        "--quant_granularity",
        "per-token-tile",
        "--tile_size",
        "64",
        "--residual_length",
        "128",
    ]
    if task.enable_hadamard:
        cmd += ["--enable_hadamard", "--hadamard_mode", "tile", "--hadamard_group_size", "64"]
    else:
        cmd += ["--hadamard_mode", "none"]
    cmd += ["--run_eval"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(task.log_path, "a", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)


def append_work_log(line: str):
    with WORK_LOG.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def render_threebit_section() -> str:
    # Completed models only; unfinished cells shown as --
    rows = []

    def val(tag: str):
        data = read_result(OUTPUT_ROOT / tag / "result.json")
        return None if data is None else data["average"]

    model_specs = [
        ("Meta-Llama-3-8B-Instruct", "meta"),
        ("Llama-2-7B", "llama2"),
        ("Mistral-7B-v0.1", "mistral"),
        ("Qwen2.5-7B-Instruct", "qwen"),
    ]

    def fmt(proxy, jsqkv):
        if proxy is None or jsqkv is None:
            return "--", "--"
        delta = jsqkv - proxy
        return f"{proxy:.2f} / **{jsqkv:.2f}**", f"{delta:+.2f}"

    lines = []
    lines.append("### 3bit Cross-Model Table")
    lines.append("")
    lines.append("| Model | 50% + 3-bit | Delta | 70% + 3-bit | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, prefix in model_specs:
        cell50, d50 = fmt(val(f"{prefix}_50_uniformkivi_3bit_selected6_full_4096"), val(f"{prefix}_50_jsqkv_3bit_tilehad_selected6_full_4096"))
        cell70, d70 = fmt(val(f"{prefix}_70_uniformkivi_3bit_selected6_full_4096"), val(f"{prefix}_70_jsqkv_3bit_tilehad_selected6_full_4096"))
        lines.append(f"| {label} | {cell50} | {d50} | {cell70} | {d70} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append("- Each score cell reports `M+K proxy / JSQKV(tilehad)`.")
    lines.append("- This section is updated automatically by `run_3bit_queue.py`.")
    return "\n".join(lines)


def update_summary():
    text = SUMMARY.read_text(encoding="utf-8")
    new = render_threebit_section()
    pattern = r"<!-- THREEBIT_SECTION_START -->.*?<!-- THREEBIT_SECTION_END -->"
    repl = "<!-- THREEBIT_SECTION_START -->\n\n" + new + "\n\n<!-- THREEBIT_SECTION_END -->"
    text = re.sub(pattern, repl, text, flags=re.S)
    SUMMARY.write_text(text, encoding="utf-8")


def main():
    tasks = TASKS
    # Skip tasks with existing results
    pending = [t for t in tasks if not t.result_path.exists()]
    running: dict[subprocess.Popen, tuple[Task, int]] = {}

    append_work_log("")
    append_work_log("### 3bit Cross-Model Queue Started")
    append_work_log("- Queue scope: four models, `50%/70% + 3bit`, `M+K proxy` vs `JSQKV(tilehad)`.")
    append_work_log("- Existing Meta-Llama-3-8B 3bit results are reused; missing tasks are queued.")

    update_summary()

    while pending or running:
        # Launch on free GPUs
        reserved = {gpu for _, gpu in running.values()}
        launched = False
        while pending:
            gpu = pick_free_gpu(reserved)
            if gpu is None:
                break
            task = pending.pop(0)
            proc = launch(task, gpu)
            running[proc] = (task, gpu)
            reserved.add(gpu)
            append_work_log(f"- launched `{task.tag}` on GPU {gpu}")
            launched = True
            time.sleep(2)

        # Poll running tasks
        finished = []
        for proc, (task, gpu) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            finished.append((proc, task, gpu, ret))

        for proc, task, gpu, ret in finished:
            running.pop(proc, None)
            result = read_result(task.result_path)
            if result is not None:
                append_work_log(f"- completed `{task.tag}` on GPU {gpu}: average `{result['average']:.2f}`")
            else:
                append_work_log(f"- task `{task.tag}` on GPU {gpu} exited with code `{ret}` and no result.json")
            update_summary()

        if not launched and not finished:
            time.sleep(30)

    append_work_log("- 3bit cross-model queue finished")
    update_summary()


if __name__ == "__main__":
    main()
