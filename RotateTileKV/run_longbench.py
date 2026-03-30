#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metrics import (  # noqa: E402
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)
from RotateTileKV.modeling_llama_rotatetilekv import (  # noqa: E402
    RotateTileKVConfig,
    load_rotatetilekv_llama,
)

CORE_DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
FULL_DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
CHAT_EXEMPT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
LONG_BENCH_DATA_CANDIDATES = [
    REPO_ROOT / "longbench" / "data",
    Path("/data/home/szm/backup_dataset/LongBench/data"),
    Path("/data/home/szm/dataset/LongBench/data"),
]

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run RotateTileKV LongBench evaluation.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--full-longbench", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--quant-impl", type=str, default="default", choices=["default", "kivi"])
    parser.add_argument("--k-quant-scheme", type=str, default="default")
    parser.add_argument("--v-quant-scheme", type=str, default="default")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--quant-granularity",
        type=str,
        default="per-token-tile",
        choices=["per-token", "per-token-head", "per-token-tile"],
    )
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--residual-length", type=int, default=0)
    parser.add_argument("--enable-hadamard", action="store_true")
    parser.add_argument(
        "--hadamard-mode",
        type=str,
        default="none",
        choices=["none", "full", "tile"],
    )
    parser.add_argument("--hadamard-group-size", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_chat(tokenizer, prompt: str, model_name: str) -> str:
    lower = model_name.lower()
    if "llama-3" in lower and "instruct" in lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "qwen" in lower and "instruct" in lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "mistral" in lower and "instruct" in lower:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def post_process(response: str, model_name: str) -> str:
    if "xgen" in model_name:
        return response.strip().replace("Assistant:", "")
    if "internlm" in model_name:
        return response.split("<eoa>")[0]
    return response


def resolve_max_length(model_name_or_path: str, config, cli_max_length: int | None) -> int:
    if cli_max_length is not None:
        return cli_max_length

    config_paths = [
        REPO_ROOT / "config" / "model2maxlen.json",
        REPO_ROOT / "DiffSparseKV" / "config" / "model2maxlen.json",
    ]
    names = [model_name_or_path, Path(model_name_or_path).name]
    for path in config_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        for name in names:
            if name in mapping:
                return mapping[name]

    return int(getattr(config, "max_position_embeddings", 4096))


def load_prompt_config():
    with open(REPO_ROOT / "config" / "dataset2prompt.json", "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    with open(REPO_ROOT / "config" / "dataset2maxlen.json", "r", encoding="utf-8") as f:
        dataset2maxlen = json.load(f)
    return dataset2prompt, dataset2maxlen


def resolve_longbench_local_file(dataset_name: str) -> Path | None:
    env_candidates = []
    for env_key in ("LONG_BENCH_DATA_DIR", "LONGBENCH_DATA_DIR"):
        env_value = os.environ.get(env_key)
        if env_value:
            env_candidates.append(Path(env_value))

    for data_dir in [*env_candidates, *LONG_BENCH_DATA_CANDIDATES]:
        local_file = data_dir / f"{dataset_name}.jsonl"
        if local_file.exists():
            return local_file
    return None


def load_longbench_dataset(dataset_name: str):
    local_file = resolve_longbench_local_file(dataset_name)
    if local_file is not None:
        print(f"[dataset] using local file for {dataset_name}: {local_file}")
        return load_dataset("json", data_files={"test": str(local_file)}, split="test")

    print(f"[dataset] falling back to THUDM/LongBench for {dataset_name}")
    return load_dataset("THUDM/LongBench", dataset_name, split="test", trust_remote_code=True)


def score_dataset(dataset_name: str, rows):
    predictions = [row["pred"] for row in rows]
    answers = [row["answers"] for row in rows]
    all_classes = rows[0]["all_classes"] if rows else None
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, DATASET2METRIC[dataset_name](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2) if predictions else 0.0


def generate_predictions(model, tokenizer, dataset_name: str, rows, prompt_format: str, max_length: int, max_gen: int):
    device = next(model.parameters()).device
    outputs = []
    max_prompt_length = max(1, max_length - max_gen - 8)
    for json_obj in tqdm(rows, desc=dataset_name):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_prompt_length:
            half = max_prompt_length // 2
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        if dataset_name not in CHAT_EXEMPT_DATASETS:
            prompt = build_chat(tokenizer, prompt, model.config._name_or_path)

        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]
        with torch.inference_mode():
            if dataset_name == "samsum":
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]

        pred = tokenizer.decode(generated[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model.config._name_or_path)
        outputs.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return outputs


def main():
    args = parse_args()
    seed_everything(args.seed)

    quant_cfg = RotateTileKVConfig(
        enable_hadamard=args.enable_hadamard,
        hadamard_mode=args.hadamard_mode,
        hadamard_group_size=args.hadamard_group_size,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        quant_impl=args.quant_impl,
        k_quant_scheme=args.k_quant_scheme,
        v_quant_scheme=args.v_quant_scheme,
        group_size=args.group_size,
        quant_granularity=args.quant_granularity,
        tile_size=args.tile_size,
        residual_length=args.residual_length,
    )
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    tokenizer, model = load_rotatetilekv_llama(
        args.model_name_or_path,
        quant_cfg=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        local_files_only=not args.allow_download,
    )

    dataset2prompt, dataset2maxlen = load_prompt_config()
    model_name = Path(args.model_name_or_path).name
    max_length = resolve_max_length(args.model_name_or_path, model.config, args.max_length)

    if args.datasets:
        datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    else:
        datasets = FULL_DATASETS if args.full_longbench else CORE_DATASETS

    exp_name = (
        f"{model_name}_k{args.k_bits}_v{args.v_bits}_{args.quant_granularity}"
        f"_{args.hadamard_mode if args.enable_hadamard else 'nohadamard'}"
        f"_res{args.residual_length}"
        f"_qimpl{args.quant_impl}"
        f"_ks{args.k_quant_scheme}_vs{args.v_quant_scheme}_g{args.group_size}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "RotateTileKV" / "pred" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": args.model_name_or_path,
                "max_length": max_length,
                "datasets": datasets,
                "limit": args.limit,
                "quant_cfg": quant_cfg.to_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_scores = {}
    for dataset_name in datasets:
        ds = load_longbench_dataset(dataset_name)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))

        rows = generate_predictions(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            rows=ds,
            prompt_format=dataset2prompt[dataset_name],
            max_length=max_length,
            max_gen=dataset2maxlen[dataset_name],
        )

        with open(output_dir / f"{dataset_name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

        score = score_dataset(dataset_name, rows)
        all_scores[dataset_name] = score
        print(f"{dataset_name:25s}: {score:6.2f}")

    all_scores["average"] = round(sum(all_scores.values()) / max(len(all_scores), 1), 2)
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    print("-" * 60)
    print(f"{'average':25s}: {all_scores['average']:6.2f}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("WANDB_DISABLED", "true")
    main()
