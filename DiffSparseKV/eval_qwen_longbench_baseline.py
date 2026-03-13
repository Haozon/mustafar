#!/usr/bin/env python3
"""
Qwen Native LongBench Evaluation Script

This script is isolated from the MUSTAFAR/Llama-specific loading path.
It loads Qwen models through Hugging Face AutoModelForCausalLM directly.
"""

import argparse
import json
import os
import subprocess
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def build_chat(tokenizer, prompt, model_name):
    """Build chat prompt only for instruct-style models with chat template."""
    if "instruct" in model_name.lower() and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def get_pred(model, tokenizer, data, max_length, max_gen, prompt_format, dataset, device, model_name):
    """Generate predictions for one dataset."""
    preds = []

    for json_obj in tqdm(data, desc=f"Processing {dataset}"):
        prompt = prompt_format.format(**json_obj)

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        input_tokens = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input_tokens.input_ids.shape[-1]

        with torch.no_grad():
            if dataset == "samsum":
                output = model.generate(
                    **input_tokens,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length + 1,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.encode("\n", add_special_tokens=False)[-1],
                    ],
                )[0]
            else:
                output = model.generate(
                    **input_tokens,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

    return preds


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Native LongBench Evaluation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/zh/model/Qwen2.5-7B",
        help="Model path",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum context length",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
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
        ],
        help="LongBench datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pred",
        help="Base output directory",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="",
        help="Optional output subdir name",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run eval_results.py automatically after prediction generation",
    )
    return parser.parse_args()


def get_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_path.rstrip("/").split("/")[-1]
    output_subdir = args.output_subdir or f"{model_name}_{args.max_length}_qwen_native_baseline"
    output_path = os.path.join(args.output_dir, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Model type from AutoConfig: {getattr(config, 'model_type', 'unknown')}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=get_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset2prompt = {
        "narrativeqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "qasper": "{context}\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_en": "{context}\n\nQuestion: {input}\nAnswer:",
        "multifieldqa_zh": "{context}\n\n问题: {input}\n答案:",
        "hotpotqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "2wikimqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "musique": "{context}\n\nQuestion: {input}\nAnswer:",
        "dureader": "{context}\n\n问题: {input}\n答案:",
        "gov_report": "{context}\n\nQuestion: {input}\nAnswer:",
        "qmsum": "{context}\n\nQuestion: {input}\nAnswer:",
        "multi_news": "{context}\n\nQuestion: {input}\nAnswer:",
        "vcsum": "{context}\n\n问题: {input}\n答案:",
        "trec": "{input}",
        "triviaqa": "{context}\n\nQuestion: {input}\nAnswer:",
        "samsum": "{context}\n\nSummarize the above conversation.",
        "lsht": "{input}",
        "passage_retrieval_en": "{input}",
        "passage_count": "{context}\n\nHow many passages are there?",
        "passage_retrieval_zh": "{input}",
        "lcc": "{input}",
        "repobench-p": "{input}",
    }

    dataset2maxlen = {
        "narrativeqa": 100,
        "qasper": 100,
        "multifieldqa_en": 100,
        "multifieldqa_zh": 100,
        "hotpotqa": 100,
        "2wikimqa": 100,
        "musique": 100,
        "dureader": 100,
        "gov_report": 100,
        "qmsum": 100,
        "multi_news": 100,
        "vcsum": 100,
        "trec": 50,
        "triviaqa": 100,
        "samsum": 100,
        "lsht": 50,
        "passage_retrieval_en": 100,
        "passage_count": 50,
        "passage_retrieval_zh": 100,
        "lcc": 100,
        "repobench-p": 100,
    }

    try:
        with open("config/dataset2prompt.json", "r", encoding="utf-8") as f:
            dataset2prompt.update(json.load(f))
        with open("config/dataset2maxlen.json", "r", encoding="utf-8") as f:
            dataset2maxlen.update(json.load(f))
        print("Loaded dataset configs from config/*.json")
    except FileNotFoundError:
        print("Config files not found, using built-in defaults")

    print(f"Output directory: {output_path}")

    for dataset in args.datasets:
        print(f"\n{'=' * 50}")
        print(f"Processing dataset: {dataset}")
        print(f"{'=' * 50}")

        data = load_dataset("THUDM/LongBench", dataset, split="test", trust_remote_code=True)
        prompt_format = dataset2prompt.get(dataset, "{input}")
        max_gen = dataset2maxlen.get(dataset, 100)

        preds = get_pred(
            model=model,
            tokenizer=tokenizer,
            data=data,
            max_length=args.max_length,
            max_gen=max_gen,
            prompt_format=prompt_format,
            dataset=dataset,
            device=device,
            model_name=model_name,
        )

        output_file = os.path.join(output_path, f"{dataset}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
        print(f"Saved {len(preds)} predictions to {output_file}")

    run_cfg = {
        "model_path": args.model_path,
        "model_name": model_name,
        "model_type": getattr(config, "model_type", "unknown"),
        "max_length": args.max_length,
        "dtype": args.dtype,
        "datasets": args.datasets,
    }
    with open(os.path.join(output_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print("Prediction generation completed")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 50}")

    if args.run_eval:
        print("\nRunning eval_results.py ...")
        subprocess.run(
            [sys.executable, "eval_results.py", "--result_dir", output_path],
            check=False,
        )


if __name__ == "__main__":
    main()
