import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset

from common import ensure_dir, load_model_and_tokenizer, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze massive-activation tokens and attention diagonals.")
    parser.add_argument("--model-path", type=str, default="/home/zh/model/Llama-2-7b-hf")
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 3, 4, 10, 20, 31])
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--massive-threshold", type=float, default=100.0)
    parser.add_argument("--diag-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="aidcs_repro/results/massive_token_analysis")
    parser.add_argument("--seed", type=int, default=20260324)
    return parser.parse_args()


def load_gsm8k_samples(tokenizer, num_samples: int, max_length: int) -> List[torch.Tensor]:
    dataset = Dataset.from_file(
        "/home/zh/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/"
        "cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow"
    )
    samples = []
    for example in dataset:
        text = f"Question: {example['question']}\nAnswer: {example['answer']}"
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        if encoded.numel() < 32:
            continue
        samples.append(encoded)
        if len(samples) >= num_samples:
            break
    return samples


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"[info] loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    samples = load_gsm8k_samples(tokenizer, args.num_samples, args.max_length)
    print(f"[info] loaded {len(samples)} gsm8k samples")

    layer_accumulators: Dict[int, Dict[str, List[float]]] = {
        layer_idx: {
            "massive_diag": [],
            "normal_diag": [],
            "massive_count": [],
            "token_count": [],
        }
        for layer_idx in args.layers
    }

    with torch.inference_mode():
        for sample_id, sample in enumerate(samples):
            input_ids = sample.unsqueeze(0).to("cuda")
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions

            for layer_idx in args.layers:
                hidden = hidden_states[layer_idx][0].float()
                diag_attn = attentions[layer_idx][0].float().diagonal(dim1=-2, dim2=-1).mean(dim=0)
                massive_mask = hidden.abs().amax(dim=-1) >= args.massive_threshold
                normal_mask = ~massive_mask
                if massive_mask.any():
                    layer_accumulators[layer_idx]["massive_diag"].extend(diag_attn[massive_mask].cpu().tolist())
                if normal_mask.any():
                    layer_accumulators[layer_idx]["normal_diag"].extend(diag_attn[normal_mask].cpu().tolist())
                layer_accumulators[layer_idx]["massive_count"].append(int(massive_mask.sum().item()))
                layer_accumulators[layer_idx]["token_count"].append(int(hidden.shape[0]))

            print(f"[run] sample={sample_id + 1}/{len(samples)}")

    rows = []
    summary = {
        "model_path": args.model_path,
        "layers": args.layers,
        "num_samples": len(samples),
        "max_length": args.max_length,
        "massive_threshold": args.massive_threshold,
        "diag_threshold": args.diag_threshold,
        "results": {},
    }

    for layer_idx in args.layers:
        acc = layer_accumulators[layer_idx]
        massive_diag = torch.tensor(acc["massive_diag"], dtype=torch.float32)
        normal_diag = torch.tensor(acc["normal_diag"], dtype=torch.float32)
        total_massive = sum(acc["massive_count"])
        total_tokens = sum(acc["token_count"])
        massive_ratio = total_massive / max(1, total_tokens)

        massive_mean = float(massive_diag.mean()) if massive_diag.numel() else 0.0
        normal_mean = float(normal_diag.mean()) if normal_diag.numel() else 0.0
        separation = massive_mean / normal_mean if normal_mean > 0 else 0.0
        protected_by_diag = float((massive_diag >= args.diag_threshold).float().mean()) if massive_diag.numel() else 0.0

        result = {
            "massive_token_count": total_massive,
            "token_count": total_tokens,
            "massive_token_ratio": massive_ratio,
            "massive_diag_mean": massive_mean,
            "normal_diag_mean": normal_mean,
            "diag_separation_ratio": separation,
            "massive_diag_ge_threshold_ratio": protected_by_diag,
        }
        summary["results"][str(layer_idx)] = result
        rows.append({"layer": layer_idx, **result})

    csv_path = output_dir / "massive_token_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "massive_token_count",
                "token_count",
                "massive_token_ratio",
                "massive_diag_mean",
                "normal_diag_mean",
                "diag_separation_ratio",
                "massive_diag_ge_threshold_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    save_json(summary, output_dir / "massive_token_analysis.json")

    latex_lines = [
        "% Auto-generated by aidcs_repro/run_massive_token_analysis.py",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Layer & Massive Ratio & Massive Diag & Normal Diag & Separation \\\\",
        "\\midrule",
    ]
    for row in rows:
        latex_lines.append(
            f"{row['layer']} & {row['massive_token_ratio']:.4f} & {row['massive_diag_mean']:.4f} & "
            f"{row['normal_diag_mean']:.4f} & {row['diag_separation_ratio']:.2f}$\\times$ \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "massive_token_analysis_table.tex").write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
