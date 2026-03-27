import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

from common import (
    build_tokenized_samples,
    ensure_dir,
    get_qkv_layer,
    load_model_and_tokenizer,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect threshold distributions for qkv inputs across datasets.")
    parser.add_argument("--model-path", type=str, default="/home/zh/model/Llama-2-7b-hf")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext2", "gsm8k", "qasper", "multifieldqa_en", "narrativeqa", "hotpotqa", "musique"],
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--max-vectors-per-layer", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="aidcs_repro/results/threshold_stability")
    parser.add_argument("--seed", type=int, default=20260324)
    return parser.parse_args()


def vector_thresholds(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    dim = x.shape[-1]
    num_keep = max(1, int(round((1.0 - sparsity) * dim)))
    kth_smallest = max(1, dim - num_keep + 1)
    flat = x.abs().reshape(-1, dim)
    return flat.kthvalue(kth_smallest, dim=-1).values


def make_hook(layer_idx: int, storage: Dict[int, List[float]], sparsity: float, max_vectors: int):
    def hook(_module, inputs):
        if not inputs:
            return
        x = inputs[0].detach()
        thresholds = vector_thresholds(x, sparsity).float().cpu()
        remain = max_vectors - len(storage[layer_idx])
        if remain <= 0:
            return
        if thresholds.numel() > remain:
            thresholds = thresholds[:remain]
        storage[layer_idx].extend(thresholds.tolist())

    return hook


def summarize(values: List[float]) -> Dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    q25, median, q75 = torch.quantile(tensor, torch.tensor([0.25, 0.5, 0.75]))
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "max": float(tensor.max()),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"[info] loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    summary = {
        "model_path": args.model_path,
        "datasets": args.datasets,
        "layers": args.layers,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "sparsity": args.sparsity,
        "stats": {},
        "raw_values": {},
    }
    csv_rows = []

    for dataset_name in args.datasets:
        print(f"[run] dataset={dataset_name}")
        samples = build_tokenized_samples(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            num_samples=args.num_samples,
            max_length=args.max_length,
        )

        storage: Dict[int, List[float]] = defaultdict(list)
        handles = []
        for layer_idx in args.layers:
            q_proj = get_qkv_layer(model, layer_idx)
            handles.append(
                q_proj.register_forward_pre_hook(
                    make_hook(layer_idx, storage, args.sparsity, args.max_vectors_per_layer),
                    with_kwargs=False,
                )
            )

        with torch.inference_mode():
            for sample in samples:
                _ = model(input_ids=sample.unsqueeze(0).to("cuda"), use_cache=False)

        for handle in handles:
            handle.remove()

        dataset_stats = {}
        for layer_idx in args.layers:
            stats = summarize(storage[layer_idx])
            dataset_stats[str(layer_idx)] = stats
            csv_rows.append(
                {
                    "dataset": dataset_name,
                    "layer": layer_idx,
                    **stats,
                }
            )
        summary["stats"][dataset_name] = dataset_stats
        summary["raw_values"][dataset_name] = {str(layer_idx): storage[layer_idx] for layer_idx in args.layers}

    csv_path = output_dir / "threshold_stability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "layer", "count", "mean", "std", "min", "q25", "median", "q75", "max"],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    save_json(summary, output_dir / "threshold_stability.json")

    raw_path = output_dir / "threshold_stability_raw_values.json"
    save_json(summary["raw_values"], raw_path)

    for layer_idx in args.layers:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_data = [summary["raw_values"][dataset_name][str(layer_idx)] for dataset_name in args.datasets]
        ax.boxplot(plot_data, tick_labels=args.datasets, showfliers=False)
        ax.set_title(f"QKV Input Threshold Distribution at Layer {layer_idx}")
        ax.set_ylabel(f"Threshold at sparsity={args.sparsity:.2f}")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / f"layer_{layer_idx}_threshold_boxplot.png", dpi=200)
        fig.savefig(output_dir / f"layer_{layer_idx}_threshold_boxplot.pdf")
        plt.close(fig)

    latex_lines = [
        "% Auto-generated by aidcs_repro/run_threshold_stability.py",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & Layer & Mean & Std & Median \\\\",
        "\\midrule",
    ]
    for row in csv_rows:
        latex_lines.append(
            f"{row['dataset']} & {row['layer']} & {row['mean']:.4f} & {row['std']:.4f} & {row['median']:.4f} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "threshold_stability_table.tex").write_text("\n".join(latex_lines), encoding="utf-8")

    print(f"[done] wrote {csv_path}")


if __name__ == "__main__":
    main()
