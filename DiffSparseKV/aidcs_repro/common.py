import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_SEED = 20260324


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def _arc_choices_text(example: Dict) -> str:
    choices = example.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    choice_text = " ".join(f"({lab}) {txt}" for lab, txt in zip(labels, texts))
    answer = example.get("answerKey", "")
    return f"Question: {example.get('question', '')} Choices: {choice_text} Answer: {answer}"


def _hellaswag_text(example: Dict) -> str:
    endings = " ".join(f"Option {idx + 1}: {text}" for idx, text in enumerate(example.get("endings", [])))
    return f"Context: {example.get('ctx', '')} {endings}"


DATASET_SPECS: Dict[str, Dict] = {
    "wikitext2": {
        "loader": lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="validation"),
        "text_fn": lambda ex: ex.get("text", ""),
    },
    "c4": {
        "loader": lambda: load_dataset("allenai/c4", "en", split="validation", streaming=True),
        "text_fn": lambda ex: ex.get("text", ""),
    },
    "redpajama_en": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/togethercomputer___red_pajama-data-v2/sample/1.0.0/"
            "47f2eb82c4957040090a6cb2e596da9b9b632341367d2e7ab04abb65ce7dbf9d/"
            "red_pajama-data-v2-train-00000-of-00022.arrow"
        ),
        "text_fn": lambda ex: ex.get("raw_content", "")
        if '"language": "en"' in ex.get("meta", "")
        else "",
    },
    "gsm8k": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/"
            "cc7b047b6e5bb11b4f1af84efc572db110a51b3c/gsm8k-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('question', '')}\nAnswer: {ex.get('answer', '')}",
    },
    "qasper": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/THUDM___long_bench/qasper/1.0.0/"
            "4a916a4bde5c3481ac49b84d5dde69a9d2eefcd67f884ef65b3d97ee7cc91f3e/long_bench-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('input', '')}\nContext: {ex.get('context', '')}\nAnswer: {ex.get('answers', [])}",
    },
    "multifieldqa_en": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/THUDM___long_bench/multifieldqa_en/1.0.0/"
            "4a916a4bde5c3481ac49b84d5dde69a9d2eefcd67f884ef65b3d97ee7cc91f3e/long_bench-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('input', '')}\nContext: {ex.get('context', '')}\nAnswer: {ex.get('answers', [])}",
    },
    "narrativeqa": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/THUDM___long_bench/narrativeqa/1.0.0/"
            "4a916a4bde5c3481ac49b84d5dde69a9d2eefcd67f884ef65b3d97ee7cc91f3e/long_bench-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('input', '')}\nContext: {ex.get('context', '')}\nAnswer: {ex.get('answers', [])}",
    },
    "hotpotqa": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/THUDM___long_bench/hotpotqa/1.0.0/"
            "4a916a4bde5c3481ac49b84d5dde69a9d2eefcd67f884ef65b3d97ee7cc91f3e/long_bench-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('input', '')}\nContext: {ex.get('context', '')}\nAnswer: {ex.get('answers', [])}",
    },
    "2wikimqa": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/THUDM___long_bench/2wikimqa/1.0.0/"
            "4a916a4bde5c3481ac49b84d5dde69a9d2eefcd67f884ef65b3d97ee7cc91f3e/long_bench-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('input', '')}\nContext: {ex.get('context', '')}\nAnswer: {ex.get('answers', [])}",
    },
    "musique": {
        "loader": lambda: Dataset.from_file(
            "/home/zh/.cache/huggingface/datasets/THUDM___long_bench/musique/1.0.0/"
            "4a916a4bde5c3481ac49b84d5dde69a9d2eefcd67f884ef65b3d97ee7cc91f3e/long_bench-test.arrow"
        ),
        "text_fn": lambda ex: f"Question: {ex.get('input', '')}\nContext: {ex.get('context', '')}\nAnswer: {ex.get('answers', [])}",
    },
}


def iter_texts(dataset_name: str) -> Iterable[str]:
    spec = DATASET_SPECS[dataset_name]
    dataset = spec["loader"]()
    text_fn: Callable[[Dict], str] = spec["text_fn"]
    for example in dataset:
        text = text_fn(example)
        if text is None:
            continue
        text = text.strip()
        if len(text) < 16:
            continue
        yield text


def build_tokenized_samples(
    tokenizer,
    dataset_name: str,
    num_samples: int,
    max_length: int,
) -> List[torch.Tensor]:
    samples: List[torch.Tensor] = []
    for text in iter_texts(dataset_name):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"][0]
        if encoded.numel() < 32:
            continue
        samples.append(encoded)
        if len(samples) >= num_samples:
            break
    if not samples:
        raise RuntimeError(f"No usable samples loaded for dataset={dataset_name}")
    return samples


def topk_prune_last_dim(x: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity <= 0.0:
        return x
    if sparsity >= 1.0:
        return torch.zeros_like(x)

    dim = x.shape[-1]
    num_keep = max(1, int(round((1.0 - sparsity) * dim)))
    kth_smallest = max(1, dim - num_keep + 1)
    flat = x.reshape(-1, dim)
    abs_flat = flat.abs()
    threshold = abs_flat.kthvalue(kth_smallest, dim=-1, keepdim=True).values
    mask = abs_flat >= threshold
    return (flat * mask).reshape_as(x)


def resolve_target_modules(model, target_kind: str) -> List[Tuple[str, torch.nn.Module]]:
    pairs: List[Tuple[str, torch.nn.Module]] = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        if target_kind == "qkv":
            pairs.extend(
                [
                    (f"layers.{layer_idx}.self_attn.q_proj", attn.q_proj),
                    (f"layers.{layer_idx}.self_attn.k_proj", attn.k_proj),
                    (f"layers.{layer_idx}.self_attn.v_proj", attn.v_proj),
                ]
            )
        elif target_kind == "out":
            pairs.append((f"layers.{layer_idx}.self_attn.o_proj", attn.o_proj))
        elif target_kind == "up":
            pairs.extend(
                [
                    (f"layers.{layer_idx}.mlp.up_proj", mlp.up_proj),
                    (f"layers.{layer_idx}.mlp.gate_proj", mlp.gate_proj),
                ]
            )
        elif target_kind == "down":
            pairs.append((f"layers.{layer_idx}.mlp.down_proj", mlp.down_proj))
        elif target_kind == "gate":
            pairs.append((f"layers.{layer_idx}.mlp.gate_proj", mlp.gate_proj))
        else:
            raise ValueError(f"Unsupported target_kind={target_kind}")
    return pairs


@contextmanager
def prune_linear_inputs(model, target_kind: str, sparsity: float):
    handles = []

    def hook(_module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        return (topk_prune_last_dim(x, sparsity),) + tuple(inputs[1:])

    for _name, module in resolve_target_modules(model, target_kind):
        handles.append(module.register_forward_pre_hook(hook, with_kwargs=False))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate_language_model(
    model,
    samples: Sequence[torch.Tensor],
    device: str = "cuda",
) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    losses: List[float] = []

    with torch.inference_mode():
        for sample in samples:
            input_ids = sample.unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = float(outputs.loss.detach().cpu())
            valid_tokens = max(1, input_ids.shape[1] - 1)
            total_nll += loss * valid_tokens
            total_tokens += valid_tokens
            losses.append(loss)

    mean_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(mean_loss)
    return {
        "loss": mean_loss,
        "ppl": ppl,
        "num_tokens": total_tokens,
        "num_samples": len(samples),
        "mean_sample_loss": sum(losses) / max(1, len(losses)),
    }


def get_qkv_layer(model, layer_idx: int):
    return model.model.layers[layer_idx].self_attn.q_proj
