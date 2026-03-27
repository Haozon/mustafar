# JSQKV End-to-End Experiment Plan

Date: 2026-03-21

## Goal

Replace the mixed "compression microbenchmark + end-to-end throughput" story with an
end-to-end-first evaluation that still quantifies the effect of the KV compression path.

This plan answers three questions:

1. In real generation, when does sparse / quant-sparse beat dense?
2. How much of the gain comes from the decode phase?
3. How much prompt-side overhead is introduced by cache building / compression?

## Workload

Use one primary workload throughout the section:

- Model: `Meta-Llama-3-8B-Instruct`
- Input length: `4096`
- Output length: `256`
- Batch size: `1..8`

This keeps the entire section aligned with the paper's main batch-size experiment.

## Main Experiment

Measure end-to-end generation on the following five configs:

- `Dense`
- `Sparse50`
- `Sparse70`
- `Sparse50+2bit`
- `Sparse70+2bit`

Report:

- throughput (`tokens/s`)
- TTFT
- TPOT
- peak memory

Requirements:

- each `BS` is run in isolation
- no model reuse across batch sizes
- at least `3` repeats per point

## Attribution Experiment

Use the same workload and batch sizes, but decompose runtime into:

- `online_total_ms`: full prompt-to-output generation
- `prefill_ms`: prompt processing with cache build enabled
- `cached_decode_ms`: decode from cached prompt state

Interpretation:

- `prefill_ms` captures prompt-side work, including cache build / compression
- `cached_decode_ms` captures the post-cache decode path
- comparing sparse FP16 vs sparse 2bit on the same `BS` shows whether quantization
  reduces decode-side cost enough to offset prompt-side overhead

This is not a synthetic kernel benchmark. All measurements are taken on the real model,
real prompt length, and real decode loop.

## Suggested Figures

Main text:

- throughput vs batch size (5-line)
- dense / sparse50 / sparse50+2bit (3-line)
- dense / sparse70 / sparse70+2bit (3-line)
- prefill vs cached decode grouped bars for `BS=1,4,8`

Optional appendix:

- compression-kernel-only microbenchmark

## Practical Notes

- If GPU is shared, treat results as preview only.
- Prefer running the full sweep when GPU utilization is close to idle.
- If a point shows `BS+1` but lower `batch_ms`, rerun that point before using it in paper text.
