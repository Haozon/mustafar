# JSQKV Joint Accuracy Status

Date: `2026-04-04`

## Scope

This note summarizes what the current workspace can and cannot support for the
`Joint sparsity-quantization results` subsection of the JSQKV paper.

It focuses on three questions:

1. Which existing results are publication-usable?
2. Which current paper tables/claims are not supported?
3. What is the minimum rerun plan for recovering a defensible joint-accuracy section?

## Current Evidence Boundary

### 1. DiffSparseKV

`DiffSparseKV` is mature enough for the sparse-only subsection.

- Meta-Llama-3-8B finalized sparse-only results exist for:
  - `50%`
  - `70%`
- Reporting is already conservative:
  - task-adaptive calibration
  - full-task follow-up
  - fallback-to-uniform for failed tasks

Relevant source:

- `DiffSparseKV/solver_runs/per_task_current_summary.md`

### 2. RotateTileKV

`RotateTileKV` is mature enough for the quantization-only subsection.

- `Meta-Llama-3-8B-Instruct`: `full LongBench`, 16 tasks
- `llama-2-7b`: `full LongBench`, 16 tasks
- `Mistral-7B-v0.1`: `selected6`
- `Qwen2.5-7B-instruct`: `selected6`

The result pattern is mixed rather than uniformly positive:

- Llama3:
  - `4bit`: KIVI slightly better
  - `3bit`: KIVI slightly better
  - `2bit`: Per-Token-Tile + tile Hadamard(64) better
- Llama2:
  - `4bit`: our method slightly better
  - `3bit/2bit`: KIVI slightly better
- Mistral:
  - `4bit/3bit`: KIVI slightly better
  - `2bit`: our method better
- Qwen:
  - all three bits: KIVI clearly better

Relevant source:

- `RotateTileKV/ALL_MODELS_CURRENT_RESULTS.md`
- `RotateTileKV/LLAMA3_8B_CURRENT_RESULTS.md`

### 3. JSQKV

`JSQKV` is **not** ready for the main paper accuracy table.

The only currently retained joint results are small-sample diagnostics on:

- model: `Meta-Llama-3-8B-Instruct`
- tasks: `selected6`
- sample limit: `12`
- context length: `4096`
- budget: `70%`
- quantization: `2bit`

Those results are negative:

- `uniform + KIVI-align fake`: `31.28`
- `DiffSparseKV only`: `47.59`
- `JSQKV-lite`: `16.90`

Relevant source:

- `JSQKV_runs/compare_limit12_meta70_len4096/COMPARE_SUMMARY.md`

## Unsupported Current Paper Content

The current draft table

- `tab:main-accuracy`

is not supported by existing workspace results.

Specifically, the workspace does **not** currently contain publication-grade
joint results for:

- `50/50 + 4-bit`
- `70/70 + 4-bit`
- `50/50 + 2-bit`
- `70/70 + 2-bit` full-task/finalized
- `GovReport` / `PCount` under JSQKV
- any cross-model JSQKV table

Therefore the following claims are currently unsupported:

- `JSQKV` outperforms `M+K` across all four compression settings
- the gain becomes larger as the budget becomes tighter
- the cross-model advantage remains consistent across Llama-2, Mistral, and Qwen

## Implementation Findings

### 1. Current JSQKV path is not a faithful joint implementation

In the original `DiffSparseKV` prefill path, prefix compression is performed by
calling:

- `self.sparsity_applier.classify_and_apply_sparsity(...)`

which jointly handles:

- token classification
- token eviction for level-2
- feature-level top-k pruning for level-1

Relevant source:

- `DiffSparseKV/diffsparsekv/llama_integration.py`
- `DiffSparseKV/diffsparsekv/sparsity_applier.py`

In the current `JSQKV` path before patching, this was replaced by:

- direct token classification
- keeping all non-level-2 tokens
- quantizing only level-1 tokens

This means the joint path had already diverged from the sparse-only baseline:

- level-1 feature sparsity was not being applied through the original sparse
  operator
- the joint method therefore did not actually match the intended sparse+quant
  execution semantics

### 2. Decode-time re-quantization is already disabled

The current code already sets:

- `self.quantize_decode_cache = False`

so the old hypothesis that the failure is mainly caused by decode-time repeated
re-quantization is no longer sufficient.

The remaining issue is more likely a semantic mismatch between:

- the sparse cache produced by `DiffSparseKV`
- and the quantization assumptions inherited from `RotateTileKV`

### 3. JSQKV currently supports only Llama

The current loader is:

- `load_jsqkv_llama(...)`

and explicitly raises `NotImplementedError` for non-Llama model types.

Therefore the draft cross-model JSQKV table is not just missing data; it is also
missing model support in the current implementation.

## Minimum Rerun Matrix

The minimum rerun plan should be staged, not broad.

### Stage A: recover one positive Meta-Llama-3-8B joint sanity result

Run only `selected6` first.

Priority order:

1. `70/70 + 4bit`
2. `70/70 + 2bit`

Rationale:

- `4bit` is the safer debugging target
- `70%` is the stronger sparse-only regime
- this is enough to determine whether the joint path is recoverable at all

### Stage B: expand Meta-Llama-3-8B only after Stage A is non-negative

Then run:

1. `50/50 + 4bit`
2. `50/50 + 2bit`

Only after that should the paper reintroduce a four-setting main table.

### Stage C: cross-model only after Meta-Llama-3-8B is repaired

Cross-model JSQKV should not start before:

- Meta-Llama-3-8B joint path beats or at least matches `M+K` on `selected6`

Even then, model support must be generalized beyond Llama before using:

- `Mistral`
- `Qwen`

## Recommended Paper Action Right Now

For the current draft:

1. Keep the sparse-only subsection based on finalized `DiffSparseKV`.
2. Keep the quantization/Hadamard subsection based on `RotateTileKV`.
3. Remove or defer the current `Joint sparsity-quantization results` table.
4. Replace it with one of the following until new runs are available:
   - a short limitation statement
   - or a small diagnostic paragraph stating that the current naive joint path
     does not yet preserve the sparse-only and quant-only gains

## Current Active Repair Direction

The current repair direction in code is:

- restore the original `DiffSparseKV` prefix sparsification operator first
- then quantize the resulting compressed prefix once

This is the minimum faithful baseline before any larger rerun.

## Patch1 Sanity Check

A first repair pass was applied in:

- `JSQKV/integration.py`

The change restores the original sparse prefix operator before quantizing the
compressed prefix.

Patched sanity run:

- model: `Meta-Llama-3-8B-Instruct`
- task: `qasper`
- budget: `70%`
- quantization: `2bit`
- sample limit: `12`
- output:
  - `JSQKV_runs/sanity_after_patch/meta70_qasper_2bit_limit12_patch1/result.json`

Observed result:

- `qasper = 16.07`

Interpretation:

- this is still far below the sparse-only and sequential baselines on the same
  small-sample regime
- therefore the joint failure is not explained by a single obvious sparse-path
  wiring bug
- the next experiments should prioritize a safer `4bit` setting before spending
  time on full 2bit joint sweeps

## Wrapper Isolation Findings

Current same-environment sanity checks on `qasper`, `limit=12`, `70%`:

- current `DiffSparseKV` baseline rerun:
  - `37.44`
  - output:
    - `tmp_jsqkv_compare/Meta-Llama-3-8B-Instruct_4096_diff_sparse_kv_0.70_diffsparse_qasper_limit12/result.json`
- `JSQKV wrapper` with `fp16`, `no hadamard`:
  - `21.49`
  - outputs:
    - `JSQKV_runs/sanity_after_patch/meta70_qasper_fp16_nohad_limit12_wrapper_v4/result.json`
    - `JSQKV_runs/sanity_after_patch/meta70_qasper_fp16_nohad_limit12_wrapper_v5/result.json`
- `JSQKV wrapper` with `fp16`, `tile hadamard`:
  - `18.12`
  - output:
    - `JSQKV_runs/sanity_after_patch/meta70_qasper_fp16_tilehad_limit12_wrapper/result.json`

Interpretation:

- the current JSQKV wrapper is still damaging the sparse-only behavior even
  before low-bit quantization is applied
- therefore the first blocker is **not** low-bit quantization quality alone
- the immediate priority is to make:
  - `JSQKV(fp16, no hadamard)`
  match
  - `DiffSparseKV only`
  under the same sampling setup

Until that equivalence is restored, larger joint-accuracy experiments are not
trustworthy enough for the main paper table
