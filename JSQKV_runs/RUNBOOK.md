# JSQKV Experiment Runbook

Date: `2026-04-04`

This file is a reusable operator guide for running the JSQKV recovery
experiments on any supported model.

It is intentionally written as a model-agnostic runbook rather than as a
single-day log.

## 1. What This Pipeline Is Trying To Measure

The current experiment stack separates three things:

1. `DiffSparseKV only`
   - sparse-only reference
2. `uniform + KIVI-style quant`
   - sequential proxy baseline
3. `JSQKV`
   - differential sparsity + Per-Token-Tile quantization

The minimum acceptable sanity condition is:

- `JSQKV(fp16, no hadamard)` should behave close to `DiffSparseKV only`

If that condition fails, do **not** trust any low-bit result.

## 2. Supported Scope

Current JSQKV implementation support:

- `Llama`-family path is the actively repaired and validated path

Before switching to a new model family:

- confirm the model can be loaded through the same integration path
- do not assume `Mistral` / `Qwen` support is already equivalent unless that
  path has been explicitly validated

## 3. Dataset Modes

### `selected6 full`

Use:

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `trec`
- `lcc`

Meaning:

- only these six focal tasks are evaluated
- each task uses its full dataset split

This is the default fast-turnaround protocol for paper-facing accuracy recovery.

### `limit=12`

Meaning:

- only 12 examples per task

Use only for:

- rapid sanity checking
- hadamard/no-hadamard comparisons
- verifying that a code change did not break the path

Never treat `limit=12` results as final paper numbers.

### Single-task full

Recommended when:

- one setting already shows positive small-sample signal
- you want the cheapest strong validation before launching a full selected6
  sweep

Best candidate task so far:

- `qasper`

## 4. Stable Setting Choices So Far

These are the current empirically best choices for `Meta-Llama-3-8B-Instruct`.

### 70% budget

- `4bit`
  - prefer `no hadamard`
- `2bit`
  - prefer `tile hadamard(64)`

### 50% budget

- `4bit`
  - `no hadamard` is currently acceptable
- `2bit`
  - still under active completion / comparison

## 5. Naming Convention

Use output tags that encode:

- budget
- method
- bit width
- hadamard choice
- scope
- context length

Examples:

- `meta70_jsqkv_4bit_nohad_selected6_full_4096`
- `meta70_uniformkivi_2bit_selected6_full_4096`
- `meta70_jsqkv_2bit_tilehad_qasper_full_4096`

When switching models, only replace the model prefix:

- `llama2_70_jsqkv_4bit_nohad_selected6_full_4096`
- `mistral70_jsqkv_2bit_tilehad_selected6_full_4096`
- `qwen70_jsqkv_4bit_nohad_selected6_full_4096`

## 6. Command Templates

### 6.1 JSQKV selected6 full

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> conda run --no-capture-output -n mustafar \
python JSQKV/eval_jsqkv_longbench.py \
  --model_path <model_path> \
  --max_length 4096 \
  --datasets narrativeqa qasper multifieldqa_en hotpotqa trec lcc \
  --output_dir JSQKV_runs/<output_root> \
  --output_tag <tag> \
  --target_distribution <p0,p1,p2> \
  --sparsity_levels <rho0,rho1,rho2> \
  --importance_mode value_aware \
  --head_aggregation_mode max \
  --value_sink_keep 4 \
  --level_2_mode evict \
  --k_bits <bits> \
  --v_bits <bits> \
  --quant_impl default \
  --k_quant_scheme per-token-tile \
  --v_quant_scheme per-token-tile \
  --group_size 128 \
  --quant_granularity per-token-tile \
  --tile_size 64 \
  --residual_length 128 \
  [--enable_hadamard --hadamard_mode tile --hadamard_group_size 64] \
  --run_eval
```

### 6.2 Uniform+KIVI selected6 full

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> conda run --no-capture-output -n mustafar \
python JSQKV/eval_jsqkv_longbench.py \
  --model_path <model_path> \
  --max_length 4096 \
  --datasets narrativeqa qasper multifieldqa_en hotpotqa trec lcc \
  --output_dir JSQKV_runs/<output_root> \
  --output_tag <tag> \
  --target_distribution 0.0,1.0,0.0 \
  --sparsity_levels <rho0,rho1,rho2> \
  --importance_mode value_aware \
  --head_aggregation_mode max \
  --value_sink_keep 4 \
  --level_2_mode evict \
  --k_bits <bits> \
  --v_bits <bits> \
  --quant_impl kivi \
  --k_quant_scheme kivi-channel \
  --v_quant_scheme per-token-head \
  --group_size 128 \
  --quant_granularity per-token-tile \
  --tile_size 64 \
  --residual_length 128 \
  --hadamard_mode none \
  --run_eval
```

### 6.3 Single-task full

Replace the dataset list with a single task:

```bash
--datasets qasper
```

## 7. Recommended Experiment Order

When moving to a new model, run in this order:

1. `JSQKV(fp16, no hadamard, limit=12, qasper)`
2. `DiffSparseKV only(limit=12, qasper)`
3. compare the two:
   - they should be close
4. `JSQKV 4bit(no had, limit=12, qasper)`
5. `JSQKV 2bit(tile had, limit=12, qasper)`
6. `qasper full` on the strongest setting
7. `selected6 full` on the strongest setting
8. only then broaden to other budgets / bit widths

## 8. Decision Rules

### If `JSQKV(fp16, no hadamard)` is much worse than `DiffSparseKV only`

Interpretation:

- wrapper/integration still broken

Action:

- stop running broader low-bit experiments
- fix integration first

### If `4bit` is positive and `2bit` is negative

Interpretation:

- joint path works
- low-bit configuration still needs tuning

Action:

- keep `4bit` as the first paper-facing main result
- continue only targeted `2bit` ablations

### If `tile hadamard` helps at `2bit` but hurts at `4bit`

Interpretation:

- this is acceptable and expected in a sparse+quant system
- hadamard is helping quantization while hurting magnitude separability for
  pruning

Action:

- treat hadamard as bit-dependent rather than universal

## 9. Runtime Estimates

For the six focal tasks on `Meta-Llama-3-8B-Instruct`:

- `narrativeqa`: about `5-6 min`
- `qasper`: about `7-8 min`
- `multifieldqa_en`: about `4 min`
- `hotpotqa`: about `5 min`
- `trec`: about `11-12 min`
- `lcc`: about `24-27 min`

Selected6 full total:

- about `55-65 min` per configuration

Single-task full:

- `qasper`: about `25-40 min`

## 10. Output Locations

Use these output roots consistently:

- `JSQKV_runs/sanity_after_patch`
  - for one-off debugging
- `JSQKV_runs/aligned_selected6`
  - for `limit=12` aligned sanity
- `JSQKV_runs/final_selected6`
  - for selected6 full outputs
- `JSQKV_runs/full_qasper`
  - for single-task full outputs
- `JSQKV_runs/ablations_4096`
  - for hadamard / no-hadamard / bit-width ablations

## 11. What To Read First

When multiple results complete, read in this order:

1. `full_qasper` strongest setting
2. `selected6 full` strongest setting
3. matched sequential proxy for the same setting
4. only then read weaker settings / ablations
