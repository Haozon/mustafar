# Server Handoff: Llama3-8B KV Quantization Work

Date:

- `2026-03-27`

Workspace:

- `/mnt/home/zh/mustafar`

Main folders:

- `/mnt/home/zh/mustafar/RotateTileKV`
- `/mnt/home/zh/mustafar/KIVI_upstream`

Model used throughout:

- `/mnt/home/zh/model/Meta-Llama-3-8B-Instruct`

Main evaluation subset used so far:

- `trec,triviaqa,passage_count,qasper`

Default larger subset sample count:

- `limit=10`

## 1. Goal

The current project goal is:

- reproduce KV cache quantization variants inside `RotateTileKV`
- compare:
  - `per-token`
  - `per-token-head`
  - `per-token-tile`
  - `Hadamard on/off`
  - `full Hadamard` vs `tile Hadamard(64)`
  - `residual window`
  - `V-only`
  - `K-only`
  - local `KIVI`-style fake quant
- evaluate them on the same Llama3-8B LongBench subset
- then compare against original KIVI

## 2. What Was Implemented

### 2.1 RotateTileKV core implementation

Files:

- `/mnt/home/zh/mustafar/RotateTileKV/fake_quant.py`
- `/mnt/home/zh/mustafar/RotateTileKV/modeling_llama_rotatetilekv.py`
- `/mnt/home/zh/mustafar/RotateTileKV/run_longbench.py`

Implemented features:

- fake quantization for KV cache
- quantization granularities:
  - `per-token`
  - `per-token-head`
  - `per-token-tile`
- Hadamard modes:
  - `none`
  - `full`
  - `tile`
- tile-sized Hadamard with `--hadamard-group-size 64`
- residual window support:
  - `--residual-length`
- separate local quantization schemes:
  - `quant_impl = default`
  - `quant_impl = kivi`
  - `k_quant_scheme = kivi-channel`
  - `v_quant_scheme = per-token-head / per-token-tile / ...`

### 2.2 KIVI upstream integration

Files:

- `/mnt/home/zh/mustafar/KIVI_upstream`
- `/mnt/home/zh/mustafar/KIVI_upstream/run_longbench_subset.py`

KIVI upstream source:

- `https://github.com/jy-yuan/KIVI.git`

Commit used:

- `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`

Important local fix applied:

- `/mnt/home/zh/mustafar/KIVI_upstream/models/llama_kivi.py`

Reason:

- upstream `main` on Llama3/GQA decode had a `value_states_full` head-mismatch issue
- fixed by using `repeat_kv(value_states_full, self.num_key_value_groups)` on the relevant decode branches

### 2.3 Queue runner

File:

- `/mnt/home/zh/mustafar/RotateTileKV/queue_longbench_jobs.py`

Purpose:

- background GPU-aware job scheduler
- starts jobs only when free memory is above threshold
- supports many experiment JSON job lists

Important local fix:

- queue runner now ignores its own `queue_longbench_jobs.py` process when counting active workers

## 3. Meaning of “Pilot”

In all current markdown files, `pilot` means:

- only the 4-task subset is used:
  - `trec`
  - `triviaqa`
  - `passage_count`
  - `qasper`
- and only `3` samples per dataset are used
- pilot runs are for quick trend checking only
- pilot numbers are not final benchmark claims

## 4. Main Result Files

Read these first:

- `/mnt/home/zh/mustafar/RotateTileKV/LLAMA3_8B_CURRENT_RESULTS.md`
- `/mnt/home/zh/mustafar/RotateTileKV/LOCAL_KIVI_FAKE_COMPARISON_LLAMA3_8B.md`
- `/mnt/home/zh/mustafar/RotateTileKV/KIVI_COMPARISON_LLAMA3_8B.md`

## 5. Current Best Important Results

All below are on:

- `Meta-Llama-3-8B-Instruct`
- datasets `trec,triviaqa,passage_count,qasper`
- `limit=10`
- `residual_length=128`

### 5.1 Our per-token-tile path

| Config | average |
|---|---:|
| per-token-tile 4bit | 58.22 |
| per-token-tile 4bit + tile hadamard(64) | 58.58 |
| per-token-tile 3bit | 55.60 |
| per-token-tile 3bit + tile hadamard(64) | 57.13 |
| per-token-tile 2bit | 12.25 |
| per-token-tile 2bit + tile hadamard(64) | 46.55 |

### 5.2 V-only

This means:

- `K = fp16`
- only `V` is quantized

| Config | average |
|---|---:|
| V-only tile 4bit | 59.81 |
| V-only tile 3bit | 58.04 |
| V-only tile 2bit | 57.72 |
| V-only head 4bit | 57.21 |
| V-only head 2bit | 54.67 |

Key conclusion:

- V-only remains strong even at low bit
- V quantization is not the main bottleneck

### 5.3 K-only

This means:

- `V = fp16`
- only `K` is quantized

Current local K-only tile results:

| Config | average |
|---|---:|
| K-only tile 4bit | 60.91 |
| K-only tile 4bit + tile hadamard(64) | 55.92 |
| K-only tile 3bit | 54.12 |
| K-only tile 3bit + tile hadamard(64) | 58.03 |
| K-only tile 2bit | 18.66 |
| K-only tile 2bit + tile hadamard(64) | 46.25 |

Key conclusion:

- K is much more sensitive than V
- tile Hadamard helps K noticeably at `3bit` and `2bit`
- tile Hadamard is not clearly beneficial at `4bit`

### 5.4 Local KIVI-style fake quant

This is the local reproduction inside `RotateTileKV`, not upstream runtime KIVI:

- `quant_impl = kivi`
- `k_quant_scheme = kivi-channel`
- `v_quant_scheme = per-token-head`
- `group_size = 128`

| Config | average |
|---|---:|
| KIVI 4bit | 60.29 |
| KIVI 2bit | 47.23 |

### 5.5 Original KIVI (upstream implementation)

Subset result directories:

- `/mnt/home/zh/mustafar/KIVI_upstream/exp_llama3_kivi_l10/kivi2`
- `/mnt/home/zh/mustafar/KIVI_upstream/exp_llama3_kivi_l10/kivi4`

| Config | average |
|---|---:|
| upstream KIVI 4bit | 60.54 |
| upstream KIVI 2bit | 59.10 |

## 6. Most Important Interpretation

### 6.1 Are we aligned with KIVI now?

Mostly yes.

On the same subset and same `residual_length=128`:

- upstream KIVI 4bit = `60.54`
- local KIVI-style fake 4bit = `60.29`

Difference:

- about `0.25`

This is close enough to treat the local reproduction as largely aligned at `4bit`.

At `2bit`:

- local KIVI-style fake 2bit = `47.23`
- our per-token-tile 2bit + tile hadamard(64) = `46.55`

Difference:

- about `0.68`

Meaning:

- after alignment, the local results are already quite close
- previous large gaps were mostly due to different quantization designs, not just “Hadamard is wrong”

### 6.2 Is the main problem K or V?

Mainly `K`.

Reason:

- V-only remains strong even at `2bit`
- K-only drops much more
- adding Hadamard mostly helps K-side low-bit runs

### 6.3 Is Hadamard useful?

It depends on bit width and where it is applied.

Observed trend:

- `4bit`: Hadamard is not clearly needed; full or tile may be neutral or slightly harmful
- `3bit`: Hadamard starts to help
- `2bit`: tile Hadamard is clearly useful

### 6.4 full Hadamard vs tile Hadamard

Current practical conclusion:

- `full Hadamard` is not a good default for `per-token-tile`
- `tile Hadamard(64)` is the better aligned choice for tile quantization

## 7. Background Queue Files

Existing job lists:

- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_tile_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_token_head_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_token_head_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_vonly_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_vonly_head_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_konly_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_kivi_align_fake_limit10.json`
- `/mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_kivi_keyonly_limit10.json`
- `/mnt/home/zh/mustafar/KIVI_upstream/jobs_llama3_kivi_full_longbench.json`

Logs:

- `/mnt/home/zh/mustafar/RotateTileKV/bg_logs`
- `/mnt/home/zh/mustafar/KIVI_upstream/bg_logs`

## 8. Full LongBench Status

Full upstream KIVI run is prepared but not yet completed:

- job file:
  `/mnt/home/zh/mustafar/KIVI_upstream/jobs_llama3_kivi_full_longbench.json`

It targets these 16 tasks:

- `narrativeqa`
- `qasper`
- `multifieldqa_en`
- `hotpotqa`
- `2wikimqa`
- `musique`
- `gov_report`
- `qmsum`
- `multi_news`
- `trec`
- `triviaqa`
- `samsum`
- `passage_count`
- `passage_retrieval_en`
- `lcc`
- `repobench-p`

Current output directories:

- `/mnt/home/zh/mustafar/KIVI_upstream/exp_llama3_kivi_full/kivi2`
- `/mnt/home/zh/mustafar/KIVI_upstream/exp_llama3_kivi_full/kivi4`

At the time of writing, these full result folders do not yet contain `result.json`.

## 9. Recommended Next Steps on New Server

Priority order:

1. Run full upstream KIVI LongBench first
2. Run full local aligned KIVI-style fake quant
3. Run full local per-token-tile + residual128 + tile hadamard
4. Compare full results, not just subset results

### 9.1 Check environment

Recommended checks:

```bash
cd /mnt/home/zh/mustafar
python -V
nvidia-smi
python - <<'PY'
import torch, transformers
print(torch.__version__)
print(transformers.__version__)
PY
```

### 9.2 Make sure KIVI quant extension is available

```bash
cd /mnt/home/zh/mustafar/KIVI_upstream/quant
pip install -e .
```

### 9.3 Resume upstream full KIVI queue

```bash
cd /mnt/home/zh/mustafar
PYTHONUNBUFFERED=1 python RotateTileKV/queue_longbench_jobs.py \
  --jobs-file /mnt/home/zh/mustafar/KIVI_upstream/jobs_llama3_kivi_full_longbench.json \
  --repo-root /mnt/home/zh/mustafar/KIVI_upstream \
  --max-total-workers 1 \
  --min-free-mib 30000 \
  --poll-seconds 30 \
  --process-match KIVI_upstream/run_longbench_subset.py
```

### 9.4 Resume local KIVI-key-only queue

```bash
cd /mnt/home/zh/mustafar
PYTHONUNBUFFERED=1 python RotateTileKV/queue_longbench_jobs.py \
  --jobs-file /mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_kivi_keyonly_limit10.json \
  --repo-root /mnt/home/zh/mustafar \
  --max-total-workers 1 \
  --min-free-mib 30000 \
  --poll-seconds 30
```

### 9.5 Resume V-only head queue

```bash
cd /mnt/home/zh/mustafar
PYTHONUNBUFFERED=1 python RotateTileKV/queue_longbench_jobs.py \
  --jobs-file /mnt/home/zh/mustafar/RotateTileKV/jobs_llama3_res128_vonly_head_limit10.json \
  --repo-root /mnt/home/zh/mustafar \
  --max-total-workers 1 \
  --min-free-mib 30000 \
  --poll-seconds 30
```

## 10. Short Final Summary

- The project started from “Per-Token / Per-Token-Head / Per-Token-Tile + Hadamard” KV fake quantization.
- After adding:
  - `tile Hadamard(64)`
  - `residual_length=128`
  - local KIVI-style K/V quantization
- the gap to KIVI became small.
- `V-only` is already strong, so the remaining difficulty is mainly on `K`.
- For low-bit K quantization, `tile Hadamard(64)` is useful.
- The next meaningful step is to finish full LongBench and compare full-task results rather than the current 4-task subset.
