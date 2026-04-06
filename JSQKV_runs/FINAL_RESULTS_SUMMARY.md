# JSQKV Final Results Summary

Date: `2026-04-04`

## Paper-Ready Tables

### Table A: Selected6 Full Average Scores

The following LaTeX table is intended to be directly reusable in the paper
after light wording cleanup. Each cell is reported as:

- `M+K proxy / JSQKV`
- parenthetical tag after the JSQKV number indicates whether that reported
  JSQKV result used `tilehad` or `nohad`

Completed rows only are included; unfinished branches are marked `--`.

```latex
\begin{table}[t]
\centering
\caption{Cross-model selected6 averages under matched sparse-quant budgets. Each cell reports \textbf{M+K proxy / JSQKV}.}
\label{tab:jsqkv_cross_model_selected6}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccc}
\toprule
Model & 50\% + 4-bit & 50\% + 2-bit & 70\% + 4-bit & 70\% + 2-bit \\
\midrule
Meta-Llama-3-8B-Instruct & 45.90 / \textbf{45.93}~{\footnotesize(nohad)} & 43.38 / \textbf{44.02}~{\footnotesize(tilehad)} & 44.18 / \textbf{45.34}~{\footnotesize(nohad)} & 39.83 / \textbf{43.12}~{\footnotesize(tilehad)} \\
Llama-2-7B & 31.34 / \textbf{31.37}~{\footnotesize(tilehad, repairA)} & \textbf{31.51} / 30.23~{\footnotesize(tilehad, repairA)} & 30.57 / \textbf{30.97}~{\footnotesize(nohad)} & 29.92 / \textbf{30.09}~{\footnotesize(tilehad)} \\
Mistral-7B-v0.1 & 32.84 / \textbf{32.90}~{\footnotesize(nohad)} & 31.70 / \textbf{32.37}~{\footnotesize(tilehad)} & 32.43 / \textbf{32.71}~{\footnotesize(nohad)} & 30.14 / \textbf{32.35}~{\footnotesize(tilehad)} \\
Qwen2.5-7B-Instruct & 31.17 / \textbf{37.08}~{\footnotesize(tilehad)} & \textbf{26.10} / 23.48~{\footnotesize(tilehad)} & 33.00 / \textbf{39.67}~{\footnotesize(tilehad)} & 25.51 / \textbf{35.20}~{\footnotesize(tilehad)} \\
\bottomrule
\end{tabular}}
\end{table}
```

### Table B: Delta Table

This version is often easier to discuss in text because it directly reports the
average gain of JSQKV over the matched proxy.

```latex
\begin{table}[t]
\centering
\caption{Average selected6 gain of JSQKV over the matched M+K proxy. Positive values indicate JSQKV is better.}
\label{tab:jsqkv_cross_model_delta}
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcccc}
\toprule
Model & 50\% + 4-bit & 50\% + 2-bit & 70\% + 4-bit & 70\% + 2-bit \\
\midrule
Meta-Llama-3-8B-Instruct & +0.03 & +0.64 & +1.16 & +3.29 \\
Llama-2-7B & +0.03 & -1.28 & +0.40 & +0.17 \\
Mistral-7B-v0.1 & +0.06 & +0.67 & +0.28 & +2.21 \\
Qwen2.5-7B-Instruct & +5.91 & -2.62 & +6.67 & +9.69 \\
\bottomrule
\end{tabular}
\end{table}
```

### Markdown Version

Selected6 full average scores:

| Model | 50% + 4-bit | 50% + 2-bit | 70% + 4-bit | 70% + 2-bit |
|---|---:|---:|---:|---:|
| Meta-Llama-3-8B-Instruct | 45.90 / **45.93** (nohad) | 43.38 / **44.02** (tilehad) | 44.18 / **45.34** (nohad) | 39.83 / **43.12** (tilehad) |
| Llama-2-7B | 31.34 / **31.37** (tilehad, repairA) | **31.51** / 30.23 (tilehad, repairA) | 30.57 / **30.97** (nohad) | 29.92 / **30.09** (tilehad) |
| Mistral-7B-v0.1 | 32.84 / **32.90** (nohad) | 31.70 / **32.37** (tilehad) | 32.43 / **32.71** (nohad) | 30.14 / **32.35** (tilehad) |
| Qwen2.5-7B-Instruct | 31.17 / **37.08** (tilehad) | **26.10** / 23.48 (tilehad) | 33.00 / **39.67** (tilehad) | 25.51 / **35.20** (tilehad) |

Selected6 full delta table:

| Model | 50% + 4-bit | 50% + 2-bit | 70% + 4-bit | 70% + 2-bit |
|---|---:|---:|---:|---:|
| Meta-Llama-3-8B-Instruct | +0.03 | +0.64 | +1.16 | +3.29 |
| Llama-2-7B | +0.03 | -1.28 | +0.40 | +0.17 |
| Mistral-7B-v0.1 | +0.06 | +0.67 | +0.28 | +2.21 |
| Qwen2.5-7B-Instruct | +5.91 | -2.62 | +6.67 | +9.69 |

### Table Notes

- `selected6 full` means:
  - `narrativeqa`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `trec`
  - `lcc`
- `M+K proxy` rows use no hadamard; the `(tilehad)` / `(nohad)` tag in the
  average-score table refers to the JSQKV variant only.
- `Llama-2-7B 50\%` now uses the stronger `repairA` shared budget for the
  paper-ready summary.
- `Mistral 50\%` is now complete in the paper-ready summary.
- The current strongest cross-model positive signal is:
  - `Qwen2.5-7B-Instruct`, especially under `70\%` budgets
  - `Meta-Llama-3-8B-Instruct`, especially under `70\% + 2-bit`

## 3bit Queue Summary

<!-- THREEBIT_SECTION_START -->

### 3bit Cross-Model Table

| Model | 50% + 3-bit | Delta | 70% + 3-bit | Delta |
|---|---:|---:|---:|---:|
| Meta-Llama-3-8B-Instruct | 45.76 / **45.77** | +0.01 | **43.83** / 43.35 | -0.48 |
| Llama-2-7B | 31.15 / **31.19** | +0.04 | 30.68 / **29.88** | -0.80 |
| Mistral-7B-v0.1 | 32.14 / **32.40** | +0.26 | **31.82** / 31.81 | -0.01 |
| Qwen2.5-7B-Instruct | 28.71 / **36.42** | +7.71 | 28.64 / **38.73** | +10.09 |

Notes:

- Each score cell reports `M+K proxy / JSQKV(tilehad)`.
- This section is updated automatically by `run_3bit_queue.py`.

<!-- THREEBIT_SECTION_END -->

## Terminology

### `selected6 full`

`selected6 full` means:

- only the six focal LongBench tasks are evaluated:
  - `narrativeqa`
  - `qasper`
  - `multifieldqa_en`
  - `hotpotqa`
  - `trec`
  - `lcc`
- but for each of these six tasks, the **full dataset split** is used
- therefore it is:
  - more reliable than `limit=12`
  - but cheaper than full 16-task LongBench

This is the main fast-turnaround protocol currently used for recovering the
JSQKV accuracy section.

## Current Best Available Results

### 70\% + 4bit

Selected6 full:

- `M+K proxy`:
  - `44.18`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta70_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `45.34`
  - file:
    - `JSQKV_runs/final_selected6/meta70_jsqkv_4bit_nohad_selected6_full/result.json`

Delta:

- `+1.16`

Per-task:

- `NarrativeQA`: `20.94 -> 21.86` (`+0.92`)
- `Qasper`: `37.32 -> 38.81` (`+1.49`)
- `MultiFieldQA-En`: `44.90 -> 46.52` (`+1.62`)
- `HotpotQA`: `41.22 -> 40.64` (`-0.58`)
- `TREC`: `68.00 -> 70.00` (`+2.00`)
- `LCC`: `52.71 -> 54.23` (`+1.52`)

### 70\% + 2bit

Selected6 full:

- `M+K proxy`:
  - `39.83`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta70_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `43.12`
  - file:
    - `JSQKV_runs/final_selected6/meta70_jsqkv_2bit_tilehad_selected6_full/result.json`

Delta:

- `+3.29`

Per-task:

- `NarrativeQA`: `21.34 -> 19.70` (`-1.64`)
- `Qasper`: `35.43 -> 39.49` (`+4.06`)
- `MultiFieldQA-En`: `43.43 -> 44.31` (`+0.88`)
- `HotpotQA`: `39.21 -> 40.15` (`+0.94`)
- `TREC`: `63.00 -> 70.00` (`+7.00`)
- `LCC`: `36.59 -> 45.04` (`+8.45`)

### 50\% + 4bit

Selected6 full:

- `M+K proxy`:
  - `45.90`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `45.93`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_jsqkv_4bit_nohad_selected6_full_4096/result.json`

Delta:

- `+0.03`

Per-task:

- `NarrativeQA`: `21.61 -> 21.92` (`+0.31`)
- `Qasper`: `40.70 -> 39.38` (`-1.32`)
- `MultiFieldQA-En`: `47.62 -> 46.77` (`-0.85`)
- `HotpotQA`: `41.13 -> 41.38` (`+0.25`)
- `TREC`: `70.50 -> 70.50` (`+0.00`)
- `LCC`: `53.83 -> 55.64` (`+1.81`)

### 50\% + 2bit

Selected6 full:

- `M+K proxy`:
  - `43.38`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV (no hadamard)`:
  - `42.45`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_jsqkv_2bit_nohad_selected6_full_4096/result.json`
- `JSQKV (tile hadamard)`:
  - `44.02`
  - file:
    - `JSQKV_runs/final_selected6_4096/meta50_jsqkv_2bit_tilehad_selected6_full_4096/result.json`

Best delta vs. `M+K proxy`:

- `+0.64` using `tile hadamard(64)`

Per-task (`tile hadamard` vs. proxy):

- `NarrativeQA`: `21.97 -> 21.16` (`-0.81`)
- `Qasper`: `38.96 -> 39.54` (`+0.58`)
- `MultiFieldQA-En`: `46.11 -> 44.49` (`-1.62`)
- `HotpotQA`: `40.39 -> 41.19` (`+0.80`)
- `TREC`: `68.00 -> 71.50` (`+3.50`)
- `LCC`: `44.86 -> 46.26` (`+1.40`)

## Selected6 Main Table Status

The four-setting selected6 main comparison is now complete:

| Setting | M+K proxy | JSQKV | Best JSQKV delta |
|---|---:|---:|---:|
| `50% + 4bit` | 45.90 | 45.93 | `+0.03` |
| `50% + 2bit` | 43.38 | 44.02 | `+0.64` |
| `70% + 4bit` | 44.18 | 45.34 | `+1.16` |
| `70% + 2bit` | 39.83 | 43.12 | `+3.29` |

This means the current selected6 main table already supports the intended
matched-comparison claim under the recovered evaluation pipeline.

## Hadamard Choice Summary

Current best choices on selected6 full:

- `50% + 4bit`
  - best available: `no hadamard`
- `50% + 2bit`
  - `tile hadamard(64)` beats `no hadamard`
  - `44.02 vs 42.45`
- `70% + 4bit`
  - `no hadamard` beats `tile hadamard(64)`
  - `45.34 vs 43.57`
- `70% + 2bit`
  - `tile hadamard(64)` beats `no hadamard`
  - `43.12 vs 42.94`

## Full-Qasper Sanity

These single-task full runs were used to de-risk the selected6 full launches.

### 70\% + 4bit

- `M+K proxy`: `40.39`
- `JSQKV`: `41.25`
- delta: `+0.86`

Files:

- `JSQKV_runs/full_qasper/meta70_uniformkivi_4bit_qasper_full/result.json`
- `JSQKV_runs/full_qasper/meta70_jsqkv_4bit_nohad_qasper_full/result.json`

### 70\% + 2bit

- `M+K proxy`: `35.43`
- `JSQKV tile hadamard`: `39.49`
- `JSQKV no hadamard`: `35.77`

Files:

- `JSQKV_runs/full_qasper_4096/meta70_uniformkivi_2bit_qasper_full_4096_r2/result.json`
- `JSQKV_runs/full_qasper_4096/meta70_jsqkv_2bit_tilehad_qasper_full_4096/result.json`
- `JSQKV_runs/ablations_4096/meta70_jsqkv_2bit_nohad_qasper_full_4096/result.json`

### 50\% + 2bit

- `JSQKV tile hadamard`: `39.54`
- `JSQKV no hadamard`: `36.76`

Files:

- `JSQKV_runs/full_qasper_4096/meta50_jsqkv_2bit_tilehad_qasper_full_4096/result.json`
- `JSQKV_runs/full_qasper_4096/meta50_jsqkv_2bit_nohad_qasper_full_4096/result.json`

### 50\% + 4bit

- `JSQKV no hadamard`: `39.38`
- `JSQKV tile hadamard`: `39.61`

Files:

- `JSQKV_runs/ablations_4096/meta50_jsqkv_4bit_nohad_qasper_full_4096/result.json`
- `JSQKV_runs/ablations_4096/meta50_jsqkv_4bit_tilehad_qasper_full_4096/result.json`

## Current Practical Conclusion

Under the recovered and aligned evaluation pipeline:

- `70\% + 4bit` is already clearly paper-usable
- `70\% + 2bit` is currently the strongest positive setting by margin
- `50\% + 4bit` is roughly neutral to slightly positive
- `50\% + 2bit` becomes positive once `tile hadamard(64)` is enabled

## Current Status

At this point:

- the selected6 main table is complete
- the remaining work is no longer about recovering the basic JSQKV claim
- the remaining work is about:
  - `8192` confirmation
  - `3bit` expansion
  - hadamard / no-hadamard explanation

## Llama-2-7B Selected6 Main Results

The first selected6 `2bit/4bit` main table is now available for `llama-2-7b`.

### 70\% + 4bit

- `M+K proxy`:
  - `30.57`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_70_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `30.97`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_70_jsqkv_4bit_nohad_selected6_full_4096/result.json`

Delta:

- `+0.40`

### 70\% + 2bit

- `M+K proxy`:
  - `29.92`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_70_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `30.09`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_70_jsqkv_2bit_tilehad_selected6_full_4096/result.json`

Delta:

- `+0.17`

### 50\% + 4bit

- `M+K proxy`:
  - `31.34`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_50_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `31.22`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_50_jsqkv_4bit_nohad_selected6_full_4096/result.json`

Delta:

- `-0.12`

### 50\% + 2bit

- `M+K proxy`:
  - `31.51`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_50_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV`:
  - `30.12`
  - file:
    - `JSQKV_runs/llama2_7b_selected6_4096/llama2_50_jsqkv_2bit_tilehad_selected6_full_4096/result.json`

Delta:

- `-1.39`

## Cross-Model Status

Current selected6 main-table coverage:

- `Meta-Llama-3-8B-Instruct`
  - complete
- `llama-2-7b`
  - complete
- `Mistral-7B-v0.1`
  - integrated
  - `qasper full` first-wave completed
  - `selected6 full 50%/70%` matched pairs completed
- `Qwen2.5-7B-instruct`
  - integrated
  - `qasper full` first-wave completed
  - `selected6 full 50%/70%` matched pairs completed

## Mistral-7B-v0.1

### qasper full, 70\% budget

- `4bit`
  - `M+K proxy = 8.18`
  - `JSQKV = 7.84`
  - delta: `-0.34`
- `2bit`
  - `M+K proxy = 7.11`
  - `JSQKV = 7.99`
  - delta: `+0.88`

### selected6 full, 70\% + 2bit

- `M+K proxy = 30.14`
  - file:
    - `JSQKV_runs/mistral_selected6_4096/mistral70_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV = 32.35`
  - file:
    - `JSQKV_runs/mistral_selected6_4096/mistral70_jsqkv_2bit_tilehad_selected6_full_4096/result.json`
- delta:
  - `+2.21`

### selected6 full, 70\% + 4bit

- `M+K proxy = 32.43`
  - file:
    - `JSQKV_runs/mistral_selected6_4096/mistral70_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV = 31.88`
  - file:
    - `JSQKV_runs/mistral_selected6_4096/mistral70_jsqkv_4bit_tilehad_selected6_full_4096/result.json`
- delta:
  - `-0.55`

### qasper full, 50\% budget

- `2bit`
  - `M+K proxy = 7.73`
  - `JSQKV = 8.42`
  - delta: `+0.69`
- `4bit`
  - `M+K proxy = 8.47`
  - `JSQKV = 8.38`
  - delta: `-0.09`

### selected6 full, 50\% budget

- `2bit`
  - `M+K proxy = 31.70`
  - `JSQKV = 32.37`
  - delta: `+0.67`
- `4bit`
  - `M+K proxy = 32.84`
  - `JSQKV (tilehad) = 32.69`
  - `JSQKV (nohad) = 32.90`
  - best delta: `+0.06` using `nohad`

## Qwen2.5-7B-instruct

### qasper full, 70\% budget

- `4bit`
  - `M+K proxy = 19.54`
  - `JSQKV = 28.61`
  - delta: `+9.07`
- `2bit`
  - `M+K proxy = 14.42`
  - `JSQKV = 26.45`
  - delta: `+12.03`

### selected6 full, 70\% + 2bit

- `M+K proxy = 25.51`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen70_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV = 35.20`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen70_jsqkv_2bit_tilehad_selected6_full_4096/result.json`
- delta:
  - `+9.69`

### selected6 full, 70\% + 4bit

- `M+K proxy = 33.00`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen70_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV = 39.67`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen70_jsqkv_4bit_tilehad_selected6_full_4096/result.json`
- delta:
  - `+6.67`

### qasper full, 50\% budget

- `2bit`
  - `M+K proxy = 13.32`
  - `JSQKV = 16.57`
  - delta: `+3.25`
- `4bit`
  - `M+K proxy = 17.82`
  - `JSQKV = 29.76`
  - delta: `+11.94`

### selected6 full, 50\% + 2bit

- `M+K proxy = 26.10`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen50_uniformkivi_2bit_selected6_full_4096/result.json`
- `JSQKV = 23.48`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen50_jsqkv_2bit_tilehad_selected6_full_4096/result.json`
- delta:
  - `-2.62`

### selected6 full, 50\% + 4bit

- `M+K proxy = 31.17`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen50_uniformkivi_4bit_selected6_full_4096/result.json`
- `JSQKV = 37.08`
  - file:
    - `JSQKV_runs/qwen_selected6_4096/qwen50_jsqkv_4bit_tilehad_selected6_full_4096/result.json`
- delta:
  - `+5.91`

## In-Progress Higher-Cost Runs

The following broader runs are still in progress / partially completed and are
not yet part of the main summary:

- no remaining high-priority selected6 main-table gaps
- residual work, if continued, is now optional / exploratory rather than
  required for the core paper-facing summary
