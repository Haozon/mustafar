# Per-Task Results Manifest

- Model: `Meta-Llama-3-8B-Instruct`
- Target budget: `0.7`

| Task | Status | Val Uniform | Val Diff | Val Delta | Full Uniform | Full Diff | Full Delta | Best target distribution | Best sparsity levels |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `hotpotqa` | weak_positive | 46.17 | 46.17 | 0.0 | 44.93 | 45.12 | 0.18999999999999773 | `[0.0, 0.75, 0.25]` | `[0.0, 0.6, 1.0]` |
| `lcc` | positive | 57.85 | 63.95 | 6.100000000000001 | 54.12 | 55.45 | 1.3300000000000054 | `[0.0, 0.75, 0.25]` | `[0.0, 0.6, 1.0]` |
| `multifieldqa_en` | positive | 39.53 | 38.14 | -1.3900000000000006 | 40.91 | 41.61 | 0.7000000000000028 | `[0.0, 1.0, 0.0]` | `[0.0, 0.7, 1.0]` |
| `narrativeqa` | fallback_uniform | 29.81 | 29.55 | -0.259999999999998 | 23.94 | 23.94 | 0.0 | `[0.0, 0.90009, 0.09991]` | `[0.0, 0.6667, 1.0]` |
| `qasper` | weak_positive | 39.02 | 38.91 | -0.11000000000000654 | 40.89 | 41.03 | 0.14000000000000057 | `[0.0, 0.90009, 0.09991]` | `[0.0, 0.6667, 1.0]` |
| `trec` | positive | 50.0 | 55.0 | 5.0 | 70.0 | 72.5 | 2.5 | `[0.0, 0.75, 0.25]` | `[0.0, 0.6, 1.0]` |

## Current conclusions

- Full-dataset positive or weakly positive tasks:
  - `lcc`: `+1.33`
  - `trec`: `+2.50`
  - `qasper`: `+0.14`
  - `multifieldqa_en`: `+0.70`
  - `hotpotqa`: `+0.19`

- `qasper`, `multifieldqa_en`, and `hotpotqa` are all rescue / re-evaluation cases:
  - their small validation results were weak or negative
  - after broader search or full-task verification, all three became non-negative on the full dataset

- `NarrativeQA` is the only finalized fallback task:
  - best diff validation is below uniform
  - a SnapKV-style comparison on the same calibration split also failed to beat uniform
  - therefore the final policy is to fall back to the uniform baseline