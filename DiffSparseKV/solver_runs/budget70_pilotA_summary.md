# Budget Solver Summary: `budget70_pilotA`

## Objective

Find a lightweight differential-sparsity configuration under target budget `0.70`
that outperforms `uniform 70%` on a small calibration split, then validate it on
a larger held-out split.

## Search space

- Model: `Meta-Llama-3-8B-Instruct`
- Fixed hyperparameters:
  - `importance_mode = value_aware`
  - `head_aggregation_mode = max`
  - `value_sink_keep = 2`
  - `level_2_mode = evict`
- Candidate grids:
  - `p0 ∈ {0.05, 0.10, 0.15}`
  - `rho1 ∈ {0.60, 0.6667, 0.7143}`
- Derived analytically:
  - `p1`
  - `p2`

## Calibration split

- Datasets: `narrativeqa`, `qasper`, `multifieldqa_en`
- `10` random samples per dataset
- `seed = 42`

### Uniform baseline

- Average: `42.42`

### Best candidate

- `p0 = 0.05`
- `p1 = 0.750075`
- `p2 = 0.199925`
- `rho1 = 0.6667`
- Expected budget: `0.70`

Calibration scores:

- `multifieldqa_en`: `50.75`
- `narrativeqa`: `28.52`
- `qasper`: `53.01`
- Average: `44.09`

Calibration gain over uniform:

- `+1.67`

## Validation split

- Datasets: `narrativeqa`, `qasper`, `multifieldqa_en`, `hotpotqa`, `trec`, `lcc`
- `30` random samples per dataset
- `seed = 42`

### Uniform 70%

- `hotpotqa`: `51.24`
- `lcc`: `49.47`
- `multifieldqa_en`: `40.58`
- `narrativeqa`: `30.65`
- `qasper`: `43.23`
- `trec`: `60.00`
- Average: `45.86`

### Best DiffSparseKV config

- `hotpotqa`: `51.24`
- `lcc`: `47.43`
- `multifieldqa_en`: `42.21`
- `narrativeqa`: `30.79`
- `qasper`: `43.26`
- `trec`: `63.33`
- Average: `46.38`

Validation gain over uniform:

- `+0.52`

## Interpretation

- The solver succeeded in finding a configuration that is better than `uniform 70%`.
- The gain generalizes from the tiny calibration split (`+1.67`) to a larger validation split (`+0.52`).
- The winning configuration is very close to the previously hand-tuned `default_3level` family:
  - approximately `5% dense`, `75% medium`, `20% evict`
  - medium sparsity about `66.7%`

## Key files

- Search table:
  - `solver_runs/budget70_pilotA_calibration_results.csv`
- Search summary:
  - `solver_runs/budget70_pilotA_summary.json`
- This readable summary:
  - `solver_runs/budget70_pilotA_summary.md`

