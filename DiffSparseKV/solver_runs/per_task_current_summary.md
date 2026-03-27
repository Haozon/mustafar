# Current Per-Task Solver Summary

This file tracks the latest finished `per-task` runs and full-dataset follow-ups.

## Finished tasks

| Task | Val Delta | Full Delta | Status |
|---|---:|---:|---|
| `hotpotqa` | 0.0 | 0.18999999999999773 | weak_positive |
| `lcc` | 6.100000000000001 | 1.3300000000000054 | positive |
| `multifieldqa_en` | -1.3900000000000006 | 0.7000000000000028 | positive |
| `narrativeqa` | -0.259999999999998 | 0.0 | fallback_uniform |
| `qasper` | -0.11000000000000654 | 0.14000000000000057 | weak_positive |
| `trec` | 5.0 | 2.5 | positive |

## Notes

- `lcc` and `trec` are stable full-dataset positives.
- `qasper`, `multifieldqa_en`, and `hotpotqa` all ended up non-negative on the full dataset after broader search or full re-evaluation.
- `narrativeqa` is finalized as a fallback-to-uniform task (`23.94`).