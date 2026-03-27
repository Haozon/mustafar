# SnapKV-style NarrativeQA Compare

- task: `narrativeqa`
- split: `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/narrative_joint_search1_per_task_indices/narrativeqa_calibration_indices.json`
- target budget: `0.7`

| Method | Score | Notes |
|---|---:|---|
| `uniform` | 9.50 | MUSTAFAR uniform baseline |
| `diff_uniform_like` | 8.76 | diff path with `[0,1,0] + rho1=0.7` |
| `snapkv_mean` | 5.64 | `selector_mode=snapkv` |
| `snapkv_max` | 8.76 | `selector_mode=snapkv` |
| `snapkv_top2_mean` | 8.76 | `selector_mode=snapkv` |
| `snapkv_hybrid` | 5.64 | `selector_mode=snapkv` |

## Conclusion

On the same `NarrativeQA` calibration split, neither the current `diff_sparse_kv`
uniform-like fallback nor the minimal SnapKV-style variants beat the true
MUSTAFAR `uniform` baseline:

- `uniform = 9.50`
- best non-uniform result here = `8.76`

This suggests that `NarrativeQA` is not simply suffering from an under-searched
budget template. Instead, the task itself appears less compatible with
observation-window-based KV compression under the current setup.
