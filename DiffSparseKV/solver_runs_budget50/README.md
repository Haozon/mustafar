# Meta-Llama-3-8B Budget50 Summary

当前目录保存 `Meta-Llama-3-8B-Instruct` 在目标稀疏度 `50%` 下的基础 per-task 搜索结果。

## 实验设置

- Model: `Meta-Llama-3-8B-Instruct`
- Target budget: `0.50`
- Calibration size: `8`
- Validation size: `20`
- Seeds: `17 / 29`
- Search grid:
  - `p0 ∈ {0.0, 0.05, 0.10}`
  - `rho1 ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65}`
- Fixed strategy:
  - `importance_mode = value_aware`
  - `head_aggregation_mode = max`
  - `value_sink_keep = 2`
  - `level_2_mode = evict`

## 结果文件

- `rep6_budget50_try1_per_task_summary.json`
- `rep6_budget50_try1_per_task_results.csv`
