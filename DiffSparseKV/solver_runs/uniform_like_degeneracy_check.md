# Uniform-like Degeneracy Check

This note records a direct check of whether the current `diff_sparse_kv`
implementation naturally degenerates to the true MUSTAFAR `uniform` path when
the configuration is set to:

- `target_distribution = [0.0, 1.0, 0.0]`
- `sparsity_levels = [0.0, 0.7, 1.0]`

Interpretation:

- No dense bucket
- No explicit eviction bucket
- All tokens fall into the middle bucket with `70%` feature sparsity

## NarrativeQA calibration check

Calibration split:

- search tag: `narrative_joint_search1`
- task: `narrativeqa`
- split file:
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/narrative_joint_search1_per_task_indices/narrativeqa_calibration_indices.json`

Uniform baseline result:

- result file:
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_uniform_0.70_narrative_joint_search1_narrativeqa_uniform_calib/result.json`
- score:
  - `9.50`

Uniform-like diff_sparse results:

- attention_only + mean + sink=0
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_attention_only_head_mean_sink_0/result.json`
  - `8.76`

- attention_only + mean + sink=2
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_attention_only_head_mean_sink_2/result.json`
  - `8.76`

- attention_only + max + sink=0
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_attention_only_head_max_sink_0/result.json`
  - `8.76`

- attention_only + max + sink=2
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_attention_only_head_max_sink_2/result.json`
  - `8.76`

- value_aware + mean + sink=0
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_value_aware_head_mean_sink_0/result.json`
  - `8.76`

- value_aware + mean + sink=2
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_value_aware_head_mean_sink_2/result.json`
  - `8.76`

- value_aware + max + sink=0
  - `/mnt/home/zh/mustafar/DiffSparseKV/solver_runs/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.70_narrative_joint_search1_narrativeqa_cand5_p0_0p00_p1_1p00_p2_0p00_rho1_0p7000_imp_value_aware_head_max_sink_0/result.json`
  - `8.76`

## Conclusion

For the current implementation, setting the diff_sparse configuration to the
uniform-like special case does **not** reproduce the true MUSTAFAR `uniform`
baseline numerically on the same split:

- true uniform: `9.50`
- uniform-like diff_sparse: `8.76`

So the current implementation supports an *approximate* degeneration toward
uniform, but not an *exact* degeneration to the true uniform path.
