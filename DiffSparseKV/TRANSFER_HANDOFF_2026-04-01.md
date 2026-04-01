# Transfer Handoff 2026-04-01

## Current State

- All experiment processes have been stopped on request.
- GPU is idle again:
  - used `~4 MB`
  - free `~81150 MB`
  - utilization `0%`
- `solver_runs/per_task_current_summary.md` has been synced to the latest results currently on disk.

## Important Files

- Main summary:
  - `solver_runs/per_task_current_summary.md`
- Running work log:
  - `SCRIPT_UPDATES.md`
- Generic section updater:
  - `solver_runs/update_model_section.py`
- Generic completion / repair driver:
  - `solver_runs/run_section_completion.py`
- Slot watcher:
  - `solver_runs/run_when_slot_free.py`
- Focused completion driver:
  - `solver_runs/run_llama2_7b_70_completion.py`

## Completed Sections

- `Meta-Llama-3-8B 50%`
  - full results complete for all 6 tasks
  - average full delta: `+0.15`
- `Mistral-7B 50%`
  - full results complete for all 6 tasks
  - average full delta: `-0.08`
- `Mistral-7B 70%`
  - repaired full results are available for all 6 tasks
  - average full delta: `+0.05`

## Partially Completed Sections

- `Llama-2-7B 70%`
  - full results complete for all 6 tasks
  - current full deltas:
    - `hotpotqa +0.39`
    - `lcc +2.70`
    - `multifieldqa_en +3.31`
    - `qasper +0.19`
    - `trec +1.00`
    - `narrativeqa -0.21`
  - only remaining negative task: `narrativeqa`

- `Llama-2-13B 70%`
  - full results complete for all 6 tasks
  - current full deltas:
    - `hotpotqa -3.32`
    - `lcc +7.52`
    - `multifieldqa_en +3.94`
    - `narrativeqa +4.16`
    - `qasper +1.35`
    - `trec +19.00`
  - only remaining negative task: `hotpotqa`

- `Llama-2-7B 50%`
  - current full-task state:
    - `hotpotqa 7.45 -> 7.15` (`-0.30`)
    - `lcc 66.88 -> 66.91` (`+0.03`)
    - `multifieldqa_en 20.90 -> 22.92` (`+2.02`)
    - `narrativeqa 15.04 -> 16.94` (`+1.90`)
    - `qasper 9.07 -> 8.52` (`-0.55`)
    - `trec` missing `diff full` result
  - unfinished / negative:
    - `hotpotqa`
    - `qasper`
    - `trec diff full`

- `Llama-2-13B 50%`
  - no full-task runs started yet

## Best Repair Snapshots

- `Llama-2-7B 70% / narrativeqa`
  - repair output root:
    - `solver_runs_llama2_7b_70_repair`
  - completed candidates when stopped:
    - `1695`
  - best calibration score:
    - `13.89`
  - best candidate config file:
    - `solver_runs_llama2_7b_70_repair/Llama-2-7b-hf_4096_diff_sparse_kv_0.70_llama2_7b_70_narrativeqa_repair_r1_narrativeqa_cand28_p0_0p10_p1_0p44_p2_0p46_rho1_0p5500_imp_value_aware_head_mean_sink_8_alpha_0.50_l2_evict_sel_diffsparse_phr_0.10_prr_1.00/sparsity_config.json`
  - current full baseline to beat:
    - `13.75`

- `Llama-2-13B 70% / hotpotqa`
  - repair output root:
    - `solver_runs_llama2_13b_70_repair`
  - completed candidates when stopped:
    - `17`
  - best calibration score:
    - `20.44`
  - best candidate config file:
    - `solver_runs_llama2_13b_70_repair/Llama-2-13b-hf_4096_diff_sparse_kv_0.70_Llama_2_13b_hf_70_hotpotqa_repair_r1_hotpotqa_cand1_p0_0p00_p1_0p55_p2_0p45_rho1_0p4500_imp_value_aware_head_max_sink_8_alpha_0.50_l2_evict_sel_diffsparse_phr_0.10_prr_0.75/sparsity_config.json`
  - current full baseline to beat:
    - `13.91`

- `Mistral-7B 70% / qasper`
  - wide-search output root:
    - `solver_runs_mistral_qasper_wide70`
  - completed candidates when stopped:
    - `776`
  - best calibration score:
    - `15.94`
  - best candidate config file:
    - `solver_runs_mistral_qasper_wide70/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_mistral_qasper_wide70_r1_qasper_cand11_p0_0p03_p1_0p79_p2_0p19_rho1_0p6500_imp_value_aware_head_mean_sink_8_alpha_0.50_l2_evict_sel_diffsparse_phr_0.10_prr_1.00/sparsity_config.json`
  - repaired full result already exists:
    - `solver_runs_mistral_repaired_full/Mistral-7B-Instruct-v0.1_8192_uniform_0.70_wide70_qasper_uniform_full/result.json`
    - `solver_runs_mistral_repaired_full/Mistral-7B-Instruct-v0.1_8192_diff_sparse_kv_0.70_wide70_qasper_diff_full/result.json`

## Recommended Resume Order

1. Finish `Llama-2-7B 70% / narrativeqa`
   - this section is otherwise complete
   - likely highest-value next step
2. Finish `Llama-2-13B 70% / hotpotqa`
   - this section is otherwise complete
3. Finish `Llama-2-7B 50%`
   - complete `trec diff full`
   - then repair `hotpotqa` and `qasper`
4. Start `Llama-2-13B 50%`
5. Revisit `Mistral-7B 50%` and the remaining negative repaired `Mistral-7B 70%` tasks if needed

## Recommended Resume Commands

```bash
/home/zh/miniconda3/envs/mustar/bin/python solver_runs/run_llama2_7b_70_completion.py
```

```bash
/home/zh/miniconda3/envs/mustar/bin/python solver_runs/run_section_completion.py \
  --section_title 'Llama-2-13B 70%' \
  --model_path /home/zh/model/Llama-2-13b-hf \
  --max_length 4096 \
  --budget 0.70 \
  --per_task_summary_json solver_runs_llama2_13b_budget70/rep6_budget70_try1_per_task_summary.json \
  --task_order hotpotqa,lcc,multifieldqa_en,narrativeqa,qasper,trec \
  --base_tag llama2_13b_70_full \
  --repair_output_root /mnt/home/zh/mustafar/DiffSparseKV/solver_runs_llama2_13b_70_repair \
  --repair_calib_limit 12 \
  --repair_val_limit 30
```

```bash
/home/zh/miniconda3/envs/mustar/bin/python solver_runs/run_section_completion.py \
  --section_title 'Llama-2-7B 50%' \
  --model_path /home/zh/model/Llama-2-7b-hf \
  --max_length 4096 \
  --budget 0.50 \
  --per_task_summary_json solver_runs_llama2_7b_budget50/rep6_budget50_try1_per_task_summary.json \
  --task_order hotpotqa,lcc,multifieldqa_en,narrativeqa,qasper,trec \
  --base_tag llama2_7b_50_full \
  --repair_output_root /mnt/home/zh/mustafar/DiffSparseKV/solver_runs_llama2_7b_50_repair \
  --repair_calib_limit 12 \
  --repair_val_limit 30
```

## Notes

- In this environment, detached `nohup` launches were unreliable more than once.
- The most reliable resume method was to start long-running jobs in a live session / PTY.
- `solver_runs/per_task_current_summary.md` is the main human-facing status file.
- `SCRIPT_UPDATES.md` is the detailed chronological work log.
