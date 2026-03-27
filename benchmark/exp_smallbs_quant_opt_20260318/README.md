# Small-BS Quant Opt Results

该目录汇总了当前阶段“小 batch 场景量化优化”的五线端到端结果归档版本。目录名保留了
`smallbs`，但当前内容实际覆盖 `BS=1..8` 的完整五组配置对比，并以 2026-03-21 的隔离式
重跑结果为准。

实验设置：

- 模型：`Meta-Llama-3-8B-Instruct`
- 输入长度：`4096`
- 输出长度：`256`
- Batch Size：`1,2,3,4,5,6,7,8`
- 对比配置：
  - `Dense`
  - `Sparse50`
  - `Sparse70`
  - `Sparse50+2bit`
  - `Sparse70+2bit`

当前采用的量化路径配置：

- `Sparse50+2bit`:
  - 启用 `quant_k_use_meta = true`
  - `BS=3,4,6,7` 使用 `quant_v_split_k=8`
  - `BS=3,4,6,7` 使用 `quant_v_tile_config=1`
  - `BS=1,2,5,8` 保持默认 `quant_v_split_k=4, quant_v_tile_config=0`
- `Sparse70+2bit`:
  - 启用 `quant_k_use_meta = true`
  - `BS=3,4,6,7,8` 使用 `quant_v_split_k=8`
  - `BS=3,4,6,7,8` 使用 `quant_v_tile_config=1`
  - `BS=5` 使用 `quant_v_split_k=4, quant_v_tile_config=1`
  - `BS=1,2` 保持默认 `quant_v_split_k=4, quant_v_tile_config=0`
- `quant_v_decode_n1=True` 已验证为负收益，未纳入最终配置

主要文件：

- `dense_summary.csv`
- `sparse50_summary.csv`
- `sparse70_summary.csv`
- `quant50_summary.csv`
- `quant70_summary.csv`
- `compression_stats_bs1_8.csv`
- `compare_five_configs_bs1_8.csv`
- `smallbs_quant_opt_results.json`
- `throughput_vs_bs_smallbs_quant_opt.pdf`
- `throughput_vs_bs_smallbs_quant_opt.png`
- `compression_time_vs_bs.pdf`
- `compression_ratio_vs_bs.pdf`
- `paper_ready/`

说明：

- 当前结果来自共享状态下的 `NVIDIA A100 80GB PCIe`，但使用了 `benchmark_throughput_isolated.py`
  对每个 `BS` 单独加载模型并做 `3` 次重复，口径比早先混合拼接版更一致。
- `BS=1` 时最优配置仍为 `Dense`，吞吐量为 `28.55` tokens/s。
- `BS=2` 时最优配置转移到 `Sparse50+2bit`，吞吐量为 `42.76` tokens/s。
- `BS=3,4,5,6` 时最优配置均为 `Sparse70+2bit`，吞吐量分别为 `57.81 / 66.42 / 74.22 / 80.38`
  tokens/s。
- `BS=7` 与 `BS=8` 时，最优配置不再是量化路径，而分别变为 `Sparse50` 与 `Sparse70`。
- `Sparse50+2bit` 相比 `Sparse50` 在 `BS=1..5` 上保持领先，但从 `BS=6` 开始被 sparse-only 版本追平并反超。
- `Sparse70+2bit` 相比 `Sparse70` 在 `BS=1..7` 上保持领先，但在 `BS=8` 时被 `Sparse70` 反超。
- 这说明量化收益在小到中等 batch 区间最明显；在更大 batch 下，sparse-only 路径的吞吐扩展更强，量化带来的额外开销开始抵消收益。
- 本目录只保留汇总结果，不再保留中间 benchmark 目录。
- 当前实际完整结果来源于 `JSQKV_benchmark/benchmark_config_llama3_bs_4096_256_e2e_paper.yaml`。
