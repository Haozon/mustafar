# Paper-Ready Small-BS Quant-Opt Results

数据来源：

- [dense_summary.csv](../dense_summary.csv)
- [sparse50_summary.csv](../sparse50_summary.csv)
- [sparse70_summary.csv](../sparse70_summary.csv)
- [quant50_summary.csv](../quant50_summary.csv)
- [quant70_summary.csv](../quant70_summary.csv)
- [compare_five_configs_bs1_8.csv](../compare_five_configs_bs1_8.csv)

主要文件：

- `end_to_end_metrics_bs1_8.csv`
- `throughput_table_bs1_8.tex`
- `ttft_table_bs1_8.tex`
- `tpot_table_bs1_8.tex`
- `end_to_end_batchsize_section.tex`
- `end_to_end_batchsize_section_compact.tex`
- `compression_stats_bs1_8.csv`
- `compression_summary_bs1_4_8.csv`
- `compression_summary_table_bs1_4_8.tex`
- `compression_module_section.tex`
- `compression_module_section_compact.tex`

图文件：

- `../throughput_vs_bs_smallbs_quant_opt.pdf`
- `../throughput_vs_bs_dense_sparse50_quant50.pdf`
- `../throughput_vs_bs_dense_sparse70_quant70.pdf`
- `../compression_time_vs_bs.pdf`
- `../compression_ratio_vs_bs.pdf`

说明：

- 吞吐量基于完整 `output_length=256` 的生成过程统计。
- 本组材料覆盖 `BS=1..8` 的完整五组配置对比，当前版本基于 2026-03-21 的隔离式重跑结果。
- 本轮结果显示量化路径的优势主要集中在 `BS=2..6`；在 `BS=7,8` 上，sparse-only 配置重新取得最优吞吐量。
- 压缩模块统计直接插入真实模型的 prefill 压缩路径中，记录了原始 KV 大小、压缩后 KV 大小与压缩时间。
- 当前未采集 TTFT/TPOT，因此相关表格使用 `--` 作为占位。
- 本组结果用于论文草稿整理；定稿时建议在独占 GPU 条件下复测后替换数值。
