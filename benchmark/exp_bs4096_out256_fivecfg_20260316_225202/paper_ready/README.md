# Paper-Ready End-to-End Results

数据来源：

- [dense_summary.csv](../dense_summary.csv)
- [sparse50_summary.csv](../sparse50_summary.csv)
- [sparse70_summary.csv](../sparse70_summary.csv)
- [quant50_summary.csv](../quant50_summary.csv)
- [quant70_summary.csv](../quant70_summary.csv)

主要文件：

- `end_to_end_metrics_bs1_6.csv`
- `throughput_table_bs1_6.tex`
- `ttft_table_bs1_6.tex`
- `tpot_table_bs1_6.tex`
- `end_to_end_batchsize_section.tex`
- `end_to_end_batchsize_section_compact.tex`

说明：

- 吞吐量基于完整 `output_length=256` 的生成过程统计。
- TTFT/TPOT 基于前 `64` 个 decode steps 的独立计时窗口统计，用于在单卡 `RTX 3090 24GB` 条件下稳定完成 batch sweep。
- 表格中的 `--` 表示该配置在对应 batch size 下未取得完整可比指标。
