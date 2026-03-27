# DiffSparseKV Thesis Materials

本目录整理了当前可直接放入论文的 DiffSparseKV 材料。

## 文件说明

- `diffsparsekv_experiment_section.tex`
  - 论文实验段落
  - 包含：
    - 配置生成器介绍
    - calibration / validation / full task set 协议
    - 代表任务结果表
    - 每任务配置表
    - NarrativeQA 上的 SnapKV-style 对照表
    - 当前实验结论

- `diffsparsekv_flow_figure.tex`
  - 一张 TikZ 流程图
  - 左图：配置生成器
  - 右图：DiffSparseKV 总体流程
  - 使用前请确保导言区包含：
    - `\usepackage{tikz}`
    - `\usetikzlibrary{positioning}`

## 主要数据来源

- `../per_task_results_manifest.json`
- `../snapkv_narrative_compare_summary.json`

## 当前 full-dataset 正例

- `lcc`
- `trec`
- `qasper`
- `multifieldqa_en`

## 当前中性/负例

- `hotpotqa`: neutral
- `narrativeqa`: negative
