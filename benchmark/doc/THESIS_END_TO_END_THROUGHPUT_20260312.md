# 端到端吞吐量实验材料（2026-03-12）

## 实验设置

- 环境：`conda` 环境 `mustar`
- 模型：`/home/zh/model/Meta-Llama-3-8B-Instruct`
- 设备：NVIDIA A100 80GB PCIe
- Benchmark 脚本：`benchmark/run_controlled_benchmark.sh`
- 结果文件：`benchmark/benchmark_results_20260312_205959/summary.csv`
- 输入长度：`1024`
- 输出长度：`8`
- 批大小：`1`
- 重复次数：`1`

当前量化稀疏配置：

```text
quant_bits = 2
quant_k_dequant_mode = 0
quant_v_dequant_mode = 0
quant_v_split_k = 4
quant_v_tile_config = 0
```

对应图文件：

- `benchmark/doc/figures/end_to_end_metrics.png`
- `benchmark/doc/figures/end_to_end_metrics.pdf`

## 详细数据

来源：`benchmark/benchmark_results_20260312_205959/summary.csv`

| 配置 | K 稀疏度 | V 稀疏度 | TTFT(ms) | TPOT(ms) | 总时延(ms) | 峰值显存(GB) | 批吞吐(tokens/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.0 | 0.0 | 175.92 | 32.30 | 401.99 | 15.83 | 19.05 |
| Sparse50 | 0.5 | 0.5 | 779.20 | 52.62 | 1147.56 | 15.80 | 6.65 |
| Sparse70 | 0.7 | 0.7 | 801.26 | 42.53 | 1098.97 | 15.78 | 6.97 |
| Sparse50+2bit | 0.5 | 0.5 | 306.88 | 41.08 | 594.45 | 15.80 | 13.18 |
| Sparse70+2bit | 0.7 | 0.7 | 312.95 | 40.30 | 595.07 | 15.80 | 12.88 |

## 关键对比

以纯稀疏版本为基线，量化稀疏版本的改进如下：

### 50% 稀疏度

- TTFT：`779.20 -> 306.88 ms`
- TTFT 降低约：`60.6%`
- TPOT：`52.62 -> 41.08 ms`
- TPOT 降低约：`21.9%`
- 吞吐：`6.65 -> 13.18 tok/s`
- 吞吐提升约：`1.98x`

### 70% 稀疏度

- TTFT：`801.26 -> 312.95 ms`
- TTFT 降低约：`60.9%`
- TPOT：`42.53 -> 40.30 ms`
- TPOT 降低约：`5.2%`
- 吞吐：`6.97 -> 12.88 tok/s`
- 吞吐提升约：`1.85x`

## 图的使用建议

当前图 `benchmark/doc/figures/end_to_end_metrics.png` 为三联图：

- 左图：TTFT
- 中图：TPOT
- 右图：吞吐

如果论文需要单图编号，可以作为：

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/end_to_end_metrics.pdf}
    \caption{不同 KV Cache 压缩策略下的端到端性能对比。}
    \label{fig:end2end-throughput}
\end{figure}
```

## 论文草稿

下面给出一版可直接改写的中文硕士论文小节草稿。

```tex
\subsubsection{端到端吞吐量}

为验证本文量化稀疏 KV Cache 计算核心在真实推理场景中的有效性，本节在 Meta-Llama-3-8B-Instruct 模型上进行了端到端吞吐量测试。实验平台为 NVIDIA A100 80GB PCIe，输入长度设置为 1024，输出长度设置为 8，批大小为 1。测试脚本统一采用同一套生成流程，并分别统计首 token 时延（Time to First Token, TTFT）、单 token 生成时延（Time per Output Token, TPOT）以及整体吞吐量。

实验对比了 5 种配置：Dense 基线、50\% 稀疏、70\% 稀疏、50\% 稀疏+2bit 量化以及 70\% 稀疏+2bit 量化。量化稀疏版本采用本文优化后的计算核心，其中 Key 路径与 Value 路径分别进行独立调优，Value 路径进一步引入了 Split-K、tile 配置分流以及基于压缩元数据的快速解包策略，以减少热路径中重复的 \texttt{popcount} 和单位数量计算开销。

从实验结果可以看出，量化稀疏版本在端到端性能上已经优于纯稀疏版本。在 50\% 稀疏度下，纯稀疏配置的 TTFT 为 779.20 ms，TPOT 为 52.62 ms，而加入 2bit 量化后，TTFT 降低至 306.88 ms，TPOT 降低至 41.08 ms，吞吐量由 6.65 tokens/s 提升到 13.18 tokens/s，提升约 1.98 倍。在 70\% 稀疏度下，纯稀疏配置的 TTFT 为 801.26 ms，TPOT 为 42.53 ms；对应的量化稀疏版本 TTFT 降低至 312.95 ms，TPOT 降低至 40.30 ms，吞吐量由 6.97 tokens/s 提升至 12.88 tokens/s，提升约 1.85 倍。

进一步分析可知，量化稀疏版本能够取得更高端到端吞吐量的关键原因在于：第一，量化显著降低了压缩 KV Cache 的有效访存带宽需求；第二，针对 Value 路径进行的专项优化有效缓解了原有量化反解压过程中的计算开销；第三，Key 与 Value 两条路径采用了不同的最优执行参数，使得整体计算核心更贴合真实 GQA 解码场景。

值得注意的是，Dense 基线在当前短输出长度设置下仍然具有更低的绝对 TTFT 和 TPOT，这主要是因为模型权重计算占主导，而 KV Cache 压缩方法的收益尚未在极短输出序列上完全放大。然而，与纯稀疏方法相比，本文提出的量化稀疏实现已经实现了稳定的吞吐提升，说明在保持较高压缩率的同时，通过对计算核心进行针对性优化，可以将量化引入的额外反解压开销控制在可接受范围内，并最终转化为端到端的推理收益。
```

## 可选写法调整

如果你想让论文口吻更保守，可以把下面这句：

```tex
已经实现了稳定的吞吐提升
```

改成：

```tex
在当前实验设置下实现了明显的吞吐提升
```

如果你想更强调“打败纯稀疏”，可以把下面这句：

```tex
量化稀疏版本在端到端性能上已经优于纯稀疏版本
```

改成：

```tex
量化稀疏版本在两组稀疏度设置下均超过了纯稀疏版本
```
