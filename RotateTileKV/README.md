# RotateTileKV

量化版复现目录，目标是先验证精度，不改 KV cache 内存格式。

当前实现：

- 在 `RoPE` 之后对 `Q/K` 可选执行 Hadamard 变换
- 在写入 KV cache 之前对 `K/V` 做 fake quant，再以解量化后的张量参与后续计算
- 支持三种量化粒度：
  - `per-token`
  - `per-token-head`
  - `per-token-tile`
- `per-token-tile` 默认 `tile_size = head_dim / 2`

快速运行示例：

```bash
python RotateTileKV/run_longbench.py \
  --model-name-or-path meta-llama/Llama-2-7b-hf \
  --k-bits 4 \
  --v-bits 4 \
  --quant-granularity per-token-tile \
  --enable-hadamard \
  --limit 10
```

如果想跑完整 16 个任务：

```bash
python RotateTileKV/run_longbench.py \
  --model-name-or-path meta-llama/Llama-2-7b-hf \
  --k-bits 4 \
  --v-bits 4 \
  --quant-granularity per-token-tile \
  --enable-hadamard \
  --full-longbench
```
