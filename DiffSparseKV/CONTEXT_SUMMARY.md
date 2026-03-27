# DiffSparseKV / JSQKV 当前进展总结

## 1. 总体目标

当前工作的目标分成两层：

1. 算法层
   - 研究在固定平均稀疏度预算下，`Token-level` 差分稀疏是否能够优于 `Uniform` 稀疏。
   - 已经固定的设定：
     - `100%` 稀疏 = `Token Exit`
     - 不再考虑 `100% zero`

2. 论文层
   - 当前章节是 `JSQKV`，它不只是差分稀疏，还包括：
     - 差分稀疏
     - Per-Token-Tile 量化
     - Hadamard 稳定化
     - bitmap-based sparse-quant 格式
     - 稀疏-量化协同算子
   - 当前实验说明：
     - 仅靠差分稀疏还没有形成“稳定跨任务优于 uniform”的结果
     - 但 JSQKV 作为联合框架仍然可以成立

## 2. 已完成的代码修改

已经实现的核心能力：

- `value-aware importance`
- 多种 `head aggregation mode`
  - `mean`
  - `max`
  - `hybrid`
  - `top2_mean`
- `100% = token exit`
- 多级稀疏支持（不再硬编码 3 级）
- budget generator
- 随机子集采样
- 多随机 split 搜索脚本
- 一个最小版 `SnapKV` selector 模式

主要改动文件：

- `diffsparsekv/config.py`
- `diffsparsekv/llama_integration.py`
- `diffsparsekv/sparsity_applier.py`
- `diffsparsekv/threshold_manager.py`
- `diffsparsekv/budget_generator.py`
- `eval_diff_sparse_kv_longbench.py`
- `pred_long_bench_diff_sparse.py`
- `utils/process_args.py`
- `summarize_search_results.py`
- `multi_random_split_search.py`
- `run_diffsparse.sh`

## 3. 当前已经确定的策略

当前已经定下来的策略：

- `100%` 稀疏就是 `Token Exit`
- 不再讨论 `100% zero`
- `value-aware` 比 `attention_only` 稍微更稳，但提升有限
- `head=max` 是当前最稳的头聚合方式
- 4 级 family 没有稳定优于 3 级
- 当前最强的 family 仍然是：
  - `target_distribution = [0.05, 0.75, 0.20]`
  - `sparsity_levels = [0.0, 0.6667, 1.0]`
  - `importance_mode = value_aware`
  - `head_aggregation_mode = max`
  - `value_sink_keep = 2`
  - `level_2_mode = evict`

## 4. 当前的问题定位

当前最核心的问题不是 `Token Exit` 本身，而是：

- `selector / token importance` 的精度还不够
- 少量关键证据 token 被误删后，会把差分稀疏的收益抵消掉

当前更可信的判断是：

- `Token Exit` 在方向上是对的
- 整个框架是有潜力的
- 真正的瓶颈是 `eviction precision`

## 5. 已获得的全量结果

### 5.1 qasper，固定 70% 预算

- `uniform70`: `40.89`
- 当前最优 Diff：`41.03`

当前最优全量配置：

- `target_distribution = [0.05, 0.75, 0.20]`
- `sparsity_levels = [0.0, 0.6667, 1.0]`
- `importance_mode = value_aware`
- `head_aggregation_mode = max`
- `value_sink_keep = 2`
- `level_2_mode = evict`

对应结果文件：

- `tmp_eval/Meta-Llama-3-8B-Instruct_8192_uniform_0.70_qasper_full_uniform70/result.json`
- `tmp_eval/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_0.00_qasper_full_bestcfg70/result.json`

### 5.2 narrativeqa，固定 70% 预算

- `uniform70`: `23.94`
- 当前搜索得到的 best Diff70：`23.27`

对应结果文件：

- `tmp_eval/Meta-Llama-3-8B-Instruct_8192_uniform_0.70_narrativeqa_full_uniform70/result.json`
- `tmp_eval/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_budget_0.70_narrativeqa_full_best70/result.json`

### 5.3 multifieldqa_en，固定 70% 预算

- `uniform70`: `40.91`
- 当前搜索得到的 best Diff70：`40.36`

对应结果文件：

- `tmp_eval/Meta-Llama-3-8B-Instruct_8192_uniform_0.70_multifieldqa_en_full_uniform70/result.json`
- `tmp_eval/Meta-Llama-3-8B-Instruct_8192_diff_sparse_kv_budget_0.70_multifieldqa_en_full_best70/result.json`

## 6. 30 条样本搜索表

已经整理好的 30 条样本搜索表：

- `outputs/qasper30_search_table_real.md`
- `outputs/narrativeqa30_search_table_real.md`
- `outputs/multifieldqa_en30_search_table_real.md`

这些表说明：

- 小样本搜索经常能找到“看起来更优”的配置
- 但很多配置在全量上无法泛化
- 固定前 30 条样本的搜索很容易过拟合

## 7. qasper 上的多随机 split 搜索结果

为了降低“前 30 条过拟合”的影响，已经加入并跑过一个小规模多随机 split 搜索：

- 脚本：`multi_random_split_search.py`
- 评测脚本新增 `--sample_seed`

在 `qasper` 上，预算固定为 `70%`，使用 3 个随机 split、每个 split 30 条样本：

- `uniform70`
  - mean `37.01`
  - std `3.19`
  - scores `[40.53, 32.81, 37.70]`

- `default70_max`
  - `distribution = [0.05, 0.75, 0.20]`
  - `levels = [0.0, 0.6667, 1.0]`
  - `value-aware + head=max + evict`
  - mean `37.72`
  - std `3.29`
  - scores `[42.17, 34.33, 36.65]`

- `conservative70_max`
  - `distribution = [0.10, 0.70, 0.20]`
  - `levels = [0.0, 0.7143, 1.0]`
  - `value-aware + head=max + evict`
  - mean `38.30`
  - std `4.75`
  - scores `[44.32, 32.71, 37.86]`

之后把 `conservative70_max` 推到 `qasper` 全量验证：

- `uniform70` 全量：`40.89`
- `default70_max` 全量：`41.03`
- `conservative70_max` 全量：`40.28`

结论：

- `conservative70_max` 在随机 split 上均值更高
- 但推到全量后并没有超过 `uniform70`
- 目前更稳的 family 仍然是 `default70_max`

## 8. 已尝试过但效果不理想的方向

### 8.1 100% zero

已经确认：

- `100% zero` 明显不好
- 这条路线已经放弃

### 8.2 4 级 family

例如：

- `0 / 50 / 70 / 100`

当前结果：

- 没有稳定优于 3 级 family

### 8.3 没有 dense bucket 的 `50/70/100` family

代表性配置：

- `target_distribution = [0.20, 0.6667, 0.1333]`
- `levels = [0.5, 0.7, 1.0]`

观察：

- 小样本上有一点信号
- 全量 `qasper` 上没有超过当前 best family

结论：

- 保留一小部分 `0% dense` 桶目前更好

### 8.4 最小版 SnapKV-style selector

已经做了一个最小版 `selector_mode = snapkv`：

- `observation window top-k prefix`
- `recent window protection`

在 `qasper` 30 条、`70%` 预算下：

- `uniform70`: `41.89`
- 当前 best Diff70: `41.95`
- `snapkv selector`: `41.43`

结论：

- 静态 prefill-only 的 SnapKV 风格 selector 还不够
- 更可能缺的是 decode-time dynamic update / heavy hitter maintenance

## 9. 失败样本分析的观察

从全量 `uniform70` 与当前 best Diff70 的逐样本对比来看：

- 大多数样本其实是打平的
- 均值下降主要来自少量失败样本
- 这些失败样本的特征是：
  - 关键答案细节缺失
  - 实体名不完整
  - 语义接近但证据不准确
- 这更像是错误的 `Token Exit` 决策，而不是整体生成能力下降

这进一步支持当前判断：

- 真正的瓶颈是 `eviction precision`

## 10. 与 SnapKV / H2O 的关系理解

当前一个很重要的结论是：

- 当前 DiffSparseKV 的实现并不等价于 H2O / SnapKV
- 差别不在于“有没有做 token exit”
- 真正差别在于：
  - H2O / SnapKV 更依赖 decode-time 的动态 recent/heavy-hitter 维护
  - 当前主实验路径跑的是简化版 decode，而不是完整动态 selector

因此：

- 如果要真正把 token-level 稀疏做好
- 最有价值的下一步不是继续做静态 family 搜索
- 而是重建一个动态 decode-time selector

## 11. 论文写作建议

### 章节定位

不要把这一章写成：

- “差分稀疏全面显著优于 uniform”

更安全、也更符合现有结果的定位是：

- `JSQKV` 是一个联合稀疏-量化-算子协同框架
- 差分稀疏是其中一个关键模块
- 它在部分任务和配置下优于统一稀疏
- 其稳定收益依赖 token importance 估计精度
- 更稳的整体收益来自联合设计，而不是只靠差分稀疏本身

### 建议的实验结构

1. 稀疏模块评估
   - `DiffSparseKV vs Uniform vs Mustafar/H2O/SnapKV`

2. 联合压缩综合实验
   - `JSQKV vs Mustafar+KIVI`

3. 消融实验
   - `value-aware`
   - `head=max`
   - 不同 family
   - Hadamard / quantization granularity

### 避免的表述

不要写：

- “各模型和各设置下均一致优于 baseline”
- “已经找到统一最优配置”
- “只是没搜到，肯定还有更优配置”

更安全的写法：

- “在部分任务和配置下取得更优结果”
- “差分稀疏在固定预算下具有优于统一稀疏的潜力”
- “其稳定收益依赖 selector 精度”

## 12. 当前最值得继续做的事

如果继续推进，最有价值的方向是：

1. 重建动态 decode-time selector
   - recent window 强保护
   - 在线 heavy hitter 累积
   - 再对剩余 token 做差分分配

2. 继续多随机 split 搜索
   - 不再使用固定前 30 条样本搜索

3. 对被 exit 的 token 做定位分析
   - 看失败样本里被删掉的 token 是否命中证据区域

## 13. 当前最优可复用配置（暂时）

如果还沿用当前实现继续走，当前最优可复用配置是：

- `target_distribution = [0.05, 0.75, 0.20]`
- `sparsity_levels = [0.0, 0.6667, 1.0]`
- `importance_mode = value_aware`
- `head_aggregation_mode = max`
- `value_sink_keep = 2`
- `level_2_mode = evict`

它目前是：

- 唯一确认在全量 `qasper` 上略微超过 `uniform70` 的配置
- 同时在多随机 split 搜索里也相对稳健

## 14. 如果切换模型继续对话，建议直接从这里接着做

推荐的下一步任务：

1. 实现一个最小版 decode-time dynamic heavy-hitter protected set
2. 把论文实验部分改写成和当前结果一致的中文稿
3. 加日志记录被 exit 的 token 位置，分析失败样本
