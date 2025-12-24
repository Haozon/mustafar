0x2 LeanSparsityKV
稀疏算法 LeanKV 迁移：
每生成一个新token：
1. 将最新token加入recent window（避免过早压缩），该 recent window保证 256 的窗口而不被压缩
2. 当攒够 256 的窗口大小，就取出上一个窗口的所有 token，计算 该窗口内候选 token
3. 计算该 token 的重要性分数
4. 根据该重要性分数进行决策：
    1. 是否一个公式，能够是 该重要性分数 与 稀疏度的映射；
    2. 如果不存在，就同样采用阈值的方法进行决定；
5. 对整个窗口内的 token 进行稀疏 和 量化

Score & Sparsity 映射方案备选与设计
- 基于排名的分段映射
def token_score_to_sparsity(self, attention_scores, N):
    """
    Args:
        attention_scores: shape [B, H, T], 每个token的重要性分数
        N: 当前序列长度
    
    Returns:
        sparsity_levels: shape [B, H, T], 每个token对应的稀疏度
    """
    B, H, T = attention_scores.shape
    
    # 计算理论平均分数
    theoretical_avg = 1.0 / N
    
    # 定义分段阈值（可通过calibration调整）
    alpha_high = 2.0  # 高重要性阈值倍数
    alpha_mid = 0.5   # 中等重要性阈值倍数
    alpha_low = 0.1   # 低重要性阈值倍数
    
    # 初始化稀疏度张量
    sparsity_levels = torch.zeros_like(attention_scores)
    
    # 分段映射
    high_mask = attention_scores >= (alpha_high * theoretical_avg)
    mid_mask = (attention_scores >= (alpha_mid * theoretical_avg)) & \
               (attention_scores < (alpha_high * theoretical_avg))
    low_mask = (attention_scores >= (alpha_low * theoretical_avg)) & \
               (attention_scores < (alpha_mid * theoretical_avg))
    prune_mask = attention_scores < (alpha_low * theoretical_avg)
    
    # 分配稀疏度等级
    sparsity_levels[high_mask] = 0.0    # 不稀疏
    sparsity_levels[mid_mask] = 0.5     # 50%稀疏
    sparsity_levels[low_mask] = 0.75    # 75%稀疏
    sparsity_levels[prune_mask] = 1.0   # 完全剪枝
    
    return sparsity_levels

- 连续平滑映射
[图片]
def continuous_score_to_sparsity(self, attention_scores, N, mapping_type='exponential'):
    """
    连续映射函数
    """
    B, H, T = attention_scores.shape
    
    if mapping_type == 'exponential':
        # 指数映射
        score_max = attention_scores.max(dim=-1, keepdim=True)[0]
        normalized_scores = attention_scores / (score_max + 1e-8)
        
        beta = 3.0  # 控制衰减速度
        sparsity = 1.0 - torch.exp(-beta * normalized_scores)
        
    elif mapping_type == 'piecewise_linear':
        # 分段线性映射
        theoretical_avg = 1.0 / N
        tau_h = 2.0 * theoretical_avg
        tau_l = 0.5 * theoretical_avg
        
        sparsity = torch.zeros_like(attention_scores)
        
        # 高分区：不稀疏
        high_region = attention_scores > tau_h
        sparsity[high_region] = 0.0
        
        # 中分区：线性插值 0→0.5
        mid_region = (attention_scores >= tau_l) & (attention_scores <= tau_h)
        sparsity[mid_region] = 0.5 * (1 - (attention_scores[mid_region] - tau_l) / (tau_h - tau_l))
        
        # 低分区：线性插值 0.5→1.0
        low_region = attention_scores < tau_l
        sparsity[low_region] = 0.5 + 0.5 * (1 - attention_scores[low_region] / tau_l)
        
    return torch.clamp(sparsity, 0.0, 1.0)

- 基于百分位的自适应映射

由于我们的 baseline是 所有的token 都使用 70% 的稀疏，所以最好的方案是 基于 排名的分段映射，可以在一定程度上经过设计保持 与 baseline 相同的平均稀疏度。
理由如下：
- 与baseline 公平对比，可以有相同的平均稀疏度
- 理论基础扎实，LeanKV  已经经过验证了
- 可解释性强，易于实现，清晰的分层逻辑
- 灵活性好，4个 level 可以调整

连续平滑映射 和 百分位法，这两种方法 的不确定性太强，需要更多的实验去调优和寻找合适的映射公式，可能面临着更多的时间消耗。

Score & 分段映射 算法设计
目标： 在保持与baseline相同的平均稀疏度（70%）的前提下，根据token重要性分配不同的稀疏度，使得重要token保留更多信息。

核心假设
1. Token的重要性可以通过attention score量化
2. 重要token需要低稀疏度（保留更多信息）
3. 不重要token可以高稀疏度甚至完全剪枝
4. 使用 1/N 作为理论平均attention score进行归一化

算法流程伪代码
Algorithm 1: Adaptive Token-wise Sparsity for KV Cache (5-Level)

Input:
    K: Key states, shape [B, H, T, D]
    V: Value states, shape [B, H, T, D]  
    A: Attention weights from previous forward, shape [B, H, T_q, T_k]
    N: Current sequence length
    α_h, α_mh, α_m, α_ml: Threshold multipliers (hyperparameters)
    S_levels: Sparsity levels [s_0, s_1, s_2, s_3, s_4] = [0.0, 0.5, 0.7, 0.9, 1.0]

Output:
    K_sparse: Sparsified key states
    V_sparse: Sparsified value states
    Mask_pruned: Boolean mask indicating completely pruned tokens

// ============ Phase 1: Compute Token Importance ============
Function ComputeTokenImportance(A):
    // A is [B, H, T_q, T_k]
    // For each key token, compute how much it's attended to
    
    Importance = []
    For each token i in range(T_k):
        // Average attention score this token receives from all query tokens
        score_i = Mean(A[:, :, :, i])  // Average over T_q dimension
        
        // Alternative: Use max for GQA (as in LeanKV)
        // score_i = Max(A[:, :, :, i])
        
        Importance.append(score_i)
    
    Return Importance  // shape [B, H, T_k]


// ============ Phase 2: Assign Sparsity Levels (5-Level) ============
Function AssignSparsityLevels(Importance, N, α_h, α_mh, α_m, α_ml):
    // Importance is [B, H, T]
    
    // Compute theoretical average attention score
    avg_score = 1.0 / N
    
    // Define thresholds (从高到低)
    τ_h  = α_h × avg_score   // Highest importance threshold
    τ_mh = α_mh × avg_score  // Medium-high importance threshold
    τ_m  = α_m × avg_score   // Medium importance threshold
    τ_ml = α_ml × avg_score  // Medium-low importance threshold
    
    SparsityMap = EmptyTensor(shape=[B, H, T])
    
    For each token t in range(T):
        score_t = Importance[:, :, t]
        
        // 5-level classification
        If score_t >= τ_h:
            SparsityMap[:, :, t] = s_0  // Level 0: No sparsity (0%)
        Else If score_t >= τ_mh:
            SparsityMap[:, :, t] = s_1  // Level 1: Low sparsity (50%)
        Else If score_t >= τ_m:
            SparsityMap[:, :, t] = s_2  // Level 2: Medium sparsity (70%)
        Else If score_t >= τ_ml:
            SparsityMap[:, :, t] = s_3  // Level 3: High sparsity (90%)
        Else:
            SparsityMap[:, :, t] = s_4  // Level 4: Complete pruning (100%)
    
    Return SparsityMap  // shape [B, H, T]


// ============ Phase 3: Apply Magnitude-based Sparsity ============
Function ApplyMagnitudeSparsity(X, SparsityMap):
    // X is either K or V, shape [B, H, T, D]
    // SparsityMap is [B, H, T], each element indicates sparsity level
    
    B, H, T, D = X.shape
    X_sparse = ZeroTensor(shape=[B, H, T, D])
    
    // Process each sparsity level separately (exclude s_4 = complete pruning)
    For each sparsity_level in [s_0, s_1, s_2, s_3]:
        
        // Find all tokens assigned this sparsity level
        Mask_level = (SparsityMap == sparsity_level)  // [B, H, T]
        
        If sparsity_level == 0.0:
            // No sparsity: directly copy
            X_sparse[Mask_level, :] = X[Mask_level, :]
        
        Else:
            // Apply magnitude-based pruning
            num_to_keep = Floor((1 - sparsity_level) × D)
            
            For each token position (b, h, t) where Mask_level[b, h, t] == True:
                vector = X[b, h, t, :]  // Shape [D]
                
                // Find the threshold value: keep top-k by magnitude
                threshold = KthLargest(Abs(vector), k = num_to_keep)
                
                // Create magnitude mask
                mag_mask = (Abs(vector) >= threshold)
                
                // Apply mask
                X_sparse[b, h, t, :] = vector × mag_mask
    
    // Tokens with s_4 (100% sparsity) remain as zeros
    Mask_pruned = (SparsityMap == s_4)  // [B, H, T]
    
    Return X_sparse, Mask_pruned


// ============ Main Algorithm ============
Function AdaptiveKVCacheSparsification(K, V, A, N, α_h, α_mh, α_m, α_ml, S_levels):
    
    // Step 1: Compute token importance
    Importance = ComputeTokenImportance(A)
    
    // Step 2: Assign sparsity levels based on importance (5-level)
    SparsityMap = AssignSparsityLevels(Importance, N, α_h, α_mh, α_m, α_ml)
    
    // Step 3: Apply magnitude-based sparsity to keys
    K_sparse, Mask_pruned_K = ApplyMagnitudeSparsity(K, SparsityMap)
    
    // Step 4: Apply magnitude-based sparsity to values
    V_sparse, Mask_pruned_V = ApplyMagnitudeSparsity(V, SparsityMap)
    
    // Pruning mask should be consistent for K and V
    Assert(Mask_pruned_K == Mask_pruned_V)
    
    // Optional: Compute statistics
    avg_sparsity = Mean(SparsityMap)
    num_pruned = Sum(Mask_pruned_K)
    
    // Compute distribution of tokens across sparsity levels
    dist = []
    For each level in [s_0, s_1, s_2, s_3, s_4]:
        count = Sum(SparsityMap == level)
        percentage = count / (B × H × T)
        dist.append({
            "sparsity_level": level,
            "count": count,
            "percentage": percentage
        })
    
    Return K_sparse, V_sparse, Mask_pruned_K, {
        "avg_sparsity": avg_sparsity,
        "num_pruned_tokens": num_pruned,
        "total_tokens": B × H × T,
        "distribution": dist
    }

完整的算法流程（decoding阶段算法设计）
# === Prompt Phase (首次处理整个 prompt) ===
def compress_prompt(tokens, model):
    # Step 1: 先完整前向传播，获取 attention
    with torch.no_grad():
        outputs = model(tokens, output_attentions=True)
        attention_weights = outputs.attentions  # [num_layers, B, H, T, T]
    
    # Step 2: 对每个 layer 计算 token importance
    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx]  # [B, H, T, T]
        
        # 计算每个 token 的重要性 (被后续 token 关注的程度)
        importance = compute_importance(attn)  # [B, H, T]
        
        # Step 3: 分配稀疏度
        sparsity_map = assign_sparsity(importance, N=T)  # [B, H, T]
        
        # Step 4: 压缩 KV cache
        K_compressed = apply_magnitude_sparsity(K[layer_idx], sparsity_map)
        V_compressed = apply_magnitude_sparsity(V[layer_idx], sparsity_map)
        
        # 保存压缩后的 KV cache
        model.kv_cache[layer_idx] = (K_compressed, V_compressed)


# === Generation Phase (逐 token 生成) ===
### 这个 地方有点问题 需要优化，结合 SWA 的 decoding 
def compress_generation(new_token, model):
    # 每生成一个 token，使用**上一步的 attention**决定压缩策略
    
    # 将新 token 先加入 recent window (不压缩)
    recent_window.append(new_token)
    
    if len(recent_window) > W:  # W = 256
        # 候选 token 离开 recent window
        candidate_token = recent_window.pop(0)
        
        # 使用**上一步的 attention**评估重要性
        importance = last_attention_weights[:, :, candidate_token.position]
        
        # 分配稀疏度并压缩
        sparsity = assign_sparsity_single_token(importance, N=current_seq_len)
        compress_token(candidate_token, sparsity)

Importance Score 的计算方法
# 方法 A: 平均 (LeanKV 用这个)
def compute_importance(attention_weights):
    """
    attention_weights: [B, H, T_q, T_k]
    对于第 i 个 key token，计算它被所有 query token 关注的平均程度
    """
    importance = attention_weights.mean(dim=2)  # [B, H, T_k]
    return importance

# 方法 B: 最大 (用于 GQA)
def compute_importance_gqa(attention_weights):
    """
    GQA 中多个 query head 共享一个 KV head
    使用 max 聚合多个 query head 的 attention
    """
    # 假设 8 个 query heads 共享 1 个 KV head
    grouped = attention_weights.reshape(B, num_kv_heads, group_size, T_q, T_k)
    importance = grouped.mean(dim=3).max(dim=2)[0]  # [B, num_kv_heads, T_k]
    return importance

这一点我还不理解为什么 GQA 使用的是 max 来确定 importance  score.

参数校准策略
目标： 找到 $$(\alpha_h, \alpha_m, \alpha_l)$$使得平均稀疏度 ≈ 70%
1. 阈值的表达形式
Attention Score的归一化问题
由于softmax的归一化特性，对于长度为 $$N$$ 的序列，所有token的attention scores之和为1：
$$\sum_{i=1}^{N} \text{score}_i = 1$$
因此，理论平均attention score为：
$$\text{average score} = \frac{1}{N}$$
关键问题：理论平均score与序列长度 $N$ 成反比，导致：
- 短序列（小N）：平均score较大（如 N=10 时，平均为 0.1）
- 长序列（大N）：平均score较小（如 N=100 时，平均为 0.01）
如果使用固定阈值判断token重要性，会导致：
- 在短序列上过于宽松
- 在长序列上过于严格
归一化方案
为了消除序列长度的影响，LeanKV采用归一化策略：
归一化score定义：
$$\text{normalized\_score} = \text{score} \times N$$
阈值定义：
$$\text{threshold} = \frac{\alpha}{N}$$
判断条件：
$$\text{score} > \frac{\alpha}{N} \quad \Leftrightarrow \quad \text{score} \times N > \alpha$$

参数 α 的物理意义：
$$\alpha = \frac{\text{score}}{\text{理论平均score}} = \frac{\text{score}}{1/N} = \text{score} \times N$$
α 表示：token的重要性是理论平均值的倍数
- $$\alpha = 1.0$$：重要性等于平均水平
- $$\alpha = 3.0$$：重要性是平均水平的3倍（高重要性）
- $$\alpha = 0.5$$：重要性只有平均水平的一半（低重要性）
输入：序列长度N，token的attention score s
输出：稀疏度等级

normalized_score = s × N

If normalized_score >= α_h:
    → Level 1 (0% sparsity)
Else if normalized_score >= α_m:
    → Level 2 (50% sparsity)  
Else if normalized_score >= α_l:
    → Level 3 (85% sparsity)
Else:
    → Level 4 (100% pruned)
2. 问题建模
给定：
- $$m$$ 组 attention scores，每组有 $n$ 个 token
- 目标平均稀疏度：$\bar{s} = 70\%$
- $$k$$ 个稀疏度等级：$[s_1, s_2, ..., s_k]$，例如 $[0\%, 50\%, 85\%, 100\%]$
求解：
- $$k-1$$ 个阈值 $[\tau_1, \tau_2, ..., \tau_{k-1}]$
- 使得每个等级的token占比 $[a_1, a_2, ..., a_k]$ 满足：
$$\sum_{i=1}^{k} a_i \cdot s_i = \bar{s} = 70\%$$
$$\sum_{i=1}^{k} a_i = 1$$
3. 稀疏度方案配置
推荐采用5级方案：[0%, 50%, 70%, 90%, 100%]
理由：
1. 70%对应baseline：便于直接对比
2. 0%和100%覆盖极端：保证关键信息+激进剪枝
3. 50%和90%提供梯度：中间过渡
4. 只需4个阈值：优化难度适中
预期分布（需校准验证）：
- 0%： 10%（最重要）
- 50%：20%
- 70%：30%（类似baseline）
- 90%：20%
- 100%：20%（剪枝）
平均稀疏度验证：
$$0.1 \times 0 + 0.2 \times 50 + 0.3 \times 70 + 0.2 \times 90 + 0.2 \times 100 = 69\%$$

4. 校准策略方案
4.1 基于分位数的直接求解
基于分位数的阈值校准方案直接从attention score的经验分布中提取阈值，无需迭代优化。该方法的核心思想是：给定期望的token分布 $P_{\text{desired}} = [p_0, p_1, p_2, p_3, p_4]$，通过计算归一化importance score的分位数来确定分级阈值。
算法输入：
- 校准数据集（推荐使用MATH训练集的子集，200-500个样本）
- 目标平均稀疏度：$s_{\text{target}} = 70\%$
- 稀疏度等级：$S = [0\%, 50\%, 70\%, 90\%, 100\%]$
- 期望分布：$P_{\text{desired}} = [p_0, p_1, p_2, p_3, p_4]$
算法输出：
- 每个layer（或每个head）的阈值参数：$(\alpha_h, \alpha_{mh}, \alpha_m, \alpha_{ml})$
第一阶段：收集归一化importance scores
对校准数据集中的每个样本执行前向传播，获取attention weights。对于每个layer $l$（或每个attention head），计算所有token的importance score。具体地，对于第 $$i$$ 个token，其importance定义为它被后续token关注的平均程度：
$$\text{importance}_i = \frac{1}{N-i} \sum_{j=i+1}^{N} \text{attention\_score}_{j,i}$$
其中 $$N$$ 是序列长度。对于GQA架构，多个query head共享一个KV head时，使用max操作聚合多个query head的attention scores。
将importance score归一化以消除序列长度的影响：
$$\text{normalized\_score}_i = \text{importance}_i \times N$$
该归一化使得score的物理意义为"相对于理论平均值（$1/N$）的倍数"。收集校准数据集上所有样本、所有token的normalized scores，得到经验分布。
第二阶段：基于期望分布计算分位数阈值
给定期望的token分布 $P_{\text{desired}} = [p_0, p_1, p_2, p_3, p_4]$，其中 $$p_i$$ 表示希望分配到第 $$i$$ 级稀疏度的token占比。首先验证该分布是否满足目标平均稀疏度：
$$\sum_{i=0}^{4} p_i \cdot s_i = s_{\text{target}}$$
例如，若 $P_{\text{desired}} = [0.10, 0.20, 0.30, 0.20, 0.20]$，则平均稀疏度为：
$$0.10 \times 0\% + 0.20 \times 50\% + 0.30 \times 70\% + 0.20 \times 90\% + 0.20 \times 100\% = 69\%$$
计算累积分布以确定分位点：
$$\text{cumulative} = [p_0, p_0+p_1, p_0+p_1+p_2, p_0+p_1+p_2+p_3]$$
对于上述示例：$\text{cumulative} = [0.10, 0.30, 0.60, 0.80]$。
将收集到的所有normalized scores降序排序，计算对应分位数作为阈值：
$$\alpha_h = \text{Quantile}(\text{scores}, q = \text{cumulative}[0])$$
$$\alpha_{mh} = \text{Quantile}(\text{scores}, q = \text{cumulative}[1])$$
$$\alpha_m = \text{Quantile}(\text{scores}, q = \text{cumulative}[2])$$
$$\alpha_{ml} = \text{Quantile}(\text{scores}, q = \text{cumulative}[3])$$
这四个阈值自然满足单调性约束：$\alpha_h > \alpha_{mh} > \alpha_m > \alpha_{ml} > 0$。
第三阶段：验证与调整
使用得到的阈值在校准数据集上统计实际的token分布。对于每个token，根据其normalized score $$s$$ 分配到对应等级：
- 若 $s \geq \alpha_h$：Level 0（0%稀疏度）
- 若 $\alpha_{mh} \leq s < \alpha_h$：Level 1（50%稀疏度）
- 若 $\alpha_m \leq s < \alpha_{mh}$：Level 2（70%稀疏度）
- 若 $\alpha_{ml} \leq s < \alpha_m$：Level 3（90%稀疏度）
- 若 $s < \alpha_{ml}$：Level 4（100%剪枝）
统计实际分布 $$P_{\text{actual}} = [p_0', p_1', p_2', p_3', p_4']$$ 并计算实际平均稀疏度：
$$s_{\text{actual}} = \sum_{i=0}^{4} p_i' \cdot s_i$$
若 $|s_{\text{actual}} - s_{\text{target}}| > \epsilon$（如 $\epsilon = 2\%$），则调整 $$P_{\text{desired}}$$ 并重新计算分位数。调整策略：
- 若 $s_{\text{actual}} < s_{\text{target}}$：增加高稀疏度等级（Level 3和4）的占比
- 若 $s_{\text{actual}} > s_{\text{target}}$：增加低稀疏度等级（Level 0和1）的占比
期望分布的确定方法
可通过以下两种方法确定 $P_{\text{desired}}$：
方法一：基于约束的优化求解。给定目标平均稀疏度，求解最大熵分布：
$$\min_{\mathbf{p}} \quad -\sum_{i=0}^{4} p_i \log p_i$$
$$\text{s.t.} \quad \sum_{i=0}^{4} p_i \cdot s_i = 0.70, \quad \sum_{i=0}^{4} p_i = 1, \quad p_i \geq 0$$
方法二：基于先验知识直接指定。参考LeanKV的经验，高重要性token（Level 0）通常占10-15%，完全剪枝（Level 4）占15-20%，中间层级平滑分布。推荐的初始配置为 $[0.10, 0.20, 0.30, 0.20, 0.20]$。
粒度选择：Per-layer vs Per-head
该方法可以在不同粒度上应用：
Per-layer方案：对每个layer，聚合该layer所有attention head的scores，计算一组统一的阈值。优势是参数少、实现简单；劣势是忽略了同一layer内不同head的异质性。
Per-head方案：为每个attention head独立计算阈值。优势是能够捕捉head-specific的sparsity pattern（LeanKV Figure 4显示不同head的critical token数量可相差数倍）；劣势是参数量增加（$4 \times \text{num\_layers} \times \text{num\_heads}$）。
推荐策略：初期使用per-layer简化问题，若精度不足再尝试per-head。
Layer-wise的自适应调整
借鉴TEAL的发现，不同layer对稀疏性的敏感度不同。可以为不同layer设置不同的 $P_{\text{desired}}$：
- 早期layer（0-5）：使用更保守的分布，如 $[0.15, 0.25, 0.30, 0.20, 0.10]$（平均稀疏度约62%）
- 中间layer（6-25）：使用更激进的分布，如 $[0.08, 0.18, 0.30, 0.22, 0.22]$（平均稀疏度约72%）
- 后期layer（26-31）：使用标准分布，如 $[0.10, 0.20, 0.30, 0.20, 0.20]$（平均稀疏度约70%）
Key和Value的差异化处理
可选地，借鉴LeanKV对Key和Value差异化对待的思想，使用不同的期望分布：
- Key：更保守的分布（低稀疏度），如 $P_K = [0.15, 0.25, 0.30, 0.20, 0.10]$（平均约62%）
- Value：更激进的分布（高稀疏度），如 $P_V = [0.08, 0.18, 0.30, 0.22, 0.22]$（平均约72%）
- 整体平均：$(62\% + 72\%) / 2 \approx 67\%$
这种处理方式基于Key通过softmax影响所有token的观察，因此需要更高的保留精度。
计算复杂度分析
该方法的计算成本主要包括：
1. 数据收集阶段：在校准数据集上执行一次前向传播，时间复杂度 $O(N \cdot M)$，其中 $N$ 是样本数，$M$ 是单次前向传播时间
2. 分位数计算：排序复杂度 $O(T \log T)$，其中 $T$ 是总token数（通常 $T \approx N \times \text{seq\_len}$）
3. 验证阶段：线性扫描，复杂度 $O(T)$
总体时间成本：对于200-500个样本的校准集，在单个GPU上通常只需几分钟到十几分钟，远低于需要多次完整推理的网格搜索方法（数天）。
Input:
    Calibration dataset (如 MATH 训练集, 100-500 样本)
    Target distribution: P_desired = [p_0, p_1, p_2, p_3, p_4]
                                   = [0.10, 0.20, 0.30, 0.20, 0.20]
    Sparsity levels: S = [0%, 50%, 70%, 90%, 100%]
    
Output:
    Thresholds: (α_h, α_mh, α_m, α_ml) for each layer/head

// ============ Step 1: 收集归一化 importance scores ============
all_scores = {}  // key: (layer_id, head_id), value: list of scores

For each sample in calibration_dataset:
    outputs = model.forward(sample, output_attentions=True)
    
    For layer_id, attention_weights in enumerate(outputs.attentions):
        # attention_weights: [B, H, T_q, T_k]
        N = sequence_length
        
        For head_id in range(num_heads):
            A_h = attention_weights[:, head_id, :, :]  // [B, T_q, T_k]
            
            // 计算每个 key token 的 importance
            importance = A_h.mean(dim=1)  // [B, T_k]: 平均被关注程度
            
            // 归一化
            normalized_scores = importance * N  // [B, T_k]
            
            // 存储
            all_scores[(layer_id, head_id)].append(normalized_scores.flatten())

// 合并每个 layer-head 的所有 scores
For (layer_id, head_id) in all_scores.keys():
    all_scores[(layer_id, head_id)] = Concatenate(
        all_scores[(layer_id, head_id)]
    )


// ============ Step 2: 计算分位数阈值 ============
thresholds = {}

For (layer_id, head_id) in all_scores.keys():
    scores = all_scores[(layer_id, head_id)]
    
    // 从高到低排序
    sorted_scores = Sort(scores, descending=True)
    
    // 计算累积分布
    cumulative = [p_0, p_0+p_1, p_0+p_1+p_2, p_0+p_1+p_2+p_3]
                = [0.10,  0.30,    0.60,       0.80]
    
    // 分位数阈值（从大到小）
    α_h  = Quantile(sorted_scores, q=cumulative[0])  // 前10%的边界
    α_mh = Quantile(sorted_scores, q=cumulative[1])  // 前30%的边界
    α_m  = Quantile(sorted_scores, q=cumulative[2])  // 前60%的边界
    α_ml = Quantile(sorted_scores, q=cumulative[3])  // 前80%的边界
    
    thresholds[(layer_id, head_id)] = (α_h, α_mh, α_m, α_ml)


// ============ Step 3: 验证平均稀疏度 ============
For (layer_id, head_id) in all_scores.keys():
    scores = all_scores[(layer_id, head_id)]
    α_h, α_mh, α_m, α_ml = thresholds[(layer_id, head_id)]
    
    // 统计每个等级的token数量
    n_0 = Count(scores >= α_h)
    n_1 = Count((α_mh <= scores) & (scores < α_h))
    n_2 = Count((α_m <= scores) & (scores < α_mh))
    n_3 = Count((α_ml <= scores) & (scores < α_m))
    n_4 = Count(scores < α_ml)
    
    total = len(scores)
    actual_p = [n_0/total, n_1/total, n_2/total, n_3/total, n_4/total]
    
    actual_sparsity = Sum([actual_p[i] * S[i] for i in range(5)])
    
    Print(f"Layer {layer_id}, Head {head_id}:")
    Print(f"  Distribution: {actual_p}")
    Print(f"  Avg sparsity: {actual_sparsity:.3f}")


// ============ Step 4: (可选) Layer-wise 聚合 ============
// 如果不想 per-head，可以聚合为 per-layer
layer_thresholds = {}

For layer_id in range(num_layers):
    // 收集该 layer 所有 head 的 scores
    layer_scores = Concatenate([
        all_scores[(layer_id, head_id)] 
        for head_id in range(num_heads)
    ])
    
    // 重新计算分位数
    sorted_scores = Sort(layer_scores, descending=True)
    α_h  = Quantile(sorted_scores, q=0.10)
    α_mh = Quantile(sorted_scores, q=0.30)
    α_m  = Quantile(sorted_scores, q=0.60)
    α_ml = Quantile(sorted_scores, q=0.80)
    
    layer_thresholds[layer_id] = (α_h, α_mh, α_m, α_ml)

Return thresholds (per-head) or layer_thresholds (per-layer)
 1. 快速验证
# 伪代码框架
def calibrate_thresholds_teal_style():
    # 使用 MATH 训练集的 200 个样本
    calibration_data = load_math_train_subset(n=200)
    
    # 收集所有 layer 的 attention scores
    all_scores = {}
    for sample in calibration_data:
        outputs = model(sample, output_attentions=True)
        for layer_id, attn in enumerate(outputs.attentions):
            importance = compute_importance(attn)  # [B, H, T]
            normalized = importance * sequence_length
            all_scores[layer_id].append(normalized)
    
    # 合并每个 layer 的 scores（跨所有 heads）
    for layer_id in all_scores.keys():
        all_scores[layer_id] = np.concatenate(all_scores[layer_id])
    
    # 计算分位数阈值
    P_desired = [0.10, 0.20, 0.30, 0.20, 0.20]
    cumulative = [0.10, 0.30, 0.60, 0.80]
    
    thresholds = {}
    for layer_id, scores in all_scores.items():
        sorted_scores = np.sort(scores)[::-1]  # 降序
        α_h  = np.quantile(sorted_scores, cumulative[0])
        α_mh = np.quantile(sorted_scores, cumulative[1])
        α_m  = np.quantile(sorted_scores, cumulative[2])
        α_ml = np.quantile(sorted_scores, cumulative[3])
        thresholds[layer_id] = (α_h, α_mh, α_m, α_ml)
    
    return thresholds

2. 验证稀疏度分布
def verify_sparsity_distribution(thresholds, calibration_data):
    for layer_id, (α_h, α_mh, α_m, α_ml) in thresholds.items():
        # 统计实际分布
        n_0 = (scores >= α_h).sum()
        n_1 = ((scores >= α_mh) & (scores < α_h)).sum()
        n_2 = ((scores >= α_m) & (scores < α_mh)).sum()
        n_3 = ((scores >= α_ml) & (scores < α_m)).sum()
        n_4 = (scores < α_ml).sum()
        
        total = len(scores)
        actual_p = [n_0/total, n_1/total, n_2/total, n_3/total, n_4/total]
        actual_sparsity = sum([actual_p[i] * S[i] for i in range(5)])
        
        print(f"Layer {layer_id}:")
        print(f"  Target distribution: {P_desired}")
        print(f"  Actual distribution: {actual_p}")
        print(f"  Target sparsity: 70%")
        print(f"  Actual sparsity: {actual_sparsity*100:.1f}%")





4.2 优化求解


5. 确定稀疏度分布 $$P_{\text{desired}}$$ 的方法
$$P_{\text{desired}} = [p_0, p_1, p_2, p_3, p_4]$$ 的选择直接决定了最终的压缩效果和模型精度。我总结了以下几种方法：
 0x3 确定稀疏度分布 $$P_{\text{desired}}$$ 的方法

这是一个关键问题！$P_{\text{desired}} = [p_0, p_1, p_2, p_3, p_4]$ 的选择直接决定了最终的压缩效果和模型精度。我总结了以下几种方法：

---
方法一：基于约束的数学优化

核心思想
给定目标平均稀疏度70%，求解一个"合理"的分布，使其满足约束条件。

优化目标的选择

选项A：最大熵分布（Maximum Entropy）

动机：在满足约束的前提下，选择最"均匀"的分布，避免过度偏向某个等级。

数学形式：
$$\begin{align}
\max_{\mathbf{p}} \quad & H(\mathbf{p}) = -\sum_{i=0}^{4} p_i \log p_i \\
\text{s.t.} \quad & \sum_{i=0}^{4} p_i \cdot s_i = 0.70 \\
& \sum_{i=0}^{4} p_i = 1 \\
& p_i \geq 0, \quad \forall i
\end{align}$$

求解代码示例：
from scipy.optimize import minimize

def objective(p):
    # 最大熵 = 最小化 -entropy
    return -np.sum(p * np.log(p + 1e-10))

def constraint_avg_sparsity(p):
    s = np.array([0.0, 0.5, 0.7, 0.9, 1.0])
    return np.dot(p, s) - 0.70

def constraint_sum(p):
    return np.sum(p) - 1.0

constraints = [
    {'type': 'eq', 'fun': constraint_avg_sparsity},
    {'type': 'eq', 'fun': constraint_sum}
]

bounds = [(0, 1) for _ in range(5)]
p_init = [0.2, 0.2, 0.2, 0.2, 0.2]  # 均匀初始化

result = minimize(objective, p_init, bounds=bounds, constraints=constraints)
P_desired = result.x

print(f"最大熵分布: {P_desired}")
print(f"验证平均稀疏度: {np.dot(P_desired, [0.0, 0.5, 0.7, 0.9, 1.0])}")

预期结果（手动推导）：
由于70%恰好是五个稀疏度等级 $$\{0\%, 50\%, 70\%, 90\%, 100\%\}$$ 的中位数，最大熵解会倾向于集中在70%附近：
$$P_{\text{maxent}} \approx [0.12, 0.18, 0.40, 0.18, 0.12]$$
（实际数值需要求解器计算）

优势：
- ✅ 理论优雅，无需人工假设
- ✅ 避免过度集中在某个等级
  
劣势：
- ⚠️ 可能产生不符合实际的分布（如过度集中在70%）
- ⚠️ 忽略了不同稀疏度等级的"成本"差异
  

---

选项B：最小化与均匀分布的距离

动机：在满足稀疏度约束的前提下，尽量接近均匀分布 $\mathbf{p}_{\text{uniform}} = [0.2, 0.2, 0.2, 0.2, 0.2]$。

数学形式：
$$\begin{align}
\min_{\mathbf{p}} \quad & \|\mathbf{p} - \mathbf{p}_{\text{uniform}}\|^2 = \sum_{i=0}^{4} (p_i - 0.2)^2 \\
\text{s.t.} \quad & \sum_{i=0}^{4} p_i \cdot s_i = 0.70 \\
& \sum_{i=0}^{4} p_i = 1 \\
& p_i \geq 0
\end{align}$$

求解：
def objective_uniform(p):
    p_uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    return np.sum((p - p_uniform)**2)

result = minimize(objective_uniform, p_init, bounds=bounds, constraints=constraints)
P_desired = result.x

预期结果：
$$P_{\text{uniform-like}} \approx [0.15, 0.20, 0.30, 0.20, 0.15]$$


---

选项C：加权目标（考虑等级重要性）

动机：不同稀疏度等级对模型精度的影响不同，应该给重要等级（如0%和100%）更多"预算"。

数学形式：
$$\begin{align}
\min_{\mathbf{p}} \quad & \sum_{i=0}^{4} w_i \cdot (p_i - p_i^{\text{ref}})^2 \\
\text{s.t.} \quad & \text{same constraints}
\end{align}$$

其中 $$\mathbf{w}$$ 是权重，$\mathbf{p}^{\text{ref}}$ 是参考分布。

示例设置：
# 给极端等级（0%和100%）更高权重
w = np.array([2.0, 1.0, 1.0, 1.0, 2.0])
p_ref = np.array([0.15, 0.20, 0.30, 0.20, 0.15])


---

方法二：基于先验知识的直接设定

核心思想
根据已有研究（LeanKV、TEAL）和任务特性，直接指定一个合理的分布。

推荐的初始分布

配置A：标准分布（推荐用于初期验证）
$$P_{\text{standard}} = [0.10, 0.20, 0.30, 0.20, 0.20]$$

验证平均稀疏度：
$$0.10 \times 0\% + 0.20 \times 50\% + 0.30 \times 70\% + 0.20 \times 90\% + 0.20 \times 100\% = 69\%$$

设计理由：
- 10%的token完全保留（0%）：对应最关键的token（如recent window、高频实体）
- 20%的token低稀疏度（50%）：重要但非关键的token
- 30%的token中度稀疏（70%）：对齐baseline，占比最大
- 20%的token高稀疏度（90%）：不重要但保留少量信息
- 20%的token完全剪枝（100%）：最不重要的token
  
参考依据：
- LeanKV的经验：高重要性token占10-15%
- TEAL的发现：40-50%稀疏度下精度几乎无损 → 保留50-60%的token足够
  

---

配置B：保守分布（用于高精度要求场景）
$$P_{\text{conservative}} = [0.15, 0.25, 0.30, 0.20, 0.10]$$

平均稀疏度：
$$0.15 \times 0\% + 0.25 \times 50\% + 0.30 \times 70\% + 0.20 \times 90\% + 0.10 \times 100\% = 61.5\%$$

适用场景：
- 数学推理任务（MATH、GSM8K）
- 代码生成（HumanEval+）
- Thinking model（QwQ-32B）
  
设计理由：
- 增加Level 0和Level 1的占比（从30%提升到40%）
- 减少完全剪枝的占比（从20%降到10%）
- 更保守的策略，适合信息密度高的任务
  

---

配置C：激进分布（用于长文本场景）
$$P_{\text{aggressive}} = [0.08, 0.17, 0.30, 0.22, 0.23]$$

平均稀疏度：
$$0.08 \times 0\% + 0.17 \times 50\% + 0.30 \times 70\% + 0.22 \times 90\% + 0.23 \times 100\% = 75\%$$

适用场景：
- 长文本摘要（GovReport）
- 多文档问答（HotpotQA）
- 长上下文任务（LongBench）
  
设计理由：
- 长文本中存在大量冗余信息
- 可以更激进地剪枝不重要token
- 节省更多内存以支持更长的序列
  

---

先验知识来源

来源1：LeanKV的token重要性分析（Figure 3-4）

LeanKV发现：
- 不同layer的critical token数量差异巨大（200~1400）
- 平均保留约50-60%的token可以维持95%的attention score
- 推论：可以设置 $p_0 + p_1 \approx 30-40\%$（完全保留+低稀疏度）
  

---

来源2：TEAL的稀疏度实验（Table 1-2）

TEAL在40-50%稀疏度下：
- Llama-3-8B：MMLU准确率从66.5%降到65.8%（-0.7%）
- Llama-3-70B：MMLU准确率从81.0%降到80.5%（-0.5%）
  
推论：
- 50%左右的稀疏度是一个"安全区"，精度损失很小
- 可以大胆设置 $p_3 + p_4 \approx 40\%$（高稀疏度+剪枝）
  

---

来源3：Attention score的经验分布

在实际模型中，attention score通常呈现长尾分布：
- 少数token获得大部分attention（头部）
- 大多数token的attention score很低（尾部）
  
幂律假设：假设attention score服从幂律分布 $P(s) \propto s^{-\alpha}$，可以推导出：
- 前10%的token贡献约40-50%的总attention
- 后20%的token贡献不到5%的总attention
  
推论：
- $$p_0 = 10\%$$（捕捉头部）
- $$p_4 = 20\%$$（剪枝尾部）
  

---

方法三：数据驱动的自适应确定

核心思想
先在校准数据集上收集attention score的经验分布，根据实际分布特征确定 $P_{\text{desired}}$。

算法流程

Algorithm: Data-Driven Distribution Initialization

// Step 1: 收集 attention scores
all_scores = CollectScoresOnCalibrationSet()

// Step 2: 分析经验分布
sorted_scores = Sort(all_scores, descending=True)
cumulative_attention = ComputeCumulativeSum(sorted_scores)

// Step 3: 找到"拐点"
// 找到前k%的token贡献了90%的总attention
k_90 = FindPercentile(cumulative_attention, threshold=0.90)
// 找到前k%的token贡献了50%的总attention
k_50 = FindPercentile(cumulative_attention, threshold=0.50)

Print(f"前 {k_50}% 的token贡献了 50% 的attention")
Print(f"前 {k_90}% 的token贡献了 90% 的attention")

// Step 4: 基于拐点设计分布
// 保守策略：前k_50的token用低稀疏度，后(1-k_90)的token剪枝
p_0 = k_50 / 2           // 前一半用0%稀疏度
p_1 = k_50 / 2           // 后一半用50%稀疏度
p_4 = 1 - k_90           // 尾部剪枝
p_2 + p_3 = k_90 - k_50  // 中间部分分配到70%和90%

// Step 5: 调整使平均稀疏度=70%
AdjustToTargetSparsity(p, target=0.70)

Return P_desired


---

示例计算

假设在MATH训练集上分析后发现：
- 前15%的token贡献了50%的attention → $$k_{50} = 15\%$$
- 前65%的token贡献了90%的attention → $$k_{90} = 65\%$$
  
初始分配：
- $$p_0 = 15\% / 2 = 7.5\%$$（最重要的前7.5%）
- $$p_1 = 15\% / 2 = 7.5\%$$（重要的后7.5%）
- $$p_4 = 1 - 65\% = 35\%$$（尾部35%剪枝）
- $$p_2 + p_3 = 65\% - 15\% = 50\%$$（中间50%）
  
调整到70%稀疏度：
假设 $p_2 : p_3 = 3 : 2$（偏向70%等级），则：
- $$p_2 = 30\%$$
- $$p_3 = 20\%$$
  
验证：
$$0.075 \times 0 + 0.075 \times 50 + 0.30 \times 70 + 0.20 \times 90 + 0.35 \times 100 = 75.75\%$$

需要进一步调整：平均稀疏度偏高（75.75% vs 70%），增加低稀疏度等级的占比：

最终分布：
$$P_{\text{data-driven}} = [0.10, 0.15, 0.35, 0.20, 0.20]$$

验证：
$$0.10 \times 0 + 0.15 \times 50 + 0.35 \times 70 + 0.20 \times 90 + 0.20 \times 100 = 70\%$$ ✓


---

方法四：多候选分布的实验验证

核心思想
预先设计3-5组候选分布，在校准数据集的子集上快速评估，选择最优的一组。

算法流程

Algorithm: Multi-Candidate Selection

// 预定义候选分布
candidates = [
    [0.10, 0.20, 0.30, 0.20, 0.20],  # 标准
    [0.15, 0.25, 0.30, 0.20, 0.10],  # 保守
    [0.08, 0.17, 0.30, 0.22, 0.23],  # 激进
    [0.12, 0.22, 0.32, 0.20, 0.14],  # 平衡
    [0.08, 0.15, 0.40, 0.22, 0.15],  # 集中在70%
]

// 使用校准集的小子集（如50-100样本）快速评估
calibration_subset = LoadSubset(n=100)

best_accuracy = 0
best_distribution = None

For P_candidate in candidates:
    // 计算对应的阈值
    thresholds = ComputeThresholdsFromQuantiles(P_candidate)
    
    // 在子集上评估
    accuracy = EvaluateWithThresholds(calibration_subset, thresholds)
    avg_sparsity = ComputeAverageSparsity(thresholds)
    
    Print(f"Distribution: {P_candidate}")
    Print(f"  Accuracy: {accuracy:.2f}%")
    Print(f"  Avg Sparsity: {avg_sparsity:.1f}%")
    
    // 检查稀疏度约束
    If |avg_sparsity - 70%| <= 2%:
        If accuracy > best_accuracy:
            best_accuracy = accuracy
            best_distribution = P_candidate

Return best_distribution


---

时间成本估算

假设：
- 5组候选分布
- 每组在100个样本上评估
- 每个样本推理时间：0.5秒
  
总时间：$5 \times 100 \times 0.5 = 250$ 秒 ≈ 4分钟

相比完整的网格搜索（数天），这个成本完全可以接受。


---

推荐的实施策略

阶段1：快速探索（1-2小时）

步骤：
1. 使用方法二（先验知识），选择3个初始分布：
  - 标准：$$[0.10, 0.20, 0.30, 0.20, 0.20]$$
  - 保守：$$[0.15, 0.25, 0.30, 0.20, 0.10]$$
  - 激进：$$[0.08, 0.17, 0.30, 0.22, 0.23]$$
    
2. 在MATH训练集的200个样本上计算分位数阈值
  
3. 在MATH测试集的子集（100个样本）上快速评估
  
4. 选择精度最高且稀疏度接近70%的分布
  

---

阶段2：精细调整（半天）

步骤：
1. 使用阶段1选出的最优分布作为基础
  
2. 使用**方法三（数据驱动）**分析attention score的经验分布，检查是否有明显的"拐点"
  
3. 根据拐点微调分布：
  - 如果头部集中度高（前10%贡献>60% attention）→ 增加 $$p_0$$
  - 如果尾部很长（后30%贡献<5% attention）→ 增加 $$p_4$$
    
4. 在完整MATH测试集上验证
  

---

阶段3：任务特化（1天，可选）
步骤：
1. 在不同任务（GSM8K、HumanEval+、MMLU）上测试阶段2的分布
  
2. 如果某个任务精度明显下降，为该任务单独设计分布：
  - 代码生成：可能需要更保守（代码结构敏感）
  - 数学推理：当前分布应该已经适配
  - 常识问答：可能可以更激进（冗余度高）
3. 考虑是否需要任务条件的分布选择机制

---
关键参数的敏感性分析
$$p_0$$（0%稀疏度）的影响
$$p_0$$
预期效果
适用场景
5%
激进，可能损失精度
长文本、低精度要求
10%
标准配置
通用
15%
保守，精度更高
高精度要求、数学推理

---
$$p_4$$（100%剪枝）的影响
$$p_4$$
预期效果
适用场景
10%
保守，内存节省少
高精度要求
20%
标准配置
通用
30%+
激进，可能损失精度
长文本、冗余度高

---
$$p_2$$（70%稀疏度）的影响
由于70%对齐baseline，这个等级的占比反映了"保持原有压缩策略"的程度：
- $$p_2 = 40\%$$：大部分token使用baseline策略（保守）
- $$p_2 = 30\%$$：标准配置（平衡）
- $$p_2 = 20\%$$：更多token走极端策略（激进）

---
总结与建议
推荐方案
对于您的场景（KV cache压缩 + 数学推理任务），我建议：

1. 第一选择：先验知识 + 多候选验证（方法二 + 方法四）
  - 时间成本：1-2小时
  - 使用标准分布 $$[0.10, 0.20, 0.30, 0.20, 0.20]$$ 作为起点
  - 在3-5组候选中快速选择最优
    
2. 第二步（可选）：数据驱动微调（方法三）
  - 如果阶段1的结果不理想（精度下降>2%）
  - 分析attention score分布特征
  - 根据拐点调整分布
    
3. 避免过度优化：
  - 不建议使用数学优化（方法一），除非有特殊需求
  - 原因：最大熵等目标不一定符合实际的attention分布
    
实施检查清单

[] 确定目标平均稀疏度（70%）
[] 选择3-5组候选分布
[] 在MATH训练集（200样本）上计算分位数阈值
[] 在MATH测试集（100样本）上快速评估
[] 验证实际平均稀疏度在 68%-72% 之间
[] 选择精度最高的分布
[] 在完整测试集上验证
[] 跨任务泛化测试（GSM8K、HumanEval+）

0x4 通过分位数方法得到的阈值是否具有代表性 和 泛化能力
假设1：校准集的代表性
在校准数据集上的计算的阈值能够泛化到测试集合其他任务上；
需要证明：
$$\alpha_{\text{calibration}} \approx \alpha_{\text{test}}$$
或者更宽松地：
$$|\text{Sparsity}_{\text{test}}(\alpha_{\text{calibration}}) - \text{Sparsity}_{\text{target}}| < \epsilon$$

假设2：分位数的稳定性
陈述：在不同样本、不同任务上，attention score的分位数分布是稳定的。
需要证明：
$$\text{Var}[\alpha_h] \text{ is small across different subsets}$$
假设3：Per-layer vs Global 阈值
陈述：如果使用per-layer的阈值，每个layer的阈值应该有显著差异；如果差异不大，说明global阈值就足够。
需要证明：
$$\text{Coefficient of Variation} = \frac{\sigma(\{\alpha_l\})}{\mu(\{\alpha_l\})} > \text{threshold}$$
其中 $$\{\alpha_l\}$$ 是所有layer的阈值集合。
为什么分位数方法可能有效？
Attention score 的分布特性
研究表明（如LeanKV H2O）,attention score 在不同的任务中都呈现类似的长尾分布：
- 少数 token 获得大部分的 attention
- 大多数 token 的 attention 都很低
推论：
- 如果 分布的 “形状”相似（如都是 幂律分布）
- 那么同一个分位数（如前 10% ）对应的阈值也会相似

Central Limit Theorem 的启示
当校准样本数足够大时，样本分位数会收敛到总体分位数：
$$\hat{Q}_p \xrightarrow{n \to \infty} Q_p$$
推论：
- 只要校准集足够大（如200+样本）且具有代表性
- 得到的阈值就会接近"真实"阈值

Robustness of Quantile-based Methods
相比均值，分位数对 异常值不敏感：
- 如果某个样本的 attention score 异常高 或 低
- 不会显著影响分位数的计算
推论：
- 分位数方法比"基于均值的方法"更鲁棒
- 更适合处理长尾分布的attention score


其他
关于 一个 layer 共用一套相关参数：
Model Structure:
    ├── Layer 0
    │   ├── Head 0  ─┐
    │   ├── Head 1   ├─→ 共享 (α_h, α_m, α_l)
    │   ├── ...      │
    │   └── Head H  ─┘
    ├── Layer 1
    │   ├── Head 0  ─┐
    │   ├── ...      ├─→ 共享同一套 (α_h, α_m, α_l)
    │   └── Head H  ─┘
    └── ...
    

