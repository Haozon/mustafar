#!/usr/bin/env python3
"""
对比预测结果，找出差异原因
"""

import json
import sys

# 读取原始结果
original_file = '/home/zh/mustafar/pred/Meta-Llama-3-8B-Instruct_8192_K_0.0_V_0.0/narrativeqa.jsonl'
yours_file = '/home/zh/mustafar/DiffSparseKV/pred/pred_diff_sparse_kv/Meta-Llama-3-8B-Instruct_8192_baseline/narrativeqa.jsonl'

with open(original_file) as f:
    original = [json.loads(line) for line in f]

with open(yours_file) as f:
    yours = [json.loads(line) for line in f]

print("=" * 80)
print("预测结果对比分析")
print("=" * 80)
print()

# 统计差异
total = len(original)
different = 0
original_longer = 0
yours_longer = 0

for i in range(total):
    orig_pred = original[i]['pred']
    your_pred = yours[i]['pred']
    
    if orig_pred != your_pred:
        different += 1
        if len(your_pred) > len(orig_pred):
            yours_longer += 1
        else:
            original_longer += 1

print(f"总样本数: {total}")
print(f"不同的预测: {different} ({different/total*100:.1f}%)")
print(f"你的更长: {yours_longer} ({yours_longer/total*100:.1f}%)")
print(f"原始更长: {original_longer} ({original_longer/total*100:.1f}%)")
print()

# 分析长度差异
orig_lengths = [len(original[i]['pred']) for i in range(total)]
your_lengths = [len(yours[i]['pred']) for i in range(total)]

print(f"原始平均长度: {sum(orig_lengths)/total:.1f} 字符")
print(f"你的平均长度: {sum(your_lengths)/total:.1f} 字符")
print(f"长度差异: {(sum(your_lengths)-sum(orig_lengths))/total:.1f} 字符")
print()

# 检查常见前缀
common_prefixes = [
    "According to the text,",
    "According to the story,",
    "According to the poem,",
    "According to the passage,",
    "Based on the text,",
    "In the text,",
]

prefix_counts = {prefix: 0 for prefix in common_prefixes}

for i in range(total):
    your_pred = yours[i]['pred']
    for prefix in common_prefixes:
        if your_pred.startswith(prefix):
            prefix_counts[prefix] += 1

print("常见前缀统计（你的预测）:")
for prefix, count in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  '{prefix}': {count} ({count/total*100:.1f}%)")
print()

# 显示几个典型例子
print("=" * 80)
print("典型差异示例")
print("=" * 80)
print()

for i in range(min(10, total)):
    orig_pred = original[i]['pred']
    your_pred = yours[i]['pred']
    answer = original[i]['answers'][0]
    
    if orig_pred != your_pred:
        print(f"样本 {i+1}:")
        print(f"  答案: {answer}")
        print(f"  原始: {orig_pred}")
        print(f"  你的: {your_pred}")
        print()
