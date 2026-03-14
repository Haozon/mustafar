#!/usr/bin/env python3
"""
DiffSparseKV 结果评估脚本
基于 eval_long_bench.py 修改，支持自定义路径
"""

import os
import sys
import json
import argparse
import numpy as np

# 添加父目录到路径以导入 metrics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='结果目录路径，例如: ./pred/pred_diff_sparse_kv/Meta-Llama-3-8B-Instruct_8192_baseline')
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    
    path = args.result_dir
    if not path.endswith('/'):
        path += '/'
    
    if not os.path.exists(path):
        print(f"[ERROR] 目录不存在: {path}")
        sys.exit(1)
    
    all_files = os.listdir(path)
    jsonl_files = [f for f in all_files if f.endswith("jsonl")]
    
    if not jsonl_files:
        print(f"[ERROR] 未找到任何 .jsonl 文件在: {path}")
        sys.exit(1)
    
    print(f"评估目录: {path}")
    print(f"找到 {len(jsonl_files)} 个数据集文件")
    print()
    
    for filename in jsonl_files:
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        
        print(f"  处理: {dataset:25s}", end=" ... ")
        
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        
        scores[dataset] = score
        print(f"{score:6.2f}")
    
    # 计算平均分
    avg_score = np.mean(list(scores.values()))
    scores['average'] = round(avg_score, 2)
    
    # 保存结果
    out_path = f"{path}result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    
    print()
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    for dataset, score in sorted(scores.items()):
        if dataset != 'average':
            print(f"{dataset:25s}: {score:6.2f}")
    print("-" * 60)
    print(f"{'平均分数':25s}: {scores['average']:6.2f}")
    print("=" * 60)
    print()
    print(f"结果已保存到: {out_path}")
