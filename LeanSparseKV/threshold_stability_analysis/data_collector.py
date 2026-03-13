#!/usr/bin/env python3
"""
阈值数据收集器 (ThresholdCollector)

负责从多个数据集收集阈值数据，支持Bootstrap采样，并集成现有的LeanSparseKV校准系统。
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# 添加LeanSparseKV路径以便导入
sys.path.append(str(Path(__file__).parent.parent / "LeanSparseKV"))

try:
    from calibrate_sparsity_thresholds import DiffKVThresholdCalibrator, load_calibration_data
    LEANSPARSE_AVAILABLE = True
except ImportError:
    LEANSPARSE_AVAILABLE = False
    logging.warning("LeanSparseKV calibration system not available")

# 数据集加载
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("datasets library not available")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ThresholdRecord:
    """阈值记录数据结构"""
    dataset_name: str
    sample_size: int
    layer_id: int
    head_id: Optional[int]
    quantile_name: str  # 'alpha_h', 'alpha_mh', 'alpha_m', 'alpha_ml'
    threshold_value: float
    collection_timestamp: datetime
    model_config: Dict[str, Any]
    bootstrap_iteration: Optional[int] = None

@dataclass
class CollectionConfig:
    """数据收集配置"""
    model_path: str
    datasets: List[str]
    sample_sizes: List[int]
    target_sparsity: float = 0.70
    granularity: str = "per_layer"  # "per_layer" or "per_head"
    max_length: int = 2048
    batch_size: int = 4
    device: str = "auto"
    bootstrap_samples: int = 100
    cross_validation_folds: int = 5

class ThresholdCollector:
    """
    阈值数据收集器
    
    支持多数据集阈值收集、Bootstrap采样和与LeanSparseKV系统的集成。
    """
    
    def __init__(self, config: CollectionConfig):
        """
        初始化收集器
        
        Args:
            config: 收集配置
        """
        self.config = config
        
        # 设备配置
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        logger.info(f"从 {config.model_path} 加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # 获取模型信息
        self._extract_model_info()
        
        # 存储收集的数据
        self.collected_records: List[ThresholdRecord] = []
        
        logger.info(f"模型加载完成: {self.num_layers} 层, {self.num_heads} 头")
    
    def _extract_model_info(self):
        """提取模型架构信息"""
        # 处理不同的模型架构
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA, Mistral等
            self.num_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2, GPT-Neo等
            self.num_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'bert') and hasattr(self.model.bert.encoder, 'layer'):
            # BERT类模型
            self.num_layers = len(self.model.bert.encoder.layer)
        else:
            # 从配置推断
            if hasattr(self.model.config, 'num_hidden_layers'):
                self.num_layers = self.model.config.num_hidden_layers
            elif hasattr(self.model.config, 'n_layer'):
                self.num_layers = self.model.config.n_layer
            else:
                raise ValueError(f"无法确定模型层数: {type(self.model)}")
        
        self.num_heads = self.model.config.num_attention_heads
        
        # 模型配置信息
        self.model_config = {
            'model_path': self.config.model_path,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'hidden_size': getattr(self.model.config, 'hidden_size', None),
            'vocab_size': getattr(self.model.config, 'vocab_size', None),
            'model_type': getattr(self.model.config, 'model_type', 'unknown')
        }
    
    def collect_thresholds_multi_dataset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从多个数据集收集阈值数据
        
        Returns:
            threshold_data: 嵌套字典格式的阈值数据
        """
        logger.info("开始多数据集阈值收集...")
        
        threshold_data = {}
        
        for dataset_name in self.config.datasets:
            logger.info(f"处理数据集: {dataset_name}")
            threshold_data[dataset_name] = {}
            
            for sample_size in self.config.sample_sizes:
                logger.info(f"  样本大小: {sample_size}")
                
                # 加载校准数据
                calibration_texts = self._load_dataset(dataset_name, sample_size)
                
                # 收集该配置下的阈值
                dataset_thresholds = self._collect_single_configuration(
                    dataset_name, sample_size, calibration_texts
                )
                
                threshold_data[dataset_name][f'sample_size_{sample_size}'] = dataset_thresholds
        
        logger.info("多数据集阈值收集完成")
        return threshold_data
    
    def collect_bootstrap_samples(self, 
                                dataset: str, 
                                n_bootstrap: int = None) -> List[Dict[str, float]]:
        """
        使用Bootstrap采样收集阈值数据
        
        Args:
            dataset: 数据集名称
            n_bootstrap: Bootstrap样本数量
            
        Returns:
            bootstrap_results: Bootstrap采样结果列表
        """
        if n_bootstrap is None:
            n_bootstrap = self.config.bootstrap_samples
            
        logger.info(f"开始Bootstrap采样: {dataset}, {n_bootstrap}次采样")
        
        # 加载完整数据集
        full_dataset = self._load_dataset(dataset, num_samples=1000)  # 使用较大的基础数据集
        
        bootstrap_results = []
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap采样"):
            # 随机采样
            sample_indices = np.random.choice(
                len(full_dataset), 
                size=min(200, len(full_dataset)), 
                replace=True
            )
            bootstrap_sample = [full_dataset[idx] for idx in sample_indices]
            
            # 收集该Bootstrap样本的阈值
            bootstrap_thresholds = self._collect_single_configuration(
                dataset, len(bootstrap_sample), bootstrap_sample, bootstrap_iteration=i
            )
            
            # 转换为扁平字典格式
            flat_thresholds = self._flatten_thresholds(bootstrap_thresholds)
            bootstrap_results.append(flat_thresholds)
        
        logger.info(f"Bootstrap采样完成: {len(bootstrap_results)}个样本")
        return bootstrap_results
    
    def collect_cross_validation_thresholds(self, 
                                          dataset: str, 
                                          k_folds: int = None) -> Dict[str, List[float]]:
        """
        使用交叉验证收集阈值数据
        
        Args:
            dataset: 数据集名称
            k_folds: 交叉验证折数
            
        Returns:
            cv_results: 交叉验证结果
        """
        if k_folds is None:
            k_folds = self.config.cross_validation_folds
            
        logger.info(f"开始交叉验证: {dataset}, {k_folds}折")
        
        # 加载完整数据集
        full_dataset = self._load_dataset(dataset, num_samples=500)
        
        # 分割数据集
        fold_size = len(full_dataset) // k_folds
        cv_results = {}
        
        for fold in range(k_folds):
            logger.info(f"处理第 {fold + 1}/{k_folds} 折")
            
            # 创建训练集（排除当前折）
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size
            
            train_data = full_dataset[:start_idx] + full_dataset[end_idx:]
            
            # 收集该折的阈值
            fold_thresholds = self._collect_single_configuration(
                dataset, len(train_data), train_data
            )
            
            # 存储结果
            flat_thresholds = self._flatten_thresholds(fold_thresholds)
            
            for key, value in flat_thresholds.items():
                if key not in cv_results:
                    cv_results[key] = []
                cv_results[key].append(value)
        
        logger.info(f"交叉验证完成: {k_folds}折")
        return cv_results
    
    def _collect_single_configuration(self, 
                                    dataset_name: str, 
                                    sample_size: int, 
                                    calibration_texts: List[str],
                                    bootstrap_iteration: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        收集单个配置下的阈值数据
        
        Args:
            dataset_name: 数据集名称
            sample_size: 样本大小
            calibration_texts: 校准文本
            bootstrap_iteration: Bootstrap迭代次数（可选）
            
        Returns:
            thresholds: 阈值数据
        """
        if LEANSPARSE_AVAILABLE:
            # 使用LeanSparseKV系统
            calibrator = DiffKVThresholdCalibrator(
                model_path=self.config.model_path,
                target_sparsity=self.config.target_sparsity,
                granularity=self.config.granularity,
                device=self.device
            )
            
            # 收集分数
            calibrator.collect_calibration_scores(
                calibration_texts=calibration_texts,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size
            )
            
            # 计算阈值
            calibrator.compute_quantile_thresholds()
            
            # 转换格式并记录
            thresholds = self._convert_leansparse_thresholds(
                calibrator.thresholds, dataset_name, sample_size, bootstrap_iteration
            )
            
        else:
            # 使用内置实现
            thresholds = self._collect_thresholds_builtin(
                calibration_texts, dataset_name, sample_size, bootstrap_iteration
            )
        
        return thresholds
    
    def _convert_leansparse_thresholds(self, 
                                     leansparse_thresholds: Dict, 
                                     dataset_name: str, 
                                     sample_size: int,
                                     bootstrap_iteration: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        转换LeanSparseKV阈值格式并记录
        
        Args:
            leansparse_thresholds: LeanSparseKV格式的阈值
            dataset_name: 数据集名称
            sample_size: 样本大小
            bootstrap_iteration: Bootstrap迭代次数
            
        Returns:
            converted_thresholds: 转换后的阈值格式
        """
        converted_thresholds = {}
        timestamp = datetime.now()
        
        for key, (α_h, α_mh, α_m, α_ml) in leansparse_thresholds.items():
            if self.config.granularity == "per_layer":
                layer_id = key[0]
                head_id = None
                threshold_key = f'layer_{layer_id}'
            else:  # per_head
                layer_id, head_id = key
                threshold_key = f'layer_{layer_id}_head_{head_id}'
            
            # 存储阈值
            converted_thresholds[threshold_key] = {
                'alpha_h': α_h,
                'alpha_mh': α_mh,
                'alpha_m': α_m,
                'alpha_ml': α_ml
            }
            
            # 记录详细数据
            for quantile_name, threshold_value in [
                ('alpha_h', α_h), ('alpha_mh', α_mh), 
                ('alpha_m', α_m), ('alpha_ml', α_ml)
            ]:
                record = ThresholdRecord(
                    dataset_name=dataset_name,
                    sample_size=sample_size,
                    layer_id=layer_id,
                    head_id=head_id,
                    quantile_name=quantile_name,
                    threshold_value=threshold_value,
                    collection_timestamp=timestamp,
                    model_config=self.model_config,
                    bootstrap_iteration=bootstrap_iteration
                )
                self.collected_records.append(record)
        
        return converted_thresholds
    
    def _collect_thresholds_builtin(self, 
                                  calibration_texts: List[str], 
                                  dataset_name: str, 
                                  sample_size: int,
                                  bootstrap_iteration: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        使用内置实现收集阈值（当LeanSparseKV不可用时）
        
        Args:
            calibration_texts: 校准文本
            dataset_name: 数据集名称
            sample_size: 样本大小
            bootstrap_iteration: Bootstrap迭代次数
            
        Returns:
            thresholds: 阈值数据
        """
        logger.info("使用内置阈值收集实现")
        
        # 目标分布
        target_distribution = np.array([0.05, 0.15, 0.30, 0.30, 0.20])
        cumulative = np.cumsum(target_distribution[:-1])
        
        # 初始化存储
        all_scores = {}
        if self.config.granularity == "per_layer":
            for layer_id in range(self.num_layers):
                all_scores[(layer_id, -1)] = []
        else:  # per_head
            for layer_id in range(self.num_layers):
                for head_id in range(self.num_heads):
                    all_scores[(layer_id, head_id)] = []
        
        # 处理文本批次
        for i in tqdm(range(0, len(calibration_texts), self.config.batch_size), 
                     desc="收集注意力分数"):
            batch_texts = calibration_texts[i:i+self.config.batch_size]
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            with torch.no_grad():
                # 前向传播
                outputs = self.model(**inputs, output_attentions=True)
                attentions = outputs.attentions
                
                # 处理每一层
                for layer_id, attention_weights in enumerate(attentions):
                    # 计算重要性分数
                    importance_scores = self._compute_diffkv_importance(attention_weights)
                    
                    if self.config.granularity == "per_layer":
                        # 按层平均
                        layer_scores = importance_scores.mean(dim=1)  # [B, T]
                        flat_scores = layer_scores.flatten().cpu().numpy()
                        all_scores[(layer_id, -1)].extend(flat_scores)
                    else:  # per_head
                        # 按头存储
                        for head_id in range(self.num_heads):
                            head_scores = importance_scores[:, head_id, :]  # [B, T]
                            flat_scores = head_scores.flatten().cpu().numpy()
                            all_scores[(layer_id, head_id)].extend(flat_scores)
        
        # 计算阈值
        thresholds = {}
        timestamp = datetime.now()
        
        for key, scores in all_scores.items():
            if len(scores) == 0:
                continue
            
            scores = np.array(scores)
            
            # 清理数据
            clean_scores = scores[scores > 0]
            clean_scores = clean_scores[np.isfinite(clean_scores)]
            
            if len(clean_scores) < 10:
                # 使用默认阈值
                α_h, α_mh, α_m, α_ml = 1e-3, 1e-4, 1e-5, 1e-6
            else:
                # 计算分位数阈值
                sorted_scores = np.sort(clean_scores)[::-1]
                n = len(sorted_scores)
                positions = cumulative * (n - 1)
                
                α_h = np.interp(positions[0], np.arange(n), sorted_scores)
                α_mh = np.interp(positions[1], np.arange(n), sorted_scores)
                α_m = np.interp(positions[2], np.arange(n), sorted_scores)
                α_ml = np.interp(positions[3], np.arange(n), sorted_scores)
                
                # 确保阈值为正且递减
                α_h = max(α_h, 1e-8)
                α_mh = max(α_mh, 1e-9)
                α_m = max(α_m, 1e-10)
                α_ml = max(α_ml, 1e-11)
                
                # 确保严格递减
                min_sep = 1e-10
                if α_mh >= α_h:
                    α_mh = α_h - min_sep
                if α_m >= α_mh:
                    α_m = α_mh - min_sep
                if α_ml >= α_m:
                    α_ml = α_m - min_sep
            
            # 存储阈值
            if self.config.granularity == "per_layer":
                layer_id = key[0]
                head_id = None
                threshold_key = f'layer_{layer_id}'
            else:
                layer_id, head_id = key
                threshold_key = f'layer_{layer_id}_head_{head_id}'
            
            thresholds[threshold_key] = {
                'alpha_h': float(α_h),
                'alpha_mh': float(α_mh),
                'alpha_m': float(α_m),
                'alpha_ml': float(α_ml)
            }
            
            # 记录详细数据
            for quantile_name, threshold_value in [
                ('alpha_h', α_h), ('alpha_mh', α_mh), 
                ('alpha_m', α_m), ('alpha_ml', α_ml)
            ]:
                record = ThresholdRecord(
                    dataset_name=dataset_name,
                    sample_size=sample_size,
                    layer_id=layer_id,
                    head_id=head_id,
                    quantile_name=quantile_name,
                    threshold_value=float(threshold_value),
                    collection_timestamp=timestamp,
                    model_config=self.model_config,
                    bootstrap_iteration=bootstrap_iteration
                )
                self.collected_records.append(record)
        
        return thresholds
    
    def _compute_diffkv_importance(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算DiffKV重要性分数
        
        Args:
            attention_weights: [B, H, T_q, T_k] 注意力权重
            
        Returns:
            importance_scores: [B, H, T_k] 重要性分数
        """
        B, H, T_q, T_k = attention_weights.shape
        
        # 步骤1: 注意力累积
        importance = attention_weights.mean(dim=2)  # [B, H, T_k]
        
        return importance
    
    def _load_dataset(self, dataset_name: str, num_samples: int) -> List[str]:
        """
        加载数据集
        
        Args:
            dataset_name: 数据集名称
            num_samples: 样本数量
            
        Returns:
            texts: 文本列表
        """
        if LEANSPARSE_AVAILABLE:
            return load_calibration_data(dataset_name, num_samples)
        else:
            # 使用内置数据集
            return self._load_builtin_dataset(dataset_name, num_samples)
    
    def _load_builtin_dataset(self, dataset_name: str, num_samples: int) -> List[str]:
        """
        加载内置数据集
        
        Args:
            dataset_name: 数据集名称
            num_samples: 样本数量
            
        Returns:
            texts: 文本列表
        """
        logger.info(f"加载内置数据集: {dataset_name}")
        
        if dataset_name.lower() == "math":
            texts = [
                "What is 2 + 2?",
                "Calculate 15 * 7.",
                "What is the square root of 64?",
                "Solve for x: 2x + 5 = 13",
                "What is 100 divided by 4?",
                "Calculate 3^4.",
                "What is the area of a circle with radius 5?",
                "Solve: 2x - 3 = 7",
                "What is 25% of 80?",
                "Calculate the factorial of 5."
            ] * (num_samples // 10 + 1)
            
        elif dataset_name.lower() == "gsm8k":
            texts = [
                "A store sells pencils for $0.25 each. If Sarah buys 8 pencils, how much does she spend?",
                "Tom has 24 marbles. He gives away 1/3 of them. How many marbles does Tom have left?",
                "A recipe calls for 2.5 cups of flour to make 12 cookies. How many cups for 30 cookies?",
                "There are 30 students in a class. If 60% are girls, how many boys are there?",
                "A car travels 60 miles in 1 hour. How far will it travel in 2.5 hours?"
            ] * (num_samples // 5 + 1)
            
        elif dataset_name.lower() == "wikitext":
            texts = [
                "The capital of France is Paris, known for its beautiful architecture and rich history.",
                "Artificial intelligence is transforming industries including healthcare and finance.",
                "Climate change represents one of the most pressing challenges of our time.",
                "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
                "Renewable energy sources like solar and wind are becoming increasingly cost-effective."
            ] * (num_samples // 5 + 1)
            
        elif dataset_name.lower() == "alpaca":
            texts = [
                "Explain the concept of machine learning in simple terms.",
                "Write a short story about a robot learning to paint.",
                "Describe the process of photosynthesis in plants.",
                "What are the benefits of regular exercise for mental health?",
                "How do you make a simple chocolate cake from scratch?"
            ] * (num_samples // 5 + 1)
            
        else:
            # 默认使用通用文本
            texts = [
                "This is a sample text for threshold calibration.",
                "Natural language processing involves understanding human language.",
                "Machine learning models require large amounts of training data.",
                "Deep learning has revolutionized computer vision and speech recognition.",
                "Attention mechanisms allow models to focus on relevant information."
            ] * (num_samples // 5 + 1)
        
        return texts[:num_samples]
    
    def _flatten_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        将嵌套阈值字典展平
        
        Args:
            thresholds: 嵌套阈值字典
            
        Returns:
            flat_thresholds: 展平的阈值字典
        """
        flat_thresholds = {}
        
        for layer_key, layer_thresholds in thresholds.items():
            for quantile_name, threshold_value in layer_thresholds.items():
                flat_key = f"{layer_key}_{quantile_name}"
                flat_thresholds[flat_key] = threshold_value
        
        return flat_thresholds
    
    def get_collected_records(self) -> List[ThresholdRecord]:
        """
        获取收集的记录
        
        Returns:
            collected_records: 收集的阈值记录列表
        """
        return self.collected_records.copy()
    
    def clear_collected_records(self):
        """清空收集的记录"""
        self.collected_records.clear()
        logger.info("已清空收集的记录")