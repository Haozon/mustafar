# 阈值稳定性分析系统

用于分析Llama3模型中DiffKV算法的阈值稳定性，评估固定阈值替代动态校准的可行性。

## 系统概述

本系统实现了完整的阈值数据收集、存储和分析流程，支持多数据集、多样本大小的阈值收集，以及Bootstrap采样和交叉验证等统计分析方法。

## 核心组件

### 1. 数据收集器 (ThresholdCollector)

负责从多个数据集收集阈值数据，支持Bootstrap采样和与LeanSparseKV系统的集成。

**主要功能:**
- ✅ 多数据集阈值收集
- ✅ Bootstrap采样方法
- ✅ 交叉验证支持
- ✅ 与LeanSparseKV校准系统集成
- ✅ 内置数据集支持（当外部数据集不可用时）
- ✅ 灵活的粒度配置（按层或按头）

**支持的数据集:**
- `math`: 数学问题
- `gsm8k`: GSM8K数学推理
- `wikitext`: WikiText文本
- `alpaca`: Alpaca指令数据

### 2. 数据存储系统 (ThresholdDataStorage)

提供多种格式的数据存储和管理功能。

**主要功能:**
- ✅ SQLite数据库存储
- ✅ JSON文件备份
- ✅ Pickle文件备份
- ✅ 数据查询和统计
- ✅ 会话管理
- ✅ CSV导出功能
- ✅ 阈值矩阵提取

## 数据结构

### ThresholdRecord
```python
@dataclass
class ThresholdRecord:
    dataset_name: str              # 数据集名称
    sample_size: int               # 样本大小
    layer_id: int                  # 层ID
    head_id: Optional[int]         # 头ID（可选）
    quantile_name: str             # 分位点名称
    threshold_value: float         # 阈值
    collection_timestamp: datetime # 收集时间戳
    model_config: Dict[str, Any]   # 模型配置
    bootstrap_iteration: Optional[int] # Bootstrap迭代次数
```

### 配置类
```python
@dataclass
class CollectionConfig:
    model_path: str                # 模型路径
    datasets: List[str]            # 数据集列表
    sample_sizes: List[int]        # 样本大小列表
    target_sparsity: float = 0.70  # 目标稀疏度
    granularity: str = "per_layer" # 粒度配置
    # ... 其他配置参数

@dataclass
class StorageConfig:
    storage_dir: str               # 存储目录
    use_database: bool = True      # 使用数据库
    use_json_backup: bool = True   # JSON备份
    use_pickle_backup: bool = True # Pickle备份
    # ... 其他存储配置
```

## 使用方法

### 基本使用

```python
from threshold_stability_analysis.data_collector import ThresholdCollector, CollectionConfig
from threshold_stability_analysis.data_storage import ThresholdDataStorage, StorageConfig

# 1. 配置数据收集
config = CollectionConfig(
    model_path="/path/to/llama3/model",
    datasets=["math", "gsm8k"],
    sample_sizes=[50, 100, 200],
    target_sparsity=0.70,
    granularity="per_layer"
)

# 2. 初始化收集器
collector = ThresholdCollector(config)

# 3. 收集多数据集阈值
threshold_data = collector.collect_thresholds_multi_dataset()

# 4. Bootstrap采样
bootstrap_results = collector.collect_bootstrap_samples("math", n_bootstrap=100)

# 5. 交叉验证
cv_results = collector.collect_cross_validation_thresholds("math", k_folds=5)
```

### 数据存储和管理

```python
# 1. 配置存储
storage_config = StorageConfig(
    storage_dir="./results",
    use_database=True,
    use_json_backup=True
)

# 2. 保存数据
with ThresholdDataStorage(storage_config) as storage:
    records = collector.get_collected_records()
    session_id = storage.save_threshold_records(records)
    
    # 3. 查询数据
    loaded_records = storage.load_threshold_records(dataset_name="math")
    
    # 4. 获取统计信息
    stats = storage.get_dataset_statistics()
    
    # 5. 导出CSV
    storage.export_to_csv("threshold_data.csv")
```

### 高级功能

```python
# 获取阈值矩阵用于分析
matrix = storage.get_threshold_matrix("math", "alpha_h")

# 计算稳定性指标
mean_vals = np.mean(matrix, axis=1)
std_vals = np.std(matrix, axis=1)
cv_vals = std_vals / mean_vals  # 变异系数

# 识别稳定的层
stable_layers = np.where(cv_vals < 0.1)[0]
```

## 与LeanSparseKV集成

系统自动检测并集成现有的LeanSparseKV校准系统：

```python
# 自动使用LeanSparseKV（如果可用）
if LEANSPARSE_AVAILABLE:
    calibrator = DiffKVThresholdCalibrator(...)
    # 使用现有的校准逻辑
else:
    # 使用内置实现
    # 提供相同的功能接口
```

## 文件结构

```
threshold_stability_analysis/
├── __init__.py              # 包初始化
├── data_collector.py        # 数据收集器
├── data_storage.py          # 数据存储系统
└── README.md               # 本文档

# 生成的数据文件
results/
├── thresholds.db           # SQLite数据库
├── session_*.json          # JSON备份文件
├── session_*.pkl           # Pickle备份文件
└── *.csv                   # 导出的CSV文件
```

## 依赖要求

```bash
# 核心依赖
pip install torch transformers numpy pandas sqlite3

# 可选依赖（用于数据集加载）
pip install datasets

# 可视化依赖（后续模块需要）
pip install matplotlib seaborn plotly
```

## 性能优化

1. **批处理**: 支持批量处理以提高效率
2. **内存管理**: 自动清理GPU内存
3. **数据压缩**: 可选的数据压缩存储
4. **索引优化**: 数据库索引加速查询
5. **并行处理**: 支持多进程数据收集（未来版本）

## 错误处理

系统提供完善的错误处理机制：

1. **网络错误**: 自动回退到内置数据集
2. **模型加载错误**: 提供详细的错误信息和解决建议
3. **数据验证**: 自动验证和清理异常数据
4. **存储错误**: 多重备份机制确保数据安全

## 扩展性

系统设计具有良好的扩展性：

1. **新数据集**: 易于添加新的数据集支持
2. **新模型**: 支持不同的Transformer架构
3. **新分析方法**: 模块化设计便于添加新功能
4. **新存储格式**: 灵活的存储接口

## 测试

运行测试以验证系统功能：

```bash
# 基本功能测试
python test_threshold_collector.py

# 完整示例
python example_usage.py
```

## 下一步开发

1. **统计分析器**: 实现StabilityAnalyzer类
2. **可视化引擎**: 实现VisualizationEngine类
3. **固定阈值评估器**: 实现FixedThresholdEvaluator类
4. **预测模型**: 实现ThresholdPredictor类
5. **报告生成器**: 实现ReportGenerator类

## 许可证

本项目遵循MIT许可证。