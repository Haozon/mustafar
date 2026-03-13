"""配置加载器"""
import yaml
import os

def load_config(config_path='benchmark_config.yaml'):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_test_configs(config):
    """获取所有测试配置"""
    return config['test_configs']

def get_model_configs(config):
    """获取所有模型配置"""
    return config['models']

def get_plot_schemes(config):
    """获取所有绘图方案"""
    return config['plot_schemes']

def validate_config(config):
    """验证配置文件的完整性"""
    required_keys = ['models', 'test_configs', 'plot_schemes', 'batch_sizes']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")
    
    # 验证绘图方案中引用的配置是否存在
    test_config_names = set(config['test_configs'].keys())
    
    for scheme_name, scheme_config in config['plot_schemes'].items():
        for config_name in scheme_config['configs']:
            if config_name not in test_config_names:
                raise ValueError(
                    f"Plot scheme '{scheme_name}' references unknown config '{config_name}'"
                )
    
    print("✅ Configuration validated successfully")
    return True
