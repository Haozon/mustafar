"""
JSQKV Benchmark - 吞吐量测试主脚本
测试不同配置下的模型吞吐量性能
"""
import os
import sys
import json
import torch
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, validate_config
from utils.model_loader import load_model
from utils.metrics import measure_throughput

def run_single_benchmark(model_name, model_config, config_name, test_config, batch_size, global_config):
    """运行单个配置的benchmark"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name} | {config_name} | Batch Size = {batch_size}")
    print(f"{'='*70}")
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 添加全局配置参数
    test_config_with_globals = test_config.copy()
    test_config_with_globals['group_size'] = global_config.get('group_size', 32)
    test_config_with_globals['residual_length'] = global_config.get('residual_length', 256)
    
    try:
        # 加载模型
        model, tokenizer = load_model(model_config, test_config_with_globals)
        
        # 测量性能
        results = measure_throughput(
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            input_length=model_config['input_length'],
            output_length=model_config['output_length'],
            num_repeats=global_config.get('num_repeats', 3),
            warmup_tokens=global_config.get('warmup_tokens', 10)
        )
        
        # 清理显存
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"❌ Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_all_benchmarks(config, models_to_test=None, configs_to_test=None):
    """运行所有benchmark测试"""
    
    # 验证配置
    validate_config(config)
    
    # 确定要测试的模型和配置
    if models_to_test is None:
        models_to_test = list(config['models'].keys())
    if configs_to_test is None:
        configs_to_test = list(config['test_configs'].keys())
    
    print(f"\n{'='*70}")
    print(f"JSQKV Benchmark - Throughput Testing")
    print(f"{'='*70}")
    print(f"Models to test: {models_to_test}")
    print(f"Configs to test: {configs_to_test}")
    print(f"Batch sizes: {config['batch_sizes']}")
    print(f"{'='*70}\n")
    
    # 存储所有结果
    all_results = {}
    
    # 遍历所有模型
    for model_name in models_to_test:
        if model_name not in config['models']:
            print(f"⚠️  Warning: Model '{model_name}' not found in config, skipping...")
            continue
            
        model_config = config['models'][model_name]
        all_results[model_name] = {}
        
        print(f"\n{'#'*70}")
        print(f"# Testing Model: {model_config['display_name']}")
        print(f"# Path: {model_config['path']}")
        print(f"# Input Length: {model_config['input_length']}, Output Length: {model_config['output_length']}")
        print(f"{'#'*70}\n")
        
        # 遍历所有测试配置
        for config_name in configs_to_test:
            if config_name not in config['test_configs']:
                print(f"⚠️  Warning: Config '{config_name}' not found, skipping...")
                continue
                
            test_config = config['test_configs'][config_name]
            all_results[model_name][config_name] = {}
            
            print(f"\n{'*'*70}")
            print(f"* Testing Configuration: {test_config['display_name']}")
            print(f"{'*'*70}")
            
            # 遍历所有batch size
            for batch_size in config['batch_sizes']:
                results = run_single_benchmark(
                    model_name=model_name,
                    model_config=model_config,
                    config_name=config_name,
                    test_config=test_config,
                    batch_size=batch_size,
                    global_config=config
                )
                
                if results is not None:
                    all_results[model_name][config_name][batch_size] = results
                    print(f"\n✅ Completed: {model_name} | {config_name} | BS={batch_size}")
                    print(f"   Throughput: {results['throughput']:.2f} tokens/sec")
                    print(f"   Peak Memory: {results['peak_memory']:.2f} GB")
                else:
                    print(f"\n❌ Failed: {model_name} | {config_name} | BS={batch_size}")
    
    return all_results

def save_results(results, output_dir='results/raw_data'):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_name, model_results in results.items():
        # 保存为JSON
        output_file = os.path.join(output_dir, f"{model_name}_results.json")
        with open(output_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"✅ Saved results: {output_file}")
        
        # 同时保存带时间戳的备份
        backup_file = os.path.join(output_dir, f"{model_name}_results_{timestamp}.json")
        with open(backup_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"✅ Saved backup: {backup_file}")

def print_summary(results):
    """打印测试结果摘要"""
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"{'-'*70}")
        
        for config_name, config_results in model_results.items():
            print(f"\n  {config_name}:")
            for batch_size, metrics in sorted(config_results.items()):
                print(f"    BS={batch_size}: {metrics['throughput']:.2f} tokens/sec, "
                      f"Memory={metrics['peak_memory']:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='JSQKV Benchmark - Throughput Testing')
    parser.add_argument('--config', type=str, default='benchmark_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to test (default: all)')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Specific configs to test (default: all)')
    parser.add_argument('--output-dir', type=str, default='results/raw_data',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 运行benchmark
    results = run_all_benchmarks(
        config=config,
        models_to_test=args.models,
        configs_to_test=args.configs
    )
    
    # 保存结果
    save_results(results, output_dir=args.output_dir)
    
    # 打印摘要
    print_summary(results)
    
    print(f"\n{'='*70}")
    print(f"✅ All benchmarks completed!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
