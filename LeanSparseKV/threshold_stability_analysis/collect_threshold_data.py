#!/usr/bin/env python3
"""
Threshold Data Collection Script

Specialized script for collecting multi-dimensional threshold data for stability validation
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_collector import ThresholdCollector, CollectionConfig, ThresholdRecord
from data_storage import ThresholdDataStorage, StorageConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThresholdDataCollector:
    """
    Threshold Data Collector
    
    Specialized for collecting multi-dimensional threshold data
    """
    
    def __init__(self, model_path: str, output_dir: str = "./threshold_data"):
        """
        Initialize data collector
        
        Args:
            model_path: Model path
            output_dir: Output directory
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure data collection
        self.collection_config = CollectionConfig(
            model_path=model_path,
            datasets=['wikitext'],  # Can extend to ['wikitext', 'math', 'gsm8k']
            sample_sizes=[20, 50],  # Different sample sizes
            target_sparsity=0.70,
            granularity='per_layer',
            max_length=2048,
            batch_size=2,
            device='auto',
            bootstrap_samples=10  # Bootstrap sampling iterations
        )
        
        # Configure data storage
        self.storage_config = StorageConfig(
            storage_dir=str(self.output_dir),
            use_database=True,
            use_json_backup=True,
            use_pickle_backup=False
        )
        
        logger.info(f"Data collector initialized, output directory: {self.output_dir}")
    
    def collect_all_data(self) -> str:
        """
        Collect all threshold data
        
        Returns:
            session_id: Data session ID
        """
        logger.info("🚀 Starting threshold data collection...")
        
        collector = ThresholdCollector(self.collection_config)
        all_records = []
        
        total_combinations = (
            len(self.collection_config.datasets) * 
            len(self.collection_config.sample_sizes) * 
            self.collection_config.bootstrap_samples
        )
        
        current_combination = 0
        
        for dataset_name in self.collection_config.datasets:
            for sample_size in self.collection_config.sample_sizes:
                logger.info(f"📊 Collecting data: {dataset_name}, sample size: {sample_size}")
                
                # Perform multiple Bootstrap sampling
                for bootstrap_iter in range(self.collection_config.bootstrap_samples):
                    current_combination += 1
                    try:
                        logger.info(f"Bootstrap {bootstrap_iter+1}/{self.collection_config.bootstrap_samples} "
                                  f"({current_combination}/{total_combinations})")
                        
                        # Load dataset
                        calibration_texts = collector._load_dataset(dataset_name, sample_size)
                        
                        # Collect thresholds
                        thresholds = collector._collect_single_configuration(
                            dataset_name, sample_size, calibration_texts
                        )
                        
                        # Convert to record format
                        timestamp = datetime.now()
                        for layer_id, layer_thresholds in thresholds.items():
                            for quantile_name, threshold_value in layer_thresholds.items():
                                record = ThresholdRecord(
                                    dataset_name=dataset_name,
                                    sample_size=sample_size,
                                    layer_id=layer_id,
                                    head_id=None,
                                    quantile_name=quantile_name,
                                    threshold_value=threshold_value,
                                    collection_timestamp=timestamp,
                                    model_config=collector.model_config,
                                    bootstrap_iteration=bootstrap_iter
                                )
                                all_records.append(record)
                        
                        logger.info(f"✅ Bootstrap {bootstrap_iter+1} completed, collected {len(thresholds)} layers")
                        
                    except Exception as e:
                        logger.error(f"❌ Bootstrap {bootstrap_iter} failed: {e}")
                        continue
        
        # Save data
        session_id = None
        with ThresholdDataStorage(self.storage_config) as storage:
            if all_records:
                session_id = storage.save_threshold_records(all_records)
                logger.info(f"💾 Data saved, session ID: {session_id}, total records: {len(all_records)}")
                
                # Save collection configuration
                config_path = self.output_dir / f"collection_config_{session_id}.json"
                config_dict = {
                    'model_path': self.model_path,
                    'datasets': self.collection_config.datasets,
                    'sample_sizes': self.collection_config.sample_sizes,
                    'bootstrap_samples': self.collection_config.bootstrap_samples,
                    'target_sparsity': self.collection_config.target_sparsity,
                    'total_records': len(all_records),
                    'collection_timestamp': datetime.now().isoformat()
                }
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📋 Collection config saved: {config_path}")
            else:
                logger.error("❌ No data collected!")
        
        return session_id
    
    def get_data_summary(self, session_id: str = None) -> Dict[str, Any]:
        """
        Get data summary
        
        Args:
            session_id: Session ID, if None get latest data
            
        Returns:
            summary: Data summary
        """
        with ThresholdDataStorage(self.storage_config) as storage:
            if session_id:
                records = storage.load_threshold_records(session_id=session_id)
            else:
                records = storage.load_threshold_records()
            
            if not records:
                return {'error': 'No data found'}
            
            # Statistics
            datasets = set(r.dataset_name for r in records)
            sample_sizes = set(r.sample_size for r in records)
            layers = set(r.layer_id for r in records)
            quantiles = set(r.quantile_name for r in records)
            bootstrap_iterations = set(r.bootstrap_iteration for r in records if r.bootstrap_iteration is not None)
            
            summary = {
                'total_records': len(records),
                'datasets': sorted(list(datasets)),
                'sample_sizes': sorted(list(sample_sizes)),
                'layers': sorted(list(layers)),
                'quantiles': sorted(list(quantiles)),
                'bootstrap_iterations': len(bootstrap_iterations),
                'layer_count': len(layers),
                'quantile_count': len(quantiles),
                'data_completeness': {
                    'expected_combinations': len(datasets) * len(sample_sizes) * len(layers) * len(quantiles) * len(bootstrap_iterations),
                    'actual_records': len(records),
                    'completeness_ratio': len(records) / (len(datasets) * len(sample_sizes) * len(layers) * len(quantiles) * len(bootstrap_iterations)) if len(bootstrap_iterations) > 0 else 0
                }
            }
            
            return summary


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Threshold Data Collection Script')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Model path')
    parser.add_argument('--output-dir', type=str, default='./threshold_data',
                       help='Output directory')
    parser.add_argument('--datasets', nargs='+', default=['wikitext'],
                       choices=['wikitext', 'math', 'gsm8k', 'alpaca'],
                       help='Dataset list')
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[20, 50],
                       help='Sample size list')
    parser.add_argument('--bootstrap-samples', type=int, default=10,
                       help='Bootstrap sampling iterations')
    
    args = parser.parse_args()
    
    # Create collector
    collector = ThresholdDataCollector(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # Update configuration
    collector.collection_config.datasets = args.datasets
    collector.collection_config.sample_sizes = args.sample_sizes
    collector.collection_config.bootstrap_samples = args.bootstrap_samples
    
    # Collect data
    session_id = collector.collect_all_data()
    
    if session_id:
        # Show summary
        summary = collector.get_data_summary(session_id)
        
        print("\n" + "="*60)
        print("📊 DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"Session ID: {session_id}")
        print(f"Total records: {summary['total_records']}")
        print(f"Datasets: {summary['datasets']}")
        print(f"Sample sizes: {summary['sample_sizes']}")
        print(f"Layer count: {summary['layer_count']}")
        print(f"Quantiles: {summary['quantiles']}")
        print(f"Bootstrap samples: {summary['bootstrap_iterations']} iterations")
        print(f"Data completeness: {summary['data_completeness']['completeness_ratio']:.1%}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
        print(f"\n💡 Next step - run visualization script:")
        print(f"python visualize_threshold_stability.py --data-dir {args.output_dir} --session-id {session_id}")
    else:
        print("❌ Data collection failed!")


if __name__ == "__main__":
    main()