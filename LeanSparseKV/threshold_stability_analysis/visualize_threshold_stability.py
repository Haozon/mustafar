#!/usr/bin/env python3
"""
Threshold Stability Visualization Script

Specialized script for visualizing threshold stability analysis with English labels
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_collector import ThresholdRecord
from data_storage import ThresholdDataStorage, StorageConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StabilityValidationResult:
    """Stability validation result"""
    layer_id: int
    quantile_name: str
    mean_threshold: float
    std_threshold: float
    coefficient_of_variation: float  # CV = std/mean
    median_threshold: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    iqr: float  # Interquartile range
    outlier_count: int
    total_samples: int
    outlier_ratio: float
    is_stable: bool  # CV < 0.1 and outlier_ratio < 0.05

class ThresholdStabilityVisualizer:
    """
    Threshold Stability Visualizer
    
    Specialized for visualizing threshold stability analysis results
    """
    
    def __init__(self, data_dir: str, output_dir: str = "./visualization_results"):
        """
        Initialize visualizer
        
        Args:
            data_dir: Data directory containing threshold data
            output_dir: Output directory for visualizations
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure data storage
        self.storage_config = StorageConfig(
            storage_dir=str(self.data_dir),
            use_database=True,
            use_json_backup=True,
            use_pickle_backup=False
        )
        
        # Stability thresholds
        self.cv_threshold = 0.1  # Coefficient of variation threshold
        self.outlier_threshold = 0.05  # Outlier ratio threshold
        
        logger.info(f"Visualizer initialized, output directory: {self.output_dir}")
    
    def run_complete_visualization(self, session_id: str = None) -> Dict[str, Any]:
        """
        Run complete visualization pipeline
        
        Args:
            session_id: Session ID to load data from
            
        Returns:
            visualization_results: Visualization results summary
        """
        logger.info("🚀 Starting threshold stability visualization...")
        
        # Step 1: Load threshold data
        logger.info("📊 Step 1: Loading threshold data...")
        threshold_data = self._load_threshold_data(session_id)
        
        # Step 2: Analyze stability
        logger.info("🔍 Step 2: Analyzing stability...")
        stability_results = self._analyze_stability(threshold_data)
        
        # Step 3: Generate visualizations
        logger.info("📈 Step 3: Generating visualizations...")
        viz_results = self._generate_all_visualizations(threshold_data, stability_results)
        
        # Step 4: Generate report
        logger.info("📋 Step 4: Generating validation report...")
        validation_summary = self._generate_validation_report(stability_results)
        
        logger.info("✅ Threshold stability visualization completed!")
        return {
            'visualization_files': viz_results,
            'validation_summary': validation_summary
        }
    
    def _load_threshold_data(self, session_id: str = None) -> List[ThresholdRecord]:
        """Load threshold data from storage"""
        with ThresholdDataStorage(self.storage_config) as storage:
            if session_id:
                records = storage.load_threshold_records(session_id=session_id)
                logger.info(f"Loaded {len(records)} records from session {session_id}")
            else:
                records = storage.load_threshold_records()
                logger.info(f"Loaded {len(records)} records from latest data")
            
            if not records:
                raise ValueError("No threshold data found!")
            
            return records
    
    def _analyze_stability(self, threshold_data: List[ThresholdRecord]) -> Dict[Tuple[int, str], StabilityValidationResult]:
        """
        Analyze stability for each layer-quantile combination
        
        Args:
            threshold_data: List of threshold records
            
        Returns:
            stability_results: {(layer_id, quantile_name): StabilityValidationResult}
        """
        # Group data by layer and quantile
        grouped_data = {}
        for record in threshold_data:
            key = (record.layer_id, record.quantile_name)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(record.threshold_value)
        
        stability_results = {}
        
        for (layer_id, quantile_name), values in grouped_data.items():
            values_array = np.array(values)
            
            # Basic statistics
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            cv = std_val / mean_val if mean_val > 0 else float('inf')
            median_val = np.median(values_array)
            q25, q75 = np.percentile(values_array, [25, 75])
            iqr = q75 - q25
            
            # Outlier detection (using IQR method)
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = values_array[(values_array < lower_bound) | (values_array > upper_bound)]
            outlier_count = len(outliers)
            outlier_ratio = outlier_count / len(values_array)
            
            # Stability assessment
            is_stable = (cv < self.cv_threshold) and (outlier_ratio < self.outlier_threshold)
            
            result = StabilityValidationResult(
                layer_id=layer_id,
                quantile_name=quantile_name,
                mean_threshold=mean_val,
                std_threshold=std_val,
                coefficient_of_variation=cv,
                median_threshold=median_val,
                q25=q25,
                q75=q75,
                iqr=iqr,
                outlier_count=outlier_count,
                total_samples=len(values_array),
                outlier_ratio=outlier_ratio,
                is_stable=is_stable
            )
            
            stability_results[(layer_id, quantile_name)] = result
        
        return stability_results
    
    def _generate_all_visualizations(self, threshold_data: List[ThresholdRecord], 
                                   stability_results: Dict[Tuple[int, str], StabilityValidationResult]) -> List[str]:
        """Generate all visualizations"""
        
        # Prepare DataFrame
        df_data = []
        for record in threshold_data:
            df_data.append({
                'layer_id': record.layer_id,
                'quantile_name': record.quantile_name,
                'threshold_value': record.threshold_value,
                'dataset_name': record.dataset_name,
                'sample_size': record.sample_size,
                'bootstrap_iteration': record.bootstrap_iteration
            })
        
        df = pd.DataFrame(df_data)
        
        # Get unique values
        layers = sorted(df['layer_id'].unique())
        quantiles = ['alpha_h', 'alpha_mh', 'alpha_m', 'alpha_ml']
        
        viz_files = []
        
        # 1. Main boxplot
        main_boxplot_file = self._create_main_boxplot(df, layers, quantiles, stability_results)
        if main_boxplot_file:
            viz_files.append(main_boxplot_file)
        
        # 2. CV heatmap
        cv_heatmap_file = self._create_cv_heatmap(stability_results)
        if cv_heatmap_file:
            viz_files.append(cv_heatmap_file)
        
        # 3. Stability statistics
        stats_plot_file = self._create_stability_stats_plot(stability_results)
        if stats_plot_file:
            viz_files.append(stats_plot_file)
        
        return viz_files
    
    def _create_main_boxplot(self, df: pd.DataFrame, layers: List[int], quantiles: List[str],
                            stability_results: Dict[Tuple[int, str], StabilityValidationResult]) -> str:
        """Create main boxplot for threshold stability validation"""
        
        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create large figure
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Prepare plot data
        plot_data = []
        positions = []
        labels = []
        colors = []
        
        color_map = {'alpha_h': '#FF6B6B', 'alpha_mh': '#4ECDC4', 'alpha_m': '#45B7D1', 'alpha_ml': '#96CEB4'}
        
        pos = 0
        for layer_id in layers:
            for i, quantile in enumerate(quantiles):
                layer_quantile_data = df[(df['layer_id'] == layer_id) & (df['quantile_name'] == quantile)]
                if not layer_quantile_data.empty:
                    plot_data.append(layer_quantile_data['threshold_value'].values)
                    positions.append(pos)
                    colors.append(color_map[quantile])
                    
                    # Add stability marker
                    key = (layer_id, quantile)
                    if key in stability_results:
                        stability_mark = "✓" if stability_results[key].is_stable else "✗"
                        cv = stability_results[key].coefficient_of_variation
                        labels.append(f'L{layer_id}-{quantile}\n{stability_mark} CV:{cv:.3f}')
                    else:
                        labels.append(f'L{layer_id}-{quantile}')
                    
                    pos += 1
            
            # Add spacing between layers
            if layer_id < max(layers):
                pos += 0.5
        
        # Create boxplot
        if plot_data and len(plot_data) == len(positions):
            bp = ax.boxplot(plot_data, positions=positions, patch_artist=True, 
                           widths=0.6, showfliers=True, flierprops={'marker': 'o', 'markersize': 3})
            
            # Set colors
            for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
                if i < len(colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            # Set labels
            if len(positions) == len(labels):
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            
            # Set Y-axis to log scale
            ax.set_yscale('log')
            ax.set_ylabel('Threshold Value (Log Scale)', fontsize=14)
            ax.set_xlabel('Layer-Quantile Combinations', fontsize=14)
            
            # Add title
            ax.set_title('Threshold Stability Validation: Layer-Quantile Boxplot Analysis\n'
                        '✓ = Stable (CV<0.1, Outliers<5%), ✗ = Unstable', 
                        fontsize=16, pad=20)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=quantile) 
                             for quantile, color in color_map.items()]
            ax.legend(handles=legend_elements, loc='upper right', title='Quantiles')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            output_path = self.output_dir / "main_boxplot_stability_validation.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Main boxplot saved: {output_path}")
            
            plt.close()
            return str(output_path)
        else:
            logger.error("No data available for boxplot")
            plt.close()
            return ""
    
    def _create_cv_heatmap(self, stability_results: Dict[Tuple[int, str], StabilityValidationResult]) -> str:
        """Create coefficient of variation heatmap"""
        
        # Prepare data
        layers = sorted(set(key[0] for key in stability_results.keys()))
        quantiles = ['alpha_h', 'alpha_mh', 'alpha_m', 'alpha_ml']
        
        cv_matrix = np.zeros((len(layers), len(quantiles)))
        
        for i, layer_id in enumerate(layers):
            for j, quantile in enumerate(quantiles):
                key = (layer_id, quantile)
                if key in stability_results:
                    cv_matrix[i, j] = stability_results[key].coefficient_of_variation
                else:
                    cv_matrix[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use green for stable (low CV), red for unstable (high CV)
        sns.heatmap(cv_matrix, 
                   xticklabels=quantiles,
                   yticklabels=[f'Layer {l}' for l in layers],
                   annot=True, fmt='.3f', cmap='RdYlGn_r',
                   vmin=0, vmax=0.2, ax=ax,
                   cbar_kws={'label': 'Coefficient of Variation (CV)'})
        
        ax.set_title('Threshold Stability Heatmap\n(Green=Stable, Red=Unstable, CV<0.1 is Stable)', fontsize=14)
        ax.set_xlabel('Quantiles', fontsize=12)
        ax.set_ylabel('Layers', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "cv_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"CV heatmap saved: {output_path}")
        
        plt.close()
        return str(output_path)
    
    def _create_stability_stats_plot(self, stability_results: Dict[Tuple[int, str], StabilityValidationResult]) -> str:
        """Create stability statistics plot"""
        
        # Calculate stability statistics
        total_combinations = len(stability_results)
        stable_combinations = sum(1 for result in stability_results.values() if result.is_stable)
        stability_rate = stable_combinations / total_combinations if total_combinations > 0 else 0
        
        # Statistics by quantile
        quantile_stats = {}
        for quantile in ['alpha_h', 'alpha_mh', 'alpha_m', 'alpha_ml']:
            quantile_results = [result for (layer_id, q), result in stability_results.items() if q == quantile]
            if quantile_results:
                stable_count = sum(1 for result in quantile_results if result.is_stable)
                quantile_stats[quantile] = {
                    'total': len(quantile_results),
                    'stable': stable_count,
                    'stability_rate': stable_count / len(quantile_results),
                    'avg_cv': np.mean([result.coefficient_of_variation for result in quantile_results])
                }
        
        # Create statistics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Stability ratio
        quantiles = list(quantile_stats.keys())
        stability_rates = [quantile_stats[q]['stability_rate'] for q in quantiles]
        
        bars = ax1.bar(quantiles, stability_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        ax1.set_title('Stability Rate by Quantile', fontsize=14)
        ax1.set_ylabel('Stability Rate', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target Threshold (80%)')
        
        # Add value labels
        for bar, rate in zip(bars, stability_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Right plot: Average coefficient of variation
        avg_cvs = [quantile_stats[q]['avg_cv'] for q in quantiles]
        
        bars2 = ax2.bar(quantiles, avg_cvs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        ax2.set_title('Average Coefficient of Variation by Quantile', fontsize=14)
        ax2.set_ylabel('Average CV', fontsize=12)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Stability Threshold (0.1)')
        
        # Add value labels
        for bar, cv in zip(bars2, avg_cvs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{cv:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "stability_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Stability statistics plot saved: {output_path}")
        
        plt.close()
        return str(output_path)
    
    def _generate_validation_report(self, stability_results: Dict[Tuple[int, str], StabilityValidationResult]) -> Dict[str, Any]:
        """Generate validation report"""
        
        # Overall statistics
        total_combinations = len(stability_results)
        stable_combinations = sum(1 for result in stability_results.values() if result.is_stable)
        overall_stability_rate = stable_combinations / total_combinations if total_combinations > 0 else 0
        
        # Statistics by quantile
        quantile_analysis = {}
        for quantile in ['alpha_h', 'alpha_mh', 'alpha_m', 'alpha_ml']:
            quantile_results = [result for (layer_id, q), result in stability_results.items() if q == quantile]
            if quantile_results:
                stable_count = sum(1 for result in quantile_results if result.is_stable)
                quantile_analysis[quantile] = {
                    'total_layers': len(quantile_results),
                    'stable_layers': stable_count,
                    'stability_rate': stable_count / len(quantile_results),
                    'avg_cv': np.mean([result.coefficient_of_variation for result in quantile_results]),
                    'min_cv': min(result.coefficient_of_variation for result in quantile_results),
                    'max_cv': max(result.coefficient_of_variation for result in quantile_results)
                }
        
        # Most and least stable combinations
        sorted_results = sorted(stability_results.items(), key=lambda x: x[1].coefficient_of_variation)
        most_stable = sorted_results[:5]
        least_stable = sorted_results[-5:]
        
        # Generate report
        validation_summary = {
            'overall_statistics': {
                'total_layer_quantile_combinations': total_combinations,
                'stable_combinations': stable_combinations,
                'overall_stability_rate': overall_stability_rate,
                'hypothesis_supported': overall_stability_rate >= 0.8
            },
            'quantile_analysis': quantile_analysis,
            'most_stable_combinations': [
                {
                    'layer_id': layer_id,
                    'quantile': quantile,
                    'cv': result.coefficient_of_variation,
                    'mean_threshold': result.mean_threshold,
                    'samples': result.total_samples
                }
                for (layer_id, quantile), result in most_stable
            ],
            'least_stable_combinations': [
                {
                    'layer_id': layer_id,
                    'quantile': quantile,
                    'cv': result.coefficient_of_variation,
                    'mean_threshold': result.mean_threshold,
                    'samples': result.total_samples
                }
                for (layer_id, quantile), result in least_stable
            ]
        }
        
        # Save report
        report_path = self.output_dir / "validation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate text report
        self._generate_text_report(validation_summary)
        
        logger.info(f"Validation report saved: {report_path}")
        return validation_summary
    
    def _generate_text_report(self, validation_summary: Dict[str, Any]):
        """Generate text format report"""
        
        report_lines = [
            "=" * 80,
            "THRESHOLD STABILITY VALIDATION REPORT",
            "=" * 80,
            "",
            "🎯 VALIDATION OBJECTIVE:",
            "Prove that thresholds for the same layer-quantile combination have low variability,",
            "enabling the use of fixed thresholds instead of dynamic calibration.",
            "",
            "📊 OVERALL RESULTS:",
            f"  • Total test combinations: {validation_summary['overall_statistics']['total_layer_quantile_combinations']}",
            f"  • Stable combinations: {validation_summary['overall_statistics']['stable_combinations']}",
            f"  • Overall stability rate: {validation_summary['overall_statistics']['overall_stability_rate']:.1%}",
            f"  • Hypothesis supported: {'✅ YES' if validation_summary['overall_statistics']['hypothesis_supported'] else '❌ NO'}",
            "",
            "📈 QUANTILE ANALYSIS:",
        ]
        
        for quantile, stats in validation_summary['quantile_analysis'].items():
            report_lines.extend([
                f"  {quantile}:",
                f"    - Stability rate: {stats['stability_rate']:.1%} ({stats['stable_layers']}/{stats['total_layers']})",
                f"    - Average CV: {stats['avg_cv']:.4f}",
                f"    - CV range: {stats['min_cv']:.4f} - {stats['max_cv']:.4f}",
                ""
            ])
        
        report_lines.extend([
            "🏆 MOST STABLE COMBINATIONS (Top 5):",
        ])
        
        for combo in validation_summary['most_stable_combinations']:
            report_lines.append(
                f"  Layer {combo['layer_id']} - {combo['quantile']}: "
                f"CV={combo['cv']:.4f}, Threshold={combo['mean_threshold']:.6f}"
            )
        
        report_lines.extend([
            "",
            "⚠️  LEAST STABLE COMBINATIONS (Bottom 5):",
        ])
        
        for combo in validation_summary['least_stable_combinations']:
            report_lines.append(
                f"  Layer {combo['layer_id']} - {combo['quantile']}: "
                f"CV={combo['cv']:.4f}, Threshold={combo['mean_threshold']:.6f}"
            )
        
        report_lines.extend([
            "",
            "💡 CONCLUSION:",
            f"{'✅ Hypothesis SUPPORTED! Most layer-quantile combinations show good stability.' if validation_summary['overall_statistics']['hypothesis_supported'] else '❌ Hypothesis NOT sufficiently supported.'}",
            f"{'Fixed thresholds can be considered as a replacement for dynamic calibration.' if validation_summary['overall_statistics']['hypothesis_supported'] else 'Further optimization needed or continue using dynamic calibration.'}",
            "",
            "=" * 80
        ])
        
        # Save text report
        report_path = self.output_dir / "validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Text report saved: {report_path}")
        
        # Print to console
        print('\n'.join(report_lines))


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Threshold Stability Visualization Script')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Data directory containing threshold data')
    parser.add_argument('--output-dir', type=str, default='./visualization_results',
                       help='Output directory for visualizations')
    parser.add_argument('--session-id', type=str, default=None,
                       help='Session ID to load specific data (optional)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ThresholdStabilityVisualizer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run visualization
    results = visualizer.run_complete_visualization(session_id=args.session_id)
    
    print(f"\n🎉 Visualization completed! Results saved in: {args.output_dir}")
    print(f"Generated files:")
    for file_path in results['visualization_files']:
        if file_path:
            print(f"  📄 {file_path}")
    
    hypothesis_supported = results['validation_summary']['overall_statistics']['hypothesis_supported']
    print(f"\nHypothesis supported: {'✅ YES' if hypothesis_supported else '❌ NO'}")


if __name__ == "__main__":
    main()