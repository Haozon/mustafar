#!/usr/bin/env python3
"""
Visualize DiffKV sparsity thresholds with English labels
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def visualize_thresholds(thresholds_file):
    """Create comprehensive threshold analysis plots"""
    
    # Read thresholds file
    with open(thresholds_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    layers = []
    alpha_h_values = []
    alpha_mh_values = []
    alpha_m_values = []
    alpha_ml_values = []
    
    for layer_name, thresholds in data['thresholds'].items():
        layer_num = int(layer_name.split('_')[1])
        layers.append(layer_num)
        alpha_h_values.append(thresholds['alpha_h'])
        alpha_mh_values.append(thresholds['alpha_mh'])
        alpha_m_values.append(thresholds['alpha_m'])
        alpha_ml_values.append(thresholds['alpha_ml'])
    
    # Sort by layer index
    sorted_indices = np.argsort(layers)
    layers = np.array(layers)[sorted_indices]
    alpha_h_values = np.array(alpha_h_values)[sorted_indices]
    alpha_mh_values = np.array(alpha_mh_values)[sorted_indices]
    alpha_m_values = np.array(alpha_m_values)[sorted_indices]
    alpha_ml_values = np.array(alpha_ml_values)[sorted_indices]
    
    # Plot 1: All thresholds comparison (log scale)
    plt.figure(figsize=(12, 8))
    plt.semilogy(layers, alpha_h_values, 'ro-', label='α_h (0% sparsity)', linewidth=2, markersize=4)
    plt.semilogy(layers, alpha_mh_values, 'go-', label='α_mh (50% sparsity)', linewidth=2, markersize=4)
    plt.semilogy(layers, alpha_m_values, 'bo-', label='α_m (70% sparsity)', linewidth=2, markersize=4)
    plt.semilogy(layers, alpha_ml_values, 'mo-', label='α_ml (90% sparsity)', linewidth=2, markersize=4)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Threshold Value (Log Scale)')
    plt.title('DiffKV Sparsity Thresholds Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('test_calibration_results/thresholds_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Dynamic range analysis
    dynamic_ranges = [alpha_h_values[i] / alpha_ml_values[i] for i in range(len(layers))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, dynamic_ranges, 'ko-', linewidth=2, markersize=4)
    plt.xlabel('Layer Index')
    plt.ylabel('Dynamic Range (α_h / α_ml)')
    plt.title('Threshold Dynamic Range Analysis')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    mean_range = np.mean(dynamic_ranges)
    plt.axhline(y=mean_range, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_range:.1f}x')
    plt.legend()
    
    plt.savefig('test_calibration_results/dynamic_range.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Individual layer threshold quality
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 3a: Threshold separation quality
    separations = []
    layer_labels = []
    for i, layer in enumerate(layers):
        # Calculate threshold separations (gaps between adjacent thresholds)
        gaps = [
            alpha_h_values[i] - alpha_mh_values[i],
            alpha_mh_values[i] - alpha_m_values[i], 
            alpha_m_values[i] - alpha_ml_values[i]
        ]
        # Calculate relative separations (gap / higher threshold)
        rel_gaps = [
            gaps[0] / alpha_h_values[i],
            gaps[1] / alpha_mh_values[i],
            gaps[2] / alpha_m_values[i]
        ]
        separations.append(np.mean(rel_gaps))  # Average relative separation
        layer_labels.append(f'L{layer}')
    
    ax1.bar(range(len(layers)), separations, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Relative Threshold Separation')
    ax1.set_title('Threshold Separation Quality by Layer')
    ax1.set_xticks(range(0, len(layers), 4))
    ax1.set_xticklabels([f'L{layers[i]}' for i in range(0, len(layers), 4)])
    ax1.grid(True, alpha=0.3)
    
    # Add quality threshold line
    ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Good Separation (>0.1)')
    ax1.legend()
    
    # Plot 3b: Dynamic range by layer
    ax2.bar(range(len(layers)), dynamic_ranges, alpha=0.7, color='coral')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Dynamic Range (α_h / α_ml)')
    ax2.set_title('Threshold Dynamic Range by Layer')
    ax2.set_xticks(range(0, len(layers), 4))
    ax2.set_xticklabels([f'L{layers[i]}' for i in range(0, len(layers), 4)])
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add quality range
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.7, label='Good Range (>50x)')
    ax2.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='Very High (>500x)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('test_calibration_results/layer_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("DiffKV Threshold Calibration Summary")
    print("=" * 60)
    print(f"Model: {data['model_path'].split('/')[-1]}")
    print(f"Layers: {len(layers)}")
    print(f"Granularity: {data['granularity']}")
    print(f"Target sparsity: {data['target_sparsity']}")
    
    target_distribution = np.array(data['target_distribution'])
    sparsity_levels = np.array([0.0, 0.5, 0.7, 0.9, 1.0])
    calculated_avg_sparsity = np.sum(target_distribution * sparsity_levels)
    
    print(f"\nDistribution Analysis:")
    print(f"  Target distribution: {target_distribution}")
    print(f"  Calculated avg sparsity: {calculated_avg_sparsity:.3f}")
    print(f"  Match with target: {'✓' if abs(calculated_avg_sparsity - data['target_sparsity']) < 0.01 else '✗'}")
    
    print(f"\nThreshold Quality by Layer:")
    print(f"  Dynamic range: {min(dynamic_ranges):.1f}x - {max(dynamic_ranges):.1f}x (avg: {np.mean(dynamic_ranges):.1f}x)")
    
    # Analyze threshold quality per layer
    good_separation_count = 0
    good_range_count = 0
    
    for i, layer in enumerate(layers):
        # Calculate relative separations
        gaps = [
            alpha_h_values[i] - alpha_mh_values[i],
            alpha_mh_values[i] - alpha_m_values[i], 
            alpha_m_values[i] - alpha_ml_values[i]
        ]
        rel_gaps = [
            gaps[0] / alpha_h_values[i],
            gaps[1] / alpha_mh_values[i],
            gaps[2] / alpha_m_values[i]
        ]
        avg_separation = np.mean(rel_gaps)
        
        if avg_separation > 0.1:
            good_separation_count += 1
        if dynamic_ranges[i] > 50:
            good_range_count += 1
    
    print(f"\nPer-Layer Quality Assessment:")
    print(f"  Layers with good threshold separation (>10%): {good_separation_count}/{len(layers)} ({100*good_separation_count/len(layers):.1f}%)")
    print(f"  Layers with good dynamic range (>50x): {good_range_count}/{len(layers)} ({100*good_range_count/len(layers):.1f}%)")
    
    # Overall quality assessment
    overall_quality = "Excellent" if (good_separation_count >= len(layers)*0.8 and good_range_count >= len(layers)*0.8) else \
                     "Good" if (good_separation_count >= len(layers)*0.6 and good_range_count >= len(layers)*0.6) else \
                     "Acceptable"
    
    print(f"\n✅ Overall Calibration Quality: {overall_quality}")
    print(f"   Each layer has been calibrated based on its own attention patterns")

if __name__ == "__main__":
    print("Visualizing DiffKV sparsity thresholds...")
    try:
        visualize_thresholds('test_calibration_results/thresholds.json')
        print("\n📊 Plots saved:")
        print("  - thresholds_overview.png")
        print("  - dynamic_range.png") 
        print("  - layer_quality_analysis.png")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()