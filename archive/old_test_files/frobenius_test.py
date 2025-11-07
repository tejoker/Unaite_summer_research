#!/usr/bin/env python3
"""
frobenius_test_weights.py - Compare two weights CSV files using Frobenius norm distance

This script computes the Frobenius norm between two weights CSV files by comparing
only the weight values for matching (window_idx, lag, i, j) combinations.

Expected CSV format: window_idx,lag,i,j,weight

Usage:
    python frobenius_test_weights.py --file1 weights1.csv --file2 weights2.csv
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_weights_csv(file_path):
    """
    Load weights CSV file with format: window_idx,lag,i,j,weight
    
    Args:
        file_path: Path to weights CSV file
    
    Returns:
        pandas DataFrame with weights data
    """
    df = pd.read_csv(file_path)
    
    # Validate expected columns
    expected_cols = ['window_idx', 'lag', 'i', 'j', 'weight']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"CSV must contain columns: {expected_cols}. Found: {list(df.columns)}")
    
    # Ensure proper data types
    df['window_idx'] = df['window_idx'].astype(int)
    df['lag'] = df['lag'].astype(int) 
    df['i'] = df['i'].astype(int)
    df['j'] = df['j'].astype(int)
    df['weight'] = df['weight'].astype(float)
    
    print(f"  Loaded {len(df)} weight entries")
    print(f"  Windows: {df['window_idx'].min()}-{df['window_idx'].max()}")
    print(f"  Lags: {df['lag'].min()}-{df['lag'].max()}")
    print(f"  Matrix dimensions: i={df['i'].min()}-{df['i'].max()}, j={df['j'].min()}-{df['j'].max()}")
    
    return df


def compute_weights_frobenius_distance(df1, df2):
    """
    Compute Frobenius norm distance between two weights DataFrames.
    Only compares weights for matching (window_idx, lag, i, j) combinations.
    
    Args:
        df1: First weights DataFrame  
        df2: Second weights DataFrame
    
    Returns:
        dict with Frobenius distance metrics
    """
    print("  Merging weights on matching (window_idx, lag, i, j) combinations...")
    
    # Merge on matching indices
    merged = pd.merge(
        df1, df2, 
        on=['window_idx', 'lag', 'i', 'j'], 
        suffixes=('_1', '_2'),
        how='inner'
    )
    
    if len(merged) == 0:
        raise ValueError("No matching (window_idx, lag, i, j) combinations found between files")
    
    print(f"  Found {len(merged)} matching weight entries out of {len(df1)} and {len(df2)}")
    
    # Calculate weight differences
    weight_diff = merged['weight_1'] - merged['weight_2']
    
    # Frobenius norm is sqrt of sum of squared differences
    frobenius_distance = np.sqrt(np.sum(weight_diff ** 2))
    
    # Normalized Frobenius distance (divide by Frobenius norm of first weights)
    norm_weights1 = np.sqrt(np.sum(merged['weight_1'] ** 2))
    normalized_distance = frobenius_distance / norm_weights1 if norm_weights1 > 0 else float('inf')
    
    # Relative Frobenius distance (as percentage)
    relative_distance = normalized_distance * 100
    
    # Statistics on weight differences
    abs_diff = np.abs(weight_diff)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff) 
    std_diff = np.std(abs_diff)
    
    # Find location of maximum difference
    max_idx = np.argmax(abs_diff)
    max_diff_row = merged.iloc[max_idx]
    max_diff_location = {
        'window_idx': int(max_diff_row['window_idx']),
        'lag': int(max_diff_row['lag']),
        'i': int(max_diff_row['i']),
        'j': int(max_diff_row['j']),
        'weight_1': float(max_diff_row['weight_1']),
        'weight_2': float(max_diff_row['weight_2']),
        'difference': float(weight_diff.iloc[max_idx])
    }
    
    # Find top 5 largest differences
    top5_indices = abs_diff.nlargest(5).index
    top5_differences = []
    for idx in top5_indices:
        row = merged.iloc[idx]
        top5_differences.append({
            'window_idx': int(row['window_idx']),
            'lag': int(row['lag']),
            'i': int(row['i']),
            'j': int(row['j']),
            'weight_1': float(row['weight_1']),
            'weight_2': float(row['weight_2']),
            'difference': float(weight_diff.iloc[idx]),
            'abs_difference': float(abs_diff.iloc[idx])
        })
    
    # Coverage statistics
    total_possible_1 = len(df1)
    total_possible_2 = len(df2)
    matched_count = len(merged)
    coverage_1 = (matched_count / total_possible_1) * 100 if total_possible_1 > 0 else 0
    coverage_2 = (matched_count / total_possible_2) * 100 if total_possible_2 > 0 else 0
    
    return {
        'frobenius_distance': float(frobenius_distance),
        'normalized_distance': float(normalized_distance),
        'relative_distance_percent': float(relative_distance),
        'max_element_difference': float(max_diff),
        'max_difference_location': max_diff_location,
        'top5_differences': top5_differences,
        'mean_element_difference': float(mean_diff),
        'std_element_difference': float(std_diff),
        'matched_entries': int(matched_count),
        'total_entries_file1': int(total_possible_1),
        'total_entries_file2': int(total_possible_2),
        'coverage_percent_file1': float(coverage_1),
        'coverage_percent_file2': float(coverage_2),
        'weight_difference_stats': {
            'min': float(np.min(weight_diff)),
            'max': float(np.max(weight_diff)), 
            'mean': float(np.mean(weight_diff)),
            'std': float(np.std(weight_diff)),
            'median': float(np.median(weight_diff))
        }
    }


def create_weights_comparison_plots(merged_df, output_dir, file1_name, file2_name):
    """
    Create visualization plots comparing the two weight datasets.
    """
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Weights Comparison: {file1_name} vs {file2_name}', fontsize=14)
    
    # Plot 1: Scatter plot of weights
    axes[0, 0].scatter(merged_df['weight_1'], merged_df['weight_2'], alpha=0.6, s=1)
    axes[0, 0].plot([merged_df['weight_1'].min(), merged_df['weight_1'].max()], 
                    [merged_df['weight_1'].min(), merged_df['weight_1'].max()], 
                    'r--', alpha=0.8, label='Perfect match')
    axes[0, 0].set_xlabel(f'Weights from {file1_name}')
    axes[0, 0].set_ylabel(f'Weights from {file2_name}')
    axes[0, 0].set_title('Weight Correlation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of weight differences
    weight_diff = merged_df['weight_1'] - merged_df['weight_2']
    axes[0, 1].hist(weight_diff, bins=50, alpha=0.7, density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.8, label='Zero difference')
    axes[0, 1].set_xlabel('Weight Difference (File1 - File2)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution of Weight Differences')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Differences by lag
    if merged_df['lag'].nunique() > 1:
        lag_diffs = merged_df.groupby('lag')['weight_1'].apply(lambda x: x - merged_df.loc[x.index, 'weight_2']).reset_index()
        lag_means = merged_df.groupby('lag').apply(lambda x: np.abs(x['weight_1'] - x['weight_2']).mean()).reset_index()
        lag_means.columns = ['lag', 'mean_abs_diff']
        axes[1, 0].bar(lag_means['lag'], lag_means['mean_abs_diff'])
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('Mean Absolute Difference')
        axes[1, 0].set_title('Mean Absolute Difference by Lag')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Only one lag present', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Differences by Lag (N/A)')
    
    # Plot 4: Differences by window (sample of windows)
    if merged_df['window_idx'].nunique() > 1:
        # Sample up to 50 windows for readability
        unique_windows = sorted(merged_df['window_idx'].unique())
        if len(unique_windows) > 50:
            sampled_windows = unique_windows[::len(unique_windows)//50]
        else:
            sampled_windows = unique_windows
            
        window_diffs = merged_df[merged_df['window_idx'].isin(sampled_windows)].groupby('window_idx').apply(
            lambda x: np.abs(x['weight_1'] - x['weight_2']).mean()
        ).reset_index()
        window_diffs.columns = ['window_idx', 'mean_abs_diff']
        axes[1, 1].plot(window_diffs['window_idx'], window_diffs['mean_abs_diff'], 'o-', markersize=2)
        axes[1, 1].set_xlabel('Window Index')
        axes[1, 1].set_ylabel('Mean Absolute Difference')
        axes[1, 1].set_title('Mean Absolute Difference by Window')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Only one window present', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Differences by Window (N/A)')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f'weights_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare two weights CSV files using Frobenius norm distance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two weights CSV files
  python frobenius_test_weights.py --file1 weights_golden.csv --file2 weights_anomaly.csv
  
  # Compare with custom output directory and plots
  python frobenius_test_weights.py --file1 weights1.csv --file2 weights2.csv --output-dir custom_results/ --save-plot
        """
    )
    
    parser.add_argument('--file1', required=True, help='Path to first weights CSV file')
    parser.add_argument('--file2', required=True, help='Path to second weights CSV file') 
    parser.add_argument('--output-dir', default='results/Test', help='Output directory for results')
    parser.add_argument('--save-plot', action='store_true', help='Save comparison plots')
    
    args = parser.parse_args()
    
    # Validate input files
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    
    if not file1_path.exists():
        print(f"Error: File1 not found: {file1_path}")
        return 1
        
    if not file2_path.exists():
        print(f"Error: File2 not found: {file2_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔬 Frobenius Distance Test for Weights")
    print(f"📄 File 1: {file1_path}")
    print(f"📄 File 2: {file2_path}")
    print(f"📁 Output: {output_dir}")
    
    try:
        # Load weights DataFrames
        print("\n📊 Loading weights files...")
        df1 = load_weights_csv(file1_path)
        df2 = load_weights_csv(file2_path)
        
        # Compute Frobenius distance
        print("\n🧮 Computing Frobenius distance for matching entries...")
        results = compute_weights_frobenius_distance(df1, df2)
        
        # Display results
        print(f"\n✅ Results:")
        print(f"  Matched Entries: {results['matched_entries']}")
        print(f"  Coverage File1: {results['coverage_percent_file1']:.1f}%")
        print(f"  Coverage File2: {results['coverage_percent_file2']:.1f}%")
        print(f"  Frobenius Distance: {results['frobenius_distance']:.6f}")
        print(f"  Normalized Distance: {results['normalized_distance']:.6f}")
        print(f"  Relative Distance: {results['relative_distance_percent']:.2f}%")
        print(f"  Max Element Difference: {results['max_element_difference']:.6f}")
        print(f"  Mean Element Difference: {results['mean_element_difference']:.6f}")
        
        # Create output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file1_name = file1_path.stem
        file2_name = file2_path.stem
        
        # Save JSON results
        results['metadata'] = {
            'file1': str(file1_path),
            'file2': str(file2_path),
            'file1_name': file1_name,
            'file2_name': file2_name,
            'timestamp': timestamp
        }
        
        json_path = output_dir / f'frobenius_weights_{file1_name}_vs_{file2_name}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  JSON: {json_path}")
        
        # Create plots if requested
        if args.save_plot:
            print(f"\n📊 Creating comparison plots...")
            # Merge data for plotting
            merged = pd.merge(df1, df2, on=['window_idx', 'lag', 'i', 'j'], suffixes=('_1', '_2'), how='inner')
            
            plot_path = create_weights_comparison_plots(
                merged, output_dir, file1_name, file2_name
            )
            print(f"  Plot: {plot_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())