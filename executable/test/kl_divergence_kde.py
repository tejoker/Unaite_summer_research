#!/usr/bin/env python3
"""
kl_divergence_kde.py - Compare two CSV files using KL divergence with KDE estimation

This script computes the Kullback-Leibler (KL) divergence between distributions
of two CSV files using Kernel Density Estimation (KDE) for probability density estimation.

Usage:
    python kl_divergence_kde.py --file1 data/file1.csv --file2 data/file2.csv
    python kl_divergence_kde.py --file1 data/Golden/chunking/output_of_the_1th_chunk.csv --file2 data/Anomaly/output_of_the_1th_chunk_spike.csv
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.integrate import quad
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')


def load_csv_data(file_path, exclude_datetime=True):
    """
    Load CSV file and extract numerical data for KL divergence calculation.
    
    Args:
        file_path: Path to CSV file
        exclude_datetime: Whether to exclude datetime columns
    
    Returns:
        DataFrame with numerical data and column names
    """
    df = pd.read_csv(file_path)
    
    # Convert datetime columns if present
    if exclude_datetime:
        numeric_cols = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                print(f"  Skipping non-numeric column: {col}")
                continue
        
        if numeric_cols:
            df_numeric = df[numeric_cols]
        else:
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
    else:
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns with all NaN values
    df_numeric = df_numeric.dropna(axis=1, how='all')
    
    # Fill NaN values with column mean
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    return df_numeric


def estimate_kde_density(data, bandwidth='scott', kernel='gaussian'):
    """
    Estimate probability density using Kernel Density Estimation.
    
    Args:
        data: 1D array of data points
        bandwidth: Bandwidth for KDE ('scott', 'silverman', or float)
        kernel: Kernel type for KDE
    
    Returns:
        KDE object and evaluation points
    """
    # Remove any remaining NaN or infinite values
    data_clean = data[np.isfinite(data)]
    
    if len(data_clean) == 0:
        raise ValueError("No valid data points found")
    
    # Determine bandwidth
    if bandwidth == 'scott':
        bw = len(data_clean) ** (-1. / 5.)
    elif bandwidth == 'silverman':
        bw = (len(data_clean) * 3. / 4.) ** (-1. / 5.)
    else:
        bw = float(bandwidth)
    
    # Fit KDE
    kde = KernelDensity(kernel=kernel, bandwidth=bw)
    kde.fit(data_clean.reshape(-1, 1))
    
    # Create evaluation points
    data_min, data_max = np.min(data_clean), np.max(data_clean)
    data_range = data_max - data_min
    x_eval = np.linspace(data_min - 0.1 * data_range, 
                        data_max + 0.1 * data_range, 1000)
    
    return kde, x_eval, data_clean


def compute_kl_divergence_kde(data1, data2, bandwidth='scott', epsilon=1e-10):
    """
    Compute KL divergence between two datasets using KDE.
    
    Args:
        data1: First dataset (1D array)
        data2: Second dataset (1D array)  
        bandwidth: Bandwidth for KDE
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence value and additional metrics
    """
    # Estimate KDE for both datasets
    kde1, x_eval1, clean_data1 = estimate_kde_density(data1, bandwidth)
    kde2, x_eval2, clean_data2 = estimate_kde_density(data2, bandwidth)
    
    # Use common evaluation points
    x_min = min(np.min(clean_data1), np.min(clean_data2))
    x_max = max(np.max(clean_data1), np.max(clean_data2))
    x_range = x_max - x_min
    x_common = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
    
    # Evaluate densities
    log_density1 = kde1.score_samples(x_common.reshape(-1, 1))
    log_density2 = kde2.score_samples(x_common.reshape(-1, 1))
    
    density1 = np.exp(log_density1)
    density2 = np.exp(log_density2)
    
    # Add epsilon to avoid log(0)
    density1 = density1 + epsilon
    density2 = density2 + epsilon
    
    # Normalize to ensure they integrate to 1
    density1 = density1 / np.trapz(density1, x_common)
    density2 = density2 / np.trapz(density2, x_common)
    
    # Compute KL divergence: KL(P||Q) = integral(P * log(P/Q))
    ratio = density1 / density2
    log_ratio = np.log(ratio)
    kl_integrand = density1 * log_ratio
    
    # Numerical integration
    kl_divergence = np.trapz(kl_integrand, x_common)
    
    # Compute symmetric KL divergence (JS divergence approximation)
    kl_reverse = np.trapz(density2 * np.log(density2 / density1), x_common)
    symmetric_kl = (kl_divergence + kl_reverse) / 2
    
    # Additional statistics
    stats_dict = {
        'kl_divergence_forward': float(kl_divergence),
        'kl_divergence_reverse': float(kl_reverse), 
        'symmetric_kl_divergence': float(symmetric_kl),
        'jensen_shannon_divergence': float(symmetric_kl / 2),  # Approximation
        'data1_stats': {
            'mean': float(np.mean(clean_data1)),
            'std': float(np.std(clean_data1)),
            'min': float(np.min(clean_data1)),
            'max': float(np.max(clean_data1)),
            'n_points': len(clean_data1)
        },
        'data2_stats': {
            'mean': float(np.mean(clean_data2)),
            'std': float(np.std(clean_data2)), 
            'min': float(np.min(clean_data2)),
            'max': float(np.max(clean_data2)),
            'n_points': len(clean_data2)
        },
        'evaluation_points': x_common,
        'density1': density1,
        'density2': density2
    }
    
    return stats_dict


def compute_column_kl_divergences(df1, df2, bandwidth='scott'):
    """
    Compute KL divergence for each column pair between two dataframes.
    """
    results = {}
    common_columns = list(set(df1.columns) & set(df2.columns))
    
    if not common_columns:
        raise ValueError("No common columns found between the two files")
    
    print(f"  Computing KL divergence for {len(common_columns)} common columns...")
    
    for col in common_columns:
        try:
            col_data1 = df1[col].dropna().values
            col_data2 = df2[col].dropna().values
            
            if len(col_data1) < 10 or len(col_data2) < 10:
                print(f"    Warning: Column '{col}' has insufficient data points")
                continue
                
            kl_result = compute_kl_divergence_kde(col_data1, col_data2, bandwidth)
            results[col] = kl_result
            
            print(f"    {col}: KL={kl_result['kl_divergence_forward']:.6f}")
            
        except Exception as e:
            print(f"    Error computing KL for column '{col}': {e}")
            continue
    
    return results


def create_kl_visualization(results, output_dir, file1_name, file2_name):
    """
    Create visualization plots for KL divergence analysis.
    """
    n_cols = len(results)
    if n_cols == 0:
        return None
        
    # Determine subplot layout
    n_rows = min(3, n_cols)  # Limit to 3 rows max
    n_plot_cols = min(3, (n_cols + n_rows - 1) // n_rows)
    
    fig, axes = plt.subplots(n_rows, n_plot_cols, figsize=(5*n_plot_cols, 4*n_rows))
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_plot_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'KL Divergence Analysis\n{file1_name} vs {file2_name}', fontsize=14)
    
    plot_idx = 0
    for col_name, kl_data in results.items():
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        # Plot KDE densities
        x_eval = kl_data['evaluation_points']
        density1 = kl_data['density1']
        density2 = kl_data['density2']
        
        ax.plot(x_eval, density1, label=f'{file1_name}', alpha=0.7, linewidth=2)
        ax.plot(x_eval, density2, label=f'{file2_name}', alpha=0.7, linewidth=2)
        ax.fill_between(x_eval, density1, alpha=0.3)
        ax.fill_between(x_eval, density2, alpha=0.3)
        
        ax.set_title(f'{col_name}\nKL={kl_data["kl_divergence_forward"]:.4f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f'kl_divergence_kde_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare two CSV files using KL divergence with KDE estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two CSV files
  python kl_divergence_kde.py --file1 data/Golden/chunking/output_of_the_1th_chunk.csv --file2 data/Anomaly/output_of_the_1th_chunk_spike.csv
  
  # Compare with custom bandwidth
  python kl_divergence_kde.py --file1 data/file1.csv --file2 data/file2.csv --bandwidth 0.1
  
  # Save visualization plots
  python kl_divergence_kde.py --file1 data/file1.csv --file2 data/file2.csv --save-plot
        """
    )
    
    parser.add_argument('--file1', required=True, help='Path to first CSV file')
    parser.add_argument('--file2', required=True, help='Path to second CSV file') 
    parser.add_argument('--output-dir', default='results/Test', help='Output directory for results')
    parser.add_argument('--bandwidth', default='scott', help='KDE bandwidth (scott, silverman, or float)')
    parser.add_argument('--save-plot', action='store_true', help='Save visualization plots')
    
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
    
    print(f"🎯 KL Divergence with KDE Test")
    print(f"📄 File 1: {file1_path}")
    print(f"📄 File 2: {file2_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🔧 Bandwidth: {args.bandwidth}")
    
    try:
        # Load data
        print(f"\n📊 Loading data...")
        df1 = load_csv_data(file1_path)
        df2 = load_csv_data(file2_path)
        
        print(f"  File 1 shape: {df1.shape}")
        print(f"  File 2 shape: {df2.shape}")
        
        # Compute KL divergences
        print(f"\n🧮 Computing KL divergences...")
        kl_results = compute_column_kl_divergences(df1, df2, args.bandwidth)
        
        if not kl_results:
            print("❌ No KL divergences could be computed")
            return 1
        
        # Compute summary statistics
        kl_values = [result['kl_divergence_forward'] for result in kl_results.values()]
        symmetric_kl_values = [result['symmetric_kl_divergence'] for result in kl_results.values()]
        
        summary = {
            'mean_kl_divergence': float(np.mean(kl_values)),
            'std_kl_divergence': float(np.std(kl_values)),
            'max_kl_divergence': float(np.max(kl_values)),
            'min_kl_divergence': float(np.min(kl_values)),
            'mean_symmetric_kl': float(np.mean(symmetric_kl_values)),
            'total_columns_analyzed': len(kl_results),
            'column_results': {}
        }
        
        # Display results
        print(f"\n✅ Results Summary:")
        print(f"  Columns analyzed: {len(kl_results)}")
        print(f"  Mean KL divergence: {summary['mean_kl_divergence']:.6f}")
        print(f"  Max KL divergence: {summary['max_kl_divergence']:.6f}")
        print(f"  Mean Symmetric KL: {summary['mean_symmetric_kl']:.6f}")
        
        # Store detailed results (without large arrays for JSON)
        for col, result in kl_results.items():
            summary['column_results'][col] = {
                'kl_divergence_forward': result['kl_divergence_forward'],
                'kl_divergence_reverse': result['kl_divergence_reverse'],
                'symmetric_kl_divergence': result['symmetric_kl_divergence'],
                'jensen_shannon_divergence': result['jensen_shannon_divergence'],
                'data1_stats': result['data1_stats'],
                'data2_stats': result['data2_stats']
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file1_name = file1_path.stem
        file2_name = file2_path.stem
        
        summary['metadata'] = {
            'file1': str(file1_path),
            'file2': str(file2_path),
            'file1_name': file1_name,
            'file2_name': file2_name,
            'timestamp': timestamp,
            'bandwidth': args.bandwidth
        }
        
        json_path = output_dir / f'kl_divergence_kde_{file1_name}_vs_{file2_name}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  JSON: {json_path}")
        
        # Create visualization if requested
        if args.save_plot:
            print(f"\n📊 Creating visualization...")
            plot_path = create_kl_visualization(kl_results, output_dir, file1_name, file2_name)
            if plot_path:
                print(f"  Plot: {plot_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())