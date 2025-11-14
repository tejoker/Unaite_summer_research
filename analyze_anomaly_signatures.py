#!/usr/bin/env python3
"""
Analyze Anomaly Signatures
Generates comprehensive analysis and visualizations of anomaly signatures.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def analyze_signatures(catalog_csv, results_dir, output_dir):
    """
    Analyze anomaly signature catalog and generate visualizations.
    
    Args:
        catalog_csv: Path to anomaly signature catalog
        results_dir: Directory containing individual Tucker-CAM results
        output_dir: Directory to save analysis outputs
    """
    
    print("="*80)
    print("ANALYZING ANOMALY SIGNATURES")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load catalog
    print(f"\n[1/5] Loading signature catalog from: {catalog_csv}")
    df = pd.read_csv(catalog_csv)
    print(f"  ✓ Loaded {len(df)} anomaly signatures")
    
    # Filter successful only
    successful = df[df['status'] == 'success'].copy()
    print(f"  ✓ {len(successful)} successful signatures")
    
    if len(successful) == 0:
        print("  ✗ No successful signatures to analyze!")
        return
    
    # ========================================================================
    # ANALYSIS 1: Anomaly Class Comparison
    # ========================================================================
    
    print(f"\n[2/5] Analyzing by anomaly class...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Signatures by Class', fontsize=16, fontweight='bold')
    
    # 1. Edge count distribution
    ax = axes[0, 0]
    for cls in successful['anomaly_class'].unique():
        data = successful[successful['anomaly_class'] == cls]['total_edges']
        ax.hist(data, alpha=0.6, label=cls, bins=20)
    ax.set_xlabel('Total Edges')
    ax.set_ylabel('Frequency')
    ax.set_title('Edge Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Mean weight by class
    ax = axes[0, 1]
    class_stats = successful.groupby('anomaly_class')['mean_weight'].agg(['mean', 'std'])
    ax.bar(range(len(class_stats)), class_stats['mean'], yerr=class_stats['std'], capsize=5)
    ax.set_xticks(range(len(class_stats)))
    ax.set_xticklabels(class_stats.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Edge Weight')
    ax.set_title('Average Edge Weight by Class')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Max weight by class
    ax = axes[1, 0]
    class_max = successful.groupby('anomaly_class')['max_weight'].agg(['mean', 'std'])
    ax.bar(range(len(class_max)), class_max['mean'], yerr=class_max['std'], capsize=5)
    ax.set_xticks(range(len(class_max)))
    ax.set_xticklabels(class_max.index, rotation=45, ha='right')
    ax.set_ylabel('Max Edge Weight')
    ax.set_title('Maximum Edge Weight by Class')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Number of source variables
    ax = axes[1, 1]
    class_sources = successful.groupby('anomaly_class')['num_source_vars'].agg(['mean', 'std'])
    ax.bar(range(len(class_sources)), class_sources['mean'], yerr=class_sources['std'], capsize=5)
    ax.set_xticks(range(len(class_sources)))
    ax.set_xticklabels(class_sources.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Source Variables')
    ax.set_title('Active Source Variables by Class')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_path / "class_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # ========================================================================
    # ANALYSIS 2: Temporal Patterns
    # ========================================================================
    
    print(f"\n[3/5] Analyzing temporal patterns...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temporal Characteristics', fontsize=16, fontweight='bold')
    
    # 1. Anomaly length vs edge count
    ax = axes[0, 0]
    for cls in successful['anomaly_class'].unique():
        data = successful[successful['anomaly_class'] == cls]
        ax.scatter(data['anomaly_length'], data['total_edges'], alpha=0.6, label=cls, s=50)
    ax.set_xlabel('Anomaly Length (timesteps)')
    ax.set_ylabel('Total Edges')
    ax.set_title('Anomaly Length vs Edge Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Anomaly length vs max weight
    ax = axes[0, 1]
    for cls in successful['anomaly_class'].unique():
        data = successful[successful['anomaly_class'] == cls]
        ax.scatter(data['anomaly_length'], data['max_weight'], alpha=0.6, label=cls, s=50)
    ax.set_xlabel('Anomaly Length (timesteps)')
    ax.set_ylabel('Max Edge Weight')
    ax.set_title('Anomaly Length vs Maximum Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Lag distribution
    ax = axes[1, 0]
    if 'mean_lag' in successful.columns:
        for cls in successful['anomaly_class'].unique():
            data = successful[successful['anomaly_class'] == cls]['mean_lag'].dropna()
            if len(data) > 0:
                ax.hist(data, alpha=0.6, label=cls, bins=15)
        ax.set_xlabel('Mean Lag')
        ax.set_ylabel('Frequency')
        ax.set_title('Average Lag Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Lagged vs contemporaneous edges
    ax = axes[1, 1]
    if 'lagged_edges' in successful.columns and 'contemporaneous_edges' in successful.columns:
        lagged_pct = successful.groupby('anomaly_class')['lagged_edges'].sum()
        contemp_pct = successful.groupby('anomaly_class')['contemporaneous_edges'].sum()
        
        x = np.arange(len(lagged_pct))
        width = 0.35
        
        ax.bar(x - width/2, lagged_pct, width, label='Lagged', alpha=0.8)
        ax.bar(x + width/2, contemp_pct, width, label='Contemporaneous', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(lagged_pct.index, rotation=45, ha='right')
        ax.set_ylabel('Total Edges')
        ax.set_title('Lagged vs Contemporaneous Edges')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_path / "temporal_patterns.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()
    
    # ========================================================================
    # ANALYSIS 3: Spacecraft Comparison
    # ========================================================================
    
    print(f"\n[4/5] Analyzing by spacecraft...")
    
    if 'spacecraft' in successful.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Spacecraft Comparison', fontsize=16, fontweight='bold')
        
        # 1. Edge count by spacecraft
        ax = axes[0]
        spacecraft_stats = successful.groupby('spacecraft')['total_edges'].agg(['mean', 'std', 'count'])
        ax.bar(range(len(spacecraft_stats)), spacecraft_stats['mean'], 
               yerr=spacecraft_stats['std'], capsize=5, alpha=0.8)
        ax.set_xticks(range(len(spacecraft_stats)))
        ax.set_xticklabels([f"{sc}\n(n={spacecraft_stats.loc[sc, 'count']})" 
                            for sc in spacecraft_stats.index], rotation=0)
        ax.set_ylabel('Mean Total Edges')
        ax.set_title('Average Edge Count by Spacecraft')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Max weight by spacecraft
        ax = axes[1]
        spacecraft_max = successful.groupby('spacecraft')['max_weight'].agg(['mean', 'std'])
        ax.bar(range(len(spacecraft_max)), spacecraft_max['mean'], 
               yerr=spacecraft_max['std'], capsize=5, alpha=0.8)
        ax.set_xticks(range(len(spacecraft_max)))
        ax.set_xticklabels(spacecraft_max.index, rotation=0)
        ax.set_ylabel('Mean Max Weight')
        ax.set_title('Average Maximum Weight by Spacecraft')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_path / "spacecraft_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    # ========================================================================
    # ANALYSIS 4: Baseline Comparison (if available)
    # ========================================================================
    
    print(f"\n[5/5] Analyzing baseline comparison...")
    
    if 'new_edges' in successful.columns and 'removed_edges' in successful.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Causal Graph Changes vs Baseline', fontsize=16, fontweight='bold')
        
        # 1. New edges by class
        ax = axes[0, 0]
        for cls in successful['anomaly_class'].unique():
            data = successful[successful['anomaly_class'] == cls]['new_edges']
            ax.hist(data, alpha=0.6, label=cls, bins=15)
        ax.set_xlabel('New Edges')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of New Edges')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Removed edges by class
        ax = axes[0, 1]
        for cls in successful['anomaly_class'].unique():
            data = successful[successful['anomaly_class'] == cls]['removed_edges']
            ax.hist(data, alpha=0.6, label=cls, bins=15)
        ax.set_xlabel('Removed Edges')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Removed Edges')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Net change (new - removed)
        ax = axes[1, 0]
        successful['net_edge_change'] = successful['new_edges'] - successful['removed_edges']
        for cls in successful['anomaly_class'].unique():
            data = successful[successful['anomaly_class'] == cls]['net_edge_change']
            ax.hist(data, alpha=0.6, label=cls, bins=15)
        ax.set_xlabel('Net Edge Change')
        ax.set_ylabel('Frequency')
        ax.set_title('Net Change in Edges (New - Removed)')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary by class
        ax = axes[1, 1]
        class_changes = successful.groupby('anomaly_class')[['new_edges', 'removed_edges', 'common_edges']].mean()
        
        x = np.arange(len(class_changes))
        width = 0.25
        
        ax.bar(x - width, class_changes['new_edges'], width, label='New', alpha=0.8)
        ax.bar(x, class_changes['removed_edges'], width, label='Removed', alpha=0.8)
        ax.bar(x + width, class_changes['common_edges'], width, label='Common', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(class_changes.index, rotation=45, ha='right')
        ax.set_ylabel('Average Edge Count')
        ax.set_title('Edge Changes by Class')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_path / "baseline_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file}")
        plt.close()
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    
    print(f"\nGenerating summary statistics...")
    
    summary = {
        'Total Anomalies': len(successful),
        'Anomaly Classes': successful['anomaly_class'].nunique(),
        'Spacecraft': successful['spacecraft'].nunique() if 'spacecraft' in successful.columns else 'N/A',
        'Mean Edges per Anomaly': successful['total_edges'].mean(),
        'Std Edges per Anomaly': successful['total_edges'].std(),
        'Min Edges': successful['total_edges'].min(),
        'Max Edges': successful['total_edges'].max(),
        'Mean Max Weight': successful['max_weight'].mean(),
        'Std Max Weight': successful['max_weight'].std(),
    }
    
    if 'new_edges' in successful.columns:
        summary['Mean New Edges'] = successful['new_edges'].mean()
        summary['Mean Removed Edges'] = successful['removed_edges'].mean()
        summary['Mean Net Change'] = successful['net_edge_change'].mean()
    
    # Save summary
    summary_file = output_path / "summary_statistics.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANOMALY SIGNATURE ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("BY ANOMALY CLASS\n")
        f.write("="*80 + "\n\n")
        
        for cls in successful['anomaly_class'].unique():
            cls_data = successful[successful['anomaly_class'] == cls]
            f.write(f"\n{cls.upper()} (n={len(cls_data)})\n")
            f.write("-"*40 + "\n")
            f.write(f"  Mean edges: {cls_data['total_edges'].mean():.1f} ± {cls_data['total_edges'].std():.1f}\n")
            f.write(f"  Mean weight: {cls_data['mean_weight'].mean():.4f} ± {cls_data['mean_weight'].std():.4f}\n")
            f.write(f"  Max weight: {cls_data['max_weight'].mean():.4f} ± {cls_data['max_weight'].std():.4f}\n")
            if 'new_edges' in cls_data.columns:
                f.write(f"  New edges: {cls_data['new_edges'].mean():.1f} ± {cls_data['new_edges'].std():.1f}\n")
                f.write(f"  Removed edges: {cls_data['removed_edges'].mean():.1f} ± {cls_data['removed_edges'].std():.1f}\n")
    
    print(f"  ✓ Saved: {summary_file}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  1. class_comparison.png - Anomaly signatures by class")
    print(f"  2. temporal_patterns.png - Temporal characteristics")
    print(f"  3. spacecraft_comparison.png - Spacecraft comparison")
    if 'new_edges' in successful.columns:
        print(f"  4. baseline_comparison.png - Causal graph changes")
    print(f"  5. summary_statistics.txt - Numerical summary")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze anomaly signatures")
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to anomaly signature catalog CSV"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing Tucker-CAM results"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for analysis"
    )
    
    args = parser.parse_args()
    
    analyze_signatures(
        catalog_csv=args.catalog,
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Done! Check {args.output_dir}/ for analysis results")
