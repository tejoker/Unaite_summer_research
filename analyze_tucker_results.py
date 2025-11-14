#!/usr/bin/env python3
"""
Comprehensive Analysis of Tucker-CAM Option D Results

Analyzes the learned causal graphs from rolling window analysis:
1. Edge statistics per window
2. Weight distribution analysis
3. Top causal relationships
4. Temporal stability
5. Network structure metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_results(weights_file):
    """Load Tucker-CAM results"""
    print(f"Loading results from {weights_file}...")
    df = pd.read_csv(weights_file)
    print(f"  Loaded {len(df):,} edges across {df['window_idx'].nunique()} windows")
    return df

def analyze_edge_statistics(df):
    """Analyze edge counts and weights per window"""
    print("\n" + "="*80)
    print("EDGE STATISTICS PER WINDOW")
    print("="*80)
    
    window_stats = df.groupby('window_idx').agg({
        'weight': ['count', 'mean', 'std', 'min', 'max'],
    }).round(3)
    
    window_stats.columns = ['Edge Count', 'Mean Weight', 'Std Weight', 'Min Weight', 'Max Weight']
    
    print("\nSummary across all windows:")
    print(window_stats.describe().round(3))
    
    # Check stability
    edge_counts = window_stats['Edge Count']
    print(f"\nEdge count stability:")
    print(f"  Mean: {edge_counts.mean():.1f}")
    print(f"  Std: {edge_counts.std():.1f}")
    print(f"  Min: {edge_counts.min():.0f}")
    print(f"  Max: {edge_counts.max():.0f}")
    print(f"  CV: {edge_counts.std() / edge_counts.mean() * 100:.2f}%")
    
    # Plot edge counts over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Edge counts
    axes[0, 0].plot(window_stats.index, window_stats['Edge Count'], linewidth=1)
    axes[0, 0].set_xlabel('Window Index')
    axes[0, 0].set_ylabel('Edge Count')
    axes[0, 0].set_title('Edge Count per Window')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean weight
    axes[0, 1].plot(window_stats.index, window_stats['Mean Weight'], linewidth=1, color='orange')
    axes[0, 1].set_xlabel('Window Index')
    axes[0, 1].set_ylabel('Mean Weight')
    axes[0, 1].set_title('Mean Edge Weight per Window')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight distribution (histogram)
    axes[1, 0].hist(df['weight'], bins=100, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Edge Weight')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Overall Weight Distribution')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Max weight per window
    axes[1, 1].plot(window_stats.index, window_stats['Max Weight'], linewidth=1, color='red')
    axes[1, 1].set_xlabel('Window Index')
    axes[1, 1].set_ylabel('Max Weight')
    axes[1, 1].set_title('Maximum Edge Weight per Window')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/tucker_option_d/analysis_edge_statistics.png', dpi=150)
    print(f"\n✓ Saved plot: results/tucker_option_d/analysis_edge_statistics.png")
    plt.close()
    
    return window_stats

def analyze_top_edges(df, top_n=20):
    """Identify strongest and most stable causal relationships"""
    print("\n" + "="*80)
    print(f"TOP {top_n} STRONGEST CAUSAL RELATIONSHIPS")
    print("="*80)
    
    # Group by parent-child pair
    edge_stats = df.groupby(['parent_name', 'child_name']).agg({
        'weight': ['mean', 'std', 'max', 'count'],
        'lag': 'first'
    }).round(3)
    
    edge_stats.columns = ['Mean Weight', 'Std Weight', 'Max Weight', 'Frequency', 'Lag']
    edge_stats = edge_stats.sort_values('Mean Weight', ascending=False)
    
    print(f"\nTop {top_n} by mean weight:")
    print(edge_stats.head(top_n).to_string())
    
    # Most stable edges (appear in many windows)
    print(f"\n\nMost stable edges (appear in most windows):")
    stable_edges = edge_stats.sort_values('Frequency', ascending=False).head(top_n)
    print(stable_edges.to_string())
    
    # Save to CSV
    edge_stats.to_csv('results/tucker_option_d/edge_statistics.csv')
    print(f"\n✓ Saved: results/tucker_option_d/edge_statistics.csv")
    
    return edge_stats

def analyze_lag_structure(df):
    """Analyze temporal lag structure"""
    print("\n" + "="*80)
    print("TEMPORAL LAG ANALYSIS")
    print("="*80)
    
    lag_counts = df['lag'].value_counts().sort_index()
    
    print("\nEdge distribution by lag:")
    for lag, count in lag_counts.items():
        pct = count / len(df) * 100
        print(f"  Lag {lag:2d}: {count:8,} edges ({pct:5.2f}%)")
    
    # Plot lag distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    axes[0].bar(lag_counts.index, lag_counts.values, edgecolor='black')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Edge Count')
    axes[0].set_title('Edge Distribution by Temporal Lag')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Weight by lag
    lag_weights = df.groupby('lag')['weight'].agg(['mean', 'std'])
    axes[1].errorbar(lag_weights.index, lag_weights['mean'], 
                     yerr=lag_weights['std'], fmt='o-', capsize=5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Mean Weight')
    axes[1].set_title('Mean Edge Weight by Lag')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/tucker_option_d/analysis_lag_structure.png', dpi=150)
    print(f"\n✓ Saved plot: results/tucker_option_d/analysis_lag_structure.png")
    plt.close()

def analyze_variable_importance(df):
    """Analyze which variables are most influential"""
    print("\n" + "="*80)
    print("VARIABLE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Out-degree (parent count)
    parent_counts = df['parent_name'].value_counts().head(20)
    print("\nTop 20 variables by out-degree (most causal):")
    for var, count in parent_counts.items():
        print(f"  {var:30s}: {count:6,} edges")
    
    # In-degree (child count)
    child_counts = df['child_name'].value_counts().head(20)
    print("\nTop 20 variables by in-degree (most affected):")
    for var, count in child_counts.items():
        print(f"  {var:30s}: {count:6,} edges")
    
    # Combined importance (weighted by edge weights)
    parent_importance = df.groupby('parent_name')['weight'].agg(['sum', 'count', 'mean']).round(3)
    parent_importance.columns = ['Total Weight', 'Out-Degree', 'Mean Weight']
    parent_importance = parent_importance.sort_values('Total Weight', ascending=False)
    
    print("\nTop 20 variables by total causal influence:")
    print(parent_importance.head(20).to_string())
    
    # Save
    parent_importance.to_csv('results/tucker_option_d/variable_importance.csv')
    print(f"\n✓ Saved: results/tucker_option_d/variable_importance.csv")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top parents
    top_parents = parent_counts.head(15)
    axes[0].barh(range(len(top_parents)), top_parents.values)
    axes[0].set_yticks(range(len(top_parents)))
    axes[0].set_yticklabels(top_parents.index, fontsize=8)
    axes[0].set_xlabel('Out-Degree (Number of Outgoing Edges)')
    axes[0].set_title('Top 15 Most Causal Variables')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Top children
    top_children = child_counts.head(15)
    axes[1].barh(range(len(top_children)), top_children.values, color='orange')
    axes[1].set_yticks(range(len(top_children)))
    axes[1].set_yticklabels(top_children.index, fontsize=8)
    axes[1].set_xlabel('In-Degree (Number of Incoming Edges)')
    axes[1].set_title('Top 15 Most Affected Variables')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/tucker_option_d/analysis_variable_importance.png', dpi=150)
    print(f"\n✓ Saved plot: results/tucker_option_d/analysis_variable_importance.png")
    plt.close()

def analyze_temporal_stability(df):
    """Analyze which edges persist across windows"""
    print("\n" + "="*80)
    print("TEMPORAL STABILITY ANALYSIS")
    print("="*80)
    
    # Count how many windows each edge appears in
    edge_persistence = df.groupby(['parent_name', 'child_name']).size()
    
    n_windows = df['window_idx'].nunique()
    
    print(f"\nEdge persistence across {n_windows} windows:")
    persistence_dist = edge_persistence.value_counts().sort_index()
    
    print(f"  1 window only:  {persistence_dist[persistence_dist.index == 1].sum():8,} edges")
    print(f"  2-10 windows:   {persistence_dist[(persistence_dist.index >= 2) & (persistence_dist.index <= 10)].sum():8,} edges")
    print(f"  11-50 windows:  {persistence_dist[(persistence_dist.index >= 11) & (persistence_dist.index <= 50)].sum():8,} edges")
    print(f"  51-100 windows: {persistence_dist[(persistence_dist.index >= 51) & (persistence_dist.index <= 100)].sum():8,} edges")
    print(f"  >100 windows:   {persistence_dist[persistence_dist.index > 100].sum():8,} edges")
    
    # Most persistent edges
    most_persistent = edge_persistence.sort_values(ascending=False).head(20)
    print(f"\nMost persistent edges (appear in most windows):")
    for (parent, child), count in most_persistent.items():
        pct = count / n_windows * 100
        print(f"  {parent:30s} → {child:30s}: {count:3d}/{n_windows} ({pct:5.1f}%)")

def generate_summary_report(df, window_stats, edge_stats):
    """Generate summary report"""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    n_windows = df['window_idx'].nunique()
    n_edges = len(df)
    n_unique_edges = df.groupby(['parent_name', 'child_name']).ngroups
    n_variables = len(set(df['parent_name'].unique()) | set(df['child_name'].unique()))
    
    summary = f"""
Tucker-CAM Option D Results Summary
====================================

Dataset:
  Total windows: {n_windows}
  Total edges: {n_edges:,}
  Unique edge types: {n_unique_edges:,}
  Variables involved: {n_variables}

Edge Statistics:
  Mean edges/window: {n_edges / n_windows:.1f}
  Std edges/window: {window_stats['Edge Count'].std():.1f}
  Edge count stability (CV): {window_stats['Edge Count'].std() / window_stats['Edge Count'].mean() * 100:.2f}%

Weight Statistics:
  Overall mean weight: {df['weight'].mean():.3f}
  Overall std weight: {df['weight'].std():.3f}
  Min weight: {df['weight'].min():.3f}
  Max weight: {df['weight'].max():.3f}

Temporal Structure:
  Contemporaneous edges (lag=0): {(df['lag'] == 0).sum():,} ({(df['lag'] == 0).sum() / n_edges * 100:.1f}%)
  Lagged edges (lag>0): {(df['lag'] > 0).sum():,} ({(df['lag'] > 0).sum() / n_edges * 100:.1f}%)
  Max lag: {df['lag'].max()}

Top 5 Strongest Edges:
{edge_stats.head(5)[['Mean Weight', 'Frequency', 'Lag']].to_string()}

Analysis files generated:
  - results/tucker_option_d/edge_statistics.csv
  - results/tucker_option_d/variable_importance.csv
  - results/tucker_option_d/analysis_edge_statistics.png
  - results/tucker_option_d/analysis_lag_structure.png
  - results/tucker_option_d/analysis_variable_importance.png
"""
    
    print(summary)
    
    # Save report
    with open('results/tucker_option_d/ANALYSIS_SUMMARY.txt', 'w') as f:
        f.write(summary)
    
    print(f"\n✓ Saved: results/tucker_option_d/ANALYSIS_SUMMARY.txt")

def main():
    """Main analysis pipeline"""
    weights_file = 'results/tucker_option_d/weights/weights_enhanced.csv'
    
    # Load data
    df = load_results(weights_file)
    
    # Run analyses
    window_stats = analyze_edge_statistics(df)
    edge_stats = analyze_top_edges(df, top_n=20)
    analyze_lag_structure(df)
    analyze_variable_importance(df)
    analyze_temporal_stability(df)
    
    # Generate summary
    generate_summary_report(df, window_stats, edge_stats)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved in: results/tucker_option_d/")

if __name__ == '__main__':
    main()
