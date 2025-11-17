#!/usr/bin/env python3
"""
Visualize per-window weight changes to detect anomalies

Shows:
1. Max weight difference per window
2. Mean weight difference per window
3. Number of large changes per window
4. Spike location overlay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def analyze_per_window_changes(golden_file, spike_file, spike_metadata_file=None):
    """
    Analyze weight changes on a per-window basis
    """
    print("="*80)
    print("WINDOW-BY-WINDOW WEIGHT CHANGE ANALYSIS")
    print("="*80)

    # Load weights
    df_g = pd.read_csv(golden_file)
    df_s = pd.read_csv(spike_file)

    print(f"\nGolden: {len(df_g)} edges across {df_g['window_idx'].nunique()} windows")
    print(f"Spike:  {len(df_s)} edges across {df_s['window_idx'].nunique()} windows")

    # Merge on edge identity
    merged = pd.merge(
        df_g[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df_s[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='inner',
        suffixes=('_g', '_s')
    )

    merged['abs_diff'] = np.abs(merged['weight_s'] - merged['weight_g'])

    # Calculate per-window statistics
    window_stats = merged.groupby('window_idx').agg({
        'abs_diff': ['mean', 'max', 'std', 'count'],
        'weight_g': 'mean',
        'weight_s': 'mean'
    }).reset_index()

    window_stats.columns = ['window_idx', 'mean_diff', 'max_diff', 'std_diff', 'n_edges',
                            'mean_weight_g', 'mean_weight_s']

    # Count large changes per window
    threshold = 0.1
    large_changes = merged[merged['abs_diff'] > threshold].groupby('window_idx').size()
    window_stats['n_large_changes'] = window_stats['window_idx'].map(large_changes).fillna(0)

    # Load spike metadata if available
    spike_windows = None
    if spike_metadata_file and Path(spike_metadata_file).exists():
        with open(spike_metadata_file, 'r') as f:
            metadata = json.load(f)

        spike_row = metadata['start']
        window_size = 100  # typical
        lag = 10

        # Calculate which windows contain the spike
        spike_windows = []
        for w in range(spike_row - window_size - lag, spike_row + 10):
            start = w + lag
            end = start + window_size
            if start <= spike_row < end:
                spike_windows.append(w)

        print(f"\nSpike at row {spike_row}")
        print(f"Expected spike windows: {spike_windows[0]} to {spike_windows[-1]}")

    return window_stats, spike_windows, merged


def plot_weight_changes(window_stats, spike_windows=None, output_file="weight_changes_by_window.png"):
    """
    Create comprehensive visualization of weight changes
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    windows = window_stats['window_idx']

    # Plot 1: Max difference per window
    ax = axes[0]
    ax.plot(windows, window_stats['max_diff'], 'b-', linewidth=1, alpha=0.7)
    ax.set_ylabel('Max Weight\nDifference', fontsize=10)
    ax.set_title('Window-by-Window Weight Changes: Golden vs Spike', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add spike region
    if spike_windows:
        ax.axvspan(spike_windows[0], spike_windows[-1], alpha=0.2, color='red',
                   label=f'Spike Region (windows {spike_windows[0]}-{spike_windows[-1]})')
        ax.legend(loc='upper right')

    # Highlight windows with large changes
    large_max = window_stats[window_stats['max_diff'] > 0.5]
    if len(large_max) > 0:
        ax.scatter(large_max['window_idx'], large_max['max_diff'],
                  color='red', s=30, zorder=5, alpha=0.7, label='Max diff > 0.5')

    # Plot 2: Mean difference per window
    ax = axes[1]
    ax.plot(windows, window_stats['mean_diff'], 'g-', linewidth=1, alpha=0.7)
    ax.set_ylabel('Mean Weight\nDifference', fontsize=10)
    ax.grid(True, alpha=0.3)

    if spike_windows:
        ax.axvspan(spike_windows[0], spike_windows[-1], alpha=0.2, color='red')

    # Plot 3: Number of large changes per window
    ax = axes[2]
    ax.bar(windows, window_stats['n_large_changes'], width=1,
           color='orange', alpha=0.7, edgecolor='none')
    ax.set_ylabel('# Large Changes\n(diff > 0.1)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    if spike_windows:
        ax.axvspan(spike_windows[0], spike_windows[-1], alpha=0.2, color='red')

    # Plot 4: Standard deviation of differences
    ax = axes[3]
    ax.plot(windows, window_stats['std_diff'], 'purple', linewidth=1, alpha=0.7)
    ax.set_ylabel('Std Dev of\nDifferences', fontsize=10)
    ax.set_xlabel('Window Index', fontsize=11)
    ax.grid(True, alpha=0.3)

    if spike_windows:
        ax.axvspan(spike_windows[0], spike_windows[-1], alpha=0.2, color='red')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    return fig


def print_top_anomalous_windows(window_stats, n=20):
    """
    Print windows with largest changes
    """
    print("\n" + "="*80)
    print(f"TOP {n} WINDOWS WITH LARGEST MAX WEIGHT CHANGES")
    print("="*80)

    top_windows = window_stats.nlargest(n, 'max_diff')

    print(f"\n{'Window':<8} {'Max Diff':<12} {'Mean Diff':<12} {'# Large':<10} {'# Edges':<10}")
    print("-"*80)

    for _, row in top_windows.iterrows():
        print(f"{int(row['window_idx']):<8} "
              f"{row['max_diff']:<12.6f} "
              f"{row['mean_diff']:<12.6f} "
              f"{int(row['n_large_changes']):<10} "
              f"{int(row['n_edges']):<10}")

    return top_windows


def analyze_specific_windows(merged, window_indices, top_n=5):
    """
    Show which specific edges changed in given windows
    """
    print("\n" + "="*80)
    print("DETAILED EDGE ANALYSIS FOR TOP ANOMALOUS WINDOWS")
    print("="*80)

    for window_idx in window_indices[:5]:  # Top 5 windows
        print(f"\n{'='*80}")
        print(f"WINDOW {window_idx}")
        print(f"{'='*80}")

        window_data = merged[merged['window_idx'] == window_idx].copy()
        window_data = window_data.nlargest(top_n, 'abs_diff')

        for i, row in window_data.iterrows():
            print(f"\n  Edge: {row['parent_name']} -> {row['child_name']}")
            print(f"    Lag: {row['lag']}")
            print(f"    Golden weight: {row['weight_g']:8.6f}")
            print(f"    Spike weight:  {row['weight_s']:8.6f}")
            print(f"    Difference:    {row['abs_diff']:8.6f}")
            print(f"    Ratio: {row['weight_s']/(abs(row['weight_g'])+1e-10):.3f}")


def main():
    golden_file = "results/Golden/weights/weights_enhanced_20251006_154344.csv"
    spike_file = "results/Anomaly/weights/weights_enhanced_20251006_160916.csv"
    metadata_file = "data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__spike.json"

    if not Path(golden_file).exists() or not Path(spike_file).exists():
        print(f"Error: Weight files not found")
        print(f"  Golden: {Path(golden_file).exists()}")
        print(f"  Spike:  {Path(spike_file).exists()}")
        return

    # Analyze
    window_stats, spike_windows, merged = analyze_per_window_changes(
        golden_file, spike_file, metadata_file
    )

    # Print top anomalous windows
    top_windows = print_top_anomalous_windows(window_stats, n=20)

    # Analyze specific edges in top windows
    analyze_specific_windows(merged, top_windows['window_idx'].values, top_n=5)

    # Create visualization
    plot_weight_changes(window_stats, spike_windows)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nOverall:")
    print(f"  Total windows: {len(window_stats)}")
    print(f"  Mean of max_diff: {window_stats['max_diff'].mean():.6f}")
    print(f"  Std of max_diff: {window_stats['max_diff'].std():.6f}")
    print(f"  Max of max_diff: {window_stats['max_diff'].max():.6f} (window {window_stats.loc[window_stats['max_diff'].idxmax(), 'window_idx']:.0f})")

    if spike_windows:
        spike_stats = window_stats[window_stats['window_idx'].isin(spike_windows)]
        other_stats = window_stats[~window_stats['window_idx'].isin(spike_windows)]

        print(f"\nSpike windows ({spike_windows[0]}-{spike_windows[-1]}):")
        print(f"  Mean of max_diff: {spike_stats['max_diff'].mean():.6f}")
        print(f"  Max of max_diff:  {spike_stats['max_diff'].max():.6f}")

        print(f"\nOther windows:")
        print(f"  Mean of max_diff: {other_stats['max_diff'].mean():.6f}")
        print(f"  Max of max_diff:  {other_stats['max_diff'].max():.6f}")

        ratio = spike_stats['max_diff'].mean() / other_stats['max_diff'].mean()
        print(f"\nRatio (spike/other): {ratio:.2f}x")

        if ratio > 2:
            print("  -> STRONG anomaly signal in spike windows!")
        elif ratio > 1.5:
            print("  -> MODERATE anomaly signal in spike windows")
        else:
            print("  -> WEAK anomaly signal in spike windows")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
