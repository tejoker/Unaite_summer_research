#!/usr/bin/env python3
"""
Compare all anomaly types against Golden baseline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def find_weight_file(anomaly_type):
    """Find the weight file for a given anomaly type"""
    # Search for most recent files containing the anomaly type
    patterns = [
        f"results/no_mi_rolling_*/Anomaly/*{anomaly_type}*/weights/weights_enhanced*.csv",
        f"results/Anomaly/weights/weights_enhanced*.csv",
        f"results/*Anomaly*{anomaly_type}*/weights/weights_enhanced*.csv",
    ]

    all_files = []
    for pattern in patterns:
        files = list(Path(".").glob(pattern))
        all_files.extend(files)

    if all_files:
        # Return most recent by modification time
        most_recent = max(all_files, key=lambda p: p.stat().st_mtime)
        return str(most_recent)

    return None


def load_anomaly_metadata(anomaly_type):
    """Load metadata for anomaly"""
    metadata_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"

    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return None


def analyze_anomaly(golden_file, anomaly_file, anomaly_type, metadata):
    """Analyze weight differences for one anomaly type"""
    print(f"\n{'='*80}")
    print(f"ANOMALY: {anomaly_type.upper()}")
    print(f"{'='*80}")

    if metadata:
        print(f"\nMetadata:")
        print(f"  Start row: {metadata.get('start', 'N/A')}")
        print(f"  Length: {metadata.get('length', 'N/A')}")
        print(f"  Magnitude: {metadata.get('magnitude', 'N/A')}")

    df_g = pd.read_csv(golden_file)
    df_a = pd.read_csv(anomaly_file)

    # Merge and calculate differences
    merged = pd.merge(
        df_g[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df_a[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='inner',
        suffixes=('_g', '_a')
    )

    merged['abs_diff'] = np.abs(merged['weight_a'] - merged['weight_g'])

    # Per-window statistics
    window_stats = merged.groupby('window_idx').agg({
        'abs_diff': ['mean', 'max', 'count']
    }).reset_index()
    window_stats.columns = ['window_idx', 'mean_diff', 'max_diff', 'n_edges']

    # Overall statistics
    print(f"\nOverall statistics:")
    print(f"  Total windows: {len(window_stats)}")
    print(f"  Mean of max_diff: {window_stats['max_diff'].mean():.6f}")
    print(f"  Std of max_diff: {window_stats['max_diff'].std():.6f}")
    print(f"  Max of max_diff: {window_stats['max_diff'].max():.6f} (window {int(window_stats.loc[window_stats['max_diff'].idxmax(), 'window_idx'])})")

    # Top anomalous windows
    top_windows = window_stats.nlargest(10, 'max_diff')
    print(f"\nTop 10 anomalous windows:")
    print(f"  {'Window':<8} {'Max Diff':<12} {'Mean Diff':<12}")
    print(f"  {'-'*32}")
    for _, row in top_windows.iterrows():
        print(f"  {int(row['window_idx']):<8} {row['max_diff']:<12.6f} {row['mean_diff']:<12.6f}")

    # Calculate expected windows if metadata available
    if metadata and 'start' in metadata:
        spike_row = metadata['start']
        window_size = 100
        lag = 10

        # Expected spike windows
        expected_start = max(0, spike_row - window_size - lag + 1)
        expected_end = spike_row

        print(f"\nExpected anomalous windows (based on position {spike_row}):")
        print(f"  Range: {expected_start} to {expected_end}")

        # Check if top windows overlap with expected
        top_window_indices = top_windows['window_idx'].values
        in_range = [(expected_start <= w <= expected_end) for w in top_window_indices]

        print(f"  Top windows in expected range: {sum(in_range)}/{len(in_range)}")

    return window_stats, top_windows


def compare_all_anomalies():
    """Compare all anomaly types"""
    golden_file = "results/Golden/weights/weights_enhanced_20251006_154344.csv"

    if not Path(golden_file).exists():
        print(f"Error: Golden weights not found: {golden_file}")
        return

    anomaly_types = [
        'spike',
        'drift',
        'level_shift',
        'trend_change',
        'amplitude_change',
        'variance_burst'
    ]

    results = {}

    for anomaly_type in anomaly_types:
        # Find weight file
        anomaly_file = find_weight_file(anomaly_type)

        if not anomaly_file:
            print(f"\nWarning: No weight file found for {anomaly_type}")
            continue

        # Load metadata
        metadata = load_anomaly_metadata(anomaly_type)

        # Analyze
        window_stats, top_windows = analyze_anomaly(
            golden_file, anomaly_file, anomaly_type, metadata
        )

        results[anomaly_type] = {
            'window_stats': window_stats,
            'top_windows': top_windows,
            'metadata': metadata,
            'file': anomaly_file
        }

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: ALL ANOMALY TYPES")
    print(f"{'='*80}")

    print(f"\n{'Anomaly Type':<20} {'Max Diff':<12} {'Mean Diff':<12} {'Top Window':<12} {'Detection':<10}")
    print(f"{'-'*80}")

    for anomaly_type in anomaly_types:
        if anomaly_type in results:
            ws = results[anomaly_type]['window_stats']
            tw = results[anomaly_type]['top_windows']

            max_diff = ws['max_diff'].max()
            mean_diff = ws['max_diff'].mean()
            top_window = int(tw.iloc[0]['window_idx'])

            # Simple detection criterion
            detection = "STRONG" if max_diff > 1.0 else "MODERATE" if max_diff > 0.5 else "WEAK"

            print(f"{anomaly_type:<20} {max_diff:<12.6f} {mean_diff:<12.6f} {top_window:<12} {detection:<10}")

    # Create comparison plot
    if len(results) > 0:
        create_comparison_plot(results, anomaly_types)


def create_comparison_plot(results, anomaly_types):
    """Create comparison visualization"""
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3*len(results)), sharex=True)

    if len(results) == 1:
        axes = [axes]

    for idx, anomaly_type in enumerate(anomaly_types):
        if anomaly_type not in results:
            continue

        ax = axes[idx]
        ws = results[anomaly_type]['window_stats']
        metadata = results[anomaly_type]['metadata']

        # Plot max diff per window
        ax.plot(ws['window_idx'], ws['max_diff'], 'b-', linewidth=1, alpha=0.7)
        ax.set_ylabel(f'{anomaly_type}\nMax Diff', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Mark expected region if metadata available
        if metadata and 'start' in metadata:
            spike_row = metadata['start']
            window_size = 100
            lag = 10
            expected_start = max(0, spike_row - window_size - lag + 1)
            expected_end = spike_row

            ax.axvspan(expected_start, expected_end, alpha=0.2, color='red',
                      label=f'Expected region (row {spike_row})')
            ax.legend(loc='upper right', fontsize=8)

        # Mark top window
        top_window = results[anomaly_type]['top_windows'].iloc[0]
        ax.scatter([top_window['window_idx']], [top_window['max_diff']],
                  color='red', s=50, zorder=5, marker='*')

    axes[-1].set_xlabel('Window Index', fontsize=10)
    axes[0].set_title('Weight Changes by Window for All Anomaly Types', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('all_anomalies_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: all_anomalies_comparison.png")


def main():
    print("="*80)
    print("COMPARING ALL ANOMALY TYPES AGAINST GOLDEN BASELINE")
    print("="*80)

    compare_all_anomalies()

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print("\nGenerated files:")
    print("  - all_anomalies_comparison.png")


if __name__ == "__main__":
    main()
