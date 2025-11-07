#!/usr/bin/env python3
"""
Advanced detection methods for gradual anomalies

Implements:
1. Cumulative Divergence
2. Regional Aggregation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def cumulative_divergence_detection(window_stats, threshold_percentile=95):
    """
    Method 1: Cumulative Divergence Detection

    For gradual anomalies, small differences accumulate over many windows.
    """
    # Calculate cumulative sum of mean differences
    window_stats['cumsum_mean'] = window_stats['mean_diff'].cumsum()

    # Calculate rate of change (derivative)
    window_stats['cumsum_rate'] = window_stats['cumsum_mean'].diff()

    # Normalize cumsum to [0, 1]
    cumsum_norm = (window_stats['cumsum_mean'] - window_stats['cumsum_mean'].min()) / \
                  (window_stats['cumsum_mean'].max() - window_stats['cumsum_mean'].min() + 1e-10)

    # Detect sustained elevation
    threshold = np.percentile(window_stats['mean_diff'], threshold_percentile)
    sustained_elevation = window_stats['mean_diff'].rolling(window=20, min_periods=1).mean() > threshold

    # Find contiguous regions
    regions = []
    in_region = False
    start = None

    for idx, is_elevated in enumerate(sustained_elevation):
        if is_elevated and not in_region:
            start = idx
            in_region = True
        elif not is_elevated and in_region:
            regions.append((start, idx-1))
            in_region = False

    if in_region:
        regions.append((start, len(sustained_elevation)-1))

    return {
        'cumsum': window_stats['cumsum_mean'].values,
        'rate': window_stats['cumsum_rate'].values,
        'regions': regions,
        'sustained_threshold': threshold
    }


def regional_aggregation_detection(golden_df, anomaly_df, region_size=50):
    """
    Method 2: Regional Aggregation

    Compare aggregated statistics over regions instead of individual windows.
    Better for detecting 100-row gradual anomalies.
    """
    # Merge datasets
    merged = pd.merge(
        golden_df[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        anomaly_df[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='inner',
        suffixes=('_g', '_a')
    )

    merged['abs_diff'] = np.abs(merged['weight_a'] - merged['weight_g'])

    max_window = merged['window_idx'].max()

    # Divide into regions
    regions = []
    for start in range(0, max_window, region_size):
        end = min(start + region_size, max_window + 1)

        region_data = merged[
            (merged['window_idx'] >= start) &
            (merged['window_idx'] < end)
        ]

        if len(region_data) == 0:
            continue

        # Calculate regional statistics
        mean_diff = region_data['abs_diff'].mean()
        max_diff = region_data['abs_diff'].max()
        std_diff = region_data['abs_diff'].std()
        n_edges = len(region_data)

        regions.append({
            'start': start,
            'end': end - 1,
            'center': (start + end) / 2,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'std_diff': std_diff,
            'n_edges': n_edges
        })

    regions_df = pd.DataFrame(regions)

    # Detect anomalous regions
    mean_threshold = regions_df['mean_diff'].mean() + 2 * regions_df['mean_diff'].std()
    anomalous_regions = regions_df[regions_df['mean_diff'] > mean_threshold]

    return {
        'regions': regions_df,
        'anomalous_regions': anomalous_regions,
        'threshold': mean_threshold
    }


def hybrid_detection(golden_file, anomaly_file, anomaly_type, metadata):
    """
    Combine both methods for robust detection
    """
    print(f"\n{'='*80}")
    print(f"GRADUAL ANOMALY DETECTION: {anomaly_type.upper()}")
    print(f"{'='*80}")

    # Load data
    df_g = pd.read_csv(golden_file)
    df_a = pd.read_csv(anomaly_file)

    # Calculate per-window statistics (existing method)
    merged = pd.merge(
        df_g[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df_a[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='inner',
        suffixes=('_g', '_a')
    )
    merged['abs_diff'] = np.abs(merged['weight_a'] - merged['weight_g'])

    window_stats = merged.groupby('window_idx').agg({
        'abs_diff': ['mean', 'max', 'std', 'count']
    }).reset_index()
    window_stats.columns = ['window_idx', 'mean_diff', 'max_diff', 'std_diff', 'n_edges']

    # Method 1: Cumulative divergence
    print("\nMethod 1: Cumulative Divergence")
    cumsum_result = cumulative_divergence_detection(window_stats)
    print(f"  Sustained elevation regions: {len(cumsum_result['regions'])}")
    for start, end in cumsum_result['regions']:
        print(f"    Windows {start:3d} to {end:3d} (length: {end-start+1})")

    # Method 2: Regional aggregation
    print("\nMethod 2: Regional Aggregation (50-window regions)")
    regional_result = regional_aggregation_detection(df_g, df_a, region_size=50)
    print(f"  Anomalous regions detected: {len(regional_result['anomalous_regions'])}")
    for _, region in regional_result['anomalous_regions'].iterrows():
        print(f"    Windows {int(region['start']):3d}-{int(region['end']):3d}  "
              f"mean_diff={region['mean_diff']:.6f}")

    # Expected position from metadata
    if metadata and 'start' in metadata:
        true_start = metadata['start']
        true_length = metadata.get('length', 0)

        if true_length > 0:
            true_end = true_start + true_length - 1
            expected_window_start = max(0, true_start - 100 - 10)
            expected_window_end = true_end
        else:
            expected_window_start = max(0, true_start - 100 - 10)
            expected_window_end = true_start

        print(f"\nExpected anomaly windows: {expected_window_start} to {expected_window_end}")

        # Check if any detected region overlaps with expected
        print("\nValidation:")

        # For cumulative method
        cumsum_matches = []
        for start, end in cumsum_result['regions']:
            if start <= expected_window_end and end >= expected_window_start:
                overlap = min(end, expected_window_end) - max(start, expected_window_start) + 1
                cumsum_matches.append((start, end, overlap))

        if cumsum_matches:
            print(f"  Cumulative method: DETECTED")
            for start, end, overlap in cumsum_matches:
                print(f"    Region {start}-{end} overlaps by {overlap} windows")
        else:
            print(f"  Cumulative method: MISSED")

        # For regional method
        regional_matches = []
        for _, region in regional_result['anomalous_regions'].iterrows():
            if region['start'] <= expected_window_end and region['end'] >= expected_window_start:
                overlap = min(region['end'], expected_window_end) - max(region['start'], expected_window_start) + 1
                regional_matches.append((int(region['start']), int(region['end']), overlap))

        if regional_matches:
            print(f"  Regional method: DETECTED")
            for start, end, overlap in regional_matches:
                print(f"    Region {start}-{end} overlaps by {overlap} windows")
        else:
            print(f"  Regional method: MISSED")

    return {
        'window_stats': window_stats,
        'cumsum_result': cumsum_result,
        'regional_result': regional_result
    }


def visualize_gradual_detection(results, anomaly_type, metadata, output_file):
    """
    Create visualization showing both detection methods
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    window_stats = results['window_stats']
    cumsum_result = results['cumsum_result']
    regional_result = results['regional_result']

    windows = window_stats['window_idx']

    # Plot 1: Original max_diff (baseline method)
    ax = axes[0]
    ax.plot(windows, window_stats['max_diff'], 'b-', linewidth=1, alpha=0.7)
    ax.set_ylabel('Max Diff\n(Baseline)', fontsize=10)
    ax.set_title(f'Gradual Anomaly Detection: {anomaly_type}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative sum
    ax = axes[1]
    ax.plot(windows, cumsum_result['cumsum'], 'g-', linewidth=1.5)
    ax.set_ylabel('Cumulative\nMean Diff', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Highlight sustained regions
    for start, end in cumsum_result['regions']:
        ax.axvspan(start, end, alpha=0.3, color='yellow', label='Sustained elevation')

    # Plot 3: Regional mean differences
    ax = axes[2]
    regional_df = regional_result['regions']
    ax.bar(regional_df['center'], regional_df['mean_diff'], width=45,
           color='orange', alpha=0.7, edgecolor='black')
    ax.axhline(regional_result['threshold'], color='red', linestyle='--',
               label=f'Threshold ({regional_result["threshold"]:.4f})')
    ax.set_ylabel('Regional\nMean Diff', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Rate of change
    ax = axes[3]
    ax.plot(windows[1:], cumsum_result['rate'][1:], 'purple', linewidth=1)
    ax.set_ylabel('Rate of\nChange', fontsize=10)
    ax.set_xlabel('Window Index', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add expected region to all plots
    if metadata and 'start' in metadata:
        true_start = metadata['start']
        true_length = metadata.get('length', 0)

        if true_length > 0:
            expected_window_start = max(0, true_start - 100 - 10)
            expected_window_end = true_start + true_length
        else:
            expected_window_start = max(0, true_start - 100 - 10)
            expected_window_end = true_start

        for ax in axes:
            ax.axvspan(expected_window_start, expected_window_end, alpha=0.15, color='red',
                      label='Expected anomaly')

    axes[0].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")


def main():
    """
    Test gradual anomaly detection on all anomaly types
    """
    print("="*80)
    print("GRADUAL ANOMALY DETECTION TEST")
    print("="*80)

    golden_file = "results/Golden/weights/weights_enhanced_20251006_154344.csv"
    anomaly_types = ['drift', 'level_shift', 'trend_change', 'amplitude_change', 'variance_burst']

    for anomaly_type in anomaly_types:
        # Find most recent weight file
        pattern = f"results/no_mi_rolling_*/Anomaly/*{anomaly_type}*/weights/weights_enhanced*.csv"
        files = list(Path(".").glob(pattern))

        if not files:
            print(f"\nSkipping {anomaly_type}: no weight file found")
            continue

        anomaly_file = str(max(files, key=lambda p: p.stat().st_mtime))

        # Load metadata
        metadata_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"
        metadata = None
        if Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Run detection
        results = hybrid_detection(golden_file, anomaly_file, anomaly_type, metadata)

        # Visualize
        output_file = f"gradual_detection_{anomaly_type}.png"
        visualize_gradual_detection(results, anomaly_type, metadata, output_file)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)
    print("\nGenerated files:")
    for anomaly_type in anomaly_types:
        print(f"  - gradual_detection_{anomaly_type}.png")


if __name__ == "__main__":
    main()
