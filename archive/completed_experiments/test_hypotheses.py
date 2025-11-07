#!/usr/bin/env python3
"""
Test specific hypotheses about why spike isn't detected in weight comparison

H1: Preprocessing removes the spike (differencing smooths it out)
H2: Spike changes weight VALUES a lot, but not causal STRUCTURE (edge existence)
H3: Spike is too localized - only affects 1 sensor
H4: Lack of random seed creates variance >> spike signal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def test_h1_preprocessing_removes_spike():
    """
    H1: Does preprocessing remove the spike?

    Check if spike is still visible after differencing
    """
    print("\n" + "="*80)
    print("H1: DOES PREPROCESSING REMOVE THE SPIKE?")
    print("="*80)

    # Load original spike data
    spike_original = "data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__spike.csv"
    if not Path(spike_original).exists():
        print(f"File not found: {spike_original}")
        return

    df_orig = pd.read_csv(spike_original)
    print(f"\nOriginal spike data: {df_orig.shape}")
    print(f"Columns: {list(df_orig.columns)}")

    # The spike is at row 200 in "Temperatur Exzenterlager links"
    spike_col = "Temperatur Exzenterlager links"
    if spike_col not in df_orig.columns:
        print(f"Column '{spike_col}' not found")
        return

    # Check spike metadata
    metadata_file = "data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__spike.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    spike_row = metadata['start']
    spike_magnitude = metadata['magnitude']

    print(f"\nSpike metadata:")
    print(f"  Row: {spike_row}")
    print(f"  Magnitude: {spike_magnitude}")
    print(f"  Column: {spike_col}")

    # Values around spike
    print(f"\nOriginal values around spike (rows {spike_row-2} to {spike_row+2}):")
    for i in range(spike_row-2, spike_row+3):
        val = df_orig.iloc[i][spike_col]
        marker = " <-- SPIKE" if i == spike_row else ""
        print(f"  Row {i}: {val:.4f}{marker}")

    # Check if preprocessed file exists
    preprocessed_files = list(Path("results").glob("**/Anomaly/**/preprocessing/*differenced*.csv"))

    if not preprocessed_files:
        print("\nNo preprocessed differenced files found")
        print("Searching for any preprocessing output...")
        preprocessed_files = list(Path("results").glob("**/Anomaly/**/preprocessing/*.csv"))

    if preprocessed_files:
        print(f"\nFound {len(preprocessed_files)} preprocessed files")
        # Use the most recent
        preproc_file = preprocessed_files[0]
        print(f"Checking: {preproc_file}")

        df_preproc = pd.read_csv(preproc_file)
        print(f"Preprocessed shape: {df_preproc.shape}")
        print(f"Columns: {list(df_preproc.columns)}")

        # Look for differenced column
        diff_col = spike_col + "_diff"
        if diff_col in df_preproc.columns:
            print(f"\nDifferenced values around spike position (rows {spike_row-2} to {spike_row+2}):")
            for i in range(max(0, spike_row-2), min(len(df_preproc), spike_row+3)):
                val = df_preproc.iloc[i][diff_col]
                marker = " <-- SPIKE POSITION" if i == spike_row else ""
                print(f"  Row {i}: {val:.4f}{marker}")

            # Calculate magnitude in differenced data
            diff_at_spike = df_preproc.iloc[spike_row][diff_col]
            diff_mean = df_preproc[diff_col].mean()
            diff_std = df_preproc[diff_col].std()

            print(f"\nSpike statistics in DIFFERENCED data:")
            print(f"  Value at spike: {diff_at_spike:.4f}")
            print(f"  Mean: {diff_mean:.4f}")
            print(f"  Std: {diff_std:.4f}")
            print(f"  Z-score: {(diff_at_spike - diff_mean)/diff_std:.2f}")

            if abs((diff_at_spike - diff_mean)/diff_std) > 3:
                print("  -> SPIKE IS STILL VISIBLE (Z-score > 3)")
            else:
                print("  -> SPIKE IS SMOOTHED OUT (Z-score < 3)")
        else:
            print(f"Column '{diff_col}' not found in preprocessed data")
    else:
        print("\nNo preprocessed files found")

    print("\n" + "-"*80)
    print("CONCLUSION H1:")
    print("Check above if spike is still visible after preprocessing")
    print("-"*80)


def test_h2_weights_vs_structure():
    """
    H2: Does spike change weight VALUES but not STRUCTURE?

    Compare:
    - Number of edges (structure)
    - Weight magnitudes (values)
    """
    print("\n" + "="*80)
    print("H2: SPIKE CHANGES WEIGHTS BUT NOT STRUCTURE?")
    print("="*80)

    golden = "results/Golden/weights/weights_enhanced_20251006_154344.csv"
    spike = "results/Anomaly/weights/weights_enhanced_20251006_160916.csv"

    if not Path(golden).exists() or not Path(spike).exists():
        print("Files not found")
        return

    df_g = pd.read_csv(golden)
    df_s = pd.read_csv(spike)

    print(f"\nTotal edges:")
    print(f"  Golden: {len(df_g)}")
    print(f"  Spike:  {len(df_s)}")
    print(f"  Difference: {abs(len(df_g) - len(df_s))} ({100*abs(len(df_g)-len(df_s))/len(df_g):.1f}%)")

    # Group by window to see structure
    print(f"\nEdges per window:")
    g_per_window = df_g.groupby('window_idx').size()
    s_per_window = df_s.groupby('window_idx').size()

    print(f"  Golden: mean={g_per_window.mean():.1f}, std={g_per_window.std():.1f}")
    print(f"  Spike:  mean={s_per_window.mean():.1f}, std={s_per_window.std():.1f}")

    # Merge to compare same edges
    merged = pd.merge(
        df_g[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df_s[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='outer',
        suffixes=('_g', '_s'),
        indicator=True
    )

    print(f"\nEdge overlap:")
    print(f"  In both:        {(merged['_merge'] == 'both').sum()}")
    print(f"  Only in Golden: {(merged['_merge'] == 'left_only').sum()}")
    print(f"  Only in Spike:  {(merged['_merge'] == 'right_only').sum()}")

    # For edges in both, compare weights
    both = merged[merged['_merge'] == 'both'].copy()
    both['weight_g'] = both['weight_g'].fillna(0)
    both['weight_s'] = both['weight_s'].fillna(0)
    both['abs_diff'] = np.abs(both['weight_s'] - both['weight_g'])

    print(f"\nWeight differences (for {len(both)} edges in both):")
    print(f"  Mean abs diff: {both['abs_diff'].mean():.6f}")
    print(f"  Max abs diff:  {both['abs_diff'].max():.6f}")
    print(f"  Median abs diff: {both['abs_diff'].median():.6f}")

    # Focus on windows around spike (91-190)
    spike_windows = both[(both['window_idx'] >= 91) & (both['window_idx'] <= 190)]
    other_windows = both[(both['window_idx'] < 91) | (both['window_idx'] > 190)]

    print(f"\nWeight differences in SPIKE windows (91-190):")
    print(f"  N edges: {len(spike_windows)}")
    print(f"  Mean abs diff: {spike_windows['abs_diff'].mean():.6f}")
    print(f"  Max abs diff:  {spike_windows['abs_diff'].max():.6f}")

    print(f"\nWeight differences in OTHER windows:")
    print(f"  N edges: {len(other_windows)}")
    print(f"  Mean abs diff: {other_windows['abs_diff'].mean():.6f}")
    print(f"  Max abs diff:  {other_windows['abs_diff'].max():.6f}")

    if len(spike_windows) > 0 and len(other_windows) > 0:
        ratio = spike_windows['abs_diff'].mean() / other_windows['abs_diff'].mean()
        print(f"\n  Ratio (spike/other): {ratio:.2f}x")

        if ratio > 2:
            print("  -> Spike windows have MUCH larger weight changes")
        elif ratio > 1.2:
            print("  -> Spike windows have moderately larger weight changes")
        else:
            print("  -> Spike windows DON'T have larger weight changes")

    print("\n" + "-"*80)
    print("CONCLUSION H2:")
    print("Check if spike changes WEIGHT VALUES more than STRUCTURE (edge count)")
    print("-"*80)


def test_h3_spike_localization():
    """
    H3: Is spike too localized?

    Check if spike only affects one sensor or propagates
    """
    print("\n" + "="*80)
    print("H3: IS SPIKE TOO LOCALIZED (ONLY 1 SENSOR)?")
    print("="*80)

    golden = "results/Golden/weights/weights_enhanced_20251006_154344.csv"
    spike = "results/Anomaly/weights/weights_enhanced_20251006_160916.csv"

    df_g = pd.read_csv(golden)
    df_s = pd.read_csv(spike)

    # Merge and calculate differences
    merged = pd.merge(
        df_g[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df_s[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='inner',
        suffixes=('_g', '_s')
    )

    merged['abs_diff'] = np.abs(merged['weight_s'] - merged['weight_g'])

    # Focus on spike windows
    spike_windows = merged[(merged['window_idx'] >= 91) & (merged['window_idx'] <= 190)]

    # Which nodes show the largest changes?
    print(f"\nEdges with largest weight changes in spike windows:")
    top_changes = spike_windows.nlargest(20, 'abs_diff')[
        ['parent_name', 'child_name', 'weight_g', 'weight_s', 'abs_diff', 'window_idx', 'lag']
    ]

    for idx, row in top_changes.iterrows():
        print(f"  {row['parent_name']:40} -> {row['child_name']:40}")
        print(f"    Golden: {row['weight_g']:8.4f}, Spike: {row['weight_s']:8.4f}, Diff: {row['abs_diff']:8.4f}")
        print(f"    Window: {row['window_idx']}, Lag: {row['lag']}")

    # Count how many unique nodes are affected
    affected_nodes = set()
    threshold = 0.1  # significant change
    large_changes = spike_windows[spike_windows['abs_diff'] > threshold]

    for _, row in large_changes.iterrows():
        affected_nodes.add(row['parent_name'])
        affected_nodes.add(row['child_name'])

    print(f"\nNodes with significant weight changes (>0.1):")
    print(f"  Total unique nodes: {len(affected_nodes)}")
    print(f"  Nodes: {sorted(affected_nodes)}")

    spike_sensor = "Temperatur Exzenterlager links_diff"
    if spike_sensor in affected_nodes:
        print(f"\n  -> Spike sensor '{spike_sensor}' IS in affected nodes")

    if len(affected_nodes) == 1:
        print("\n  -> Only 1 sensor affected - VERY LOCALIZED")
    elif len(affected_nodes) <= 2:
        print("\n  -> 2 sensors affected - SOMEWHAT LOCALIZED")
    else:
        print(f"\n  -> {len(affected_nodes)} sensors affected - PROPAGATES")

    print("\n" + "-"*80)
    print("CONCLUSION H3:")
    print("Check how many sensors show significant weight changes")
    print("-"*80)


def test_h4_random_seed_variance():
    """
    H4: Does lack of random seed create high variance?

    This is already tested in test_reproducibility.py
    Just summarize the finding
    """
    print("\n" + "="*80)
    print("H4: DOES LACK OF RANDOM SEED CREATE HIGH VARIANCE?")
    print("="*80)

    print("\nFrom test_reproducibility.py:")
    print("  - NO random seed found in dynotears.py")
    print("  - Golden vs Golden 99th percentile: 0.85")
    print("  - Golden vs Spike 99th percentile: 0.51")
    print()
    print("  -> Variance from no seed (0.85) > Signal from spike (0.51)")
    print("  -> This IS a major problem")

    print("\n" + "-"*80)
    print("CONCLUSION H4:")
    print("YES - lack of random seed creates variance LARGER than spike signal")
    print("RECOMMENDATION: Add random seed to dynotears.py")
    print("-"*80)


def main():
    print("="*80)
    print("TESTING HYPOTHESES ABOUT SPIKE DETECTION FAILURE")
    print("="*80)

    test_h1_preprocessing_removes_spike()
    test_h2_weights_vs_structure()
    test_h3_spike_localization()
    test_h4_random_seed_variance()

    print("\n" + "="*80)
    print("SUMMARY OF ALL HYPOTHESES")
    print("="*80)
    print("\nCheck the conclusions above for each hypothesis:")
    print("  H1: Does preprocessing remove spike?")
    print("  H2: Does spike change weights but not structure?")
    print("  H3: Is spike too localized (1 sensor)?")
    print("  H4: Does no random seed create high variance?")
    print("\nBased on results, we can determine the ROOT CAUSE")
    print("="*80)

if __name__ == "__main__":
    main()
