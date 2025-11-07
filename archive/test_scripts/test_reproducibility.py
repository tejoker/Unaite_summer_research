#!/usr/bin/env python3
"""
Test DynoTEARS reproducibility and assumptions about weight variability

This script tests:
1. Whether DynoTEARS is deterministic (same data -> same weights)
2. Whether random initialization causes weight differences
3. What magnitude of differences to expect from optimization noise
4. Appropriate threshold for detecting real anomalies vs noise
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_weight_differences(file1, file2, label1="Run 1", label2="Run 2"):
    """
    Compare two weight files and analyze differences

    Returns statistics about weight differences
    """
    print(f"\n{'='*80}")
    print(f"COMPARING: {label1} vs {label2}")
    print(f"{'='*80}")

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    print(f"{label1}: {df1.shape[0]} edges")
    print(f"{label2}: {df2.shape[0]} edges")

    # Merge on window, lag, parent, child to align edges
    merged = pd.merge(
        df1[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df2[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='outer',
        suffixes=('_1', '_2')
    )

    # Fill NaN for edges that only exist in one file
    merged['weight_1'] = merged['weight_1'].fillna(0.0)
    merged['weight_2'] = merged['weight_2'].fillna(0.0)

    # Calculate differences
    merged['abs_diff'] = np.abs(merged['weight_2'] - merged['weight_1'])
    merged['rel_diff'] = np.abs(merged['weight_2'] - merged['weight_1']) / (np.abs(merged['weight_1']) + 1e-10)

    # Edges that exist in both
    both_nonzero = (np.abs(merged['weight_1']) > 1e-6) & (np.abs(merged['weight_2']) > 1e-6)
    only_in_1 = (np.abs(merged['weight_1']) > 1e-6) & (np.abs(merged['weight_2']) < 1e-6)
    only_in_2 = (np.abs(merged['weight_1']) < 1e-6) & (np.abs(merged['weight_2']) > 1e-6)

    print(f"\nEdge overlap:")
    print(f"  Edges in both:       {both_nonzero.sum()}")
    print(f"  Edges only in {label1}: {only_in_1.sum()}")
    print(f"  Edges only in {label2}: {only_in_2.sum()}")

    print(f"\nAbsolute differences (for edges in both):")
    if both_nonzero.sum() > 0:
        abs_diffs = merged.loc[both_nonzero, 'abs_diff']
        print(f"  Min:    {abs_diffs.min():.6f}")
        print(f"  Max:    {abs_diffs.max():.6f}")
        print(f"  Mean:   {abs_diffs.mean():.6f}")
        print(f"  Median: {abs_diffs.median():.6f}")
        print(f"  Std:    {abs_diffs.std():.6f}")
        print(f"  95th percentile: {abs_diffs.quantile(0.95):.6f}")
        print(f"  99th percentile: {abs_diffs.quantile(0.99):.6f}")

    print(f"\nRelative differences (for edges in both):")
    if both_nonzero.sum() > 0:
        rel_diffs = merged.loc[both_nonzero, 'rel_diff']
        print(f"  Mean:   {rel_diffs.mean():.6f}")
        print(f"  Median: {rel_diffs.median():.6f}")
        print(f"  95th percentile: {rel_diffs.quantile(0.95):.6f}")

    # Test different thresholds
    print(f"\nNumber of 'anomalous' edges at different thresholds:")
    for threshold in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        n_anomalous = (merged['abs_diff'] > threshold).sum()
        pct = 100 * n_anomalous / len(merged)
        print(f"  threshold={threshold:.3f}: {n_anomalous}/{len(merged)} edges ({pct:.1f}%)")

    # Analyze by window
    print(f"\nWindow-level analysis:")
    window_stats = merged.groupby('window_idx').agg({
        'abs_diff': ['mean', 'max', 'count']
    }).reset_index()
    window_stats.columns = ['window_idx', 'mean_diff', 'max_diff', 'n_edges']

    # Windows with large differences
    large_diff_windows = window_stats[window_stats['max_diff'] > 0.1]['window_idx'].tolist()
    print(f"  Windows with max_diff > 0.1: {len(large_diff_windows)}")
    if large_diff_windows and len(large_diff_windows) <= 20:
        print(f"  Window indices: {large_diff_windows}")

    return {
        'merged': merged,
        'window_stats': window_stats,
        'both_nonzero': both_nonzero,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2
    }

def test_same_dataset_reproducibility():
    """
    Test if running on the same Golden dataset twice gives identical results
    """
    print("\n" + "="*80)
    print("TEST 1: REPRODUCIBILITY - Do we get same results on same data?")
    print("="*80)

    # Check if we have multiple Golden runs
    golden_weights = list(Path("results").glob("**/Golden/**/weights_enhanced*.csv"))

    if len(golden_weights) < 2:
        print("Need at least 2 Golden weight files to test reproducibility")
        print(f"Found: {len(golden_weights)} files")
        for f in golden_weights:
            print(f"  {f}")
        return None

    print(f"Found {len(golden_weights)} Golden weight files")
    file1 = golden_weights[0]
    file2 = golden_weights[1] if len(golden_weights) > 1 else golden_weights[0]

    print(f"\nComparing:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")

    return analyze_weight_differences(file1, file2, "Golden Run 1", "Golden Run 2")

def test_golden_vs_spike():
    """
    Test Golden vs Spike to see if differences are larger than noise
    """
    print("\n" + "="*80)
    print("TEST 2: GOLDEN vs SPIKE - Are anomaly differences larger than noise?")
    print("="*80)

    golden = "results/Golden/weights/weights_enhanced_20251006_154344.csv"
    spike = "results/Anomaly/weights/weights_enhanced_20251006_160916.csv"

    if not Path(golden).exists() or not Path(spike).exists():
        print(f"Files not found:")
        print(f"  Golden: {Path(golden).exists()}")
        print(f"  Spike:  {Path(spike).exists()}")
        return None

    return analyze_weight_differences(golden, spike, "Golden", "Spike")

def check_random_seed():
    """
    Check if DynoTEARS sets random seed
    """
    print("\n" + "="*80)
    print("TEST 3: RANDOM SEED - Is random seed set in DynoTEARS?")
    print("="*80)

    dynotears_file = Path("executable/final_pipeline/dynotears.py")
    if not dynotears_file.exists():
        print(f"File not found: {dynotears_file}")
        return

    with open(dynotears_file, 'r') as f:
        content = f.read()

    seed_patterns = [
        'torch.manual_seed',
        'np.random.seed',
        'random.seed',
        'torch.cuda.manual_seed'
    ]

    print("Searching for random seed initialization...")
    found_seeds = []
    for pattern in seed_patterns:
        if pattern in content:
            found_seeds.append(pattern)
            # Find line number
            for i, line in enumerate(content.split('\n'), 1):
                if pattern in line:
                    print(f"  Found '{pattern}' at line {i}: {line.strip()}")

    if not found_seeds:
        print("  WARNING: No random seed initialization found!")
        print("  This means results are NOT reproducible - different runs give different weights")
    else:
        print(f"\n  Found {len(found_seeds)} seed initialization(s)")

def main():
    print("="*80)
    print("DYNOTEARS REPRODUCIBILITY AND THRESHOLD ANALYSIS")
    print("="*80)

    # Test 3: Check for random seed
    check_random_seed()

    # Test 1: Reproducibility on same data
    golden_results = test_same_dataset_reproducibility()

    # Test 2: Golden vs Spike
    spike_results = test_golden_vs_spike()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY AND CONCLUSIONS")
    print("="*80)

    print("\nKey findings:")
    print("1. Check above if random seed is set (affects reproducibility)")
    print("2. Compare 'Golden vs Golden' differences to 'Golden vs Spike' differences")
    print("3. If Golden-Golden diffs are similar to Golden-Spike diffs, then:")
    print("   -> You're just seeing optimization noise, NOT anomaly detection")
    print("4. If Golden-Spike diffs are MUCH LARGER than Golden-Golden diffs, then:")
    print("   -> Real anomaly signal exists, but threshold needs tuning")

    if golden_results and spike_results:
        print("\nRecommended threshold:")
        # Get 99th percentile of Golden-Golden differences as baseline noise
        if golden_results['both_nonzero'].sum() > 0:
            noise_level = golden_results['merged'].loc[golden_results['both_nonzero'], 'abs_diff'].quantile(0.99)
            print(f"  Baseline noise (99th percentile of Golden-Golden): {noise_level:.6f}")
            print(f"  Suggested threshold: {2*noise_level:.6f} (2x baseline noise)")

        # Check if spike has larger differences
        if spike_results['both_nonzero'].sum() > 0:
            spike_95th = spike_results['merged'].loc[spike_results['both_nonzero'], 'abs_diff'].quantile(0.95)
            print(f"  Golden-Spike 95th percentile: {spike_95th:.6f}")

if __name__ == "__main__":
    main()
