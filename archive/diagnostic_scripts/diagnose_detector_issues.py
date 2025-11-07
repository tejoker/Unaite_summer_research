#!/usr/bin/env python3
"""
Comprehensive diagnostic script to test multiple hypotheses
about why the robust detector still finds 900+ anomalies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def hypothesis1_find_comparable_golden_runs():
    """
    Hypothesis 1: Two golden runs are incomparable due to different settings

    Test: Find all golden runs and group by configuration
    """
    print("\n" + "="*100)
    print("HYPOTHESIS 1: Finding Comparable Golden Runs")
    print("="*100)

    all_golden_files = list(Path('results').rglob('weights_enhanced*.csv'))
    golden_chunk1 = [f for f in all_golden_files if 'Golden' in str(f) and '1th_chunk' in str(f)]

    print(f"\nFound {len(golden_chunk1)} Golden chunk 1 weight files:")

    run_info = []
    for f in golden_chunk1:
        df = pd.read_csv(f)

        # FILTER: Only include runs with enhanced format (must have t_end, i, j columns)
        required_columns = {'window_idx', 't_end', 'ts_end', 't_center', 'ts_center',
                           'lag', 'i', 'j', 'child_name', 'parent_name', 'weight'}
        if not required_columns.issubset(df.columns):
            print(f"  ⚠️  Skipping {f.name} - missing enhanced format columns")
            continue

        n_edges = len(df)
        n_windows = len(df['window_idx'].unique())

        # Extract configuration from path
        path_str = str(f)
        if 'no_mi' in path_str:
            config = 'no_mi'
        elif 'full_mi' in path_str:
            config = 'full_mi'
        elif 'baseline' in path_str:
            config = 'baseline'
        else:
            config = 'unknown'

        run_info.append({
            'path': f.relative_to('results'),
            'config': config,
            'n_edges': n_edges,
            'n_windows': n_windows,
            'edges_per_window': n_edges / n_windows if n_windows > 0 else 0
        })

    # Group by configuration
    for info in sorted(run_info, key=lambda x: x['config']):
        print(f"\n  Config: {info['config']:<15} Edges: {info['n_edges']:<8} Windows: {info['n_windows']:<5}")
        print(f"    {info['path']}")

    # Find pairs with same configuration
    configs = {}
    for info in run_info:
        config = info['config']
        if config not in configs:
            configs[config] = []
        configs[config].append(info)

    print("\n" + "-"*100)
    print("COMPARABLE PAIRS (same configuration):")

    comparable_pairs = []
    for config, runs in configs.items():
        if len(runs) >= 2:
            print(f"\n  Configuration '{config}': {len(runs)} runs available")
            comparable_pairs.append((runs[0], runs[1], config))
        else:
            print(f"\n  Configuration '{config}': Only {len(runs)} run(s) - CANNOT COMPARE")

    return comparable_pairs


def hypothesis2_debug_baseline_calculation(golden1_path, golden2_path):
    """
    Hypothesis 2: Baseline noise calculation fails silently

    Test: Add detailed logging to see what's happening
    """
    print("\n" + "="*100)
    print("HYPOTHESIS 2: Debugging Baseline Noise Calculation")
    print("="*100)

    g1 = pd.read_csv(golden1_path)
    g2 = pd.read_csv(golden2_path)

    print(f"\nGolden 1: {len(g1)} edges, {len(g1['window_idx'].unique())} windows")
    print(f"Golden 2: {len(g2)} edges, {len(g2['window_idx'].unique())} windows")

    all_diffs = []
    merge_stats = []

    for window_idx in sorted(g1['window_idx'].unique())[:10]:  # Test first 10 windows
        g1_window = g1[g1['window_idx'] == window_idx]
        g2_window = g2[g2['window_idx'] == window_idx]

        # Merge on child-parent-lag
        merged = pd.merge(
            g1_window,
            g2_window,
            on=['child_name', 'parent_name', 'lag'],
            suffixes=('_1', '_2'),
            how='inner'
        )

        merge_stats.append({
            'window': window_idx,
            'g1_edges': len(g1_window),
            'g2_edges': len(g2_window),
            'merged_edges': len(merged),
            'match_rate': len(merged) / max(len(g1_window), len(g2_window))
        })

        if len(merged) > 0:
            merged['abs_diff'] = abs(merged['weight_1'] - merged['weight_2'])
            all_diffs.extend(merged['abs_diff'].values)

    # Print merge statistics
    print("\nMerge Statistics (first 10 windows):")
    print(f"{'Window':<8} {'G1 Edges':<12} {'G2 Edges':<12} {'Merged':<12} {'Match Rate':<12}")
    print("-"*60)
    for stat in merge_stats:
        print(f"{stat['window']:<8} {stat['g1_edges']:<12} {stat['g2_edges']:<12} "
              f"{stat['merged_edges']:<12} {stat['match_rate']:<12.2%}")

    avg_match_rate = np.mean([s['match_rate'] for s in merge_stats])
    print(f"\nAverage match rate: {avg_match_rate:.2%}")

    if avg_match_rate < 0.5:
        print("⚠️  WARNING: Low match rate suggests incompatible runs!")

    if len(all_diffs) == 0:
        print("\n❌ FAILURE: No matching edges found - cannot calculate baseline noise")
        return None, None

    all_diffs = np.array(all_diffs)
    mean = all_diffs.mean()
    std = all_diffs.std()

    print(f"\nBaseline Noise from {len(all_diffs)} matched edges:")
    print(f"  Mean (μ): {mean:.6f}")
    print(f"  Std (σ):  {std:.6f}")
    print(f"  Min:      {all_diffs.min():.6f}")
    print(f"  Max:      {all_diffs.max():.6f}")
    print(f"  Median:   {np.median(all_diffs):.6f}")

    print(f"\nRecommended thresholds:")
    for sigma_mult in [3, 5, 10, 20]:
        threshold = mean + sigma_mult * std
        print(f"  μ + {sigma_mult}σ = {threshold:.6f}")

    return mean, std


def hypothesis3_test_sigma_multipliers(golden_path, anomaly_path, baseline_mean, baseline_std):
    """
    Hypothesis 3: Adaptive threshold is too low even with proper noise calculation

    Test: Try different sigma multipliers
    """
    print("\n" + "="*100)
    print("HYPOTHESIS 3: Testing Different Sigma Multipliers")
    print("="*100)

    if baseline_mean is None or baseline_std is None:
        print("⚠️  Skipping - need valid baseline noise")
        return

    golden = pd.read_csv(golden_path)
    anomaly = pd.read_csv(anomaly_path)

    print(f"\nTesting on: {Path(anomaly_path).parent.parent.name}")

    sigma_multipliers = [1, 2, 3, 5, 10, 20, 50]

    print(f"\n{'Sigma Mult':<12} {'Threshold':<15} {'Windows Detected':<20} {'% of Windows':<15}")
    print("-"*70)

    for sigma_mult in sigma_multipliers:
        threshold = baseline_mean + sigma_mult * baseline_std
        n_detected_windows = 0

        for window_idx in golden['window_idx'].unique():
            g_window = golden[golden['window_idx'] == window_idx]
            a_window = anomaly[anomaly['window_idx'] == window_idx]

            if len(a_window) == 0:
                continue

            merged = pd.merge(
                g_window, a_window,
                on=['child_name', 'parent_name', 'lag'],
                suffixes=('_g', '_a'),
                how='outer'
            ).fillna(0)

            merged['abs_diff'] = abs(merged['weight_a'] - merged['weight_g'])

            if (merged['abs_diff'] > threshold).any():
                n_detected_windows += 1

        total_windows = len(golden['window_idx'].unique())
        pct = 100 * n_detected_windows / total_windows

        print(f"{sigma_mult:<12} {threshold:<15.6f} {n_detected_windows:<20} {pct:<15.1f}%")


def hypothesis4_measure_dynotears_variance():
    """
    Hypothesis 4: DynoTEARS has enormous optimization variance

    Test: Check variance between two identical golden runs (if they exist with same settings)
    """
    print("\n" + "="*100)
    print("HYPOTHESIS 4: Measuring DynoTEARS Optimization Variance")
    print("="*100)

    # Find runs with same configuration
    pairs = hypothesis1_find_comparable_golden_runs()

    if not pairs:
        print("\n⚠️  No comparable pairs found - cannot test")
        return

    for run1, run2, config in pairs:
        print(f"\nAnalyzing variance for config '{config}':")

        df1 = pd.read_csv(Path('results') / run1['path'])
        df2 = pd.read_csv(Path('results') / run2['path'])

        # Sample 5 random windows
        sample_windows = np.random.choice(df1['window_idx'].unique(), min(5, len(df1['window_idx'].unique())), replace=False)

        for window_idx in sample_windows:
            w1 = df1[df1['window_idx'] == window_idx]
            w2 = df2[df2['window_idx'] == window_idx]

            merged = pd.merge(w1, w2, on=['child_name', 'parent_name', 'lag'],
                            suffixes=('_1', '_2'), how='inner')

            if len(merged) > 0:
                merged['abs_diff'] = abs(merged['weight_1'] - merged['weight_2'])
                merged['rel_diff'] = merged['abs_diff'] / (abs(merged['weight_1']) + 1e-6)

                print(f"\n  Window {window_idx}:")
                print(f"    Matched edges: {len(merged)}")
                print(f"    Mean abs diff: {merged['abs_diff'].mean():.6f}")
                print(f"    Max abs diff:  {merged['abs_diff'].max():.6f}")
                print(f"    Mean rel diff: {merged['rel_diff'].mean():.2%}")


def hypothesis5_verify_input_data():
    """
    Hypothesis 5: The "Golden" runs aren't actually identical data

    Test: Compare input CSV files
    """
    print("\n" + "="*100)
    print("HYPOTHESIS 5: Verifying Input Data is Identical")
    print("="*100)

    golden_csv = Path('data/Golden/chunking/output_of_the_1th_chunk.csv')

    if not golden_csv.exists():
        print(f"⚠️  Golden input file not found: {golden_csv}")
        return

    df = pd.read_csv(golden_csv, index_col=0)

    print(f"\nGolden input data:")
    print(f"  File: {golden_csv}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Data hash: {pd.util.hash_pandas_object(df).sum()}")

    print("\n✓ All Golden runs should use this exact same input file")
    print("  If they don't, weight differences are expected!")


def main():
    """Run all hypothesis tests"""
    print("="*100)
    print("COMPREHENSIVE DIAGNOSTIC: Why Are There 900+ Detections?")
    print("="*100)

    # Test 1: Find comparable runs
    comparable_pairs = hypothesis1_find_comparable_golden_runs()

    # Test 2: Debug baseline calculation
    if comparable_pairs:
        pair = comparable_pairs[0]  # Use first available pair
        g1_path = Path('results') / pair[0]['path']
        g2_path = Path('results') / pair[1]['path']

        baseline_mean, baseline_std = hypothesis2_debug_baseline_calculation(g1_path, g2_path)

        # Test 3: Try different thresholds
        # Find a spike anomaly for testing
        spike_files = list(Path('results').rglob('*spike*/weights/weights_enhanced*.csv'))
        if spike_files and baseline_mean is not None:
            spike_path = max([f for f in spike_files if 'Anomaly' in str(f)],
                           key=lambda p: p.stat().st_mtime)
            hypothesis3_test_sigma_multipliers(g1_path, spike_path, baseline_mean, baseline_std)

    # Test 4: Measure variance
    hypothesis4_measure_dynotears_variance()

    # Test 5: Verify input data
    hypothesis5_verify_input_data()

    print("\n" + "="*100)
    print("DIAGNOSTIC COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
