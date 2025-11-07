#!/usr/bin/env python3
"""
Diagnose why gradual anomaly detection is failing

Check if the problem is:
1. The anomalies aren't actually in the preprocessed data
2. DynoTEARS weights genuinely don't differ
3. Our detection methods are too insensitive
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def check_raw_data_has_anomaly(anomaly_type):
    """
    Verify that the anomaly is actually present in the raw data
    """
    print(f"\n{'='*80}")
    print(f"CHECKING RAW DATA: {anomaly_type.upper()}")
    print(f"{'='*80}")

    # Load metadata
    metadata_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"
    if not Path(metadata_file).exists():
        print("Metadata not found")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    start = metadata['start']
    length = metadata.get('length', 0)
    magnitude = metadata.get('magnitude', 'N/A')

    # Load golden and anomaly data
    golden_file = "data/Golden/chunking/output_of_the_1th_chunk.csv"
    anomaly_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.csv"

    if not Path(golden_file).exists() or not Path(anomaly_file).exists():
        print("Data files not found")
        return

    df_golden = pd.read_csv(golden_file)
    df_anomaly = pd.read_csv(anomaly_file)

    sensor = "Temperatur Exzenterlager links"

    print(f"\nMetadata:")
    print(f"  Start: {start}")
    print(f"  Length: {length}")
    print(f"  Magnitude: {magnitude}")

    # Check values before, during, and after anomaly
    if length > 0:
        before = slice(max(0, start-10), start)
        during = slice(start, start + length)
        after = slice(start + length, min(start + length + 10, len(df_anomaly)))
    else:
        before = slice(max(0, start-10), start)
        during = slice(start, start+1)
        after = slice(start+1, min(start + 11, len(df_anomaly)))

    print(f"\nGolden data:")
    print(f"  Before anomaly ({before}): mean={df_golden[sensor].iloc[before].mean():.2f}, std={df_golden[sensor].iloc[before].std():.2f}")
    print(f"  During anomaly ({during}): mean={df_golden[sensor].iloc[during].mean():.2f}, std={df_golden[sensor].iloc[during].std():.2f}")
    print(f"  After anomaly  ({after}):  mean={df_golden[sensor].iloc[after].mean():.2f}, std={df_golden[sensor].iloc[after].std():.2f}")

    print(f"\nAnomaly data:")
    print(f"  Before anomaly ({before}): mean={df_anomaly[sensor].iloc[before].mean():.2f}, std={df_anomaly[sensor].iloc[before].std():.2f}")
    print(f"  During anomaly ({during}): mean={df_anomaly[sensor].iloc[during].mean():.2f}, std={df_anomaly[sensor].iloc[during].std():.2f}")
    print(f"  After anomaly  ({after}):  mean={df_anomaly[sensor].iloc[after].mean():.2f}, std={df_anomaly[sensor].iloc[after].std():.2f}")

    # Calculate actual difference
    diff_during = abs(df_anomaly[sensor].iloc[during].mean() - df_golden[sensor].iloc[during].mean())
    print(f"\nActual difference during anomaly: {diff_during:.2f}°C")

    if diff_during < 1.0:
        print("  ⚠️  WARNING: Difference is very small!")
    elif diff_during < 5.0:
        print("  ⚠️  Difference is moderate")
    else:
        print("  ✓  Difference is large and detectable")

    return {
        'diff_during': diff_during,
        'golden_before': df_golden[sensor].iloc[before].mean(),
        'golden_during': df_golden[sensor].iloc[during].mean(),
        'anomaly_during': df_anomaly[sensor].iloc[during].mean(),
    }


def check_preprocessed_data_has_anomaly(anomaly_type):
    """
    Check if anomaly survives preprocessing (differencing)
    """
    print(f"\n{'='*80}")
    print(f"CHECKING PREPROCESSED DATA: {anomaly_type.upper()}")
    print(f"{'='*80}")

    # Find most recent preprocessed file
    pattern = f"results/no_mi_rolling_*/Anomaly/*{anomaly_type}*/preprocessing/*differenced*.csv"
    files = list(Path(".").glob(pattern))

    if not files:
        print("No preprocessed file found")
        return

    preproc_file = str(max(files, key=lambda p: p.stat().st_mtime))
    print(f"Using: {preproc_file}")

    df_anomaly_diff = pd.read_csv(preproc_file)

    # Find golden preprocessed
    golden_pattern = "results/Golden/preprocessing/*differenced*.csv"
    golden_files = list(Path(".").glob(golden_pattern))
    if not golden_files:
        print("No golden preprocessed found")
        return

    df_golden_diff = pd.read_csv(golden_files[0])

    sensor_diff = "Temperatur Exzenterlager links_diff"

    # Load metadata
    metadata_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    start = metadata['start']
    length = metadata.get('length', 0)

    # Note: differencing shifts indices by 1
    start_diff = start - 1

    if length > 0:
        during_diff = slice(start_diff, start_diff + length)
    else:
        during_diff = slice(start_diff, start_diff + 1)

    print(f"\nDifferenced data statistics:")
    print(f"  Golden during anomaly: mean={df_golden_diff[sensor_diff].iloc[during_diff].mean():.6f}, std={df_golden_diff[sensor_diff].iloc[during_diff].std():.6f}")
    print(f"  Anomaly during anomaly: mean={df_anomaly_diff[sensor_diff].iloc[during_diff].mean():.6f}, std={df_anomaly_diff[sensor_diff].iloc[during_diff].std():.6f}")

    diff_mean = abs(df_anomaly_diff[sensor_diff].iloc[during_diff].mean() - df_golden_diff[sensor_diff].iloc[during_diff].mean())
    diff_std = abs(df_anomaly_diff[sensor_diff].iloc[during_diff].std() - df_golden_diff[sensor_diff].iloc[during_diff].std())

    print(f"\n  Difference in mean: {diff_mean:.6f}")
    print(f"  Difference in std:  {diff_std:.6f}")

    if diff_mean < 0.001 and diff_std < 0.01:
        print("  ❌ PROBLEM: Anomaly has been eliminated by differencing!")
    elif diff_mean < 0.01 and diff_std < 0.05:
        print("  ⚠️  WARNING: Anomaly signal is very weak after differencing")
    else:
        print("  ✓  Anomaly signal survives differencing")


def compare_weight_statistics(anomaly_type):
    """
    Check the actual magnitude of weight differences
    """
    print(f"\n{'='*80}")
    print(f"WEIGHT DIFFERENCE STATISTICS: {anomaly_type.upper()}")
    print(f"{'='*80}")

    golden_file = "results/Golden/weights/weights_enhanced_20251006_154344.csv"

    # Find anomaly weight file
    pattern = f"results/no_mi_rolling_*/Anomaly/*{anomaly_type}*/weights/weights_enhanced*.csv"
    files = list(Path(".").glob(pattern))

    if not files:
        print("No weight file found")
        return

    anomaly_file = str(max(files, key=lambda p: p.stat().st_mtime))

    df_g = pd.read_csv(golden_file)
    df_a = pd.read_csv(anomaly_file)

    # Load metadata
    metadata_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    start = metadata['start']
    length = metadata.get('length', 0)

    # Expected window range
    if length > 0:
        expected_start = max(0, start - 110)
        expected_end = start + length
    else:
        expected_start = max(0, start - 110)
        expected_end = start

    print(f"\nExpected anomalous windows: {expected_start} to {expected_end}")

    # Merge and calculate differences
    merged = pd.merge(
        df_g[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        df_a[['window_idx', 'lag', 'parent_name', 'child_name', 'weight']],
        on=['window_idx', 'lag', 'parent_name', 'child_name'],
        how='inner',
        suffixes=('_g', '_a')
    )
    merged['abs_diff'] = np.abs(merged['weight_a'] - merged['weight_g'])

    # Compare expected region vs rest
    expected_region = merged[
        (merged['window_idx'] >= expected_start) &
        (merged['window_idx'] <= expected_end)
    ]
    rest_region = merged[
        (merged['window_idx'] < expected_start) |
        (merged['window_idx'] > expected_end)
    ]

    print(f"\nWeight differences in EXPECTED region ({expected_start}-{expected_end}):")
    print(f"  Mean: {expected_region['abs_diff'].mean():.6f}")
    print(f"  Max:  {expected_region['abs_diff'].max():.6f}")
    print(f"  Std:  {expected_region['abs_diff'].std():.6f}")

    print(f"\nWeight differences in REST of timeline:")
    print(f"  Mean: {rest_region['abs_diff'].mean():.6f}")
    print(f"  Max:  {rest_region['abs_diff'].max():.6f}")
    print(f"  Std:  {rest_region['abs_diff'].std():.6f}")

    ratio = expected_region['abs_diff'].mean() / rest_region['abs_diff'].mean()
    print(f"\n  Ratio (expected/rest): {ratio:.3f}")

    if ratio < 1.1:
        print("  ❌ PROBLEM: Expected region has NO HIGHER differences than baseline!")
    elif ratio < 1.5:
        print("  ⚠️  WARNING: Expected region has only slightly higher differences")
    else:
        print("  ✓  Expected region shows elevated differences")


def main():
    print("="*80)
    print("DIAGNOSING GRADUAL ANOMALY DETECTION FAILURE")
    print("="*80)

    anomaly_types = ['drift', 'trend_change', 'amplitude_change', 'variance_burst']

    for anomaly_type in anomaly_types:
        print(f"\n\n{'#'*80}")
        print(f"# {anomaly_type.upper()}")
        print(f"{'#'*80}")

        # Check 1: Raw data
        raw_result = check_raw_data_has_anomaly(anomaly_type)

        # Check 2: Preprocessed data
        check_preprocessed_data_has_anomaly(anomaly_type)

        # Check 3: Weight statistics
        compare_weight_statistics(anomaly_type)

    print(f"\n\n{'='*80}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*80}")
    print("""
Based on the checks above, the problem is likely one of:

1. Anomalies are eliminated by differencing
   → Gradual changes look like trends → differencing removes them
   → Solution: Use raw data or different preprocessing

2. Anomaly signal is too weak in differenced data
   → 0.5°C/row drift becomes 0.0005 after differencing
   → Solution: Increase anomaly magnitude or use non-stationary methods

3. DynoTEARS treats gradual changes as normal variance
   → 100-row drift is slower than natural fluctuations
   → Solution: Use longer windows or compare regions

Check the detailed output above to determine which is the case.
    """)


if __name__ == "__main__":
    main()
