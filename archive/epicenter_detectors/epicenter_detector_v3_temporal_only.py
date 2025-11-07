#!/usr/bin/env python3
"""
Epicenter-Based Anomaly Detection V3 - Temporal-Only Strategy

Key insight: In causal graphs, anomaly SOURCE appears EARLIEST, not necessarily STRONGEST.
Hub nodes amplify signals, making peripheral anomaly sources look weaker in magnitude.

Strategy: Pure temporal reasoning - find sensor with earliest significant weight changes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse


def load_weights(weight_file):
    """Load weights file and return as DataFrame"""
    df = pd.read_csv(weight_file)
    required = {'window_idx', 'child_name', 'parent_name', 'lag', 'weight'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected {required}, got {df.columns}")
    return df


def calculate_weight_differences(golden_weights, anomaly_weights):
    """Calculate weight differences between Golden and Anomaly runs."""
    print(f"Golden: {len(golden_weights)} edges, {golden_weights['window_idx'].nunique()} windows")
    print(f"Anomaly: {len(anomaly_weights)} edges, {anomaly_weights['window_idx'].nunique()} windows")

    merged = pd.merge(
        golden_weights,
        anomaly_weights,
        on=['window_idx', 'child_name', 'parent_name', 'lag'],
        suffixes=('_golden', '_anomaly'),
        how='outer'
    )

    merged['weight_golden'] = merged['weight_golden'].fillna(0)
    merged['weight_anomaly'] = merged['weight_anomaly'].fillna(0)
    merged['weight_diff'] = abs(merged['weight_anomaly'] - merged['weight_golden'])

    return merged


def identify_epicenter_temporal(weight_diffs, significance_threshold_percentile=75):
    """
    Pure temporal epicenter identification.

    Strategy:
    1. For each sensor, find its EARLIEST window with significant weight change
    2. Significance = above 75th percentile of all weight changes for that sensor
    3. The sensor with the EARLIEST significant change is the epicenter

    Rationale: Anomaly source manifests first, propagates to connected sensors later.
    """
    if weight_diffs.empty:
        return [], 0.0, {}

    # Get all sensors
    all_sensors = set(weight_diffs['child_name'].unique()) | set(weight_diffs['parent_name'].unique())

    earliest_significant_windows = {}
    sensor_stats = {}

    for sensor in all_sensors:
        # Get all edges involving this sensor
        mask = (weight_diffs['child_name'] == sensor) | (weight_diffs['parent_name'] == sensor)
        sensor_edges = weight_diffs[mask]

        if sensor_edges.empty or len(sensor_edges[sensor_edges['weight_diff'] > 0]) == 0:
            continue

        # Calculate significance threshold for this sensor
        threshold = np.percentile(sensor_edges['weight_diff'].values, significance_threshold_percentile)

        # Find significant changes
        significant_changes = sensor_edges[sensor_edges['weight_diff'] >= threshold]

        if not significant_changes.empty:
            # Find EARLIEST window with significant change
            earliest_window = significant_changes['window_idx'].min()
            earliest_significant_windows[sensor] = earliest_window

            # Collect stats
            sensor_stats[sensor] = {
                'earliest_window': earliest_window,
                'threshold': threshold,
                'n_significant_changes': len(significant_changes),
                'total_impact': sensor_edges['weight_diff'].sum(),
                'max_change': sensor_edges['weight_diff'].max()
            }

    if not earliest_significant_windows:
        return [], 0.0, {}

    # Find sensor(s) with earliest significant window
    min_window = min(earliest_significant_windows.values())
    epicenter_candidates = [s for s, w in earliest_significant_windows.items() if w == min_window]

    # If multiple sensors tie for earliest window, use total_impact as tie-breaker
    if len(epicenter_candidates) > 1:
        impacts = {s: sensor_stats[s]['total_impact'] for s in epicenter_candidates}
        epicenter = max(impacts, key=impacts.get)
        tie_breaker = 'total_impact'
    else:
        epicenter = epicenter_candidates[0]
        tie_breaker = 'temporal_only'

    # Calculate confidence based on temporal separation
    all_windows = sorted(earliest_significant_windows.values())
    if len(all_windows) > 1:
        separation = all_windows[1] - all_windows[0]
        # Confidence = how many windows separate #1 from #2
        confidence = min(1.0, separation / 10.0)  # 10 windows apart = 100% confidence
    else:
        confidence = 1.0

    candidates_info = {
        'epicenter': epicenter,
        'earliest_window': min_window,
        'tie_breaker': tie_breaker,
        'all_earliest_windows': earliest_significant_windows,
        'sensor_stats': sensor_stats,
        'confidence': confidence
    }

    return [epicenter], confidence, candidates_info


def filter_epicenter_edges(weight_diffs, epicenter_sensors, min_diff_threshold=0.0):
    """Filter weight differences to only include edges involving epicenter sensors."""
    mask = (
        weight_diffs['child_name'].isin(epicenter_sensors) |
        weight_diffs['parent_name'].isin(epicenter_sensors)
    )

    filtered = weight_diffs[mask & (weight_diffs['weight_diff'] > min_diff_threshold)]
    return filtered.sort_values('weight_diff', ascending=False)


def detect_anomaly_windows(epicenter_edges, threshold_percentile=95):
    """Identify which windows contain significant anomalies."""
    window_changes = epicenter_edges.groupby('window_idx')['weight_diff'].sum().sort_values(ascending=False)
    threshold = np.percentile(window_changes.values, threshold_percentile)
    anomalous_windows = window_changes[window_changes > threshold].index.tolist()

    return anomalous_windows, window_changes


def generate_report(epicenter_sensors, epicenter_edges, anomalous_windows,
                    confidence=None, candidates_info=None, ground_truth_file=None):
    """Generate human-readable anomaly detection report"""

    print("\n" + "="*100)
    print("EPICENTER-BASED ANOMALY DETECTION REPORT (V3 - Temporal-Only)")
    print("="*100)

    print("\n1. TEMPORAL ANALYSIS:")
    print("-"*100)
    if candidates_info and 'sensor_stats' in candidates_info:
        # Sort sensors by earliest window
        stats_df = pd.DataFrame(candidates_info['sensor_stats']).T
        stats_df = stats_df.sort_values('earliest_window')
        print("\nTop 10 Sensors by Earliest Significant Activity:")
        print(stats_df[['earliest_window', 'n_significant_changes', 'total_impact', 'max_change']].head(10).to_string())

    print(f"\n2. IDENTIFIED EPICENTER(S): {epicenter_sensors}")
    print("-"*100)
    if confidence is not None:
        print(f"  Confidence: {confidence:.2%}")
    if candidates_info:
        print(f"  Earliest significant window: {candidates_info.get('earliest_window', 'N/A')}")
        print(f"  Tie-breaker used: {candidates_info.get('tie_breaker', 'N/A')}")

        if epicenter_sensors and 'sensor_stats' in candidates_info:
            epicenter = epicenter_sensors[0]
            if epicenter in candidates_info['sensor_stats']:
                stats = candidates_info['sensor_stats'][epicenter]
                print(f"\n  Epicenter Statistics:")
                print(f"    Earliest window: {stats['earliest_window']}")
                print(f"    Significant changes: {stats['n_significant_changes']}")
                print(f"    Total impact: {stats['total_impact']:.4f}")
                print(f"    Max single change: {stats['max_change']:.4f}")

    print(f"\n3. EPICENTER EDGES:")
    print("-"*100)
    print(f"Total edges involving epicenter: {len(epicenter_edges)}")

    if len(epicenter_edges) > 0:
        print(f"\nTop 10 largest weight changes:")
        print(epicenter_edges[['window_idx', 'child_name', 'parent_name', 'lag', 'weight_diff']].head(10).to_string(index=False))

    print(f"\n4. ANOMALOUS WINDOWS:")
    print("-"*100)
    print(f"Detected {len(anomalous_windows)} anomalous windows: {sorted(anomalous_windows)}")

    # Validation
    if ground_truth_file and Path(ground_truth_file).exists():
        print(f"\n5. VALIDATION AGAINST GROUND TRUTH:")
        print("-"*100)

        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

        gt_sensor = ground_truth.get('ts_col', ground_truth.get('affected_sensor', 'Unknown'))
        gt_start = ground_truth.get('start', ground_truth.get('anomaly_start_row', 0))
        gt_length = ground_truth.get('length', 1)
        gt_end = gt_start + gt_length if gt_length > 0 else gt_start + 1
        gt_type = ground_truth.get('anomaly', ground_truth.get('anomaly_type', 'Unknown'))

        print(f"Ground Truth:")
        print(f"  Affected Sensor: {gt_sensor}")
        print(f"  Anomaly Type: {gt_type}")
        print(f"  Rows: {gt_start}-{gt_end}")

        epicenter_match = any(
            gt_sensor in sensor or
            sensor.replace('_diff', '') == gt_sensor or
            gt_sensor.replace(' ', '_').lower() in sensor.lower()
            for sensor in epicenter_sensors
        )
        print(f"\nEpicenter Detection: {'✅ CORRECT' if epicenter_match else '❌ INCORRECT'}")
        print(f"  Detected: {epicenter_sensors}")
        print(f"  Expected: {gt_sensor} (or variants with _diff suffix)")

        return epicenter_match

    return None


def main():
    parser = argparse.ArgumentParser(description='Epicenter-based anomaly detection V3 (temporal-only)')
    parser.add_argument('--golden', required=True, help='Golden weights file')
    parser.add_argument('--anomaly', required=True, help='Anomaly weights file')
    parser.add_argument('--ground-truth', help='Ground truth JSON file (optional)')
    parser.add_argument('--output', help='Output CSV file for detected anomalies')
    parser.add_argument('--min-diff', type=float, default=0.0, help='Minimum weight difference threshold')

    args = parser.parse_args()

    print("="*100)
    print("LOADING DATA")
    print("="*100)
    print(f"Golden: {args.golden}")
    print(f"Anomaly: {args.anomaly}")

    # Load weights
    golden_weights = load_weights(args.golden)
    anomaly_weights = load_weights(args.anomaly)

    # Calculate weight differences
    print("\n" + "="*100)
    print("CALCULATING WEIGHT DIFFERENCES")
    print("="*100)
    weight_diffs = calculate_weight_differences(golden_weights, anomaly_weights)
    print(f"Total edges compared: {len(weight_diffs)}")
    print(f"Edges with changes: {len(weight_diffs[weight_diffs['weight_diff'] > 0])}")

    # Identify epicenter using temporal-only approach
    print("\n" + "="*100)
    print("TEMPORAL EPICENTER IDENTIFICATION")
    print("="*100)
    epicenter_sensors, confidence, candidates_info = identify_epicenter_temporal(weight_diffs)
    print(f"Detected {len(epicenter_sensors)} epicenter(s) with {confidence:.1%} confidence")

    # Filter epicenter edges
    epicenter_edges = filter_epicenter_edges(weight_diffs, epicenter_sensors, args.min_diff)

    # Detect anomalous windows
    anomalous_windows, window_changes = detect_anomaly_windows(epicenter_edges)

    # Generate report
    match = generate_report(epicenter_sensors, epicenter_edges,
                           anomalous_windows, confidence, candidates_info, args.ground_truth)

    # Save results
    if args.output:
        epicenter_edges.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "="*100)
    print("DETECTION COMPLETE")
    print("="*100)

    return 0 if match is None or match else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
