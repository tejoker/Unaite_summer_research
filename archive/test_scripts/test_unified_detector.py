#!/usr/bin/env python3
"""
Test the unified anomaly detector on all 6 synthetic anomaly types
"""

import pandas as pd
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / 'executable' / 'final_pipeline' / 'anomaly_detection'))

from unified_anomaly_detector import UnifiedAnomalyDetector

def test_all_anomalies():
    """Test unified detector on all 6 anomaly types"""

    # Load golden baseline
    golden_raw = pd.read_csv('data/Golden/chunking/output_of_the_1th_chunk.csv', index_col=0)

    # Find golden differenced data
    golden_diff_files = list(Path('results').glob('*/Golden/*/preprocessing/*_differenced_stationary_series.csv'))
    if not golden_diff_files:
        golden_diff_files = list(Path('results/Golden/preprocessing').glob('*_differenced_stationary_series.csv'))

    if not golden_diff_files:
        print("ERROR: Could not find golden differenced data")
        return

    golden_diff = pd.read_csv(golden_diff_files[0], index_col=0)

    anomaly_types = ['spike', 'level_shift', 'drift', 'trend_change', 'amplitude_change', 'variance_burst']

    results_summary = []

    for anomaly_type in anomaly_types:
        print("\n" + "="*80)
        print(f"Testing: {anomaly_type}")
        print("="*80)

        # Load anomaly raw data
        anomaly_raw_file = f'data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.csv'

        try:
            anomaly_raw = pd.read_csv(anomaly_raw_file, index_col=0)
        except FileNotFoundError:
            print(f"  SKIP: {anomaly_raw_file} not found")
            continue

        # Find anomaly differenced data - use recursive glob
        anomaly_diff_pattern = f'**/*{anomaly_type}*_differenced_stationary_series.csv'
        anomaly_diff_files = [
            f for f in Path('results').rglob(anomaly_diff_pattern)
            if 'Anomaly' in str(f)
        ]

        if not anomaly_diff_files:
            print(f"  SKIP: No differenced data found for {anomaly_type}")
            continue

        # Use most recent file
        anomaly_diff_file = max(anomaly_diff_files, key=lambda p: p.stat().st_mtime)
        print(f"  Loading diff data from: {anomaly_diff_file.relative_to('results')}")
        anomaly_diff = pd.read_csv(anomaly_diff_file, index_col=0)

        # Find weight files (optional) - use recursive search
        golden_weight_files = list(Path('results').rglob('Golden/weights/weights_enhanced*.csv'))
        anomaly_weight_files = [
            f for f in Path('results').rglob('weights/weights_enhanced*.csv')
            if 'Anomaly' in str(f) and anomaly_type in str(f)
        ]

        golden_weights = None
        anomaly_weights = None

        if golden_weight_files and anomaly_weight_files:
            golden_weights = pd.read_csv(max(golden_weight_files, key=lambda p: p.stat().st_mtime))
            anomaly_weights = pd.read_csv(max(anomaly_weight_files, key=lambda p: p.stat().st_mtime))
            print(f"  Using weights: {len(golden_weights)} golden, {len(anomaly_weights)} anomaly edges")
        else:
            print(f"  No weights found - running without weight-based detection")

        # Run unified detector
        detector = UnifiedAnomalyDetector(
            double_diff_threshold=3.0,
            volatility_window=30,
            volatility_spike_threshold=3.0,
            volatility_changepoint_pen=1.5,
            weight_ratio_threshold=2.0
        )

        detections = detector.detect_all(
            raw_data=anomaly_raw,
            differenced_data=anomaly_diff,
            golden_weights=golden_weights,
            anomaly_weights=anomaly_weights
        )

        # Analyze results
        detected = len(detections) > 0
        methods_used = set([d.method for d in detections]) if detections else set()
        types_detected = set([d.anomaly_type for d in detections]) if detections else set()

        results_summary.append({
            'anomaly_type': anomaly_type,
            'detected': '✅' if detected else '❌',
            'n_detections': len(detections),
            'methods': ', '.join(methods_used),
            'detected_as': ', '.join(types_detected) if types_detected else 'N/A'
        })

        if detections:
            print(f"\n  ✅ DETECTED: {len(detections)} anomalies")
            for d in detections[:3]:  # Show first 3
                location = f"window {d.window_idx}" if d.window_idx else f"row {d.row_idx}"
                print(f"    - {d.method}: {d.anomaly_type} at {location} (confidence: {d.confidence:.2f})")
        else:
            print(f"\n  ❌ NOT DETECTED")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Unified Detector Performance")
    print("="*80)
    print(f"\n{'Anomaly Type':<20} {'Status':<10} {'# Det':<10} {'Methods Used':<30} {'Detected As':<20}")
    print("-"*90)

    for result in results_summary:
        print(f"{result['anomaly_type']:<20} {result['detected']:<10} {result['n_detections']:<10} "
              f"{result['methods']:<30} {result['detected_as']:<20}")

    total_detected = sum(1 for r in results_summary if r['detected'] == '✅')
    total_tested = len(results_summary)

    print("-"*90)
    if total_tested > 0:
        print(f"\nDetection Rate: {total_detected}/{total_tested} ({100*total_detected/total_tested:.0f}%)")
    else:
        print("\nNo anomalies were tested (check data availability)")
    print("="*80)


if __name__ == '__main__':
    test_all_anomalies()
