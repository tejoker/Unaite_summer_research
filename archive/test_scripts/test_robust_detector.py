#!/usr/bin/env python3
"""
Test the robust weight-based detector on all 6 synthetic anomalies
with validation against ground truth metadata
"""

import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'executable' / 'final_pipeline' / 'anomaly_detection'))

from robust_weight_detector import RobustWeightDetector


def load_ground_truth(anomaly_type: str) -> dict:
    """Load ground truth from JSON metadata"""
    json_file = Path(f'data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json')

    if not json_file.exists():
        return None

    with open(json_file, 'r') as f:
        metadata = json.load(f)

    return {
        'sensor': metadata['ts_col'],
        'anomaly': metadata['anomaly'],
        'start_row': metadata['start'],
        'length': metadata['length'],
        'end_row': metadata['start'] + metadata.get('length', 0),
        'magnitude': metadata['magnitude']
    }


def validate_detection(detection, ground_truth, window_size=100, window_overlap=90) -> dict:
    """
    Validate if detection matches ground truth

    Returns dict with validation results
    """
    if not detection or not ground_truth:
        return {'valid': False, 'reason': 'Missing data'}

    # Check if epicenter matches ground truth sensor
    gt_sensor = ground_truth['sensor']
    detected_sensor = detection.epicenter_sensor.replace('_diff', '')

    sensor_match = gt_sensor == detected_sensor

    # Check if detection time range overlaps with ground truth
    gt_start = ground_truth['start_row']
    gt_end = ground_truth['end_row']

    det_start, det_end = detection.window_time_range

    # Allow some tolerance due to window alignment
    time_overlap = not (det_end < gt_start - 20 or det_start > gt_end + 20)

    # Calculate expected window indices for ground truth anomaly
    # Window i starts at row i * (window_size - window_overlap)
    stride = window_size - window_overlap
    expected_windows = set()

    for row in range(gt_start, gt_end + 1):
        window_idx = row // stride
        expected_windows.add(window_idx)

    detected_window = detection.window_idx
    window_match = detected_window in expected_windows

    is_valid = sensor_match and (time_overlap or window_match)

    result = {
        'valid': is_valid,
        'sensor_match': sensor_match,
        'time_overlap': time_overlap,
        'window_match': window_match,
        'detected_sensor': detected_sensor,
        'expected_sensor': gt_sensor,
        'detected_window': detected_window,
        'expected_windows': sorted(expected_windows),
        'detected_time_range': (det_start, det_end),
        'expected_time_range': (gt_start, gt_end)
    }

    if not is_valid:
        if not sensor_match:
            result['reason'] = f"Wrong sensor: detected '{detected_sensor}', expected '{gt_sensor}'"
        elif not time_overlap and not window_match:
            result['reason'] = f"Wrong timing: window {detected_window} not in expected {sorted(expected_windows)}"

    return result


def test_all_anomalies():
    """Test robust detector on all 6 anomaly types"""

    # Find golden weights - need two runs of SAME settings for noise calculation
    all_golden_files = list(Path('results').rglob('weights_enhanced*.csv'))
    # Look for Golden runs (either with 1th_chunk in path, or Golden_no_mi_run in path)
    golden_files = [f for f in all_golden_files
                    if 'Golden' in str(f) and ('1th_chunk' in str(f) or 'Golden_no_mi_run' in str(f))]

    if len(golden_files) < 1:
        print("ERROR: Need at least 1 golden weight file")
        print(f"Searched in: {Path('results').absolute()}")
        return

    # Sort by modification time
    golden_files = sorted(golden_files, key=lambda p: p.stat().st_mtime, reverse=True)

    golden_weights_1 = golden_files[0]
    print(f"Using Golden weights 1: {golden_weights_1.relative_to('results')}")

    # Check if we have two golden runs for noise calculation
    golden_weights_2 = None
    if len(golden_files) >= 2:
        # Get second run
        golden_weights_2 = golden_files[1]
        print(f"Using Golden weights 2: {golden_weights_2.relative_to('results')}")
    else:
        print("WARNING: Only 1 golden run found - will use default threshold")

    golden_df_1 = pd.read_csv(golden_weights_1)
    golden_df_2 = pd.read_csv(golden_weights_2) if golden_weights_2 else None

    # Calculate baseline noise if we have 2 golden runs
    baseline_mean = None
    baseline_std = None

    if golden_df_2 is not None:
        print("\n" + "="*100)
        print("CALCULATING BASELINE NOISE FROM GOLDEN-GOLDEN COMPARISON")
        print("="*100)
        baseline_mean, baseline_std = RobustWeightDetector.calculate_baseline_noise(
            golden_df_1, golden_df_2
        )

    # Create detector
    detector = RobustWeightDetector(
        baseline_noise_mean=baseline_mean,
        baseline_noise_std=baseline_std,
        sigma_multiplier=5.0,
        min_epicenter_impact=10.0
    )

    anomaly_types = ['spike', 'level_shift', 'drift', 'trend_change', 'amplitude_change', 'variance_burst']
    results_summary = []

    for anomaly_type in anomaly_types:
        print("\n" + "="*100)
        print(f"Testing: {anomaly_type.upper()}")
        print("="*100)

        # Load ground truth
        ground_truth = load_ground_truth(anomaly_type)

        if ground_truth is None:
            print(f"  SKIP: No ground truth metadata found")
            continue

        print(f"\nGround Truth:")
        print(f"  Sensor: {ground_truth['sensor']}")
        print(f"  Time Range: rows {ground_truth['start_row']}-{ground_truth['end_row']}")
        print(f"  Magnitude: {ground_truth['magnitude']}")

        # Find anomaly weights
        anomaly_weight_pattern = f'**/Anomaly/**/*{anomaly_type}*/weights/weights_enhanced*.csv'
        anomaly_weight_files = list(Path('results').glob(anomaly_weight_pattern))

        if not anomaly_weight_files:
            print(f"  SKIP: No weight files found")
            results_summary.append({
                'anomaly_type': anomaly_type,
                'status': '⚠️ SKIP',
                'reason': 'No weight files'
            })
            continue

        anomaly_weights_file = max(anomaly_weight_files, key=lambda p: p.stat().st_mtime)
        print(f"\nUsing anomaly weights: {anomaly_weights_file.relative_to('results').parent.name}")

        anomaly_df = pd.read_csv(anomaly_weights_file)

        # Run detection
        print(f"\nRunning robust detection...")
        detections = detector.detect_anomalies(
            golden_weights=golden_df_1,
            anomaly_weights=anomaly_df,
            window_size=100,
            window_overlap=90
        )

        # Validate results
        if len(detections) == 0:
            print(f"\n❌ FAILED: No detections")
            results_summary.append({
                'anomaly_type': anomaly_type,
                'status': '❌ MISS',
                'reason': 'No detections above threshold',
                'n_detections': 0
            })
            continue

        print(f"\n✓ Found {len(detections)} detection(s)")

        # Find best matching detection
        best_match = None
        best_validation = None

        for detection in detections:
            validation = validate_detection(detection, ground_truth)
            if validation['valid']:
                if best_match is None or detection.epicenter_impact_score > best_match.epicenter_impact_score:
                    best_match = detection
                    best_validation = validation

        if best_match:
            print(f"\n✅ SUCCESS: Valid detection found")
            print(f"   Window: {best_match.window_idx}")
            print(f"   Epicenter: {best_match.epicenter_sensor}")
            print(f"   Type: {best_match.anomaly_type}")
            print(f"   Impact Score: {best_match.epicenter_impact_score:.2f}")
            print(f"   Max Weight Change: {best_match.max_weight_change:.4f}")
            print(f"\n   {best_match.root_cause_summary}")

            results_summary.append({
                'anomaly_type': anomaly_type,
                'status': '✅ PASS',
                'n_detections': len(detections),
                'window': best_match.window_idx,
                'epicenter': best_match.epicenter_sensor,
                'impact_score': f"{best_match.epicenter_impact_score:.1f}"
            })
        else:
            print(f"\n❌ FAILED: Detection(s) found but none match ground truth")
            print(f"\n   Detected windows: {[d.window_idx for d in detections]}")
            print(f"   Expected windows: {validation['expected_windows']}")
            print(f"   Detected sensors: {[d.epicenter_sensor for d in detections]}")
            print(f"   Expected sensor: {ground_truth['sensor']}")

            results_summary.append({
                'anomaly_type': anomaly_type,
                'status': '❌ FALSE',
                'reason': 'Wrong location/sensor',
                'n_detections': len(detections)
            })

    # Print summary table
    print("\n" + "="*100)
    print("VALIDATION SUMMARY: Robust Weight-Based Detector")
    print("="*100)
    print(f"\n{'Anomaly Type':<20} {'Status':<12} {'# Det':<8} {'Details':<50}")
    print("-"*100)

    for result in results_summary:
        details = ""
        if result['status'] == '✅ PASS':
            details = f"Window {result['window']}, {result['epicenter'].replace('_diff', '')}, Score: {result['impact_score']}"
        elif 'reason' in result:
            details = result['reason']

        n_det = result.get('n_detections', '-')

        print(f"{result['anomaly_type']:<20} {result['status']:<12} {n_det:<8} {details:<50}")

    passed = sum(1 for r in results_summary if r['status'] == '✅ PASS')
    total = len(results_summary)

    print("-"*100)
    print(f"\nValidation Rate: {passed}/{total} ({100*passed/total if total > 0 else 0:.0f}%)")
    print("="*100)


if __name__ == '__main__':
    test_all_anomalies()
