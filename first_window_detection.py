#!/usr/bin/env python3
"""
First Window Detection - Extract the FIRST window where anomaly appears
This helps identify the root cause onset point.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / 'executable' / 'final_pipeline'))

from weight_corrector import load_all_windows_from_csv


def detect_first_anomaly_window(golden_weights_path: Path,
                                 anomaly_weights_path: Path,
                                 threshold_percentile: float = 95.0) -> Dict:
    """
    Find the FIRST window where anomaly appears by comparing edge weights.

    Args:
        golden_weights_path: Path to golden baseline weights CSV
        anomaly_weights_path: Path to anomaly weights CSV
        threshold_percentile: Percentile for anomaly threshold

    Returns:
        Dictionary with first anomaly window info
    """
    print(f"\nLoading golden weights from: {golden_weights_path}")
    golden_windows, golden_indices = load_all_windows_from_csv(golden_weights_path)

    print(f"Loading anomaly weights from: {anomaly_weights_path}")
    anomaly_windows, anomaly_indices = load_all_windows_from_csv(anomaly_weights_path)

    num_windows = len(golden_windows)
    print(f"Total windows: {num_windows}")

    # Compute edge differences for each window
    window_differences = []

    for w_idx in range(num_windows):
        g_data = golden_windows[w_idx]
        a_data = anomaly_windows[w_idx]

        # Extract W (instantaneous) and A_list (lag) matrices
        # Format: {'W': np.ndarray, 'A_list': List[np.ndarray]}
        g_W = g_data['W']
        a_W = a_data['W']
        g_A_list = g_data['A_list']
        a_A_list = a_data['A_list']

        # Compute differences for instantaneous weights
        abs_diffs = []
        d = g_W.shape[0]

        for i in range(d):
            for j in range(d):
                g_val = g_W[i, j]
                a_val = a_W[i, j]
                abs_diffs.append(abs(a_val - g_val))

        # Add lag weight differences
        max_lag = min(len(g_A_list), len(a_A_list))
        for lag_idx in range(max_lag):
            g_A = g_A_list[lag_idx]
            a_A = a_A_list[lag_idx]
            for i in range(d):
                for j in range(d):
                    g_val = g_A[i, j]
                    a_val = a_A[i, j]
                    abs_diffs.append(abs(a_val - g_val))

        # Use mean absolute difference as window score
        mean_diff = np.mean(abs_diffs) if abs_diffs else 0.0
        max_diff = np.max(abs_diffs) if abs_diffs else 0.0

        window_differences.append({
            'window_idx': w_idx,
            'mean_abs_diff': mean_diff,
            'max_abs_diff': max_diff,
            'num_edges': len(abs_diffs)
        })

    # Compute threshold based on percentile of mean differences
    mean_diffs = [w['mean_abs_diff'] for w in window_differences]
    threshold = np.percentile(mean_diffs, threshold_percentile)

    print(f"\nAnomaly threshold (p={threshold_percentile}): {threshold:.4f}")
    print(f"Mean diff range: [{np.min(mean_diffs):.4f}, {np.max(mean_diffs):.4f}]")

    # Find FIRST window exceeding threshold
    first_anomaly_idx = None
    for i, w in enumerate(window_differences):
        if w['mean_abs_diff'] > threshold:
            first_anomaly_idx = i
            break

    if first_anomaly_idx is None:
        print("WARNING: No anomaly windows detected above threshold!")
        # Use window with maximum difference
        first_anomaly_idx = max(range(len(window_differences)),
                                key=lambda i: window_differences[i]['mean_abs_diff'])
        print(f"Using window with maximum difference instead: {first_anomaly_idx}")

    result = {
        'first_anomaly_window': first_anomaly_idx,
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile,
        'window_stats': window_differences[first_anomaly_idx],
        'total_windows_above_threshold': sum(1 for w in window_differences if w['mean_abs_diff'] > threshold),
        'all_window_diffs': window_differences
    }

    return result


def run_all_anomalies(results_base: Path, golden_dir: Path):
    """
    Run first window detection on all anomaly types from zoom analysis.
    """
    anomaly_types = [
        'drift', 'spike', 'variance_burst', 'level_shift',
        'amplitude_change', 'missing_block', 'trend_change'
    ]

    # Ground truth data
    ground_truth = {
        'drift': (150, 250),
        'spike': (200, 200),
        'variance_burst': (150, 250),
        'level_shift': (150, 150),  # Permanent shift starting at 150
        'amplitude_change': (150, 250),
        'missing_block': (150, 200),
        'trend_change': (150, 250)
    }

    print("="*80)
    print("FIRST WINDOW DETECTION - Root Cause Localization")
    print("="*80)

    results_summary = []

    for anom_type in anomaly_types:
        print(f"\n{'='*80}")
        print(f"ANOMALY TYPE: {anom_type.upper()}")
        print(f"{'='*80}")

        # Find weights files
        anomaly_dir = results_base / f"anomaly_{anom_type}"

        golden_weights = golden_dir / "weights" / "weights_enhanced.csv"
        anomaly_weights = anomaly_dir / "weights" / "weights_enhanced.csv"

        if not golden_weights.exists():
            print(f"ERROR: Golden weights not found: {golden_weights}")
            continue

        if not anomaly_weights.exists():
            print(f"ERROR: Anomaly weights not found: {anomaly_weights}")
            continue

        # Run detection
        result = detect_first_anomaly_window(golden_weights, anomaly_weights)

        first_window = result['first_anomaly_window']
        stats = result['window_stats']

        gt_start, gt_end = ground_truth[anom_type]

        print(f"\n{'─'*80}")
        print(f"RESULTS:")
        print(f"{'─'*80}")
        print(f"First anomaly window:     {first_window}")
        print(f"Mean abs difference:      {stats['mean_abs_diff']:.4f}")
        print(f"Max abs difference:       {stats['max_abs_diff']:.4f}")
        print(f"Windows above threshold:  {result['total_windows_above_threshold']}")
        print(f"\nGround truth range:       [{gt_start}, {gt_end}]")

        # Evaluation
        contains_gt = (first_window <= gt_start <= first_window + 100) or \
                     (first_window <= gt_end <= first_window + 100)
        start_error = abs(first_window - gt_start)

        print(f"\nEVALUATION:")
        print(f"Contains ground truth:    {contains_gt}")
        print(f"Start error (samples):    {start_error}")
        print(f"Relative error:           {start_error / gt_start * 100:.1f}%")

        results_summary.append({
            'anomaly_type': anom_type,
            'first_window': first_window,
            'ground_truth_start': gt_start,
            'ground_truth_end': gt_end,
            'start_error': start_error,
            'contains_gt': contains_gt,
            'mean_diff': stats['mean_abs_diff'],
            'max_diff': stats['max_abs_diff'],
            'threshold': result['threshold'],
            'windows_above_threshold': result['total_windows_above_threshold']
        })

        # Save detailed results
        output_dir = results_base / f"first_window_{anom_type}"
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "first_window_results.json", 'w') as f:
            json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)

        print(f"\nDetailed results saved to: {output_dir / 'first_window_results.json'}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY - ALL ANOMALIES")
    print(f"{'='*80}")
    print(f"{'Anomaly Type':<20} {'First Window':<15} {'GT Start':<12} {'Error':<12} {'Contains GT':<15}")
    print(f"{'─'*80}")

    for r in results_summary:
        print(f"{r['anomaly_type']:<20} {r['first_window']:<15} {r['ground_truth_start']:<12} "
              f"{r['start_error']:<12} {str(r['contains_gt']):<15}")

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = results_base / "first_window_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Overall statistics
    if len(results_summary) > 0:
        success_rate = sum(1 for r in results_summary if r['contains_gt']) / len(results_summary) * 100
        avg_error = np.mean([r['start_error'] for r in results_summary])
    else:
        success_rate = 0.0
        avg_error = 0.0

    print(f"\n{'='*80}")
    print(f"OVERALL PERFORMANCE:")
    print(f"{'='*80}")
    print(f"Success rate (contains GT): {success_rate:.1f}%")
    print(f"Average start error:        {avg_error:.1f} samples")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="First Window Anomaly Detection")
    parser.add_argument('--results-base', type=str,
                        default='results/complete_zoom_20251016_143412',
                        help='Base directory with all anomaly results')
    parser.add_argument('--golden-dir', type=str,
                        default='results/complete_zoom_20251016_143412/golden',
                        help='Golden baseline directory')
    parser.add_argument('--threshold-percentile', type=float, default=95.0,
                        help='Percentile threshold for anomaly detection (default: 95)')

    args = parser.parse_args()

    results_base = Path(args.results_base)
    golden_dir = Path(args.golden_dir)

    if not results_base.exists():
        print(f"ERROR: Results base directory not found: {results_base}")
        sys.exit(1)

    if not golden_dir.exists():
        print(f"ERROR: Golden directory not found: {golden_dir}")
        sys.exit(1)

    run_all_anomalies(results_base, golden_dir)
