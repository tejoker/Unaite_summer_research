#!/usr/bin/env python3
"""
Analyze window-by-window detection results to find FIRST anomaly window for each type.
This extracts the root cause onset from UnifiedAnomalyDetectionSuite results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def analyze_first_anomaly_window(summary_csv: Path, anomaly_type: str, ground_truth: int):
    """
    Find the first window where is_anomaly=True.

    Args:
        summary_csv: Path to window_by_window_summary.csv
        anomaly_type: Name of anomaly type
        ground_truth: Ground truth timepoint

    Returns:
        Dictionary with analysis results
    """
    if not summary_csv.exists():
        return {
            'anomaly_type': anomaly_type,
            'error': 'File not found',
            'file': str(summary_csv)
        }

    df = pd.read_csv(summary_csv)

    # Find first window with is_anomaly=True
    anomaly_windows = df[df['is_anomaly'] == True]

    if len(anomaly_windows) == 0:
        return {
            'anomaly_type': anomaly_type,
            'first_window_idx': None,
            'first_window_start': None,
            'first_window_end': None,
            'ground_truth': ground_truth,
            'error': 'No anomaly windows detected'
        }

    # Get first anomaly window
    first_row = anomaly_windows.iloc[0]

    first_idx = int(first_row['window_idx'])
    first_start = int(first_row['timepoint_start'])
    first_end = int(first_row['timepoint_end'])
    ensemble_score = float(first_row['ensemble_score'])
    classification = str(first_row['classification'])
    classification_conf = float(first_row['classification_confidence'])

    # Evaluation
    contains_gt = (first_start <= ground_truth <= first_end)
    start_error = abs(first_start - ground_truth)

    # Count total anomaly windows
    total_anomaly_windows = len(anomaly_windows)
    total_windows = len(df)

    return {
        'anomaly_type': anomaly_type,
        'first_window_idx': first_idx,
        'first_window_start': first_start,
        'first_window_end': first_end,
        'ground_truth': ground_truth,
        'start_error': start_error,
        'contains_gt': contains_gt,
        'ensemble_score': ensemble_score,
        'classification': classification,
        'classification_confidence': classification_conf,
        'total_anomaly_windows': total_anomaly_windows,
        'total_windows': total_windows,
        'anomaly_ratio': total_anomaly_windows / total_windows if total_windows > 0 else 0
    }


def main():
    results_base = Path("results/complete_zoom_20251016_143412")

    anomaly_types = [
        'drift', 'spike', 'variance_burst', 'level_shift',
        'amplitude_change', 'missing_block', 'trend_change'
    ]

    ground_truth = {
        'drift': 150,
        'spike': 200,
        'variance_burst': 150,
        'level_shift': 150,
        'amplitude_change': 150,
        'missing_block': 150,
        'trend_change': 150
    }

    print("="*80)
    print("FIRST ANOMALY WINDOW ANALYSIS")
    print("Window-by-Window Detection with UnifiedAnomalyDetectionSuite")
    print("="*80)
    print()

    results = []

    for anom_type in anomaly_types:
        summary_csv = results_base / f"window_detection_{anom_type}" / "window_by_window_summary.csv"
        gt = ground_truth[anom_type]

        print(f"Analyzing {anom_type}...")
        result = analyze_first_anomaly_window(summary_csv, anom_type, gt)
        results.append(result)

        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  First window: {result['first_window_idx']} "
                  f"(samples {result['first_window_start']}-{result['first_window_end']})")
            print(f"  Ground truth: {result['ground_truth']}")
            print(f"  Start error: {result['start_error']} samples")
            print(f"  Contains GT: {result['contains_gt']}")
            print(f"  Classification: {result['classification']} (conf={result['classification_confidence']:.2f})")
            print(f"  Anomaly windows: {result['total_anomaly_windows']}/{result['total_windows']} "
                  f"({result['anomaly_ratio']*100:.1f}%)")
        print()

    # Summary table
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Anomaly Type':<20} {'First Window':<15} {'Start Sample':<15} "
          f"{'GT':<8} {'Error':<10} {'Contains GT':<12}")
    print("-"*80)

    valid_results = [r for r in results if 'error' not in r]

    for r in valid_results:
        print(f"{r['anomaly_type']:<20} {r['first_window_idx']:<15} "
              f"{r['first_window_start']:<15} {r['ground_truth']:<8} "
              f"{r['start_error']:<10} {str(r['contains_gt']):<12}")

    # Overall statistics
    if valid_results:
        success_rate = sum(1 for r in valid_results if r['contains_gt']) / len(valid_results) * 100
        avg_error = np.mean([r['start_error'] for r in valid_results])
        avg_anomaly_ratio = np.mean([r['anomaly_ratio'] for r in valid_results]) * 100

        print()
        print("="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"Success rate (contains GT):  {success_rate:.1f}%")
        print(f"Average start error:         {avg_error:.1f} samples")
        print(f"Average anomaly ratio:       {avg_anomaly_ratio:.1f}%")
        print("="*80)

        # Save results
        results_df = pd.DataFrame(valid_results)
        output_path = results_base / "first_window_analysis.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    else:
        print("\nNo valid results found. Make sure window-by-window detection has been run.")


if __name__ == "__main__":
    main()
