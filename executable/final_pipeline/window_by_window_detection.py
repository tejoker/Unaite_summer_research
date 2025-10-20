#!/usr/bin/env python3
"""
Window-by-Window Anomaly Detection

Compares baseline and anomaly weights window-by-window instead of using
averaged weights. This is critical for detecting localized anomalies like
spikes that occur at specific time points.

Usage:
    python window_by_window_detection.py \\
        --baseline-weights results/Golden_NoMI/weights/weights_enhanced.csv \\
        --anomaly-weights results/Spike_NoMI/weights/weights_enhanced.csv \\
        --output-dir results/window_analysis \\
        --anomaly-time 200
"""

import numpy as np
import pandas as pd
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'test' / 'anomaly_detection_suite'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'test'))

from final_pipeline.weight_corrector import (
    load_weights_from_csv, load_all_windows_from_csv, get_window_info
)

# Direct imports from anomaly detection suite
from anomaly_detection_suite import UnifiedAnomalyDetectionSuite

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_window_for_timepoint(timepoint: int, window_size: int, window_stride: int, start_idx: int = 0) -> int:
    """
    Find which window index contains a given timepoint.

    Args:
        timepoint: The time index to find
        window_size: Size of each rolling window
        window_stride: Stride between windows
        start_idx: Starting index (default 0)

    Returns:
        Window index that contains the timepoint
    """
    # Window i covers data from [start_idx + i*stride, start_idx + i*stride + window_size)
    # We want the window that has timepoint in its range
    if timepoint < start_idx:
        raise ValueError(f"Timepoint {timepoint} is before start index {start_idx}")

    window_idx = (timepoint - start_idx) // window_stride
    return window_idx


def _process_single_window(window_idx: int,
                           baseline_W: np.ndarray,
                           anomaly_W: np.ndarray,
                           window_size: int,
                           window_stride: int) -> Dict:
    """
    Process a single window comparison. Used for parallel processing.

    Args:
        window_idx: Window index
        baseline_W: Baseline weight matrix
        anomaly_W: Anomaly weight matrix
        window_size: Window size
        window_stride: Window stride

    Returns:
        Detection result dict
    """
    # Create detector instance (each process needs its own)
    detector = UnifiedAnomalyDetectionSuite()

    # Run detection
    result = detector.analyze_single_comparison(
        W_baseline=baseline_W,
        W_current=anomaly_W
    )

    # Add window metadata
    result['window_idx'] = window_idx
    result['window_timepoint_start'] = window_idx * window_stride
    result['window_timepoint_end'] = window_idx * window_stride + window_size

    return result


def detect_window_by_window(baseline_weights_path: str,
                            anomaly_weights_path: str,
                            output_dir: str,
                            anomaly_timepoint: Optional[int] = None,
                            window_size: int = 100,
                            window_stride: int = 1,
                            n_jobs: int = -1) -> Dict:
    """
    Perform window-by-window anomaly detection with parallel processing.

    Args:
        baseline_weights_path: Path to baseline weights CSV
        anomaly_weights_path: Path to anomaly weights CSV
        output_dir: Output directory for results
        anomaly_timepoint: If provided, focus on specific timepoint
        window_size: Rolling window size used in DynoTEARS
        window_stride: Stride between windows
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)

    Returns:
        Dict with detection results for all windows
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("WINDOW-BY-WINDOW ANOMALY DETECTION")
    logger.info("=" * 80)

    # Get window information
    baseline_info = get_window_info(baseline_weights_path)
    anomaly_info = get_window_info(anomaly_weights_path)

    logger.info(f"\nBaseline: {baseline_info['num_windows']} windows")
    logger.info(f"Anomaly:  {anomaly_info['num_windows']} windows")
    logger.info(f"Variables: {baseline_info['num_variables']}")
    logger.info(f"Max lag: {baseline_info['max_lag']}")

    # Load all windows
    logger.info("\nLoading all windows...")
    baseline_windows, baseline_indices = load_all_windows_from_csv(baseline_weights_path)
    anomaly_windows, anomaly_indices = load_all_windows_from_csv(anomaly_weights_path)

    # Results storage
    all_results = []
    anomalous_windows = []

    # Determine which windows to compare
    if anomaly_timepoint is not None:
        target_window = find_window_for_timepoint(
            anomaly_timepoint, window_size, window_stride
        )
        logger.info(f"\nFocusing on timepoint {anomaly_timepoint} -> window {target_window}")
        windows_to_check = [target_window] if target_window < len(anomaly_windows) else []
    else:
        # Compare all overlapping windows
        windows_to_check = list(range(min(len(baseline_windows), len(anomaly_windows))))

    # Determine number of workers
    # Use sequential mode for very few windows (overhead not worth it)
    if len(windows_to_check) <= 2:
        n_workers = 1
        logger.info(f"\nOnly {len(windows_to_check)} window(s) - using sequential mode")
    elif n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = min(n_jobs, cpu_count())

    logger.info(f"\nComparing {len(windows_to_check)} windows using {n_workers} workers...")
    logger.info("=" * 80)

    # Prepare data for parallel processing
    window_data = []
    for window_idx in windows_to_check:
        if window_idx >= len(baseline_windows) or window_idx >= len(anomaly_windows):
            logger.warning(f"Window {window_idx} out of range, skipping")
            continue
        window_data.append((
            window_idx,
            baseline_windows[window_idx]['W'],
            anomaly_windows[window_idx]['W'],
            window_size,
            window_stride
        ))

    # Run detection in parallel or sequential
    if n_workers > 1:
        logger.info("Running parallel detection...")
        with Pool(processes=n_workers) as pool:
            all_results = pool.starmap(_process_single_window, window_data)
    else:
        logger.info("Running sequential detection...")
        all_results = [_process_single_window(*args) for args in window_data]

    # Process results and display
    for result in all_results:
        window_idx = result['window_idx']
        is_anomaly = result['phase1_binary_detection']['binary_detection']['is_anomaly']
        ensemble_score = result['phase1_binary_detection']['binary_detection']['ensemble_score']

        if is_anomaly:
            anomalous_windows.append(window_idx)
            classification = result.get('phase2_classification', {})
            rule_based = classification.get('rule_based_result', {})
            anomaly_type = rule_based.get('prediction', 'unknown')
            confidence = rule_based.get('confidence', 0.0)

            logger.info(f"\n*** ANOMALY DETECTED in Window {window_idx} ***")
            logger.info(f"  Timepoint range: [{window_idx * window_stride}, {window_idx * window_stride + window_size})")
            logger.info(f"  Ensemble Score: {ensemble_score:.4f}")
            logger.info(f"  Classification: {anomaly_type} (confidence: {confidence:.2%})")

            # Root cause analysis
            root_cause = result.get('phase3_root_cause', {})
            if root_cause:
                top_edges = root_cause.get('top_changed_edges', [])[:5]
                logger.info(f"  Top changed edges:")
                for edge in top_edges:
                    logger.info(f"    {edge['from']} -> {edge['to']}: change={edge['change']:.4f}, importance={edge['importance']:.4f}")
        else:
            logger.info(f"Window {window_idx}: Normal (score={ensemble_score:.4f})")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total windows analyzed: {len(windows_to_check)}")
    logger.info(f"Anomalous windows: {len(anomalous_windows)}")
    if anomalous_windows:
        logger.info(f"Anomaly window indices: {anomalous_windows}")

        # Find highest scoring anomaly
        max_score_result = max(all_results, key=lambda x: x['phase1_binary_detection']['binary_detection']['ensemble_score'])
        max_window = max_score_result['window_idx']
        max_score = max_score_result['phase1_binary_detection']['binary_detection']['ensemble_score']
        logger.info(f"\nHighest anomaly score: {max_score:.4f} at window {max_window}")
        logger.info(f"  Timepoint range: [{max_window * window_stride}, {max_window * window_stride + window_size})")

    # Save detailed results
    results_file = output_path / "window_by_window_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'baseline_weights': baseline_weights_path,
                'anomaly_weights': anomaly_weights_path,
                'num_windows_analyzed': len(windows_to_check),
                'anomalous_windows': anomalous_windows,
                'window_size': window_size,
                'window_stride': window_stride
            },
            'window_results': all_results
        }, f, indent=2)

    logger.info(f"\nDetailed results saved to: {results_file}")

    # Save summary CSV
    summary_data = []
    for result in all_results:
        window_idx = result['window_idx']
        detection = result['phase1_binary_detection']['binary_detection']
        classification = result.get('phase2_classification', {}).get('rule_based_result', {})

        summary_data.append({
            'window_idx': window_idx,
            'timepoint_start': window_idx * window_stride,
            'timepoint_end': window_idx * window_stride + window_size,
            'is_anomaly': detection['is_anomaly'],
            'ensemble_score': detection['ensemble_score'],
            'frobenius': detection['metrics_raw']['frobenius_distance'],
            'structural_hamming': detection['metrics_raw']['structural_hamming_distance'],
            'spectral': detection['metrics_raw']['spectral_distance'],
            'max_edge_change': detection['metrics_raw']['max_edge_change'],
            'classification': classification.get('prediction', 'normal'),
            'classification_confidence': classification.get('confidence', 0.0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "window_by_window_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary CSV saved to: {summary_file}")

    return {
        'all_results': all_results,
        'anomalous_windows': anomalous_windows,
        'summary_df': summary_df
    }


def main():
    parser = argparse.ArgumentParser(description='Window-by-window anomaly detection with parallel processing')
    parser.add_argument('--baseline-weights', required=True, help='Path to baseline weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Path to anomaly weights CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--anomaly-time', type=int, help='Focus on specific timepoint (optional)')
    parser.add_argument('--window-size', type=int, default=100, help='Rolling window size')
    parser.add_argument('--window-stride', type=int, default=1, help='Window stride')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1=all CPUs, 1=sequential)')

    args = parser.parse_args()

    detect_window_by_window(
        baseline_weights_path=args.baseline_weights,
        anomaly_weights_path=args.anomaly_weights,
        output_dir=args.output_dir,
        anomaly_timepoint=args.anomaly_time,
        window_size=args.window_size,
        window_stride=args.window_stride,
        n_jobs=args.n_jobs
    )


if __name__ == '__main__':
    main()
