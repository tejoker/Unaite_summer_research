#!/usr/bin/env python3
"""
Evaluate Anomaly Detection Performance

This script evaluates the dual-metric anomaly detection results against
the labeled anomalies from the NASA Telemanom dataset.

Metrics:
- Precision: % of detected anomalies that are true positives
- Recall: % of labeled anomalies that were detected
- F1-score: Harmonic mean of precision and recall
- Point-wise metrics (strict matching)
- Range-based metrics (overlap-based matching)

Reference:
- Hundman et al. 2018 "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_labeled_anomalies(labels_csv: str) -> List[Tuple[int, int, str]]:
    """
    Load labeled anomalies from Telemanom labeled_anomalies.csv.

    Args:
        labels_csv: Path to labeled_anomalies.csv

    Returns:
        List of (start, end, channel) tuples
    """
    df = pd.read_csv(labels_csv)

    # Parse anomaly sequences
    anomalies = []

    for _, row in df.iterrows():
        chan_id = row['chan_id']

        # Parse anomaly_sequences - format: [[start1, end1], [start2, end2], ...]
        sequences_str = row['anomaly_sequences']

        # Clean up the string and parse
        sequences_str = sequences_str.replace('[', '').replace(']', '')
        if not sequences_str.strip():
            continue

        # Split by comma and group pairs
        values = [int(x.strip()) for x in sequences_str.split(',') if x.strip()]

        # Group into pairs
        for i in range(0, len(values), 2):
            if i + 1 < len(values):
                start = values[i]
                end = values[i + 1]
                anomalies.append((start, end, chan_id))

    logger.info(f"Loaded {len(anomalies)} labeled anomaly ranges")
    return anomalies


def load_detection_results(results_csv: str) -> pd.DataFrame:
    """
    Load dual-metric anomaly detection results.

    Args:
        results_csv: Path to anomaly_detection_results.csv

    Returns:
        DataFrame with detection results
    """
    df = pd.read_csv(results_csv)
    logger.info(f"Loaded {len(df)} windows from detection results")
    return df


def extract_detected_ranges(df_results: pd.DataFrame, status_filter: Optional[List[str]] = None) -> List[Tuple[int, int]]:
    """
    Extract detected anomaly ranges from detection results.

    Args:
        df_results: Detection results DataFrame
        status_filter: List of statuses to consider as anomalies (None = all non-NORMAL)

    Returns:
        List of (start, end) tuples for detected anomaly ranges
    """
    if status_filter is None:
        # All non-NORMAL statuses are anomalies
        df_anomalies = df_results[df_results['status'] != 'NORMAL']
    else:
        df_anomalies = df_results[df_results['status'].isin(status_filter)]

    if len(df_anomalies) == 0:
        return []

    # Group consecutive windows into ranges
    ranges = []
    current_start = None
    current_end = None

    for _, row in df_anomalies.iterrows():
        t_center = int(row['t_center'])
        window_idx = int(row['window_idx'])

        if current_start is None:
            # Start new range
            current_start = t_center
            current_end = t_center
        elif window_idx == ranges[-1][2] + 1 if ranges else True:
            # Consecutive window, extend range
            current_end = t_center
        else:
            # Gap detected, save current range and start new one
            ranges.append((current_start, current_end, window_idx - 1))
            current_start = t_center
            current_end = t_center

    # Save final range
    if current_start is not None:
        ranges.append((current_start, current_end, len(df_anomalies) - 1))

    # Convert to (start, end) tuples
    detected_ranges = [(start, end) for start, end, _ in ranges]

    logger.info(f"Extracted {len(detected_ranges)} detected anomaly ranges from {len(df_anomalies)} anomalous windows")
    return detected_ranges


def compute_point_wise_metrics(labeled_points: np.ndarray, detected_points: np.ndarray) -> Dict[str, float]:
    """
    Compute point-wise precision, recall, F1-score.

    Args:
        labeled_points: Binary array of labeled anomalies (1 = anomaly)
        detected_points: Binary array of detected anomalies (1 = anomaly)

    Returns:
        Dictionary with precision, recall, F1
    """
    # True positives: detected AND labeled
    tp = np.sum((detected_points == 1) & (labeled_points == 1))

    # False positives: detected but NOT labeled
    fp = np.sum((detected_points == 1) & (labeled_points == 0))

    # False negatives: labeled but NOT detected
    fn = np.sum((detected_points == 0) & (labeled_points == 1))

    # True negatives: NOT detected AND NOT labeled
    tn = np.sum((detected_points == 0) & (labeled_points == 0))

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def compute_range_based_metrics(labeled_ranges: List[Tuple[int, int]],
                                detected_ranges: List[Tuple[int, int]],
                                min_overlap: float = 0.1) -> Dict[str, float]:
    """
    Compute range-based precision, recall, F1-score.

    A detected range is a true positive if it overlaps with at least one labeled range.
    A labeled range is detected if at least one detection overlaps with it.

    Args:
        labeled_ranges: List of (start, end) tuples for labeled anomalies
        detected_ranges: List of (start, end) tuples for detected anomalies
        min_overlap: Minimum overlap fraction to consider a match

    Returns:
        Dictionary with precision, recall, F1
    """
    if len(detected_ranges) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp_ranges': 0,
            'fp_ranges': 0,
            'fn_ranges': 0
        }

    if len(labeled_ranges) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0 if len(detected_ranges) > 0 else 1.0,
            'f1': 0.0,
            'tp_ranges': 0,
            'fp_ranges': len(detected_ranges),
            'fn_ranges': 0
        }

    # Check which detected ranges match labeled ranges
    tp_detected = set()
    for i, (det_start, det_end) in enumerate(detected_ranges):
        for lab_start, lab_end in labeled_ranges:
            # Compute overlap
            overlap_start = max(det_start, lab_start)
            overlap_end = min(det_end, lab_end)

            if overlap_end > overlap_start:
                overlap_length = overlap_end - overlap_start
                det_length = det_end - det_start
                lab_length = lab_end - lab_start

                # Check if sufficient overlap
                overlap_frac_det = overlap_length / det_length if det_length > 0 else 0
                overlap_frac_lab = overlap_length / lab_length if lab_length > 0 else 0

                if overlap_frac_det >= min_overlap or overlap_frac_lab >= min_overlap:
                    tp_detected.add(i)
                    break

    # Check which labeled ranges were detected
    tp_labeled = set()
    for i, (lab_start, lab_end) in enumerate(labeled_ranges):
        for det_start, det_end in detected_ranges:
            # Compute overlap
            overlap_start = max(det_start, lab_start)
            overlap_end = min(det_end, lab_end)

            if overlap_end > overlap_start:
                overlap_length = overlap_end - overlap_start
                lab_length = lab_end - lab_start
                det_length = det_end - det_start

                # Check if sufficient overlap
                overlap_frac_lab = overlap_length / lab_length if lab_length > 0 else 0
                overlap_frac_det = overlap_length / det_length if det_length > 0 else 0

                if overlap_frac_lab >= min_overlap or overlap_frac_det >= min_overlap:
                    tp_labeled.add(i)
                    break

    # Compute metrics
    tp = len(tp_detected)
    fp = len(detected_ranges) - tp
    fn = len(labeled_ranges) - len(tp_labeled)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = len(tp_labeled) / len(labeled_ranges) if len(labeled_ranges) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp_ranges': tp,
        'fp_ranges': fp,
        'fn_ranges': fn
    }


def evaluate_detection_performance(
    labels_csv: str,
    results_csv: str,
    output_csv: Optional[str] = None,
    status_filter: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Evaluate anomaly detection performance against labeled data.

    Args:
        labels_csv: Path to labeled_anomalies.csv
        results_csv: Path to anomaly_detection_results.csv
        output_csv: Optional path to save detailed evaluation results
        status_filter: Which statuses to consider as anomalies

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("="*80)
    logger.info("ANOMALY DETECTION EVALUATION")
    logger.info("="*80)
    logger.info(f"Labeled anomalies: {labels_csv}")
    logger.info(f"Detection results: {results_csv}")
    logger.info("")

    # Load data
    labeled_anomalies = load_labeled_anomalies(labels_csv)
    df_results = load_detection_results(results_csv)

    # Extract detected ranges
    detected_ranges = extract_detected_ranges(df_results, status_filter)

    # Get only the (start, end) tuples from labeled anomalies (ignore channel)
    labeled_ranges = [(start, end) for start, end, _ in labeled_anomalies]

    # Determine time range for point-wise evaluation
    all_times = []
    for start, end in labeled_ranges:
        all_times.extend([start, end])
    for start, end in detected_ranges:
        all_times.extend([start, end])

    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
    else:
        min_time = 0
        max_time = 0

    logger.info(f"Time range: [{min_time}, {max_time}]")

    # Create binary point-wise arrays
    time_length = max_time - min_time + 1 if max_time > min_time else 0

    if time_length > 0:
        labeled_points = np.zeros(time_length, dtype=int)
        detected_points = np.zeros(time_length, dtype=int)

        # Mark labeled anomalies
        for start, end in labeled_ranges:
            labeled_points[start - min_time:end - min_time + 1] = 1

        # Mark detected anomalies
        for start, end in detected_ranges:
            detected_points[start - min_time:end - min_time + 1] = 1

        # Compute point-wise metrics
        point_metrics = compute_point_wise_metrics(labeled_points, detected_points)
    else:
        point_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    # Compute range-based metrics
    range_metrics = compute_range_based_metrics(labeled_ranges, detected_ranges)

    # Compile results
    evaluation = {
        'metadata': {
            'labels_file': labels_csv,
            'results_file': results_csv,
            'num_labeled_ranges': len(labeled_ranges),
            'num_detected_ranges': len(detected_ranges),
            'time_range': (int(min_time), int(max_time)),
            'status_filter': status_filter
        },
        'point_wise': point_metrics,
        'range_based': range_metrics,
        'labeled_ranges': labeled_ranges,
        'detected_ranges': detected_ranges
    }

    # Log results
    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Labeled anomaly ranges:  {len(labeled_ranges)}")
    logger.info(f"Detected anomaly ranges: {len(detected_ranges)}")
    logger.info("")

    logger.info("Point-Wise Metrics (strict timestep matching):")
    logger.info(f"  Precision: {point_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {point_metrics['recall']:.4f}")
    logger.info(f"  F1-score:  {point_metrics['f1']:.4f}")
    logger.info(f"  TP: {point_metrics['tp']:,}  FP: {point_metrics['fp']:,}  FN: {point_metrics['fn']:,}")
    logger.info("")

    logger.info("Range-Based Metrics (overlap-based matching):")
    logger.info(f"  Precision: {range_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {range_metrics['recall']:.4f}")
    logger.info(f"  F1-score:  {range_metrics['f1']:.4f}")
    logger.info(f"  TP ranges: {range_metrics['tp_ranges']}  FP ranges: {range_metrics['fp_ranges']}  FN ranges: {range_metrics['fn_ranges']}")
    logger.info("")
    logger.info("="*80)

    # Save detailed results
    if output_csv:
        # Create summary DataFrame
        summary_data = {
            'metric_type': ['point_wise', 'range_based'],
            'precision': [point_metrics['precision'], range_metrics['precision']],
            'recall': [point_metrics['recall'], range_metrics['recall']],
            'f1_score': [point_metrics['f1'], range_metrics['f1']]
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_csv, index=False)
        logger.info(f"Evaluation summary saved to: {output_csv}")

    return evaluation


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate Anomaly Detection Performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script evaluates the dual-metric anomaly detection results against
the labeled anomalies from the NASA Telemanom dataset.

Metrics:
- Point-wise: Strict timestep-level precision/recall/F1
- Range-based: Overlap-based anomaly range matching

Examples:
  # Basic evaluation
  python evaluate_anomaly_detection.py \\
    --labels telemanom/labeled_anomalies.csv \\
    --results results/anomaly_detection/anomaly_detection_results.csv \\
    --output results/anomaly_detection/evaluation_summary.csv

  # Evaluate only NEW_ANOMALY_ONSET detections
  python evaluate_anomaly_detection.py \\
    --labels telemanom/labeled_anomalies.csv \\
    --results results/anomaly_detection/anomaly_detection_results.csv \\
    --status NEW_ANOMALY_ONSET
        """
    )

    parser.add_argument('--labels', required=True, help='Path to labeled_anomalies.csv')
    parser.add_argument('--results', required=True, help='Path to anomaly_detection_results.csv')
    parser.add_argument('--output', help='Output CSV file for evaluation summary')
    parser.add_argument('--status', nargs='+', help='Filter by status (e.g., NEW_ANOMALY_ONSET)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.labels).exists():
        logger.error(f"Labels file not found: {args.labels}")
        return 1

    if not Path(args.results).exists():
        logger.error(f"Results file not found: {args.results}")
        return 1

    # Run evaluation
    evaluation = evaluate_detection_performance(
        labels_csv=args.labels,
        results_csv=args.results,
        output_csv=args.output,
        status_filter=args.status
    )

    logger.info("")
    logger.info("Evaluation complete!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
