#!/usr/bin/env python3
"""
Compare anomaly detection results with Telemanom ground truth labels.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path


def parse_anomaly_sequences(seq_str):
    """Parse anomaly sequence string from labeled_anomalies.csv."""
    return ast.literal_eval(seq_str)


def create_ground_truth_mask(anomaly_sequences, total_length):
    """Create binary mask for ground truth anomalies."""
    mask = np.zeros(total_length, dtype=bool)
    for start, end in anomaly_sequences:
        mask[start:end+1] = True
    return mask


def window_to_timesteps(window_idx, window_size=100, stride=10):
    """Convert window index to timestep range."""
    t_start = window_idx * stride
    t_end = t_start + window_size
    t_center = t_start + window_size // 2
    return t_start, t_end, t_center


def evaluate_detection(predictions, ground_truth):
    """
    Evaluate anomaly detection performance.

    Args:
        predictions: Boolean array of predictions (per timestep)
        ground_truth: Boolean array of ground truth (per timestep)

    Returns:
        Dictionary with metrics
    """
    tp = np.sum(predictions & ground_truth)
    fp = np.sum(predictions & ~ground_truth)
    fn = np.sum(~predictions & ground_truth)
    tn = np.sum(~predictions & ~ground_truth)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }


def point_adjust_evaluate(predictions, ground_truth):
    """
    Point-adjust evaluation: credit detection if ANY point in anomaly sequence is detected.
    This is the standard metric for time series anomaly detection.
    """
    # Find anomaly segments in ground truth
    diff = np.diff(np.concatenate([[False], ground_truth, [False]]).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    detected_segments = 0
    total_segments = len(starts)

    for start, end in zip(starts, ends):
        if np.any(predictions[start:end]):
            detected_segments += 1

    recall = detected_segments / total_segments if total_segments > 0 else 0

    # Count false positive segments (predicted anomalies in normal regions)
    pred_diff = np.diff(np.concatenate([[False], predictions, [False]]).astype(int))
    pred_starts = np.where(pred_diff == 1)[0]
    pred_ends = np.where(pred_diff == -1)[0]

    fp_segments = 0
    for pred_start, pred_end in zip(pred_starts, pred_ends):
        if not np.any(ground_truth[pred_start:pred_end]):
            fp_segments += 1

    precision = detected_segments / (detected_segments + fp_segments) if (detected_segments + fp_segments) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Detected_Segments': detected_segments,
        'Total_Segments': total_segments,
        'FP_Segments': fp_segments,
        'Precision_PA': precision,
        'Recall_PA': recall,
        'F1_PA': f1
    }


def main():
    # Load ground truth labels
    labels_file = Path('telemanom/labeled_anomalies.csv')
    df_labels = pd.read_csv(labels_file)

    # For P-1 channel (first in test dataset)
    p1_row = df_labels[df_labels['chan_id'] == 'P-1'].iloc[0]
    anomaly_sequences = parse_anomaly_sequences(p1_row['anomaly_sequences'])
    num_values = int(p1_row['num_values'])

    print("="*80)
    print("GROUND TRUTH (P-1 Channel)")
    print("="*80)
    print(f"Total timesteps: {num_values}")
    print(f"Anomaly sequences: {anomaly_sequences}")
    for i, (start, end) in enumerate(anomaly_sequences, 1):
        print(f"  Anomaly {i}: [{start}, {end}] (duration: {end-start+1} timesteps)")
    print()

    # Create ground truth mask
    gt_mask = create_ground_truth_mask(anomaly_sequences, num_values)
    print(f"Total anomalous timesteps: {gt_mask.sum()}/{num_values} ({100*gt_mask.sum()/num_values:.1f}%)")
    print()

    # Load our detection results
    results_file = Path('results/anomaly_detection_results.csv')
    df_results = pd.read_csv(results_file)

    print("="*80)
    print("OUR DETECTION RESULTS")
    print("="*80)
    print(f"Total windows analyzed: {len(df_results)}")
    print(f"Windows flagged as anomalies: {len(df_results[df_results['status'] != 'NORMAL'])}")
    print()

    # Map window-level detections to timesteps
    window_size = 100
    stride = 10
    pred_mask = np.zeros(num_values, dtype=bool)

    anomaly_windows = df_results[df_results['status'] != 'NORMAL']
    print("Detected anomaly windows:")
    for _, row in anomaly_windows.iterrows():
        window_idx = int(row['window_idx'])
        t_start, t_end, t_center = window_to_timesteps(window_idx, window_size, stride)
        t_start = min(t_start, num_values-1)
        t_end = min(t_end, num_values)

        pred_mask[t_start:t_end] = True
        print(f"  Window {window_idx}: t=[{t_start}, {t_end}], status={row['status']}")

    print()
    print(f"Total predicted anomalous timesteps: {pred_mask.sum()}/{num_values} ({100*pred_mask.sum()/num_values:.1f}%)")
    print()

    # Evaluate with standard metrics
    print("="*80)
    print("EVALUATION: Point-wise Metrics")
    print("="*80)
    metrics = evaluate_detection(pred_mask, gt_mask)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:15s}: {value:.4f}")
        else:
            print(f"{key:15s}: {value}")
    print()

    # Evaluate with point-adjust (standard for time series anomaly detection)
    print("="*80)
    print("EVALUATION: Point-Adjust Metrics (Standard for Time Series)")
    print("="*80)
    print("(Credits detection if ANY point in anomaly sequence is detected)")
    print()
    pa_metrics = point_adjust_evaluate(pred_mask, gt_mask)
    for key, value in pa_metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print()

    # Detailed comparison
    print("="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    for i, (start, end) in enumerate(anomaly_sequences, 1):
        detected = np.any(pred_mask[start:end+1])
        overlap = np.sum(pred_mask[start:end+1])
        status = "DETECTED" if detected else "MISSED"
        print(f"Anomaly {i} [{start:4d}, {end:4d}]: {status} (overlap: {overlap}/{end-start+1} timesteps)")
    print()

    print("="*80)
    print("CONCLUSION")
    print("="*80)
    if pa_metrics['Recall_PA'] > 0.7:
        print(f"Good detection! Detected {pa_metrics['Detected_Segments']}/{pa_metrics['Total_Segments']} anomaly segments")
    elif pa_metrics['Recall_PA'] > 0.3:
        print(f"Moderate detection. Detected {pa_metrics['Detected_Segments']}/{pa_metrics['Total_Segments']} anomaly segments")
    else:
        print(f"Poor detection. Only detected {pa_metrics['Detected_Segments']}/{pa_metrics['Total_Segments']} anomaly segments")

    if pa_metrics['FP_Segments'] > 5:
        print(f"High false positive rate: {pa_metrics['FP_Segments']} false alarms")
    print()


if __name__ == '__main__':
    main()
