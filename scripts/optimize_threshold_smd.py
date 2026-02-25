#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc

def load_labels(label_file):
    return np.loadtxt(label_file, delimiter=',')

def map_labels_to_windows(labels_point, num_windows, window_size=100, stride=10):
    y_true = np.zeros(num_windows)
    for w in range(num_windows):
        start = w * stride
        end = start + window_size
        if end > len(labels_point):
            break
        chunk = labels_point[start:end]
        if np.any(chunk == 1):
            y_true[w] = 1
    return y_true

def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score.
        label (np.ndarray): The ground-truth label.
        threshold (float): The threshold of anomaly score.
        pred (np.ndarray): The predicted label.
        calc_latency (bool):
    Returns:
        np.ndarray: The predicted label.
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def optimize(detection_file, label_file):
    print(f"Loading detections: {detection_file}")
    df = pd.read_csv(detection_file)
    
    if 'abs_score' not in df.columns:
        print("Error: 'abs_score' column missing.")
        return

    print(f"Loading labels: {label_file}")
    labels_point = load_labels(label_file)
    
    num_windows = df['window_idx'].max() + 1
    print(f"Mapping labels to {num_windows} windows...")
    y_true = map_labels_to_windows(labels_point, num_windows)
    
    scores = np.zeros(num_windows)
    for _, row in df.iterrows():
        idx = int(row['window_idx'])
        if idx < num_windows:
            scores[idx] = row['abs_score']
            
    # Calculate AUC-PR first
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(recall, precision)
    print(f"\nBaseline AUC-PR: {auc_pr:.4f}")
    
    # Sweep thresholds
    print("optimizing threshold...")
    thresholds = np.percentile(scores, np.linspace(0, 99.5, 100))
    thresholds = np.unique(thresholds)
    
    best_f1 = -1
    best_res = None
    
    results = []
    
    for thresh in thresholds:
        y_pred = (scores > thresh).astype(int)
        y_pred = (scores > thresh).astype(int)
        
        # Calculate Standard Metrics
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Calculate Point-Adusted Metrics
        y_pred_pa = adjust_predicts(scores, y_true, threshold=thresh)
        p_pa, r_pa, f1_pa, _ = precision_recall_fscore_support(y_true, y_pred_pa, average='binary', zero_division=0)
        
        if f1_pa > best_f1:
            best_f1 = f1_pa
            best_res = {
                'thresh': thresh, 
                'precision': p, 'recall': r, 'f1': f1,
                'precision_pa': p_pa, 'recall_pa': r_pa, 'f1_pa': f1_pa
            }
            
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Threshold: {best_res['thresh']:.6f}")
    print(f"Best F1 Score:  {best_res['f1']:.4f}")
    print("="*60)
    print("STANDARD METRICS (Window-wise)")
    print(f"Precision:      {best_res['precision']:.4f}")
    print(f"Recall:         {best_res['recall']:.4f}")
    print(f"F1 Score:       {best_res['f1']:.4f}")
    print("-" * 60)
    print("POINT-ADJUSTED METRICS (SOTA Comparable)")
    print(f"PA-Precision:   {best_res['precision_pa']:.4f}")
    print(f"PA-Recall:      {best_res['recall_pa']:.4f}")
    print(f"PA-F1 Score:    {best_res['f1_pa']:.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('label')
    args = parser.parse_args()
    optimize(args.file, args.label)
