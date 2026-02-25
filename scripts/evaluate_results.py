
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def get_events(y_test):
    """
    Extract continuous anomaly events from binary labels.
    Returns list of (start, end) tuples.
    """
    events = []
    start = None
    for i, val in enumerate(y_test):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(y_test) - 1))
    return events

def evaluate(detections_csv, label_file, window_size=100, stride=10):
    # Load detections
    df_det = pd.read_csv(detections_csv)
    
    # Load labels
    # Labels are usually point-wise. We need to map them to windows.
    labels_point = np.loadtxt(label_file, delimiter=',')
    
    # Assuming detections are indexed by 'window_idx' which maps to time
    # We need to reconstruct the time range for each window or map labels to windows.
    # Standard mapping: window i covers [i*stride, i*stride + window_size]
    
    # Strategy: Convert window anomalies to point anomalies (OR logic)
    # If a window is anomalous, all points in it are candidates (or we just map points to windows)
    # Better: Map point labels to window labels.
    # If any point in a window is 1, the window is 1.
    
    if 'window_idx' not in df_det.columns:
        print("Error: 'window_idx' column missing in detections.")
        return

    # Create array of window labels
    num_windows = df_det['window_idx'].max() + 1
    y_true_windows = np.zeros(num_windows)
    
    # Max possible window index based on label length?
    # n_windows = (n_points - window_size) // stride + 1
    
    for w in range(num_windows):
        start = w * stride
        end = start + window_size
        if end > len(labels_point):
            break
        chunk = labels_point[start:end]
        if np.any(chunk == 1):
            y_true_windows[w] = 1
            
    # Create array of predicted window labels
    y_pred_windows = np.zeros(num_windows)
    
    # Filter for anomalous windows in detection
    # Column is likely 'status' based on dual_metric_anomaly_detection.py output
    anomalies = df_det[df_det['status'].isin(['NEW_ANOMALY_ONSET', 'CASCADE_OR_PERSISTENT'])]
    for idx in anomalies['window_idx']:
        if idx < num_windows:
            y_pred_windows[idx] = 1
            
    # Calculate Point-adjustment F1 (PA-F1) or standard F1?
    # Standard F1 on windows is safer for now.
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_windows, y_pred_windows, average='binary')
    
    print(f"Metrics (Window-wise):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Calculate AUC-PR
    # We use 'abs_score' as the anomaly likelihood score
    if 'abs_score' in df_det.columns:
        from sklearn.metrics import precision_recall_curve, auc
        
        # We need continuous scores for all windows
        # Map abs_score to window index y_pred_scores
        y_scores = np.zeros(num_windows)
        
        for _, row in df_det.iterrows():
            idx = int(row['window_idx'])
            if idx < num_windows:
                # Use abs_score as the continuous anomaly score
                y_scores[idx] = row['abs_score']
                
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_windows, y_scores)
        auc_pr = auc(recall_curve, precision_curve)
        print(f"AUC-PR:    {auc_pr:.4f}")
    else:
        print("AUC-PR:    N/A ('abs_score' column missing)")
    
    # Point-Adjustment simulate
    # If we hit an event, we get the whole event?
    # Let's stick to raw metrics first as they are strictest.
    
    print(f"Total Windows: {num_windows}")
    print(f"True Anomalies: {y_true_windows.sum()}")
    print(f"Predicted Anomalies: {y_pred_windows.sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('detections', help='Path to detection CSV')
    parser.add_argument('labels', help='Path to label TXT')
    args = parser.parse_args()
    
    evaluate(args.detections, args.labels)
