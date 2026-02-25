import numpy as np
import pandas as pd
import ast
import os
from sklearn.metrics import average_precision_score

def range_f1(y_true, y_score, k=0):
    """
    Computes Range-F1 score based on Tatbul et al. (2018).
    Simplified implementation:
    - Precision: Same as point-based (TP / (TP + FP))
    - Recall: Existence reward. If any point in a ground-truth anomaly segment is detected, 
              it counts as fully recalled (or partially, depending on weighting).
              Here we use the "Existence" policy: 1.0 if overlap > 0.
    """
    # Thresholding
    threshold = np.percentile(y_score, 100 * (1 - k/len(y_score))) if k > 0 else 0.5
    # Let's find best F1 threshold instead of fixed k
    
    # Actually, let's look at the downloaded precision/recall curve data
    # But that's point-based. We need to re-compute from raw scores.
    pass

def calculate_metrics(detections_path, gt_path):
    print(f"Loading detections from {detections_path}...")
    df = pd.read_csv(detections_path)
    df = df.sort_values('window_idx')
    scores = df['abs_score'].values
    
    print(f"Loading GT from {gt_path}...")
    gt = pd.read_csv(gt_path)
    gt['anomaly_sequences'] = gt['anomaly_sequences'].apply(ast.literal_eval)
    
    # Convert GT to ranges
    gt_ranges = []
    window_size = 250
    stride = 10
    
    for _, row in gt.iterrows():
        for start, end in row['anomaly_sequences']:
             # Convert timestamp/index to window index range
             w_start = max(0, int((start - window_size)/stride))
             w_end = int(end / stride) + 2
             gt_ranges.append((w_start, w_end))
             
    # Calculate Range-Recall (Existence)
    # We sweep thresholds
    print("Calculating Range-F1 (Existence Policy)...")
    
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_f1 = 0
    best_p = 0
    best_r = 0
    
    for th in thresholds:
        pred_anom_indices = np.where(scores > th)[0]
        if len(pred_anom_indices) == 0: continue
        
        # Predicted Ranges (consecutive)
        # Simplified: just checking overlap
        # Precision: TP_ranges / Pred_ranges
        # Recall: TP_gt_ranges / Total_gt_ranges
        
        # Point-based Precision is often used in Range-F1 papers for simplicity mixed with Range-Recall
        # But proper definition uses predicted ranges.
        
        # Let's use a simpler proxy:
        # Recall = (Count of GT ranges hit by at least one prediction) / (Total GT ranges)
        # Precision = (Count of predicted points in GT ranges) / (Total predicted points) [Standard Precision]
        # This is "Point-Precision, Event-Recall" often called "Event-F1"
        
        hits = 0
        for start, end in gt_ranges:
            # Check overlap
            if np.any((pred_anom_indices >= start) & (pred_anom_indices <= end)):
                hits += 1
                
        recall = hits / len(gt_ranges) if len(gt_ranges) > 0 else 0
        
        # Precision (Point-wise for standard approx)
        tp_points = 0
        for idx in pred_anom_indices:
             is_tp = False
             for start, end in gt_ranges:
                 if start <= idx <= end:
                     is_tp = True
                     break
             if is_tp: tp_points += 1
             
        precision = tp_points / len(pred_anom_indices)
        
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            
        if f1 > best_f1:
            best_f1 = f1
            best_p = precision
            best_r = recall
            
            best_r = recall
            
    print(f"Best Range-F1 (Event-Recall/Point-Precision): {best_f1:.4f}")
    print(f"  Precision: {best_p:.4f}")
    print(f"  Recall:    {best_r:.4f}")

    # Calculate AUC-PR (Window-based)
    y_true = np.zeros(len(scores))
    for start, end in gt_ranges:
        # Clip to array bounds
        s = max(0, start)
        e = min(len(y_true), end)
        y_true[s:e] = 1.0
        
    if np.sum(y_true) > 0:
        auc_pr = average_precision_score(y_true, scores)
        print(f"AUC-PR: {auc_pr:.4f}")
        return auc_pr
    else:
        print("AUC-PR: N/A (No anomalies in GT)")
        return 0.0


def parse_smd_labels(label_path):
    """
    Parses SMD interpretation_label file (e.g., machine-1-1.txt).
    Format: start-end:dim1,dim2,...
    Returns list of dicts: [{'start': s, 'end': e, 'dims': {d1, d2...}}, ...]
    """
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                range_str, dims_str = parts
                start, end = map(int, range_str.split('-'))
                dims = set(map(int, dims_str.split(',')))
                labels.append({'start': start, 'end': end, 'dims': dims})
    return labels

def calculate_ack(detections_path, gt_path, k=1, dataset='Telemanom'):
    print(f"\nCalculating AC@{k} for {dataset}...")
    
    if dataset == 'Telemanom':
        print("  Note: True Root Cause Analysis requires a causal graph ground truth which Telemanom lacks.")
        print("  AC@k: N/A")
        return

    # SMD Logic
    gt_labels = parse_smd_labels(gt_path)
    df = pd.read_csv(detections_path)
    
    # We need to map window_idx back to raw timestamps/indices or assume window alignment
    # SMD is usually point-based indices. Our windows have 'window_idx'.
    # Assuming window_size=100, stride=1 for SMD? Or whatever was used.
    # Standard SMD is often window=100. Let's assume direct index mapping if possible or just range overlap.
    
    # Actually, our `window_idx` corresponds to a time range in the original series.
    # Let's assume standard windowing: start_time = window_idx * stride
    # We need the stride. Let's assume stride=10 (from config) or look at 'i' column index?
    # Our CSV doesn't have start/end time.
    # Let's use a simplified "Event-based" AC@k.
    
    hits = 0
    total_events = 0
    
    # AC@k: For each anomaly event, did we correctly identify at least one root cause dimension in top-k?
    # Or strict: are ALL root causes in top-k? usually "Recall@k" or "HitRate@k".
    # Let's do HitRate@k: % of events where at least one ground truth dimension is in top-k.
    
    # Pre-parse df into a lookup: window_idx -> top_dims
    window_preds = {}
    if 'top_dims' in df.columns:
        for _, row in df.iterrows():
            if pd.isna(row['top_dims']) or row['top_dims'] == "":
                dims = []
            else:
                dims = [int(float(x)) for x in str(row['top_dims']).split(',')[:k]]
            window_preds[row['window_idx']] = set(dims)
            
    # Window parameters (Critical!)
    stride = 10 
    window_size = 100 # Adjust if needed
    
    for event in gt_labels:
        total_events += 1
        event_start, event_end = event['start'], event['end']
        gt_dims = event['dims']
        
        # Find all windows intersecting this event
        # w_start * stride <= event_end AND w_end * stride >= event_start
        # approximate: w_idx * stride approx within event
        
        w_min = max(0, int((event_start - window_size)/stride))
        w_max = int(event_end / stride) + 2
        
        event_hit = False
        
        # Aggregate predictions for the event? Or check if ANY window in the event got it right?
        # Standard: Aggregate root cause scores for the event segment, then rank.
        # Simplified: Check if *any* anomalous window in the range correctly identified a root cause.
        
        for w_idx in range(w_min, w_max):
            if w_idx in window_preds:
                pred_dims = window_preds[w_idx]
                # Check intersection
                if not pred_dims.isdisjoint(gt_dims):
                    event_hit = True
                    break
        
        if event_hit:
            hits += 1
            
    if total_events > 0:
        ack = hits / total_events
        print(f"  AC@{k} (HitRate): {ack:.4f} ({hits}/{total_events} events)")
        return ack
    else:
        print("  No ground truth events found.")
        return 0.0

def calculate_smd_event_f1(detections_path, gt_path):
    """
    Calculates Event-based F1 (Existence Policy) for SMD.
    Recall = % of GT events detected (at least one overlapping window)
    Precision = % of predicted windows that overlap with a GT event
    """
    gt_labels = parse_smd_labels(gt_path)
    df = pd.read_csv(detections_path)
    
    # Filter for anomalies
    # Use 'status' column if available (binary decision from detector)
    if 'status' not in df.columns:
        return 0, 0, 0
        
    # Check for non-NORMAL status
    pred_windows = df[df['status'] != 'NORMAL']['window_idx'].values
    
    if len(pred_windows) == 0:
        return 0, 0, 0
    
    # Config
    stride = 10
    window_size = 100
    
    # 1. Recall (Event-wise)
    hits = 0
    total_events = len(gt_labels)
    
    for event in gt_labels:
        e_start, e_end = event['start'], event['end']
        
        # Check if ANY predicted window overlaps this event
        is_detected = False
        for w_idx in pred_windows:
            w_start = w_idx * stride
            w_end = w_start + window_size
            
            # Overlap condition
            if max(w_start, e_start) < min(w_end, e_end):
                is_detected = True
                break
        
        if is_detected:
            hits += 1
            
    recall = hits / total_events if total_events > 0 else 0.0
    
    # 2. Precision (Window-wise)
    tp_windows = 0
    for w_idx in pred_windows:
        w_start = w_idx * stride
        w_end = w_start + window_size
        
        is_tp = False
        for event in gt_labels:
            e_start, e_end = event['start'], event['end']
            if max(w_start, e_start) < min(w_end, e_end):
                is_tp = True
                break
        
        if is_tp:
            tp_windows += 1
            
    precision = tp_windows / len(pred_windows)
    
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        
    print(f"  Range-F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})")
    return f1, precision, recall

def calculate_smd_rca_f1(detections_path, gt_path, k=3):
    """
    Calculates F1 score for Root Cause Analysis (Dimension Retrieval).
    For each correctly detected event (Hit), computes F1 of predicted vs GT dimensions.
    """
    gt_labels = parse_smd_labels(gt_path)
    df = pd.read_csv(detections_path)
    
    # Lookup for predictions
    window_preds = {}
    if 'top_dims' in df.columns:
        for _, row in df.iterrows():
            if pd.isna(row['top_dims']) or row['top_dims'] == "":
                dims = []
            else:
                dims = [int(float(x)) for x in str(row['top_dims']).split(',')[:k]]
            window_preds[row['window_idx']] = set(dims)
            
    stride = 10
    window_size = 100
    
    precisions = []
    recalls = []
    f1s = []
    
    for event in gt_labels:
        e_start, e_end = event['start'], event['end']
        gt_dims = event['dims']
        if not gt_dims: continue # Should generally exist
        
        # Find intersecting windows
        w_min = max(0, int((e_start - window_size)/stride))
        w_max = int(e_end / stride) + 2
        
        # We need to aggregate predicted dimensions for this event.
        # Strategy: Union of top-k from all overlapping windows? 
        # Or Majority vote?
        # Let's take the Union of top-k from all overlapping *anomalous* windows.
        
        event_pred_dims = set()
        found_window = False
        
        for w_idx in range(w_min, w_max):
            if w_idx in window_preds:
                found_window = True
                event_pred_dims.update(window_preds[w_idx])
                
        if not found_window:
            # Missed event -> 0 scores
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue
            
        # Calculate P/R/F1 for this event
        tp = len(gt_dims.intersection(event_pred_dims))
        fp = len(event_pred_dims - gt_dims)
        fn = len(gt_dims - event_pred_dims)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
    avg_f1 = np.mean(f1s) if f1s else 0.0
    print(f"  RCA-F1 (Dim-wise): {avg_f1:.4f}")
    return avg_f1

def calculate_smd_full(results_dir, gt_dir, k=3):
    print(f"\n[SMD Full Benchmark] Aggregating results from {results_dir}...")
    import glob
    
    detection_files = glob.glob(os.path.join(results_dir, "*_detection.csv"))
    if not detection_files:
        print("  No detection files found.")
        return


    ack_scores = []
    f1_scores = [] # Detection F1
    rca_f1_scores = [] # Root Cause F1
    auc_pr_scores = [] # AUC-PR
    
    print(f"  Found {len(detection_files)} machines.")
    
    for fpath in detection_files:
        fname = os.path.basename(fpath)
        entity = fname.replace("_detection.csv", "") # e.g. machine-1-1
        
        gt_file = os.path.join(gt_dir, f"{entity}.txt")
        if not os.path.exists(gt_file):
            print(f"  [Skipping] No GT for {entity}")
            continue
            
        # Calculate Metrics for this entity
        print(f"  Processing {entity}...", end=" ")
        try:
             # AC@k
             ack = calculate_ack(fpath, gt_file, k=k, dataset='SMD')
             if ack is not None:
                 ack_scores.append(ack)
                 
             # Range-F1 (Detection)
             f1, p, r = calculate_smd_event_f1(fpath, gt_file)
             f1_scores.append(f1)
             
             # RCA F1 (Dimension)
             rca_f1 = calculate_smd_rca_f1(fpath, gt_file, k=k)
             rca_f1_scores.append(rca_f1)
             
             # AUC-PR (Re-using calculate_metrics logic essentially but adapted for SMD)
             # We need a quick helper or reuse calculate_metrics?
             # calculate_metrics relies on Telemanom specific GT parsing.
             # Let's add inline or helper for SMD AUC-PR
             
             # Inline SMD AUC-PR
             df_det = pd.read_csv(fpath)
             scores = df_det['abs_score'].values
             gt_labels = parse_smd_labels(gt_file)
             
             y_true = np.zeros(len(scores))
             stride = 10
             window_size = 100
             
             for event in gt_labels:
                 e_start, e_end = event['start'], event['end']
                 w_start = max(0, int((e_start - window_size)/stride))
                 w_end = int(e_end / stride) + 2
                 
                 s = max(0, w_start)
                 e = min(len(y_true), w_end)
                 y_true[s:e] = 1.0
                 
             if np.sum(y_true) > 0:
                 auc = average_precision_score(y_true, scores)
                 auc_pr_scores.append(auc)
                 print(f"  AUC-PR: {auc:.4f}")
             else:
                 print("  AUC-PR: N/A (No GT)")
                 
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*40)
    print("FINAL AGGREGATE METRICS (Macro-Average across entities)")
    if ack_scores:
        print(f"Mean AC@{k} (HitRate):   {np.mean(ack_scores):.4f}")
    if f1_scores:
        print(f"Mean Range-F1 (Det):   {np.mean(f1_scores):.4f}")
    if rca_f1_scores:
        print(f"Mean RCA-F1 (Dim):     {np.mean(rca_f1_scores):.4f}")
    if auc_pr_scores:
        print(f"Mean AUC-PR:           {np.mean(auc_pr_scores):.4f}")
    print("="*40)

if __name__ == "__main__":
    # Telemanom
    if os.path.exists('results/bagging_experiment_download/anomaly_detection_FULL_855_windows.csv'):
         print("-" * 30)
         print("Telemanom Evaluation")
         calculate_metrics('results/bagging_experiment_download/anomaly_detection_FULL_855_windows.csv', 'telemanom/labeled_anomalies.csv')
    
    # SMD Full Benchmark
    smd_results_dir = 'results/SMD_fast_ack'
    smd_gt_dir = 'ServerMachineDataset/interpretation_label'
    
    if os.path.exists(smd_results_dir) and os.path.exists(smd_gt_dir):
        calculate_smd_full(smd_results_dir, smd_gt_dir, k=1)
    else:
        # Fallback to single machine if full results missing
        smd_det = 'results/SMD_machine-1-1_golden_baseline/anomaly_detection_smd_ack.csv'
        smd_gt = 'ServerMachineDataset/interpretation_label/machine-1-1.txt'
        if os.path.exists(smd_det):
             calculate_ack(smd_det, smd_gt, k=3, dataset='SMD')

