#!/usr/bin/env python3
import pandas as pd
import numpy as np
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def calculate_metrics(df_det, df_gt, threshold, num_windows, gt_mask):
    # Apply Threshold
    # pred_mask is boolean array for window-wise metrics
    pred_mask = (df_det['abs_score'] > threshold).values
    
    # ---------------------------------------------------------
    # 1. Window-wise Accuracy
    # ---------------------------------------------------------
    tp_w = np.sum(gt_mask & pred_mask)
    tn_w = np.sum((~gt_mask) & (~pred_mask))
    accuracy = (tp_w + tn_w) / num_windows
    
    # ---------------------------------------------------------
    # 2. Point-Adjusted (Sequence-Level) Evaluation
    # ---------------------------------------------------------
    total_sequences = 0
    detected_sequences = 0
    
    # Identify windows that are predicted as anomaly
    detected_window_indices = np.where(pred_mask)[0]
    detected_window_set = set(detected_window_indices)
    
    # Group detections into 'events' for Precision
    # (Simplified: detected events that don't overlap ANY GT are False Positives)
    
    detected_events = []
    if len(detected_window_indices) > 0:
        current_event = [detected_window_indices[0]]
        for w in detected_window_indices[1:]:
            if w == current_event[-1] + 1:
                current_event.append(w)
            else:
                detected_events.append(current_event)
                current_event = [w]
        detected_events.append(current_event)
    
    # Pre-compute GT windows for faster FP checking
    # (We can use gt_mask directly: if any window in event has gt_mask=True, it's a valid detection)
    
    # Count TPs (Sequences detected)
    for _, row in df_gt.iterrows():
        sequences = ast.literal_eval(row['anomaly_sequences'])
        for seq in sequences:
            total_sequences += 1
            start, end = seq
            
            min_w = max(0, int((start - 100) / 10))
            max_w = min(num_windows - 1, int(end / 10) + 1)
            
            # Check if this sequence is hit
            hit = False
            # Check overlap logic
            # Use explicit loop for correctness on range
            for w in range(min_w, max_w + 1):
                if w in detected_window_set:
                    hit = True
                    break
            
            if hit:
                detected_sequences += 1

    # Count FPs (Detected events that don't match any GT)
    fp_events = 0
    for event in detected_events:
        # Check if any window in the event overlaps with ANY GT
        # We can check gt_mask[w] for every w in event
        overlap = False
        for w in event:
            if w < num_windows and gt_mask[w]:
                overlap = True
                break
        if not overlap:
            fp_events += 1
            
    precision = detected_sequences / (detected_sequences + fp_events) if (detected_sequences + fp_events) > 0 else 0
    recall = detected_sequences / total_sequences if total_sequences > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, accuracy, fp_events, detected_sequences

def main():
    # 1. Load Data
    detection_file = 'results/bagging_experiment/anomaly_detection_FULL_855_windows.csv'
    gt_file = 'telemanom/labeled_anomalies.csv'
    
    logger.info("Loading data...")
    df_det = pd.read_csv(detection_file)
    df_gt = pd.read_csv(gt_file)
    
    scores = df_det['abs_score'].values
    num_windows = df_det['window_idx'].max() + 1
    
    # 2. Build Global GT Mask (Window-wise)
    logger.info("Building ground truth mask...")
    gt_mask = np.zeros(num_windows, dtype=bool)
    
    for _, row in df_gt.iterrows():
        sequences = ast.literal_eval(row['anomaly_sequences'])
        for seq in sequences:
            start, end = seq
            min_w = max(0, int((start - 100) / 10))
            max_w = min(num_windows - 1, int(end / 10) + 1)
            
            # Mark windows
            for w in range(min_w, max_w + 1):
                # Using simple window-overlap logic:
                # w_start = w*10, w_end = w*10 + 100
                # if max(start, w_start) < min(end, w_end):
                w_start = w * 10
                w_end = w_start + 100
                if max(start, w_start) < min(end, w_end):
                    gt_mask[w] = True

    # 3. Determine Search Range
    percentiles = np.linspace(80, 99.9, 50) 
    thresholds = np.percentile(scores, percentiles)
    thresholds = sorted(list(set(thresholds)))
    
    logger.info(f"Sweeping {len(thresholds)} thresholds...")
    
    results = []
    best_f1 = -1
    best_res = None
    
    for thresh in thresholds:
        prec, rec, f1, acc, fp, tp = calculate_metrics(df_det, df_gt, thresh, num_windows, gt_mask)
        
        results.append({
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc,
            'fp_events': fp,
            'tp_seqs': tp
        })
        
        if f1 >= best_f1: # Prefer higher threshold if F1 tie? Or lower? usually higher recall is better if F1 tied.
            best_f1 = f1
            best_res = results[-1]
            
    # Print Top 10
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('f1', ascending=False)
    
    logger.info("\n" + "="*90)
    logger.info("THRESHOLD OPTIMIZATION RESULTS")
    logger.info("="*90)
    logger.info(f"{'Threshold':<12} | {'F1':<8} | {'Recall':<8} | {'Precision':<10} | {'Accuracy':<8} | {'TP':<4} | {'FP':<4}")
    logger.info("-" * 90)
    
    for i in range(min(15, len(df_res))):
        r = df_res.iloc[i]
        logger.info(f"{r['threshold']:<12.6f} | {r['f1']:<8.4f} | {r['recall']:<8.4f} | {r['precision']:<10.4f} | {r['accuracy']:<8.4f} | {int(r['tp_seqs']):<4} | {int(r['fp_events']):<4}")
        
    logger.info("="*90)
    logger.info("="*90)
    if best_res is not None:
        logger.info(f"\nBest Configuration Found:")
        logger.info(f"Threshold: {best_res['threshold']:.6f}")
        logger.info(f"F1-Score:  {best_res['f1']:.4f}")
        logger.info(f"Recall:    {best_res['recall']:.4f} ({int(best_res['tp_seqs'])}/105)")
        logger.info(f"Precision: {best_res['precision']:.4f}")
        logger.info(f"Accuracy:  {best_res['accuracy']:.4f}")

    # Calculate Exact AUC-PR (Average Precision)
    # We use ALL unique scores as thresholds to trace the full curve
    logger.info("\nCalculating EXACT AUC-PR (sweeping all unique scores)...")
    
    # Get all unique scores, sorted descending
    unique_scores = np.sort(df_det['abs_score'].unique())[::-1]
    
    # We need to trace the curve. 
    # For efficiency, we can just compute P/R at every point.
    # Since N=855 is small, we can just run the metrics loop 855 times.
    
    all_precisions = []
    all_recalls = []
    
    # Optimization: Only evaluate if the classification changes? 
    # Yes, unique scores.
    
    for thresh in unique_scores:
        # Pass full=False (or similar) to avoid printing? 
        # We just call calculate_metrics
        p, r, _, _, _, _ = calculate_metrics(df_det, df_gt, thresh, num_windows, gt_mask)
        all_precisions.append(p)
        all_recalls.append(r)
        
    # Integrate using trapezoidal rule on the exact set of points
    # Ensure sorted by recall for integration
    recalls_arr = np.array(all_recalls)
    precisions_arr = np.array(all_precisions)
    
    # Sort by recall (ascending)
    sorted_indices = np.argsort(recalls_arr)
    sorted_recalls = recalls_arr[sorted_indices]
    sorted_precisions = precisions_arr[sorted_indices]
    
    exact_auc_pr = np.trapz(sorted_precisions, sorted_recalls)
    
    logger.info(f"Exact AUC-PR: {exact_auc_pr:.4f}")
    if best_res:
        logger.info(f"(Max F1 Configuration: Threshold={best_res['threshold']:.6f} -> F1={best_res['f1']:.4f})")

if __name__ == "__main__":
    main()
