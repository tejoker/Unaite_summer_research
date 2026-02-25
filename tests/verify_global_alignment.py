#!/usr/bin/env python3
import pandas as pd
import numpy as np
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def main():
    # 1. Load Detection Results
    detection_file = 'results/bagging_experiment/anomaly_detection_FULL_855_windows.csv'
    logger.info(f"Loading detection results from {detection_file}...")
    df_det = pd.read_csv(detection_file)
    
    # 2. Load Ground Truth
    gt_file = 'telemanom/labeled_anomalies.csv'
    logger.info(f"Loading ground truth from {gt_file}...")
    df_gt = pd.read_csv(gt_file)
    
    # 3. Create Global GT Mask (T=8550, assuming 855 windows * 10 stride)
    # Actually, let's map everything to Windows.
    # Window w covers [w*10, w*10 + 100].
    # If ANY anomaly overlaps with Window w, then GT(w) = 1.
    
    num_windows = df_det['window_idx'].max() + 1
    stride = 10
    window_size = 100
    gt_mask = np.zeros(num_windows, dtype=int)
    
    anomalies_found = 0
    total_anomalies = 0
    
    # Iterate over all entities and their anomalies
    # Assuming GLOBAL ALIGNMENT: Start time 0 is the same for everyone.
    for _, row in df_gt.iterrows():
        sequences = ast.literal_eval(row['anomaly_sequences'])
        for seq in sequences:
            start, end = seq
            total_anomalies += 1
            
            # Find windows overlapping this sequence
            # Window w range: [w*10, w*10 + 100]
            # Overlap if: max(start, w_start) < min(end, w_end)
            
            # Optimization: 
            # w*10 + 100 > start  =>  w*10 > start - 100  =>  w > (start - 100)/10
            # w*10 < end          =>  w < end/10
            
            min_w = max(0, int((start - 100) / 10))
            max_w = min(num_windows - 1, int(end / 10) + 1)
            
            for w in range(min_w, max_w + 1):
                w_start = w * stride
                w_end = w_start + window_size
                if max(start, w_start) < min(end, w_end):
                    gt_mask[w] = 1
                    
    # 4. Compare
    df_det['is_anomaly'] = df_det['status'] != 'NORMAL'
    df_det['gt_anomaly'] = gt_mask[df_det['window_idx']]
    
    tp = len(df_det[(df_det['is_anomaly']) & (df_det['gt_anomaly'] == 1)])
    fp = len(df_det[(df_det['is_anomaly']) & (df_det['gt_anomaly'] == 0)])
    fn = len(df_det[(~df_det['is_anomaly']) & (df_det['gt_anomaly'] == 1)])
    tn = len(df_det[(~df_det['is_anomaly']) & (df_det['gt_anomaly'] == 0)])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info("=" * 40)
    logger.info("GLOBAL VALIDATION RESULTS")
    logger.info("=" * 40)
    logger.info(f"Total Windows: {num_windows}")
    logger.info(f"Ground Truth Anomaly Windows: {sum(gt_mask)}")
    logger.info(f"Detected Anomaly Windows: {len(df_det[df_det['is_anomaly']])}")
    logger.info("-" * 40)
    logger.info(f"True Positives: {tp}")
    logger.info(f"False Positives: {fp}")
    logger.info(f"False Negatives: {fn}")
    logger.info(f"True Negatives: {tn}")
    logger.info("-" * 40)
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info("=" * 40)

    # 6. Point-Adjusted (Sequence-Level) Evaluation
    logger.info("\nPOINT-ADJUSTED SEQUENTIAL EVALUATION")
    logger.info("-" * 40)
    
    total_sequences = 0
    detected_sequences = 0
    
    # Iterate over all GT sequences and check if ANY of our 19 detections hit them
    detected_window_indices = set(df_det[df_det['is_anomaly']]['window_idx'].values)
    
    missed_sequences = []
    
    for _, row in df_gt.iterrows():
        entity_id = row['chan_id']
        sequences = ast.literal_eval(row['anomaly_sequences'])
        for i, seq in enumerate(sequences):
            start, end = seq
            total_sequences += 1
            
            # Find windows covering this sequence
            min_w = max(0, int((start - 100) / 10))
            max_w = min(num_windows - 1, int(end / 10) + 1)
            
            # Check overlap
            hit = False
            for w in range(min_w, max_w + 1):
                if w in detected_window_indices:
                    hit = True
                    break
            
            if hit:
                detected_sequences += 1
            else:
                missed_sequences.append(f"{entity_id} Seq {i} [{start}-{end}]")

    seq_recall = detected_sequences / total_sequences if total_sequences > 0 else 0
    logger.info(f"Total GT Sequences: {total_sequences}")
    logger.info(f"Detected Sequences: {detected_sequences}")
    logger.info(f"Sequence Recall:    {seq_recall:.4f}")
    
    if missed_sequences:
        logger.info(f"\nMissed {len(missed_sequences)} sequences. First 5:")
        for m in missed_sequences[:5]:
            logger.info(f"  - {m}")
    else:
        logger.info("\nALL SEQUENCES DETECTED!")

if __name__ == "__main__":
    main()
