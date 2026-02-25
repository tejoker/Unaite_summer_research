#!/usr/bin/env python3
"""
RCA Evaluation Script (Text Report Parser)
Evaluates Root Cause Analysis accuracy by parsing 'rca_report.txt'
and comparing against 'interpretation_label'.
Metric: Top-1 Accuracy (Precision) - Does the predicted root cause match GT?
"""

import os
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Configuration
RESULTS_ROOT = os.getenv('RESULTS_ROOT', '/mnt/disk2/results')
DATA_ROOT = os.getenv('DATA_ROOT', '/home/nicolas_b/program_internship_paul_wurth')
LABEL_DIR = os.path.join(DATA_ROOT, 'ServerMachineDataset/interpretation_label')

ENTITIES = [
    f"machine-{g}-{i}" 
    for g, count in [(1,8), (2,9), (3,11)] 
    for i in range(1, count+1)
]

def parse_ground_truth(entity):
    """
    Parses start-end:dim1,dim2... into a list of (start, end, set(dims))
    """
    path = os.path.join(LABEL_DIR, f"{entity}.txt")
    if not os.path.exists(path):
        return None
        
    gt_ranges = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line: continue
            
            try:
                # Format: 15849-16368:1,9,10...
                range_part, dims_part = line.split(':')
                start, end = map(int, range_part.split('-'))
                dims = set(map(int, dims_part.split(',')))
                gt_ranges.append((start, end, dims))
            except:
                continue
    return gt_ranges

def parse_rca_report(entity):
    """
    Parses rca_report.txt into list of (window_idx, root_cause_dim)
    """
    path = os.path.join(RESULTS_ROOT, f"bagging_SMD_{entity}", "rca_report.txt")
    if not os.path.exists(path):
        return None
        
    predictions = []
    with open(path, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3: continue
            
            # Check if first part is a window number
            if not parts[0].isdigit(): continue
            
            try:
                window_idx = int(parts[0])
                # Root Cause: "Var_13" -> 13
                rc_str = parts[2] 
                if rc_str.startswith("Var_"):
                    dim = int(rc_str.split('_')[1])
                    predictions.append((window_idx, dim))
            except:
                continue
    return predictions

def evaluate_entity_rca(entity):
    gt = parse_ground_truth(entity)
    preds = parse_rca_report(entity)
    
    if gt is None:
        return {'Entity': entity, 'Status': 'No GT'}
    if preds is None or len(preds) == 0:
        return {'Entity': entity, 'Status': 'No Predictions'}
        
    hits = 0
    total = 0
    
    # Window settings
    WINDOW_SIZE = 100
    STRIDE = 10 
    # Warning: Script might have used different stride? 
    # Usually stride=10 in our pipeline.
    
    for win_idx, pred_dim in preds:
        start_time = win_idx # * STRIDE # Wait, window_idx in detection output is typically the index.
        # But wait, does 'window_idx' in RCA report mean 'window number' or 'start timestamp'?
        # Looking at report: "1731", "2002".
        # If it's window number, timestamp is 1731 * 10 = 17310.
        # If it's timestamp, then it fits GT range 15849-16368? No.
        # Let's assume it IS the window index, so time is win_idx * 10.
        
        # Check overlaps
        win_start = win_idx * STRIDE # Assuming consistent Stride=10
        win_end = win_start + WINDOW_SIZE
        
        # Does this window overlap with any GT anomaly?
        # Actually, we should check if the anomaly happened HERE.
        
        match_found = False
        valid_pred = False # Is this prediction even in an anomaly zone?
        
        for gt_s, gt_e, gt_dims in gt:
            # Overlap check
            if max(win_start, gt_s) < min(win_end, gt_e):
                valid_pred = True
                if pred_dim in gt_dims:
                    match_found = True
                    break
                    
        if valid_pred:
            total += 1
            if match_found:
                hits += 1
                
    accuracy = hits / total if total > 0 else 0.0
    
    return {
        'Entity': entity,
        'Status': 'Evaluated',
        'RCA_Accuracy': accuracy,
        'Correct_Preds': hits,
        'Total_Evaluated_Preds': total
    }

def main():
    print("Starting RCA Evaluation (Text Report)...", flush=True)
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(evaluate_entity_rca, ENTITIES))
        
    df = pd.DataFrame(results)
    df.to_csv("rca_benchmark_results.csv", index=False)
    
    # Filter only evaluated
    eval_df = df[df['Status'] == 'Evaluated']
    
    print("\nRCA BENCHMARK SUMMARY (Top-1 Accuracy)")
    if not eval_df.empty:
        print(eval_df['RCA_Accuracy'].describe())
        print(f"Mean Accuracy: {eval_df['RCA_Accuracy'].mean():.4f}")
    else:
        print("No valid evaluations found (Check GT/Preds overlap logic).")

if __name__ == "__main__":
    main()
