#!/usr/bin/env python3
"""
Academic Standard Evaluation Script (v2 - Fixed Causal Parsing)
Implements:
1. Point Adjustment (PA-F1)
2. Best-F1 Search
3. Robust Causal Stability (PARSES EDGE LISTS CORRECTLY)
"""

import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')

# Configuration
RESULTS_ROOT = os.getenv('RESULTS_ROOT', '/mnt/disk2/results')
DATA_ROOT = os.getenv('DATA_ROOT', '/home/nicolas_b/program_internship_paul_wurth')
LABEL_DIR = os.path.join(DATA_ROOT, 'ServerMachineDataset/test_label')

ENTITIES = [
    f"machine-{g}-{i}" 
    for g, count in [(1,8), (2,9), (3,11)] 
    for i in range(1, count+1)
]

def get_anomaly_segments(y_true):
    events = []
    start = None
    for i, val in enumerate(y_true):
        if val == 1 and start is None:
            start = i
        elif val == 0 and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(y_true) - 1))
    return events

def point_adjustment(y_score, y_true, threshold):
    y_pred = (y_score >= threshold).astype(int)
    y_pred_pa = y_pred.copy()
    segments = get_anomaly_segments(y_true)
    for start, end in segments:
        if np.sum(y_pred[start : end + 1]) > 0:
            y_pred_pa[start : end + 1] = 1
    return y_pred_pa

def find_best_f1_pa(y_score, y_true, num_steps=100):
    if len(y_score) == 0: return 0,0,0,0.0
    min_score, max_score = np.min(y_score), np.max(y_score)
    thresholds = [min_score] if min_score == max_score else np.linspace(min_score, max_score, num_steps)
    
    best_stats = (0.0, 0.0, 0.0, 0.0)
    
    for th in thresholds:
        y_pred_pa = point_adjustment(y_score, y_true, th)
        pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_pa, average='binary', zero_division=0)
        if f1 > best_stats[0]:
            best_stats = (f1, pre, rec, th)
    return best_stats

def evaluate_entity_academic(entity):
    result = {'Entity': entity}
    
    # --- Part 1: Anomaly Detection ---
    try:
        detection_file = os.path.join(RESULTS_ROOT, f"bagging_SMD_{entity}", "anomaly_detection_bagged.csv")
        label_file = os.path.join(LABEL_DIR, f"{entity}.txt")
        
        if os.path.exists(detection_file) and os.path.exists(label_file):
            df_det = pd.read_csv(detection_file)
            labels_point = np.loadtxt(label_file, delimiter=',')
            
            if 'window_idx' in df_det.columns and 'abs_score' in df_det.columns:
                num_windows = int(df_det['window_idx'].max()) + 1
                y_true = np.zeros(num_windows)
                y_scores = np.zeros(num_windows)
                
                window_size, stride = 100, 10
                for w in range(num_windows):
                    start = w * stride
                    end = start + window_size
                    if end > len(labels_point): break
                    if np.any(labels_point[start:end] == 1): y_true[w] = 1
                
                for _, row in df_det.iterrows():
                    idx = int(row['window_idx'])
                    if idx < num_windows: y_scores[idx] = row['abs_score']

                if np.sum(y_true) > 0:
                    best_f1, best_pre, best_rec, best_th = find_best_f1_pa(y_scores, y_true)
                else:
                    best_f1, best_pre, best_rec, best_th = 0.0, 0.0, 0.0, 0.0
                    
                result.update({
                    'Best_F1_PA': best_f1, 
                    'Best_Precision_PA': best_pre, 
                    'Best_Recall_PA': best_rec
                })
            else:
                 result.update({'Best_F1_PA': 0, 'Error': 'Missing columns'})
        else:
             result.update({'Best_F1_PA': 0, 'Error': 'Files missing'})
    except Exception as e:
        result.update({'Best_F1_PA': 0, 'Error': str(e)})

    # --- Part 2: Causal Stability (Fixed for Edge Lists) ---
    try:
        runs_dir = os.path.join(RESULTS_ROOT, f"bagging_SMD_{entity}", "runs")
        if os.path.exists(runs_dir):
            weight_files = glob.glob(os.path.join(runs_dir, "run_*/weights/weights_enhanced.csv"))
            
            if len(weight_files) >= 5:
                # Determine max dimension (SMD is typically 38 dimensions)
                # But we can infer max(i, j) from headers or edge lists.
                N_VARS = 38 # SMD standard
                
                dense_matrices = []
                
                for f in weight_files[:20]:
                    try:
                        df = pd.read_csv(f)
                        # Expect columns: window_idx, i, j, lag, weight
                        if 'i' not in df.columns or 'weight' not in df.columns: continue
                        
                        # Pivot to dense N x N (ignoring lag for simplicity, or summing weights across lags)
                        # We want the aggregate adjacency matrix (A->B strength)
                        # Sum absolute weights for same i,j pair
                        adj = np.zeros((N_VARS, N_VARS))
                        
                        for _, row in df.iterrows():
                            i, j = int(row['i']), int(row['j'])
                            if i < N_VARS and j < N_VARS:
                                adj[i, j] += abs(row['weight'])
                        
                        dense_matrices.append(adj)
                    except:
                        continue
                
                if len(dense_matrices) >= 2:
                    stack = np.array(dense_matrices)
                    # Variance
                    edge_variances = np.var(stack, axis=0)
                    mean_var = np.mean(edge_variances)
                    # Scale variance? If mean edge weight is small, variance is small.
                    stability = np.exp(-mean_var * 100) # Scaling factor to make it sensitive?
                    # Or just 1 - normalized variance.
                    # Let's stick to exp(-var) but be aware if weights are tiny (0.005), var is tinier (0.000025).
                    # Stability will represent "consistency of edge strengths".
                    stability = np.exp(-mean_var)

                    result.update({
                        'Causal_Stability': stability,
                        'Edge_Variance': mean_var,
                        'n_valid_runs': len(dense_matrices)
                    })
                else:
                    result.update({'Causal_Stability': 0, 'n_valid_runs': len(dense_matrices)})
            else:
                 result.update({'Causal_Stability': 0, 'n_valid_runs': len(weight_files)})
        else:
            result.update({'Causal_Stability': 0, 'n_valid_runs': 0})
    except Exception as e:
        result['Causal_Error'] = str(e)
        result.update({'Causal_Stability': 0})
        
    return result

def main():
    print(f"Starting ACADEMIC Evaluation v2...", flush=True)
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(evaluate_entity_academic, ENTITIES))
        
    df = pd.DataFrame(results)
    df.fillna(0, inplace=True)
    df.to_csv("academic_benchmark_results_v2.csv", index=False)
    
    cols = ['Best_F1_PA', 'Causal_Stability', 'Edge_Variance']
    print("\nBENCHMARK SUMMARY v2")
    print(df[cols].describe().loc[['mean', 'max']])

if __name__ == "__main__":
    main()
