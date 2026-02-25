#!/usr/bin/env python3
"""
Full Benchmark Evaluation Script (Parallel & Robust)
Evaluates Anomaly Detection (F1, AUC-PR) and Causal Stability across all 28 SMD entities.
"""

import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_ROOT = os.getenv('RESULTS_ROOT', '/mnt/disk2/results')
DATA_ROOT = os.getenv('DATA_ROOT', '/home/nicolas_b/program_internship_paul_wurth')
LABEL_DIR = os.path.join(DATA_ROOT, 'ServerMachineDataset/test_label')

# Generate Entity List
ENTITIES = [
    f"machine-{g}-{i}" 
    for g, count in [(1,8), (2,9), (3,11)] 
    for i in range(1, count+1)
]

def evaluate_entity(entity):
    """
    Evaluate a single entity for both Anomaly Detection and Causality.
    Returns a dict with metrics or None if failed.
    """
    result = {'Entity': entity}
    
    # --- Part 1: Anomaly Detection ---
    try:
        detection_file = os.path.join(RESULTS_ROOT, f"bagging_SMD_{entity}", "anomaly_detection_bagged.csv")
        label_file = os.path.join(LABEL_DIR, f"{entity}.txt")
        
        if os.path.exists(detection_file) and os.path.exists(label_file):
            df_det = pd.read_csv(detection_file)
            labels_point = np.loadtxt(label_file, delimiter=',')
            
            if 'window_idx' in df_det.columns:
                num_windows = int(df_det['window_idx'].max()) + 1
                y_true = np.zeros(num_windows)
                y_pred = np.zeros(num_windows)
                y_scores = np.zeros(num_windows)
                
                # Ground Truth Mapping
                window_size = 100
                stride = 10
                for w in range(num_windows):
                    start = w * stride
                    end = start + window_size
                    if end > len(labels_point): break
                    if np.any(labels_point[start:end] == 1):
                        y_true[w] = 1
                
                # Predictions
                anomalies = df_det[df_det['status'].isin(['NEW_ANOMALY_ONSET', 'CASCADE_OR_PERSISTENT'])]
                for idx in anomalies['window_idx']:
                    if idx < num_windows:
                        y_pred[int(idx)] = 1
                        
                if 'abs_score' in df_det.columns:
                    for _, row in df_det.iterrows():
                        idx = int(row['window_idx'])
                        if idx < num_windows:
                            y_scores[idx] = row['abs_score']

                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
                
                if np.sum(y_true) > 0 and 'abs_score' in df_det.columns:
                    p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
                    auc_pr = auc(r_curve, p_curve)
                else:
                    auc_pr = 0.0
                    
                result.update({
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1,
                    'AUC_PR': auc_pr
                })
            else:
                result.update({'F1': 0, 'AUC_PR': 0, 'Error': 'Missing window_idx'})
        else:
             result.update({'F1': 0, 'AUC_PR': 0, 'Error': 'files_missing'})
             
    except Exception as e:
        result.update({'F1': 0, 'AUC_PR': 0, 'Error': str(e)})

    # --- Part 2: Causal Stability ---
    try:
        runs_dir = os.path.join(RESULTS_ROOT, f"bagging_SMD_{entity}", "runs")
        if os.path.exists(runs_dir):
            weight_files = glob.glob(os.path.join(runs_dir, "run_*/weights/weights_enhanced.csv"))
            
            if len(weight_files) >= 5: # Need a few runs for stability
                matrices = []
                target_shape = None
                
                for f in weight_files[:20]: # Limit to 20 files to speed up stats if many runs
                    try:
                        df = pd.read_csv(f, index_col=0)
                        mat = df.values
                        if target_shape is None:
                            target_shape = mat.shape
                        
                        if mat.shape == target_shape:
                            matrices.append(mat)
                    except:
                        continue
                
                if matrices:
                    stack = np.array(matrices)
                    # Mean Variance of edges
                    edge_variances = np.var(stack, axis=0)
                    mean_var = np.mean(edge_variances)
                    # Stability = exp(-variance)
                    stability = np.exp(-mean_var) # 0 to 1
                    
                    result.update({
                        'Causal_Stability': stability,
                        'Edge_Variance': mean_var,
                        'n_valid_runs': len(matrices)
                    })
                else:
                    result.update({'Causal_Stability': 0, 'n_valid_runs': 0})
            else:
                 result.update({'Causal_Stability': 0, 'n_valid_runs': len(weight_files)})
        else:
            result.update({'Causal_Stability': 0, 'n_valid_runs': 0})
            
    except Exception as e:
        result['Causal_Error'] = str(e)
        result.update({'Causal_Stability': 0})
        
    return result

def main():
    print(f"Starting Benchmark Evaluation (28 Entities, Parallel)...", flush=True)
    
    # Parallel Execution
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(evaluate_entity, ENTITIES))
        
    # Analyze Results
    df = pd.DataFrame(results)
    
    # Fill NaNs
    df.fillna(0, inplace=True)
    
    # Save Detailed
    out_file = "benchmark_summary_detailed.csv"
    df.to_csv(out_file, index=False)
    
    # Calculate Averages
    numeric_cols = ['Precision', 'Recall', 'F1', 'AUC_PR', 'Causal_Stability']
    avg = df[numeric_cols].mean()
    
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY RESULTS (Aggregated)")
    print("="*50)
    print(df[numeric_cols].describe().loc[['mean', 'min', 'max']])
    print("="*50)
    print(f"Detailed CSV saved to: {out_file}")

if __name__ == "__main__":
    main()
