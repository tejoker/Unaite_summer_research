#!/usr/bin/env python3
"""
RCA Runner Script (Embedded Logic)

This script performs Root Cause Analysis on detected anomalies by comparing
the causal graph weights of the anomalous window against the Golden Baseline.

It identifies the "Root Cause" node as the one with the highest outgoing influence
change (highest sum of absolute weight differences in outgoing edges).
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_weights_matrix(csv_path, window_idx, n_nodes):
    """
    Load weight matrix from weights_enhanced.csv for a specific window.
    Returns an (N, N) numpy matrix.
    """
    try:
        # Optimized load: read only relevant columns
        df = pd.read_csv(csv_path, usecols=['window_idx', 'i', 'j', 'weight', 'lag'])
        
        # Filter for window and lag 0 (contemporaneous/instantaneous effects)
        # We focus on lag 0 for root cause usually, but could sum lags.
        # Let's stick to lag 0 for interpreting "current" causal drivers.
        df_window = df[(df['window_idx'] == window_idx) & (df['lag'] == 0)]
        
        W = np.zeros((n_nodes, n_nodes))
        
        if df_window.empty:
            logging.warning(f"No weights found for window {window_idx} in {csv_path}")
            return W
            
        # Fill matrix
        for _, row in df_window.iterrows():
            i, j = int(row['i']), int(row['j'])
            if i < n_nodes and j < n_nodes:
                W[i, j] = row['weight']
                
        return W
    except Exception as e:
        logging.error(f"Failed to load weights from {csv_path}: {e}")
        return np.zeros((n_nodes, n_nodes))

def load_golden_average(csv_path, n_nodes):
    """
    Compute average Golden Baseline matrix (across all windows).
    """
    try:
        df = pd.read_csv(csv_path, usecols=['i', 'j', 'weight', 'lag'])
        df = df[df['lag'] == 0]
        
        # Group by edge (i, j) and mean
        df_avg = df.groupby(['i', 'j'])['weight'].mean().reset_index()
        
        W = np.zeros((n_nodes, n_nodes))
        for _, row in df_avg.iterrows():
            i, j = int(row['i']), int(row['j'])
            if i < n_nodes and j < n_nodes:
                W[i, j] = row['weight']
        return W
    except Exception as e:
        logging.error(f"Failed to load golden baseline: {e}")
        return np.zeros((n_nodes, n_nodes))

def perform_rca(W_golden, W_current, columns):
    """
    Identify Root Causes by comparing W_current and W_golden.
    Metric: Out-degree of Difference Matrix (D = |W_cur - W_gold|).
    """
    # 1. Compute Difference Matrix
    Delta = np.abs(W_current - W_golden)
    
    # 2. Compute Node Scores
    # Out-degree (sum of row) -> "How much did this node change its influence on others?"
    # In-degree (sum of col) -> "How much was this node affected by others?"
    
    node_out_scores = np.sum(Delta, axis=1) # Sum of outgoing weight changes
    node_in_scores = np.sum(Delta, axis=0)  # Sum of incoming weight changes
    
    # 3. Identify Top Contributors
    top_cause_idx = np.argmax(node_out_scores)
    top_effect_idx = np.argmax(node_in_scores)
    
    root_cause = columns[top_cause_idx]
    primary_effect = columns[top_effect_idx]
    
    # 4. Find Top Changed Edge
    # Flatten and find max
    flat_idx = np.argmax(Delta)
    source_idx, target_idx = np.unravel_index(flat_idx, Delta.shape)
    
    top_edge = f"{columns[source_idx]} -> {columns[target_idx]}"
    edge_change_val = W_current[source_idx, target_idx] - W_golden[source_idx, target_idx]
    
    return root_cause, primary_effect, top_edge, edge_change_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', help='Results directory (containing anomaly_detection_bagged.csv)')
    parser.add_argument('detections_file', help='Full path to anomaly_detection_bagged.csv')
    parser.add_argument('golden_baseline', help='Full path to weights_enhanced.csv (Golden)')
    args = parser.parse_args()

    results_dir = args.results_dir
    detections_file = args.detections_file
    baseline_file = args.golden_baseline

    # 1. Load Detections
    if not os.path.exists(detections_file):
        logging.error(f"Detections file not found: {detections_file}")
        sys.exit(1)
        
    df_det = pd.read_csv(detections_file)
    anomalies = df_det[df_det['status'] != 'NORMAL']
    
    if anomalies.empty:
        logging.info("No anomalies detected. Skipping RCA.")
        # Create empty report
        with open(os.path.join(results_dir, "rca_report.txt"), "w") as f:
            f.write("No anomalies detected.\n")
        return

    # 2. Sort by score and pick Top 5
    top_anomalies = anomalies.sort_values('abs_score', ascending=False).head(5)
    
    # 3. Determine Columns / Dimensions
    # Try to find machine-1-6_columns.npy first (standard location)
    # Since we don't have the entity name passed explicitly, rely on standard paths or fallback
    # But wait, we can infer n_nodes from the CSV itself if needed.
    # However, having names is better.
    
    # Look for any *_columns.npy in the golden baseline folder?
    # Or just default to "Var_X"
    columns = []
    
    # HACK: Try to find columns file in obvious places
    possible_col_files = [
        os.path.join(results_dir, "../data/SMD/test/machine-1-6_columns.npy"), # Standard check
        os.path.join(results_dir, "feature_names.txt")
    ]
    
    col_file_found = None
    for p in possible_col_files:
        if os.path.exists(p):
            col_file_found = p
            break
            
    if col_file_found and col_file_found.endswith('.npy'):
        try:
            columns = list(np.load(col_file_found, allow_pickle=True))
        except:
            pass
    elif col_file_found and col_file_found.endswith('.txt'):
        try:
            with open(col_file_found, 'r') as f:
                content = f.read().strip()
                if ',' in content:
                    columns = content.split(',')
        except:
            pass
            
    # Fallback: scan CSV to find max index
    if not columns:
        logging.info("Could not auto-detect column names. Using generic Var_i.")
        # Scan baseline to find max dimension
        df_base = pd.read_csv(baseline_file, nrows=1000)
        max_idx = max(df_base['i'].max(), df_base['j'].max())
        columns = [f"Var_{i}" for i in range(max_idx + 1)]
        
    n_nodes = len(columns)
    
    # 4. Load Golden Baseline (Average)
    logging.info("Loading Golden Baseline...")
    W_golden = load_golden_average(baseline_file, n_nodes)
    
    # 5. Analyze Each Anomaly
    summary_table = []
    
    # Guess the location of the "Current" weights.
    # In bagging detection, we compared Test vs Golden.
    # The Test Weights come from the Bagging Runs. 
    # Usually "results/bagging_SMD_.../runs/run_000/weights/weights_enhanced.csv"
    # We will assume run_000 is the representative "current" timeline.
    current_weights_path = os.path.join(results_dir, "runs", "run_000", "weights", "weights_enhanced.csv")
    
    if not os.path.exists(current_weights_path):
        logging.error(f"Could not find test weights at {current_weights_path}")
        return

    logging.info("Starting RCA Loop...")
    
    for _, row in top_anomalies.iterrows():
        w_idx = int(row['window_idx'])
        score = row['abs_score']
        
        # Load W_current
        W_current = load_weights_matrix(current_weights_path, w_idx, n_nodes)
        
        # Perform RCA
        root, effect, changed_edge, delta = perform_rca(W_golden, W_current, columns)
        
        summary_table.append({
            'window': w_idx,
            'score': score,
            'root': root,
            'effect': effect,
            'edge': f"{changed_edge} ({delta:+.3f})"
        })

    # 6. Print Report
    print("\n" + "="*110)
    print("ROOT CAUSE ANALYSIS REPORT (Top Anomalies)")
    print("-" * 110)
    print(f"{'Window':<8} | {'Score':<8} | {'Root Cause':<25} | {'Primary Effect':<25} | {'Major Edge Change':<30}")
    print("-" * 110)
    for row in summary_table:
        print(f"{row['window']:<8} | {row['score']:<8.4f} | {row['root']:<25} | {row['effect']:<25} | {row['edge']:<30}")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
