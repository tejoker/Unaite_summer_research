#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
from scipy.linalg import norm
from tqdm import tqdm

def load_multiple_windows(csv_file, window_indices, lag, fixed_dim, chunk_size=1000000):
    target_windows = set(window_indices)
    matrices = {w: np.zeros((fixed_dim, fixed_dim)) for w in target_windows}
    
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        mask = (chunk['window_idx'].isin(target_windows)) & (chunk['lag'] == lag)
        relevant = chunk[mask]
        
        if len(relevant) > 0:
            for w_idx, group in relevant.groupby('window_idx'):
                i_arr = group['i'].values.astype(int)
                j_arr = group['j'].values.astype(int)
                w_arr = group['weight'].values
                W = matrices[w_idx]
                for idx in range(len(i_arr)):
                    i, j = i_arr[idx], j_arr[idx]
                    if i < fixed_dim and j < fixed_dim:
                        W[i, j] = w_arr[idx]
    return matrices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixed-dim', type=int, default=659)
    parser.add_argument('--lag', type=int, default=0)
    args = parser.parse_args()
    
    # 1. Count windows
    max_window = -1
    for chunk in pd.read_csv(args.golden, chunksize=1000000, usecols=['window_idx']):
        m = chunk['window_idx'].max()
        if not pd.isna(m) and m > max_window:
            max_window = int(m)
    num_windows = max_window + 1
    print(f"Found {num_windows} windows")
    
    # 2. Select 200 samples
    samples = np.linspace(0, num_windows-1, min(200, num_windows), dtype=int).tolist()
    
    # 3. Load
    print("Loading samples...")
    matrices_map = load_multiple_windows(args.golden, samples, args.lag, args.fixed_dim)
    W_list = list(matrices_map.values())
    
    if not W_list:
        print("Error: No samples loaded")
        return
        
    # 4. Average
    W_avg = np.mean(W_list, axis=0)
    print(f"Baseline Norm: {norm(W_avg, 'fro')}")
    
    # 5. Save
    rows = []
    d = args.fixed_dim
    for i in range(d):
        for j in range(d):
            w = W_avg[i, j]
            if abs(w) > 1e-8:
                rows.append({'window_idx': 0, 'lag': args.lag, 'i': i, 'j': j, 'weight': w})
    
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
