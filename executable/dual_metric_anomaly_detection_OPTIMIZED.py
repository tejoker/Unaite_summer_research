#!/usr/bin/env python3
"""
Memory-Optimized Dual-Metric Anomaly Detection for Full Dataset
"""

import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.linalg import norm
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def frobenius_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """Frobenius norm distance between two weight matrices."""
    return norm(W1 - W2, ord='fro')

def get_top_k_dimensions(W1: np.ndarray, W2: np.ndarray, k: int = 5) -> str:
    """
    Identify dimensions contributing most to the difference.
    Heuristic: Sum of absolute differences in rows + cols for each node.
    Returns: string of top-k indices (e.g., "1,5,10")
    """
    Diff = np.abs(W1 - W2)
    # Contribution of node i is sum of differences in its row (outgoing) and col (incoming)
    # Axis 0 is col (incoming), Axis 1 is row (outgoing)
    # We sum both to catch any change involving the node
    node_scores = np.sum(Diff, axis=1) + np.sum(Diff, axis=0)
    
    # Get top k indices
    top_indices = np.argsort(node_scores)[::-1][:k]
    return ",".join(map(str, top_indices))



def load_window_matrix(csv_file: str, window_idx: int, lag: int, fixed_dim: int, chunk_size: int = 1000000) -> np.ndarray:
    """
    Load a single window's weight matrix from CSV using chunked reading.
    Memory efficient - only loads relevant chunks.
    """
    W = np.zeros((fixed_dim, fixed_dim))
    
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        # Filter this chunk for our window and lag
        mask = (chunk['window_idx'] == window_idx) & (chunk['lag'] == lag)
        df_window = chunk[mask]
        
        if len(df_window) > 0:
            # Vectorized assignment
            i_arr = df_window['i'].values.astype(int)
            j_arr = df_window['j'].values.astype(int)
            weight_arr = df_window['weight'].values
            
            for idx in range(len(i_arr)):
                i, j = i_arr[idx], j_arr[idx]
                if i < fixed_dim and j < fixed_dim:
                    W[i, j] = weight_arr[idx]
    
    return W

def load_multiple_windows(csv_file: str, window_indices: list, lag: int, fixed_dim: int, chunk_size: int = 1000000) -> dict:
    """
    Load multiple windows efficiently in a single pass.
    Returns dict: {window_idx: W_matrix}
    """
    target_windows = set(window_indices)
    matrices = {w: np.zeros((fixed_dim, fixed_dim)) for w in target_windows}
    
    # We need to scan the whole file once
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
        # Filter for ANY of our target windows
        mask = (chunk['window_idx'].isin(target_windows)) & (chunk['lag'] == lag)
        relevant = chunk[mask]
        
        if len(relevant) > 0:
            for w_idx, group in relevant.groupby('window_idx'):
                i_arr = group['i'].values.astype(int)
                j_arr = group['j'].values.astype(int)
                w_arr = group['weight'].values
                
                # Assign to correct matrix
                W = matrices[w_idx]
                for idx in range(len(i_arr)):
                    i, j = i_arr[idx], j_arr[idx]
                    if i < fixed_dim and j < fixed_dim:
                        W[i, j] = w_arr[idx]
                        
    return matrices


def process_chunk(args):
    """
    Worker function to process a chunk of windows.
    Args: (chunk_windows, csv_file, lag, fixed_dim, chunk_size, W_golden_avg, lookback)
    """
    chunk_windows, csv_file, lag, fixed_dim, chunk_size, W_golden_avg, lookback = args
    
    results_chunk = []
    
    # Pre-load needed windows for this chunk (and 1 prior for change score)
    # This optimizes the worker to read the file ONCE for its chunk
    # instead of N times.
    
    windows_to_load = list(chunk_windows)
    start_w = chunk_windows[0]
    if start_w > 0:
        windows_to_load.append(start_w - 1)
        
    local_matrices = load_multiple_windows(csv_file, windows_to_load, lag, fixed_dim, chunk_size)
    
    # Now process sequentially
    W_prev = None
    if start_w > 0:
        W_prev = local_matrices.get(start_w - 1)

    for window_idx in chunk_windows:
        W_curr = local_matrices.get(window_idx, np.zeros((fixed_dim, fixed_dim)))
        
        # metrics
        abs_score = frobenius_distance(W_curr, W_golden_avg)
        
        if W_prev is not None:
            change_score = frobenius_distance(W_curr, W_prev)
        else:
            change_score = 0.0
            
        res = {
            'window_idx': window_idx,
            'abs_score': abs_score,
            'change_score': change_score,
            'top_dims': get_top_k_dimensions(W_curr, W_golden_avg, k=5)

        }
        results_chunk.append(res)
        
        W_prev = W_curr
        
    return results_chunk

def main():
    parser = argparse.ArgumentParser(description='Memory-optimized dual-metric anomaly detection')
    parser.add_argument('--golden', required=True, help='Golden baseline weights CSV')
    parser.add_argument('--test', required=True, help='Test timeline weights CSV')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--metric', default='frobenius', choices=['frobenius'], help='Distance metric')
    parser.add_argument('--lookback', type=int, default=5, help='Lookback for trend')
    parser.add_argument('--lag', type=int, default=0, help='Which lag to analyze')
    parser.add_argument('--fixed-dim', type=int, default=659, help='Matrix dimension')
    parser.add_argument('--chunk-size', type=int, default=1000000, help='CSV chunk size')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    logger.info(f"Starting memory-optimized anomaly detection (Workers: {args.workers})")
    logger.info(f"Golden: {args.golden}")
    logger.info(f"Test: {args.test}")
    logger.info(f"Output: {args.output}")
    
    # Step 1: Determine number of windows in test data
    logger.info("Scanning test dataset for window count...")
    max_window = -1
    # Check if we can just read the last line for speed? No, assuming standard structure.
    # Fast scan of just the 'window_idx' col?
    # pd.read_csv(usecols=['window_idx']) is faster
    for chunk in pd.read_csv(args.test, chunksize=args.chunk_size, usecols=['window_idx']):
        chunk_max = chunk['window_idx'].max()
        if not pd.isna(chunk_max) and chunk_max > max_window:
            max_window = int(chunk_max)
    
    num_windows = max_window + 1
    logger.info(f"Found {num_windows} windows in test dataset")
    
    # Step 2: Compute reference golden baseline (average of sample)
    logger.info("Computing reference golden baseline from 200 sample windows...")
    golden_sample_indices = np.linspace(0, num_windows-1, min(200, num_windows), dtype=int).tolist()
    
    # Optimization: Scan CSV once to load all samples at once
    # This reduces time from 5h to a few minutes
    logger.info(f"Optimized loading: Scanning golden file once for {len(golden_sample_indices)} windows...")
    golden_matrices_map = load_multiple_windows(args.golden, golden_sample_indices, args.lag, args.fixed_dim, args.chunk_size)
    
    W_golden_list = list(golden_matrices_map.values())
    if len(W_golden_list) == 0:
        logger.warning("No golden samples found! Using zeros.")
        W_golden_list = [np.zeros((args.fixed_dim, args.fixed_dim))]

    W_golden_avg = np.mean(W_golden_list, axis=0)
    logger.info(f"Reference baseline Frobenius norm: {norm(W_golden_avg, 'fro'):.6e}")
    
    # Save Golden Baseline for RCA
    baseline_path = os.path.dirname(args.output) + "/golden_baseline_matrix.csv"
    logger.info(f"Saving golden baseline matrix to {baseline_path}...")
    
    # Convert matrix to sparse CSV format (i, j, weight, window_idx, lag)
    rows = []
    d = W_golden_avg.shape[0]
    for i in range(d):
        for j in range(d):
            w = W_golden_avg[i, j]
            if abs(w) > 1e-8: # Sparsity check
                rows.append({
                    'window_idx': 0, # Dummy index foundation
                    'lag': args.lag,
                    'i': i,
                    'j': j,
                    'weight': w
                })
    pd.DataFrame(rows).to_csv(baseline_path, index=False)
    
    # Step 2.5: Compute Adaptive Thresholds from Golden Data
    logger.info("Computing adaptive thresholds from golden samples...")
    golden_scores = []
    for W in W_golden_list:
        score = frobenius_distance(W, W_golden_avg)
        golden_scores.append(score)
    
    golden_mean = np.mean(golden_scores)
    golden_std = np.std(golden_scores)
    
    # Adaptive Threshold: Mean + 3*Std
    # Robustness: Ensure minimum threshold
    threshold_abs_calc = max(golden_mean + 3 * golden_std, 0.001)
    
    logger.info(f"Golden scores: mean={golden_mean:.6f}, std={golden_std:.6f}")
    logger.info(f"Calculated adaptive threshold (mean+3std): {threshold_abs_calc:.6f}")
    
    # Step 3: Process Test Windows
    logger.info(f"Processing {num_windows} test windows...")

    # RESUME LOGIC
    processed_windows = set()
    if os.path.exists(args.output):
        try:
            # Check if file has header
            df_existing = pd.read_csv(args.output)
            if 'window_idx' in df_existing.columns:
                processed_windows = set(df_existing['window_idx'].unique())
                logger.info(f"Resuming: Found {len(processed_windows)} already processed windows.")
        except Exception as e:
            logger.warning(f"Could not read existing output file for resume: {e}")

    # Identify remaining windows
    all_window_indices = np.arange(num_windows)
    remaining_windows = [w for w in all_window_indices if w not in processed_windows]
    
    if len(remaining_windows) == 0:
        logger.info("All windows already processed. Skipping detection step.")
    else:
        logger.info(f"remaining windows to process: {len(remaining_windows)}")
        
        # Prepare for incremental writing
        write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
        
        # Helper to Append Results
        def save_batch(batch_results):
            nonlocal write_header
            if not batch_results: return
            
            # Post-process batch (basic metrics only, trends need history which we might lack in batch)
            # For resumability, we just save raw scores now and do post-processing trend analysis in a final pass
            rows = []
            for res in batch_results:
                rows.append({
                    'window_idx': res['window_idx'],
                    'abs_score': res['abs_score'],
                    'change_score': res['change_score'],
                    'top_dims': res.get('top_dims', ""),
                    # Placeholders for trend/status, updated in final pass
                    't_center': res['window_idx'],
                    'status': 'PENDING', 
                    'abs_trend': 0.0
                })
            
            df_batch = pd.DataFrame(rows)
            df_batch.to_csv(args.output, mode='a', header=write_header, index=False)
            write_header = False 

        if args.workers > 1:
            from multiprocessing import Pool
            
            # Split remaining windows
            chunk_len = max(1, len(remaining_windows) // args.workers) 
            # Ensure we don't hold too many chunks in memory, just stream them
            # Actually, let's make chunks fixed size for safety? 
            # 855 is small, but if 855 was gigantic... 
            # Stick to worker count split but respect 'remaining' list
            
            # We must pass contiguous chunks? No, `load_multiple_windows` takes a list.
            # So random access is fine!
            
            # Create sub-chunks of the remaining list
            sub_chunks = [remaining_windows[i:i + chunk_len] for i in range(0, len(remaining_windows), chunk_len)]
            
            pool_args = []
            for ch in sub_chunks:
                pool_args.append((ch, args.test, args.lag, args.fixed_dim, args.chunk_size, W_golden_avg, args.lookback))
            
            with Pool(args.workers) as pool:
                for batch in tqdm(pool.imap_unordered(process_chunk, pool_args), total=len(pool_args), desc="Processing Chunks"):
                    save_batch(batch)
                
        else:
            # Sequential
            # For sequential resume, we need W_prev logic if we want change_score correct?
            # If we skip, we might miss W_prev.
            # `process_chunk` handles loading W_prev (start_w - 1).
            # So we can just reuse process_chunk logic even for sequential!
            # Just verify distinct chunks.
            chunk_size_seq = 10 # Process 10 windows at a time and save
            sub_chunks = [remaining_windows[i:i + chunk_size_seq] for i in range(0, len(remaining_windows), chunk_size_seq)]
            
            for ch in tqdm(sub_chunks, desc="Sequential Processing"):
                # Adapt args for process_chunk
                args_tuple = (ch, args.test, args.lag, args.fixed_dim, args.chunk_size, W_golden_avg, args.lookback)
                batch = process_chunk(args_tuple)
                save_batch(batch)

    # Step 4: Final Post-Processing (Trend & Classification)
    # Re-read full file (now complete) and update Status/Trend
    logger.info("Finalizing: calculating trends and status on full dataset...")
    df_final = pd.read_csv(args.output)
    df_final.sort_values('window_idx', inplace=True)
    
    # Calculate Thresholds
    threshold_abs = threshold_abs_calc
    threshold_change = threshold_abs * 1.5
    threshold_trend = threshold_abs * 0.5
    
    abs_scores_series = df_final['abs_score'].values
    
    new_statuses = []
    new_trends = []
    
    for i in range(len(df_final)):
        abs_score = df_final.iloc[i]['abs_score']
        change_score = df_final.iloc[i]['change_score']
        
        # Trend
        if i >= args.lookback:
            trend = abs_scores_series[i] - abs_scores_series[i - args.lookback]
        else:
            trend = 0.0
            
        # Classify
        if abs_score < threshold_abs:
            s = "NORMAL"
        elif change_score > threshold_change and trend > threshold_trend:
            s = "NEW_ANOMALY_ONSET"
        elif change_score > threshold_change and trend < -threshold_trend:
            s = "RECOVERY_FLUCTUATION"
        elif abs_score > threshold_abs:
            s = "CASCADE_OR_PERSISTENT"
        else:
            s = "NORMAL"
            
        new_statuses.append(s)
        new_trends.append(trend)
        
    df_final['status'] = new_statuses
    df_final['abs_trend'] = new_trends
    
    # Save Final
    df_final.to_csv(args.output, index=False)
    
    # Step 6: Summary statistics
    logger.info("\n" + "="*80)
    logger.info("DETECTION SUMMARY")
    logger.info("="*80)
    status_counts = df_final['status'].value_counts()
    for status, count in status_counts.items():
        pct = 100 * count / len(df_final)
        logger.info(f"{status:25s}: {count:4d} windows ({pct:5.1f}%)")
    
    anomalies = df_final[df_final['status'] != 'NORMAL']
    logger.info(f"\nTotal anomaly windows: {len(anomalies)}")
    
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()

