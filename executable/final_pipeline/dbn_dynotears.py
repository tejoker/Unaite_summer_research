#!/usr/bin/env python3
"""
DynoTears DBN Analysis WITHOUT MI Masking
Simplified version that removes all MI-masking functionality
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import os
import sys
import pickle
import csv
import time
import logging
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler
from dynotears import from_pandas_dynamic, extract_matrices, generate_histogram_and_kde

# --------------------------- Configuration ---------------------------
DATA_FILE = os.getenv('INPUT_DIFFERENCED_CSV', "differenced_stationary_series.csv")
OPTIMAL_LAGS_FILE = os.getenv('INPUT_LAGS_CSV', "optimal_lags.csv")
RESULT_DIR = os.getenv('RESULT_DIR')
if RESULT_DIR:
    OUTPUT_DIR = os.path.join(RESULT_DIR, 'weights')
    CHECKPOINT_FILE = os.path.join(RESULT_DIR, 'history', "rolling_checkpoint.pkl")
    WEIGHTS_HISTORY_CSV = os.path.join(RESULT_DIR, 'history', "weights_history.csv")
    WEIGHTS_HISTORY_CKPT = os.path.join(RESULT_DIR, 'history', "weights_history.pkl")
else:
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', "dynotears_results")
    CHECKPOINT_FILE = os.getenv('CHECKPOINT_FILE', "rolling_checkpoint.pkl")
    WEIGHTS_HISTORY_CSV = os.getenv('WEIGHTS_HISTORY_CSV', "weights_history.csv")
    WEIGHTS_HISTORY_CKPT = os.getenv('WEIGHTS_HISTORY_CKPT', "weights_history.pkl")

# --------------------------- Logging Setup ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --------------------------- Utility Functions ---------------------------

def load_optimal_lags(path):
    """Load optimal lags from CSV file"""
    if not os.path.exists(path):
        logger.warning(f"Optimal lags file not found: {path}")
        return None
    
    df = pd.read_csv(path)
    if 'variable' in df.columns and 'optimal_lag' in df.columns:
        return df.set_index('variable')['optimal_lag'].to_dict()
    return None

def estimate_optimal_window_size(df, efold=1/np.e, candidates_multiplier=[2,5,10]):
    """Estimate optimal window size based on autocorrelation"""
    logger.info("Estimating optimal window size...")
    
    # Calculate autocorrelation for each variable
    acf_values = []
    for col in df.columns:
        try:
            acf_result = acf(df[col].dropna(), nlags=min(50, len(df)//4), fft=True)
            acf_values.append(acf_result)
        except Exception as e:
            logger.warning(f"Could not calculate ACF for {col}: {e}")
            continue
    
    if not acf_values:
        logger.warning("Could not calculate ACF for any variable, using default window size")
        return 100
    
    # Find the lag where ACF drops below efold
    min_lag = len(df)
    for acf_series in acf_values:
        for i, val in enumerate(acf_series):
            if val < efold:
                min_lag = min(min_lag, i)
                break
    
    # Generate candidate window sizes
    candidates = [min_lag * mult for mult in candidates_multiplier]
    candidates = [c for c in candidates if 10 <= c <= len(df)//2]
    
    if not candidates:
        window_size = min(100, len(df)//4)
    else:
        window_size = min(candidates)
    
    logger.info(f"Estimated optimal window size: {window_size}")
    return window_size

def rolling_window_analysis(df, window_size, step_size, optimal_lags, variable_names):
    """Perform rolling window analysis without MI masking"""
    logger.info(f"Starting rolling window analysis: window_size={window_size}, step_size={step_size}")
    
    n_samples, n_vars = df.shape
    max_lag = max(optimal_lags.values()) if optimal_lags else 1
    
    # Initialize results storage
    results = []
    weights_history = []
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    
    # Rolling window analysis
    window_idx = 0
    for start_idx in range(0, n_samples - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_data = df.iloc[start_idx:end_idx].copy()
        
        logger.info(f"Processing window {window_idx}: rows {start_idx}-{end_idx}")
        
        try:
            # Run DynoTEARS without MI masking
            result = from_pandas_dynamic(
                window_data,
                p=max_lag,  # Required: lag order
                lambda_w=0.1,
                lambda_a=0.1,
                max_iter=100,
                h_tol=1e-8,
                w_threshold=0.1
            )
            
            # Extract matrices
            W_est, A_est_list = extract_matrices(result, variable_names, max_lag)
            
            # Store results
            window_result = {
                'window_idx': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'W_est': W_est,
                'A_est_list': A_est_list,
                'timestamp': time.time()
            }
            results.append(window_result)
            
            # Store weights for history
            weights_entry = {
                'window_idx': window_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'W_est': W_est.detach().cpu().numpy() if hasattr(W_est, 'detach') else W_est,
                'timestamp': time.time()
            }
            weights_history.append(weights_entry)
            
            # Save checkpoint every 10 windows
            if window_idx % 10 == 0:
                with open(CHECKPOINT_FILE, 'wb') as f:
                    pickle.dump(results, f)
                logger.info(f"Checkpoint saved at window {window_idx}")
            
            window_idx += 1
            
        except Exception as e:
            logger.error(f"Error processing window {window_idx}: {e}")
            continue
    
    # Save final results
    logger.info(f"Analysis complete. Processed {len(results)} windows")
    
    # Save weights history
    if weights_history:
        weights_df = pd.DataFrame(weights_history)
        weights_df.to_csv(WEIGHTS_HISTORY_CSV, index=False)
        with open(WEIGHTS_HISTORY_CKPT, 'wb') as f:
            pickle.dump(weights_history, f)
        logger.info(f"Weights history saved to {WEIGHTS_HISTORY_CSV}")
    
    return results

def export_results(results, variable_names, max_lag, window_size, step_size):
    """Export results to CSV format with enhanced columns"""
    logger.info("Exporting results to CSV with enhanced format...")

    # Prepare data for export
    export_data = []

    for result in results:
        window_idx = result['window_idx']
        start_idx = result['start_idx']
        end_idx = result['end_idx']
        W_est = result['W_est']
        A_est_list = result['A_est_list']

        # Calculate temporal information
        t_end = end_idx
        ts_end = end_idx + 1  # ts is typically t+1
        t_center = (start_idx + end_idx) // 2
        ts_center = t_center + 1

        # Convert to numpy if needed
        if hasattr(W_est, 'detach'):
            W_est = W_est.detach().cpu().numpy()

        # Export contemporaneous weights (lag 0)
        for i in range(len(variable_names)):
            for j in range(len(variable_names)):
                if abs(W_est[i, j]) > 1e-6:  # Only export non-zero weights
                    export_data.append({
                        'window_idx': window_idx,
                        't_end': t_end,
                        'ts_end': ts_end,
                        't_center': t_center,
                        'ts_center': ts_center,
                        'lag': 0,
                        'i': i,
                        'j': j,
                        'child_name': variable_names[i],
                        'parent_name': variable_names[j],
                        'weight': float(W_est[i, j])
                    })

        # Export autoregressive weights (lags 1+)
        for lag_idx, A_est in enumerate(A_est_list):
            lag = lag_idx + 1
            if hasattr(A_est, 'detach'):
                A_est = A_est.detach().cpu().numpy()

            for i in range(len(variable_names)):
                for j in range(len(variable_names)):
                    if abs(A_est[i, j]) > 1e-6:  # Only export non-zero weights
                        export_data.append({
                            'window_idx': window_idx,
                            't_end': t_end,
                            'ts_end': ts_end,
                            't_center': t_center,
                            'ts_center': ts_center,
                            'lag': lag,
                            'i': i,
                            'j': j,
                            'child_name': variable_names[i],
                            'parent_name': variable_names[j],
                            'weight': float(A_est[i, j])
                        })
    
    # Save to CSV
    if export_data:
        output_file = os.path.join(OUTPUT_DIR, "weights_enhanced.csv")
        df_export = pd.DataFrame(export_data)
        df_export.to_csv(output_file, index=False)
        logger.info(f"Results exported to {output_file}")
        logger.info(f"Total edges exported: {len(export_data)}")
    else:
        logger.warning("No results to export")

def main():
    """Main analysis function"""
    logger.info("=== DYNOTEARS ANALYSIS WITHOUT MI MASKING ===")
    
    try:
        # Load data
        logger.info(f"Loading data from: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} variables")
        
        # Load optimal lags
        optimal_lags = load_optimal_lags(OPTIMAL_LAGS_FILE)
        if optimal_lags:
            logger.info(f"Loaded optimal lags for {len(optimal_lags)} variables")
        else:
            logger.warning("No optimal lags found, using default lag=1")
            optimal_lags = {col: 1 for col in df.columns}
        
        # Get variable names
        variable_names = list(df.columns)
        max_lag = max(optimal_lags.values()) if optimal_lags else 1
        
        # Dynamically estimate window size, but with sensible bounds
        estimated_size = estimate_optimal_window_size(df)
        window_size = max(50, min(estimated_size, 200)) # Bound between 50 and 200
        step_size = max(1, window_size // 10) # Overlap of ~90%
        
        # Run rolling window analysis
        results = rolling_window_analysis(df, window_size, step_size, optimal_lags, variable_names)

        # Export results
        export_results(results, variable_names, max_lag, window_size, step_size)
        
        logger.info("=== ANALYSIS COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
