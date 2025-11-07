#!/usr/bin/env python3
"""
DynoTEARS with Fixed Lambda Strategy

Uses fixed lambda values (calibrated on golden data) for both baseline
and anomaly runs instead of performing separate hyperparameter searches.

Purpose:
Eliminate the "Butterfly Effect" where a future anomaly influences
global lambda selection, causing weight differences in early windows.
By using the same lambdas for both Golden and Anomaly runs, any observed
weight difference is guaranteed to result from the data itself, not from
different regularization parameters.
"""

import os
import sys
import logging
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json

# Add parent directory to path to allow imports from the final_pipeline package
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dynotears import from_pandas_dynamic
except ImportError:
    print("Error: Could not import 'from_pandas_dynamic' from 'dynotears'. Make sure dynotears.py is in the same directory.")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def find_best_lambdas(df: pd.DataFrame, p: int, result_dir: Path, candidates=None):
    """
    Performs a search on a sample of the data to find the best lambdas and saves them.
    The 'best' is defined as the one producing a reasonable number of edges.
    """
    n_vars = df.shape[1]
    
    # Adaptive lambda candidates based on dimensionality
    if candidates is None:
        if n_vars < 20:
            candidates = [0.01, 0.05, 0.1, 0.5]
        elif n_vars < 100:
            candidates = [0.05, 0.1, 0.3, 0.5]
        elif n_vars < 500:
            candidates = [0.1, 0.3, 0.5, 1.0]
        else:
            # For very high dimensional problems, use much stronger regularization
            scale = np.sqrt(n_vars / 100)
            candidates = [0.3 * scale, 0.5 * scale, 1.0 * scale, 2.0 * scale]
            logger.info(f"High-dimensional dataset ({n_vars} vars), using scaled candidates: {[f'{c:.2f}' for c in candidates]}")
    
    logger.info(f"Starting hyperparameter search for best lambdas. Candidates: {candidates}")
    best_lambdas = {'lambda_w': 0.1, 'lambda_a': 0.1} # Default
    best_score = float('inf')

    # Use a sample of the data for speed
    sample_df = df.sample(n=min(len(df), 300), random_state=42)
    target_edges = df.shape[1] * 1.5 # Target a graph that's slightly denser than one edge per node

    for lw in candidates:
        for la in candidates:
            try:
                model = from_pandas_dynamic(sample_df, p=p, lambda_w=lw, lambda_a=la, max_iter=100, w_threshold=0.05)
                num_edges = len(model.edges())

                # Score is the squared difference from the target number of edges
                score = (num_edges - target_edges) ** 2
                logger.info(f"  Testing (lw={lw:.3f}, la={la:.3f}): Found {num_edges} edges. Score={score:.2f}")

                if score < best_score:
                    best_score = score
                    best_lambdas = {'lambda_w': lw, 'lambda_a': la}

            except Exception as e:
                logger.warning(f"  Failed test for (lw={lw:.3f}, la={la:.3f}): {e}")
                continue

    logger.info(f"Best lambdas found: lambda_w={best_lambdas['lambda_w']:.3f}, lambda_a={best_lambdas['lambda_a']:.3f}")

    lambda_file = result_dir / "best_lambdas.json"
    with open(lambda_file, 'w') as f:
        json.dump(best_lambdas, f, indent=4)
    logger.info(f"Saved best lambdas to {lambda_file}")
    return best_lambdas['lambda_w'], best_lambdas['lambda_a']


def run_rolling_window_analysis(
    df_differenced: pd.DataFrame,
    p: int,
    window_size: int,
    stride: int,
    output_dir: Path,
    lambda_w: float,
    lambda_a: float
):
    """
    Performs rolling window causal discovery using DynoTEARS with fixed lambdas.
    """
    n_samples = len(df_differenced)
    all_weights = []

    logger.info(f"Starting rolling window analysis with FIXED lambdas (lambda_w={lambda_w}, lambda_a={lambda_a})")
    logger.info(f"Window size: {window_size}, Stride: {stride}")

    # Create subdirectories
    weights_dir = output_dir / "weights"
    history_dir = output_dir / "history"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = history_dir / "rolling_checkpoint.pkl"
    weights_history_file = history_dir / "weights_history.csv"

    start_window = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_window = checkpoint_data.get('last_completed_window', -1) + 1
                all_weights = checkpoint_data.get('all_weights', [])
                logger.info(f"Resuming from checkpoint. Starting at window {start_window}.")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Checkpoint file is corrupted, deleting and starting fresh: {e}")
            checkpoint_file.unlink()
            start_window = 0
            all_weights = []

    for i in range(start_window, n_samples - window_size + 1, stride):
        window_start_time = time.time()
        window_df = df_differenced.iloc[i : i + window_size]
        
        logger.info(f"Processing window {i // stride} (rows {i}-{i+window_size})...")

        try:
            # Run DynoTEARS with the fixed "Golden" lambdas
            sm = from_pandas_dynamic(
                window_df,
                p=p,
                lambda_w=lambda_w,
                lambda_a=lambda_a,
                max_iter=100, # Standard iterations for rolling window
                w_threshold=0.01,
                logger_prefix=f"[Window {i//stride}]"
            )

            # Process and store weights
            t_end = i + window_size - 1
            ts_end = window_df.index[t_end - i] if isinstance(window_df.index, pd.DatetimeIndex) else None
            t_center = i + window_size // 2
            ts_center = window_df.index[t_center - i] if isinstance(window_df.index, pd.DatetimeIndex) else None

            for u, v, data in sm.edges.data():
                parent_name, parent_lag_str = u.rsplit('_lag', 1)
                child_name, child_lag_str = v.rsplit('_lag', 1)
                
                all_weights.append({
                    'window_idx': i // stride,
                    't_end': t_end,
                    'ts_end': ts_end,
                    't_center': t_center,
                    'ts_center': ts_center,
                    'lag': int(parent_lag_str),
                    'parent_name': parent_name,
                    'child_name': child_name,
                    'weight': data['weight']
                })

        except Exception as e:
            logger.error(f"Failed to process window {i // stride}: {e}", exc_info=True)
            continue

        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'last_completed_window': i // stride, 'all_weights': all_weights}, f)

        logger.info(f"Window {i // stride} finished in {time.time() - window_start_time:.2f}s.")

    # Save final results
    if all_weights:
        weights_df = pd.DataFrame(all_weights)
        # A simplified version of the enhanced format
        weights_df['i'] = weights_df['child_name'].astype('category').cat.codes
        weights_df['j'] = weights_df['parent_name'].astype('category').cat.codes
        
        output_path = weights_dir / "weights_enhanced.csv"
        weights_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved final weights to {output_path}")
    else:
        logger.warning("No weights were generated during the analysis.")

def main():
    """Main execution function."""
    # Get file paths from environment variables set by launcher.py
    differenced_csv = os.getenv('INPUT_DIFFERENCED_CSV')
    lags_csv = os.getenv('INPUT_LAGS_CSV')
    result_dir = os.getenv('RESULT_DIR')

    # Get fixed lambdas from environment variables, with a default of 0.1
    # This makes the script configurable without changing the code.
    lambda_w = float(os.getenv('FIXED_LAMBDA_W', 0.1))
    lambda_a = float(os.getenv('FIXED_LAMBDA_A', 0.1))
    
    # Check for calibration mode - accept 'true', '1', 'yes', 'on'
    calibrate_env = os.getenv('CALIBRATE_LAMBDAS', 'false').lower()
    calibrate_mode = calibrate_env in ('true', '1', 'yes', 'on')

    if not all([differenced_csv, lags_csv, result_dir]):
        logger.error("Missing required environment variables (INPUT_DIFFERENCED_CSV, INPUT_LAGS_CSV, RESULT_DIR).")
        logger.error("This script should be run via launcher.py or with environment variables set manually.")
        sys.exit(1)

    logger.info(f"Input differenced data: {differenced_csv}")
    logger.info(f"Input lags data: {lags_csv}")
    logger.info(f"Output directory: {result_dir}")

    # Load data
    df_differenced = pd.read_csv(differenced_csv, index_col=0, parse_dates=True)
    df_lags = pd.read_csv(lags_csv)
    
    # Adaptive lambda scaling for high-dimensional problems
    n_vars = df_differenced.shape[1]
    if n_vars > 500:
        # For very high dimensional problems, scale lambdas
        scale_factor = np.sqrt(n_vars / 100)  # Scale based on sqrt of dimension ratio
        lambda_w_scaled = lambda_w * scale_factor
        lambda_a_scaled = lambda_a * scale_factor
        logger.warning(f"High-dimensional dataset detected ({n_vars} variables)")
        logger.warning(f"Scaling regularization: lambda_w {lambda_w:.3f} -> {lambda_w_scaled:.3f}")
        logger.warning(f"Scaling regularization: lambda_a {lambda_a:.3f} -> {lambda_a_scaled:.3f}")
        lambda_w = lambda_w_scaled
        lambda_a = lambda_a_scaled

    # Determine parameters
    p = int(df_lags['optimal_lag'].max()) if not df_lags.empty else 1
    logger.info("="*80)
    logger.info(f"🔍 CRITICAL PARAMETER: max_lag p = {p}")
    logger.info(f"   Source: {lags_csv}")
    logger.info(f"   Individual lags: {df_lags['optimal_lag'].tolist()}")
    logger.info("="*80)

    if calibrate_mode:
        logger.info("CALIBRATION MODE: Finding and saving best lambdas.")
        find_best_lambdas(df_differenced, p, Path(result_dir))
        logger.info("Calibration complete. Exiting.")
        sys.exit(0)

    # --- Execution Mode ---
    # Use a fixed window size and stride for consistency in tests.
    window_size = 100
    stride = 10
    
    run_rolling_window_analysis(df_differenced, p, window_size, stride, Path(result_dir), lambda_w, lambda_a)

if __name__ == "__main__":
    main()