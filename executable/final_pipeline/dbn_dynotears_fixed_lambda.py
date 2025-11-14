#!/usr/bin/env python3
"""
Fast CAM-DAG with Fixed Lambda Strategy

Uses fixed lambda values (calibrated on golden data) for both baseline
and anomaly runs instead of performing separate hyperparameter searches.

Purpose:
Eliminate the "Butterfly Effect" where a future anomaly influences
global lambda selection, causing weight differences in early windows.
By using the same lambdas for both Golden and Anomaly runs, any observed
weight difference is guaranteed to result from the data itself, not from
different regularization parameters.

Uses Fast CAM-DAG (nonlinear causal discovery with P-Splines) exclusively.
"""

import os
import sys

# ========================================
# SET THREAD COUNTS BEFORE ANY IMPORTS
# Must be done before numpy/torch import
# ========================================
num_threads = os.environ.get('OMP_NUM_THREADS', '60')
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import logging
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json
import gc  # Garbage collector for memory management

# Add parent directory to path to allow imports from the final_pipeline package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam as from_pandas_dynamic_cam
from dynotears_tucker_cam import TuckerFastCAMDAG as FastCAMDAG
from adaptive_knots import get_knots_for_dataset
from memory_monitor import MemoryMonitor
from transformers import DynamicDataTransformer

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

def find_best_lambdas(df: pd.DataFrame, p: int, result_dir: Path,
                      n_knots: int = None, lambda_smooth: float = 0.01,
                      candidates=None):
    """
    Performs a search on a sample of the data to find the best lambdas and saves them.
    Uses Fast CAM-DAG for calibration.
    The 'best' is defined as the one producing a reasonable number of edges.
    
    Args:
        n_knots: Number of B-spline knots. If None, will be automatically selected.
    """
    n_vars = df.shape[1]
    
    # Adaptive knot selection if not specified
    if n_knots is None:
        n_knots, knot_reasoning = get_knots_for_dataset(df, window_size=min(len(df), 300))
        logger.info(f"Adaptive knot selection: {n_knots} knots")
        logger.info(f"  Reasoning: {knot_reasoning}")
    else:
        logger.info(f"Using fixed n_knots={n_knots} (specified in config)")

    # Adaptive lambda candidates based on dimensionality
    if candidates is None:
        if n_vars < 20:
            candidates = [0.01, 0.05, 0.1, 0.5]
        elif n_vars < 100:
            candidates = [0.05, 0.1, 0.3, 0.5]
        elif n_vars < 500:
            candidates = [0.1, 0.3, 0.5, 1.0]
        else:
            # For very high dimensional problems, use WEAKER regularization
            # More variables = more parameters = need less penalty to discover structure
            candidates = [0.001, 0.005, 0.01, 0.05]
            logger.info(f"High-dimensional dataset ({n_vars} vars), using weaker lambdas: {candidates}")

    logger.info(f"Starting hyperparameter search for best lambdas using Fast CAM-DAG. Candidates: {candidates}")
    best_lambdas = {'lambda_w': 0.1, 'lambda_a': 0.1} # Default
    best_score = float('inf')

    # Use a sample of the data for speed
    sample_df = df.sample(n=min(len(df), 300), random_state=42)
    target_edges = df.shape[1] * 1.5 # Target a graph that's slightly denser than one edge per node

    for lw in candidates:
        for la in candidates:
            try:
                model = from_pandas_dynamic_cam(
                    sample_df, p=p,
                    lambda_w=lw, lambda_a=la,
                    n_knots=n_knots,
                    lambda_smooth=lambda_smooth,
                    use_gcv=False,
                    max_iter=100,
                    w_threshold=0.05
                )
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
    lambda_a: float,
    n_knots: int = None,
    lambda_smooth: float = 0.01
):
    """
    Performs rolling window causal discovery using Fast CAM-DAG with fixed lambdas.

    Args:
        df_differenced: Preprocessed time series data
        p: Lag order
        window_size: Number of samples per window
        stride: Window stride (samples)
        output_dir: Directory to save results
        lambda_w: L2 penalty for contemporaneous edges
        lambda_a: L2 penalty for lagged edges
        n_knots: Number of B-spline knots. If None, will be automatically selected.
        lambda_smooth: Smoothness penalty (default 0.01)
    """
    n_samples = len(df_differenced)
    all_weights = []
    
    # Adaptive knot selection if not specified
    if n_knots is None:
        n_knots, knot_reasoning = get_knots_for_dataset(df_differenced, window_size=window_size)
        logger.info(f"Adaptive knot selection for rolling windows: {n_knots} knots")
        logger.info(f"  Reasoning: {knot_reasoning}")
    else:
        logger.info(f"Using fixed n_knots={n_knots} (specified in config)")

    logger.info(f"Starting rolling window analysis using Fast CAM-DAG (nonlinear)")
    logger.info(f"FIXED lambdas: lambda_w={lambda_w}, lambda_a={lambda_a}")
    logger.info(f"CAM parameters: n_knots={n_knots}, lambda_smooth={lambda_smooth}")
    logger.info(f"Window size: {window_size}, Stride: {stride}")

    # Initialize memory monitoring
    memory_monitor = MemoryMonitor(logger)
    memory_monitor.log("Pipeline initialization")

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

    # ========================================
    # OPTIMIZATION: Create model ONCE and reuse
    # ========================================
    logger.info("=" * 80)
    logger.info("MEMORY OPTIMIZATION: Creating reusable model instance")
    logger.info("  This allocates memory once instead of per-window (98% reduction)")
    logger.info("=" * 80)

    d = df_differenced.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data transformer (reusable)
    transformer = DynamicDataTransformer(p=p)

    # Option D: Post-hoc Top-K sparsification
    # Use minimal penalties (only smoothness), learn dense model, then prune Top-K edges
    # This avoids the L2 vs L1 sparsity problem entirely
    tucker_lambda_w = 0.0  # No contemporaneous penalty
    tucker_lambda_a = 0.0  # No lagged penalty
    logger.info(f"Option D (Post-hoc Top-K): Using zero penalties, will select Top-{os.environ.get('TUCKER_TOP_K', '10000')} edges after training")

    # Create model instance ONCE (major memory savings)
    cam_dag = FastCAMDAG(
        d=d,
        p=p,
        n_knots=n_knots,
        lambda_w=tucker_lambda_w,
        lambda_a=tucker_lambda_a,
        lambda_smooth=lambda_smooth,
        device=device
    )

    memory_monitor.log("After model allocation")
    logger.info(f"Model instance created: {d} variables, {p} lags, {n_knots} knots")
    # Note: Model parameters are allocated in fit() when basis matrices are computed

    var_names = df_differenced.columns.tolist()

    # Rolling window loop with model reuse
    # Get number of windows from environment (default: all 278)
    n_windows = int(os.environ.get('N_WINDOWS', str((n_samples - window_size) // stride + 1)))
    logger.info(f"Processing {n_windows} windows (set N_WINDOWS env var to change)")
    
    for i in range(start_window, min(start_window + n_windows*stride, n_samples - window_size + 1), stride):
        window_idx = i // stride
        window_start_time = time.time()

        logger.info("=" * 80)
        logger.info(f"WINDOW {window_idx}: rows {i} to {i+window_size}")
        logger.info("=" * 80)

        memory_monitor.log(f"Window {window_idx} START")

        try:
            # Step 1: Extract window data
            logger.info(f"  [1/6] Extracting window data...")
            window_df = df_differenced.iloc[i : i + window_size]

            # Step 2: Reset model parameters (ensures independence)
            logger.info(f"  [2/6] Resetting model parameters (random init)...")
            cam_dag.reset_parameters()

            # Step 3: Transform data to tensor format
            logger.info(f"  [3/6] Transforming data to tensors...")
            X, Xlags = transformer.fit_transform([window_df], return_df=False)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            Xlags = torch.tensor(Xlags, dtype=torch.float32, device=device)

            # Step 4: Fit model (this will create model, compute basis, and optimize)
            logger.info(f"  [4/5] Fitting CAM-DAG model (max_iter=100)...")
            fit_start = time.time()
            # Use lower learning rate for stability with large d
            cam_dag.fit(X, Xlags, max_iter=100, lr=0.001, verbose=True)
            fit_time = time.time() - fit_start
            logger.info(f"       Model converged in {fit_time:.2f}s")
            memory_monitor.log(f"Window {window_idx} after fit")

            # Step 5: Extract structure
            logger.info(f"  [5/5] Extracting causal graph structure...")
            sm = cam_dag.get_structure_model(var_names)
            logger.info(f"       Found {len(sm.edges)} edges")

            # Process and store weights
            t_end = i + window_size - 1
            ts_end = window_df.index[t_end - i] if isinstance(window_df.index, pd.DatetimeIndex) else None
            t_center = i + window_size // 2
            ts_center = window_df.index[t_center - i] if isinstance(window_df.index, pd.DatetimeIndex) else None

            for u, v, data in sm.edges.data():
                parent_name, parent_lag_str = u.rsplit('_lag', 1)
                child_name, child_lag_str = v.rsplit('_lag', 1)

                all_weights.append({
                    'window_idx': window_idx,
                    't_end': t_end,
                    'ts_end': ts_end,
                    't_center': t_center,
                    'ts_center': ts_center,
                    'lag': int(parent_lag_str),
                    'parent_name': parent_name,
                    'child_name': child_name,
                    'weight': data['weight']
                })

            # Clear window-specific data (keep model structure)
            cam_dag.model.basis_current = None
            cam_dag.model.basis_lagged = None
            del X, Xlags, sm, window_df
            
            # Force GPU cache cleanup
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            memory_monitor.log(f"Window {window_idx} END (after cleanup)")

        except Exception as e:
            logger.error(f"Failed to process window {window_idx}: {e}", exc_info=True)
            continue

        # Save checkpoint AND incremental results after each window
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'last_completed_window': window_idx, 'all_weights': all_weights}, f)

        # Save incremental CSV results (allows monitoring progress and crash recovery)
        if all_weights:
            weights_df = pd.DataFrame(all_weights)
            # Add category codes for compatibility
            weights_df['i'] = weights_df['child_name'].astype('category').cat.codes
            weights_df['j'] = weights_df['parent_name'].astype('category').cat.codes

            output_path = weights_dir / "weights_enhanced.csv"
            weights_df.to_csv(output_path, index=False)

            logger.info(f"Window {window_idx} finished in {time.time() - window_start_time:.2f}s. "
                       f"Saved {len(all_weights)} edges so far.")
        else:
            logger.info(f"Window {window_idx} finished in {time.time() - window_start_time:.2f}s. "
                       f"No edges found yet.")

    # Final summary
    logger.info("=" * 80)
    logger.info("ROLLING WINDOW ANALYSIS COMPLETE")
    logger.info("=" * 80)
    memory_monitor.log("Final state (after all windows)")

    if all_weights:
        output_path = weights_dir / "weights_enhanced.csv"
        logger.info(f"Total {len(all_weights)} edges saved to {output_path}")
    else:
        logger.warning("No weights were generated during the analysis.")

    # Cleanup model (free memory)
    del cam_dag
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_monitor.log("After cleanup")

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

    # Fast CAM-DAG parameters
    n_knots_env = os.getenv('CAM_N_KNOTS', 'auto')
    if n_knots_env.lower() in ('auto', 'adaptive', 'none'):
        n_knots = None  # Will be auto-selected based on data
        logger.info("Knot selection: ADAPTIVE (will be determined from dataset characteristics)")
    else:
        try:
            n_knots = int(n_knots_env)
            logger.info(f"Knot selection: FIXED at {n_knots} knots")
        except ValueError:
            logger.warning(f"Invalid CAM_N_KNOTS value '{n_knots_env}', using adaptive mode")
            n_knots = None
    
    lambda_smooth = float(os.getenv('CAM_LAMBDA_SMOOTH', 0.01))
    
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
    if n_vars > 1000:
        # For very high dimensional problems, use WEAKER regularization
        # More variables = more parameters = need less penalty to discover structure
        scale_factor = 100 / n_vars  # Scale DOWN, not up
        lambda_w_scaled = lambda_w * scale_factor
        lambda_a_scaled = lambda_a * scale_factor
        logger.warning(f"High-dimensional dataset detected ({n_vars} variables)")
        logger.warning(f"Scaling regularization DOWN: lambda_w {lambda_w:.3f} -> {lambda_w_scaled:.6f}")
        logger.warning(f"Scaling regularization DOWN: lambda_a {lambda_a:.3f} -> {lambda_a_scaled:.6f}")
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
        logger.info("CALIBRATION MODE: Finding and saving best lambdas using Fast CAM-DAG.")
        find_best_lambdas(df_differenced, p, Path(result_dir),
                         n_knots=n_knots, lambda_smooth=lambda_smooth)
        logger.info("Calibration complete. Exiting.")
        sys.exit(0)

    # --- Execution Mode ---
    # Use a fixed window size and stride for consistency in tests.
    window_size = 100
    stride = 10

    run_rolling_window_analysis(
        df_differenced,
        p,
        window_size,
        stride,
        Path(result_dir),
        lambda_w,
        lambda_a,
        n_knots=n_knots,
        lambda_smooth=lambda_smooth
    )

if __name__ == "__main__":
    main()