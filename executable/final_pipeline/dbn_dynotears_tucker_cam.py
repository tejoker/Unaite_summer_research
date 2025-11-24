#!/usr/bin/env python3
"""
Tucker-CAM Rolling Window Analysis

Uses Tucker-decomposed Fast CAM-DAG for memory-efficient nonlinear causal discovery.

Option D Strategy: Zero L1 penalties (lambda_w=0, lambda_a=0) + post-hoc Top-K sparsification
"""

import os
import sys
import logging
import time
import pickle
import gc
import csv
from pathlib import Path

import numpy as np
import polars as pl
import torch
import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam
except ImportError:
    print("Error: Could not import Tucker-CAM. Make sure dynotears_tucker_cam.py is in the same directory.")
    sys.exit(1)

# Configure CPU parallelism for high-performance computing
# Use all available CPU cores for intra-op parallelism
num_threads = os.cpu_count()
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(max(1, num_threads // 2))

# Configure NVIDIA GPU optimizations for maximum performance
if torch.cuda.is_available():
    # Enable cuDNN auto-tuner to select fastest algorithms
    torch.backends.cudnn.benchmark = True

    # Enable TF32 on Ampere+ GPUs (3070/3080/3090/A100, etc.)
    # TF32 uses 19-bit precision: faster than FP32, more stable than FP16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def run_rolling_window_tucker_cam(
    data: np.ndarray,
    var_names: list,
    p: int,
    window_size: int,
    stride: int,
    output_dir: Path,
    rank_w: int = 20,
    rank_a: int = 10,
    n_knots: int = 5,
    lambda_smooth: float = 0.01,
    lambda_w: float = 0.0,
    lambda_a: float = 0.0,
    top_k: int = 10000
):
    """
    Performs rolling window causal discovery using Tucker-CAM with fixed parameters.

    Optimized for memory and performance:
    - Uses numpy arrays instead of pandas where possible
    - Creates minimal pandas DataFrames only when required by Tucker-CAM
    - Streams results directly to CSV without intermediate DataFrames

    Args:
        data: Differenced time series data as numpy array (n_samples, n_vars)
        var_names: List of variable names
        p: Maximum lag order
        window_size: Size of rolling window
        stride: Window stride
        output_dir: Output directory
        rank_w: Tucker rank for contemporaneous edges
        rank_a: Tucker rank for lagged edges
        n_knots: Number of B-spline knots
        lambda_smooth: Smoothness penalty
        lambda_w: L1 penalty contemporaneous (0 = Option D)
        lambda_a: L1 penalty lagged (0 = Option D)
        top_k: Keep top-k edges per window (post-hoc sparsification)
    """
    n_samples = len(data)
    import pandas as pd  # Only import when needed for Tucker-CAM compatibility
    all_weights = []

    # Use GPU if available, fallback to CPU with high parallelism
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        logger.info(f"Starting Tucker-CAM rolling window analysis (GPU mode)")
    else:
        logger.info(f"Starting Tucker-CAM rolling window analysis (CPU mode with {torch.get_num_threads()} threads)")

    logger.info(f"Starting Tucker-CAM rolling window analysis")
    logger.info(f"Tucker ranks: r_w={rank_w}, r_a={rank_a}")
    logger.info(f"P-splines: n_knots={n_knots}, lambda_smooth={lambda_smooth}")
    logger.info(f"Option D: lambda_w={lambda_w}, lambda_a={lambda_a}")
    logger.info(f"Post-hoc sparsity: Top-K={top_k} edges per window")
    logger.info(f"Window size: {window_size}, Stride: {stride}")
    logger.info(f"Device: {device}")

    # Create subdirectories
    weights_dir = output_dir / "weights"
    history_dir = output_dir / "history"
    weights_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = history_dir / "rolling_checkpoint_tucker.pkl"
    weights_history_file = history_dir / "weights_history_tucker.csv"

    start_window = 0
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                start_window = checkpoint_data.get('last_completed_window', -1) + 1
                logger.info(f"Resuming from checkpoint. Starting at window {start_window}.")
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Checkpoint file is corrupted, deleting and starting fresh: {e}")
            checkpoint_file.unlink()
            start_window = 0

    # Stream weights to CSV instead of keeping in RAM
    weights_csv_path = weights_dir / "weights_streaming.csv"
    csv_header_written = weights_csv_path.exists() and start_window > 0

    # Check for restart interval (to prevent memory accumulation)
    restart_interval = int(os.getenv('RESTART_INTERVAL', 1))
    max_windows_this_run = start_window + restart_interval

    # Convert window index to row index
    start_row = start_window * stride
    for i in range(start_row, n_samples - window_size + 1, stride):
        current_window = i // stride

        # Check if we should exit for restart
        if current_window >= max_windows_this_run:
            logger.info(f"Reached restart interval ({restart_interval} windows). Exiting for memory cleanup...")
            logger.info(f"Completed windows {start_window} to {current_window - 1}")
            break

        # Force garbage collection BEFORE window processing to prevent accumulation
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        window_start_time = time.time()
        # Extract window as numpy slice, convert to minimal DataFrame only for Tucker-CAM
        window_data = data[i : i + window_size]
        window_df = pd.DataFrame(window_data, columns=var_names)

        logger.info(f"Processing window {i // stride} (rows {i}-{i+window_size})...")

        try:
            # Run Tucker-CAM with Option D (zero L1 penalties)
            # Returns edge list directly (NO NetworkX - massive memory savings!)
            edges = from_pandas_dynamic_tucker_cam(
                window_df,
                p=p,
                rank_w=rank_w,
                rank_a=rank_a,
                n_knots=n_knots,
                lambda_smooth=lambda_smooth,
                lambda_w=lambda_w,
                lambda_a=lambda_a,
                max_iter=7,  # Phase 2: Reduced from 10 (early stopping usually triggers before max_iter)
                lr=0.05,  # Increased from default 0.01 for faster convergence
                w_threshold=0.0,  # No threshold during optimization
                h_tol=1e-3,  # Relaxed for Option D (post-hoc Top-K sparsification)
                device=device
            )

            # Post-hoc Top-K sparsification (Option D)
            if len(edges) > top_k:
                # Sort by absolute weight
                edges_sorted = sorted(edges, key=lambda e: abs(e[2]['weight']), reverse=True)
                edges = edges_sorted[:top_k]
                logger.info(f"  Sparsified: {len(edges_sorted)} -> {top_k} edges (Top-K)")

            # Process and store weights
            t_end = i + window_size - 1
            ts_end = window_df.index[t_end - i] if isinstance(window_df.index, pd.DatetimeIndex) else None
            t_center = i + window_size // 2
            ts_center = window_df.index[t_center - i] if isinstance(window_df.index, pd.DatetimeIndex) else None

            # Stream weights to CSV file (avoid accumulating in RAM)
            window_weights = []
            for u, v, edge_data in edges:
                parent_name, parent_lag_str = u.rsplit('_lag', 1)
                child_name, child_lag_str = v.rsplit('_lag', 1)

                window_weights.append({
                    'window_idx': i // stride,
                    't_end': t_end,
                    'ts_end': ts_end,
                    't_center': t_center,
                    'ts_center': ts_center,
                    'lag': int(parent_lag_str),
                    'parent_name': parent_name,
                    'child_name': child_name,
                    'weight': edge_data['weight']
                })

            # Append to CSV file using csv.DictWriter (faster than pandas DataFrame)
            # No DataFrame creation = less memory overhead
            with open(weights_csv_path, 'a', newline='') as f:
                if not csv_header_written:
                    fieldnames = ['window_idx', 't_end', 'ts_end', 't_center', 'ts_center',
                                  'lag', 'parent_name', 'child_name', 'weight']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    csv_header_written = True
                else:
                    fieldnames = ['window_idx', 't_end', 'ts_end', 't_center', 'ts_center',
                                  'lag', 'parent_name', 'child_name', 'weight']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(window_weights)
            del window_weights

            # Memory diagnostics
            window_elapsed = time.time() - window_start_time
            gc.collect()
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_gb = mem_info.rss / (1024**3)

            if device == 'cuda':
                gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"Window {i // stride} finished in {window_elapsed:.2f}s. "
                           f"RAM: {mem_gb:.2f}GB, GPU: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved")
            else:
                logger.info(f"Window {i // stride} finished in {window_elapsed:.2f}s. RAM: {mem_gb:.2f}GB")

            # Clean up memory after processing window
            # No NetworkX cleanup needed anymore (using edge lists)
            del edges, window_df

            if device == 'cuda':
                torch.cuda.synchronize()  # Wait for all operations to complete
                torch.cuda.empty_cache()   # Free cached memory including CUDA graphs
                torch.cuda.reset_peak_memory_stats()  # Reset memory tracking

        except Exception as e:
            logger.error(f"Failed to process window {i // stride}: {e}", exc_info=True)
            # Clean up GPU memory even on failure
            gc.collect()
            if device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            continue

        # Save checkpoint (no longer storing all_weights in RAM)
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'last_completed_window': i // stride}, f)

    # Save final results from streaming CSV
    if weights_csv_path.exists():
        logger.info("Reading streaming weights file...")
        weights_df = pd.read_csv(weights_csv_path)

        # Enhanced format with variable indices
        weights_df['i'] = weights_df['child_name'].astype('category').cat.codes
        weights_df['j'] = weights_df['parent_name'].astype('category').cat.codes

        output_path = weights_dir / "weights_enhanced.csv"
        weights_df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved final weights to {output_path}")

        # Statistics
        total_edges = len(weights_df)
        unique_windows = weights_df['window_idx'].nunique()
        avg_edges_per_window = total_edges / unique_windows if unique_windows > 0 else 0
        logger.info(f"Tucker-CAM Results: {total_edges:,} total edges, {avg_edges_per_window:.0f} avg/window")
    else:
        logger.warning("No weights were generated during the analysis.")


def main():
    """Main execution function."""
    # Get file paths from environment variables set by launcher.py
    differenced_csv = os.getenv('INPUT_DIFFERENCED_CSV')
    lags_csv = os.getenv('INPUT_LAGS_CSV')
    result_dir = os.getenv('RESULT_DIR')

    # Tucker-CAM parameters from environment
    rank_w = int(os.getenv('TUCKER_RANK_W', 20))
    rank_a = int(os.getenv('TUCKER_RANK_A', 10))
    n_knots = int(os.getenv('N_KNOTS', 5))
    lambda_smooth = float(os.getenv('LAMBDA_SMOOTH', 0.01))

    # Option D: zero L1 penalties (can override via env)
    lambda_w = float(os.getenv('LAMBDA_W', 0.0))
    lambda_a = float(os.getenv('LAMBDA_A', 0.0))

    # Top-K sparsification
    top_k = int(os.getenv('TOP_K', 10000))

    if not all([differenced_csv, lags_csv, result_dir]):
        logger.error("Missing required environment variables (INPUT_DIFFERENCED_CSV, INPUT_LAGS_CSV, RESULT_DIR).")
        logger.error("This script should be run via launcher.py or with environment variables set manually.")
        sys.exit(1)

    logger.info(f"Input differenced data: {differenced_csv}")
    logger.info(f"Input lags data: {lags_csv}")
    logger.info(f"Output directory: {result_dir}")

    # Load data using polars (much faster than pandas for large CSV files)
    logger.info("Loading data with polars (optimized CSV reader)...")
    df_diff_pl = pl.read_csv(differenced_csv)
    df_lags_pl = pl.read_csv(lags_csv)

    # Extract variable names from columns (skip first column which is index)
    var_names = df_diff_pl.columns[1:]  # Skip index column

    # Convert to numpy array (efficient, single memory copy)
    data_np = df_diff_pl.select(var_names).to_numpy().astype(np.float32)

    # Determine parameters
    p = int(df_lags_pl['optimal_lag'].max()) if df_lags_pl.height > 0 else 1
    n_vars = data_np.shape[1]

    logger.info(f"Data loaded: {data_np.shape[0]} samples, {n_vars} variables")

    logger.info("="*80)
    logger.info(f"Tucker-CAM Configuration:")
    logger.info(f"  Variables: {n_vars}")
    logger.info(f"  Max lag p: {p}")
    logger.info(f"  Tucker ranks: r_w={rank_w}, r_a={rank_a}")
    logger.info(f"  P-splines: n_knots={n_knots}, lambda_smooth={lambda_smooth}")
    logger.info(f"  Option D: lambda_w={lambda_w}, lambda_a={lambda_a}")
    logger.info(f"  Top-K: {top_k} edges/window")
    logger.info("="*80)

    # Rolling window parameters
    window_size = int(os.getenv('WINDOW_SIZE', 100))
    stride = int(os.getenv('STRIDE', 10))

    run_rolling_window_tucker_cam(
        data_np, var_names, p, window_size, stride, Path(result_dir),
        rank_w=rank_w, rank_a=rank_a, n_knots=n_knots,
        lambda_smooth=lambda_smooth, lambda_w=lambda_w, lambda_a=lambda_a,
        top_k=top_k
    )

if __name__ == "__main__":
    main()
