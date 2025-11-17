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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam
except ImportError:
    print("Error: Could not import Tucker-CAM. Make sure dynotears_tucker_cam.py is in the same directory.")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def run_rolling_window_tucker_cam(
    df_differenced: pd.DataFrame,
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

    Args:
        df_differenced: Differenced time series data
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
    n_samples = len(df_differenced)
    all_weights = []

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            # Run Tucker-CAM with Option D (zero L1 penalties)
            sm = from_pandas_dynamic_tucker_cam(
                window_df,
                p=p,
                rank_w=rank_w,
                rank_a=rank_a,
                n_knots=n_knots,
                lambda_smooth=lambda_smooth,
                lambda_w=lambda_w,
                lambda_a=lambda_a,
                max_iter=100,
                w_threshold=0.0,  # No threshold during optimization
                device=device
            )

            # Extract all edges
            edges = list(sm.edges.data())

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

            for u, v, data in edges:
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

    # Load data
    df_differenced = pd.read_csv(differenced_csv, index_col=0, parse_dates=True)
    df_lags = pd.read_csv(lags_csv)

    # Determine parameters
    p = int(df_lags['optimal_lag'].max()) if not df_lags.empty else 1
    n_vars = df_differenced.shape[1]

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
        df_differenced, p, window_size, stride, Path(result_dir),
        rank_w=rank_w, rank_a=rank_a, n_knots=n_knots,
        lambda_smooth=lambda_smooth, lambda_w=lambda_w, lambda_a=lambda_a,
        top_k=top_k
    )

if __name__ == "__main__":
    main()
