#!/usr/bin/env python3
"""
DynoTEARS DBN Analysis WITHOUT Rolling Windows - Single Global Analysis

This script runs the full DynoTEARS optimization algorithm on the entire dataset
without rolling windows, providing a single global causal graph.
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
import os
import sys
import csv
import time
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from dynotears import from_pandas_dynamic, extract_matrices, generate_histogram_and_kde

# --------------------------- Configuration ---------------------------
DATA_FILE = os.getenv('INPUT_DIFFERENCED_CSV', "differenced_stationary_series.csv")
OPTIMAL_LAGS_FILE = os.getenv('INPUT_LAGS_CSV', "optimal_lags.csv")
MI_MASK_FILE = os.getenv('INPUT_MI_MASK_CSV', "mi_mask_edges.csv")
RESULT_DIR = os.getenv('RESULT_DIR')
if RESULT_DIR:
    OUTPUT_DIR = os.path.join(RESULT_DIR, 'weights')
    WEIGHTS_HISTORY_CSV = os.path.join(RESULT_DIR, 'history', "weights_history.csv")
else:
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', "dynotears_results")
    WEIGHTS_HISTORY_CSV = os.getenv('WEIGHTS_HISTORY_CSV', "weights_history.csv")

# DynoTEARS parameters for single global optimization
LAMBDA_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0]  # Regularization candidates
MAX_ITER_GLOBAL = 1000     # Higher iterations for global optimization
H_TOL = 1e-8               # Stricter acyclicity tolerance
LOSS_TOL = 1e-6            # Stricter loss convergence tolerance

# --------------------------- Logging Setup ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
if RESULT_DIR:
    os.makedirs(os.path.join(RESULT_DIR, 'history'), exist_ok=True)

logger = logging.getLogger("dbn_dynotears_no_rolling")
logger.setLevel(logging.INFO)

run_id = time.strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(OUTPUT_DIR, f"dynotears_no_rolling_{run_id}.log")
file_handler = logging.FileHandler(log_file_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

# Reduce verbosity of external loggers
logging.getLogger('dynotears').setLevel(logging.WARNING)
logging.getLogger('structuremodel').setLevel(logging.WARNING)

def load_mi_mask(mi_mask_file, variable_names, L):
    """Load MI mask from CSV file with proper variable name matching"""
    try:
        if not os.path.exists(mi_mask_file):
            logger.warning(f"MI mask file not found: {mi_mask_file}, allowing all edges")
            d = len(variable_names)
            return np.ones((d, d, L + 1), dtype=bool)

        mask_df = pd.read_csv(mi_mask_file)
        logger.info(f"Loaded MI mask from {mi_mask_file} with {len(mask_df)} entries")

        d = len(variable_names)
        # Initialize mask (False = not allowed, True = allowed)
        mask = np.zeros((d, d, L + 1), dtype=bool)

        # Create name to index mapping
        name_to_idx = {name: i for i, name in enumerate(variable_names)}

        for _, row in mask_df.iterrows():
            if 'parent' in mask_df.columns and 'child' in mask_df.columns:
                parent_name = row['parent']
                child_name = row['child']
                lag = int(row['lag'])
                allowed = bool(int(row['allowed']))

                # Skip if lag is out of range
                if lag > L:
                    continue

                # Find indices using variable names
                parent_idx = None
                child_idx = None

                for idx, var_name in enumerate(variable_names):
                    if parent_name in var_name or var_name in parent_name:
                        parent_idx = idx
                    if child_name in var_name or var_name in child_name:
                        child_idx = idx

                if parent_idx is not None and child_idx is not None:
                    if 0 <= parent_idx < d and 0 <= child_idx < d and 0 <= lag <= L:
                        mask[child_idx, parent_idx, lag] = allowed

        allowed_edges = np.sum(mask)
        total_edges = d * d * (L + 1)
        logger.info(f"MI mask allows {allowed_edges}/{total_edges} edges ({100*allowed_edges/total_edges:.1f}%)")

        return mask

    except Exception as e:
        logger.error(f"Error loading MI mask: {e}, allowing all edges")
        d = len(variable_names)
        return np.ones((d, d, L + 1), dtype=bool)

def main():
    """Main function for DynoTEARS without rolling windows"""

    logger.info("=== DYNOTEARS WITHOUT ROLLING WINDOWS ===")
    logger.info("Single global optimization on entire dataset")

    # Load data
    logger.info(f"Loading data from {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file not found: {DATA_FILE}")
        return 1

    df = pd.read_csv(DATA_FILE, index_col=0)
    logger.info(f"Loaded data: {df.shape} (samples x variables)")

    # Load optimal lags
    logger.info(f"Loading optimal lags from {OPTIMAL_LAGS_FILE}")
    if os.path.exists(OPTIMAL_LAGS_FILE):
        lags_df = pd.read_csv(OPTIMAL_LAGS_FILE)
        if 'optimal_lag' in lags_df.columns:
            L = int(lags_df['optimal_lag'].max())
        else:
            L = 1
        logger.info(f"Using maximum lag order L={L}")
    else:
        L = 1
        logger.warning(f"Lags file not found, using default L={L}")

    # Load MI mask
    variable_names = list(df.columns)
    d = len(variable_names)
    mask = load_mi_mask(MI_MASK_FILE, variable_names, L)

    # Prepare data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    data_df = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)

    logger.info(f"Data standardized: mean≈0, std≈1")
    logger.info(f"Starting DynoTEARS global optimization...")
    logger.info(f"Parameters: d={d}, L={L}, n_samples={len(data_df)}")
    logger.info(f"Lambda candidates: {LAMBDA_CANDIDATES}")
    logger.info(f"Max iterations: {MAX_ITER_GLOBAL}")

    start_time = time.time()

    try:
        # Run DynoTEARS on entire dataset (no rolling windows)
        best_result = None
        best_lambda = None
        best_score = float('inf')

        for lambda_val in LAMBDA_CANDIDATES:
            logger.info(f"Trying lambda_w={lambda_val}, lambda_a={lambda_val}")

            try:
                # Run DynoTEARS optimization
                result = from_pandas_dynamic(
                    data_df,
                    p=L,
                    lambda_w=lambda_val,
                    lambda_a=lambda_val,
                    max_iter=MAX_ITER_GLOBAL,
                    h_tol=H_TOL,
                    w_threshold=0.3,
                    save_every=50
                )

                if result is not None:
                    # Extract loss or use a heuristic (number of edges)
                    variable_names = list(data_df.columns)
                    W, A_list = extract_matrices(result, variable_names, L)

                    # Convert torch tensors to numpy
                    W_np = W.detach().cpu().numpy()
                    A_list_np = [A.detach().cpu().numpy() for A in A_list]

                    total_edges = np.sum(np.abs(W_np) > 1e-6) + sum(np.sum(np.abs(A) > 1e-6) for A in A_list_np)

                    # Use negative log likelihood or edge count as score
                    # For simplicity, use regularized edge count as score
                    score = total_edges + lambda_val * (np.sum(np.abs(W_np)) + sum(np.sum(np.abs(A)) for A in A_list_np))

                    logger.info(f"Lambda {lambda_val}: edges={total_edges}, score={score:.6f}")

                    if score < best_score:
                        best_score = score
                        best_result = result
                        best_lambda = lambda_val
                        logger.info(f"New best lambda: {lambda_val} (score={score:.6f})")
                else:
                    logger.warning(f"Lambda {lambda_val}: optimization failed or incomplete")

            except Exception as e:
                logger.error(f"Lambda {lambda_val}: error during optimization: {e}")
                continue

        if best_result is None:
            logger.error("All lambda values failed. Cannot proceed.")
            return 1

        logger.info(f"Best lambda: {best_lambda} (loss={best_score:.6f})")

        # Extract matrices
        variable_names = list(data_df.columns)
        W, A_list = extract_matrices(best_result, variable_names, L)

        # Convert torch tensors to numpy
        W = W.detach().cpu().numpy()
        A_list = [A.detach().cpu().numpy() for A in A_list]

        logger.info(f"Extracted matrices: W shape={W.shape}, A_list length={len(A_list)}")

        # Apply MI mask if provided
        if mask is not None:
            logger.info(f"Applying MI mask: mask shape={mask.shape}, W shape={W.shape}, A_list length={len(A_list)}")

            # Apply mask to W (contemporaneous, lag=0)
            if mask.shape[2] > 0:  # Check if lag 0 exists in mask
                W_masked = W * mask[:, :, 0]
            else:
                W_masked = W
                logger.warning("No lag 0 in MI mask, keeping W unchanged")

            # Apply mask to A matrices
            A_list_masked = []
            for lag_idx, A in enumerate(A_list):
                lag = lag_idx + 1  # A_list[0] corresponds to lag 1
                if lag < mask.shape[2]:  # Check if this lag exists in mask
                    A_masked = A * mask[:, :, lag]
                    logger.debug(f"Applied mask to A[{lag_idx}] (lag {lag})")
                else:
                    A_masked = A  # No mask available for this lag
                    logger.debug(f"No mask for A[{lag_idx}] (lag {lag}), keeping unchanged")
                A_list_masked.append(A_masked)

            W = W_masked
            A_list = A_list_masked
            logger.info("Applied MI mask to results")

        # Count edges
        w_edges = np.sum(np.abs(W) > 1e-6)
        a_edges = sum(np.sum(np.abs(A) > 1e-6) for A in A_list)
        total_edges = w_edges + a_edges

        logger.info(f"Final results: W edges={w_edges}, A edges={a_edges}, total={total_edges}")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save weight matrices
        W_file = os.path.join(OUTPUT_DIR, f'W_no_rolling_{timestamp}.npy')
        np.save(W_file, W)
        logger.info(f"Saved W matrix to {W_file}")

        for i, A in enumerate(A_list):
            A_file = os.path.join(OUTPUT_DIR, f'A_lag{i+1}_no_rolling_{timestamp}.npy')
            np.save(A_file, A)
        logger.info(f"Saved {len(A_list)} A matrices")

        # Save detailed results CSV
        variable_names = list(df.columns)
        rows = []

        # Add W entries (lag 0)
        for i in range(d):
            for j in range(d):
                if abs(W[i, j]) > 1e-6:
                    rows.append({
                        'parent': variable_names[j],
                        'child': variable_names[i],
                        'lag': 0,
                        'weight': W[i, j],
                        'abs_weight': abs(W[i, j]),
                        'method': 'DynoTEARS_no_rolling',
                        'lambda': best_lambda,
                        'timestamp': timestamp
                    })

        # Add A entries (lag >= 1)
        for lag_idx, A in enumerate(A_list):
            lag = lag_idx + 1
            for i in range(d):
                for j in range(d):
                    if abs(A[i, j]) > 1e-6:
                        rows.append({
                            'parent': variable_names[j],
                            'child': variable_names[i],
                            'lag': lag,
                            'weight': A[i, j],
                            'abs_weight': abs(A[i, j]),
                            'method': 'DynoTEARS_no_rolling',
                            'lambda': best_lambda,
                            'timestamp': timestamp
                        })

        if rows:
            results_df = pd.DataFrame(rows)
            results_file = os.path.join(OUTPUT_DIR, f'weights_no_rolling_{timestamp}.csv')
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved {len(rows)} edges to {results_file}")
        else:
            logger.warning("No significant edges found")

        # Save summary
        summary = {
            'method': 'DynoTEARS_no_rolling',
            'timestamp': timestamp,
            'data_shape': list(df.shape),
            'lag_order': L,
            'best_lambda': best_lambda,
            'best_loss': best_score,
            'w_edges': int(w_edges),
            'a_edges': int(a_edges),
            'total_edges': int(total_edges),
            'runtime_seconds': time.time() - start_time,
            'max_iterations': MAX_ITER_GLOBAL
        }

        import json
        summary_file = os.path.join(OUTPUT_DIR, f'summary_no_rolling_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

        total_time = time.time() - start_time
        logger.info(f"=== DYNOTEARS NO ROLLING COMPLETE ===")
        logger.info(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Results saved to: {OUTPUT_DIR}")

        return 0

    except Exception as e:
        logger.error(f"Fatal error in DynoTEARS no rolling: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())