#!/usr/bin/env python3
"""
Enhanced DynoTears DBN Analysis with Detailed Time Mapping

TIME MAPPING RULE:
================================================================================
Each exported record describes the effect at t_end on child i.
- If lag = L > 0, the source sample time is t_src = t_end - L on parent j.
- If lag = 0, source and effect are the same timestamp t_end.

MATH SUMMARY:
We are exporting estimates of W (lag 0) and A(L) (lags 1…p). Each exported 
nonzero weight corresponds to:
- x_i(t_end) ⊃ W[i,j]⋅x_j(t_end) if L=0,
- x_i(t_end) ⊃ A(L)[i,j]⋅x_j(t_end-L) if L≥1.

CANONICAL TIME USAGE:
- ts_center: Use for "window-level" storytelling (what period the edge belongs to)
- ts_end ± lag: Use for pinpointing the exact sample timestamp
================================================================================
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
MI_MASK_FILE = os.getenv('INPUT_MI_MASK_CSV', "mi_mask_edges.csv")  # Add MI mask input
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
LAMBDA_CANDIDATES = [0.05, 0.1, 0.5, 1.0, 2.0]  # more conservative regularization 
MAX_ITER_HYPER = 200    # max iterations for hyperparameter tuning
MAX_ITER_WINDOW = 300   # reduced max iterations for rolling window optimization
H_TOL = 1e-6            # more relaxed acyclicity tolerance 
LOSS_TOL = 1e-4         # more relaxed loss convergence tolerance for rolling windows
SAVE_EVERY = 10         # save frequency (iterations) for inner optimization

# --------------------------- Logging Setup ---------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
if RESULT_DIR:
    os.makedirs(os.path.join(RESULT_DIR, 'history'), exist_ok=True)
logger = logging.getLogger("dbn_dynotears")
logger.setLevel(logging.INFO)
# Prepare log file (append if resuming, otherwise new)
resume_mode = False
run_id = time.strftime("%Y%m%d_%H%M%S")
if os.path.exists(CHECKPOINT_FILE):
    try:
        ckpt_temp = pickle.load(open(CHECKPOINT_FILE, "rb"))
        run_id = ckpt_temp.get("timestamp", run_id)
        resume_mode = True
    except Exception:
        pass
log_file_path = os.path.join(OUTPUT_DIR, f"dynotears_{run_id}.log")
file_handler = logging.FileHandler(log_file_path, mode='a' if resume_mode else 'w')
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
logging.getLogger('dynotears').handlers.clear()
logging.getLogger('dynotears').setLevel(logging.WARNING)
logging.getLogger('structuremodel').setLevel(logging.WARNING)

# --------------------------- Helper Functions ---------------------------
def get_gpu_memory_usage():
    """Get current GPU memory usage in MB (NVIDIA GPUs only)."""
    try:
        import subprocess
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(result.strip().split('\n')[0])
    except Exception:
        return 0

def log_gpu_stats():
    """Log current GPU memory usage and clear cache."""
    if torch.cuda.is_available():
        mem_used = get_gpu_memory_usage()
        total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        logger.info(f"GPU Memory: {mem_used} MB / {total} MB used")
        torch.cuda.empty_cache()

def load_optimal_lags(path):
    """Load optimal lag value from CSV (if available). Supports a single lag or per-variable lags."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    if df.empty:
        return None
    if 'optimal_lag' in df.columns and df['optimal_lag'].notna().all():
        if df['optimal_lag'].nunique() == 1:
            return int(df['optimal_lag'].iloc[0])
        return df.set_index('variable')['optimal_lag'].to_dict()
    return None

def load_mi_mask(path, variable_names, max_lag):
    """
    Load MI mask from CSV and convert to numpy array format
    Expected CSV format: parent, child, lag, allowed
    Returns: numpy array of shape (d, d, max_lag + 1) with boolean values
    """
    logger.info(f"Loading MI mask from {path}")
    try:
        if not os.path.exists(path):
            logger.warning(f"MI mask file {path} not found. Using dense (all connections allowed) mask.")
            d = len(variable_names)
            return np.ones((d, d, max_lag + 1), dtype=bool)
        
        df_mask = pd.read_csv(path)
        
        # Create variable name to index mapping
        var_to_idx = {name: idx for idx, name in enumerate(variable_names)}
        d = len(variable_names)
        
        # Initialize mask with all False
        mask = np.zeros((d, d, max_lag + 1), dtype=bool)
        
        # Fill mask based on CSV data
        for _, row in df_mask.iterrows():
            if row['parent'] in var_to_idx and row['child'] in var_to_idx:
                parent_idx = var_to_idx[row['parent']]
                child_idx = var_to_idx[row['child']]
                lag = int(row['lag'])
                if 0 <= lag <= max_lag:
                    mask[child_idx, parent_idx, lag] = bool(row['allowed'])
        
        allowed_edges = np.sum(mask)
        total_edges = mask.size
        logger.info(f"MI mask loaded: {allowed_edges}/{total_edges} edges allowed ({100*allowed_edges/total_edges:.1f}%)")
        
        return mask
        
    except Exception as e:
        
        logger.error(f"Error loading MI mask: {e}")
        logger.warning("Using dense (all connections allowed) mask as fallback.")
        d = len(variable_names)
        return np.ones((d, d, max_lag + 1), dtype=bool)

def apply_mi_mask_to_matrices(W_est, A_est_list, mi_mask):
    """
    Apply MI mask to estimated weight matrices
    W_est: contemporaneous weights (lag 0)
    A_est_list: list of autoregressive matrices (lags 1, 2, ...)
    mi_mask: boolean mask of shape (d, d, max_lag + 1)
    """
    logger.info("Applying MI mask to estimated matrices")
    
    # Apply mask to contemporaneous weights (lag 0)
    if mi_mask.shape[2] > 0:  # if lag 0 exists in mask
        if hasattr(W_est, 'clone'):
            W_masked = W_est.clone()
        else:
            W_masked = W_est.copy()
        W_masked[~mi_mask[:, :, 0]] = 0.0
        if hasattr(W_masked, 'sum'):
            masked_w_edges = int((W_masked != 0).sum().item())
            original_w_edges = int((W_est != 0).sum().item())
        else:
            masked_w_edges = int(np.sum(W_masked != 0))
            original_w_edges = int(np.sum(W_est != 0))
        logger.info(f"Contemporaneous matrix: {masked_w_edges}/{original_w_edges} edges kept after MI masking")
    else:
        W_masked = W_est
    
    # Apply mask to autoregressive matrices (lags 1, 2, ...)
    A_masked_list = []
    for lag_idx, A_est in enumerate(A_est_list):
        lag = lag_idx + 1  # A_est_list[0] corresponds to lag 1
        if lag < mi_mask.shape[2]:
            if hasattr(A_est, 'clone'):
                A_masked = A_est.clone()
            else:
                A_masked = A_est.copy()
            A_masked[~mi_mask[:, :, lag]] = 0.0
            if hasattr(A_masked, 'sum'):
                masked_a_edges = int((A_masked != 0).sum().item())
                original_a_edges = int((A_est != 0).sum().item())
            else:
                masked_a_edges = int(np.sum(A_masked != 0))
                original_a_edges = int(np.sum(A_est != 0))
            logger.info(f"Autoregressive matrix (lag {lag}): {masked_a_edges}/{original_a_edges} edges kept after MI masking")
        else:
            A_masked = A_est
        A_masked_list.append(A_masked)
    
    return W_masked, A_masked_list

def estimate_optimal_window_size(df, efold=1/np.e, candidates_multiplier=[2,5,10]):
    """Estimate an optimal window size based on autocorrelation (ACF) decay."""
    logger.info("Estimating optimal window size via ACF")
    max_lag = min(500, len(df)//2)
    acf_vals = np.vstack([acf(df[col], nlags=max_lag, fft=True) for col in df.columns])
    mean_acf = acf_vals.mean(axis=0)
    below_idx = np.where(mean_acf < efold)[0]
    h = int(below_idx[0]) if len(below_idx) > 0 else 1
    logger.info(f"Characteristic decay length h = {h}")
    candidates = [int(max(m * h, 1)) for m in candidates_multiplier]
    # ensure each candidate is at least lag+1 and at most (num_vars * lag)
    return candidates, h

def _init_enhanced_weights_csv():
    """Initialize enhanced CSV file for storing edge weights with complete time mapping."""
    global weights_csv_path
    
    # Define the output path for the enhanced weights CSV file
    weights_csv_path = os.path.join(OUTPUT_DIR, f"weights_enhanced_{run_id}.csv")
    
    # Create the CSV file with enhanced headers
    with open(weights_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'window_idx', 't_end', 'ts_end', 't_center', 'ts_center', 
            'lag', 'i', 'j', 'child_name', 'parent_name', 'weight'
        ])
    
    logger.info(f"Initialized enhanced CSV file for edge weights: {weights_csv_path}")
    return weights_csv_path

def _append_enhanced_weights_csv(df):
    """Append weight records to the enhanced weights CSV file."""
    global weights_csv_path
    
    # Initialize the CSV file if it doesn't exist
    if 'weights_csv_path' not in globals():
        _init_enhanced_weights_csv()
    
    # Append to CSV file without writing headers
    df.to_csv(weights_csv_path, mode='a', header=False, index=False)

def close_weights_csv():
    """Placeholder function to maintain compatibility with the original code."""
    logger.info("Enhanced weights CSV file completed")

def test_window_size_tensor(
    data_tensor: torch.Tensor,
    lag: int,
    window_size: int,
    var_names: list,
    batch_size: int = 32,
    max_windows: int = 2000,
    ckpt_path: str = os.path.join(OUTPUT_DIR, "window_test_ckpt.pkl"),
    save_every: int = 100
) -> float:
    """
    Estimate the stability of W across sliding windows of length `window_size` by
    running DynoTears on each window (batched), then computing the relative variance
    of consecutive Frobenius distances ||W_t - W_{t-1}||_F.

    - Échantillonne au plus `max_windows` fenêtres uniformément réparties.
    - Sauvegarde un checkpoint pickle toutes les `save_every` batches.
    - Reprend automatiquement si un checkpoint existe.

    Returns
    -------
    float
        Relative variance of the Frobenius distances between successive W.
    """
    import pickle

    T, D = data_tensor.shape
    window_size = max(window_size, lag + 1)
    indices = list(range(window_size, T + 1))
    if len(indices) < 2:
        logger.warning("Not enough windows to compute stability; returning 0.")
        return 0.0

    # Échantillonnage
    if max_windows is not None and len(indices) > max_windows:
        step = len(indices) // max_windows
        indices = indices[::step][:max_windows]

    total_windows = len(indices)
    total_batches = (total_windows + batch_size - 1) // batch_size

    # Reprise
    start_batch = 0
    W_chunks = []  # stocke des tensors [B, D, D]
    if os.path.exists(ckpt_path):
        try:
            ck = pickle.load(open(ckpt_path, "rb"))
            if ck.get("window_size") == window_size and ck.get("lag") == lag:
                start_batch = ck.get("batch_idx", 0)
                # W_partial est une liste de tensors convertibles
                W_chunks = [torch.tensor(w, device='cpu') for w in ck.get("W_partial", [])]
                logger.info(f"[WINTEST] Resume at batch {start_batch}/{total_batches}")
        except Exception as e:
            logger.warning(f"[WINTEST] Failed to load checkpoint ({e}), restarting test.")
            start_batch = 0
            W_chunks = []

    logger.info(f"Testing window size {window_size} on {total_windows} windows "
                f"(batch_size={batch_size}, batches={total_batches})")

    # Boucle batches
    for batch_idx in range(start_batch, total_batches):
        b_start = batch_idx * batch_size
        b_end = min((batch_idx + 1) * batch_size, total_windows)
        batch_inds = indices[b_start:b_end]
        logger.info(f"[WINTEST] Batch {batch_idx+1}/{total_batches}: windows {batch_inds[0]}–{batch_inds[-1]}")

        # Empilement des slices
        batch_slices = torch.stack(
            [data_tensor[t - window_size:t] for t in batch_inds],
            dim=0
        ).to(device)

        W_batch_list = []
        for s in range(batch_slices.shape[0]):
            df_win = pd.DataFrame(batch_slices[s].cpu().numpy(), columns=var_names)
            sm = from_pandas_dynamic(
                df_win,
                p=lag,
                lambda_w=2.0,
                lambda_a=2.0,
                max_iter=100,
                h_tol=1e-8,
                loss_tol=1e-6
            )
            W, _ = extract_matrices(sm, var_names, lag)
            W_batch_list.append(W.cpu())

        W_chunk = torch.stack(W_batch_list, dim=0)  # [B, D, D]
        W_chunks.append(W_chunk)

        # checkpoint périodique
        if ((batch_idx + 1) % save_every == 0) or (batch_idx + 1 == total_batches):
            ck = {
                "window_size": window_size,
                "lag": lag,
                "batch_idx": batch_idx + 1,
                "W_partial": [w.numpy() for w in W_chunks]  # vers numpy pour alléger pickle
            }
            with open(ckpt_path, "wb") as f:
                pickle.dump(ck, f)

        del batch_slices
        torch.cuda.empty_cache()

    # Concatène toutes les matrices
    W_all = torch.cat(W_chunks, dim=0)  # [N, D, D]

    # Nettoie le checkpoint terminé
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # Distances Frobenius successives
    diffs = W_all[1:] - W_all[:-1]
    dist = torch.norm(diffs.flatten(1), p='fro', dim=1).cpu().numpy()
    rel_var = dist.var() / (dist.mean() + 1e-8)

    return float(rel_var)

def process_enhanced_window_batch(batch_indices, data_tensor, var_names, lag, original_df):
    """
    Enhanced version that processes a batch of windows with complete time mapping.
    Returns a list of W matrices and exports enhanced edge information.
    """
    batch_results = []
    eps = 1e-4  # threshold for storing edges

    for local_idx, t_end in enumerate(batch_indices):

        start_idx = t_end - best_window_size
        window_tensor = data_tensor[start_idx:t_end]  # [L, D] on device
        df_window = pd.DataFrame(window_tensor.cpu().numpy(), columns=var_names)

        # Calculate window timing information
        window_start_in_original = start_idx
        window_end_in_original = t_end - 1  # t_end is exclusive, so last index is t_end-1
        window_center_in_original = (window_start_in_original + window_end_in_original) // 2
        
        # Get timestamps from original DataFrame
        ts_end = original_df.index[window_end_in_original]
        ts_center = original_df.index[window_center_in_original]
        
        # Log first and last window timestamps for verification (only for first few windows)
        if current_window_idx < 3:
            ts_start = original_df.index[window_start_in_original]
            logger.info(f"Window {current_window_idx}: t_start={window_start_in_original} ({ts_start}), "
                       f"t_end={window_end_in_original} ({ts_end}), "
                       f"t_center={window_center_in_original} ({ts_center})")

        # Clean any previous checkpoint/history (from the *previous window*)
        for path in (WEIGHTS_HISTORY_CKPT, WEIGHTS_HISTORY_CSV):
            if os.path.exists(path):
                os.remove(path)

        # ----- Run DynoTears -----
        start_time = time.time()
        sm = from_pandas_dynamic(
            df_window,
            p=lag,
            lambda_w=best_lambda_w,
            lambda_a=best_lambda_a,
            max_iter=MAX_ITER_WINDOW,
            h_tol=H_TOL,
            loss_tol=LOSS_TOL,
            save_csv_path=WEIGHTS_HISTORY_CSV,
            save_every=SAVE_EVERY,
            checkpoint_path=WEIGHTS_HISTORY_CKPT,
            logger_prefix=f"[Window {current_window_idx:06d}]"
        )
        elapsed = time.time() - start_time

        W, A = extract_matrices(sm, var_names, lag)
        
        # Apply MI mask to sparsify edges based on mutual information
        A_list = [A[:, :, lag_idx] for lag_idx in range(A.shape[2])]
        W_masked, A_masked_list = apply_mi_mask_to_matrices(W, A_list, mi_mask)
        
        # Reconstruct A tensor from masked list
        if A_masked_list:
            A_masked = torch.stack([torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in A_masked_list], dim=2)
        else:
            A_masked = A
        
        # Use masked matrices for edge collection
        W, A = W_masked, A_masked

        # ----- Collect enhanced sparse edges (W: lag 0, A: lags 1..p) -----
        records = []
        
        # W (instantaneous, lag = 0)
        Wi, Wj = W.shape
        for i in range(Wi):
            for j in range(Wj):
                val = float(W[i, j].item()) if hasattr(W[i, j], 'item') else float(W[i, j])
                if abs(val) > eps:
                    records.append({
                        'window_idx': current_window_idx,
                        't_end': window_end_in_original,
                        'ts_end': ts_end,
                        't_center': window_center_in_original,
                        'ts_center': ts_center,
                        'lag': 0,
                        'i': i,
                        'j': j,
                        'child_name': var_names[i],
                        'parent_name': var_names[j],
                        'weight': val
                    })

        # A (temporal, lags 1..p)
        Ai, Aj, Plags = A.shape
        for lag_idx in range(Plags):
            actual_lag = lag_idx + 1  # lag_idx 0 corresponds to lag 1
            for i in range(Ai):
                for j in range(Aj):
                    val = float(A[i, j, lag_idx].item()) if hasattr(A[i, j, lag_idx], 'item') else float(A[i, j, lag_idx])
                    if abs(val) > eps:
                        records.append({
                            'window_idx': current_window_idx,
                            't_end': window_end_in_original,
                            'ts_end': ts_end,
                            't_center': window_center_in_original,
                            'ts_center': ts_center,
                            'lag': actual_lag,
                            'i': i,
                            'j': j,
                            'child_name': var_names[i],
                            'parent_name': var_names[j],
                            'weight': val
                        })

        if records:

            df_rec = pd.DataFrame(records)
            _append_enhanced_weights_csv(df_rec)

        # ----- Logging convergence -----
        losses = sm.history.get("loss", [])
        final_loss = losses[-1] if losses else float("nan")
        converged = (len(losses) < MAX_ITER_WINDOW)
        if converged:
            logger.info(f"Window {current_window_idx} converged in {len(losses) - 1} iterations "
                       f"(loss={final_loss:.6f}, time={elapsed:.2f}s)")
        else:
            logger.warning(f"Window {current_window_idx} reached max_iter "
                          f"(loss={final_loss:.6f}, time={elapsed:.2f}s)")

        batch_results.append(W.cpu().numpy())

        # Increment global window counter
        globals()["current_window_idx"] += 1

    return batch_results


# Determine device for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Import resource manager
from resource_manager import initialize_resources, shutdown_resources, get_resource_manager

if __name__ == "__main__":
    # Initialize resource management
    resource_config = None
    
    try:
        # 1) Load & preprocess data
        try:
            df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        except Exception as e:
            logger.error(f"Failed to load data file '{DATA_FILE}': {e}")
            sys.exit(1)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Initialize resource management based on data size
        resource_config = initialize_resources(df.shape)
        logger.info(f"Enhanced pipeline with resource management: {resource_config}")
        
        # Get resource stats
        resource_manager = get_resource_manager()
        stats = resource_manager.get_resource_stats()
        logger.info(f"System resources: CPU cores={stats.get('cpu_count', 'N/A')}, "
                   f"GPU count={stats.get('gpu_count', 0)}, "
                   f"Available memory={stats.get('memory_available_mb', 0)}MB")
        
        # Store original DataFrame for timestamp mapping
        original_df = df.copy()
    
        # Check original data range
        original_max = float(np.max(np.abs(df.values)))
        logger.info(f"Original data range: max|x|={original_max:.3f}")
    
        # Apply standard scaling only - DO NOT MODIFY THE DATA DISTRIBUTION
        scaler = StandardScaler()
        data_np = scaler.fit_transform(df.values).astype('float32')
        X = pd.DataFrame(data_np, columns=df.columns, index=df.index)
    
        max_abs = float(np.max(np.abs(data_np)))
        n_inf = int(np.isinf(data_np).sum())
        n_nan = int(np.isnan(data_np).sum())
        logger.info(f"Scaled data stats: max|x|={max_abs:.3f}, infs={n_inf}, nans={n_nan}")
    
        if n_inf > 0 or n_nan > 0:
    
            logger.error("Data contains non-finite values. Please clean the dataset.")
            sys.exit(1)
        
        # Warning for potentially problematic data ranges
        if max_abs > 6.0:
            logger.warning(f"Large scaled values detected ({max_abs:.3f}). This may cause numerical instability in optimization.")
        # Move data to GPU (if available)
        data_tensor = torch.from_numpy(data_np).to(device)
        # 2) Determine lag p
        opt_lag = load_optimal_lags(OPTIMAL_LAGS_FILE)
        if opt_lag is None:
            lag = 1
        elif isinstance(opt_lag, int):
            lag = opt_lag
        else:
            lag = max(opt_lag.values()) if len(opt_lag) > 0 else 1
        logger.info(f"Using lag p = {lag}")
        # 3) Hyperparameter sweep for lambda_w and lambda_a
        results = []
        best_loss = float('inf')
        best_lambda_w = None
        best_lambda_a = None
        logger.info("Starting hyperparameter sweep over λ_w and λ_a")
        for lambda_w in LAMBDA_CANDIDATES:
            for lambda_a in LAMBDA_CANDIDATES:
                logger.info(f"Testing λ_w={lambda_w}, λ_a={lambda_a}")
                sm = from_pandas_dynamic(X, p=lag, lambda_w=lambda_w, lambda_a=lambda_a,
                                         max_iter=MAX_ITER_HYPER, h_tol=H_TOL)
                loss_hist = sm.history.get('loss', [])
                final_loss = loss_hist[-1] if loss_hist else float('nan')
                results.append({'lambda_w': lambda_w, 'lambda_a': lambda_a, 'loss': final_loss})
                logger.info(f"Result: final loss = {final_loss:.6f}")
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_lambda_w = lambda_w
                    best_lambda_a = lambda_a
        if best_lambda_w is None or best_lambda_a is None:
            logger.warning("Hyperparameter sweep did not find a valid result. Using default λ_w=0.1, λ_a=0.1")
            best_lambda_w, best_lambda_a = 0.1, 0.1
        logger.info(f"Best hyperparameters: λ_w={best_lambda_w}, λ_a={best_lambda_a} (loss={best_loss:.6f})")
        # 4) Estimate window size via ACF and test candidates
        window_candidates, h = estimate_optimal_window_size(df)
        max_window_cap = df.shape[1] * lag
        window_candidates = sorted(set(min(max(w, lag+1), max_window_cap) for w in window_candidates))
        logger.info(f"Window size candidates (capped at {max_window_cap}): {window_candidates}")
        var_names = list(df.columns)
    
        # Load MI mask for edge sparsing
        mi_mask = load_mi_mask(MI_MASK_FILE, var_names, lag)
    
        window_results = []
        best_window_size = None
        best_var_score = float('inf')
        for w in window_candidates:
            try:
                rv = test_window_size_tensor(data_tensor, lag, w, var_names)
            except Exception as e:
                logger.error(f"Window size test failed for w={w}: {e}")
                rv = float('inf')
            window_results.append({'window_size': w, 'relative_variance': rv})
            logger.info(f"Window {w}: relative variance = {rv:.6f}")
            if rv < best_var_score:
                best_var_score = rv
                best_window_size = w
        log_gpu_stats()
        if best_window_size is None:
            best_window_size = window_candidates[0] if window_candidates else (lag+1)
        logger.info(f"Selected optimal window size L = {best_window_size}")
        # 5) Rolling window DynoTears with enhanced time mapping
        T = data_tensor.shape[0]
        L = best_window_size
        indices = list(range(L, T+1))
        if device.type == 'cuda':
            gpu_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            batch_size = min(50, max(1, gpu_total_mb // 1000))
        else:
            batch_size = 5
        total_batches = (len(indices) + batch_size - 1) // batch_size
        logger.info(f"Rolling window processing: total windows={len(indices)}, batch_size={batch_size}, total_batches={total_batches}")
        start_batch = 0
        matrices_csv_path = os.path.join(OUTPUT_DIR, f"W_matrices_{run_id}.csv")
        if os.path.exists(CHECKPOINT_FILE):
            try:
                ckpt = pickle.load(open(CHECKPOINT_FILE, 'rb'))
                start_batch = ckpt.get('batch_idx', 0)
                run_id_old = ckpt.get('timestamp', run_id)
                if run_id_old != run_id:
                    run_id = run_id_old
                    matrices_csv_path = os.path.join(OUTPUT_DIR, f"W_matrices_{run_id}.csv")
                logger.info(f"Resuming from batch {start_batch}/{total_batches}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint file, starting from batch 0. Error: {e}")
                start_batch = 0
        else:
            with open(matrices_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["window_idx", "coef_idx", "value"])
        logger.info(f"Created output CSV for weight matrices: {matrices_csv_path}")
        ckpt = {'batch_idx': 0, 'timestamp': run_id}
        pickle.dump(ckpt, open(CHECKPOINT_FILE, 'wb'))
        current_window_idx = start_batch * batch_size
        for batch_idx in range(start_batch, total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(indices))
        batch_inds = indices[batch_start:batch_end]
        logger.info(f"Processing batch {batch_idx+1}/{total_batches}: windows {batch_inds[0]}–{batch_inds[-1]}")
        try:
            batch_W_results = process_enhanced_window_batch(batch_inds, data_tensor, var_names, lag, original_df)
        except KeyboardInterrupt:
            logger.info("Execution interrupted by user. Saving progress and exiting.")
            ckpt['batch_idx'] = batch_idx
            ckpt['timestamp'] = run_id
            pickle.dump(ckpt, open(CHECKPOINT_FILE, 'wb'))
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during batch {batch_idx}: {e}. Saving checkpoint and aborting.")
            ckpt['batch_idx'] = batch_idx
            ckpt['timestamp'] = run_id
            pickle.dump(ckpt, open(CHECKPOINT_FILE, 'wb'))
            sys.exit(1)
        with open(matrices_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for w_idx, W in enumerate(batch_W_results, start=batch_start):
                flat = W.flatten()
                for coef_idx, val in enumerate(flat):
                    writer.writerow([w_idx, coef_idx, float(val)])
        ckpt['batch_idx'] = batch_idx + 1
        ckpt['timestamp'] = run_id
        pickle.dump(ckpt, open(CHECKPOINT_FILE, 'wb'))
        log_gpu_stats()
        logger.info("All windows processed. Starting post-processing of results.")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        df_w = pd.read_csv(matrices_csv_path)
        num_windows = df_w['window_idx'].nunique()
        D = len(var_names)
        W_flat = df_w.pivot(index='window_idx', columns='coef_idx', values='value').values.astype(np.float32)
        distances = []
        if W_flat.shape[0] > 1:
            diffs = W_flat[1:] - W_flat[:-1]
        distances = np.linalg.norm(diffs, axis=1)
        pd.DataFrame({'distance': distances}).to_csv(os.path.join(OUTPUT_DIR, f"distances_{run_id}.csv"), index=False)
        all_weights = W_flat.flatten()
        nonzero_weights = all_weights[all_weights != 0]
        if nonzero_weights.size > 0:
            stats = generate_histogram_and_kde(nonzero_weights, bin_step=0.02)
            stats['histogram'].to_csv(os.path.join(OUTPUT_DIR, f"histogram_{run_id}.csv"), index=False)
            stats['kde'].to_csv(os.path.join(OUTPUT_DIR, f"kde_{run_id}.csv"), index=False)
        else:
            logger.info("No nonzero weights found; skipping histogram and KDE generation.")
        close_weights_csv()
        logger.info("=== Enhanced DynoTears DBN analysis complete ===")
        
    finally:
        # Cleanup resource management
        if resource_config:
            shutdown_resources()
            logger.info("Enhanced pipeline resource management shutdown complete")