#!/usr/bin/env python3
"""
Tucker-CAM Rolling Window Analysis - PARALLEL VERSION

Processes multiple windows in parallel to fully utilize all CPU cores.
Each window is independent, so we can run N_WORKERS windows simultaneously.
"""

# Configure CPU threading BEFORE any imports that use parallelism
import os
# Use all available CPU cores for parallel operations (force 64 for systems that report lower)
num_cores = max(os.cpu_count() or 1, 64)

# Divide cores among workers - each worker gets its fair share
# If N_WORKERS is set, divide cores equally: threads_per_worker = total_cores / N_WORKERS
if 'OMP_NUM_THREADS' in os.environ:
    threads_per_worker = int(os.environ['OMP_NUM_THREADS'])
else:
    n_workers = int(os.environ.get('N_WORKERS', 1))
    threads_per_worker = max(1, num_cores // n_workers)  # Allow 1 thread per worker

# Update env vars to ensure libraries respect the limit
os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
os.environ['MKL_NUM_THREADS'] = str(threads_per_worker)
os.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_worker)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(threads_per_worker)
os.environ['NUMEXPR_NUM_THREADS'] = str(threads_per_worker)
os.environ['OPENBLAS_MAIN_FREE'] = '1'

import sys
from pathlib import Path
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import traceback as tb
from sklearn.cluster import KMeans

torch.set_num_threads(threads_per_worker)
torch.set_num_interop_threads(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def adaptive_hard_threshold(weights, min_cluster_size=100):
    """
    Adaptive hard thresholding using K-means clustering in log-space.
    
    Separates signal from noise by clustering edge weights into two groups.
    More principled than fixed top-k: adapts to each window's weight distribution.
    
    Args:
        weights: np.ndarray of absolute weight values
        min_cluster_size: Minimum edges to keep (safety threshold)
        
    Returns:
        np.ndarray: Boolean mask indicating signal edges
    """
    if len(weights) == 0:
        return np.array([], dtype=bool)
    
    # Handle zeros/negatives
    log_weights = np.log1p(np.abs(weights))  # log(1 + |w|)
    
    # Two-cluster K-means in log-space
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(log_weights.reshape(-1, 1))
    
    # Identify signal cluster (higher center)
    centers = kmeans.cluster_centers_.flatten()
    signal_cluster = np.argmax(centers)
    signal_mask = (kmeans.labels_ == signal_cluster)
    
    # Safety: Keep at least min_cluster_size edges
    if signal_mask.sum() < min_cluster_size and len(weights) >= min_cluster_size:
        top_indices = np.argsort(weights)[-min_cluster_size:]
        signal_mask = np.zeros(len(weights), dtype=bool)
        signal_mask[top_indices] = True
    
    # DEBUG LOGGING
    n_kept = signal_mask.sum()
    if n_kept == 0 and len(weights) > 0:
        # Fallback: if clustering removes everything, keep top k (safety net)
        # But only if we have enough weights
        fallback_k = min(len(weights), 50)
        top_indices = np.argsort(weights)[-fallback_k:]
        signal_mask[top_indices] = True
        # logger.info(f"  Threshold warning: Clustering kept 0/{len(weights)} edges. Fallback to top-{fallback_k}.")
    
    return signal_mask


def process_single_window(args):
    """Process a single window (runs in separate process)."""
    window_idx, data_file, columns_file, window_start, window_end, p, rank_w, rank_a, n_knots, lambda_smooth, lambda_w, lambda_a, shm_name, shm_shape, shm_dtype, n_threads = args

    import sys
    import traceback
    from pathlib import Path
    import os
    import numpy as np
    from datetime import datetime
    import time as time_module
    
    try:
        import psutil
    except ImportError:
        psutil = None

    # CRITICAL: Configure threads for this worker immediately
    # This must be done at runtime since this process might be forked
    import torch
    torch.set_num_threads(n_threads)
    # torch.set_num_interop_threads(1)
    # Attempts to restrict MKL/BLAS if not already too late
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)


    # Import Tucker-CAM inside worker (critical for multiprocessing)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam

    # Memory tracking helper
    def get_memory_mb():
        if psutil:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        return 0.0

    def log_with_timestamp(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [Window {window_idx}] {msg}", file=sys.stderr, flush=True)

    try:
        start_time = time_module.time()
        log_with_timestamp(f"STARTING (range: {window_start}-{window_end})")

        mem_start = get_memory_mb()
        log_with_timestamp(f"Memory at start: {mem_start:.1f} MB")

        # Load ONLY this window's data using SHARED MEMORY (zero-copy across all workers!)
        log_with_timestamp(f"Accessing shared memory dataset...")
        
        if shm_name:
            # Use shared memory (preferred - true zero-copy)
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(name=shm_name)
            data_full = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
            data_chunk = data_full[window_start:window_end].copy()  # Copy window to worker's RAM
            shm.close()  # Don't unlink - main process owns it
        else:
            # Fallback to memory-mapped file
            data_full = np.load(data_file, mmap_mode='r')
            data_chunk = data_full[window_start:window_end].copy()
            del data_full
        
        var_names = np.load(columns_file, allow_pickle=True).tolist()

        mem_after_load = get_memory_mb()
        log_with_timestamp(f"Data loaded - Memory: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f} MB)")

        # Create DataFrame for this window
        log_with_timestamp(f"Creating DataFrame...")
        window_df = pd.DataFrame(data_chunk, columns=var_names)
        del data_chunk  # Free immediately

        mem_before_training = get_memory_mb()
        log_with_timestamp(f"DataFrame created - Memory: {mem_before_training:.1f} MB")

        # Run Tucker-CAM (no threshold - rely on top-k filtering)
        log_with_timestamp(f"Starting Tucker-CAM training (p={p}, rank_w={rank_w}, rank_a={rank_a})...")
        training_start = time_module.time()

        edges = from_pandas_dynamic_tucker_cam(
            window_df,
            p=p,
            rank_w=rank_w,
            rank_a=rank_a,
            n_knots=n_knots,
            lambda_smooth=lambda_smooth,
            lambda_w=lambda_w,
            lambda_a=lambda_a,
            max_iter=100,
            lr=0.01,
            w_threshold=0.0,  # Disabled - Tucker compression produces small weights
            device='cpu',
            return_indices=True # CRITICAL FIX: Return raw indices to avoid float conversion errors
        )

        training_elapsed = time_module.time() - training_start
        mem_after_training = get_memory_mb()
        log_with_timestamp(f"Training complete in {training_elapsed:.2f}s - Memory: {mem_after_training:.1f} MB")
        log_with_timestamp(f"Peak memory delta: +{mem_after_training - mem_start:.1f} MB")

        # Extract weights
        log_with_timestamp(f"Extracting edge weights...")
        
        # Optimize: Use NumPy array instead of list of dicts to save space (500MB -> 10MB)
        # Format: [window_idx, source, target, lag, weight]
        num_edges = len(edges)
        if num_edges > 0:
            weights_arr = np.zeros((num_edges, 5), dtype=np.float32)
            
            weights_arr = np.zeros((num_edges, 5), dtype=np.float32)
            
            for i, edge in enumerate(edges):
                # New tuple format from return_indices=True: (src, tgt, lag, weight)
                # No strings involved!
                src, tgt, lag, weight = edge
                
                weights_arr[i, 0] = float(window_idx)
                weights_arr[i, 1] = float(src)
                weights_arr[i, 2] = float(tgt)
                weights_arr[i, 3] = float(lag)
                weights_arr[i, 4] = abs(float(weight))
        else:
            weights_arr = np.zeros((0, 5), dtype=np.float32)

        mem_final = get_memory_mb()
        total_elapsed = time_module.time() - start_time
        log_with_timestamp(f"COMPLETED in {total_elapsed:.2f}s - Extracted {num_edges} edges - Memory: {mem_final:.1f} MB")

        return (window_idx, weights_arr, None)
        
    except MemoryError as e:
        error_msg = f"MemoryError: {e}"
        print(f"[Window {window_idx}] {error_msg}", file=sys.stderr, flush=True)
        return (window_idx, None, error_msg)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[Window {window_idx}] ERROR:\n{error_msg}", file=sys.stderr, flush=True)
        return (window_idx, None, error_msg)


def run_parallel_tucker_cam(
    data: np.ndarray,
    var_names: list,
    p: int,
    window_size: int,
    stride: int,
    output_dir: Path,
    data_file: str,
    columns_file: str,
    rank_w: int = 20,
    rank_a: int = 10,
    n_knots: int = 5,
    lambda_smooth: float = 0.01,
    lambda_w: float = 0.0,
    lambda_a: float = 0.0,
    top_k: int = 10000,
    n_workers: int = 8,
    shm_name: str = None,
    shm_shape: tuple = None,
    shm_dtype: np.dtype = None,
    n_threads: int = 1,
    max_windows: int = None
):
    """
    Process windows in parallel using multiple processes.
    
    Args:
        n_workers: Number of parallel workers (default 8 for 64-core machine)
        shm_name: Name of shared memory segment (if available)
        shm_shape: Shape of data in shared memory
        shm_dtype: Data type in shared memory
        n_threads: Number of threads per worker
        max_windows: Limit number of windows to process
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = len(data)
    num_windows = (n_samples - window_size) // stride + 1
    
    logger.info(f"="*80)
    logger.info(f"PARALLEL Tucker-CAM Analysis")
    logger.info(f"="*80)
    logger.info(f"Total windows: {num_windows}")
    logger.info(f"Parallel workers: {n_workers}")
    logger.info(f"Threads per worker: {os.environ.get('OMP_NUM_THREADS', 'unknown')}")
    logger.info(f"="*80)
    
    # Check for existing results (resume support)
    results_file = output_dir / "window_edges_incremental.npy"
    completed_windows = set()
    
    if results_file.exists():
        try:
            # Only load metadata, not full edge data (to avoid 20GB memory bloat)
            existing_data = np.load(results_file, allow_pickle=True)
            completed_windows = {w['window'] for w in existing_data}
            logger.info(f"Found {len(completed_windows)} completed windows, resuming...")
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}")
            
    # FIX: Also check for temp files to mark windows as completed (Robust Resume)
    temp_dir = output_dir / "temp_windows"
    if temp_dir.exists():
        temp_files_found = 0
        for temp_file in temp_dir.glob("window_*.npy"):
            try:
                # Format: window_00123.npy or window_00123.tmp.npy
                if temp_file.name.startswith("window_") and temp_file.suffix == '.npy' and '.tmp' not in temp_file.name:
                    idx_str = temp_file.stem.split('_')[1]
                    idx = int(idx_str)
                    if idx not in completed_windows:
                        completed_windows.add(idx)
                        temp_files_found += 1
            except (ValueError, IndexError):
                pass
        
        if temp_files_found > 0:
            logger.info(f"Found {temp_files_found} additional completed windows in {temp_dir.name}, marking as completed.")
    
    # Prepare all window arguments (skip completed ones)
    window_args = []
    for win_idx in range(num_windows):
        if win_idx in completed_windows:
            continue
            
        start_idx = win_idx * stride
        end_idx = start_idx + window_size
        
        if end_idx > n_samples:
            break
        
        # Pass file paths + indices + shared memory info
        window_args.append((
            win_idx, data_file, columns_file, start_idx, end_idx, p,
            rank_w, rank_a, n_knots, lambda_smooth, lambda_w, lambda_a,
            shm_name, shm_shape, shm_dtype,
            n_threads # Pass calculated thread count
        ))
    
    if max_windows and len(window_args) > max_windows:
        logger.info(f"Limiting execution to first {max_windows} windows (of {len(window_args)} remaining)")
        window_args = window_args[:max_windows]
    
    if not window_args:
        logger.info("All windows already completed! Proceeding to merge check...")
        # Do not return here! We must fallback to the merge step.
        # Ensure we don't try to run empty executor
    else:
        logger.info(f"Processing {len(window_args)} remaining windows...")
    
    # Process windows in parallel
    # DON'T accumulate all edges in memory - causes 20GB bloat!
    # Write incrementally to temp files instead
    completed = len(completed_windows)
    failed = 0
    temp_dir = output_dir / "temp_windows"
    temp_dir.mkdir(exist_ok=True)
    
    # Create progress file
    progress_file = output_dir / "progress.txt"
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_window, args): args[0] for args in window_args}
        
        for future in as_completed(futures):
            win_idx = futures[future]
            try:
                window_idx, weights, error = future.result()
                
                if error:
                    logger.error(f"Window {window_idx} FAILED: {error}")
                    failed += 1
                else:
                    # Save weights to individual temp file (don't accumulate in RAM!)
                    if weights is not None and len(weights) > 0:
                        temp_file = temp_dir / f"window_{window_idx:05d}.npy"
                        # Fix: np.save appends .npy if missing. Use .tmp.npy explicitly.
                        temp_file_tmp = temp_dir / f"window_{window_idx:05d}.tmp.npy" # Explicit extension
                        
                        # Ensure dir exists (race condition safety)
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Atomic write: save to .tmp.npy, then rename to .npy
                        np.save(temp_file_tmp, weights, allow_pickle=False)
                        os.rename(temp_file_tmp, temp_file)
                        
                        logger.info(f"Window {window_idx} completed - Saved {len(weights)} edges to {temp_file.name}")

                    completed += 1

                    # Update progress file
                    with open(progress_file, 'w') as f:
                        f.write(f"Completed: {completed}/{num_windows}\n")
                        f.write(f"Failed: {failed}\n")
                        f.write(f"In progress: {len(window_args) - (completed - len(completed_windows)) - failed}\n")
                        f.write(f"Last completed window: {window_idx}\n")


                    progress_pct = (completed / num_windows) * 100
                    logger.info(f"Progress: {completed}/{num_windows} windows ({progress_pct:.1f}%), {failed} failed)")
                
                # CRITICAL: Release memory immediately!
                # The Future object holds the result (weights) which is heavy.
                # We saved it to disk, so we don't need it in RAM anymore.
                del weights
                del futures[future]
                        
            except Exception as e:
                logger.error(f"Window {win_idx} exception: {e}")
                failed += 1
                
                # Update progress even on failure
                with open(progress_file, 'w') as f:
                    f.write(f"Completed: {completed}/{num_windows}\n")
                    f.write(f"Failed: {failed}\n")
                    f.write(f"In progress: {len(window_args) - (completed - len(completed_windows)) - failed}\n")
                
                # Release reference on failure too
                del futures[future]

    
    # Merge all temp files into final result (only loads into memory once at end)
    logger.info(f"="*80)
    logger.info(f"POST-PROCESSING: Merging temp files and applying top-k filter...")
    
    if max_windows is not None:
        logger.info(f"Skipping merge step because --max-windows is set (Stress Test Mode).")
        logger.info(f"Parallel processing finished successfully.")
        return

    # STREAMING MERGE: Process temp files one by one and write to CSV
    # This avoids loading all data into memory at once
    import csv
    
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_csv = weights_dir / "weights_enhanced.csv"
    
    logger.info(f"MERGE: Streaming results to {output_csv}...")
    
    temp_files = sorted(temp_dir.glob("window_*.npy"))
    total_edges = 0
    
    # SAFETY: If no temp files are found, do NOT overwrite the existing weights file.
    # This prevents data loss if the script is re-run on a fully completed experiment.
    if not temp_files:
        if output_csv.exists():
             logger.info(f"No new temp files to merge. Keeping existing {output_csv}.")
        else:
             logger.warning(f"No temp files found and no existing output {output_csv}. Something might be wrong.")
        
        logger.info(f"="*80)
        return

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header matching dual_metric_anomaly_detection.py expectation (i, j for indices)
        writer.writerow(['window_idx', 'i', 'j', 'lag', 'weight'])
        
        for i, temp_file in enumerate(temp_files):
            try:
                # Load one window
                weights_arr = np.load(temp_file, allow_pickle=False)
                
                # Apply adaptive thresholding per window
                # Extract weight values (column 4)
                weight_values = weights_arr[:, 4]
                
                if len(weight_values) > 0:
                    # Apply adaptive thresholding
                    logger.info(f"    Window {i}: Raw edges={len(weight_values)}, Max weight={np.max(weight_values):.4f}")
                    signal_mask = adaptive_hard_threshold(weight_values, min_cluster_size=min(100, len(weight_values)))
                    logger.info(f"    Window {i}: Kept {signal_mask.sum()} edges after thresholding")
                    
                    # Write kept edges
                    rows_to_write = []
                    # Filter rows using boolean mask
                    kept_rows = weights_arr[signal_mask]
                    
                    for row in kept_rows:
                        # [window_idx, source, target, lag, weight]
                        rows_to_write.append([
                            int(row[0]),  # window
                            int(row[1]),  # source(i)
                            int(row[2]),  # target(j)
                            int(row[3]),  # lag
                            float(row[4]) # weight
                        ])
                    
                    if rows_to_write:
                        writer.writerows(rows_to_write)
                        total_edges += len(rows_to_write)
                
                # Clean up immediately
                del weights_arr
                temp_file.unlink()
                
                if i % 10 == 0:
                    logger.info(f"Merged {i+1}/{len(temp_files)} windows... ({total_edges} edges written)")
                    
            except Exception as e:
                logger.warning(f"Failed to merge {temp_file.name}: {e}")

    logger.info(f"="*80)
    logger.info(f"PARALLEL PROCESSING COMPLETE")
    logger.info(f"Total edges saved: {total_edges}")
    logger.info(f"Output: {output_csv}")
    
    # Remove temp dir if empty
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    logger.info(f"="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .npy data file')
    parser.add_argument('--columns', required=True, help='Path to .npy columns file')
    parser.add_argument('--lags', required=True, help='Path to .npy lags file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers (2-4 recommended for d=2889)')
    parser.add_argument('--max-windows', type=int, default=None, help='Maximum number of windows to process')
    
    args = parser.parse_args()
    
    # Load data
    data_np = np.load(args.data)
    var_names = np.load(args.columns, allow_pickle=True).tolist()
    
    # Load optimal lags (handle both dictionary and array formats)
    optimal_lags_raw = np.load(args.lags, allow_pickle=True)
    optimal_lags = {}
    
    if isinstance(optimal_lags_raw, dict):
        optimal_lags = optimal_lags_raw
    elif isinstance(optimal_lags_raw, np.ndarray):
        if optimal_lags_raw.shape == ():  # 0-d array containing dict
            optimal_lags = optimal_lags_raw.item()
        elif len(optimal_lags_raw) > 0:
            # Check first item to determine format
            first_item = optimal_lags_raw[0]
            
            # Try to extract as tuple (works for both regular tuples and structured arrays)
            try:
                if hasattr(first_item, '__len__') and len(first_item) == 2:
                    # Array of tuples/pairs: [(var_name, lag), ...]
                    for item in optimal_lags_raw:
                        var_name = str(item[0])  # Convert to string in case it's bytes
                        lag = int(item[1])
                        optimal_lags[var_name] = lag
                elif isinstance(first_item, (int, np.integer)):
                    # Just lag values in order
                    optimal_lags = {var_names[i]: int(lag) for i, lag in enumerate(optimal_lags_raw)}
                elif optimal_lags_raw.dtype.names and 'optimal_lag' in optimal_lags_raw.dtype.names:
                    # Structured array with field names
                    optimal_lags_arr = optimal_lags_raw['optimal_lag']
                    optimal_lags = {var_names[i]: int(lag) for i, lag in enumerate(optimal_lags_arr)}
            except (TypeError, IndexError, ValueError):
                pass
    
    if not optimal_lags:
        logger.error("Failed to load optimal lags!")
        logger.error(f"Raw type: {type(optimal_lags_raw)}")
        logger.error(f"Raw shape: {optimal_lags_raw.shape if hasattr(optimal_lags_raw, 'shape') else 'N/A'}")
        logger.error(f"Raw content (first 5): {optimal_lags_raw[:5] if hasattr(optimal_lags_raw, '__getitem__') else optimal_lags_raw}")
        sys.exit(1)
    
    p = max(optimal_lags.values())
    
    logger.info(f"Loaded data: {data_np.shape}")
    logger.info(f"Variables: {len(var_names)}")
    logger.info(f"Max lag: {p}")
    
    # Create shared memory for dataset (zero-copy across all workers!)
    logger.info(f"Creating shared memory buffer ({data_np.nbytes / 1024 / 1024:.1f} MB)...")
    shm = None
    try:
        shm = shared_memory.SharedMemory(create=True, size=data_np.nbytes)
        shm_array = np.ndarray(data_np.shape, dtype=data_np.dtype, buffer=shm.buf)
        shm_array[:] = data_np[:] # Copy data to shared memory
        logger.info(f"Shared memory created: {shm.name}")
    except Exception as e:
        logger.warning(f"Failed to create shared memory: {e}. Falling back to mmap mode.")
        if shm:
            shm.close()
            shm.unlink()
        shm = None
    
    # Run parallel processing
    try:
        # Register signal handler to ensure cleanup on interrupt
        import signal
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, cleaning up...")
            if shm:
                try:
                    shm.close()
                    shm.unlink()
                    logger.info("Shared memory unlinked successfully.")
                except:
                    pass
            sys.exit(1)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Calculate optimal threads per worker
        # Reserve 1 core for system/main process overhead if possible
        total_cores = os.cpu_count() or 1
        n_threads = max(1, total_cores // args.workers)
        
        # Cap threads to avoid diminishing returns (PyTorch intra-op usually saturates around 4-8)
        n_threads = min(n_threads, 8) 
        
        logger.info(f"Thread configuration: {args.workers} workers x {n_threads} threads (Total cores: {total_cores})")

        run_parallel_tucker_cam(
            data_np, var_names, p,
            args.window_size, args.stride,
            Path(args.output),
            str(args.data),
            str(args.columns),
            n_workers=args.workers,
            shm_name=shm.name if shm else None,
            shm_shape=data_np.shape,
            shm_dtype=data_np.dtype,
            n_threads=n_threads,
            max_windows=args.max_windows
        )
    except Exception as e:
        logger.error(f"Global error in parallel execution: {e}")
        raise
    finally:
        # Cleanup shared memory
        if shm:
            try:
                # Check if it still exists before unlinking
                # On some systems, unlinking twice throws error, on others it's fine
                shm.close()
                shm.unlink()
                logger.info("Shared memory cleaned up.")
            except FileNotFoundError:
                pass  # Already cleaned up
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory: {e}")


if __name__ == '__main__':
    main()
