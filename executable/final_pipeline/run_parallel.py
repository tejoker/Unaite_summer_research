#!/usr/bin/env python3
"""
Parallel Launcher for Tucker-CAM Rolling Window Analysis
=========================================================

Splits the rolling window analysis into chunks and runs them in parallel.
Each worker processes a subset of windows using the existing launcher.

This is safer than multiprocessing within Python because:
1. Each process has isolated memory (no CUDA conflicts)
2. Each process has independent GPU streams
3. Failures in one worker don't affect others
4. Results are written independently (no race conditions)
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def get_total_windows(data_path: str, window_size: int = 100, stride: int = 10):
    """Calculate total number of windows from data."""
    import pandas as pd
    df = pd.read_csv(data_path, index_col=0)
    n_samples = len(df)
    n_windows = (n_samples - window_size) // stride + 1
    return n_windows


def run_worker_chunk(worker_id: int, start_window: int, end_window: int, 
                     n_workers: int, log_dir: Path):
    """
    Run a worker process for a specific window range.
    
    Each worker runs the existing launcher script with modified window range.
    """
    log_file = log_dir / f"worker_{worker_id}.log"
    
    # Create a temporary modified launcher for this worker
    worker_script = log_dir / f"worker_{worker_id}_launcher.py"
    
    # Read the original launcher
    launcher_path = Path(__file__).parent / "dbn_dynotears_fixed_lambda.py"
    with open(launcher_path, 'r') as f:
        launcher_code = f.read()
    
    # Modify the window range in the loop
    # Find: for i in range(start_window, n_samples - window_size + 1, stride):
    # Replace with: for i in range(START, END, stride):
    
    modified_code = launcher_code.replace(
        'for i in range(start_window, n_samples - window_size + 1, stride):',
        f'for i in range({start_window}, {end_window}, stride):'
    )
    
    # Write worker script
    with open(worker_script, 'w') as f:
        f.write(modified_code)
    
    logger.info(f"[Worker {worker_id}/{n_workers}] Starting windows {start_window//10}-{end_window//10}")
    
    # Run the worker script
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(worker_script)],
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            check=True
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[Worker {worker_id}/{n_workers}] Completed in {elapsed:.1f}s")
        
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"[Worker {worker_id}/{n_workers}] Failed after {elapsed:.1f}s: {e}")
        return False, elapsed


def parallel_run(n_workers: int = 12, data_path: str = None, 
                window_size: int = 100, stride: int = 10):
    """
    Run rolling window analysis in parallel by splitting work across processes.
    
    Args:
        n_workers: Number of parallel workers
        data_path: Path to data file
        window_size: Window size
        stride: Stride between windows
    """
    # Set up
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "Golden" / "cleaned_dataset.csv"
    
    log_dir = Path(__file__).parent.parent.parent / "logs" / "parallel_run"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total windows
    total_windows = get_total_windows(str(data_path), window_size, stride)
    total_rows = total_windows * stride
    
    logger.info("="*80)
    logger.info("PARALLEL TUCKER-CAM ANALYSIS")
    logger.info("="*80)
    logger.info(f"Total windows: {total_windows}")
    logger.info(f"Parallel workers: {n_workers}")
    logger.info(f"Windows per worker: ~{total_windows // n_workers}")
    logger.info(f"Expected speedup: ~{n_workers}×")
    logger.info("="*80)
    
    # Split windows across workers
    windows_per_worker = (total_rows + n_workers - 1) // n_workers
    windows_per_worker = (windows_per_worker // stride) * stride  # Round to stride
    
    workers = []
    for worker_id in range(n_workers):
        start_row = worker_id * windows_per_worker
        end_row = min(start_row + windows_per_worker, total_rows)
        
        if start_row >= total_rows:
            break
        
        workers.append((worker_id, start_row, end_row))
    
    # Run workers sequentially (subprocess handles parallelism better than multiprocessing)
    # Actually, let's use concurrent.futures for better control
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    start_time = time.time()
    results = {}
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_worker = {
            executor.submit(run_worker_chunk, wid, start, end, n_workers, log_dir): wid
            for wid, start, end in workers
        }
        
        for future in as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                success, elapsed = future.result()
                results[worker_id] = (success, elapsed)
            except Exception as e:
                logger.error(f"Worker {worker_id} raised exception: {e}")
                results[worker_id] = (False, 0)
    
    total_elapsed = time.time() - start_time
    
    # Summary
    n_successful = sum(1 for success, _ in results.values() if success)
    n_failed = len(results) - n_successful
    
    logger.info("="*80)
    logger.info("PARALLEL ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Successful workers: {n_successful}/{len(workers)}")
    logger.info(f"Failed workers: {n_failed}")
    logger.info(f"Actual speedup: ~{(total_windows * 60) / total_elapsed:.1f}×")  # Assume 60s/window serial
    logger.info(f"Worker logs: {log_dir}")
    logger.info("="*80)
    
    if n_failed > 0:
        logger.warning(f"Some workers failed. Check logs in {log_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Parallel Tucker-CAM Analysis')
    parser.add_argument('--workers', type=int, default=12, help='Number of parallel workers (default: 12)')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--window-size', type=int, default=100, help='Window size')
    parser.add_argument('--stride', type=int, default=10, help='Stride')
    
    args = parser.parse_args()
    
    # Check system resources
    import multiprocessing as mp
    n_cpu = mp.cpu_count()
    
    if args.workers > n_cpu:
        logger.warning(f"Requested {args.workers} workers but only {n_cpu} CPUs available")
    
    logger.info(f"System: {n_cpu} CPUs available")
    logger.info(f"Using: {args.workers} workers")
    
    # Run
    parallel_run(
        n_workers=args.workers,
        data_path=args.data,
        window_size=args.window_size,
        stride=args.stride
    )


if __name__ == "__main__":
    main()
