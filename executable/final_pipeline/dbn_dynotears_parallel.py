#!/usr/bin/env python3
"""
Parallel Rolling Window Analysis with Tucker-CAM
=================================================

Processes multiple windows in parallel for massive speedup.
Uses multiprocessing to parallelize across CPU cores and GPU streams.
"""

import os
import sys
import logging
import time
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dbn_dynotears_fixed_lambda import (
    load_and_prepare_data,
    process_single_window,
    get_config_params
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def worker_process_window(args):
    """
    Worker function to process a single window.
    
    Each worker gets its own GPU stream to avoid conflicts.
    """
    (window_idx, window_start, window_end, df_differenced, 
     config_params, worker_id, n_workers) = args
    
    try:
        # Set GPU device and stream for this worker
        if torch.cuda.is_available():
            # Round-robin GPU assignment (if multiple GPUs, otherwise all use GPU 0)
            device_id = worker_id % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            
            # Create separate CUDA stream for this worker
            torch.cuda.set_stream(torch.cuda.Stream())
        
        logger.info(f"[Worker {worker_id}/{n_workers}] Processing window {window_idx} "
                   f"(rows {window_start}-{window_end})")
        
        # Process window
        result = process_single_window(
            window_idx=window_idx,
            window_start=window_start,
            window_end=window_end,
            df_differenced=df_differenced,
            **config_params
        )
        
        logger.info(f"[Worker {worker_id}/{n_workers}] Window {window_idx} complete: "
                   f"{result['n_edges']} edges, {result['elapsed']:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"[Worker {worker_id}/{n_workers}] Window {window_idx} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parallel_rolling_window_analysis(
    df_differenced: pd.DataFrame,
    window_size: int,
    stride: int,
    start_window: int,
    n_workers: int = 12,
    **config_params
):
    """
    Run rolling window analysis in parallel.
    
    Args:
        df_differenced: Preprocessed dataframe
        window_size: Size of each window
        stride: Stride between windows
        start_window: Starting row index
        n_workers: Number of parallel workers (default: 12)
        **config_params: Additional parameters (lambda_w, lambda_a, etc.)
    """
    n_samples = len(df_differenced)
    
    # Generate all window parameters
    window_args = []
    window_indices = []
    
    for i in range(start_window, n_samples - window_size + 1, stride):
        window_idx = i // stride
        window_start = i
        window_end = i + window_size
        
        window_args.append((
            window_idx, window_start, window_end,
            df_differenced, config_params,
            len(window_args) % n_workers,  # worker_id
            n_workers
        ))
        window_indices.append(window_idx)
    
    n_windows = len(window_args)
    logger.info("="*80)
    logger.info(f"PARALLEL ROLLING WINDOW ANALYSIS")
    logger.info("="*80)
    logger.info(f"Total windows: {n_windows}")
    logger.info(f"Parallel workers: {n_workers}")
    logger.info(f"Expected speedup: {min(n_workers, n_windows)}×")
    logger.info(f"Window size: {window_size}, Stride: {stride}")
    logger.info("="*80)
    
    # Run parallel processing
    start_time = time.time()
    
    # Use spawn method to avoid CUDA initialization issues
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker_process_window, window_args)
    
    elapsed = time.time() - start_time
    
    # Filter out failed windows
    successful_results = [r for r in results if r is not None]
    n_successful = len(successful_results)
    n_failed = n_windows - n_successful
    
    total_edges = sum(r['n_edges'] for r in successful_results)
    avg_time_per_window = elapsed / n_windows if n_windows > 0 else 0
    
    logger.info("="*80)
    logger.info("PARALLEL ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Average time per window: {avg_time_per_window:.1f}s")
    logger.info(f"Successful windows: {n_successful}/{n_windows}")
    logger.info(f"Failed windows: {n_failed}")
    logger.info(f"Total edges learned: {total_edges:,}")
    logger.info(f"Speedup vs sequential: ~{n_workers:.1f}×")
    logger.info("="*80)
    
    return successful_results


def main():
    """Main entry point for parallel rolling window analysis."""
    
    # Get configuration
    config_params = get_config_params()
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df_differenced = load_and_prepare_data(config_params['data_path'])
    
    # Determine number of workers based on system resources
    n_cpu = mp.cpu_count()
    available_ram_gb = 120  # From free -h output
    ram_per_worker = 5  # GB per worker
    max_workers_ram = int(available_ram_gb / ram_per_worker)
    
    # GPU memory constraint (assume 1 GB per worker)
    gpu_mem_gb = 24
    gpu_per_worker = 1
    max_workers_gpu = int(gpu_mem_gb / gpu_per_worker)
    
    # Use conservative estimate
    n_workers = min(12, max_workers_ram, max_workers_gpu, n_cpu // 2)
    
    logger.info(f"System resources: {n_cpu} CPUs, {available_ram_gb} GB RAM, {gpu_mem_gb} GB GPU")
    logger.info(f"Using {n_workers} parallel workers")
    
    # Run parallel analysis
    results = parallel_rolling_window_analysis(
        df_differenced=df_differenced,
        window_size=config_params['window_size'],
        stride=config_params['stride'],
        start_window=config_params['start_window'],
        n_workers=n_workers,
        **config_params
    )
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
