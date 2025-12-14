#!/usr/bin/env python3
"""
Benchmark GPU vs CPU performance for Tucker-CAM on your actual data.
Tests first 2 windows to estimate which device is faster.
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add executable/final_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "executable" / "final_pipeline"))

from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam
import pandas as pd

def benchmark_device(data, var_names, p, device, n_windows=2):
    """Benchmark Tucker-CAM on specified device."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {device.upper()}")
    print(f"{'='*80}")
    
    window_size = 100
    times = []
    
    for win_idx in range(n_windows):
        start_idx = win_idx * 10  # stride=10
        end_idx = start_idx + window_size
        
        if end_idx > len(data):
            break
        
        window_data = data[start_idx:end_idx]
        window_df = pd.DataFrame(window_data, columns=var_names)
        
        print(f"\nWindow {win_idx} (rows {start_idx}-{end_idx})...")
        
        start_time = time.time()
        
        try:
            edges = from_pandas_dynamic_tucker_cam(
                window_df,
                p=p,
                rank_w=20,
                rank_a=10,
                n_knots=5,
                lambda_smooth=0.01,
                lambda_w=0.0,
                lambda_a=0.0,
                max_iter=100,
                lr=0.01,
                w_threshold=0.01,
                device=device
            )
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"  ✓ Completed in {elapsed:.1f}s ({len(edges)} edges)")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ✗ CUDA OOM: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    avg_time = np.mean(times)
    print(f"\n{device.upper()} Average: {avg_time:.1f}s per window")
    
    return avg_time


def main():
    print("\n" + "="*80)
    print("Tucker-CAM GPU vs CPU Benchmark")
    print("="*80)
    
    # Generate synthetic test data matching real problem dimensions
    print("\nGenerating synthetic test data...")
    
    # Match your actual data dimensions
    n_samples = 500  # Enough for 2 windows (100 each, stride 10)
    n_vars = 2889    # Your actual number of variables
    p = 20           # Your max lag
    
    print(f"  Samples: {n_samples}")
    print(f"  Variables: {n_vars}")
    print(f"  Max lag: {p}")
    
    # Generate random data (standardized, like preprocessing output)
    np.random.seed(42)
    data_np = np.random.randn(n_samples, n_vars).astype(np.float32)
    var_names = [f"var_{i}" for i in range(n_vars)]
    
    print(f"\nTest data created: {data_np.shape[0]} samples × {data_np.shape[1]} variables")
    print(f"Testing first 2 windows (stride=10, window_size=100)")
    
    # Automatic device selection (same logic as real pipeline)
    gpu_time = None
    if n_vars > 2000:
        print(f"\n⚠️  {n_vars} variables > 2000 threshold")
        print("    Automatic device selection: CPU (GPU would OOM)")
        print("    GPU benchmark skipped to match real pipeline behavior")
    elif torch.cuda.is_available():
        print(f"\n✓ {n_vars} variables ≤ 2000 threshold, testing GPU...")
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        gpu_time = benchmark_device(data_np, var_names, p, 'cuda', n_windows=2)
        
        # Report GPU memory
        if gpu_time is not None:
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak GPU memory: {peak_mem:.2f} GB")
            torch.cuda.empty_cache()
    else:
        print("\n✗ CUDA not available, skipping GPU benchmark")
    
    # Test CPU
    cpu_time = benchmark_device(data_np, var_names, p, 'cpu', n_windows=2)
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    if gpu_time is not None and cpu_time is not None:
        print(f"GPU: {gpu_time:.1f}s per window")
        print(f"CPU: {cpu_time:.1f}s per window")
        print(f"\nSpeedup: {cpu_time / gpu_time:.2f}x ({'GPU' if gpu_time < cpu_time else 'CPU'} faster)")
        
        total_windows = 421
        gpu_estimate = gpu_time * total_windows / 3600
        cpu_estimate = cpu_time * total_windows / 3600
        
        print(f"\nEstimated time for {total_windows} windows:")
        print(f"  GPU: {gpu_estimate:.1f} hours")
        print(f"  CPU: {cpu_estimate:.1f} hours")
        
        if gpu_time < cpu_time:
            print(f"\n✓ Recommendation: Use GPU (saves {cpu_estimate - gpu_estimate:.1f} hours)")
        else:
            print(f"\n✓ Recommendation: Use CPU (saves {gpu_estimate - cpu_estimate:.1f} hours)")
    elif cpu_time is not None:
        print(f"CPU: {cpu_time:.1f}s per window")
        total_windows = 421
        cpu_estimate = cpu_time * total_windows / 3600
        print(f"Estimated time for {total_windows} windows: {cpu_estimate:.1f} hours")
        print(f"\n✓ Using CPU (GPU unavailable or OOM)")
    else:
        print("✗ Both devices failed - problem too large!")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
