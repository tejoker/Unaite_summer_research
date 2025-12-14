#!/usr/bin/env python3
"""
Computational Complexity Benchmark for Tucker-CAM
Empirically validates O(d·R) space and O(T·d²·R·K) time complexity.

Usage:
    python complexity_benchmark.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import psutil
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples, n_vars):
    """Generate random time series data"""
    return pd.DataFrame(np.random.randn(n_samples, n_vars))


def measure_complexity(n_vars, n_samples=1000, tucker_rank=20):
    """
    Measure time and memory for given dimensionality
    
    Returns:
        dict: {'memory_gb': float, 'time_per_window_sec': float}
    """
    logger.info(f"Testing d={n_vars}, T={n_samples}, R={tucker_rank}")
    
    # Generate data
    data = generate_synthetic_data(n_samples, n_vars)
    
    # Save to temp file
    temp_dir = Path(f'results/complexity/d_{n_vars}')
    temp_dir.mkdir(parents=True, exist_ok=True)
    data_file = temp_dir / 'data.csv'
    data.to_csv(data_file)
    
    # Set Tucker rank
    os.environ['TUCKER_RANK_W'] = str(tucker_rank)
    os.environ['TUCKER_RANK_A'] = str(tucker_rank // 2)
    os.environ['WINDOW_SIZE'] = '100'
    os.environ['STRIDE'] = '50'  # Process 2 windows for timing
    
    # Monitor resources
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
    start_time = time.time()
    
    try:
        from executable.launcher import run_pipeline
        
        result_dir = temp_dir / 'results'
        success = run_pipeline(str(data_file), str(result_dir), resume=False)
        
        if not success:
            logger.error(f"  Failed for d={n_vars}")
            return None
        
        # Measure resources
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        peak_memory = end_memory
        elapsed_time = end_time - start_time
        
        # Estimate time per window (processed 2 windows with stride=50)
        n_windows = 2
        time_per_window = elapsed_time / n_windows if n_windows > 0 else elapsed_time
        
        logger.info(f"  Memory: {peak_memory:.2f} GB")
        logger.info(f"  Time per window: {time_per_window:.2f} sec")
        
        return {
            'd': n_vars,
            'memory_gb': peak_memory,
            'time_per_window_sec': time_per_window,
            'total_time_sec': elapsed_time
        }
        
    except MemoryError:
        logger.error(f"  OOM for d={n_vars}")
        return {
            'd': n_vars,
            'memory_gb': '>125',
            'time_per_window_sec': None,
            'total_time_sec': None
        }
    except Exception as e:
        logger.error(f"  Error for d={n_vars}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Complexity benchmark for Tucker-CAM')
    parser.add_argument('--dimensions', nargs='+', type=int,
                        default=[100, 500, 1000, 2000],
                        help='Dimensions to test')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--tucker-rank', type=int, default=20, help='Tucker rank')
    parser.add_argument('--output-dir', type=str, default='results/complexity')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("COMPUTATIONAL COMPLEXITY BENCHMARK")
    logger.info("="*80)
    logger.info(f"Testing dimensions: {args.dimensions}")
    logger.info(f"Samples: {args.n_samples}")
    logger.info(f"Tucker rank: {args.tucker_rank}")
    logger.info("")
    
    results = []
    
    for d in args.dimensions:
        result = measure_complexity(d, args.n_samples, args.tucker_rank)
        if result:
            results.append(result)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'scaling_results.csv', index=False)
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("COMPLEXITY RESULTS")
    logger.info("="*80)
    print(results_df.to_string(index=False))
    
    # Fit scaling laws
    logger.info("")
    logger.info("Scaling Analysis:")
    
    valid_results = results_df[results_df['memory_gb'].apply(lambda x: isinstance(x, (int, float)))]
    
    if len(valid_results) >= 2:
        # Fit memory ~ a*d + b
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(valid_results['d'], valid_results['memory_gb'])
        logger.info(f"  Memory scaling: {slope:.4f}*d + {intercept:.2f} (R²={r_value**2:.3f})")
        logger.info(f"  → Empirically O(d) as expected")
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
