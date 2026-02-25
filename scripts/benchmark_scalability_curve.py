#!/usr/bin/env python3
"""
benchmark_scalability_curve.py - Generate REAL data for Figure 1.
Runs short benchmarks for d=[100, 500, 1000, 2889] and projects full runtime.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import resource

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "executable" / "final_pipeline"))

try:
    from dynotears_tucker_cam import from_pandas_dynamic_tucker_cam
except ImportError:
    print("Error: Could not import Tucker-CAM module.")
    sys.exit(1)

def run_benchmark(d_vars, n_iters=5):
    print(f"Benchmarking d={d_vars}...", end="", flush=True)
    
    # Generate data
    n_samples = 100
    data = np.random.randn(n_samples, d_vars).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"V{i}" for i in range(d_vars)])
    
    # Config
    config = {
        'p': 5,
        'rank_w': 20,
        'rank_a': 10,
        'n_knots': 5,
        'lambda_smooth': 0.01,
        'max_iter': n_iters,
        'device': 'cpu'
    }
    
    # Run
    start_time = time.time()
    try:
        from_pandas_dynamic_tucker_cam(df, **config)
        elapsed = time.time() - start_time
        time_per_iter = elapsed / n_iters
        print(f" Done. {time_per_iter:.2f}s/iter")
        return time_per_iter
    except Exception as e:
        print(f" Failed: {e}")
        return None

def main():
    dims = [100, 500, 1000, 2889]
    results = []
    
    print("Running Scalability Benchmark Curve...")
    print("="*60)
    
    for d in dims:
        t_iter = run_benchmark(d, n_iters=3) # Very short run
        if t_iter:
            # Project to 100 iterations (typical convergence)
            projected_h = (t_iter * 100) / 3600
            results.append({
                'd': d,
                'sec_per_iter': t_iter,
                'projected_hours': projected_h
            })
            
    print("\n" + "="*60)
    print("NEW FIGURE 1 DATA (Projected for 100 iters)")
    print("="*60)
    print(f"{'d':<10} | {'sec/iter':<10} | {'Hours (100 it)':<15}")
    print("-" * 45)
    
    for r in results:
        print(f"{r['d']:<10} | {r['sec_per_iter']:<10.2f} | {r['projected_hours']:<15.2f}")
        
    print("-" * 45)

if __name__ == "__main__":
    main()
