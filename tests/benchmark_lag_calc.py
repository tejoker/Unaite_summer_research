import time
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def mock_find_optimal_lag(args):
    """
    Same logic as in preprocessing_no_mi.py
    """
    name, series_array, max_lag, lb_lags, alpha = args
    try:
        series_clean = series_array[~np.isnan(series_array)]
        if len(series_clean) < 10 or np.std(series_clean) == 0:
            return name, 1
            
        epsilon = 1e-10
        min_val = np.min(series_clean)
        shift = abs(min_val) + epsilon if min_val <= 0 else epsilon
        y = np.log1p(series_clean + shift)
        
        p_max = max(1, min(max_lag, len(y) // 5))

        for p in range(1, p_max + 1):
            try:
                mod = AutoReg(y, lags=p)
                res = mod.fit()
                resid = res.resid
                lb_res = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
                pval = lb_res['lb_pvalue'].iloc[0]
                if pval > alpha:
                    return name, p
            except:
                continue

        return name, p_max
    except:
        return name, 1

def run_benchmark():
    print("Benchmarking Sequential Lag Calculation (38 variables)...")
    
    # 1. Generate Mock Data (Size of SMD: ~23k rows)
    n_rows = 24000
    n_vars = 38
    print(f"Dataset Size: {n_rows} rows x {n_vars} cols")
    
    # Random walk data to make it realistic for AR models
    data = np.cumsum(np.random.normal(size=(n_rows, n_vars)), axis=0)
    
    # 2. Sequential Benchmark
    start_time = time.time()
    
    args_list = [
        (f"var_{i}", data[:, i], 20, 10, 0.05) 
        for i in range(n_vars)
    ]
    
    results = [mock_find_optimal_lag(arg) for arg in args_list]
    
    dict_results = dict(results)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Sequential processing validation complete.")
    print(f"Total Time: {elapsed:.2f} seconds")
    print(f"Average per variable: {elapsed/n_vars:.2f} seconds")
    
    if elapsed < 30:
        print("\nCONCLUSION: Sequential is FAST ENOUGH. Parallel overhead is not worth the deadlock risk.")
    else:
        print("\nCONCLUSION: Sequential might be slow.")

if __name__ == "__main__":
    run_benchmark()
