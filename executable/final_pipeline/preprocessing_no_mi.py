#!/usr/bin/env python3
"""
Preprocessing Pipeline (Polars Optimized)
Performs: stationarity testing, causal rolling median detrending, and lag optimization
"""

import os
import sys
import logging
from multiprocessing import Pool, cpu_count
import time

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler

# Configuration
ALPHA_STATIONARITY = 0.05

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def causal_robust_detrend_numpy(series, window=60):
    """
    Numpy-based causal rolling median (helper for parallel execution).
    """
    # Use pandas rolling for convenience within parallel worker if needed, 
    # but here we receive a numpy array.
    # Actually, efficient rolling median in pure numpy is hard.
    # We will compute this using Polars BEFORE parallel step for efficiency.
    pass 

def parallel_stationarity_test(args):
    """
    Worker function for stationarity test.
    Args: (name, series_array, alpha)
    """
    name, series_array, alpha = args
    try:
        # Drop NaN
        series_clean = series_array[~np.isnan(series_array)]
        
        if len(series_clean) == 0 or np.std(series_clean) == 0:
            return name, False, 1.0, 0.0
        
        # Log transform (add epsilon)
        epsilon = 1e-10
        min_val = np.min(series_clean)
        shift = abs(min_val) + epsilon if min_val <= 0 else epsilon
        
        with np.errstate(divide='ignore', invalid='ignore'):
            series_log = np.log1p(series_clean + shift)
            
        if not np.isfinite(series_log).all():
            return name, False, 1.0, 0.0
            
        # ADF test
        with np.errstate(divide='ignore', invalid='ignore'):
            adf_stat, adf_p, _, _, _, _ = adfuller(series_log, maxlag=20)

        # KPSS test
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            kpss_stat, kpss_p, _, _ = kpss(series_log, nlags='auto')

        is_stationary = (adf_p < alpha) and (kpss_p > alpha)
        return name, is_stationary, adf_p, kpss_p
    
    except Exception:
        return name, False, 1.0, 0.0

def find_optimal_lag_worker(args):
    """
    Worker function for lag optimization.
    Args: (name, series_array, max_lag, lb_lags, alpha)
    """
    name, series_array, max_lag, lb_lags, alpha = args
    try:
        # Clean data
        series_clean = series_array[~np.isnan(series_array)]
        if len(series_clean) < 10 or np.std(series_clean) == 0:
            return name, 1
            
        # Log transform
        epsilon = 1e-10
        min_val = np.min(series_clean)
        shift = abs(min_val) + epsilon if min_val <= 0 else epsilon
        y = np.log1p(series_clean + shift)
        
        if not np.isfinite(y).all():
            return name, 1
            
        p_max = max(1, min(max_lag, len(y) // 5))

        for p in range(1, p_max + 1):
            try:
                # Use statsmodels AutoReg
                mod = AutoReg(y, lags=p)
                res = mod.fit()
                resid = res.resid
                # Ljung-Box test on residuals
                lb_res = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
                pval = lb_res['lb_pvalue'].iloc[0]
                if pval > alpha:
                    return name, p
            except:
                continue

        return name, p_max
    except:
        return name, 1

def main():
    # Get input parameters
    input_file = os.getenv('INPUT_CSV_FILE', 'cleaned_dataset.csv')
    result_dir = os.getenv('RESULT_DIR', 'results')
    provided_lags_file = os.getenv('INPUT_LAGS_CSV', None)

    logger.info("="*80)
    logger.info("PREPROCESSING PIPELINE (POLARS OPTIMIZED)")
    logger.info("="*80)
    logger.info(f"Input file: {input_file}")
    
    preproc_dir = os.path.join(result_dir, 'preprocessing')
    os.makedirs(preproc_dir, exist_ok=True)

    # 1. Load Data
    logger.info("Loading data...")
    if input_file.endswith('.npy'):
        data_np = np.load(input_file)
        # Try to find columns
        cols_file = input_file.replace('.npy', '_columns.npy')
        if os.path.exists(cols_file):
            cols = np.load(cols_file, allow_pickle=True).tolist()
        else:
            cols = [f"var_{i}" for i in range(data_np.shape[1])]
        df = pl.DataFrame(data_np, schema=cols)
    else:
        # Polars CSV read is very fast
        df = pl.read_csv(input_file)
        # Ensure first column is not index if it looks like one (heuristic)
        if df.columns[0] == "" or "index" in df.columns[0].lower():
            df = df.drop(df.columns[0])
            
    var_names = df.columns
    logger.info(f"Data shape: {df.shape}")

    # 2. NaN Handling (Forward Fill)
    logger.info("Handling NaNs (Forward Fill)...")
    # Polars fill_null strategy
    df = df.fill_null(strategy="forward")
    # Drop remaining nulls (leading)
    df = df.drop_nulls()
    logger.info(f"Shape after NaN handling: {df.shape}")

    # 3. Causal Detrending (Rolling Median)
    logger.info("Applying Causal Rolling Median Detrending (window=60)...")
    start_time = time.time()
    
    # Log transform first: log1p(x + shift)
    # We do a global valid shift for simplicity or per-column? 
    # To keep it robust/fast in Polars, let's do per-column expression
    
    exprs = []
    for col in var_names:
        # Calculate shift for this column to handle negatives
        min_val = df[col].min()
        shift = abs(min_val) + 1e-10 if min_val <= 0 else 1e-10
        
        # Log transform
        log_col = (pl.col(col) + shift).log1p()
        
        # Rolling median (causal = shift 1)
        # Polars: rolling_median(window_size, min_periods).shift(1)
        trend = log_col.rolling_median(window_size=60, min_periods=1).shift(1)
        
        # We need to fill first value which becomes null due to shift
        # Fill with first existing value (backfill logic for the very first point)
        trend = trend.fill_null(strategy="backward") 
        
        # Cycle = Log - Trend
        cycle = log_col - trend
        exprs.append(cycle.alias(col))
        
    df_detrended = df.select(exprs)
    
    # Drop the first row if it's garbage? 
    # Actually rolling_median with shift(1) puts null at pos 0.
    # We used fill_null backward, so pos 0 is cycle[0] = log[0] - log[0] = 0.
    # This is fine.
    
    elapsed = time.time() - start_time
    logger.info(f"Detrending complete in {elapsed:.2f}s")

    # 4. Stationarity Test (Parallel)
    # (Optional: Only if we want to filter variables. But pipeline usually keeps all)
    # The original script kept only stationary ones.
    
    logger.info("Testing stationarity...")
    # Convert back to dict of numpy arrays for multiprocessing
    # This overhead is small compared to ADF test time
    data_dict = {col: df_detrended[col].to_numpy() for col in var_names}
    
    with Pool(cpu_count()) as pool:
        results = pool.map(
            parallel_stationarity_test,
            [(col, data_dict[col], ALPHA_STATIONARITY) for col in var_names]
        )
        
    stationary_vars = []
    for name, is_stat, _, _ in results:
        if is_stat:
            stationary_vars.append(name)
            
    logger.info(f"Stationary variables: {len(stationary_vars)}/{len(var_names)}")
    
    if len(stationary_vars) == 0:
        logger.warning("No stationary variables found! Using all variables as fallback.")
        stationary_vars = var_names

    # Filter DataFrame
    df_final = df_detrended.select(stationary_vars)
    
    # 5. Standardization
    logger.info("Standardizing data...")
    # Polars has no built-in StandardScaler, use sklearn on numpy matrix
    data_vals = df_final.to_numpy()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_vals)
    
    # 6. Save detrended data
    diff_file = os.path.join(preproc_dir, f"{os.path.basename(input_file).split('.')[0]}_differenced_stationary_series.npy")
    np.save(diff_file, data_scaled)
    
    cols_file = os.path.join(preproc_dir, f"{os.path.basename(input_file).split('.')[0]}_columns.npy")
    np.save(cols_file, np.array(stationary_vars))
    
    logger.info(f"Saved processed data to {diff_file}")

    # 7. Lag Optimization
    if provided_lags_file and os.path.exists(provided_lags_file):
        logger.info(f"Using provided lags from {provided_lags_file}")
        # Identify type and copy
        dest = os.path.join(preproc_dir, f"{os.path.basename(input_file).split('.')[0]}_optimal_lags.npy")
        if provided_lags_file.endswith('.npy'):
            import shutil
            shutil.copy(provided_lags_file, dest)
        else:
            # Assume CSV
            lags_df = pl.read_csv(provided_lags_file)
            # Convert to structured array for compatibility
            lags_list = [(r[0], r[1]) for r in lags_df.iter_rows()]
            dtype = [('variable', 'U100'), ('optimal_lag', 'i4')]
            np.save(dest, np.array(lags_list, dtype=dtype))
    else:
        logger.info("Calculating optimal lags...")
        # Prepare args
        # We reuse data_dict but restricted to stationary_vars
        # Need to re-extract numpy arrays from scaled data?
        # Optimal lag should be calculated on DETRENDED data (before standardization or after?)
        # Original script did it on detrended (log-transformed).
        # We can use the data in data_dict (which is detrended log-transformed)
        
        args_list = [
            (col, data_dict[col], 20, 10, 0.05) 
            for col in stationary_vars
        ]
        
        with Pool(cpu_count()) as pool:
            lag_results = pool.map(find_optimal_lag_worker, args_list)
            
        # Save results
        lags_file = os.path.join(preproc_dir, f"{os.path.basename(input_file).split('.')[0]}_optimal_lags.npy")
        dtype = [('variable', 'U100'), ('optimal_lag', 'i4')]
        np.save(lags_file, np.array(lag_results, dtype=dtype))
        logger.info(f"Saved optimal lags to {lags_file}")

    logger.info("Preprocessing Pipeline Complete.")

if __name__ == "__main__":
    main()
