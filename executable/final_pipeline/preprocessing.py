#!/usr/bin/env python3
import os
import sys
import logging
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools

import numpy as np
import pandas as pd

from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.contingency_tables import Table

# Optional import for enhanced performance
try:
    from resource_manager import get_resource_manager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
    get_resource_manager = None

# ==============================
# Config
# ==============================
ALPHA_STATIONARITY = 0.05
ALPHA_MI = 0.01              # significance for chi-square dependence test used by MI mask
MI_BINS = 5                  # quantile bins for discretization
INCLUDE_LAG0_MASK = False    # usually False: do NOT constrain contemporaneous layer by MI
FORCE_ALLOW_SELF_LAGS = True # keep i->i at lag>=1 always allowed (recommended)

# ==============================
# Stationarity / lag utilities
# ==============================

def parallel_stationarity_test(series_data, alpha=ALPHA_STATIONARITY):
    """
    Apply log1p, first-difference and then perform ADF and KPSS tests to assess stationarity.
    Args:
        series_data: tuple of (series_name, series_values)
        alpha: significance level
    Returns: (series_name, is_stationary, adf_pvalue, kpss_pvalue)
    """
    series_name, series_values = series_data
    try:
        series = pd.Series(series_values, name=series_name)
        series_clean = series.dropna()
        
        if len(series_clean) < 10:  # Need minimum data points
            return series_name, False, 1.0, 0.0
        
        series_log = np.log1p(series_clean)              # stabilize variance
        series_diff = series_log.diff().dropna()         # remove stochastic trend

        if len(series_diff) < 5:  # Need minimum data for tests
            return series_name, False, 1.0, 0.0

        # ADF (H0: non-stationary)
        adf_stat, adf_p, _, _, _, _ = adfuller(series_diff, maxlag=min(20, len(series_diff)//4))
        # KPSS (H0: stationary)
        kpss_stat, kpss_p, _, _ = kpss(series_diff, nlags='auto')

        is_stationary = (adf_p < alpha) and (kpss_p > alpha)
        return series_name, is_stationary, adf_p, kpss_p
    except Exception as e:
        logging.warning(f"Stationarity test failed for {series_name}: {e}")
        return series_name, False, 1.0, 0.0

def test_stationarity_parallel(df, alpha=ALPHA_STATIONARITY, n_workers=None):
    """
    Test stationarity of all series in DataFrame using parallel processing
    Falls back to sequential processing if parallel processing fails
    """
    if n_workers is None:
        if RESOURCE_MANAGER_AVAILABLE:
            try:
                resource_manager = get_resource_manager()
                n_workers = resource_manager.get_cpu_workers() if resource_manager.current_config else cpu_count()
            except:
                n_workers = cpu_count()
        else:
            n_workers = cpu_count()
    
    logging.info(f"Testing stationarity for {len(df.columns)} series using {n_workers} workers")
    
    # Prepare data for parallel processing
    series_data = [(col, df[col].values) for col in df.columns]
    
    try:
        # Try parallel processing first
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_series = {
                executor.submit(parallel_stationarity_test, data, alpha): data[0] 
                for data in series_data
            }
            
            # Collect results with progress bar
            with tqdm(total=len(series_data), desc="Stationarity tests") as pbar:
                for future in as_completed(future_to_series):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        series_name = future_to_series[future]
                        logging.error(f"Stationarity test failed for {series_name}: {e}")
                        results.append((series_name, False, 1.0, 0.0))
                    pbar.update(1)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results, columns=['series', 'is_stationary', 'adf_pvalue', 'kpss_pvalue'])
        return results_df
        
    except Exception as e:
        logging.warning(f"Parallel processing failed ({e}), falling back to sequential processing")
        return test_stationarity_sequential(df, alpha)

def test_stationarity_sequential(df, alpha=ALPHA_STATIONARITY):
    """
    Fallback: Test stationarity sequentially
    """
    logging.info(f"Testing stationarity for {len(df.columns)} series sequentially")
    results = []
    
    for col in tqdm(df.columns, desc="Stationarity tests (sequential)"):
        try:
            series_data = (col, df[col].values)
            result = parallel_stationarity_test(series_data, alpha)
            results.append(result)
        except Exception as e:
            logging.error(f"Stationarity test failed for {col}: {e}")
            results.append((col, False, 1.0, 0.0))
    
    results_df = pd.DataFrame(results, columns=['series', 'is_stationary', 'adf_pvalue', 'kpss_pvalue'])
    return results_df


def find_optimal_lag_single(series_data, max_lag=20, lb_lags=10, alpha=0.05):
    """
    Pick the smallest AR lag p in [1..max_lag] such that AutoReg(p) residuals
    pass Ljung–Box (p-value > alpha).
    Args:
        series_data: tuple of (series_name, series_values)
    Returns: (series_name, optimal_lag)
    """
    series_name, series_values = series_data
    try:
        series = pd.Series(series_values, name=series_name)
        y = np.log1p(series.dropna()).diff().dropna()
        
        if len(y) < 20:  # Need minimum data points
            return series_name, 1
        
        p_max = max(1, min(max_lag, len(y) // 5))
        for p in range(1, p_max + 1):
            try:
                model = AutoReg(y, lags=p).fit()
                resid = model.resid
                if len(resid) <= lb_lags:
                    continue
                pval = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)['lb_pvalue'].iloc[0]
                if pval > alpha:
                    return series_name, p
            except Exception:
                continue
        return series_name, p_max
    except Exception as e:
        logging.warning(f"Optimal lag calculation failed for {series_name}: {e}")
        return series_name, 1

def find_optimal_lags_parallel(df, max_lag=20, lb_lags=10, alpha=0.05, n_workers=None):
    """
    Find optimal lag for all series in DataFrame using parallel processing
    Falls back to sequential processing if parallel processing fails
    """
    if n_workers is None:
        if RESOURCE_MANAGER_AVAILABLE:
            try:
                resource_manager = get_resource_manager()
                n_workers = resource_manager.get_cpu_workers() if resource_manager.current_config else cpu_count()
            except:
                n_workers = cpu_count()
        else:
            n_workers = cpu_count()
    
    logging.info(f"Finding optimal lags for {len(df.columns)} series using {n_workers} workers")
    
    # Prepare data for parallel processing
    series_data = [(col, df[col].values) for col in df.columns]
    
    try:
        # Try parallel processing first
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_series = {
                executor.submit(find_optimal_lag_single, data, max_lag, lb_lags, alpha): data[0] 
                for data in series_data
            }
            
            # Collect results with progress bar
            with tqdm(total=len(series_data), desc="Optimal lag calculation") as pbar:
                for future in as_completed(future_to_series):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        series_name = future_to_series[future]
                        logging.error(f"Optimal lag calculation failed for {series_name}: {e}")
                        results.append((series_name, 1))
                    pbar.update(1)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results, columns=['variable', 'optimal_lag'])
        return results_df
        
    except Exception as e:
        logging.warning(f"Parallel lag calculation failed ({e}), falling back to sequential processing")
        return find_optimal_lags_sequential(df, max_lag, lb_lags, alpha)

def find_optimal_lags_sequential(df, max_lag=20, lb_lags=10, alpha=0.05):
    """
    Fallback: Find optimal lags sequentially
    """
    logging.info(f"Finding optimal lags for {len(df.columns)} series sequentially")
    results = []
    
    for col in tqdm(df.columns, desc="Optimal lag calculation (sequential)"):
        try:
            series_data = (col, df[col].values)
            result = find_optimal_lag_single(series_data, max_lag, lb_lags, alpha)
            results.append(result)
        except Exception as e:
            logging.error(f"Optimal lag calculation failed for {col}: {e}")
            results.append((col, 1))
    
    results_df = pd.DataFrame(results, columns=['variable', 'optimal_lag'])
    return results_df


def find_cross_dependencies(stationary_series, max_lag=20, alpha=0.05):
    """
    Very simple cross-correlation scan for significant positive lags.
    Returns: {series1: {series2: optimal_lag, ...}, ...}
    """
    results = {}
    series_names = list(stationary_series.keys())
    for i, name1 in enumerate(series_names):
        results[name1] = {}
        series1 = np.log1p(stationary_series[name1].dropna()).diff().dropna()
        for name2 in series_names:
            if name1 == name2:
                continue
            series2 = np.log1p(stationary_series[name2].dropna()).diff().dropna()
            s1, s2 = series1.align(series2, join='inner')
            if len(s1) < max_lag:
                continue
            cross_corr = np.correlate(s1, s2, mode='full')
            mid = len(cross_corr) // 2
            significant = []
            threshold = 1.96 / np.sqrt(len(s1))  # 95% approx
            for lag in range(1, min(max_lag + 1, len(cross_corr) - mid)):
                denom = (np.std(s1) * np.std(s2) * len(s1))
                if denom == 0:
                    continue
                corr_value = cross_corr[mid + lag] / denom
                if abs(corr_value) > threshold:
                    significant.append((lag, abs(corr_value)))
            if significant:
                optimal_lag = sorted(significant, key=lambda x: x[1], reverse=True)[0][0]
                results[name1][name2] = optimal_lag
    return results

# ==============================
# MI-mask utilities
# ==============================

def _discretize_quantiles(x: np.ndarray, bins: int) -> np.ndarray:
    """Discretize into ~equal-frequency bins; stable even if there are ties."""
    x = np.asarray(x, dtype=float)
    n = np.count_nonzero(~np.isnan(x))
    if n == 0 or bins <= 1:
        return np.zeros_like(x, dtype=int)
    # guard for constant series
    if np.nanmax(x) - np.nanmin(x) == 0:
        return np.zeros_like(x, dtype=int)
    # quantile edges
    qs = np.linspace(0, 1, bins + 1)
    edges = np.nanquantile(x, qs)
    # make edges strictly increasing
    edges = np.unique(edges)
    if len(edges) <= 2:
        return (x > edges[0]).astype(int)
    # bin
    # rightmost inclusive
    idx = np.digitize(x, edges[1:-1], right=False)
    idx[np.isnan(x)] = -1  # mark NaNs
    return idx.astype(int)


def _chi2_pvalue_from_crosstab(x_d: np.ndarray, y_d: np.ndarray) -> float:
    """Pearson chi-square independence test p-value using statsmodels.Table."""
    # drop rows with NaN labels
    mask = (x_d >= 0) & (y_d >= 0)
    if mask.sum() < 3:
        return 1.0  # not enough data -> treat as independent
    xi = x_d[mask]
    yi = y_d[mask]
    # build contingency table
    # (using pandas crosstab is convenient & stable)
    ct = pd.crosstab(pd.Series(xi), pd.Series(yi)).to_numpy()
    if ct.size == 0:
        return 1.0
    try:
        p = Table(ct).test_nominal_association().pvalue
    except Exception:
        # Very sparse contingency -> assume independent
        p = 1.0
    return float(p)


def df_grasp_mi_lagged(df: pd.DataFrame, L: int, alpha: float = ALPHA_MI, bins: int = MI_BINS,
                       include_lag0: bool = INCLUDE_LAG0_MASK,
                       force_allow_self_lags: bool = FORCE_ALLOW_SELF_LAGS) -> np.ndarray:
    """
    Build a boolean mask M[d, d, L+1] using a GRASP-style (lag-aware) dependence screen:
      - discretize all series into quantile bins,
      - for each (child=i, parent=j, lag in 0..L), test chi-square independence for Y_i(t) vs X_j(t-lag),
      - keep edge if p < alpha.
    Notes:
      - lag 0 (contemporaneous) can be skipped via include_lag0=False.
      - self-lags (i==j, lag>=1) are force-allowed if force_allow_self_lags=True.
    """
    cols = list(df.columns)
    d = len(cols)
    if d == 0:
        raise ValueError("Empty dataframe passed to df_grasp_mi_lagged.")

    # Align all variables to same index and drop rows with any NaN
    df = df.copy()
    df = df.dropna(axis=0, how="any")
    if len(df) < (L + 5):
        logging.warning(f"[MI] Very short series vs. L={L} (rows={len(df)}). Results may be weak.")

    # Precompute discretized present and lagged versions
    disc_now = {c: _discretize_quantiles(df[c].to_numpy(), bins) for c in cols}
    disc_lag = {(c, lag): (_discretize_quantiles(df[c].shift(lag).to_numpy(), bins) if lag > 0 else disc_now[c])
                for c in cols for lag in range(L + 1)}

    # Build mask
    M = np.zeros((d, d, L + 1), dtype=bool)

    lag_range = range(0, L + 1) if include_lag0 else range(1, L + 1)

    for i, ci in enumerate(cols):                 # child
        yi = disc_now[ci]
        for j, cj in enumerate(cols):             # parent
            for lag in lag_range:
                if i == j and lag >= 1 and force_allow_self_lags:
                    M[i, j, lag] = True
                    continue
                if i == j and lag == 0:
                    # forbid contemporaneous self-loop
                    M[i, j, lag] = False
                    continue
                xj = disc_lag[(cj, lag)]
                pval = _chi2_pvalue_from_crosstab(xj, yi)
                M[i, j, lag] = (pval < alpha)

    # Enforce no contemporaneous self-loop
    for i in range(d):
        M[i, i, 0] = False

    return M


def generate_window_mi_masks(df: pd.DataFrame, L: int, window_sizes: list = [11, 22, 50], 
                             alpha: float = ALPHA_MI, bins: int = MI_BINS,
                             include_lag0: bool = INCLUDE_LAG0_MASK,
                             force_allow_self_lags: bool = FORCE_ALLOW_SELF_LAGS,
                             max_windows_per_size: int = 100) -> dict:
    """
    Generate MI masks for different window sizes by sampling windows from the dataset.
    
    Args:
        df: DataFrame with differenced stationary series
        L: Maximum lag order
        window_sizes: List of window sizes to test
        alpha: Significance level for independence test
        bins: Number of bins for discretization
        include_lag0: Whether to include lag 0 (contemporaneous) connections
        force_allow_self_lags: Whether to force allow self-connections at lag >= 1
        max_windows_per_size: Maximum number of windows to sample per window size
    
    Returns:
        Dictionary with window_size -> list of MI masks
    """
    logging.info(f'Generating window-specific MI masks for window sizes: {window_sizes}')
    
    window_mi_masks = {}
    df_clean = df.dropna()
    n_rows = len(df_clean)
    
    for window_size in window_sizes:
        if window_size >= n_rows:
            logging.warning(f'Window size {window_size} >= dataset size {n_rows}, skipping')
            continue
            
        logging.info(f'Computing MI masks for window size {window_size}')
        
        # Calculate how many windows we can extract
        max_possible_windows = n_rows - window_size + 1
        n_sample_windows = min(max_windows_per_size, max_possible_windows)
        
        # Sample window starting positions uniformly
        if n_sample_windows < max_possible_windows:
            step = max_possible_windows // n_sample_windows
            window_starts = list(range(0, max_possible_windows, step))[:n_sample_windows]
        else:
            window_starts = list(range(max_possible_windows))
        
        masks_for_size = []
        
        for i, start_idx in enumerate(window_starts):
            if i % 20 == 0:  # Progress logging
                logging.info(f'  Processing window {i+1}/{len(window_starts)} for size {window_size}')
                
            # Extract window
            window_df = df_clean.iloc[start_idx:start_idx + window_size].copy()
            
            # Reset index to ensure continuous indexing
            window_df = window_df.reset_index(drop=True)
            
            try:
                # Compute MI mask for this window
                mask = df_grasp_mi_lagged(
                    window_df, 
                    L=L, 
                    alpha=alpha, 
                    bins=bins,
                    include_lag0=include_lag0,
                    force_allow_self_lags=force_allow_self_lags
                )
                masks_for_size.append(mask)
                
            except Exception as e:
                logging.warning(f'Failed to compute MI mask for window {i} (size {window_size}): {e}')
                continue
        
        if masks_for_size:
            window_mi_masks[window_size] = masks_for_size
            logging.info(f'Generated {len(masks_for_size)} MI masks for window size {window_size}')
        else:
            logging.error(f'No valid MI masks generated for window size {window_size}')
    
    return window_mi_masks


def save_window_mi_masks(window_mi_masks: dict, preprocessing_dir: str, input_basename: str):
    """Save window-specific MI masks to files"""
    import pickle
    
    # Save as pickle file for dbn_dynotears to load
    masks_file = os.path.join(preprocessing_dir, f'{input_basename}_window_mi_masks.pkl')
    with open(masks_file, 'wb') as f:
        pickle.dump(window_mi_masks, f)
    logging.info(f'Saved window MI masks to {masks_file}')
    
    # Also save a summary CSV for inspection
    summary_rows = []
    for window_size, masks in window_mi_masks.items():
        for mask_idx, mask in enumerate(masks):
            n_allowed = int(np.sum(mask))
            total_edges = int(mask.size)
            summary_rows.append({
                'window_size': window_size,
                'mask_index': mask_idx,
                'allowed_edges': n_allowed,
                'total_edges': total_edges,
                'sparsity_ratio': n_allowed / total_edges if total_edges > 0 else 0.0
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(preprocessing_dir, f'{input_basename}_window_mi_masks_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logging.info(f'Saved window MI masks summary to {summary_file}')


# ==============================
# Main
# ==============================

def main():
    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info('Starting stationarity analysis...')

    # Load dataset
    input_file = os.getenv('INPUT_CSV_FILE', 'cleaned_dataset.csv')
    logging.info(f'Loading dataset from: {input_file}')
    df_raw = pd.read_csv(input_file)
    logging.info(f'Loaded data with {df_raw.shape[0]} rows and {df_raw.shape[1]} variables')
    
    # Create output directory and file naming based on input file
    input_path = os.path.abspath(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Determine data directory - look for project root first
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = None
    
    # Look for project root by going up from script directory to find 'data' folder
    current_dir = script_dir
    for _ in range(5):  # Search up to 5 levels up from script
        potential_data_dir = os.path.join(current_dir, 'data')
        if os.path.exists(potential_data_dir) and os.path.isdir(potential_data_dir):
            project_root = current_dir
            data_dir = potential_data_dir
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root
            break
        current_dir = parent_dir
    
    # If still not found, try to find data directory in parent directories of input file
    if project_root is None:
        current_dir = os.path.dirname(input_path)
        for _ in range(5):  # Search up to 5 levels up
            if os.path.basename(current_dir) == 'data':
                data_dir = current_dir
                project_root = os.path.dirname(current_dir)
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
    
    # Last resort: use current working directory if it has a data folder
    if project_root is None:
        cwd_data = os.path.join(os.getcwd(), 'data')
        if os.path.exists(cwd_data) and os.path.isdir(cwd_data):
            data_dir = cwd_data
            project_root = os.getcwd()
        else:
            # Fallback: create data directory relative to script
            project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up from executable/final_pipeline
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
    # Get result directory from environment variable (set by launcher.py)
    result_dir = os.getenv('RESULT_DIR')
    if result_dir:
        # Use the result directory provided by launcher
        preprocessing_dir = os.path.join(result_dir, 'preprocessing')
    else:
        # Fallback to old behavior if not launched by launcher.py
        preprocessing_dir = os.path.join(data_dir, 'preprocessing')
    
    os.makedirs(preprocessing_dir, exist_ok=True)
    
    logging.info(f'Output directory: {preprocessing_dir}')

    # Log-transform before differencing
    df = np.log1p(df_raw)
    logging.info('Applied log1p transform to raw data')

    # Parallel config
    num_workers = min(cpu_count(), 32)
    logging.info(f'Using {num_workers} workers for parallel processing')

    cols = df.columns
    n_cols = len(cols)
    batch_size = 1

    stationary_series = {}
    differenced_df = pd.DataFrame()

    # Use new parallel stationarity testing function
    try:
        logging.info('Running stationarity tests using enhanced parallel processing...')
        stationarity_results = test_stationarity_parallel(df, alpha=ALPHA_STATIONARITY)
        
        for _, row in stationarity_results.iterrows():
            name = row['series']
            is_stat = row['is_stationary']
            adf_p = row['adf_pvalue']
            kpss_p = row['kpss_pvalue']
            
            logging.info(f'Processed {name}: stationary={is_stat}, ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}')
            if is_stat:
                stationary_series[name] = df_raw[name]
                series_diff = df[name].diff().dropna()
                differenced_df = pd.concat([differenced_df, series_diff.rename(f'{name}_diff')], axis=1)
                
    except Exception as e:
        logging.error(f'Enhanced parallel stationarity testing failed: {e}')
        logging.info('Falling back to original batch processing...')
        
        # Fallback to original batch processing
        for batch_start in range(0, n_cols, batch_size):
            batch_end = min(batch_start + batch_size, n_cols)
            batch_cols = cols[batch_start:batch_end]
            logging.info(f'Processing batch {batch_start//batch_size + 1}, columns {batch_start+1} to {batch_end}')

            # Process sequentially as fallback
            results = []
            for col in tqdm(batch_cols, desc='Batch tests (sequential)'):
                try:
                    series_data = (col, df[col].values)
                    result = parallel_stationarity_test(series_data)
                    results.append(result)
                except Exception as e:
                    logging.error(f'Stationarity test failed for {col}: {e}')
                    results.append((col, False, 1.0, 0.0))

            for name, is_stat, adf_p, kpss_p in results:
                logging.info(f'Processed {name}: stationary={is_stat}, ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}')
                if is_stat:
                    stationary_series[name] = df_raw[name]
                    series_diff = df[name].diff().dropna()
                    differenced_df = pd.concat([differenced_df, series_diff.rename(f'{name}_diff')], axis=1)

    # Save differenced stationary series
    if not differenced_df.empty:
        # Create recognizable filename based on input
        output_file = os.getenv('OUTPUT_DIFFERENCED_CSV', 
                               os.path.join(preprocessing_dir, f'{input_basename}_differenced_stationary_series.csv'))
        differenced_df.to_csv(output_file)
        logging.info(f'Saved {len(differenced_df.columns)} differenced stationary series to {output_file}')
    else:
        logging.warning('No stationary series found; differenced_stationary_series.csv will not be written.')
        output_file = None

    # Find optimal AR lag for each stationary series
    logging.info(f'Found {len(stationary_series)} stationary series. Finding optimal lags...')
    
    # Use parallel processing if available, otherwise fallback to sequential
    try:
        # Create a temporary DataFrame for parallel processing
        stationary_df = pd.DataFrame(stationary_series)
        lag_results_df = find_optimal_lags_parallel(stationary_df)
        lag_results = dict(zip(lag_results_df['variable'], lag_results_df['optimal_lag']))
        logging.info(f'Optimal lags found for {len(lag_results)} series')
    except Exception as e:
        logging.warning(f'Parallel lag calculation failed ({e}), using sequential fallback')
        lag_results = {}
        for name, series in tqdm(stationary_series.items(), desc='Finding optimal lags (sequential)'):
            try:
                series_data = (name, series.values)
                _, lag = find_optimal_lag_single(series_data)
                lag_results[name] = lag
                logging.info(f'Optimal lag for {name}: {lag_results[name]}')
            except Exception as e:
                logging.error(f'Error finding optimal lag for {name}: {str(e)}')

    # Cross-dependencies (lightweight scan)
    logging.info('Finding cross-dependencies between stationary series...')
    try:
        cross_dependencies = find_cross_dependencies(stationary_series)
        logging.info(f'Found cross-dependencies for {len(cross_dependencies)} series')
    except Exception as e:
        logging.error(f'Error finding cross-dependencies: {str(e)}')
        cross_dependencies = {}

    # Summary CSV
    summary = []
    for name in stationary_series.keys():
        if name in lag_results:
            entry = {'variable': name, 'optimal_lag': lag_results[name]}
            if name in cross_dependencies:
                for dep_name, dep_lag in cross_dependencies[name].items():
                    entry[f'cross_lag_to_{dep_name}'] = dep_lag
            summary.append(entry)
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        optimal_lags_file = os.getenv('OUTPUT_LAGS_CSV', 
                                     os.path.join(preprocessing_dir, f'{input_basename}_optimal_lags.csv'))
        summary_df.to_csv(optimal_lags_file, index=False)
        logging.info(f'Results saved to {optimal_lags_file}')
    else:
        logging.warning('optimal_lags.csv not written (no stationary variables).')
        optimal_lags_file = None

    # ==============================
    # Window-specific MI Masks
    # ==============================
    # Decide L (global order) as max optimal_lag across variables (fallback 1)
    if not differenced_df.empty:
        if not summary_df.empty and 'optimal_lag' in summary_df.columns:
            L = int(max(1, int(summary_df['optimal_lag'].max())))
        else:
            logging.warning('No optimal_lags available; defaulting MI mask L=1.')
            L = 1

        # Use the differenced stationary series, aligned & NaNs dropped
        df_for_mi = differenced_df.dropna()
        if df_for_mi.shape[0] < (L + 10):
            logging.warning(f'[MI] Short series ({df_for_mi.shape[0]} rows) vs. L={L}; mask may be weak.')

        if df_for_mi.shape[1] < 2:
            logging.warning('[MI] Fewer than 2 stationary variables; mask will be trivial.')
            
        try:
            logging.info(f'Building window-specific MI masks with L={L}, alpha={ALPHA_MI}, bins={MI_BINS}, '
                         f'include_lag0={INCLUDE_LAG0_MASK}, self_lags_allowed={FORCE_ALLOW_SELF_LAGS}')
            
            # Define window sizes to test (these will be used by dbn_dynotears)
            # Base window sizes on data length and lag order
            data_length = len(df_for_mi)
            min_window = max(L + 5, 11)  # At least lag + 5, minimum 11
            max_window = min(data_length // 4, 100)  # At most 1/4 of data or 100
            
            if max_window <= min_window:
                window_sizes = [min_window]
            else:
                # Generate 3-5 window sizes logarithmically spaced
                import math
                n_sizes = min(5, max(3, (max_window - min_window) // 10))
                if n_sizes == 1:
                    window_sizes = [min_window]
                else:
                    # Logarithmic spacing
                    log_min = math.log(min_window)
                    log_max = math.log(max_window)
                    window_sizes = [int(math.exp(log_min + i * (log_max - log_min) / (n_sizes - 1))) 
                                  for i in range(n_sizes)]
                    window_sizes = sorted(list(set(window_sizes)))  # Remove duplicates and sort
            
            logging.info(f'Computing MI masks for window sizes: {window_sizes}')
            
            # Generate window-specific MI masks
            window_mi_masks = generate_window_mi_masks(
                df_for_mi,
                L=L,
                window_sizes=window_sizes,
                alpha=ALPHA_MI,
                bins=MI_BINS,
                include_lag0=INCLUDE_LAG0_MASK,
                force_allow_self_lags=FORCE_ALLOW_SELF_LAGS,
                max_windows_per_size=50  # Sample up to 50 windows per size
            )
            
            # Save window-specific MI masks
            if window_mi_masks:
                save_window_mi_masks(window_mi_masks, preprocessing_dir, input_basename)
                
                # Also create a backward-compatible single MI mask file (using first window size)
                first_window_size = min(window_mi_masks.keys())
                if window_mi_masks[first_window_size]:
                    # Average the masks for the smallest window size as a fallback
                    avg_mask = np.mean([mask.astype(float) for mask in window_mi_masks[first_window_size]], axis=0)
                    # Threshold at 0.5 to convert back to boolean
                    M_fallback = avg_mask > 0.5
                    
                    # Save fallback files for backward compatibility
                    mi_mask_npy_file = os.path.join(preprocessing_dir, f'{input_basename}_mi_mask.npy')
                    np.save(mi_mask_npy_file, M_fallback.astype(bool))
                    
                    # Human-readable CSV of averaged mask
                    d = df_for_mi.shape[1]
                    names = list(df_for_mi.columns)
                    rows = []
                    for i in range(d):           # child
                        for j in range(d):       # parent
                            for lag in range(L + 1):
                                rows.append({
                                    'parent': names[j],
                                    'child': names[i],
                                    'lag': lag,
                                    'allowed': int(M_fallback[i, j, lag])
                                })
                    mi_mask_file = os.getenv('OUTPUT_MI_MASK_CSV', 
                                           os.path.join(preprocessing_dir, f'{input_basename}_mi_mask_edges.csv'))
                    pd.DataFrame(rows).to_csv(mi_mask_file, index=False)
                    logging.info(f'[MI] Wrote backward-compatible MI mask files')
                
            else:
                logging.error('[MI] No window MI masks generated')

        except Exception as e:
            logging.error(f'[MI] Failed to build/save window MI masks: {e}')
    else:
        logging.warning('[MI] Skipping MI mask (no differenced_stationary_series).')

    logging.info('Analysis completed')

if __name__ == '__main__':
    main()
