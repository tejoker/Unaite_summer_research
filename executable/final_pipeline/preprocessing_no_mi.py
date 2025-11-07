#!/usr/bin/env python3
"""
Preprocessing Pipeline
Performs: stationarity testing, differencing, and lag optimization
"""

import os
import sys
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler

# Configuration
ALPHA_STATIONARITY = 0.05

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def parallel_stationarity_test(series, alpha=ALPHA_STATIONARITY):
    """
    Apply log1p, first-difference and perform ADF and KPSS tests
    Returns: (series_name, is_stationary, adf_pvalue, kpss_pvalue)
    """
    try:
        series_clean = series.dropna()
        
        # Check for constant/zero-variance series
        if len(series_clean) == 0 or series_clean.std() == 0:
            return series.name, False, 1.0, 0.0
        
        series_log = np.log1p(series_clean)
        
        # Check for inf/nan after log transform
        if not np.isfinite(series_log).all():
            return series.name, False, 1.0, 0.0
        
        series_diff = series_log.diff().dropna()
        
        # Check for inf/nan after differencing
        if len(series_diff) == 0 or not np.isfinite(series_diff).all():
            return series.name, False, 1.0, 0.0

        # ADF test (H0: non-stationary)
        adf_stat, adf_p, _, _, _, _ = adfuller(series_diff, maxlag=20)

        # KPSS test (H0: stationary)
        kpss_stat, kpss_p, _, _ = kpss(series_diff, nlags='auto')

        is_stationary = (adf_p < alpha) and (kpss_p > alpha)
        return series.name, is_stationary, adf_p, kpss_p
    
    except Exception as e:
        # If any test fails, mark as non-stationary and skip
        return series.name, False, 1.0, 0.0


def find_optimal_lag(series, max_lag=20, lb_lags=10, alpha=0.05):
    """
    Find smallest AR lag p where residuals pass Ljung-Box test
    """
    try:
        y = np.log1p(series.dropna()).diff().dropna()
        
        # Check for inf/nan or insufficient data
        if len(y) < 10 or not np.isfinite(y).all():
            return 1  # Default to lag 1
        
        p_max = max(1, min(max_lag, len(y) // 5))

        for p in range(1, p_max + 1):
            try:
                resid = AutoReg(y, lags=p).fit().resid
                pval = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)['lb_pvalue'].iloc[0]
                if pval > alpha:
                    return p
            except:
                continue

        return p_max
    except Exception as e:
        return 1  # Default to lag 1 on any error


def main():
    """Main preprocessing workflow"""

    # Get input parameters from environment
    input_file = os.getenv('INPUT_CSV_FILE', 'cleaned_dataset.csv')
    result_dir = os.getenv('RESULT_DIR', 'results')
    provided_lags_file = os.getenv('INPUT_LAGS_CSV', None)  # Check if lags file is provided

    logger.info("="*80)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("="*80)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Result directory: {result_dir}")
    
    if provided_lags_file:
        logger.info(f"Provided lags file: {provided_lags_file}")
        logger.info("  -> Will use existing lags instead of recalculating")

    # The parent script already provides the correct directory.
    # Just use it directly.
    preproc_dir = result_dir
    os.makedirs(preproc_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_file}...")
    df_raw = pd.read_csv(input_file, index_col=0)

    logger.info(f"Data loaded successfully!")
    logger.info(f"Data shape: {df_raw.shape} ({df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns)")
    logger.info(f"Columns: {df_raw.columns.tolist()}")

    # Get base filename
    basename = os.path.splitext(os.path.basename(input_file))[0]

    # Step 1: Test stationarity in parallel
    logger.info("\nStep 1: Testing stationarity for all series")
    logger.info("-"*40)
    logger.info(f"Running stationarity tests on {len(df_raw.columns)} columns using {cpu_count()} CPU cores...")

    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            parallel_stationarity_test,
            [(df_raw[col], ALPHA_STATIONARITY) for col in df_raw.columns]
        )

    logger.info(f"Stationarity tests completed!")

    stationary_series = {}
    non_stationary = []

    for col_name, is_stat, adf_p, kpss_p in results:
        logger.info(f"{col_name}: stationary={is_stat} (ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f})")
        if is_stat:
            stationary_series[col_name] = df_raw[col_name]
        else:
            non_stationary.append(col_name)

    if non_stationary:
        logger.warning(f"Non-stationary series: {non_stationary}")

    # Step 2: Apply differencing to all series
    logger.info("\nStep 2: Applying differencing to all series")
    logger.info("-"*40)

    df_diff = pd.DataFrame()

    for col in df_raw.columns:
        series_log = np.log1p(df_raw[col])
        series_diff = series_log.diff()
        df_diff[f"{col}_diff"] = series_diff
    
    # Remove the first row (NaN from diff) across all columns
    df_diff = df_diff.iloc[1:].reset_index(drop=True)
    
    # Clean any remaining inf/nan values that might have appeared
    df_diff = df_diff.replace([np.inf, -np.inf], np.nan)
    
    # For columns with all NaN or constant values, fill with zeros
    for col in df_diff.columns:
        if df_diff[col].isna().all() or df_diff[col].std() == 0:
            df_diff[col] = 0.0
        else:
            # Fill any remaining NaN with column mean
            df_diff[col] = df_diff[col].fillna(df_diff[col].mean())
    
    # Standardize the data to prevent numerical issues in optimization
    # This is critical for datasets with thousands of variables
    scaler = StandardScaler()
    df_diff_values = scaler.fit_transform(df_diff)
    df_diff = pd.DataFrame(df_diff_values, columns=df_diff.columns)

    logger.info(f"Differenced data shape: {df_diff.shape}")
    logger.info(f"Data range after standardization: [{df_diff.min().min():.3f}, {df_diff.max().max():.3f}]")
    logger.info(f"Data mean: {df_diff.mean().mean():.6f}, std: {df_diff.std().mean():.6f}")

    # Save differenced data
    diff_file = os.path.join(preproc_dir, f"{basename}_differenced_stationary_series.csv")
    df_diff.to_csv(diff_file)
    logger.info(f"Saved differenced data to: {diff_file}")

    # Step 3: Find optimal lags (or use provided lags)
    logger.info("\nStep 3: Optimal lags determination")
    logger.info("-"*40)
    
    if provided_lags_file and os.path.exists(provided_lags_file):
        # Use the provided lags file
        logger.info(f"Loading existing lags from: {provided_lags_file}")
        lags_df = pd.read_csv(provided_lags_file)
        logger.info(f"Loaded {len(lags_df)} lag values:")
        for _, row in lags_df.iterrows():
            logger.info(f"  {row['variable']}: optimal lag = {row['optimal_lag']}")
        
        # Copy the lags file to the result directory for consistency
        lags_file = os.path.join(preproc_dir, f"{basename}_optimal_lags.csv")
        lags_df.to_csv(lags_file, index=False)
        logger.info(f"Copied lags to: {lags_file}")
    else:
        # Calculate new lags
        logger.info(f"Running lag optimization on {len(df_raw.columns)} columns using {cpu_count()} CPU cores...")
        logger.info("This may take 1-3 minutes for large datasets...")

        optimal_lags = []

        with Pool(cpu_count()) as pool:
            lag_results = pool.starmap(
                find_optimal_lag,
                [(df_raw[col], 20, 10, 0.05) for col in df_raw.columns]
            )

        logger.info(f"Lag optimization completed!")

        for col, lag in zip(df_raw.columns, lag_results):
            logger.info(f"  {col}: optimal lag = {lag}")
            optimal_lags.append({
                'variable': col,
                'optimal_lag': lag
            })

        # Save optimal lags
        lags_df = pd.DataFrame(optimal_lags)
        lags_file = os.path.join(preproc_dir, f"{basename}_optimal_lags.csv")
        lags_df.to_csv(lags_file, index=False)
        logger.info(f"Saved optimal lags to: {lags_file}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Output files in: {preproc_dir}")
    logger.info(f"  - Differenced data: {os.path.basename(diff_file)}")
    logger.info(f"  - Optimal lags: {os.path.basename(lags_file)}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
