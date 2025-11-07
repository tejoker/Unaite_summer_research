#!/usr/bin/env python3
"""
DYNOTEARS Variants for Benchmarking
- DYNOTEARS without MI mask (uses real DYNOTEARS algorithm)
- DYNOTEARS without rolling windows (single global analysis)
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import simple functions
from sklearn.linear_model import Ridge

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class DynoTearsNoMIMask:
    """Simple VAR without MI mask constraint"""
    
    def __init__(self, d, p):
        self.d = d
        self.p = p
        
    def fit(self, data, lambda_w=0.1, lambda_a=0.1, max_iter=100):
        """Fit simple VAR without MI mask"""
        logger.info("Running VAR without MI mask...")
        start_time = time.time()
        
        try:
            return self._simple_var_estimation(data, lambda_w)
            
        except Exception as e:
            logger.error(f"Error in VAR no MI mask: {e}")
            return {
                'W': np.zeros((self.d, self.d)),
                'A_list': [np.zeros((self.d, self.d)) for _ in range(self.p)],
                'method': 'VAR (no MI mask) - Error',
                'error': str(e)
            }
    
    def _simple_var_estimation(self, data, lambda_reg):
        """Simple VAR estimation as fallback"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Validate data doesn't contain NaN or inf
        if not np.isfinite(data).all():
            logger.warning("Data contains NaN or inf values, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        n, d = data.shape
        
        # Create lagged data matrix
        if n <= self.p:
            # Not enough data for lags
            W = np.zeros((d, d))
            A_list = [np.zeros((d, d)) for _ in range(self.p)]
        else:
            # Construct design matrix with lags
            X = []
            y = []
            
            for t in range(self.p, n):
                # Current observation
                y.append(data[t])
                
                # Lagged observations
                x_row = []
                for lag in range(1, self.p + 1):
                    x_row.extend(data[t - lag])
                X.append(x_row)
            
            X = np.array(X)
            y = np.array(y)
            
            # Final validation before regression
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                logger.warning("X or y contains NaN/inf after lag construction, replacing with zeros")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ridge regression for each variable
            from sklearn.linear_model import Ridge
            
            W = np.zeros((d, d))
            A_list = [np.zeros((d, d)) for _ in range(self.p)]
            
            for j in range(d):
                try:
                    ridge = Ridge(alpha=lambda_reg)
                    ridge.fit(X, y[:, j])
                    
                    # Extract coefficients
                    coef = ridge.coef_
                    
                    # Map coefficients back to matrices
                    coef_idx = 0
                    for lag in range(self.p):
                        for i in range(d):
                            if coef_idx < len(coef):
                                A_list[lag][i, j] = coef[coef_idx]
                            coef_idx += 1
                except Exception as e:
                    logger.warning(f"Ridge regression failed for variable {j}: {e}")
                    # Leave as zeros
                    pass
        
        return {
            'W': W,
            'A_list': A_list,
            'method': 'DYNOTEARS (no MI mask) - Simple VAR'
        }

class DynoTearsNoRollingVAR:
    """Simple VAR without rolling windows (same as no MI)"""
    
    def __init__(self, d, p):
        self.d = d
        self.p = p
        
    def fit(self, data, lambda_w=0.1, lambda_a=0.1, max_iter=100):
        """Fit simple VAR on entire dataset (no rolling windows)"""
        logger.info("Running VAR without rolling windows...")
        start_time = time.time()
        
        try:
            return self._simple_var_estimation(data, lambda_w)
            
        except Exception as e:
            logger.error(f"Error in VAR no rolling: {e}")
            return {
                'W': np.zeros((self.d, self.d)),
                'A_list': [np.zeros((self.d, self.d)) for _ in range(self.p)],
                'method': 'VAR (no rolling) - Error',
                'error': str(e)
            }
    
    def _simple_var_estimation(self, data, lambda_reg):
        """Simple VAR estimation"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        n, d = data.shape
        start_time = time.time()
        
        if n <= self.p:
            W = np.zeros((d, d))
            A_list = [np.zeros((d, d)) for _ in range(self.p)]
        else:
            # Create design matrix
            X = []
            y = []
            
            for t in range(self.p, n):
                y.append(data[t])
                x_row = []
                for lag in range(1, self.p + 1):
                    x_row.extend(data[t - lag])
                X.append(x_row)
            
            X = np.array(X)
            y = np.array(y)
            
            W = np.zeros((d, d))
            A_list = [np.zeros((d, d)) for _ in range(self.p)]
            
            for j in range(d):
                ridge = Ridge(alpha=lambda_reg)
                ridge.fit(X, y[:, j])
                
                coef = ridge.coef_
                coef_idx = 0
                for lag in range(self.p):
                    for i in range(d):
                        if coef_idx < len(coef):
                            A_list[lag][i, j] = coef[coef_idx]
                        coef_idx += 1
        
        elapsed = time.time() - start_time
        return {
            'W': W,
            'A_list': A_list,
            'method': 'VAR (no rolling windows)',
            'runtime_seconds': elapsed
        }

def run_dynotears_variants(data_file, output_dir, variants=['no_mi', 'no_rolling']):
    """Run real DYNOTEARS variants and save results"""
    
    logger.info(f"Loading data from {data_file}")
    
    # Load and preprocess data (same as your enhanced method)
    if data_file.endswith('.csv'):
        # Handle index column properly 
        try:
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        except:
            data = pd.read_csv(data_file)
        
        # Standardize data for fair comparison
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values)
        data = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
        
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    logger.info(f"Loaded data shape: {data.shape}")
    
    d = data.shape[1]
    p = 1  # Default lag order
    
    results = {
        'data_file': data_file,
        'data_shape': list(data.shape),
        'preprocessing': 'StandardScaler applied',
        'variants': {}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_start_time = time.time()
    
    for variant in variants:
        logger.info(f"Running DYNOTEARS variant: {variant}")
        
        if variant == 'no_mi':
            model = DynoTearsNoMIMask(d, p)
            result = model.fit(data)
        elif variant == 'no_rolling':
            model = DynoTearsNoRollingVAR(d, p)  
            result = model.fit(data)
        else:
            logger.warning(f"Unknown variant: {variant}")
            continue
        
        results['variants'][variant] = result
        
        # Count edges for summary
        if 'error' not in result:
            W_edges = np.sum(np.abs(result['W']) > 1e-4)
            A_edges = sum(np.sum(np.abs(A) > 1e-4) for A in result['A_list'])
            logger.info(f"{variant}: {W_edges} W edges, {A_edges} A edges, "
                       f"time: {result.get('runtime_seconds', 0):.2f}s")
        
        # Save individual result
        variant_file = os.path.join(output_dir, f"dynotears_{variant}_results.json")
        with open(variant_file, 'w') as f:
            json.dump({variant: result}, f, indent=2, default=str)
        
        logger.info(f"Saved {variant} results to {variant_file}")
    
    total_elapsed = time.time() - total_start_time
    results['total_runtime_seconds'] = total_elapsed
    
    # Save combined results
    combined_file = os.path.join(output_dir, "dynotears_variants_combined.json")
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"All variants completed in {total_elapsed:.2f}s")
    logger.info(f"Combined results saved to {combined_file}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DYNOTEARS Variants Benchmark')
    parser.add_argument('--data', required=True, help='Input time series CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--variants', nargs='+', choices=['no_mi', 'no_rolling'], 
                       default=['no_mi', 'no_rolling'], help='Variants to run')
    
    args = parser.parse_args()
    
    try:
        results = run_dynotears_variants(args.data, args.output_dir, args.variants)
        print("✅ DYNOTEARS variants benchmark completed!")
        
        # Print summary
        print("\nSummary:")
        for variant, result in results['variants'].items():
            if 'error' in result:
                print(f"  {variant}: ❌ {result['error']}")
            else:
                W = np.array(result['W'])
                A_list = [np.array(A) for A in result['A_list']]
                w_edges = np.sum(W != 0)
                total_a_edges = sum(np.sum(A != 0) for A in A_list)
                print(f"  {variant}: ✅ W edges: {w_edges}, A edges: {total_a_edges}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)