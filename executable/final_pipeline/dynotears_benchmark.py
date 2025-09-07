#!/usr/bin/env python3
"""
Proper DYNOTEARS Benchmark Script

Compares:
1. Enhanced Method (your rolling windows + MI + temporal mapping)
2. Classic DYNOTEARS (proper acyclicity-constrained optimization)
3. Rolling VAR without MI (your method minus MI masking)  
4. Classic VAR with MI (traditional VAR + your MI enhancement)
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Import your existing components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynotears import from_pandas_dynamic, extract_matrices
from preprocessing import calculate_mutual_information_mask

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ProperDynoTears:
    """Proper DYNOTEARS implementation with acyclicity constraints"""
    
    def __init__(self, d, p=1, device='cpu'):
        self.d = d
        self.p = p
        self.device = device
        
    def _h_func(self, W):
        """Acyclicity constraint function"""
        d = W.shape[0]
        A = W * W  # Element-wise square
        h = torch.trace(torch.matrix_exp(A)) - d
        return h
    
    def fit(self, data, lambda_w=0.1, lambda_a=0.1, max_iter=100, h_tol=1e-6):
        """Fit proper DYNOTEARS with acyclicity constraints"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        n, d = data.shape
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Initialize matrices
        W = torch.randn(d, d, device=self.device, requires_grad=True) * 0.1
        A_list = [torch.randn(d, d, device=self.device, requires_grad=True) * 0.1 
                  for _ in range(self.p)]
        
        # Augmented Lagrangian parameters
        alpha = 0.0
        rho = 1.0
        
        optimizer = torch.optim.Adam([W] + A_list, lr=0.01)
        
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Reconstruction loss
            if n > self.p:
                X_current = data_tensor[self.p:]
                X_lagged_list = []
                for lag in range(1, self.p + 1):
                    X_lagged_list.append(data_tensor[self.p-lag:-lag])
                
                # Forward pass
                I = torch.eye(d, device=self.device)
                residual = X_current @ (I - W)
                
                # Add lagged terms
                for lag_idx, A_mat in enumerate(A_list):
                    residual -= X_lagged_list[lag_idx] @ A_mat
                
                mse_loss = torch.mean(residual ** 2)
            else:
                mse_loss = torch.tensor(0.0, device=self.device)
            
            # Regularization
            l1_W = torch.sum(torch.abs(W))
            l1_A = sum(torch.sum(torch.abs(A)) for A in A_list)
            
            # Acyclicity constraint
            h_val = self._h_func(W)
            
            # Total loss (Augmented Lagrangian)
            total_loss = mse_loss + lambda_w * l1_W + lambda_a * l1_A + alpha * h_val + 0.5 * rho * h_val ** 2
            
            total_loss.backward()
            optimizer.step()
            
            # Update dual variables
            if iteration % 10 == 0:
                h_current = self._h_func(W).item()
                logger.info(f"Iter {iteration}: Loss={total_loss.item():.6f}, h={h_current:.6f}")
                
                if abs(h_current) <= h_tol:
                    logger.info(f"Converged at iteration {iteration}")
                    break
                    
                if h_current > 0.25 * abs(alpha / rho):
                    rho *= 10
                alpha += rho * h_current
        
        # Return results
        with torch.no_grad():
            W_final = W.detach().cpu().numpy()
            A_final = [A.detach().cpu().numpy() for A in A_list]
            
        return {
            'W': W_final,
            'A_list': A_final,
            'method': 'Proper DYNOTEARS with acyclicity constraints',
            'h_final': self._h_func(W).item(),
            'converged': abs(self._h_func(W).item()) <= h_tol
        }

class RollingVARNoMI:
    """Rolling VAR without MI masking"""
    
    def __init__(self, d, p=1, window_size=50):
        self.d = d
        self.p = p
        self.window_size = window_size
        
    def fit(self, data, lambda_reg=0.1):
        """Fit rolling VAR without MI constraints"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        n, d = data.shape
        num_windows = n - self.window_size + 1
        
        all_W = []
        all_A = []
        
        for t in range(self.window_size, n):
            window_data = data[t-self.window_size:t]
            
            # Fit VAR on window
            W, A_list = self._fit_var_window(window_data, lambda_reg)
            all_W.append(W)
            all_A.append(A_list)
            
        return {
            'W_series': all_W,
            'A_series': all_A,
            'method': 'Rolling VAR without MI masking',
            'num_windows': len(all_W)
        }
    
    def _fit_var_window(self, window_data, lambda_reg):
        """Fit VAR model on single window"""
        n, d = window_data.shape
        
        if n <= self.p:
            return np.zeros((d, d)), [np.zeros((d, d)) for _ in range(self.p)]
        
        # Create design matrix
        X = []
        y = []
        
        for t in range(self.p, n):
            y.append(window_data[t])
            x_row = []
            for lag in range(1, self.p + 1):
                x_row.extend(window_data[t - lag])
            X.append(x_row)
        
        X = np.array(X)
        y = np.array(y)
        
        # Ridge regression for each variable
        W = np.zeros((d, d))  # No contemporaneous effects in VAR
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
        
        return W, A_list

class ClassicVARWithMI:
    """Classic VAR with MI-based edge filtering"""
    
    def __init__(self, d, p=1):
        self.d = d
        self.p = p
        
    def fit(self, data, lambda_reg=0.1, mi_threshold=0.1):
        """Fit VAR with MI filtering"""
        if isinstance(data, pd.DataFrame):
            data_df = data
            data = data.values
        else:
            data_df = pd.DataFrame(data)
            
        n, d = data.shape
        
        # Calculate MI mask
        logger.info("Calculating mutual information mask...")
        mi_mask = calculate_mutual_information_mask(data_df, max_lag=self.p, threshold=mi_threshold)
        
        # Fit standard VAR
        logger.info("Fitting VAR model...")
        W, A_list = self._fit_var(data, lambda_reg)
        
        # Apply MI mask
        logger.info("Applying MI mask...")
        if mi_mask.shape[2] > 0:
            W[~mi_mask[:, :, 0]] = 0.0
            for lag in range(self.p):
                if lag + 1 < mi_mask.shape[2]:
                    A_list[lag][~mi_mask[:, :, lag + 1]] = 0.0
        
        return {
            'W': W,
            'A_list': A_list,
            'method': 'Classic VAR with MI filtering',
            'mi_edges_allowed': np.sum(mi_mask),
            'mi_edges_total': mi_mask.size
        }
    
    def _fit_var(self, data, lambda_reg):
        """Standard VAR fitting"""
        n, d = data.shape
        
        if n <= self.p:
            return np.zeros((d, d)), [np.zeros((d, d)) for _ in range(self.p)]
        
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
        
        # Ridge regression
        W = np.zeros((d, d))  # No contemporaneous in standard VAR
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
        
        return W, A_list

def run_comprehensive_benchmark(data_file, output_dir, lag_order=1):
    """Run comprehensive benchmark comparing all methods"""
    
    # Load data
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    logger.info(f"Data shape: {df.shape}")
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)
    
    d = df.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        'data_info': {
            'file': data_file,
            'shape': df.shape,
            'variables': list(df.columns),
            'lag_order': lag_order
        },
        'methods': {}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Method 1: Enhanced Method (your current implementation)
    logger.info("Running Enhanced Method...")
    start_time = time.time()
    try:
        # Use your existing from_pandas_dynamic function
        sm_enhanced = from_pandas_dynamic(
            df_scaled,
            p=lag_order,
            lambda_w=0.1,
            lambda_a=0.1,
            max_iter=100
        )
        W_enh, A_enh = extract_matrices(sm_enhanced, list(df.columns), lag_order)
        
        results['methods']['enhanced'] = {
            'W_edges': int(np.sum(W_enh.cpu().numpy() != 0)),
            'A_edges': int(np.sum(A_enh.cpu().numpy() != 0)),
            'method': 'Enhanced DYNOTEARS (rolling + MI + temporal)',
            'time_seconds': time.time() - start_time,
            'status': 'success'
        }
        logger.info(f"Enhanced method completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        results['methods']['enhanced'] = {'status': 'error', 'error': str(e)}
        logger.error(f"Enhanced method failed: {e}")
    
    # Method 2: Proper DYNOTEARS
    logger.info("Running Proper DYNOTEARS...")
    start_time = time.time()
    try:
        proper_dyno = ProperDynoTears(d, lag_order, device)
        result_proper = proper_dyno.fit(df_scaled.values)
        
        results['methods']['proper_dynotears'] = {
            'W_edges': int(np.sum(result_proper['W'] != 0)),
            'A_edges': sum(int(np.sum(A != 0)) for A in result_proper['A_list']),
            'method': result_proper['method'],
            'h_final': float(result_proper['h_final']),
            'converged': result_proper['converged'],
            'time_seconds': time.time() - start_time,
            'status': 'success'
        }
        logger.info(f"Proper DYNOTEARS completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        results['methods']['proper_dynotears'] = {'status': 'error', 'error': str(e)}
        logger.error(f"Proper DYNOTEARS failed: {e}")
    
    # Method 3: Rolling VAR without MI
    logger.info("Running Rolling VAR without MI...")
    start_time = time.time()
    try:
        rolling_var = RollingVARNoMI(d, lag_order, window_size=50)
        result_rolling = rolling_var.fit(df_scaled.values)
        
        # Aggregate statistics
        avg_W_edges = np.mean([np.sum(W != 0) for W in result_rolling['W_series']])
        avg_A_edges = np.mean([sum(np.sum(A != 0) for A in A_list) for A_list in result_rolling['A_series']])
        
        results['methods']['rolling_var_no_mi'] = {
            'avg_W_edges': float(avg_W_edges),
            'avg_A_edges': float(avg_A_edges),
            'method': result_rolling['method'],
            'num_windows': result_rolling['num_windows'],
            'time_seconds': time.time() - start_time,
            'status': 'success'
        }
        logger.info(f"Rolling VAR without MI completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        results['methods']['rolling_var_no_mi'] = {'status': 'error', 'error': str(e)}
        logger.error(f"Rolling VAR without MI failed: {e}")
    
    # Method 4: Classic VAR with MI
    logger.info("Running Classic VAR with MI...")
    start_time = time.time()
    try:
        classic_mi = ClassicVARWithMI(d, lag_order)
        result_classic = classic_mi.fit(df_scaled)
        
        results['methods']['classic_var_mi'] = {
            'W_edges': int(np.sum(result_classic['W'] != 0)),
            'A_edges': sum(int(np.sum(A != 0)) for A in result_classic['A_list']),
            'method': result_classic['method'],
            'mi_edges_allowed': int(result_classic['mi_edges_allowed']),
            'mi_edges_total': int(result_classic['mi_edges_total']),
            'time_seconds': time.time() - start_time,
            'status': 'success'
        }
        logger.info(f"Classic VAR with MI completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        results['methods']['classic_var_mi'] = {'status': 'error', 'error': str(e)}
        logger.error(f"Classic VAR with MI failed: {e}")
    
    # Save results
    output_file = os.path.join(output_dir, "comprehensive_benchmark.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE BENCHMARK RESULTS")
    logger.info("="*80)
    
    for method_name, method_result in results['methods'].items():
        if method_result['status'] == 'success':
            if 'W_edges' in method_result:
                logger.info(f"{method_name.upper()}:")
                logger.info(f"  W edges: {method_result.get('W_edges', 'N/A')}")
                logger.info(f"  A edges: {method_result.get('A_edges', 'N/A')}")
                logger.info(f"  Time: {method_result['time_seconds']:.2f}s")
            else:
                logger.info(f"{method_name.upper()}:")
                logger.info(f"  Avg W edges: {method_result.get('avg_W_edges', 'N/A'):.1f}")
                logger.info(f"  Avg A edges: {method_result.get('avg_A_edges', 'N/A'):.1f}")
                logger.info(f"  Time: {method_result['time_seconds']:.2f}s")
        else:
            logger.error(f"{method_name.upper()}: FAILED - {method_result['error']}")
        logger.info("-" * 40)
    
    logger.info(f"Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive DYNOTEARS Benchmark')
    parser.add_argument('--data', required=True, help='Input CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--lag', type=int, default=1, help='Lag order')
    
    args = parser.parse_args()
    
    try:
        results = run_comprehensive_benchmark(args.data, args.output_dir, args.lag)
        print("✅ Comprehensive benchmark completed successfully!")
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        sys.exit(1)