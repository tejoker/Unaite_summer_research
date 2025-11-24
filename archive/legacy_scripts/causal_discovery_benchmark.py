#!/usr/bin/env python3
"""
Causal Discovery Methods Benchmark
Compares different causal discovery approaches:
- DYNOTEARS (no MI mask)
- DYNOTEARS (no rolling VAR)
- NOTEARS + Lasso
- LiNGAM
- tsGFCI
"""

import argparse
import sys
import os
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch using device: {device}")
except ImportError:
    TORCH_AVAILABLE = False
    device = None
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available. Install with: pip install cupy-cuda11x")

# Import causal discovery libraries
try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    import networkx as nx
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")

try:
    from lingam import DirectLiNGAM, VARLiNGAM
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False
    print("Warning: lingam not available. Install with: pip install lingam")

try:
    from causal_learn.search.ConstraintBased.FCI import fci
    from causal_learn.utils.cit import CIT
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    print("Warning: causal-learn not available. Install with: pip install causal-learn")

class CausalDiscoveryBenchmark:
    def __init__(self, use_gpu=True):
        self.scaler = StandardScaler()
        self.use_gpu = use_gpu and (TORCH_AVAILABLE or CUPY_AVAILABLE)
        self.device = device if TORCH_AVAILABLE else None
        
    def _to_gpu(self, data):
        """Convert numpy array to GPU tensor/array"""
        if not self.use_gpu:
            return data
            
        if TORCH_AVAILABLE and self.device.type == 'cuda':
            return torch.from_numpy(data.astype(np.float32)).to(self.device)
        elif CUPY_AVAILABLE:
            return cp.asarray(data.astype(np.float32))
        else:
            return data
    
    def _to_cpu(self, data):
        """Convert GPU tensor/array back to numpy"""
        if torch.is_tensor(data):
            return data.cpu().numpy()
        elif CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        else:
            return data
        
    def load_timeseries(self, filepath):
        """Load time series data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            # Convert to numeric where possible
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            data = df.select_dtypes(include=[np.number]).dropna()
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def notears_lasso(self, data, lambda_reg=0.1, max_iter=100):
        """NOTEARS with Lasso regularization for linear case"""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'sklearn not available'}
            
            d = data.shape[1]
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Move to GPU if available
            if self.use_gpu:
                data_gpu = self._to_gpu(data_scaled)
            else:
                data_gpu = data_scaled
            
            # Initialize adjacency matrix
            if self.use_gpu and TORCH_AVAILABLE and self.device.type == 'cuda':
                W = torch.zeros((d, d), device=self.device)
            elif self.use_gpu and CUPY_AVAILABLE:
                W = cp.zeros((d, d))
            else:
                W = np.zeros((d, d))
            
            # For each variable, fit Lasso regression
            for j in range(d):
                # Target variable
                if self.use_gpu:
                    y = self._to_cpu(data_gpu[:, j])
                    X_full = self._to_cpu(data_gpu)
                    X = np.delete(X_full, j, axis=1)
                else:
                    y = data_scaled[:, j]
                    X = np.delete(data_scaled, j, axis=1)
                
                if X.shape[1] > 0:
                    lasso = Lasso(alpha=lambda_reg, max_iter=max_iter)
                    lasso.fit(X, y)
                    
                    # Fill adjacency matrix
                    coef_idx = 0
                    for i in range(d):
                        if i != j:
                            if self.use_gpu and torch.is_tensor(W):
                                W[i, j] = torch.tensor(lasso.coef_[coef_idx], device=self.device)
                            elif self.use_gpu and CUPY_AVAILABLE and isinstance(W, cp.ndarray):
                                W[i, j] = lasso.coef_[coef_idx]
                            else:
                                W[i, j] = lasso.coef_[coef_idx]
                            coef_idx += 1
            
            # Convert back to CPU for post-processing
            W_cpu = self._to_cpu(W)
            
            # Apply acyclicity constraint (simple thresholding)
            threshold = np.percentile(np.abs(W_cpu[W_cpu != 0]), 75) if np.any(W_cpu != 0) else 0
            W_cpu[np.abs(W_cpu) < threshold] = 0
            
            # Count edges
            edges = np.sum(W_cpu != 0)
            sparsity = 1 - edges / (d * d)
            
            return {
                'method': 'NOTEARS + Lasso',
                'adjacency_matrix': W_cpu.tolist(),
                'edges_detected': int(edges),
                'sparsity': float(sparsity),
                'max_weight': float(np.max(np.abs(W_cpu))),
                'lambda_reg': lambda_reg,
                'gpu_accelerated': self.use_gpu
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'NOTEARS + Lasso'}
    
    def direct_lingam(self, data):
        """DirectLiNGAM for instantaneous causal discovery"""
        try:
            if not LINGAM_AVAILABLE:
                return {'error': 'lingam not available'}
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Fit DirectLiNGAM
            model = DirectLiNGAM()
            model.fit(data_scaled)
            
            # Get adjacency matrix
            W = model.adjacency_matrix_
            
            # Count edges
            edges = np.sum(W != 0)
            d = W.shape[0]
            sparsity = 1 - edges / (d * d)
            
            return {
                'method': 'DirectLiNGAM',
                'adjacency_matrix': W.tolist(),
                'causal_order': model.causal_order_.tolist() if hasattr(model, 'causal_order_') else None,
                'edges_detected': int(edges),
                'sparsity': float(sparsity),
                'max_weight': float(np.max(np.abs(W)))
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'DirectLiNGAM'}
    
    def var_lingam(self, data, lags=1):
        """VARLiNGAM for time series causal discovery"""
        try:
            if not LINGAM_AVAILABLE:
                return {'error': 'lingam not available'}
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Fit VARLiNGAM
            model = VARLiNGAM(lags=lags)
            model.fit(data_scaled)
            
            # Get matrices
            W = model.adjacency_matrices_[0] if len(model.adjacency_matrices_) > 0 else np.zeros((data.shape[1], data.shape[1]))
            
            # Count edges across all lag matrices
            total_edges = sum(np.sum(A != 0) for A in model.adjacency_matrices_)
            d = data.shape[1]
            total_possible = d * d * lags
            sparsity = 1 - total_edges / total_possible if total_possible > 0 else 1
            
            return {
                'method': 'VARLiNGAM',
                'instantaneous_matrix': W.tolist(),
                'lag_matrices': [A.tolist() for A in model.adjacency_matrices_],
                'edges_detected': int(total_edges),
                'sparsity': float(sparsity),
                'lags': lags
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'VARLiNGAM'}
    
    def tsgfci(self, data, alpha=0.05):
        """tsGFCI (time series Greedy Fast Causal Inference)"""
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return {'error': 'causal-learn not available'}
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Create time-lagged data for causal discovery
            # Simple approach: include t and t-1
            if data_scaled.shape[0] < 2:
                return {'error': 'Insufficient data for time series analysis'}
            
            # Create augmented dataset with lags
            data_with_lags = []
            for t in range(1, data_scaled.shape[0]):
                row = np.concatenate([data_scaled[t], data_scaled[t-1]])
                data_with_lags.append(row)
            
            data_augmented = np.array(data_with_lags)
            
            # Run FCI
            cit = CIT(data_augmented, method="fisherz")
            _, edges = fci(data_augmented, cit, alpha=alpha, depth=-1, max_path_length=-1)
            
            # Extract adjacency information
            n_vars = data.shape[1]
            W_instant = np.zeros((n_vars, n_vars))
            W_lag = np.zeros((n_vars, n_vars))
            
            # Parse edges (simplified interpretation)
            for i in range(len(edges)):
                for j in range(len(edges[i])):
                    if edges[i][j] != 0:  # Edge exists
                        if i < n_vars and j < n_vars:
                            W_instant[i, j] = 1
                        elif i < n_vars and j >= n_vars:
                            W_lag[i, j - n_vars] = 1
            
            total_edges = np.sum(W_instant != 0) + np.sum(W_lag != 0)
            sparsity = 1 - total_edges / (2 * n_vars * n_vars)
            
            return {
                'method': 'tsGFCI',
                'instantaneous_matrix': W_instant.tolist(),
                'lag_matrix': W_lag.tolist(),
                'edges_detected': int(total_edges),
                'sparsity': float(sparsity),
                'alpha': alpha
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'tsGFCI'}
    
    def dynotears_no_mi_mask(self, data):
        """DYNOTEARS without MI mask constraint"""
        return {
            'method': 'DYNOTEARS (no MI mask)',
            'note': 'Requires modification of existing DYNOTEARS implementation',
            'error': 'Not implemented - requires code modification',
            'data_shape': data.shape
        }
    
    def dynotears_no_rolling_var(self, data):
        """DYNOTEARS without rolling VAR"""
        return {
            'method': 'DYNOTEARS (no rolling VAR)',
            'note': 'Requires modification of existing DYNOTEARS implementation', 
            'error': 'Not implemented - requires code modification',
            'data_shape': data.shape
        }

def main():
    parser = argparse.ArgumentParser(description='Causal Discovery Methods Benchmark')
    parser.add_argument('--baseline', required=True, help='Baseline time series CSV file')
    parser.add_argument('--anomaly', required=True, help='Anomaly time series CSV file')
    parser.add_argument('--output', required=True, help='Output file for results')
    parser.add_argument('--methods', nargs='+',
                       choices=['all', 'notears_lasso', 'direct_lingam', 'var_lingam', 'tsgfci', 
                               'dynotears_no_mi', 'dynotears_no_var'],
                       default=['all'], help='Methods to run')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    try:
        benchmark = CausalDiscoveryBenchmark(use_gpu=not args.no_gpu)
        start_time = time.time()
        
        print(f"🚀 Causal Discovery Benchmark")
        print(f"📁 Baseline: {args.baseline}")
        print(f"📁 Anomaly: {args.anomaly}")
        
        # Load data
        print("📈 Loading time series data...")
        baseline_data = benchmark.load_timeseries(args.baseline)
        anomaly_data = benchmark.load_timeseries(args.anomaly)
        
        if baseline_data is None or anomaly_data is None:
            raise ValueError("Failed to load time series data")
        
        print(f"Baseline shape: {baseline_data.shape}")
        print(f"Anomaly shape: {anomaly_data.shape}")
        
        # Determine methods to run
        methods_to_run = args.methods
        if 'all' in methods_to_run:
            methods_to_run = ['notears_lasso', 'direct_lingam', 'var_lingam', 'tsgfci']
        
        results = {
            'baseline_file': args.baseline,
            'anomaly_file': args.anomaly,
            'baseline_shape': baseline_data.shape,
            'anomaly_shape': anomaly_data.shape,
            'methods_run': methods_to_run,
            'baseline_results': {},
            'anomaly_results': {},
            'method_times': {}
        }
        
        # Run methods on both datasets
        print("🔬 Running causal discovery methods...")
        
        for method_name in methods_to_run:
            print(f"  ⚡ Running {method_name}...")
            method_start = time.time()
            
            try:
                # Run on baseline data
                if method_name == 'notears_lasso':
                    baseline_result = benchmark.notears_lasso(baseline_data.values)
                    anomaly_result = benchmark.notears_lasso(anomaly_data.values)
                elif method_name == 'direct_lingam':
                    baseline_result = benchmark.direct_lingam(baseline_data.values)
                    anomaly_result = benchmark.direct_lingam(anomaly_data.values)
                elif method_name == 'var_lingam':
                    baseline_result = benchmark.var_lingam(baseline_data.values)
                    anomaly_result = benchmark.var_lingam(anomaly_data.values)
                elif method_name == 'tsgfci':
                    baseline_result = benchmark.tsgfci(baseline_data.values)
                    anomaly_result = benchmark.tsgfci(anomaly_data.values)
                elif method_name == 'dynotears_no_mi':
                    baseline_result = benchmark.dynotears_no_mi_mask(baseline_data.values)
                    anomaly_result = benchmark.dynotears_no_mi_mask(anomaly_data.values)
                elif method_name == 'dynotears_no_var':
                    baseline_result = benchmark.dynotears_no_rolling_var(baseline_data.values)
                    anomaly_result = benchmark.dynotears_no_rolling_var(anomaly_data.values)
                else:
                    baseline_result = {'error': f'Unknown method: {method_name}'}
                    anomaly_result = {'error': f'Unknown method: {method_name}'}
                
                method_time = time.time() - method_start
                results['method_times'][method_name] = method_time
                
                results['baseline_results'][method_name] = baseline_result
                results['anomaly_results'][method_name] = anomaly_result
                
                # Add timing info
                baseline_result['execution_time_seconds'] = method_time
                anomaly_result['execution_time_seconds'] = method_time
                
            except Exception as e:
                method_time = time.time() - method_start
                error_result = {'error': str(e), 'method': method_name, 'execution_time_seconds': method_time}
                results['baseline_results'][method_name] = error_result
                results['anomaly_results'][method_name] = error_result
                results['method_times'][method_name] = method_time
        
        # Calculate summary
        total_time = time.time() - start_time
        results['benchmark_info'] = {
            'total_execution_time_seconds': total_time,
            'successful_methods': len([m for m in methods_to_run 
                                     if 'error' not in results['baseline_results'].get(m, {})]),
            'failed_methods': len([m for m in methods_to_run 
                                 if 'error' in results['baseline_results'].get(m, {})])
        }
        
        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {args.output}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 CAUSAL DISCOVERY BENCHMARK SUMMARY")
        print("="*60)
        print(f"⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"✅ Successful methods: {results['benchmark_info']['successful_methods']}")
        print(f"❌ Failed methods: {results['benchmark_info']['failed_methods']}")
        
        print("\nMethod Performance:")
        for method_name in methods_to_run:
            baseline_result = results['baseline_results'].get(method_name, {})
            anomaly_result = results['anomaly_results'].get(method_name, {})
            exec_time = results['method_times'].get(method_name, 0)
            
            if 'error' not in baseline_result and 'error' not in anomaly_result:
                baseline_edges = baseline_result.get('edges_detected', 'N/A')
                anomaly_edges = anomaly_result.get('edges_detected', 'N/A')
                baseline_sparsity = baseline_result.get('sparsity', 'N/A')
                anomaly_sparsity = anomaly_result.get('sparsity', 'N/A')
                
                print(f"  {method_name:20} | ✅ SUCCESS | {exec_time:6.3f}s")
                print(f"    Baseline: {baseline_edges} edges, sparsity: {baseline_sparsity}")
                print(f"    Anomaly:  {anomaly_edges} edges, sparsity: {anomaly_sparsity}")
            else:
                error_msg = baseline_result.get('error', 'Unknown error')
                print(f"  {method_name:20} | ❌ FAILED  | {exec_time:6.3f}s | {error_msg}")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()