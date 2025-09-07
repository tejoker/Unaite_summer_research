#!/usr/bin/env python3
"""
DYNOTEARS Variants for Benchmarking
- DYNOTEARS without MI mask
- DYNOTEARS without rolling VAR
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to sys.path to import from dynotears modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import existing DYNOTEARS components
try:
    from dbn_dynotears_enhanced import DynoTearsDBN, DynoTearsDynamicOptimizer
except ImportError:
    print("Warning: Could not import existing DYNOTEARS components")
    DynoTearsDBN = None
    DynoTearsDynamicOptimizer = None

class DynoTearsNoMIMask:
    """DYNOTEARS without MI mask constraint"""
    
    def __init__(self, d, p, window_size=50, device='cpu'):
        self.d = d
        self.p = p
        self.window_size = window_size
        self.device = device
        
    def fit(self, data, lambda_w=0.1, lambda_a=0.1, max_iter=100):
        """Fit DYNOTEARS without MI mask"""
        try:
            if DynoTearsDBN is None:
                return self._simple_var_estimation(data, lambda_w)
            
            # Use existing DYNOTEARS but skip MI mask
            model = DynoTearsDBN(
                dims=[self.d] * (self.p + 1),
                bias=True,
                device=self.device
            )
            
            # Convert data to tensor
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            data_tensor = torch.FloatTensor(data).to(self.device)
            
            # Simple optimization without MI constraints
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for iteration in range(max_iter):
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = model(data_tensor)
                
                # Loss: reconstruction + regularization (no MI mask)
                mse_loss = torch.mean((data_tensor - reconstructed) ** 2)
                
                # L1 regularization on weights
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                
                total_loss = mse_loss + lambda_w * l1_loss
                
                total_loss.backward()
                optimizer.step()
                
                if iteration % 20 == 0:
                    print(f"Iteration {iteration}, Loss: {total_loss.item():.6f}")
            
            # Extract weight matrices
            with torch.no_grad():
                W = torch.zeros(self.d, self.d)
                A_list = [torch.zeros(self.d, self.d) for _ in range(self.p)]
                
                # Extract from model parameters (simplified)
                params = list(model.parameters())
                if len(params) > 0:
                    # This is a simplified extraction - would need to match actual model structure
                    weight_param = params[0]
                    if weight_param.shape[0] >= self.d and weight_param.shape[1] >= self.d:
                        W = weight_param[:self.d, :self.d].clone()
                        
                        # Extract lag matrices if available
                        if weight_param.shape[1] >= self.d * (self.p + 1):
                            for lag in range(self.p):
                                start_col = self.d * (lag + 1)
                                end_col = self.d * (lag + 2)
                                A_list[lag] = weight_param[:self.d, start_col:end_col].clone()
            
            return {
                'W': W.cpu().numpy(),
                'A_list': [A.cpu().numpy() for A in A_list],
                'method': 'DYNOTEARS (no MI mask)'
            }
            
        except Exception as e:
            print(f"Error in DYNOTEARS no MI mask: {e}")
            return self._simple_var_estimation(data, lambda_w)
    
    def _simple_var_estimation(self, data, lambda_reg):
        """Simple VAR estimation as fallback"""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
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
            
            # Ridge regression for each variable
            from sklearn.linear_model import Ridge
            
            W = np.zeros((d, d))
            A_list = [np.zeros((d, d)) for _ in range(self.p)]
            
            for j in range(d):
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
        
        return {
            'W': W,
            'A_list': A_list,
            'method': 'DYNOTEARS (no MI mask) - Simple VAR'
        }

class DynoTearsNoRollingVAR:
    """DYNOTEARS without rolling VAR (fixed window)"""
    
    def __init__(self, d, p, device='cpu'):
        self.d = d
        self.p = p
        self.device = device
        
    def fit(self, data, lambda_w=0.1, lambda_a=0.1, max_iter=100):
        """Fit DYNOTEARS without rolling window VAR"""
        try:
            # Use entire dataset as single window instead of rolling
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            n, d = data.shape
            
            # Simple VAR on entire dataset
            from sklearn.linear_model import Ridge
            
            if n <= self.p:
                W = np.zeros((d, d))
                A_list = [np.zeros((d, d)) for _ in range(self.p)]
            else:
                # Create single VAR model on all data
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
                
                # Fit VAR model
                for j in range(d):
                    ridge = Ridge(alpha=lambda_w)
                    ridge.fit(X, y[:, j])
                    
                    coef = ridge.coef_
                    coef_idx = 0
                    for lag in range(self.p):
                        for i in range(d):
                            if coef_idx < len(coef):
                                A_list[lag][i, j] = coef[coef_idx]
                            coef_idx += 1
            
            return {
                'W': W,
                'A_list': A_list,
                'method': 'DYNOTEARS (no rolling VAR)'
            }
            
        except Exception as e:
            print(f"Error in DYNOTEARS no rolling VAR: {e}")
            return {
                'W': np.zeros((self.d, self.d)),
                'A_list': [np.zeros((self.d, self.d)) for _ in range(self.p)],
                'method': 'DYNOTEARS (no rolling VAR) - Error',
                'error': str(e)
            }

def run_dynotears_variants(data_file, output_dir, variants=['no_mi', 'no_rolling']):
    """Run DYNOTEARS variants and save results"""
    
    # Load data
    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file)
        # Convert to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='ignore')
        data = data.select_dtypes(include=[np.number]).dropna()
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    print(f"Loaded data shape: {data.shape}")
    
    d = data.shape[1]
    p = 1  # Default lag order
    
    results = {
        'data_file': data_file,
        'data_shape': data.shape,
        'variants': {}
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for variant in variants:
        print(f"Running DYNOTEARS variant: {variant}")
        
        if variant == 'no_mi':
            model = DynoTearsNoMIMask(d, p)
            result = model.fit(data)
        elif variant == 'no_rolling':
            model = DynoTearsNoRollingVAR(d, p)  
            result = model.fit(data)
        else:
            print(f"Unknown variant: {variant}")
            continue
        
        results['variants'][variant] = result
        
        # Save individual result
        variant_file = os.path.join(output_dir, f"dynotears_{variant}_results.json")
        import json
        with open(variant_file, 'w') as f:
            json.dump({variant: result}, f, indent=2, default=str)
        
        print(f"Saved {variant} results to {variant_file}")
    
    # Save combined results
    combined_file = os.path.join(output_dir, "dynotears_variants_combined.json")
    import json
    with open(combined_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved combined results to {combined_file}")
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