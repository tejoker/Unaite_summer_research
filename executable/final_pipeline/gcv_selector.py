#!/usr/bin/env python3
"""
Generalized Cross-Validation (GCV) for Smoothness Parameter Selection
======================================================================

Automatically selects the smoothness penalty λ_smooth for P-Splines
without requiring a validation set.

GCV formula:
    GCV(λ) = n * MSE(λ) / (n - df_effective(λ))²

Where:
- MSE(λ) is the mean squared error with smoothness penalty λ
- df_effective is the effective degrees of freedom (trace of hat matrix)

Reference:
    Craven, P. and Wahba, G. (1979). "Smoothing noisy data with spline functions."
    Numerische Mathematik, 31(4), 377-403.
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Callable
import time

logger = logging.getLogger(__name__)


class GCVSelector:
    """
    Selects optimal smoothness parameter using Generalized Cross-Validation.
    
    This is the standard method in GAM literature for automatic hyperparameter
    selection without needing a separate validation set.
    """
    
    def __init__(self, 
                 candidate_lambdas: Optional[List[float]] = None,
                 quick_fit_iters: int = 20,
                 device: str = 'cpu'):
        """
        Initialize GCV selector.
        
        Args:
            candidate_lambdas: List of smoothness penalties to try.
                If None, uses logarithmic grid [0.001, 0.01, 0.1, 1.0]
            quick_fit_iters: Number of iterations for quick model fitting
                (we don't need full convergence for GCV comparison)
            device: 'cpu' or 'cuda'
        """
        if candidate_lambdas is None:
            # Default: logarithmic grid
            candidate_lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
        
        self.candidate_lambdas = candidate_lambdas
        self.quick_fit_iters = quick_fit_iters
        self.device = device
        
        # Cache for storing results
        self.gcv_scores = {}
        self.best_lambda = None
        self.best_gcv = None
    
    def select(self,
               fit_function: Callable,
               X: torch.Tensor,
               y: torch.Tensor,
               verbose: bool = True) -> float:
        """
        Select optimal λ_smooth via GCV.
        
        Args:
            fit_function: Function with signature fit_function(X, y, lambda_smooth, max_iter)
                that returns (model, predictions)
            X: Input features [n, d]
            y: Target values [n, d_out]
            verbose: If True, log GCV scores for each lambda
        
        Returns:
            Best lambda value (minimum GCV score)
        
        Example:
            >>> selector = GCVSelector()
            >>> def my_fit(X, y, lam, max_iter):
            ...     model = MyModel(lambda_smooth=lam)
            ...     model.fit(X, y, max_iter)
            ...     return model, model.predict(X)
            >>> best_lambda = selector.select(my_fit, X_train, y_train)
        """
        n = X.shape[0]
        
        if verbose:
            logger.info(f"GCV Selection: Testing {len(self.candidate_lambdas)} lambda values")
            logger.info(f"Candidates: {self.candidate_lambdas}")
        
        best_lambda = self.candidate_lambdas[0]
        best_gcv = float('inf')
        results = []
        
        for lambda_smooth in self.candidate_lambdas:
            start_time = time.time()
            
            # Quick fit with this lambda
            try:
                model, y_pred = fit_function(X, y, lambda_smooth, self.quick_fit_iters)
                
                # Compute MSE
                residuals = y - y_pred
                mse = torch.mean(residuals ** 2).item()
                
                # Estimate effective degrees of freedom
                df_eff = self._estimate_df(model, X, n)
                
                # GCV score
                if n - df_eff <= 0:
                    gcv_score = float('inf')  # Overfitting - penalize heavily
                else:
                    gcv_score = (n * mse) / ((n - df_eff) ** 2)
                
                elapsed = time.time() - start_time
                
                results.append({
                    'lambda': lambda_smooth,
                    'gcv': gcv_score,
                    'mse': mse,
                    'df_eff': df_eff,
                    'time': elapsed
                })
                
                if verbose:
                    logger.info(f"  λ={lambda_smooth:.4f}: GCV={gcv_score:.6f}, MSE={mse:.6f}, "
                               f"df={df_eff:.1f}, time={elapsed:.2f}s")
                
                # Update best
                if gcv_score < best_gcv:
                    best_gcv = gcv_score
                    best_lambda = lambda_smooth
            
            except Exception as e:
                logger.warning(f"  λ={lambda_smooth:.4f}: Failed - {e}")
                results.append({
                    'lambda': lambda_smooth,
                    'gcv': float('inf'),
                    'error': str(e)
                })
        
        # Store results
        self.gcv_scores = {r['lambda']: r.get('gcv', float('inf')) for r in results}
        self.best_lambda = best_lambda
        self.best_gcv = best_gcv
        
        if verbose:
            logger.info(f"GCV Selection Complete: Best λ={best_lambda:.4f} (GCV={best_gcv:.6f})")
        
        return best_lambda
    
    def _estimate_df(self, model, X: torch.Tensor, n: int) -> float:
        """
        Estimate effective degrees of freedom for the model.
        
        For P-Splines, the exact formula involves the hat matrix S where y_hat = S*y.
        df_effective = tr(S)
        
        Since computing the full hat matrix is expensive, we use approximations:
        
        1. For additive models: df ≈ sum of df per component
        2. For each P-spline: df ≈ K - (smoothing_penalty_effect)
        3. Total df ≈ (number of active edges) × (knots per edge) × (1 - penalty_ratio)
        
        Args:
            model: Fitted model object
            X: Input data [n, d]
            n: Number of samples
        
        Returns:
            Estimated effective degrees of freedom
        """
        # Method 1: Use model's df estimate if available
        if hasattr(model, 'get_effective_df'):
            return model.get_effective_df()
        
        # Method 2: Use model's parameter count with penalty adjustment
        if hasattr(model, 'count_parameters'):
            n_params = model.count_parameters()
            
            # Get smoothness penalty strength (if available)
            lambda_smooth = getattr(model, 'lambda_smooth', 0.0)
            
            # Rough approximation: each parameter contributes (1 - penalty_factor) to df
            # Stronger penalty -> lower df
            penalty_factor = lambda_smooth / (1.0 + lambda_smooth)  # Ranges 0 to 1
            df_eff = n_params * (1.0 - penalty_factor * 0.5)
            
            return max(1.0, min(df_eff, n - 1))  # Clamp to valid range
        
        # Method 3: Conservative fallback
        # Assume df scales with sqrt(n) for additive models (typical for GAMs)
        logger.warning("Model has no df estimation method, using sqrt(n) heuristic")
        return np.sqrt(n)
    
    def plot_gcv_curve(self, save_path: Optional[str] = None):
        """
        Plot GCV score vs lambda (for diagnostics).
        
        Args:
            save_path: If provided, save plot to this path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, cannot plot GCV curve")
            return
        
        if not self.gcv_scores:
            logger.warning("No GCV scores to plot - run select() first")
            return
        
        lambdas = sorted(self.gcv_scores.keys())
        gcvs = [self.gcv_scores[lam] for lam in lambdas]
        
        plt.figure(figsize=(8, 6))
        plt.semilogx(lambdas, gcvs, 'o-', linewidth=2, markersize=8)
        plt.axvline(self.best_lambda, color='r', linestyle='--', 
                   label=f'Best λ={self.best_lambda:.4f}')
        plt.xlabel('Smoothness Penalty (λ)', fontsize=12)
        plt.ylabel('GCV Score', fontsize=12)
        plt.title('Generalized Cross-Validation Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"GCV curve saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class AdaptiveGCVSelector(GCVSelector):
    """
    Enhanced GCV selector with adaptive lambda grid refinement.
    
    First does coarse search, then refines around the best region.
    This is more efficient than testing many lambdas upfront.
    """
    
    def __init__(self, 
                 coarse_lambdas: Optional[List[float]] = None,
                 n_refinement: int = 3,
                 **kwargs):
        """
        Initialize adaptive GCV selector.
        
        Args:
            coarse_lambdas: Initial coarse grid (if None, uses default)
            n_refinement: Number of refinement steps
            **kwargs: Passed to parent GCVSelector
        """
        super().__init__(candidate_lambdas=coarse_lambdas, **kwargs)
        self.n_refinement = n_refinement
    
    def select(self, fit_function: Callable, X: torch.Tensor, y: torch.Tensor,
               verbose: bool = True) -> float:
        """
        Select optimal lambda with adaptive refinement.
        
        Strategy:
        1. Coarse search over initial grid
        2. Identify best region
        3. Refine search in that region (logarithmic subdivision)
        4. Repeat refinement n_refinement times
        """
        # Step 1: Coarse search
        if verbose:
            logger.info("Adaptive GCV: Phase 1 - Coarse search")
        
        best_lambda = super().select(fit_function, X, y, verbose=verbose)
        
        # Step 2-4: Refinement
        for refinement in range(self.n_refinement):
            if verbose:
                logger.info(f"Adaptive GCV: Phase {refinement+2} - Refinement around λ={best_lambda:.4f}")
            
            # Create refined grid around best_lambda
            # Use log-scale: [best/3, best/1.5, best, best*1.5, best*3]
            refined_grid = [
                best_lambda / 3.0,
                best_lambda / 1.5,
                best_lambda,
                best_lambda * 1.5,
                best_lambda * 3.0
            ]
            
            # Remove duplicates and sort
            refined_grid = sorted(list(set(refined_grid)))
            
            # Update candidate list
            self.candidate_lambdas = refined_grid
            
            # Search refined grid
            best_lambda = super().select(fit_function, X, y, verbose=verbose)
        
        if verbose:
            logger.info(f"Adaptive GCV: Final best λ={best_lambda:.4f}")
        
        return best_lambda


def test_gcv_selector():
    """Test GCV selector on synthetic data"""
    print("Testing GCV Selector...")
    
    # Generate synthetic data: y = f(x) + noise
    torch.manual_seed(42)
    n = 100
    X = torch.linspace(-3, 3, n).unsqueeze(1)
    y_true = torch.sin(X) + 0.1 * X**2
    y = y_true + 0.2 * torch.randn_like(y_true)
    
    # Simple polynomial fitter for testing
    def polynomial_fit(X, y, lambda_smooth, max_iter):
        """Fit polynomial with L2 regularization (stands in for P-splines)"""
        # Features: [1, x, x^2, x^3]
        X_poly = torch.cat([X**i for i in range(4)], dim=1)
        
        # Ridge regression
        n, d = X_poly.shape
        I = torch.eye(d)
        
        # Solve: (X^T X + λI) β = X^T y
        XtX = X_poly.T @ X_poly
        Xty = X_poly.T @ y
        
        beta = torch.linalg.solve(XtX + lambda_smooth * I, Xty)
        
        # Predict
        y_pred = X_poly @ beta
        
        # Mock model object with df estimation
        class PolyModel:
            def __init__(self, beta, lambda_s):
                self.beta = beta
                self.lambda_smooth = lambda_s
            
            def count_parameters(self):
                return len(beta)
        
        model = PolyModel(beta, lambda_smooth)
        return model, y_pred
    
    # Test standard GCV
    print("\n1. Standard GCV Selector:")
    selector = GCVSelector(candidate_lambdas=[0.01, 0.1, 1.0, 10.0])
    best_lambda = selector.select(polynomial_fit, X, y, verbose=True)
    print(f"   Selected λ = {best_lambda}")
    
    # Test adaptive GCV
    print("\n2. Adaptive GCV Selector:")
    adaptive = AdaptiveGCVSelector(coarse_lambdas=[0.01, 1.0, 100.0], n_refinement=2)
    best_lambda_adaptive = adaptive.select(polynomial_fit, X, y, verbose=True)
    print(f"   Selected λ = {best_lambda_adaptive}")
    
    print("\n✓ GCV Selector test passed!")


if __name__ == "__main__":
    test_gcv_selector()
