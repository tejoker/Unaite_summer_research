#!/usr/bin/env python3
"""
Time Series Reconstruction from Learned Weights with MI Mask Constraints

This module implements reconstruction (simulation) of time series trajectories 
from learned DBN matrices with mutual information mask constraints applied.

Based on PROMPT C specifications:
- Uses instantaneous matrix W ∈ R^{d×d} (acyclic, zero diag)
- Uses lag matrices A^(1),...,A^(p) ∈ R^{d×d}
- Respects MI mask constraints that have already zeroed forbidden edges
- Supports both deterministic (ε=0) and stochastic reconstruction
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union, Tuple, List
from scipy.linalg import solve, LinAlgError
from sklearn.preprocessing import StandardScaler

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeSeriesReconstructor:
    """
    Reconstructs time series from learned DBN weight matrices.
    
    Handles both deterministic and stochastic reconstruction with proper
    acyclicity constraints and MI mask application.
    """
    
    def __init__(self, W: Union[torch.Tensor, np.ndarray], 
                 A_list: List[Union[torch.Tensor, np.ndarray]],
                 var_names: List[str],
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize the reconstructor with learned matrices.
        
        Args:
            W: Instantaneous weight matrix of shape [d, d], already MI-masked
            A_list: List of lag matrices [A^(1), A^(2), ..., A^(p)], each [d, d]
            var_names: List of variable names
            scaler: Optional StandardScaler used during training (for inverse transform)
        """
        self.var_names = var_names
        self.d = len(var_names)
        self.p = len(A_list)
        self.scaler = scaler
        
        # Convert to numpy arrays for computation
        self.W = self._to_numpy(W)
        self.A_list = [self._to_numpy(A) for A in A_list]
        
        # Validate matrix dimensions
        self._validate_matrices()
        
        # Compute S = I - W for matrix solve method
        self.S = np.eye(self.d) - self.W
        
        # Check invertibility of S (acyclicity condition)
        self._check_acyclicity()
        
        logger.info(f"Initialized reconstructor: {self.d} variables, lag order {self.p}")
        self._log_matrix_stats()
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor/array to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
    
    def _validate_matrices(self):
        """Validate matrix dimensions and properties."""
        if self.W.shape != (self.d, self.d):
            raise ValueError(f"W shape {self.W.shape} doesn't match expected ({self.d}, {self.d})")
        
        # Check diagonal is zero (acyclic constraint)
        if not np.allclose(np.diag(self.W), 0, atol=1e-8):
            logger.warning("W has non-zero diagonal elements - may violate acyclicity")
        
        for i, A in enumerate(self.A_list):
            if A.shape != (self.d, self.d):
                raise ValueError(f"A[{i+1}] shape {A.shape} doesn't match expected ({self.d}, {self.d})")
    
    def _check_acyclicity(self, reg_param: float = 1e-6):
        """Check if S = I - W is invertible (acyclicity condition)."""
        try:
            cond_num = np.linalg.cond(self.S)
            if cond_num > 1e12:
                logger.warning(f"S = I - W is ill-conditioned (cond={cond_num:.2e})")
                # Apply Tikhonov regularization
                self.S += reg_param * np.eye(self.d)
                logger.info(f"Applied Tikhonov regularization (λ={reg_param})")
            
            # Test inversion
            _ = np.linalg.inv(self.S)
            logger.info("Acyclicity check passed - S is invertible")
            
        except LinAlgError:
            logger.error("S = I - W is not invertible - acyclicity violated")
            # Prune small edges and retry
            self._prune_small_edges()
    
    def _prune_small_edges(self, threshold: float = 1e-8):
        """Prune edges smaller than threshold to ensure acyclicity."""
        logger.info(f"Pruning edges with |weight| < {threshold}")
        
        # Prune W
        mask = np.abs(self.W) >= threshold
        edges_before = np.sum(self.W != 0)
        self.W[~mask] = 0.0
        edges_after = np.sum(self.W != 0)
        logger.info(f"W: {edges_before} -> {edges_after} edges after pruning")
        
        # Prune A matrices
        for i, A in enumerate(self.A_list):
            mask = np.abs(A) >= threshold
            edges_before = np.sum(A != 0)
            A[~mask] = 0.0
            edges_after = np.sum(A != 0)
            logger.info(f"A[{i+1}]: {edges_before} -> {edges_after} edges after pruning")
        
        # Recompute S
        self.S = np.eye(self.d) - self.W
        
        # Retry inversion
        try:
            _ = np.linalg.inv(self.S)
            logger.info("Acyclicity achieved after pruning")
        except LinAlgError:
            raise ValueError("Cannot achieve acyclicity even after pruning small edges")
    
    def _log_matrix_stats(self):
        """Log statistics about the weight matrices."""
        w_edges = np.sum(self.W != 0)
        w_sparsity = 1 - w_edges / (self.d * self.d)
        logger.info(f"W: {w_edges}/{self.d*self.d} edges ({w_sparsity:.3f} sparsity)")
        
        for i, A in enumerate(self.A_list):
            a_edges = np.sum(A != 0)
            a_sparsity = 1 - a_edges / (self.d * self.d)
            logger.info(f"A[{i+1}]: {a_edges}/{self.d*self.d} edges ({a_sparsity:.3f} sparsity)")
    
    def reconstruct_deterministic(self, 
                                initial_conditions: np.ndarray,
                                T: int) -> pd.DataFrame:
        """
        Deterministically reconstruct time series (ε = 0).
        
        Args:
            initial_conditions: Initial values of shape [p, d] for x(0),...,x(p-1)
            T: Total number of time steps to simulate
            
        Returns:
            DataFrame with reconstructed time series
        """
        logger.info(f"Starting deterministic reconstruction for {T} time steps")
        return self._reconstruct(initial_conditions, T, stochastic=False)
    
    def reconstruct_stochastic(self, 
                             initial_conditions: np.ndarray,
                             T: int,
                             noise_std: float = 1.0) -> pd.DataFrame:
        """
        Stochastically reconstruct time series with noise.
        
        Args:
            initial_conditions: Initial values of shape [p, d] for x(0),...,x(p-1)
            T: Total number of time steps to simulate
            noise_std: Standard deviation of noise ε(t)
            
        Returns:
            DataFrame with reconstructed time series
        """
        logger.info(f"Starting stochastic reconstruction for {T} time steps (σ={noise_std})")
        return self._reconstruct(initial_conditions, T, stochastic=True, noise_std=noise_std)
    
    def predict_one_step(self, 
                        history: np.ndarray) -> np.ndarray:
        """
        Predict one step ahead using teacher forcing.
        
        Args:
            history: Historical values of shape [p, d] for x(t-p),...,x(t-1)
            
        Returns:
            Predicted values x̂(t) of shape [d]
        """
        if history.shape != (self.p, self.d):
            raise ValueError(f"History shape {history.shape} != expected ({self.p}, {self.d})")
        
        # Compute lagged contribution: h(t) = Σ A^(k) x(t-k)
        h = np.zeros(self.d)
        for k in range(self.p):
            h += self.A_list[k] @ history[self.p - 1 - k]  # history[0] = x(t-p), history[p-1] = x(t-1)
        
        # Solve instantaneous layer: x(t) = S^(-1) h(t)
        x_pred = solve(self.S, h)
        
        return x_pred
    
    def _reconstruct(self, 
                    initial_conditions: np.ndarray,
                    T: int,
                    stochastic: bool = False,
                    noise_std: float = 1.0) -> pd.DataFrame:
        """
        Core reconstruction logic.
        
        Args:
            initial_conditions: Initial values [p, d]
            T: Number of time steps
            stochastic: Whether to add noise
            noise_std: Noise standard deviation
            
        Returns:
            Reconstructed time series as DataFrame
        """
        if initial_conditions.shape != (self.p, self.d):
            raise ValueError(f"Initial conditions shape {initial_conditions.shape} != ({self.p}, {self.d})")
        
        # Initialize trajectory array
        x = np.zeros((T, self.d))
        
        # Set initial conditions
        if self.p > 0:
            x[:self.p] = initial_conditions
        
        # Pre-compute inverse of S for efficiency
        try:
            S_inv = np.linalg.inv(self.S)
        except LinAlgError:
            logger.error("Cannot invert S - using pseudo-inverse")
            S_inv = np.linalg.pinv(self.S)
        
        # Generate trajectory from t = p to T-1
        for t in range(self.p, T):
            # Compute lagged contribution: h(t) = Σ A^(k) x(t-k)
            h = np.zeros(self.d)
            for k in range(self.p):
                h += self.A_list[k] @ x[t - k - 1]
            
            # Add noise if stochastic
            if stochastic:
                epsilon = np.random.normal(0, noise_std, self.d)
                h += epsilon
            
            # Solve instantaneous layer: x(t) = S^(-1) (h(t) + ε(t))
            x[t] = S_inv @ h
            
            # Check for numerical instability
            if np.any(np.abs(x[t]) > 1e6):
                logger.warning(f"Large values detected at t={t}, simulation may be unstable")
                if np.any(np.isinf(x[t])) or np.any(np.isnan(x[t])):
                    logger.error(f"Infinite/NaN values at t={t}, stopping simulation")
                    x = x[:t]  # Truncate to valid part
                    break
        
        # Create DataFrame with proper time index
        time_index = pd.RangeIndex(len(x))
        df_result = pd.DataFrame(x, columns=self.var_names, index=time_index)
        
        logger.info(f"Reconstruction completed: {len(df_result)} time steps")
        
        return df_result
    
    def check_stability(self) -> Tuple[bool, float]:
        """
        Check stability of the system by examining the companion matrix.
        
        Returns:
            Tuple of (is_stable, spectral_radius)
        """
        if self.p == 0:
            # No lags, stability depends only on W
            eigenvals = np.linalg.eigvals(self.W)
            spectral_radius = np.max(np.abs(eigenvals))
            is_stable = spectral_radius < 1.0
        else:
            # Build companion matrix for VAR system
            # Convert to pure VAR by assuming W ≈ 0 (or solving out instantaneous effects)
            companion = self._build_companion_matrix()
            eigenvals = np.linalg.eigvals(companion)
            spectral_radius = np.max(np.abs(eigenvals))
            is_stable = spectral_radius < 1.0
        
        logger.info(f"Stability check: spectral radius = {spectral_radius:.4f}, stable = {is_stable}")
        return is_stable, spectral_radius
    
    def _build_companion_matrix(self) -> np.ndarray:
        """Build companion matrix for stability analysis."""
        if self.p == 0:
            return np.zeros((0, 0))
        
        # For simplicity, assume S^(-1) ≈ I (when W is small)
        # Full analysis would require solving S^(-1) * A_k
        try:
            S_inv = np.linalg.inv(self.S)
            effective_A = [S_inv @ A for A in self.A_list]
        except LinAlgError:
            logger.warning("Using A matrices directly for stability (S not invertible)")
            effective_A = self.A_list
        
        # Build companion form
        companion_size = self.d * self.p
        companion = np.zeros((companion_size, companion_size))
        
        # First d rows: x(t) = A^(1) x(t-1) + ... + A^(p) x(t-p)
        for k in range(self.p):
            companion[:self.d, k*self.d:(k+1)*self.d] = effective_A[k]
        
        # Remaining rows: identity shifts (x(t-k) = x(t-k))
        for i in range(1, self.p):
            start_row = i * self.d
            end_row = (i + 1) * self.d
            start_col = (i - 1) * self.d
            end_col = i * self.d
            companion[start_row:end_row, start_col:end_col] = np.eye(self.d)
        
        return companion
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformation to get back to original scale.
        
        Args:
            df: DataFrame in transformed space
            
        Returns:
            DataFrame in original scale
        """
        if self.scaler is None:
            logger.warning("No scaler provided - returning data as-is")
            return df.copy()
        
        # Apply inverse standardization
        data_original = self.scaler.inverse_transform(df.values)
        df_original = pd.DataFrame(data_original, columns=df.columns, index=df.index)
        
        logger.info("Applied inverse scaling transformation")
        return df_original


def load_matrices_from_results(results_dir: str, 
                              window_idx: int,
                              var_names: List[str],
                              p: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Load weight matrices for a specific window from DynoTears results.
    
    Args:
        results_dir: Directory containing results
        window_idx: Window index to load
        var_names: List of variable names
        p: Lag order
        
    Returns:
        Tuple of (W, A_list) matrices
    """
    # Look for weights CSV file in results_dir and subdirectories
    weights_files = []
    
    # First, check the main directory
    if os.path.exists(results_dir):
        weights_files = [f for f in os.listdir(results_dir) if f.startswith("weights_") and f.endswith(".csv")]
        weights_files = [os.path.join(results_dir, f) for f in weights_files]
    
    # If not found, check the weights subdirectory
    weights_subdir = os.path.join(results_dir, "weights")
    if os.path.exists(weights_subdir):
        subdir_files = [f for f in os.listdir(weights_subdir) if f.startswith("weights_") and f.endswith(".csv")]
        weights_files.extend([os.path.join(weights_subdir, f) for f in subdir_files])
    
    if not weights_files:
        raise FileNotFoundError(f"No weights CSV files found in {results_dir} or {weights_subdir}")
    
    weights_file = weights_files[0]  # Use first found file
    logger.info(f"Loading matrices from {weights_file} for window {window_idx}")
    
    # Load weights data
    df_weights = pd.read_csv(weights_file)
    
    # Filter for specific window
    window_data = df_weights[df_weights['window_idx'] == window_idx]
    
    if window_data.empty:
        raise ValueError(f"No data found for window {window_idx}")
    
    d = len(var_names)
    
    # Initialize matrices
    W = np.zeros((d, d))
    A_list = [np.zeros((d, d)) for _ in range(p)]
    
    # Fill matrices from edge data
    for _, row in window_data.iterrows():
        lag = int(row['lag'])
        i = int(row['i'])
        j = int(row['j'])
        weight = float(row['weight'])
        
        if lag == 0:
            W[i, j] = weight
        elif 1 <= lag <= p:
            A_list[lag - 1][i, j] = weight
    
    logger.info(f"Loaded matrices: W ({np.sum(W != 0)} edges), A_list ({[np.sum(A != 0) for A in A_list]} edges)")
    
    return W, A_list


def reconstruct_from_results(results_dir: str,
                           window_idx: int,
                           var_names: List[str], 
                           p: int,
                           T: int,
                           initial_conditions: Optional[np.ndarray] = None,
                           stochastic: bool = False,
                           noise_std: float = 1.0,
                           scaler: Optional[StandardScaler] = None) -> pd.DataFrame:
    """
    High-level function to reconstruct time series from DynoTears results.
    
    Args:
        results_dir: Directory containing DynoTears results
        window_idx: Window index to use for reconstruction
        var_names: List of variable names
        p: Lag order
        T: Number of time steps to simulate
        initial_conditions: Initial conditions [p, d], random if None
        stochastic: Whether to add noise
        noise_std: Noise standard deviation
        scaler: Scaler for inverse transformation
        
    Returns:
        Reconstructed time series DataFrame
    """
    # Load matrices
    W, A_list = load_matrices_from_results(results_dir, window_idx, var_names, p)
    
    # Create reconstructor
    reconstructor = TimeSeriesReconstructor(W, A_list, var_names, scaler)
    
    # Check stability
    is_stable, spec_radius = reconstructor.check_stability()
    if not is_stable:
        logger.warning(f"System appears unstable (spectral radius = {spec_radius:.4f})")
    
    # Generate initial conditions if not provided
    if initial_conditions is None:
        logger.info("Generating random initial conditions")
        initial_conditions = np.random.normal(0, 0.1, (p, len(var_names)))
    
    # Reconstruct
    if stochastic:
        df_result = reconstructor.reconstruct_stochastic(initial_conditions, T, noise_std)
    else:
        df_result = reconstructor.reconstruct_deterministic(initial_conditions, T)
    
    # Apply inverse transformation if scaler provided
    if scaler is not None:
        df_result = reconstructor.inverse_transform(df_result)
    
    return df_result


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reconstruct time series from DynoTears results")
    parser.add_argument("--results_dir", required=True, help="Directory with DynoTears results")
    parser.add_argument("--window_idx", type=int, default=0, help="Window index to use")
    parser.add_argument("--variables", nargs="+", required=True, help="Variable names")
    parser.add_argument("--lag", type=int, default=1, help="Lag order")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to simulate")
    parser.add_argument("--stochastic", action="store_true", help="Add noise")
    parser.add_argument("--noise_std", type=float, default=1.0, help="Noise standard deviation")
    parser.add_argument("--output", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Reconstruct
    df_recon = reconstruct_from_results(
        results_dir=args.results_dir,
        window_idx=args.window_idx,
        var_names=args.variables,
        p=args.lag,
        T=args.steps,
        stochastic=args.stochastic,
        noise_std=args.noise_std
    )
    
    # Save if requested
    if args.output:
        df_recon.to_csv(args.output, index=True)
        logger.info(f"Saved reconstruction to {args.output}")
    else:
        print(df_recon.head(10))
        print(f"... ({len(df_recon)} total steps)")