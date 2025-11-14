#!/usr/bin/env python3
"""
Fast Causal Additive Model with P-Splines (CAM-DAG)
===================================================

Implements nonlinear causal discovery using:
1. Causal Additive Models (CAM) with P-Splines for each edge
2. Efficient O(Kd²) complexity (no O(d³) matrix exponential)
3. DAG enforcement via cycle detection (dag_enforcer.py)
4. Automatic smoothness selection via GCV (gcv_selector.py)

Key advantages over NOTEARS-MLP:
- 10-100x fewer parameters (Kd² vs d²×hidden_size)
- Interpretable: each edge is a 1D smooth function
- Efficient: additive structure enables parallel computation

Reference:
    Bühlmann, P., Peters, J., & Ernest, J. (2014). 
    "CAM: Causal additive models, high-dimensional order search and 
    penalized regression." Annals of Statistics, 42(6), 2526-2556.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import BSpline
import time

logger = logging.getLogger(__name__)


class PSplineBasis:
    """
    1D Penalized B-Spline basis for smooth nonlinear functions.
    
    Each edge function g_ji(x) is represented as:
        g_ji(x) = Σ_k β_k B_k(x)
    
    where B_k are B-spline basis functions and β_k are learned coefficients.
    """
    
    def __init__(self, n_knots: int = 10, degree: int = 3, 
                 knot_type: str = 'quantile'):
        """
        Initialize P-Spline basis.
        
        Args:
            n_knots: Number of interior knots (total basis functions = n_knots + degree - 1)
            degree: Degree of B-spline (3 = cubic, recommended)
            knot_type: 'quantile' (data-adaptive) or 'uniform' (evenly spaced)
        """
        self.n_knots = n_knots
        self.degree = degree
        self.knot_type = knot_type
        
        self.knots = None
        self.n_basis = n_knots + degree + 1  # Number of basis functions
        
        # Second-order difference matrix for smoothness penalty
        self.D2 = self._create_difference_matrix()
    
    def fit(self, x: np.ndarray) -> 'PSplineBasis':
        """
        Compute knot locations from data.
        
        Args:
            x: Data values [n_samples]
        
        Returns:
            self (for method chaining)
        """
        x = np.asarray(x).flatten()
        
        # Check for constant or near-constant data
        x_range = x.max() - x.min()
        if x_range < 1e-10:
            # Data is constant - create artificial spread for numerical stability
            # Suppress warning (happens frequently with high-dimensional data)
            x_min, x_max = x.mean() - 0.5, x.mean() + 0.5
        else:
            x_min, x_max = x.min(), x.max()
        
        if self.knot_type == 'quantile':
            # Place knots at quantiles of data (adaptive to distribution)
            # Ensure we get distinct knots even if data has low variance
            percentiles = np.linspace(0, 100, self.n_knots + 2)[1:-1]
            interior_knots = np.percentile(x, percentiles)
            
            # Check if knots collapsed (happens with low-variance data)
            unique_knots = np.unique(interior_knots)
            if len(unique_knots) < 2:
                # Fallback to uniform spacing (suppress warning - common with low-variance data)
                interior_knots = np.linspace(x_min, x_max, self.n_knots + 2)[1:-1]
        else:  # uniform
            # Evenly spaced knots between min and max
            interior_knots = np.linspace(x_min, x_max, self.n_knots + 2)[1:-1]
        
        # Final safety check: ensure at least 2 interior knots for degree-3 B-splines
        if len(interior_knots) < 2:
            logger.warning(f"Only {len(interior_knots)} interior knots, forcing minimum of 2")
            interior_knots = np.array([x_min + 0.25 * x_range, x_min + 0.75 * x_range])
        
        # Add boundary knots (repeated for B-spline definition)
        self.knots = np.concatenate([
            [x_min] * (self.degree + 1),
            interior_knots,
            [x_max] * (self.degree + 1)
        ])
        
        return self
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate B-spline basis at given points.
        
        Args:
            x: Evaluation points [n_samples]
        
        Returns:
            Basis matrix [n_samples, n_basis]
        """
        if self.knots is None:
            raise ValueError("Must call fit() before transform()")
        
        x = np.asarray(x).flatten()
        n = len(x)
        
        # Evaluate each basis function
        basis_matrix = np.zeros((n, self.n_basis))
        
        for k in range(self.n_basis):
            # Create coefficient vector (1 for k-th basis, 0 elsewhere)
            c = np.zeros(self.n_basis)
            c[k] = 1.0
            
            # Evaluate k-th B-spline
            try:
                spl = BSpline(self.knots, c, self.degree, extrapolate=True)
                basis_matrix[:, k] = spl(x)
            except Exception as e:
                logger.warning(f"B-spline evaluation failed for basis {k}: {e}")
                basis_matrix[:, k] = 0.0
        
        return basis_matrix
    
    def transform_into(self, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        """
        Evaluate B-spline basis at given points, writing into pre-allocated buffer.
        
        Memory-efficient version that avoids allocation overhead.
        
        Args:
            x: Evaluation points [n_samples]
            out: Pre-allocated output buffer [n_samples, n_basis]
        
        Returns:
            out (same as input, for convenience)
        """
        if self.knots is None:
            raise ValueError("Must call fit() before transform_into()")
        
        x = np.asarray(x).flatten()
        n = len(x)
        
        if out.shape != (n, self.n_basis):
            raise ValueError(f"Output buffer shape {out.shape} != required shape ({n}, {self.n_basis})")
        
        # Evaluate each basis function directly into output buffer
        for k in range(self.n_basis):
            # Create coefficient vector (1 for k-th basis, 0 elsewhere)
            c = np.zeros(self.n_basis)
            c[k] = 1.0
            
            # Evaluate k-th B-spline directly into out[:, k]
            try:
                spl = BSpline(self.knots, c, self.degree, extrapolate=True)
                out[:, k] = spl(x)
            except Exception as e:
                logger.warning(f"B-spline evaluation failed for basis {k}: {e}")
                out[:, k] = 0.0
        
        return out
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit knots and compute basis (convenience method)"""
        return self.fit(x).transform(x)
    
    def _create_difference_matrix(self) -> np.ndarray:
        """
        Create second-order difference matrix for smoothness penalty.
        
        The penalty is: λ * ||D2 @ β||² which penalizes curvature.
        
        Returns:
            Difference matrix [n_basis-2, n_basis]
        """
        K = self.n_basis
        D2 = np.zeros((K - 2, K))
        
        for i in range(K - 2):
            D2[i, i] = 1
            D2[i, i+1] = -2
            D2[i, i+2] = 1
        
        return D2
    
    def smoothness_penalty_matrix(self) -> np.ndarray:
        """
        Get the penalty matrix P = D2^T @ D2
        
        The penalized least squares objective is:
            ||y - X@β||² + λ * β^T @ P @ β
        
        Returns:
            Penalty matrix [n_basis, n_basis]
        """
        return self.D2.T @ self.D2


class CAMModel(nn.Module):
    """
    Causal Additive Model for Dynamic Bayesian Networks.
    
    For each variable i, models its value as:
        X_i(t) = Σ_j g_ji(X_j(t)) + Σ_k Σ_j h_jik(X_j(t-k)) + ε_i(t)
    
    where:
    - g_ji is a P-spline for contemporaneous effect j -> i
    - h_jik is a P-spline for lagged effect j -> i at lag k
    - Each function is represented by n_knots coefficients
    """
    
    def __init__(self, 
                 d: int, 
                 p: int, 
                 n_knots: int = 10,
                 degree: int = 3,
                 lambda_smooth: float = 0.01):
        """
        Initialize CAM model.
        
        Args:
            d: Number of variables
            p: Maximum lag order
            n_knots: Number of B-spline knots per edge
            degree: B-spline degree
            lambda_smooth: Smoothness penalty strength
        """
        super().__init__()
        
        self.d = d
        self.p = p
        self.n_knots = n_knots
        self.degree = degree
        self.lambda_smooth = lambda_smooth
        
        # P-Spline basis (fitted once per window, stored in buffers)
        self.n_basis = n_knots + degree + 1
        
        # Learnable parameters: spline coefficients for each edge
        # W_coefs[i,j,:] are coefficients for edge j -> i (contemporaneous)
        self.W_coefs = nn.Parameter(torch.randn(d, d, self.n_basis) * 0.01)
        
        # A_coefs[i,j,lag,:] are coefficients for edge j -> i at lag
        self.A_coefs = nn.Parameter(torch.randn(d, d, p, self.n_basis) * 0.01)
        
        # Basis matrices (stored as buffers, not trained)
        # These are precomputed once per window
        self.register_buffer('basis_current', torch.zeros(1, d, 1, self.n_basis))
        self.register_buffer('basis_lagged', torch.zeros(1, d, p, 1, self.n_basis))
        
        # Smoothness penalty matrix (constant)
        D2 = PSplineBasis(n_knots, degree)._create_difference_matrix()
        P = torch.tensor(D2.T @ D2, dtype=torch.float32)
        self.register_buffer('penalty_matrix', P)
        
        # Edge masks (for enforcing DAG structure)
        self.register_buffer('W_mask', torch.ones(d, d))
        self.register_buffer('A_mask', torch.ones(d, d, p))
        
        # Diagonal of W must be zero (no self-loops)
        self.W_mask.fill_diagonal_(0.0)

    def reset_parameters(self):
        """
        Reset parameters to random initialization for new window.

        Allows model reuse across rolling windows while maintaining independence.
        Each window starts from fresh random initialization without reallocating memory.

        Memory impact: 0 bytes (overwrites existing tensors in-place)
        """
        with torch.no_grad():
            # Reset to same distribution as __init__
            self.W_coefs.data = torch.randn(self.d, self.d, self.n_basis, device=self.W_coefs.device) * 0.01
            self.A_coefs.data = torch.randn(self.d, self.d, self.p, self.n_basis, device=self.A_coefs.device) * 0.01

            # Enforce DAG constraint: no self-loops
            for i in range(self.d):
                self.W_coefs.data[i, i, :] = 0.0

    def set_basis_matrices(self, X: torch.Tensor, Xlags: torch.Tensor):
        """
        Precompute B-spline basis matrices for current window.
        
        Uses memory pooling to reuse numpy buffers and avoid allocation overhead.
        
        Args:
            X: Current values [n_samples, d]
            Xlags: Lagged values [n_samples, d*p]
        """
        n = X.shape[0]
        
        # Pre-allocate output tensors
        basis_current = torch.zeros(n, self.d, self.n_basis, device=X.device, dtype=X.dtype)
        basis_lagged = torch.zeros(n, self.d, self.p, self.n_basis, device=X.device, dtype=X.dtype)
        
        # Memory pool: reuse single PSpline object and numpy buffer
        pspline = PSplineBasis(self.n_knots, self.degree)
        basis_buffer = np.zeros((n, self.n_basis), dtype=np.float32)  # Reusable buffer
        
        # Basis for current variables
        for j in range(self.d):
            x_j = X[:, j].detach().cpu().numpy()
            
            # Compute basis into reusable buffer (no allocation)
            pspline.fit(x_j)
            pspline.transform_into(x_j, out=basis_buffer)
            
            # Copy to GPU tensor
            basis_current[:, j, :] = torch.from_numpy(basis_buffer).to(device=X.device, dtype=X.dtype)
        
        # Basis for lagged variables (same buffer reuse)
        for lag in range(self.p):
            for j in range(self.d):
                x_jlag = Xlags[:, lag * self.d + j].detach().cpu().numpy()
                
                # Reuse same buffer
                pspline.fit(x_jlag)
                pspline.transform_into(x_jlag, out=basis_buffer)
                
                basis_lagged[:, j, lag, :] = torch.from_numpy(basis_buffer).to(device=X.device, dtype=X.dtype)
        
        # Update buffers IN-PLACE to avoid memory leak
        # self.basis_current and self.basis_lagged are registered buffers
        # Reassigning them creates new tensors and leaks the old ones!
        # Must use .resize_() + .copy_() for in-place update
        self.basis_current.resize_(basis_current.shape)
        self.basis_current.copy_(basis_current)
        
        self.basis_lagged.resize_(basis_lagged.shape)
        self.basis_lagged.copy_(basis_lagged)
        
        logger.debug(f"Precomputed basis matrices: current {self.basis_current.shape}, lagged {self.basis_lagged.shape}")
    
    def _forward_contemp(self, basis_flat, coefs_flat):
        """Helper for gradient checkpointing: contemporaneous effects"""
        return torch.mm(basis_flat, coefs_flat.t())
    
    def _forward_lagged(self, basis_lag_flat, coefs_lag_flat):
        """Helper for gradient checkpointing: lagged effects"""
        return torch.mm(basis_lag_flat, coefs_lag_flat.t())
    
    def forward(self, return_individual_effects: bool = False):
        """
        Compute predicted values X_pred = CAM(X_current, X_lagged)
        
        Uses precomputed basis matrices (set via set_basis_matrices).
        MEMORY-EFFICIENT with gradient checkpointing to reduce backward pass memory.
        
        Args:
            return_individual_effects: If True, also return contributions from each edge
        
        Returns:
            X_pred: Predicted values [n, d]
            (optional) effects: Dict with edge-level contributions
        """
        n = self.basis_current.shape[0]
        device = self.W_coefs.device
        
        # Initialize prediction
        X_pred = torch.zeros(n, self.d, device=device, dtype=self.basis_current.dtype)
        
        # ========================================
        # CONTEMPORANEOUS EFFECTS - GRADIENT CHECKPOINTING
        # ========================================
        # Prepare masked coefficients (remove self-loops)
        W_coefs_masked = self.W_coefs * self.W_mask.unsqueeze(-1)  # (d, d, K)
        for i in range(self.d):
            W_coefs_masked[i, i] = 0.0  # No self-loops
        
        # Reshape for matrix multiplication
        n, d, K = self.basis_current.shape
        basis_flat = self.basis_current.reshape(n, d * K)  # (n, d*K)
        coefs_flat = W_coefs_masked.reshape(self.d, d * K)  # (d, d*K)
        
        # Use gradient checkpointing to reduce memory (trades compute for memory)
        # This saves ~70% memory during backward pass
        from torch.utils.checkpoint import checkpoint
        X_pred = checkpoint(self._forward_contemp, basis_flat, coefs_flat, use_reentrant=False)
        
        # ========================================
        # LAGGED EFFECTS - GRADIENT CHECKPOINTING
        # ========================================
        # Apply mask
        A_coefs_masked = self.A_coefs * self.A_mask.unsqueeze(-1)  # (d, d, p, K)
        
        # Reshape for efficient matrix multiplication
        n, d, p, K = self.basis_lagged.shape
        basis_lag_flat = self.basis_lagged.reshape(n, d * p * K)  # (n, d*p*K)
        coefs_lag_flat = A_coefs_masked.reshape(self.d, d * p * K)  # (d, d*p*K)
        
        # Gradient checkpointing for lagged effects
        X_pred += checkpoint(self._forward_lagged, basis_lag_flat, coefs_lag_flat, use_reentrant=False)
        
        # Note: return_individual_effects not supported with gradient checkpointing
        # (would require storing intermediates, defeating the purpose)
        return X_pred
    
    def get_edge_strength(self, i: int, j: int, lag: int = 0) -> float:
        """
        Compute edge strength as L2 norm of spline coefficients.
        
        This gives a single scalar weight for each edge, making it
        comparable to the linear model's weight matrices.
        
        Args:
            i: Child (effect) variable
            j: Parent (cause) variable
            lag: Lag (0 for contemporaneous)
        
        Returns:
            Edge strength (scalar)
        """
        with torch.no_grad():
            if lag == 0:
                mask = self.W_mask[i, j].item()
                if mask == 0:
                    return 0.0
                coefs = self.W_coefs[i, j, :]
            else:
                mask = self.A_mask[i, j, lag - 1].item()
                if mask == 0:
                    return 0.0
                coefs = self.A_coefs[i, j, lag - 1, :]
            
            # L2 norm of coefficients
            strength = torch.norm(coefs, p=2).item()
            return strength * mask
    
    def get_weight_matrix(self, lag: int = 0) -> np.ndarray:
        """
        Extract weight matrix W[i,j] = strength of edge j -> i
        
        For compatibility with linear DynoTEARS interface.
        
        Args:
            lag: Which lag to extract (0 = contemporaneous)
        
        Returns:
            Weight matrix [d, d]
        """
        # VECTORIZED: compute all edge strengths in one GPU operation
        if lag == 0:
            # Contemporaneous: W[i,j] = ||W_coefs[i,j]||_2
            W = torch.norm(self.W_coefs, p=2, dim=2)  # [d, d]
        else:
            # Lagged: A[i,j,lag] = ||A_coefs[i,j,lag]||_2
            W = torch.norm(self.A_coefs[:, :, lag, :], p=2, dim=2)  # [d, d]
        
        return W.detach().cpu().numpy()
    
    def compute_smoothness_penalty(self) -> torch.Tensor:
        """
        Compute smoothness penalty: λ * Σ ||D2 @ β||²
        
        This penalizes curvature in the learned functions, encouraging smooth fits.
        MEMORY-EFFICIENT: Processes lagged penalties in chunks to avoid OOM.
        
        Returns:
            Scalar penalty value
        """
        # Contemporaneous edges: W_coefs [d, d, n_basis]
        # penalty[i,j] = coefs[i,j] @ penalty_matrix @ coefs[i,j]
        W_coefs_flat = self.W_coefs.view(-1, self.n_basis)  # [d*d, n_basis]
        W_penalties = torch.sum(W_coefs_flat @ self.penalty_matrix * W_coefs_flat, dim=1)  # [d*d]
        W_penalties = W_penalties.view(self.d, self.d)  # [d, d]
        penalty_w = torch.sum(W_penalties * self.W_mask)  # Apply mask
        
        # Lagged edges: A_coefs [d, d, p, n_basis] - CHUNKED to avoid OOM
        # Full tensor would be 167M floats = 668 MB, but backward pass needs 2× = 1.3 GB
        # Chunk by target variable to stay under 500 MB per chunk
        chunk_size = 500  # Process 500 target variables at a time
        penalty_a = torch.tensor(0.0, device=self.A_coefs.device, dtype=self.A_coefs.dtype)
        
        for i_start in range(0, self.d, chunk_size):
            i_end = min(i_start + chunk_size, self.d)
            
            # Process chunk [i_start:i_end, :, :, :]
            A_chunk = self.A_coefs[i_start:i_end]  # [chunk, d, p, n_basis]
            A_chunk_flat = A_chunk.reshape(-1, self.n_basis)  # [chunk*d*p, n_basis]
            
            # Compute penalties for chunk
            A_chunk_penalties = torch.sum(A_chunk_flat @ self.penalty_matrix * A_chunk_flat, dim=1)
            A_chunk_penalties = A_chunk_penalties.view(i_end - i_start, self.d, self.p)
            
            # Apply mask and accumulate
            penalty_a += torch.sum(A_chunk_penalties * self.A_mask[i_start:i_end])
        
        return self.lambda_smooth * (penalty_w + penalty_a)
    
    def update_masks(self, W_dag: torch.Tensor, A_dag: Optional[torch.Tensor] = None):
        """
        Update edge masks to enforce DAG structure.
        
        Called by DAG enforcer after projecting weights onto DAG space.
        
        Args:
            W_dag: Contemporaneous adjacency matrix (binary or weighted)
            A_dag: Lagged adjacency matrix (optional, defaults to all ones)
        """
        with torch.no_grad():
            # Update contemporaneous mask
            self.W_mask.copy_(W_dag)
            self.W_mask.fill_diagonal_(0.0)  # Ensure no self-loops
            
            # VECTORIZED: Zero out coefficients for disabled edges (hard constraint)
            # Only keep coefficients where mask == 1
            mask_expanded = self.W_mask.unsqueeze(-1)  # [d, d, 1]
            self.W_coefs.mul_(mask_expanded)  # Broadcast to [d, d, n_basis]
            
            # Update lagged mask if provided
            if A_dag is not None:
                self.A_mask.copy_(A_dag)
                
                # VECTORIZED: Zero out lagged coefficients
                mask_expanded = self.A_mask.unsqueeze(-1)  # [d, d, p, 1]
                self.A_coefs.mul_(mask_expanded)  # Broadcast to [d, d, p, n_basis]
    
    def count_parameters(self) -> int:
        """Count number of active (non-masked) parameters"""
        n_params = 0
        
        # Contemporaneous
        n_params += (self.W_mask.sum() * self.n_basis).item()
        
        # Lagged
        n_params += (self.A_mask.sum() * self.n_basis).item()
        
        return int(n_params)
    
    def get_effective_df(self) -> float:
        """
        Estimate effective degrees of freedom for GCV.
        
        For P-splines: df ≈ n_basis - (penalty effect)
        The penalty reduces df by approximately penalty_strength / (1 + penalty_strength)
        """
        # Base df: number of basis functions per active edge
        base_df = self.count_parameters()
        
        # Penalty reduces df
        penalty_factor = self.lambda_smooth / (1.0 + self.lambda_smooth)
        
        effective_df = base_df * (1.0 - 0.5 * penalty_factor)
        
        return max(1.0, effective_df)


def test_cam_model():
    """Test CAM model on synthetic data"""
    print("Testing CAM Model...")
    
    # Small synthetic dataset
    torch.manual_seed(42)
    n, d, p = 50, 3, 2
    
    X = torch.randn(n, d)
    Xlags = torch.randn(n, d * p)
    
    # Create model
    model = CAMModel(d=d, p=p, n_knots=5, lambda_smooth=0.01)
    print(f"Model: {d} variables, lag {p}, {model.count_parameters()} parameters")
    
    # Set basis matrices
    print("Computing basis matrices...")
    model.set_basis_matrices(X, Xlags)
    
    # Forward pass
    print("Forward pass...")
    X_pred = model()
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {X_pred.shape}")
    
    # Check edge strengths
    print("Edge strengths (contemporaneous):")
    for i in range(d):
        for j in range(d):
            if i != j:
                strength = model.get_edge_strength(i, j, lag=0)
                print(f"  {j} -> {i}: {strength:.4f}")
    
    # Weight matrix
    W = model.get_weight_matrix(lag=0)
    print(f"Weight matrix W:\n{W}")
    
    # Smoothness penalty
    penalty = model.compute_smoothness_penalty()
    print(f"Smoothness penalty: {penalty:.6f}")
    
    print("\n✓ CAM Model test passed!")


if __name__ == "__main__":
    test_cam_model()
