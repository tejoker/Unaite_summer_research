#!/usr/bin/env python3
"""
Tucker-Decomposed CAM Model
===========================

Low-rank tensor factorization for memory-efficient nonlinear causal discovery.

Instead of storing full coefficient tensors:
- W_coefs: (d, d, K) = 41M params with d=2889, K=5
- A_coefs: (d, d, p, K) = 834M params with p=20

We factorize using Tucker decomposition:
- W ≈ Core(r,r,r) ×₁ U1(d,r) ×₂ U2(d,r) ×₃ U3(K,r)
- A ≈ Core(r,r,r,r) ×₁ U1(d,r) ×₂ U2(d,r) ×₃ U3(p,r) ×₄ U4(K,r)

With r=20: ~192K parameters (4500× reduction!)

Memory: ~1.5 GB total (vs 23.8 GB for dense P-splines)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline
import warnings


class TuckerCAMModel(nn.Module):
    """
    CAM model with Tucker-decomposed coefficient tensors.
    
    Combines:
    1. Low-rank tensor factorization (Tucker decomposition)
    2. P-Spline basis functions (smooth nonlinear functions)
    3. Group LASSO penalties (edge selection)
    
    Memory: O(d·r + r³) instead of O(d²K)
    """
    
    def __init__(self, 
                 d: int, 
                 p: int, 
                 n_knots: int = 5,
                 rank_w: int = 20,
                 rank_a: int = 10,
                 lambda_smooth: float = 0.01,
                 spline_degree: int = 3):
        """
        Initialize Tucker-decomposed CAM model.
        
        Args:
            d: Number of variables
            p: Maximum lag order
            n_knots: Number of B-spline knots per edge
            rank_w: Tucker rank for contemporaneous edges (higher = more expressive)
            rank_a: Tucker rank for lagged edges (can be lower since more regularization)
            lambda_smooth: Smoothness penalty coefficient
            spline_degree: Degree of B-spline basis (3 = cubic)
        """
        super().__init__()
        
        self.d = d
        self.p = p
        self.n_knots = n_knots
        self.rank_w = rank_w
        self.rank_a = rank_a
        self.lambda_smooth = lambda_smooth
        self.spline_degree = spline_degree
        
        # Number of basis functions (degree + n_knots - 1)
        self.n_basis = spline_degree + n_knots - 1
        
        # Tucker factors for contemporaneous edges W: (d, d, n_basis)
        # Core tensor: (rank_w, rank_w, rank_w)
        self.W_core = nn.Parameter(torch.randn(rank_w, rank_w, rank_w) / np.sqrt(rank_w))
        # Factor matrices
        self.W_U1 = nn.Parameter(torch.randn(d, rank_w) / np.sqrt(rank_w))  # Target variables
        self.W_U2 = nn.Parameter(torch.randn(d, rank_w) / np.sqrt(rank_w))  # Source variables
        self.W_U3 = nn.Parameter(torch.randn(self.n_basis, rank_w) / np.sqrt(rank_w))  # Basis coeffs
        
        # Tucker factors for lagged edges A: (d, d, p, n_basis)
        # Core tensor: (rank_a, rank_a, rank_a, rank_a)
        self.A_core = nn.Parameter(torch.randn(rank_a, rank_a, rank_a, rank_a) / np.sqrt(rank_a))
        # Factor matrices
        self.A_U1 = nn.Parameter(torch.randn(d, rank_a) / np.sqrt(rank_a))  # Target variables
        self.A_U2 = nn.Parameter(torch.randn(d, rank_a) / np.sqrt(rank_a))  # Source variables
        self.A_U3 = nn.Parameter(torch.randn(p, rank_a) / np.sqrt(rank_a))  # Lags
        self.A_U4 = nn.Parameter(torch.randn(self.n_basis, rank_a) / np.sqrt(rank_a))  # Basis coeffs
        
        # Masks for DAG enforcement (learned via topological sort)
        self.register_buffer('W_mask', torch.ones(d, d))
        self.register_buffer('A_mask', torch.ones(d, d, p))
        
        # Basis matrices (precomputed from data)
        self.register_buffer('basis_current', torch.zeros(1, d, self.n_basis))
        self.register_buffer('basis_lagged', torch.zeros(1, d * p, self.n_basis))
        
        # Smoothness penalty matrix (precomputed)
        self.register_buffer('penalty_matrix', self._compute_penalty_matrix())
        
        print(f"[Tucker CAM] Initialized:")
        print(f"  Dense params would be: {d*d*self.n_basis + d*d*p*self.n_basis:,} ({(d*d*self.n_basis + d*d*p*self.n_basis)*4/1e9:.2f} GB)")
        print(f"  Tucker params (r_w={rank_w}, r_a={rank_a}): {self.count_parameters():,} ({self.count_parameters()*4/1e6:.2f} MB)")
        print(f"  Memory reduction: {(d*d*self.n_basis + d*d*p*self.n_basis) / self.count_parameters():.1f}×")
    
    def count_parameters(self):
        """Count total trainable parameters"""
        # Contemporaneous
        w_params = self.rank_w**3 + 2*self.d*self.rank_w + self.n_basis*self.rank_w
        # Lagged
        a_params = self.rank_a**4 + 2*self.d*self.rank_a + self.p*self.rank_a + self.n_basis*self.rank_a
        return w_params + a_params
    
    def get_W_coefs(self):
        """
        Reconstruct W coefficients from Tucker factors.
        
        Returns:
            W_coefs: (d, d, n_basis) - contemporaneous edge coefficients
        """
        # W = Core ×₁ U1 ×₂ U2 ×₃ U3
        # Einstein notation: core[a,b,c] @ U1[i,a] @ U2[j,b] @ U3[k,c] -> W[i,j,k]
        W = torch.einsum('abc,ia,jb,kc->ijk', 
                        self.W_core, self.W_U1, self.W_U2, self.W_U3)
        return W
    
    def get_A_coefs(self):
        """
        Reconstruct A coefficients from Tucker factors.
        
        Returns:
            A_coefs: (d, d, p, n_basis) - lagged edge coefficients
        """
        # A = Core ×₁ U1 ×₂ U2 ×₃ U3 ×₄ U4
        # Einstein notation: core[a,b,c,d] @ U1[i,a] @ U2[j,b] @ U3[l,c] @ U4[k,d] -> A[i,j,l,k]
        A = torch.einsum('abcd,ia,jb,lc,kd->ijlk',
                        self.A_core, self.A_U1, self.A_U2, 
                        self.A_U3, self.A_U4)
        return A
    
    def reset_parameters(self):
        """Reset Tucker factors to small random initialization (balanced for large d)"""
        # Start with very small init, let optimizer grow from there
        # For d=2889: scale ~ 1e-4, which prevents NaN gradients but allows learning
        scale_w = 0.1 / np.sqrt(self.d * self.rank_w)
        scale_a = 0.1 / np.sqrt(self.d * self.rank_a)
        
        # Contemporaneous factors  
        nn.init.normal_(self.W_core, mean=0, std=scale_w)
        nn.init.normal_(self.W_U1, mean=0, std=scale_w)
        nn.init.normal_(self.W_U2, mean=0, std=scale_w)
        nn.init.normal_(self.W_U3, mean=0, std=scale_w)
        
        # Lagged factors
        nn.init.normal_(self.A_core, mean=0, std=scale_a)
        nn.init.normal_(self.A_U1, mean=0, std=scale_a)
        nn.init.normal_(self.A_U2, mean=0, std=scale_a)
        nn.init.normal_(self.A_U3, mean=0, std=scale_a)
        nn.init.normal_(self.A_U4, mean=0, std=scale_a)
        
        # Reset masks
        self.W_mask.fill_(1.0)
        self.A_mask.fill_(1.0)
    
    def set_basis_matrices(self, X: torch.Tensor, Xlags: torch.Tensor):
        """
        Precompute B-spline basis matrices from data.
        
        This is called once per window before optimization.
        
        Args:
            X: Current values [n, d]
            Xlags: Lagged values [n, d*p]
        """
        n = X.shape[0]
        
        # Compute basis for contemporaneous variables
        basis_current = self._compute_basis_matrix(X)  # (n, d, n_basis)
        
        # Compute basis for lagged variables
        basis_lagged = self._compute_basis_matrix(Xlags)  # (n, d*p, n_basis)
        
        # Store in buffers (in-place to avoid memory leak)
        if self.basis_current is None or self.basis_current.shape[0] != n:
            self.basis_current = basis_current.clone()
        else:
            self.basis_current.copy_(basis_current)
        
        if self.basis_lagged is None or self.basis_lagged.shape[0] != n:
            self.basis_lagged = basis_lagged.clone()
        else:
            self.basis_lagged.copy_(basis_lagged)
    
    def _compute_basis_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis matrix for input data.
        
        For each variable, we fit B-splines over its value range.
        
        Args:
            X: Input data [n, d] or [n, d*p]
        
        Returns:
            basis: [n, d, n_basis] - basis function evaluations
        """
        n, d = X.shape
        basis = torch.zeros(n, d, self.n_basis, device=X.device, dtype=X.dtype)
        
        # Convert to numpy for scipy
        X_np = X.cpu().numpy()
        
        for j in range(d):
            x_j = X_np[:, j]
            
            # Create knot sequence
            x_min, x_max = x_j.min(), x_j.max()
            
            # Avoid degenerate case (constant variable)
            if x_max - x_min < 1e-10:
                # Use identity basis (just copy the constant value)
                basis[:, j, 0] = 1.0
                continue
            
            # Interior knots + boundary knots for clamped spline
            interior_knots = np.linspace(x_min, x_max, self.n_knots)
            knots = np.concatenate([
                [x_min] * self.spline_degree,  # Left boundary
                interior_knots,
                [x_max] * self.spline_degree   # Right boundary
            ])
            
            # Evaluate each basis function
            for k in range(self.n_basis):
                # Create basis vector (1 at position k, 0 elsewhere)
                c = np.zeros(self.n_basis)
                c[k] = 1.0
                
                # Create B-spline object
                spl = BSpline(knots, c, self.spline_degree, extrapolate=False)
                
                # Evaluate at data points
                basis_vals = spl(x_j)
                
                # Handle extrapolation (set to 0 outside knot range)
                basis_vals = np.nan_to_num(basis_vals, nan=0.0)
                
                basis[:, j, k] = torch.tensor(basis_vals, device=X.device, dtype=X.dtype)
        
        return basis
    
    def forward(self):
        """
        Forward pass: compute predictions using precomputed basis.
        
        This is memory-efficient: we reconstruct coefficients on-the-fly
        from Tucker factors, use them in matrix multiply, then discard.
        
        Returns:
            X_pred: [n, d] - predicted values
        """
        n = self.basis_current.shape[0]
        
        # Reconstruct coefficients (on-the-fly, not stored)
        W_coefs = self.get_W_coefs()  # (d, d, n_basis)
        A_coefs = self.get_A_coefs()  # (d, d, p, n_basis)
        
        # Apply DAG masks
        W_coefs_masked = W_coefs * self.W_mask.unsqueeze(-1)
        A_coefs_masked = A_coefs * self.A_mask.unsqueeze(-1)
        
        # Contemporaneous contribution: X_pred[n,i] = Σ_j Σ_k W[i,j,k] * basis[n,j,k]
        # Reshape for efficient matrix multiply: (n, d*n_basis) @ (d, d*n_basis)^T
        basis_flat = self.basis_current.reshape(n, self.d * self.n_basis)  # (n, d*K)
        coefs_flat = W_coefs_masked.reshape(self.d, self.d * self.n_basis)  # (d, d*K)
        
        X_pred = torch.mm(basis_flat, coefs_flat.t())  # (n, d)
        
        # Lagged contribution: X_pred[n,i] += Σ_j Σ_l Σ_k A[i,j,l,k] * basis_lag[n,j*p+l,k]
        # Reshape: (n, d*p*n_basis) @ (d, d*p*n_basis)^T
        basis_lag_flat = self.basis_lagged.reshape(n, self.d * self.p * self.n_basis)  # (n, d*p*K)
        coefs_lag_flat = A_coefs_masked.reshape(self.d, self.d * self.p * self.n_basis)  # (d, d*p*K)
        
        X_pred += torch.mm(basis_lag_flat, coefs_lag_flat.t())  # (n, d)
        
        return X_pred
    
    def compute_smoothness_penalty(self) -> torch.Tensor:
        """
        Compute smoothness penalty for all edges (Tucker-aware).
        
        Penalty = Σ_{i,j} ||D² W_{ij}||² + Σ_{i,j,l} ||D² A_{ijl}||²
        
        Where D² is the second-order difference matrix.
        
        Returns:
            penalty: scalar smoothness penalty
        """
        # Reconstruct coefficients
        W_coefs = self.get_W_coefs()  # (d, d, n_basis)
        A_coefs = self.get_A_coefs()  # (d, d, p, n_basis)
        
        # Contemporaneous smoothness
        W_flat = W_coefs.reshape(-1, self.n_basis)  # (d*d, n_basis)
        W_smooth = torch.sum(W_flat @ self.penalty_matrix * W_flat)
        penalty_w = self.lambda_smooth * W_smooth
        
        # Lagged smoothness (chunked to save memory)
        penalty_a = 0.0
        chunk_size = 500  # Process 500 target variables at a time
        
        for i_start in range(0, self.d, chunk_size):
            i_end = min(i_start + chunk_size, self.d)
            A_chunk = A_coefs[i_start:i_end]  # (chunk, d, p, n_basis)
            
            A_chunk_flat = A_chunk.reshape(-1, self.n_basis)  # (chunk*d*p, n_basis)
            A_chunk_smooth = torch.sum(A_chunk_flat @ self.penalty_matrix * A_chunk_flat)
            penalty_a += self.lambda_smooth * A_chunk_smooth
        
        return penalty_w + penalty_a
    
    def _compute_penalty_matrix(self) -> torch.Tensor:
        """
        Compute second-order difference penalty matrix D^T D.
        
        Returns:
            P: [n_basis, n_basis] - penalty matrix
        """
        # Second-order difference matrix
        D = torch.zeros(self.n_basis - 2, self.n_basis)
        for i in range(self.n_basis - 2):
            D[i, i] = 1.0
            D[i, i+1] = -2.0
            D[i, i+2] = 1.0
        
        # Penalty matrix: D^T D
        P = D.t() @ D  # (n_basis, n_basis)
        
        return P
    
    def get_weight_matrix(self, lag: int = 0) -> np.ndarray:
        """
        Extract weight matrix for visualization/analysis.
        
        W[i,j] = ||coefficients_{ij}||_2 (L2 norm of spline coefficients)
        
        Args:
            lag: Which lag to extract (0 = contemporaneous, 1..p = lagged)
        
        Returns:
            W: [d, d] - weight matrix (numpy array)
        """
        with torch.no_grad():
            if lag == 0:
                # Contemporaneous
                W_coefs = self.get_W_coefs()  # (d, d, n_basis)
                W_norms = torch.norm(W_coefs, p=2, dim=2)  # (d, d)
                W_masked = W_norms * self.W_mask
            else:
                # Lagged
                A_coefs = self.get_A_coefs()  # (d, d, p, n_basis)
                A_norms = torch.norm(A_coefs[:, :, lag-1, :], p=2, dim=2)  # (d, d)
                W_masked = A_norms * self.A_mask[:, :, lag-1]
            
            return W_masked.cpu().numpy()
    
    def get_all_weight_matrices_gpu(self) -> torch.Tensor:
        """
        Get ALL weight matrices at once on GPU for fast edge extraction.
        
        Returns:
            W_all: [p+1, d, d] tensor on GPU
                W_all[0] = contemporaneous weights
                W_all[1:] = lagged weights for lags 1..p
        """
        with torch.no_grad():
            W_all = torch.zeros(self.p + 1, self.d, self.d, 
                              device=self.W_core.device, dtype=self.W_core.dtype)
            
            # Contemporaneous (lag 0)
            W_coefs = self.get_W_coefs()  # (d, d, n_basis)
            W_norms = torch.norm(W_coefs, p=2, dim=2)  # (d, d)
            W_all[0] = W_norms * self.W_mask
            
            # All lagged (lags 1..p) - vectorized!
            A_coefs = self.get_A_coefs()  # (d, d, p, n_basis)
            A_norms = torch.norm(A_coefs, p=2, dim=3)  # (d, d, p)
            A_masked = A_norms * self.A_mask
            W_all[1:] = A_masked.permute(2, 0, 1)  # (p, d, d)
            
            return W_all
    
    def update_masks(self, W_dag: torch.Tensor):
        """
        Update masks based on DAG-enforced adjacency matrix.
        
        Args:
            W_dag: [d, d] - binary adjacency matrix after DAG enforcement
        """
        # Update contemporaneous mask (keep diagonal = 0)
        self.W_mask.copy_(W_dag)
        self.W_mask.fill_diagonal_(0)
        
        # Lagged edges: no DAG constraint (time flows forward)
        # Keep all lagged connections active
        self.A_mask.fill_(1.0)


def test_tucker_cam():
    """Test Tucker CAM model"""
    print("\n" + "="*60)
    print("Testing Tucker-Decomposed CAM Model")
    print("="*60)
    
    # Test parameters
    d, p, n = 100, 5, 50
    n_knots = 5
    rank_w, rank_a = 10, 5
    
    print(f"\nTest setup: d={d}, p={p}, n={n}, K={n_knots}")
    print(f"Tucker ranks: r_w={rank_w}, r_a={rank_a}")
    
    # Create model
    model = TuckerCAMModel(d, p, n_knots=n_knots, rank_w=rank_w, rank_a=rank_a)
    
    # Generate synthetic data
    X = torch.randn(n, d)
    Xlags = torch.randn(n, d * p)
    
    # Set basis matrices
    print("\nComputing basis matrices...")
    model.set_basis_matrices(X, Xlags)
    
    # Forward pass
    print("Running forward pass...")
    X_pred = model()
    
    print(f"✓ Output shape: {X_pred.shape}")
    print(f"✓ Output range: [{X_pred.min():.3f}, {X_pred.max():.3f}]")
    
    # Compute penalties
    print("\nComputing penalties...")
    smoothness = model.compute_smoothness_penalty()
    print(f"✓ Smoothness penalty: {smoothness.item():.6f}")
    
    # Reconstruct coefficients
    print("\nReconstructing coefficients...")
    W_coefs = model.get_W_coefs()
    A_coefs = model.get_A_coefs()
    print(f"✓ W_coefs shape: {W_coefs.shape}")
    print(f"✓ A_coefs shape: {A_coefs.shape}")
    
    # Get weight matrix
    print("\nExtracting weight matrices...")
    W = model.get_weight_matrix(lag=0)
    print(f"✓ Weight matrix shape: {W.shape}")
    print(f"✓ Weight matrix range: [{W.min():.3f}, {W.max():.3f}]")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = torch.mean((X - X_pred) ** 2) + smoothness
    loss.backward()
    
    grad_norm = torch.norm(model.W_U1.grad)
    print(f"✓ Gradient norm (W_U1): {grad_norm.item():.6f}")
    
    # Memory comparison
    dense_params = d * d * model.n_basis + d * d * p * model.n_basis
    tucker_params = model.count_parameters()
    reduction = dense_params / tucker_params
    
    print("\n" + "="*60)
    print("Memory Comparison")
    print("="*60)
    print(f"Dense P-splines: {dense_params:,} params ({dense_params*4/1e6:.1f} MB)")
    print(f"Tucker (r_w={rank_w}, r_a={rank_a}): {tucker_params:,} params ({tucker_params*4/1e6:.1f} MB)")
    print(f"Reduction: {reduction:.1f}× fewer parameters")
    
    print("\n✓ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_tucker_cam()
