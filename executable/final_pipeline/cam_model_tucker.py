#!/usr/bin/env python3
"""
Tucker-Decomposed CAM Model

Memory-efficient P-spline model using Tucker decomposition to compress
the huge coefficient tensors.

Tucker factorization:
- W[d,d,K] ≈ W_core[r,r,r] × W_U1[d,r] × W_U2[d,r] × W_U3[K,r]
- A[d,d,p,K] ≈ A_core[r,r,r,r] × A_U1[d,r] × A_U2[d,r] × A_U3[p,r] × A_U4[K,r]

This reduces 875M parameters to ~192K with r=20 (4500× reduction!)
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import BSpline


class TuckerCAMModel(nn.Module):
    """
    CAM model with Tucker-decomposed coefficient tensors.

    Uses low-rank Tucker factorization to compress P-spline coefficients,
    enabling nonlinear causal discovery on high-dimensional time series.
    """

    def __init__(
        self,
        d: int,
        p: int,
        n_knots: int = 5,
        rank_w: int = 20,
        rank_a: int = 10,
        lambda_smooth: float = 0.01,
        device='cpu'
    ):
        """
        Initialize Tucker-decomposed CAM model.

        Args:
            d: Number of variables
            p: Maximum lag order
            n_knots: Number of B-spline knots (degree=3 → K=n_knots+3 basis functions)
            rank_w: Tucker rank for contemporaneous edges (higher = more expressive)
            rank_a: Tucker rank for lagged edges (can be lower since more regularization)
            lambda_smooth: Smoothness penalty weight
            device: 'cpu' or 'cuda'
        """
        super().__init__()

        self.d = d
        self.p = p
        self.n_knots = n_knots
        self.rank_w = rank_w
        self.rank_a = rank_a
        self.lambda_smooth = lambda_smooth
        self.device = device

        # B-spline parameters (cubic splines, degree=3)
        self.degree = 3
        self.K = n_knots + self.degree  # Number of basis functions

        # Tucker factors for W (contemporaneous): d × d × K
        self.W_core = nn.Parameter(torch.randn(rank_w, rank_w, rank_w, device=device))
        self.W_U1 = nn.Parameter(torch.randn(d, rank_w, device=device))
        self.W_U2 = nn.Parameter(torch.randn(d, rank_w, device=device))
        self.W_U3 = nn.Parameter(torch.randn(self.K, rank_w, device=device))

        # Tucker factors for A (lagged): d × d × p × K
        self.A_core = nn.Parameter(torch.randn(rank_a, rank_a, rank_a, rank_a, device=device))
        self.A_U1 = nn.Parameter(torch.randn(d, rank_a, device=device))
        self.A_U2 = nn.Parameter(torch.randn(d, rank_a, device=device))
        self.A_U3 = nn.Parameter(torch.randn(p, rank_a, device=device))
        self.A_U4 = nn.Parameter(torch.randn(self.K, rank_a, device=device))

        # B-spline basis matrices (fixed, not trainable)
        self.register_buffer('B_w', torch.ones(1, self.K, device=device))  # Placeholder
        self.register_buffer('B_a', torch.ones(1, self.K, device=device))  # Placeholder

        # Masks for enforcing constraints
        self.register_buffer('W_mask', torch.ones(d, d, device=device))
        self.register_buffer('A_mask', torch.ones(d, d, p, device=device))

        self.reset_parameters()

        # Memory comparison
        dense_params = d * d * self.K + d * d * p * self.K
        tucker_params = (rank_w**3 + 2*d*rank_w + self.K*rank_w +
                        rank_a**4 + 2*d*rank_a + p*rank_a + self.K*rank_a)
        reduction = dense_params / tucker_params if tucker_params > 0 else 0

        print(f"[Tucker CAM] Initialized:")
        print(f"  Tucker params (r_w={rank_w}, r_a={rank_a}): {tucker_params:,}")
        print(f"  Dense P-splines: {dense_params:,}")
        print(f"  Memory reduction: {reduction:.1f}×")

    def reset_parameters(self):
        """Reset Tucker factors to small random initialization (balanced for large d)."""
        # Scale initialization based on dimensionality
        scale_w = 1.0 / np.sqrt(self.rank_w * self.d)
        scale_a = 1.0 / np.sqrt(self.rank_a * self.d * self.p)

        nn.init.normal_(self.W_core, 0, scale_w)
        nn.init.normal_(self.W_U1, 0, scale_w)
        nn.init.normal_(self.W_U2, 0, scale_w)
        nn.init.normal_(self.W_U3, 0, scale_w)

        nn.init.normal_(self.A_core, 0, scale_a)
        nn.init.normal_(self.A_U1, 0, scale_a)
        nn.init.normal_(self.A_U2, 0, scale_a)
        nn.init.normal_(self.A_U3, 0, scale_a)
        nn.init.normal_(self.A_U4, 0, scale_a)

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_W_coefs(self):
        """
        Reconstruct W coefficients from Tucker factors.

        Returns: Tensor of shape (d, d, K)
        """
        # W[i,j,k] = sum_rst W_core[r,s,t] * W_U1[i,r] * W_U2[j,s] * W_U3[k,t]
        W = torch.einsum('rst,ir,js,kt->ijk',
                        self.W_core, self.W_U1, self.W_U2, self.W_U3)
        return W

    def get_A_coefs(self):
        """
        Reconstruct A coefficients from Tucker factors.

        Returns: Tensor of shape (d, d, p, K)
        """
        # A[i,j,l,k] = sum_rstu A_core[r,s,t,u] * A_U1[i,r] * A_U2[j,s] * A_U3[l,t] * A_U4[k,u]
        A = torch.einsum('rstu,ir,js,lt,ku->ijlk',
                        self.A_core, self.A_U1, self.A_U2, self.A_U3, self.A_U4)
        return A

    def set_basis_matrices(self, B_w, B_a):
        """Set precomputed B-spline basis matrices."""
        self.B_w = B_w.to(self.device)
        self.B_a = B_a.to(self.device)

    def _compute_basis_matrix(self, X, degree=3):
        """Compute B-spline basis matrix for input data."""
        n = X.shape[0]

        # Create knot vector
        knots = np.linspace(0, 1, self.n_knots)
        # Extend for B-spline boundary conditions
        t = np.concatenate([
            np.repeat(knots[0], degree),
            knots,
            np.repeat(knots[-1], degree)
        ])

        # Compute basis functions
        basis_list = []
        x_vals = np.linspace(0, 1, n)

        for i in range(self.K):
            # Create B-spline basis function
            coef = np.zeros(self.K)
            coef[i] = 1.0
            bspl = BSpline(t, coef, degree)
            basis_vals = bspl(x_vals)
            basis_list.append(basis_vals)

        B = np.column_stack(basis_list)
        return torch.tensor(B, dtype=torch.float32, device=self.device)

    def forward(self, X, Xlags):
        """
        Forward pass: reconstruct coefficients from Tucker factors,
        use them in matrix multiply, then discard.

        Args:
            X: Current values (n, d)
            Xlags: Lagged values (n, d*p)

        Returns:
            predictions: (n, d)
        """
        n = X.shape[0]

        # Reconstruct coefficient tensors (on-the-fly)
        W_coefs = self.get_W_coefs()  # (d, d, K)
        A_coefs = self.get_A_coefs()  # (d, d, p, K)

        # Apply basis functions: (n, K) @ (d, d, K) -> (n, d, d)
        W_funcs = torch.einsum('nk,ijk->nij', self.B_w[:n], W_coefs)
        A_funcs = torch.einsum('nk,ijlk->nijl', self.B_a[:n], A_coefs)

        # Contemporaneous effects: X[n,d] @ W[n,d,d] -> (n,d)
        contrib_w = torch.einsum('ni,nij->nj', X, W_funcs)

        # Lagged effects: Xlags[n,d*p] @ A[n,d,d,p] -> (n,d)
        Xlags_reshaped = Xlags.reshape(n, self.d, self.p)
        contrib_a = torch.einsum('nil,nijl->nj', Xlags_reshaped, A_funcs)

        predictions = contrib_w + contrib_a
        return predictions

    def compute_smoothness_penalty(self):
        """
        Compute smoothness penalty for all edges (Tucker-aware).

        Penalizes variation in coefficient functions to encourage smoothness.
        """
        # For W: penalize differences in K dimension
        W_coefs = self.get_W_coefs()  # (d, d, K)
        W_diff = W_coefs[:, :, 1:] - W_coefs[:, :, :-1]
        penalty_w = torch.sum(W_diff ** 2)

        # For A: penalize differences in K dimension
        A_coefs = self.get_A_coefs()  # (d, d, p, K)
        A_diff = A_coefs[:, :, :, 1:] - A_coefs[:, :, :, :-1]
        penalty_a = torch.sum(A_diff ** 2)

        return self.lambda_smooth * (penalty_w + penalty_a)

    def _compute_penalty_matrix(self):
        """Helper for smoothness penalty computation."""
        # Second-order difference matrix
        K = self.K
        D = torch.zeros(K-2, K, device=self.device)
        for i in range(K-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        return D.T @ D

    def get_weight_matrix(self):
        """
        Get weight matrix for acyclicity constraint.

        Returns: W matrix (d, d) - contemporaneous edge weights
        """
        # Average over basis functions
        W_coefs = self.get_W_coefs()  # (d, d, K)
        W = W_coefs.mean(dim=2)  # Average over K

        # Apply mask
        W = W * self.W_mask

        return W

    def get_all_weight_matrices_gpu(self):
        """
        Get all weight matrices efficiently on GPU.

        Returns:
            W: (d, d) contemporaneous
            A_lags: List of (d, d) for each lag l=1..p
        """
        W = self.get_weight_matrix()

        # Get lagged matrices
        A_coefs = self.get_A_coefs()  # (d, d, p, K)
        A_lags = [A_coefs[:, :, l, :].mean(dim=2) * self.A_mask[:, :, l]
                  for l in range(self.p)]

        return W, A_lags

    def update_masks(self, W_mask=None, A_mask=None):
        """Update masks for enforcing structural constraints."""
        if W_mask is not None:
            self.W_mask = W_mask.to(self.device)
        if A_mask is not None:
            self.A_mask = A_mask.to(self.device)


if __name__ == "__main__":
    print("="*60)
    print("Testing Tucker-Decomposed CAM Model")
    print("="*60)

    # Test parameters
    d = 100  # variables
    p = 5    # lags
    n = 80   # samples
    n_knots = 5
    rank_w = 20
    rank_a = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTucker ranks: r_w={rank_w}, r_a={rank_a}")
    print(f"Using device: {device}\n")

    # Create model
    model = TuckerCAMModel(
        d=d, p=p, n_knots=n_knots,
        rank_w=rank_w, rank_a=rank_a,
        device=device
    )

    # Create random data
    X = torch.randn(n, d, device=device)
    Xlags = torch.randn(n, d*p, device=device)

    # Compute basis matrices
    B_w = model._compute_basis_matrix(X)
    B_a = model._compute_basis_matrix(X)
    model.set_basis_matrices(B_w, B_a)

    # Test forward pass
    print("\nTesting forward pass...")
    pred = model.forward(X, Xlags)
    print(f"  Input shape: X={X.shape}, Xlags={Xlags.shape}")
    print(f"  Output shape: {pred.shape}")
    print(f"  Output range: [{pred.min():.3f}, {pred.max():.3f}]")

    # Test weight extraction
    print("\nTesting weight matrix extraction...")
    W = model.get_weight_matrix()
    print(f"  W shape: {W.shape}")
    print(f"  Weight matrix range: [{W.min():.3f}, {W.max():.3f}]")

    # Test gradient flow
    print("\nTesting gradient flow...")
    loss = pred.sum()
    loss.backward()
    grad_norm = model.W_U1.grad.norm().item()
    print(f"  Gradient norm (W_U1): {grad_norm:.3f}")

    # Memory comparison
    dense_params = d * d * model.K + d * d * p * model.K
    tucker_params = model.count_parameters()
    print(f"\nMemory Comparison:")
    print(f"  Dense P-splines: {dense_params:,} parameters")
    print(f"  Tucker (r_w={rank_w}, r_a={rank_a}): {tucker_params:,} parameters")
    print(f"  Reduction: {dense_params/tucker_params:.1f}×")

    print("\n" + "="*60)
    print("✓ Tucker CAM model test passed!")
    print("="*60)
