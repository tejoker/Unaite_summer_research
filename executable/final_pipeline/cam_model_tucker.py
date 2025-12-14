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

        # Memory estimation
        dense_params = d * d * self.K + d * d * p * self.K
        tucker_params = (rank_w**3 + 2*d*rank_w + self.K*rank_w +
                        rank_a**4 + 2*d*rank_a + p*rank_a + self.K*rank_a)
        reduction = dense_params / tucker_params if tucker_params > 0 else 0

        # Estimate peak memory usage during operations
        bytes_per_float = 4  # float32
        w_coefs_memory = d * d * self.K * bytes_per_float / (1024**3)  # GB
        a_coefs_memory = d * d * p * self.K * bytes_per_float / (1024**3)  # GB
        
        # Auto-calculated chunk size for A_coefs
        target_chunk_mb = 100
        auto_chunk_size = max(10, int((target_chunk_mb * 1024 * 1024) / (d * p * self.K * bytes_per_float)))
        a_chunk_memory = min(auto_chunk_size, d) * d * p * self.K * bytes_per_float / (1024**3)  # GB

        print(f"[Tucker CAM] Initialized:")
        print(f"  Variables (d): {d:,}, Lags (p): {p}, Basis (K): {self.K}")
        print(f"  Tucker params (r_w={rank_w}, r_a={rank_a}): {tucker_params:,}")
        print(f"  Dense P-splines: {dense_params:,}")
        print(f"  Memory reduction: {reduction:.1f}×")
        print(f"[Memory Estimate]")
        print(f"  W_coefs (full): {w_coefs_memory:.2f} GB")
        print(f"  A_coefs (full): {a_coefs_memory:.2f} GB {'← NEVER MATERIALIZED' if d > 500 else ''}")
        print(f"  A_coefs (chunked): {a_chunk_memory:.2f} GB (chunk_size={min(auto_chunk_size, d)})")
        print(f"  Peak memory usage (estimated): ~{w_coefs_memory + a_chunk_memory + 2:.1f} GB on {self.device.upper()}")

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
        
        WARNING: For large d (e.g., d>1000), this can be huge (d²×p×K).
        Consider using get_A_coefs_chunked() instead.
        """
        # A[i,j,l,k] = sum_rstu A_core[r,s,t,u] * A_U1[i,r] * A_U2[j,s] * A_U3[l,t] * A_U4[k,u]
        A = torch.einsum('rstu,ir,js,lt,ku->ijlk',
                        self.A_core, self.A_U1, self.A_U2, self.A_U3, self.A_U4)
        return A
    
    def get_A_coefs_chunked(self, chunk_size=None):
        """
        Reconstruct A coefficients in chunks to avoid OOM.
        
        For d=2889, full A_coefs is 2889²×20×8 = 13.4GB.
        Dynamically calculates safe chunk size based on d.
        
        Yields: (i_start, i_end, A_chunk) where A_chunk is shape (chunk, d, p, K)
        """
        if chunk_size is None:
            # Auto-calculate chunk size to keep memory per chunk < 100MB
            # A_chunk size: chunk × d × p × K × 4 bytes (float32)
            bytes_per_element = 4
            target_bytes = 100 * 1024 * 1024  # 100 MB target
            chunk_size = max(10, int(target_bytes / (self.d * self.p * self.K * bytes_per_element)))
            chunk_size = min(chunk_size, self.d)  # Don't exceed d
        
        for i in range(0, self.d, chunk_size):
            i_end = min(i + chunk_size, self.d)
            # Only compute for variables i:i_end
            A_chunk = torch.einsum('rstu,ir,js,lt,ku->ijlk',
                                   self.A_core, 
                                   self.A_U1[i:i_end], 
                                   self.A_U2, 
                                   self.A_U3, 
                                   self.A_U4)
            yield i, i_end, A_chunk

    def set_basis_matrices(self, B_w, B_a):
        """Set precomputed B-spline basis matrices."""
        self.B_w = B_w.to(self.device)
        self.B_a = B_a.to(self.device)

    def _compute_basis_matrix(self, X, degree=3):
        """
        Compute B-spline basis matrix for input data.
        
        Note: For rolling windows with same size, basis matrix stays constant.
        Could be cached at a higher level for additional speedup.
        """
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

        # --- OPTIMIZED FORWARD PASS (TENSOR CONTRACTION) ---
        # Avoids materializing full W (d,d,K) and A (d,d,p,K) tensors.
        # Complexity: O(N * d * R) instead of O(N * d^2 * K).
        # Memory: Minimal intermediate tensors.

        # 1. Contemporaneous (W)
        # Model: X @ W_mat, where W_mat[i,j] = sum_k W[i,j,k] * B_w[n,k]
        # Tucker: W[i,j,k] = G[r,s,t] * U1[i,r] * U2[j,s] * U3[k,t]
        # Contraction path:
        #   Basis_Proj[n,t] = B_w[n,k] @ U3[k,t]
        #   Input_Proj[n,r] = X[n,i] @ U1[i,r]
        #   Core_Inter[n,s] = G[r,s,t] * Input_Proj[n,r] * Basis_Proj[n,t]
        #   Output[n,j]     = Core_Inter[n,s] @ U2[j,s].T

        # Basis projection (N, r_w)
        basis_proj_w = torch.matmul(self.B_w[:n], self.W_U3)
        
        # Input projection (N, r_w)
        input_proj_w = torch.matmul(X, self.W_U1)
        
        # Core interaction (N, r_w) -> (N, r_2)
        # einsum: n r, n t, r s t -> n s
        core_inter_w = torch.einsum('nr, nt, rst -> ns', input_proj_w, basis_proj_w, self.W_core)
        
        # Output projection (N, d)
        contrib_w = torch.matmul(core_inter_w, self.W_U2.T)

        # 2. Lagged (A)
        # Model: Xlags @ A_mat
        # Tucker: A[i,j,l,k] = G[r,s,t,u] * U1[i,r] * U2[j,s] * U3[l,t] * U4[k,u]
        # Xlags structure: [Lag1 | Lag2 | ... | LagP] (stacked horizontally)
        # Reshape Xlags to (N, p, d) then permute to (N, d, p) to match A indices
        
        Xlags_reshaped = Xlags.view(n, self.p, self.d).permute(0, 2, 1) # (N, d, p)
        
        # Basis projection (N, r_a)
        basis_proj_a = torch.matmul(self.B_a[:n], self.A_U4)
        
        # Input projection (contract over d and p)
        # We need to contract X[n,i,l] with U1[i,r] and U3[l,t]
        # Step A: Contract lags (p) with U3 (p, t) -> (N, d, r_a)
        x_lag_proj = torch.einsum('nil, lt -> nit', Xlags_reshaped, self.A_U3)
        
        # Step B: Contract vars (d) with U1 (d, r) -> (N, r_a, r_a)
        # Indices: n i t, i r -> n t r (or n r t)
        x_spatial_proj = torch.einsum('nit, ir -> nrt', x_lag_proj, self.A_U1)
        
        # Core interaction (N, r_a)
        # einsum: n r t, n u, r s t u -> n s
        # Note: x_spatial_proj is (n, r, t) corresponding to (U1_dim, U3_dim)
        # Core is (r, s, t, u) -> (U1, U2, U3, U4)
        core_inter_a = torch.einsum('nrt, nu, rstu -> ns', x_spatial_proj, basis_proj_a, self.A_core)
        
        # Output projection (N, d)
        contrib_a = torch.matmul(core_inter_a, self.A_U2.T)

        return contrib_w + contrib_a

    def compute_smoothness_penalty(self):
        """
        Compute smoothness penalty for all edges (Tucker-aware).

        Penalizes variation in coefficient functions to encourage smoothness.
        Uses chunking to avoid OOM on large tensors.
        """
        # Dynamic chunk size based on problem size
        if self.d <= 25:
            chunk_size_w = self.d
            chunk_size_a = self.d
        elif self.d <= 100:
            chunk_size_w = 50
            chunk_size_a = 50
        elif self.d <= 500:
            chunk_size_w = 100
            chunk_size_a = 25  # Smaller for A since it's 4D
        else:
            # For very large d (e.g., 2889), use very small chunks
            chunk_size_w = 100
            chunk_size_a = None  # Auto-calculate

        # For W: penalize differences in K dimension (chunked)
        W_coefs = self.get_W_coefs()  # (d, d, K)
        penalty_w = 0.0
        for i in range(0, self.d, chunk_size_w):
            i_end = min(i + chunk_size_w, self.d)
            W_chunk = W_coefs[i:i_end]  # (chunk, d, K)
            W_diff_chunk = W_chunk[:, :, 1:] - W_chunk[:, :, :-1]
            penalty_w += torch.sum(W_diff_chunk ** 2)

        # For A: penalize differences in K dimension (chunked - avoid full materialization)
        penalty_a = 0.0
        for i_start, i_end, A_chunk in self.get_A_coefs_chunked(chunk_size=chunk_size_a):
            # A_chunk shape: (chunk, d, p, K)
            A_diff_chunk = A_chunk[:, :, :, 1:] - A_chunk[:, :, :, :-1]
            penalty_a += torch.sum(A_diff_chunk ** 2)
            # Free memory immediately
            del A_chunk, A_diff_chunk

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
        Get all weight matrices efficiently (2D block chunking for minimal memory).

        Returns:
            W: (d, d) contemporaneous
            A_lags: List of (d, d) for each lag l=1..p
        """
        import sys
        W = self.get_weight_matrix()

        # MEMORY-OPTIMIZED VERSION: Process in 2D blocks (row_chunk × col_chunk)
        # Instead of full (chunk, d, p, K), compute (row_chunk, col_chunk, p, K)
        # This reduces peak memory from 13.4GB to ~500MB
        
        # Initialize output
        A_lags = [torch.zeros(self.d, self.d, device=self.device) for _ in range(self.p)]
        
        # Determine block size (target 500MB per block, smaller for safety)
        bytes_per_element = 4  # float32
        target_bytes = 500 * 1024 * 1024  # 500MB per block
        # A_block: (row_chunk, col_chunk, p, K)
        # Size = row_chunk × col_chunk × p × K × 4 bytes
        block_elements = target_bytes // (self.p * self.K * bytes_per_element)
        block_size = int(block_elements ** 0.5)  # Square blocks
        block_size = max(100, min(block_size, 500))  # Between 100-500 rows/cols
        
        num_row_blocks = (self.d + block_size - 1) // block_size
        num_col_blocks = (self.d + block_size - 1) // block_size
        total_blocks = num_row_blocks * num_col_blocks
        
        print(f"    [Edge Extraction] Processing {num_row_blocks}×{num_col_blocks} = {total_blocks} blocks of {block_size}×{block_size}", flush=True)
        sys.stdout.flush()
        
        # Process in 2D blocks with progress signals
        block_count = 0
        for row_idx, i in enumerate(range(0, self.d, block_size)):
            i_end = min(i + block_size, self.d)
            
            for col_idx, j in enumerate(range(0, self.d, block_size)):
                j_end = min(j + block_size, self.d)
                block_count += 1
                
                # Progress signal every 20 blocks to reduce I/O overhead
                if block_count % 20 == 0 or block_count == total_blocks:
                    percent = int(100 * block_count / total_blocks)
                    print(f"    [Edge Extraction] Block {block_count}/{total_blocks} ({percent}% complete)", flush=True)
                    sys.stdout.flush()
                
                # Compute A[i:i_end, j:j_end, :, :] = einsum over Tucker factors
                # Shape: (row_chunk, col_chunk, p, K) - much smaller than (d, d, p, K)!
                A_block = torch.einsum('rstu,ir,js,lt,ku->ijlk',
                                       self.A_core,
                                       self.A_U1[i:i_end],
                                       self.A_U2[j:j_end],
                                       self.A_U3,
                                       self.A_U4)
                
                # Average over K and apply mask for this block
                for l in range(self.p):
                    A_lags[l][i:i_end, j:j_end] = A_block[:, :, l, :].mean(dim=2) * self.A_mask[i:i_end, j:j_end, l]
                
                # Free memory immediately after each block
                del A_block
        
        print(f"    [Edge Extraction] Complete: {total_blocks}/{total_blocks} blocks processed", flush=True)
        sys.stdout.flush()
        return W, A_lags

    def compute_core_sparsity_penalty(self):
        """
        L1 penalty on core tensors to enforce sparse factor interactions.
        
        Prevents "smearing" effect where dense cores cause spurious edges
        through indirect factor pathways. Forces model to select only
        critical factor interactions.
        
        Returns:
            torch.Tensor: Normalized L1 norm of core tensors
        """
        # L1 on core promotes element-wise sparsity
        w_core_l1 = torch.norm(self.W_core, p=1)
        a_core_l1 = torch.norm(self.A_core, p=1)
        
        # Normalize by number of elements to make lambdas comparable
        # W_core: (r_w, r_w, K), A_core: (r_a, r_a, p, K)
        w_core_l1 = w_core_l1 / self.W_core.numel()
        a_core_l1 = a_core_l1 / self.A_core.numel()
        
        return w_core_l1 + a_core_l1
    
    def compute_orthogonality_penalty(self):
        """
        Soft constraint: U^T @ U ≈ I for all factor matrices.
        
        Ensures factors are orthogonal and capture distinct features,
        preventing redundant/correlated factors from splitting edges.
        Improves interpretability and reduces smearing artifacts.
        
        Returns:
            torch.Tensor: Average Frobenius norm of (U^T U - I)
        """
        loss = 0.0
        factors = [self.W_U1, self.W_U2, self.W_U3,
                   self.A_U1, self.A_U2, self.A_U3, self.A_U4]
        
        count = 0
        for U in factors:
            if U.dim() == 2:  # Only factor matrices, not core tensors
                gram = torch.matmul(U.T, U)
                identity = torch.eye(gram.shape[0], device=U.device)
                # Frobenius norm squared: sum((Gram - I)^2)
                loss += torch.norm(gram - identity, p='fro') ** 2
                count += 1
        
        # Average over all factors to normalize
        return loss / max(1, count)

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
