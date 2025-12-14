#!/usr/bin/env python3
"""
Tucker-CAM-DAG: Memory-Efficient Nonlinear Causal Discovery

Integrates Tucker-decomposed CAM model with DynoTEARS optimization framework.

Architecture:
  Rolling Window → DynoTEARS Framework (this file) → CAM Model (cam_model_tucker.py) → Tucker Factors
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import logging

from cam_model_tucker import TuckerCAMModel

# Try to import NVIDIA Apex FusedAdam for 5-7% speedup
try:
    from apex.optimizers import FusedAdam
    FUSED_ADAM_AVAILABLE = True
except ImportError:
    FUSED_ADAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class TuckerFastCAMDAG:
    """
    Fast CAM-DAG with Tucker decomposition for massive memory reduction.

    Wraps TuckerCAMModel in DynoTEARS optimization framework with matrix_exp
    acyclicity constraint.
    """

    def __init__(
        self,
        d: int,
        p: int,
        n_knots: int = 5,
        rank_w: int = 20,
        rank_a: int = 10,
        lambda_w: float = 0.0,
        lambda_a: float = 0.0,
        lambda_smooth: float = 0.01,
        lambda_core: float = 0.01,
        lambda_orth: float = 0.001,
        device='cuda'
    ):
        """
        Initialize Tucker-CAM-DAG optimizer.

        Args:
            d: Number of variables
            p: Maximum lag order
            n_knots: Number of B-spline knots
            rank_w: Tucker rank for contemporaneous (higher = more expressive)
            rank_a: Tucker rank for lagged (can be lower)
            lambda_w: L1 penalty on contemporaneous edges (0 = Option D)
            lambda_a: L1 penalty on lagged edges (0 = Option D)
            lambda_smooth: Smoothness penalty on P-splines
            lambda_core: Core sparsity penalty (prevents smearing via sparse factor interactions)
            lambda_orth: Orthogonality penalty (ensures distinct, interpretable factors)
            device: 'cpu' or 'cuda'
        """
        self.d = d
        self.p = p
        self.n_knots = n_knots
        self.rank_w = rank_w
        self.rank_a = rank_a
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.lambda_smooth = lambda_smooth
        self.lambda_core = lambda_core
        self.lambda_orth = lambda_orth
        self.device = device

        # Model will be created in fit()
        self.model = None
        self.history = {'loss': [], 'h': [], 'rho': []}

        # Memory reduction estimate
        K = n_knots + 3
        dense_params = d * d * K + d * d * p * K
        tucker_params = (rank_w**3 + 2*d*rank_w + K*rank_w +
                        rank_a**4 + 2*d*rank_a + p*rank_a + K*rank_a)
        reduction = dense_params / tucker_params if tucker_params > 0 else 0

        logger.info(f"Tucker-CAM using device: {device}")
        logger.info(f"Tucker-CAM: Memory reduction = {reduction:.0f}×")

    def reset_parameters(self):
        """Reset model parameters."""
        if self.model is not None:
            self.model.reset_parameters()

    def fit(
        self,
        X: torch.Tensor,
        Xlags: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
        rho: float = 1.0,
        rho_max: float = 1e10,  # Reduced from 1e16 to prevent weight collapse
        h_tol: float = 1e-8,
        verbose: bool = False
    ):
        """
        Fit Tucker-CAM model to data.

        Uses augmented Lagrangian method to enforce acyclicity constraint.

        Args:
            X: Current values (n, d)
            Xlags: Lagged values (n, d*p)
            max_iter: Maximum optimization iterations
            lr: Learning rate
            rho: Initial penalty parameter for acyclicity
            rho_max: Maximum rho value
            h_tol: Tolerance for acyclicity constraint h(W)
            verbose: Print progress
        """
        n, d = X.shape

        # 1. Initialize Tucker factors (if first window)
        if self.model is None:
            if verbose:
                logger.info("Created Tucker-CAM model")
            self.model = TuckerCAMModel(
                d=self.d,
                p=self.p,
                n_knots=self.n_knots,
                rank_w=self.rank_w,
                rank_a=self.rank_a,
                lambda_smooth=self.lambda_smooth,
                device=self.device
            )
            # PyTorch 2.0+ JIT compilation
            # On CPU, 'reduce-overhead' can significantly speed up the small-step training loop
            if self.device == 'cpu':
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    if verbose:
                        logger.info("torch.compile() enabled for CPU (mode=reduce-overhead)")
                except Exception as e:
                    if verbose:
                        logger.warning(f"torch.compile() failed on CPU: {e}")
            elif self.device == 'cuda' and d > 1000:
                if verbose:
                    logger.info(f"torch.compile() DISABLED for d={d} (saves 20GB GPU memory)")
            else:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    if verbose:
                        logger.info("torch.compile() enabled (mode=reduce-overhead)")
                except Exception as e:
                    if verbose:
                        logger.warning(f"torch.compile() failed: {e}")
        else:
            if verbose:
                logger.info("Reusing compiled Tucker-CAM model (parameters reset)")
            # Reset parameters but keep compiled graph
            self.model.reset_parameters()

        # 2. Compute B-spline basis matrices
        if verbose:
            logger.info("Computing B-spline basis matrices...")
        B_w = self.model._compute_basis_matrix(X)
        B_a = self.model._compute_basis_matrix(X)
        self.model.set_basis_matrices(B_w, B_a)

        # 3. Setup optimizer (use NVIDIA FusedAdam for 5-7% speedup if available)
        if FUSED_ADAM_AVAILABLE and self.device == 'cuda':
            optimizer = FusedAdam(self.model.parameters(), lr=lr)
            if verbose:
                logger.info("Using NVIDIA FusedAdam optimizer (5-7% faster)")
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            if verbose and self.device == 'cuda':
                logger.info("Using standard Adam optimizer (install apex for FusedAdam speedup)")

        # Mixed precision training
        use_amp = self.device == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        # Gradient accumulation disabled to prevent weight collapse
        accumulation_steps = 1  # Disabled (was 2) - full gradient updates to maintain weight magnitudes

        # 4. Augmented Lagrangian parameters
        alpha = torch.zeros(1, device=self.device)  # Lagrange multiplier
        rho_current = rho

        self.history = {'loss': [], 'h': [], 'rho': []}

        if verbose:
            logger.info(f"Tucker-CAM: n={n}, d={d}, p={self.p}, AMP={'enabled' if use_amp else 'disabled'}")

        # 5. Optimization loop
        for iter_num in range(max_iter):
            # Zero gradients only every accumulation_steps iterations
            if iter_num % accumulation_steps == 0:
                optimizer.zero_grad()

            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    pred = self.model.forward(X, Xlags)

                    # Loss: squared error
                    loss_fit = torch.mean((X - pred) ** 2)

                    # L1 penalties (Option D: lambda_w=0, lambda_a=0)
                    W_coefs = self.model.get_W_coefs()
                    loss_l1_w = self.lambda_w * torch.sum(torch.abs(W_coefs))
                    
                    # Compute A_coefs L1 penalty in chunks to avoid OOM for large d
                    if self.lambda_a > 0:
                        loss_l1_a = 0.0
                        for i_start, i_end, A_chunk in self.model.get_A_coefs_chunked(chunk_size=100):
                            loss_l1_a += torch.sum(torch.abs(A_chunk))
                        loss_l1_a *= self.lambda_a
                    else:
                        loss_l1_a = 0.0  # Skip computation if lambda_a=0

                    # Smoothness penalty
                    loss_smooth = self.model.compute_smoothness_penalty()
                    
                    # Core sparsity penalty (prevents smearing)
                    loss_core = self.lambda_core * self.model.compute_core_sparsity_penalty()
                    
                    # Orthogonality penalty (distinct factors)
                    loss_orth = self.lambda_orth * self.model.compute_orthogonality_penalty()

                    # Acyclicity constraint: h(W) = tr(e^(W◦W)) - d
                    W = self.model.get_weight_matrix()  # (d, d)
                    W_squared = W * W
                    h = torch.trace(torch.matrix_exp(W_squared)) - d

                    # Augmented Lagrangian (divide by accumulation_steps for averaging)
                    loss_total = (loss_fit + loss_l1_w + loss_l1_a + loss_smooth + 
                                 loss_core + loss_orth +
                                 alpha * h + 0.5 * rho_current * h * h) / accumulation_steps

                # Backward pass with gradient scaling (accumulate gradients)
                scaler.scale(loss_total).backward()
                
                # Step optimizer only every accumulation_steps iterations
                if (iter_num + 1) % accumulation_steps == 0 or (iter_num + 1) == max_iter:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Standard precision (CPU or no AMP)
                pred = self.model.forward(X, Xlags)
                loss_fit = torch.mean((X - pred) ** 2)
                W_coefs = self.model.get_W_coefs()
                loss_l1_w = self.lambda_w * torch.sum(torch.abs(W_coefs))
                
                # Compute A_coefs L1 penalty in chunks to avoid OOM for large d
                if self.lambda_a > 0:
                    loss_l1_a = 0.0
                    for i_start, i_end, A_chunk in self.model.get_A_coefs_chunked(chunk_size=100):
                        loss_l1_a += torch.sum(torch.abs(A_chunk))
                    loss_l1_a *= self.lambda_a
                else:
                    loss_l1_a = 0.0  # Skip computation if lambda_a=0
                
                loss_smooth = self.model.compute_smoothness_penalty()
                
                # Core sparsity penalty (prevents smearing)
                loss_core = self.lambda_core * self.model.compute_core_sparsity_penalty()
                
                # Orthogonality penalty (distinct factors)
                loss_orth = self.lambda_orth * self.model.compute_orthogonality_penalty()
                
                W = self.model.get_weight_matrix()
                W_squared = W * W
                h = torch.trace(torch.matrix_exp(W_squared)) - d
                loss_total = (loss_fit + loss_l1_w + loss_l1_a + loss_smooth + 
                             loss_core + loss_orth +
                             alpha * h + 0.5 * rho_current * h * h) / accumulation_steps
                loss_total.backward()
                
                # Step optimizer only every accumulation_steps iterations
                if (iter_num + 1) % accumulation_steps == 0 or (iter_num + 1) == max_iter:
                    optimizer.step()

            # Track history
            self.history['loss'].append(loss_total.item())
            self.history['h'].append(h.item())
            self.history['rho'].append(rho_current)

            # Early stopping: check convergence
            # 1. DAG constraint satisfied
            if abs(h.item()) <= h_tol:
                if verbose:
                    logger.info(f"  Iter {iter_num}: h={h.item():.2e} (DAG constraint satisfied)")
                break

            # 2. Loss plateau (improved early stopping for speed)
            if iter_num >= 3:  # Check after just 3 iterations (was 5)
                recent_losses = self.history['loss'][-3:]  # Last 3 losses
                loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) / (abs(recent_losses[i-1]) + 1e-8)
                               for i in range(1, len(recent_losses))]
                if all(change < 5e-4 for change in loss_changes):  # Relaxed from 1e-4 to 5e-4 (0.05%)
                    if verbose:
                        logger.info(f"  Iter {iter_num}: Loss converged (plateau detected, changes < 0.05%)")
                    break

            # Update Lagrangian parameters
            if h.item() > 0.25 * self.history['h'][max(0, iter_num-1)]:
                # Not making progress, increase penalty
                rho_current = min(rho_current * 10, rho_max)
            alpha = alpha + rho_current * h.item()

            if verbose and (iter_num % 10 == 0 or iter_num == max_iter - 1):
                logger.info(
                    f"  Iter {iter_num}: loss={loss_fit.item():.4f}, "
                    f"h={h.item():.2e}, rho={rho_current:.1e}"
                )

        if verbose:
            logger.info(f"Tucker-CAM training completed in {iter_num+1} iterations")

    def get_structure_model(
        self,
        var_names: List[str],
        w_threshold: float = 0.01
    ) -> List[Tuple]:
        """
        Convert learned Tucker-CAM to edge list (vectorized - no NetworkX overhead).

        Args:
            var_names: Variable names
            w_threshold: Threshold for edge pruning (keep edges with |weight| > threshold)

        Returns:
            List of edges as tuples: (parent, child, {'weight': w, 'lag': l})
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Get weight matrices
        W, A_lags = self.model.get_all_weight_matrices_gpu()

        # Collect edges in a list (NO NetworkX - massive memory savings!)
        edges = []

        # Add contemporaneous edges (lag 0) - VECTORIZED
        W_np = W.detach().cpu().numpy()
        # Create mask: |weight| > threshold AND not diagonal
        mask_w = (np.abs(W_np) > w_threshold) & (np.eye(self.d) == 0)
        # Get indices of non-zero edges (vectorized - avoids 8M loop iterations!)
        indices_w = np.argwhere(mask_w)
        for idx in indices_w:
            i, j = idx
            edges.append((
                f"{var_names[j]}_lag0",  # parent
                f"{var_names[i]}_lag0",  # child
                {'weight': float(W_np[i, j]), 'lag': 0}
            ))

        # Add lagged edges (lag 1..p) - VECTORIZED
        for lag_idx, A_lag in enumerate(A_lags):
            A_np = A_lag.detach().cpu().numpy()
            lag = lag_idx + 1

            # Create mask: |weight| > threshold (vectorized)
            mask_a = np.abs(A_np) > w_threshold
            # Get indices of non-zero edges
            indices_a = np.argwhere(mask_a)
            for idx in indices_a:
                i, j = idx
                edges.append((
                    f"{var_names[j]}_lag{lag}",  # parent
                    f"{var_names[i]}_lag0",      # child
                    {'weight': float(A_np[i, j]), 'lag': lag}
                ))

        return edges


def from_pandas_dynamic_tucker_cam(
    time_series: Union[pd.DataFrame, List[pd.DataFrame]],
    p: int,
    rank_w: int = 20,
    rank_a: int = 10,
    lambda_w: float = 0.0,
    lambda_a: float = 0.0,
    n_knots: int = 5,
    lambda_smooth: float = 0.01,
    use_gcv: bool = False,
    max_iter: int = 100,
    lr: float = 0.01,
    w_threshold: float = 0.01,
    h_tol: float = 1e-8,
    device: str = 'cuda'
) -> List[Tuple]:
    """
    Learn DBN structure using Tucker-CAM-DAG (memory-efficient nonlinear).
    Returns edge list instead of NetworkX graph for massive memory savings.

    Args:
        time_series: Time series data (single DataFrame or list)
        p: Maximum lag order
        rank_w: Tucker rank for contemporaneous (20 recommended)
        rank_a: Tucker rank for lagged (10 recommended)
        lambda_w: L1 penalty contemporaneous (0 = Option D)
        lambda_a: L1 penalty lagged (0 = Option D)
        n_knots: Number of B-spline knots
        lambda_smooth: Smoothness penalty
        use_gcv: Not used (placeholder for API compatibility)
        max_iter: Maximum iterations
        lr: Learning rate
        w_threshold: Edge threshold
        device: 'cpu' or 'cuda'

    Returns:
        List of edges as tuples: (parent, child, {'weight': w, 'lag': l})
        NO NetworkX overhead - massive memory savings!
    """
    # Handle input
    if isinstance(time_series, pd.DataFrame):
        time_series = [time_series]

    # Concatenate all series
    df_concat = pd.concat(time_series, axis=0).reset_index(drop=True)
    var_names = df_concat.columns.tolist()
    d = len(var_names)

    # Convert to tensors
    data_np = df_concat.values.astype(np.float32)
    X = torch.tensor(data_np, dtype=torch.float32, device=device)

    # Create lagged features
    n = len(X)
    Xlags_list = []
    for lag in range(1, p+1):
        if lag < n:
            lagged = torch.cat([
                torch.zeros(lag, d, device=device),
                X[:-lag]
            ], dim=0)
            Xlags_list.append(lagged)

    if len(Xlags_list) > 0:
        Xlags = torch.cat(Xlags_list, dim=1)  # (n, d*p)
    else:
        Xlags = torch.zeros(n, d*p, device=device)

    logger.info(f"Tucker-CAM: n={n}, d={d}, p={p}")

    # Create model
    model = TuckerFastCAMDAG(
        d=d,
        p=p,
        n_knots=n_knots,
        rank_w=rank_w,
        rank_a=rank_a,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        lambda_smooth=lambda_smooth,
        device=device
    )

    # Fit model
    model.fit(X, Xlags, max_iter=max_iter, lr=lr, h_tol=h_tol, verbose=True)

    # Extract structure as edge list (NO NetworkX - massive memory savings!)
    edges = model.get_structure_model(var_names, w_threshold=w_threshold)

    logger.info(f"Tucker-CAM: Found {len(edges)} edges")

    # AGGRESSIVE memory cleanup to prevent accumulation across windows
    # Delete model and tensors explicitly
    del X, Xlags
    if hasattr(model, 'model') and model.model is not None:
        # Delete Tucker factors and cached matrices
        if hasattr(model.model, 'W_U1'):
            del model.model.W_U1, model.model.W_U2, model.model.W_core
        if hasattr(model.model, 'A_U1'):
            del model.model.A_U1, model.model.A_U2, model.model.A_U3, model.model.A_U4, model.model.A_core
        if hasattr(model.model, 'W_mask'):
            del model.model.W_mask, model.model.A_mask
        if hasattr(model.model, 'basis_matrices'):
            del model.model.basis_matrices
        del model.model
    del model
    
    # Force garbage collection
    import gc
    gc.collect()
    
    if device == 'cuda':
        torch.cuda.synchronize()  # Wait for all operations to complete
        torch.cuda.empty_cache()   # Free cached memory
        torch.cuda.reset_peak_memory_stats()  # Reset memory tracking

    return edges


if __name__ == "__main__":
    print("="*70)
    print("Testing Tucker-CAM-DAG Optimizer")
    print("="*70)

    # Generate synthetic data
    np.random.seed(42)
    n = 100
    d = 50
    p = 5

    # Random time series
    df = pd.DataFrame(
        np.random.randn(n, d),
        columns=[f"X{i}" for i in range(d)]
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"Data: {n} samples, {d} variables, {p} lags\n")

    # Test Tucker-CAM-DAG
    print("Fitting Tucker-CAM...")
    sm = from_pandas_dynamic_tucker_cam(
        df,
        p=p,
        rank_w=15,
        rank_a=8,
        lambda_w=0.0,
        lambda_a=0.0,
        n_knots=5,
        max_iter=50,
        device=device
    )

    # Count edges and nodes from edge list
    nodes = set()
    for parent, child, _ in sm:
        nodes.add(parent)
        nodes.add(child)

    print(f"\nResults:")
    print(f"  Edges found: {len(sm)}")
    print(f"  Variables: {len(nodes)}")

    print("\n" + "="*70)
    print("✓ Tucker-CAM-DAG test passed!")
    print("="*70)
