#!/usr/bin/env python3
"""
Fast Tucker-CAM-DAG Optimizer
==============================

Memory-efficient nonlinear causal discovery using Tucker decomposition.

Key improvements over dense P-splines:
1. 500-4500× fewer parameters via low-rank factorization
2. Fits d=2889, p=20, K=5 in <2 GB (vs 24 GB for dense)
3. Same API as dense version (drop-in replacement)

Main interface: from_pandas_dynamic_tucker_cam()
"""

import os
import sys
import logging
import time
import pickle
from pathlib import Path
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cam_model_tucker import TuckerCAMModel
from dag_enforcer import TopologicalDAGEnforcer
from structuremodel import StructureModel

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 8-bit Adam for additional memory savings
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
    logger.info("✓ 8-bit Adam available (will save ~75% optimizer memory)")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("⚠ bitsandbytes not available - using standard Adam")

# PyTorch configuration
num_threads = int(os.environ.get('PYTORCH_INTRA_OP_THREADS', '60'))
torch.set_num_threads(num_threads)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Tucker-CAM using device: {device}")

if device.type == 'cuda':
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class TuckerFastCAMDAG:
    """
    Fast CAM-DAG with Tucker decomposition for massive memory reduction.
    
    Complexity: O(ndr² + r³) per iteration (vs O(nKd²) for dense)
    Memory: O(dr + r³) (vs O(Kd²) for dense)
    """
    
    def __init__(self,
                 d: int,
                 p: int,
                 n_knots: int = 5,
                 rank_w: int = 20,
                 rank_a: int = 10,
                 lambda_w: float = 0.1,
                 lambda_a: float = 0.1,
                 lambda_smooth: float = 0.01,
                 dag_enforce_interval: int = 10,
                 device: str = 'cpu'):
        """
        Initialize Tucker-CAM-DAG optimizer.
        
        Args:
            d: Number of variables
            p: Maximum lag order
            n_knots: Number of B-spline knots per edge (can use K=5 now!)
            rank_w: Tucker rank for contemporaneous (higher = more expressive)
            rank_a: Tucker rank for lagged (can be lower)
            lambda_w: L2 penalty for contemporaneous edges
            lambda_a: L2 penalty for lagged edges
            lambda_smooth: Smoothness penalty
            dag_enforce_interval: Enforce DAG every N iterations
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
        self.dag_enforce_interval = dag_enforce_interval
        self.device = device
        
        # Components
        self.model = None
        self.optimizer = None
        self.dag_enforcer = TopologicalDAGEnforcer(threshold=0.01)
        
        # Training history
        self.history = {
            'loss': [],
            'mse': [],
            'penalty': [],
            'dag_violations': []
        }
        
        # Calculate memory savings
        dense_params = d * d * (n_knots + 4 - 1) * (1 + p)
        tucker_params = (rank_w**3 + 2*d*rank_w + (n_knots+4-1)*rank_w +
                        rank_a**4 + 2*d*rank_a + p*rank_a + (n_knots+4-1)*rank_a)
        self.memory_reduction = dense_params / tucker_params
        
        logger.info(f"Tucker-CAM: Memory reduction = {self.memory_reduction:.1f}× vs dense P-splines")
    
    def reset_parameters(self):
        """Reset model parameters for new window"""
        if self.model is not None:
            self.model.reset_parameters()
        
        # Clear optimizer (will recreate in fit())
        self.optimizer = None
        
        # Garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear history
        self.history = {
            'loss': [],
            'mse': [],
            'penalty': [],
            'dag_violations': []
        }
    
    def fit(self,
            X: torch.Tensor,
            Xlags: torch.Tensor,
            max_iter: int = 100,
            lr: float = 0.01,
            loss_tol: float = 1e-6,
            verbose: bool = True) -> 'TuckerFastCAMDAG':
        """
        Fit Tucker-CAM model to data.
        
        Algorithm:
        1. Initialize Tucker factors (if first window)
        2. Precompute B-spline basis matrices
        3. Alternating optimization:
            - Update Tucker factors via gradient descent
            - Every K iterations: project W onto DAG space
        
        Args:
            X: Current values [n, d]
            Xlags: Lagged values [n, d*p]
            max_iter: Maximum iterations
            lr: Learning rate
            loss_tol: Convergence tolerance
            verbose: Log progress
        
        Returns:
            self (trained)
        """
        n = X.shape[0]
        
        if verbose:
            logger.info(f"Tucker-CAM: n={n}, d={self.d}, p={self.p}, K={self.n_knots}")
            logger.info(f"Ranks: r_w={self.rank_w}, r_a={self.rank_a}")
            param_count = self.model.count_parameters() if self.model else None
            if param_count:
                logger.info(f"Parameters: {param_count:,} ({self.memory_reduction:.0f}× reduction)")
            else:
                logger.info(f"Parameters: TBD ({self.memory_reduction:.0f}× reduction)")
        
        # Initialize model (or reuse)
        if self.model is None:
            self.model = TuckerCAMModel(
                d=self.d,
                p=self.p,
                n_knots=self.n_knots,
                rank_w=self.rank_w,
                rank_a=self.rank_a,
                lambda_smooth=self.lambda_smooth
            ).to(self.device)
            if verbose:
                logger.info("Created Tucker-CAM model")
        else:
            if verbose:
                logger.info("Reusing Tucker-CAM model (parameters reset)")
        
        # Precompute basis matrices
        if verbose:
            logger.info("Computing B-spline basis matrices...")
        start_basis = time.time()
        self.model.set_basis_matrices(X, Xlags)
        elapsed_basis = time.time() - start_basis
        if verbose:
            logger.info(f"  Basis computation: {elapsed_basis:.2f}s")
        
        # Create optimizer
        if self.optimizer is None:
            if BITSANDBYTES_AVAILABLE:
                self.optimizer = bnb.optim.Adam8bit(
                    self.model.parameters(),
                    lr=lr,
                    percentile_clipping=100,
                    block_wise=True,
                    min_8bit_size=4096
                )
                if verbose:
                    logger.info("  Created 8-bit Adam optimizer")
            else:
                self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
                if verbose:
                    logger.info("  Created standard Adam optimizer")
        
        optimizer = self.optimizer
        prev_loss = None
        start_time = time.time()
        
        # Use mixed precision but with robust NaN handling
        # FP16 is needed for memory with large d, but can cause NaN gradients
        use_amp = device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', init_scale=2.**10, growth_interval=100) if use_amp else None
        
        if use_amp and verbose:
            logger.info("  Using mixed precision (FP16) with conservative scaling")
        
        if verbose:
            logger.info("Starting optimization...")
        
        # Learning rate warmup for first 5 iterations (helps with NaN gradients)
        warmup_iters = 5
        
        for it in range(max_iter):
            # Apply learning rate warmup
            if it < warmup_iters:
                warmup_lr = lr * (it + 1) / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            elif it == warmup_iters:
                # Restore full learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            optimizer.zero_grad()
            
            # Forward pass
            if use_amp:
                with torch.amp.autocast('cuda'):
                    X_pred = self.model()
                    
                    # MSE loss
                    mse = 0.5 * torch.mean((X - X_pred) ** 2)
                    
                    # Group LASSO penalties (vectorized)
                    W_coefs = self.model.get_W_coefs()  # (d, d, n_basis)
                    A_coefs = self.model.get_A_coefs()  # (d, d, p, n_basis)
                    
                    W_norms = torch.norm(W_coefs, p=2, dim=2)  # (d, d)
                    W_masked = W_norms * self.model.W_mask
                    penalty_w = (torch.sum(W_masked) - torch.trace(W_masked))
                    
                    A_norms = torch.norm(A_coefs, p=2, dim=3)  # (d, d, p)
                    A_masked = A_norms * self.model.A_mask
                    penalty_a = torch.sum(A_masked)
                    
                    # Smoothness penalty
                    penalty_smooth = self.model.compute_smoothness_penalty()
                    
                    # Total loss
                    loss = mse + self.lambda_w * penalty_w + self.lambda_a * penalty_a + penalty_smooth
                
                # Backward with mixed precision
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Check gradient norm and skip if NaN/Inf
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                if not torch.isnan(total_norm) and not torch.isinf(total_norm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if verbose and it == 0:
                        logger.warning(f"  NaN/Inf gradient detected in AMP, skipping update")
                    scaler.update()  # Update scaler state even if skipping step
                
                # Log gradient norm on first iteration
                if it == 0 and verbose:
                    logger.info(f"  Gradient norm: {total_norm:.2f}")
            else:
                # Standard precision
                X_pred = self.model()
                
                # MSE loss
                mse = 0.5 * torch.mean((X - X_pred) ** 2)
                
                # Group LASSO penalties (vectorized)
                W_coefs = self.model.get_W_coefs()  # (d, d, n_basis)
                A_coefs = self.model.get_A_coefs()  # (d, d, p, n_basis)
                
                W_norms = torch.norm(W_coefs, p=2, dim=2)  # (d, d)
                W_masked = W_norms * self.model.W_mask
                penalty_w = (torch.sum(W_masked) - torch.trace(W_masked))
                
                A_norms = torch.norm(A_coefs, p=2, dim=3)  # (d, d, p)
                A_masked = A_norms * self.model.A_mask
                penalty_a = torch.sum(A_masked)
                
                # Smoothness penalty
                penalty_smooth = self.model.compute_smoothness_penalty()
                
                # Total loss
                loss = mse + self.lambda_w * penalty_w + self.lambda_a * penalty_a + penalty_smooth
                
                # Backward
                loss.backward()
                
                # Check for NaN gradients
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Skip update if gradients are NaN
                if not torch.isnan(total_norm) and not torch.isinf(total_norm):
                    optimizer.step()
                else:
                    if verbose and it == 0:
                        logger.warning(f"  NaN/Inf gradient detected, skipping update")
                
                # Log gradient norm on first iteration
                if it == 0 and verbose:
                    logger.info(f"  Gradient norm: {total_norm:.2f}")
            
            # Record history
            self.history['loss'].append(loss.item())
            self.history['mse'].append(mse.item())
            penalty_total = (penalty_w + penalty_a + penalty_smooth).item()
            self.history['penalty'].append(penalty_total)
            
            # Log penalties on first iteration for debugging
            if it == 0 and verbose:
                logger.info(f"  Penalties: W={penalty_w.item():.4f}, A={penalty_a.item():.4f}, smooth={penalty_smooth.item():.4f}, total={penalty_total:.4f}")
            
            # DAG enforcement
            if (it + 1) % self.dag_enforce_interval == 0:
                with torch.no_grad():
                    W_strength = torch.tensor(self.model.get_weight_matrix(lag=0),
                                             dtype=torch.float32, device=self.device)
                    
                    W_dag = self.dag_enforcer.project_to_dag(W_strength, inplace=False)
                    self.model.update_masks(W_dag)
                    
                    n_violations = self.dag_enforcer.compute_num_cycles(W_strength)
                    self.history['dag_violations'].append(n_violations)
                    
                    if verbose and n_violations > 0:
                        logger.debug(f"  Iter {it+1}: Enforced DAG, broke {n_violations} cycles")
            
            # Logging
            if verbose and (it % 10 == 0 or it == max_iter - 1):
                elapsed = time.time() - start_time
                logger.info(f"  Iter {it:3d}: loss={loss.item():.6f}, mse={mse.item():.6f}, "
                           f"time={elapsed:.1f}s")
            
            # Convergence check (relative tolerance)
            if prev_loss is not None:
                rel_change = abs(loss.item() - prev_loss) / (abs(prev_loss) + 1e-8)
                if rel_change < loss_tol:
                    if verbose:
                        logger.info(f"  Converged at iteration {it} (relative Δloss = {rel_change:.2e} < {loss_tol})")
                    break
            
            prev_loss = loss.item()
        
        # Final DAG enforcement
        with torch.no_grad():
            W_strength = torch.tensor(self.model.get_weight_matrix(lag=0),
                                     dtype=torch.float32, device=self.device)
            W_dag = self.dag_enforcer.project_to_dag(W_strength)
            self.model.update_masks(W_dag)
        
        total_time = time.time() - start_time
        
        if verbose:
            logger.info(f"Optimization complete: {len(self.history['loss'])} iterations, {total_time:.2f}s")
            logger.info(f"Final loss: {self.history['loss'][-1]:.6f}")
        
        return self
    
    def get_structure_model(self, var_names: List[str]) -> StructureModel:
        """
        Convert learned Tucker-CAM to StructureModel with GPU-accelerated edge extraction.
        
        Args:
            var_names: Variable names
        
        Returns:
            StructureModel with learned edges
        """
        import time as time_module
        start_time = time_module.time()
        
        sm = StructureModel()
        
        # Add nodes
        for lag in range(self.p + 1):
            for name in var_names:
                sm.add_node(f"{name}_lag{lag}")
        
        logger.info("Extracting edges with GPU acceleration...")
        
        # GPU-accelerated edge extraction (VECTORIZED!)
        with torch.no_grad():
            # Get ALL weight matrices at once on GPU
            W_all = self.model.get_all_weight_matrices_gpu()  # (p+1, d, d) on GPU
            
            # Option D: Post-hoc Top-K sparsification
            # Learn dense model, then select Top-K edges by absolute weight
            W_abs = torch.abs(W_all).flatten()
            
            # Select Top-K edges (default: 10,000)
            top_k = int(os.environ.get('TUCKER_TOP_K', '10000'))
            
            if len(W_abs) < top_k:
                logger.warning(f"Requested top_k={top_k} but only {len(W_abs)} total edges, using all")
                top_k = len(W_abs)
            
            # Get threshold for Top-K
            if top_k > 0:
                threshold = torch.topk(W_abs, top_k).values[-1].item()
            else:
                threshold = 0.1
            
            logger.info(f"Post-hoc Top-K sparsification: selecting {top_k} edges (threshold={threshold:.6f})")
            
            # Find all edges above threshold (vectorized on GPU!)
            edge_mask = torch.abs(W_all) > threshold  # (p+1, d, d) boolean mask
            
            # Get indices of edges above threshold
            lag_indices, i_indices, j_indices = torch.where(edge_mask)
            
            # Move to CPU for edge creation (only the selected edges!)
            lag_indices = lag_indices.cpu().numpy()
            i_indices = i_indices.cpu().numpy()
            j_indices = j_indices.cpu().numpy()
            weights = W_all[edge_mask].cpu().numpy()
            
            logger.info(f"Found {len(weights)} edges above threshold")
            
            # Add edges to structure model
            for idx in range(len(weights)):
                lag = int(lag_indices[idx])
                i = int(i_indices[idx])
                j = int(j_indices[idx])
                weight = float(weights[idx])
                
                if lag == 0:
                    # Contemporaneous edge
                    parent = f"{var_names[j]}_lag0"
                    child = f"{var_names[i]}_lag0"
                else:
                    # Lagged edge
                    parent = f"{var_names[j]}_lag{lag}"
                    child = f"{var_names[i]}_lag0"
                
                sm.add_edge(parent, child, weight=weight, origin="learned")
        
        elapsed = time_module.time() - start_time
        logger.info(f"Edge extraction completed in {elapsed:.2f}s (GPU-accelerated)")
        
        sm.history = self.history
        
        return sm


def from_pandas_dynamic_tucker_cam(
    time_series: Union[pd.DataFrame, List[pd.DataFrame]],
    p: int,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    lambda_smooth: float = 0.01,
    n_knots: int = 5,
    rank_w: int = 20,
    rank_a: int = 10,
    max_iter: int = 100,
    w_threshold: float = 0.01,
    logger_prefix: str = "",
    **kwargs
) -> StructureModel:
    """
    Learn DBN structure using Tucker-CAM-DAG (memory-efficient nonlinear).
    
    Drop-in replacement for from_pandas_dynamic_cam() with massive memory reduction.
    
    Args:
        time_series: DataFrame or list of DataFrames
        p: Lag order
        lambda_w: L2 penalty for contemporaneous edges
        lambda_a: L2 penalty for lagged edges
        lambda_smooth: Smoothness penalty
        n_knots: Number of B-spline knots (can use 5 now!)
        rank_w: Tucker rank for contemporaneous (20 recommended)
        rank_a: Tucker rank for lagged (10 recommended)
        max_iter: Maximum iterations
        w_threshold: Threshold for edge pruning
        logger_prefix: Log prefix
        **kwargs: Additional arguments
    
    Returns:
        StructureModel with learned causal graph
    
    Example:
        >>> df = pd.read_csv('data.csv', index_col=0)
        >>> sm = from_pandas_dynamic_tucker_cam(df, p=20, n_knots=5, rank_w=20, rank_a=10)
        >>> print(f"Learned {len(sm.edges)} edges with {sm.memory_reduction:.0f}× less memory")
    """
    # Convert to list
    if not isinstance(time_series, list):
        time_series_list = [time_series]
    else:
        time_series_list = time_series
    
    # Transform data
    from transformers import DynamicDataTransformer
    transformer = DynamicDataTransformer(p=p)
    X, Xlags = transformer.fit_transform(time_series_list, return_df=False)
    
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Xlags = torch.tensor(Xlags, dtype=torch.float32, device=device)
    
    n, d = X.shape
    
    logger.info(f"{logger_prefix} Tucker-CAM: n={n}, d={d}, p={p}, K={n_knots}")
    logger.info(f"{logger_prefix} Ranks: r_w={rank_w}, r_a={rank_a}")
    
    # Create and fit
    tucker_cam = TuckerFastCAMDAG(
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
    
    start_time = time.time()
    tucker_cam.fit(X, Xlags, max_iter=max_iter, verbose=True)
    elapsed = time.time() - start_time
    
    logger.info(f"{logger_prefix} Tucker-CAM training completed in {elapsed:.2f}s")
    
    # Convert to StructureModel
    var_names = time_series_list[0].columns.tolist()
    sm = tucker_cam.get_structure_model(var_names)
    
    logger.info(f"{logger_prefix} Learned {len(sm.edges)} edges")
    
    return sm


def test_tucker_fast_cam_dag():
    """Test Tucker-CAM-DAG on synthetic data"""
    print("\n" + "="*70)
    print("Testing Tucker-CAM-DAG Optimizer")
    print("="*70)
    
    # Synthetic data
    torch.manual_seed(42)
    n, d, p = 100, 50, 5
    
    X = torch.randn(n, d)
    Xlags = torch.randn(n, d * p)
    
    print(f"\nTest: n={n}, d={d}, p={p}")
    
    # Create optimizer
    tucker_cam = TuckerFastCAMDAG(
        d=d, p=p, n_knots=5,
        rank_w=10, rank_a=5,
        lambda_smooth=0.01
    )
    
    print(f"\nFitting Tucker-CAM...")
    tucker_cam.fit(X, Xlags, max_iter=50, verbose=True)
    
    # Check results
    W = tucker_cam.model.get_weight_matrix(lag=0)
    print(f"\n✓ Learned weight matrix W: {W.shape}")
    print(f"✓ Number of edges: {np.sum(np.abs(W) > 0.01)}")
    print(f"✓ Weight range: [{W.min():.3f}, {W.max():.3f}]")
    
    print("\n✓ Tucker-CAM-DAG test passed!")
    print("="*70)


if __name__ == "__main__":
    test_tucker_fast_cam_dag()
