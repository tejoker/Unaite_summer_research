import logging
import time
import os
import pickle
from typing import Union, List, Tuple, Dict
import numpy as np
import pandas as pd
import torch



from structuremodel import StructureModel

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Determine computation device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --------------------------------------------------------------------------
# Main interface functions for DynoTears dynamic structure learning
# --------------------------------------------------------------------------

def from_pandas_dynamic(
    time_series: Union[pd.DataFrame, List[pd.DataFrame]],
    p: int,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    loss_tol: float = 1e-6,
    init_W: np.ndarray = None,
    init_A: np.ndarray = None,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[int, int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    # Logging and checkpoint options
    save_csv_path: str = None,
    save_every: int = None,
    checkpoint_path: str = None,
    logger_prefix: str = "",
) -> StructureModel:
    """
    Learn a Dynamic Bayesian Network (DBN) structure from time series data (pandas DataFrame(s))
    using the DynoTears algorithm.

    This function accepts one or a list of pandas DataFrames representing time series data.
    It constructs the design matrices for current values and lagged values up to order `p`,
    then optimizes an augmented Lagrangian to learn intra-slice (contemporaneous) and inter-slice (temporal lag) relationships.

    Key features:
    - Periodic checkpointing of the optimizer state (every `save_every` iterations, if specified).
    - Optionally saves weight matrices to CSV at checkpoints.
    - Resumes training from a checkpoint file if available, allowing graceful recovery from interruptions.
    - Uses GPU acceleration if available.

    Args:
        time_series: A pandas DataFrame or list of DataFrames containing the time series data.
            Each DataFrame should have shape (n_samples, n_variables) and a datetime or integer index.
        p: The number of lag time steps to consider (lag order).
        lambda_w: L1 regularization strength for intra-slice weights (W).
        lambda_a: L1 regularization strength for inter-slice weights (A).
        max_iter: Maximum number of optimization iterations.
        h_tol: Tolerance for the acyclicity condition (convergence criterion on acyclicity).
        loss_tol: Tolerance for change in loss (convergence criterion on loss improvement).
        init_W: Optional initial value for W (numpy array of shape [d, d]).
        init_A: Optional initial value for A (numpy array of shape [d, d*p]).
        w_threshold: Threshold below which learned weights are set to zero (for sparsity in output).
        tabu_edges: List of forbidden edges specified as tuples (parent_index, child_index, lag).
            Edges with these parent-child relationships will be constrained to zero weight.
            Use lag=0 for contemporaneous edges, lag>0 for lagged edges.
        tabu_parent_nodes: List of variable indices that are not allowed to be parent nodes (no outgoing edges from these).
        tabu_child_nodes: List of variable indices that are not allowed to be child nodes (no incoming edges into these).
        save_csv_path: Path to a CSV file where weight matrices will be periodically saved (appended) at checkpoints.
        save_every: Interval (in iterations) for saving CSV and checkpoint. For example, 10 means save every 10 iterations.
        checkpoint_path: Path to a pickle file to save checkpoints. If this file exists at the start, the optimizer will resume from it.
        logger_prefix: Prefix string for log messages (useful to distinguish multiple concurrent runs).

    Returns:
        A StructureModel object representing the learned dynamic structure. The nodes of the StructureModel are named as 
        "{variable_name}_lagL" for L = 0,...,p. Edges have attribute "weight" for the learned coefficient, and origin="learned".
        The StructureModel also contains a history dictionary with at least 'loss' (list of loss values per iteration).
    """
    # Ensure the input is a list of DataFrames
    if not isinstance(time_series, list):
        time_series_list = [time_series]
    else:
        time_series_list = time_series
    # Convert time series DataFrame(s) to numpy arrays (with index validation)
    X, Xlags = _to_numpy_dynamic(time_series_list, p)
    # If a checkpoint exists, load it to resume optimization
    resume_state = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                resume_state = pickle.load(f)
            logger.info(f"{logger_prefix} Resuming from {checkpoint_path} (iteration {resume_state.get('it')})")
        except Exception as e:
            logger.warning(f"{logger_prefix} Could not load checkpoint {checkpoint_path}: {e}")
    # Learn structure using numpy arrays (calls internal optimization)
    start_time = time.time()
    sm_core = from_numpy_dynamic(
        X=X,
        Xlags=Xlags,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        h_tol=h_tol,
        loss_tol=loss_tol,
        init_W=init_W,
        init_A=init_A,
        w_threshold=w_threshold,
        tabu_edges=tabu_edges,
        tabu_parent_nodes=tabu_parent_nodes,
        tabu_child_nodes=tabu_child_nodes,
        save_csv_path=save_csv_path,
        save_every=save_every,
        checkpoint_path=checkpoint_path,
        logger_prefix=logger_prefix,
        resume_state=resume_state
    )
    elapsed = time.time() - start_time
    logger.info(f"{logger_prefix} Structure learning completed in {elapsed:.2f}s. Final loss = {sm_core.history['loss'][-1]:.6f}")
    # Map numeric node labels back to original variable names for user-friendly output
    var_names = time_series_list[0].columns.tolist()
    idx_to_name = {idx: name for idx, name in enumerate(var_names)}
    sm_formatted = StructureModel()
    # Add all nodes with proper names
    for l in range(p + 1):
        for idx, name in enumerate(var_names):
            sm_formatted.add_node(f"{name}_lag{l}")
    # Add edges with remapped node names
    for u, v, w in sm_core.edges.data("weight"):
        # u and v are in the format "{i}_lag{l}"
        u_idx_str, u_lag = u.split("_lag")
        v_idx_str, v_lag = v.split("_lag")
        u_idx = int(u_idx_str); v_idx = int(v_idx_str)
        new_u = f"{idx_to_name.get(u_idx, u_idx)}_lag{u_lag}"
        new_v = f"{idx_to_name.get(v_idx, v_idx)}_lag{v_lag}"
        sm_formatted.add_edge(new_u, new_v, weight=w, origin="learned")
    # Attach history (e.g., loss trajectory) to output StructureModel
    sm_formatted.history = getattr(sm_core, "history", {})
    return sm_formatted

def _to_numpy_dynamic(time_series_list: List[pd.DataFrame], p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of pandas DataFrame time series into numpy arrays for current values (X) and lagged values (Xlags).

    Ensures that each DataFrame index is sorted in ascending order, contains no duplicates, and is strictly increasing.
    If not, the DataFrame is sorted by index and any duplicate timestamps are removed; the index is then reset to a continuous range.

    Args:
        time_series_list: List of pandas DataFrames, each representing a time series of shape (n_samples, n_vars).
        p: The number of lag steps to include.

    Returns:
        A tuple (X, Xlags):
        - X: numpy array of shape (N, D) containing the current values for each time step (after the first p observations).
        - Xlags: numpy array of shape (N, D*p) containing the lagged values for each time step (concatenated for all lags 1..p).
        Here N = total number of output time steps across all input series (taking into account p lags), and D = number of variables.
    """
    # Validate and prepare each DataFrame in the list
    for i, df in enumerate(time_series_list):
        # Check if index is sorted and unique
        if not df.index.is_monotonic_increasing or df.index.has_duplicates:
            logger.warning("Time series index of DataFrame %d is not strictly increasing or contains duplicates. Sorting and resetting index." % i)
            df_fixed = df.sort_index()
            # Remove duplicate index entries if any, keeping first occurrence
            df_fixed = df_fixed[~df_fixed.index.duplicated(keep='first')]
            # Reset to integer index (drop original index)
            df_fixed = df_fixed.reset_index(drop=True)
            time_series_list[i] = df_fixed
    # Use an external transformer utility to build design matrices for dynamic data
    from transformers import DynamicDataTransformer
    transformer = DynamicDataTransformer(p=p)
    X, Xlags = transformer.fit_transform(time_series_list, return_df=False)
    X = np.asarray(X, dtype=np.float32)
    Xlags = np.asarray(Xlags, dtype=np.float32)
    
    # Critical validation: Check for NaN/Inf values that would cause optimization failure
    if not np.isfinite(X).all():
        n_invalid = (~np.isfinite(X)).sum()
        logger.warning(f"X contains {n_invalid} NaN/Inf values. Replacing with 0.0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if not np.isfinite(Xlags).all():
        n_invalid = (~np.isfinite(Xlags)).sum()
        logger.warning(f"Xlags contains {n_invalid} NaN/Inf values. Replacing with 0.0")
        Xlags = np.nan_to_num(Xlags, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, Xlags

def from_numpy_dynamic(
    X: np.ndarray,
    Xlags: np.ndarray,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    loss_tol: float = 1e-6,
    init_W: np.ndarray = None,
    init_A: np.ndarray = None,
    w_threshold: float = 0.0,
    tabu_edges: List[Tuple[int, int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    save_csv_path: str = None,
    save_every: int = None,
    checkpoint_path: str = None,
    logger_prefix: str = "",
    resume_state: dict = None,
) -> StructureModel:
    """
    Learn a DBN structure from pre-formatted numpy arrays for data and lagged data.

    This function is similar to from_pandas_dynamic, but directly takes in the numpy arrays for current and lagged values.
    It runs the optimization to learn weight matrices W and A. Use this if you have already prepared X and Xlags.

    Args:
        X: N x D numpy array of current-time observations.
        Xlags: N x (D*p) numpy array of lagged observations (concatenated for lags 1..p).
        lambda_w: L1 regularization strength for W (intra-slice).
        lambda_a: L1 regularization strength for A (inter-slice).
        max_iter: Maximum number of iterations for the optimizer.
        h_tol: Convergence tolerance for acyclicity (constraint violation).
        loss_tol: Convergence tolerance for loss improvement between iterations.
        init_W: Optional initial W matrix (numpy array shape [D, D]).
        init_A: Optional initial A matrix (numpy array shape [D, D*p]).
        w_threshold: Threshold to zero out small weights after optimization.
        tabu_edges: List of forbidden edges (parent_index, child_index, lag) to keep weight at 0.
        tabu_parent_nodes: List of variable indices not allowed to be parents.
        tabu_child_nodes: List of variable indices not allowed to be children.
        save_csv_path: If provided, path to CSV file for saving weight matrices at checkpoints.
        save_every: Save checkpoint and CSV every this many iterations.
        checkpoint_path: If provided, path to a pickle file to save/load optimizer state for resuming.
        logger_prefix: Prefix for log messages (for identifying concurrent runs).
        resume_state: If provided, a dict containing state from a previous run to resume (typically loaded from checkpoint).

    Returns:
        A StructureModel representing the learned graph. Node naming uses numeric indices for variables (e.g., "0_lag0"),
        and edges have learned weights. The history attribute contains the loss progression.
    """
    # Transfer data to PyTorch tensors on the chosen device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Xlags_t = torch.tensor(Xlags, dtype=torch.float32, device=device)
    n_samples, d_vars = X_t.shape
    # Validate dimensions
    if n_samples == 0 or Xlags_t.shape[0] == 0:
        raise ValueError("Input data X or Xlags is empty, cannot learn structure.")
    if Xlags_t.shape[0] != n_samples:
        raise ValueError("X and Xlags must have the same number of rows.")
    if Xlags_t.shape[1] % d_vars != 0:
        raise ValueError("Number of columns in Xlags must be D * p (a multiple of number of variables in X).")
    p_orders = Xlags_t.shape[1] // d_vars
    logger.info(f"{logger_prefix} Starting optimization: n_samples={n_samples}, n_vars={d_vars}, lag_order={p_orders}")
    # If resume state not explicitly provided, check for checkpoint file
    if resume_state is None and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                resume_state = pickle.load(f)
            logger.info(f"{logger_prefix} Loaded checkpoint from {checkpoint_path} (iteration {resume_state.get('it')})")
        except Exception as e:
            logger.warning(f"{logger_prefix} Could not load checkpoint {checkpoint_path}: {e}")
            resume_state = None
    # Run the core learning algorithm
    W_final, A_final, loss_history = _learn_dynamic_structure(
        X_t,
        Xlags_t,
        lambda_w=lambda_w,
        lambda_a=lambda_a,
        max_iter=max_iter,
        h_tol=h_tol,
        loss_tol=loss_tol,
        init_W=init_W,
        init_A=init_A,
        tabu_edges=tabu_edges,
        tabu_parent_nodes=tabu_parent_nodes,
        tabu_child_nodes=tabu_child_nodes,
        save_csv_path=save_csv_path,
        save_every=save_every,
        checkpoint_path=checkpoint_path,
        resume_state=resume_state,
        logger_prefix=logger_prefix
    )
    logger.info(f"{logger_prefix} Optimization finished; final loss = {loss_history[-1]:.6f}")
    # Apply threshold to small weights for sparsity
    with torch.no_grad():
        if w_threshold > 0:
            W_final[torch.abs(W_final) < w_threshold] = 0.0
            A_final[torch.abs(A_final) < w_threshold] = 0.0
    # Convert final weights to numpy (on CPU) for building output graph
    W_np = W_final.detach().cpu().numpy()
    A_np = A_final.detach().cpu().numpy()  # shape [D, D, p]
    # Combine A matrices for all lags into one 2D matrix of shape [D*p, D]
    A_mat = np.concatenate([A_np[:, :, k] for k in range(p_orders)], axis=0)
    # Construct a StructureModel from W and A
    sm = _matrices_to_structure_model(W_np, A_mat)
    sm.history = {"loss": loss_history}
    return sm

# --------------------------------------------------------------------------
# Internal helper functions for model construction and optimization
# --------------------------------------------------------------------------

def _matrices_to_structure_model(w_est: np.ndarray, a_est: np.ndarray) -> StructureModel:
    """
    Build a StructureModel graph from weight matrices.

    This creates a StructureModel with nodes labeled as numeric indices (as strings) with lag suffix.
    For example, for 3 variables and p=2, nodes will be "0_lag0", "1_lag0", "2_lag0", "0_lag1", "1_lag1", "2_lag1".
    Intra-slice edges (lag 0) are added for each nonzero entry in w_est, and inter-slice edges are added for each nonzero entry in a_est.

    Args:
        w_est: 2D numpy array of shape [D, D] for intra-slice (contemporaneous) weights.
        a_est: 2D numpy array of shape [D*p, D] for inter-slice (lagged) weights.

    Returns:
        StructureModel with the corresponding nodes and edges.
    """
    sm = StructureModel()
    d = w_est.shape[0]
    p = a_est.shape[0] // d
    # Add nodes for lag 0,...,p
    for lag in range(p + 1):
        for i in range(d):
            sm.add_node(f"{i}_lag{lag}")
    # Add intra-slice edges (lag 0)
    for i in range(d):
        for j in range(d):
            w = w_est[i, j]
            if w != 0:
                sm.add_edge(f"{i}_lag0", f"{j}_lag0", weight=float(w), origin="learned")
    # Add inter-slice edges for each lag
    for lag in range(1, p + 1):
        for i in range(d):
            for j in range(d):
                w = a_est[(lag - 1) * d + i, j]
                if w != 0:
                    sm.add_edge(f"{i}_lag{lag}", f"{j}_lag0", weight=float(w), origin="learned")
    return sm

def _learn_dynamic_structure(
    X: torch.Tensor,
    Xlags: torch.Tensor,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    loss_tol: float = 1e-6,
    init_W: np.ndarray = None,
    init_A: np.ndarray = None,
    tabu_edges: List[Tuple[int, int, int]] = None,
    tabu_parent_nodes: List[int] = None,
    tabu_child_nodes: List[int] = None,
    save_csv_path: str = None,
    save_every: int = None,
    checkpoint_path: str = None,
    resume_state: dict = None,
    logger_prefix: str = ""
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Core optimization routine to learn W and A using augmented Lagrangian method.

    This function iteratively optimizes the weight matrices W (intra-slice) and A (inter-slice) to minimize
    reconstruction error while enforcing an acyclicity constraint on W (no cycles among contemporaneous edges).
    It uses a batched approach for large data, and updates dual variables to enforce acyclicity (through alpha and rho).

    Args:
        X: Torch tensor of shape (N, D) for current observations.
        Xlags: Torch tensor of shape (N, D*p) for lagged observations.
        lambda_w: L1 penalty coefficient for W.
        lambda_a: L1 penalty coefficient for A.
        max_iter: Maximum number of outer iterations (updates of W and A).
        h_tol: Tolerance for the acyclicity constraint (threshold for trace(expm(W^2)) - D).
        loss_tol: Tolerance for change in loss for convergence.
        init_W: Initial W weights (numpy array shape [D, D]) or None to initialize small random.
        init_A: Initial A weights (numpy array shape [D, D*p]) or None to initialize small random.
        tabu_edges: List of forbidden edges (parent_index, child_index, lag) to constrain to zero weight.
        tabu_parent_nodes: List of indices of variables forbidden to be parents.
        tabu_child_nodes: List of indices of variables forbidden to be children.
        save_csv_path: Path to CSV file for saving weight matrices at checkpoints (if provided).
        save_every: Interval (iterations) to save CSV and checkpoint.
        checkpoint_path: Path to pickle file for saving checkpoint states.
        resume_state: Dictionary containing previous state (W, A, alpha, rho, it, optimizer_state) for resuming optimization.
        logger_prefix: Prefix string for log messages (to identify this run context).

    Returns:
        A tuple (W_final, A_final, loss_history):
        - W_final: Torch tensor of shape [D, D] containing the learned intra-slice weights.
        - A_final: Torch tensor of shape [D, D, p] containing the learned inter-slice weights.
        - loss_history: List of loss values (float) for each iteration.
    """
    import csv
    # Unpack dimensions
    n_samples, d = X.shape
    p_orders = Xlags.shape[1] // d
    # Determine batch size to roughly cap memory usage per batch (~50 MB of float32 data)
    bytes_per_row = (d + d * p_orders) * 4  # 4 bytes per float32
    target_bytes = 50 * 1024 * 1024  # 50 MB
    batch_size = max(1, int(target_bytes // bytes_per_row))
    batch_size = min(batch_size, n_samples)
    logger.info(f"{logger_prefix} Using batch_size={batch_size} for {n_samples} samples")
    # Initialize W and A parameters
    if init_W is not None:
        W = torch.tensor(init_W, dtype=torch.float32, device=device, requires_grad=True)
    else:
        W = torch.full((d, d), 1e-3, dtype=torch.float32, device=device, requires_grad=True)
    if init_A is not None:
        # reshape init_A to [D, D, p] if given as [D, D*p]
        try:
            A_init = init_A.reshape(d, d, p_orders)
        except Exception as e:
            raise ValueError(f"init_A shape {init_A.shape} is incompatible with d={d}, p={p_orders}") from e
        A = torch.tensor(A_init, dtype=torch.float32, device=device, requires_grad=True)
    else:
        A = torch.full((d, d, p_orders), 1e-3, dtype=torch.float32, device=device, requires_grad=True)
    # Initialize dual variables for acyclicity constraint
    alpha = 0.0
    rho = 1.0
    start_iter = 0
    # If resuming from previous state, load values
    if resume_state is not None:
        # Check for shape consistency before loading
        W_state = resume_state.get("W")
        A_state = resume_state.get("A")
        if W_state is None or A_state is None:
            logger.warning(f"{logger_prefix} Resume state missing W or A matrices, ignoring resume_state")
        elif W_state.shape != (d, d) or A_state.shape != (d, d, p_orders):
            logger.warning(f"{logger_prefix} Checkpoint dimensions do not match current data (skip resume).")
        else:
            with torch.no_grad():
                W.copy_(torch.from_numpy(W_state).to(device))
                A.copy_(torch.from_numpy(A_state).to(device))
            alpha = resume_state.get("alpha", alpha)
            rho = resume_state.get("rho", rho)
            start_iter = resume_state.get("it", -1) + 1
            logger.info(f"{logger_prefix} Resumed state loaded: starting at iteration {start_iter}")
    
    # Adaptive learning rate based on problem size
    # For large d, use much smaller learning rate to prevent divergence
    if d < 20:
        base_lr = 0.05
    elif d < 100:
        base_lr = 0.01
    elif d < 500:
        base_lr = 0.001
    else:
        # For very large problems (like 2889 variables), use tiny learning rate
        base_lr = 0.0001
    
    logger.info(f"{logger_prefix} Using adaptive learning rate: {base_lr} (d={d})")
    
    # Set up optimizer
    optimizer = torch.optim.Adam([W, A], lr=base_lr)
    # If optimizer state was saved, restore it
    if resume_state is not None and "optimizer_state" in resume_state:
        opt_state = resume_state["optimizer_state"]
        # Ensure the optimizer's internal state exists for W and A
        optimizer.state[W] = optimizer.state.get(W, {})
        optimizer.state[A] = optimizer.state.get(A, {})
        if "exp_avg_W" in opt_state:
            optimizer.state[W]["exp_avg"] = torch.tensor(opt_state["exp_avg_W"], dtype=torch.float32, device=device)
            optimizer.state[W]["exp_avg_sq"] = torch.tensor(opt_state["exp_avg_sq_W"], dtype=torch.float32, device=device)
            optimizer.state[W]["step"] = int(opt_state.get("step_W", 0))
        if "exp_avg_A" in opt_state:
            optimizer.state[A]["exp_avg"] = torch.tensor(opt_state["exp_avg_A"], dtype=torch.float32, device=device)
            optimizer.state[A]["exp_avg_sq"] = torch.tensor(opt_state["exp_avg_sq_A"], dtype=torch.float32, device=device)
            optimizer.state[A]["step"] = int(opt_state.get("step_A", 0))
        logger.info(f"{logger_prefix} Optimizer state restored from checkpoint")
    # Prepare CSV file headers if needed
    if save_csv_path is not None and not os.path.exists(save_csv_path):
        with open(save_csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["timestamp", "tag", "iter", "matrix", "i", "j", "lag", "value"])
    loss_history: List[float] = []
    prev_loss = None
    # Iterate until max_iter or convergence
    for it in range(start_iter, max_iter):
        optimizer.zero_grad()
        # Compute residual in batches to handle large N
        I = torch.eye(d, device=device)
        total_loss_mse = 0.0
        for s in range(0, n_samples, batch_size):
            e = min(s + batch_size, n_samples)
            X_batch = X[s:e]
            Xlags_batch = Xlags[s:e]
            # Compute residual: (I - W) * X_t - A_flat * Xlags_t
            A_flat = A.permute(2, 0, 1).reshape(d * p_orders, d)
            residual = X_batch.matmul(I - W) - Xlags_batch.matmul(A_flat)
            total_loss_mse += (residual.norm(p='fro') ** 2)
        # MSE loss
        loss_mse = 0.5 / n_samples * total_loss_mse
        # Acyclicity constraint value h(W) = trace(exp(W*W)) - D
        # Use numerical stability improvements for matrix exponential
        W_squared = W.matmul(W)
        W_squared_norm = torch.norm(W_squared, p='fro')
        
        # If the matrix norm is too large, use approximation to avoid overflow
        if W_squared_norm > 10.0:  # Threshold to prevent exp() overflow
            logger.debug(f"{logger_prefix} Large W norm ({W_squared_norm:.3f}), using polynomial approximation")
            # Use polynomial approximation: exp(X) ≈ I + X + X²/2 + X³/6 for numerical stability
            I = torch.eye(d, device=W.device)
            W2 = W_squared
            W3 = W2.matmul(W_squared)
            exp_approx = I + W_squared + 0.5 * W2.matmul(W_squared) + (1.0/6.0) * W3
            h_val_tensor = torch.trace(exp_approx) - d
        else:
            try:
                h_val_tensor = torch.trace(torch.linalg.matrix_exp(W_squared)) - d
            except Exception as e:
                logger.warning(f"{logger_prefix} Matrix exponential failed, using quadratic approximation: {e}")
                # Fallback to simpler quadratic approximation
                h_val_tensor = torch.trace(W_squared) + 0.5 * torch.trace(W_squared.matmul(W_squared))
        # L1 regularization penalty
        l1_penalty = lambda_w * torch.abs(W).sum() + lambda_a * torch.abs(A).sum()
        # Total loss = MSE + acyclicity penalty + L1
        loss = loss_mse + 0.5 * rho * (h_val_tensor ** 2) + alpha * h_val_tensor + l1_penalty
        # Backpropagate
        loss.backward()
        
        # Gradient clipping to prevent explosion (critical for large d)
        torch.nn.utils.clip_grad_norm_([W, A], max_norm=1.0)
        
        # Enforce taboo edges by zeroing gradients for those weights
        if tabu_edges or tabu_parent_nodes or tabu_child_nodes:
            with torch.no_grad():
                # Tabu edges
                if tabu_edges:
                    for (pi, ci, lag) in tabu_edges:
                        if lag == 0:
                            if 0 <= pi < d and 0 <= ci < d:
                                W.grad[pi, ci] = 0.0
                        elif lag > 0 and lag <= p_orders:
                            if 0 <= pi < d and 0 <= ci < d:
                                A.grad[pi, ci, lag - 1] = 0.0
                # Tabu parent nodes: zero grad for all edges where parent index is in list
                if tabu_parent_nodes:
                    for pi in tabu_parent_nodes:
                        if 0 <= pi < d:
                            # no outgoing from pi -> zero gradients for W[pi,*] and A[pi,*,*]
                            W.grad[pi, :] = 0.0
                            if p_orders > 0:
                                A.grad[pi, :, :] = 0.0
                # Tabu child nodes: zero grad for all edges where child index is in list
                if tabu_child_nodes:
                    for ci in tabu_child_nodes:
                        if 0 <= ci < d:
                            # no incoming to ci -> zero gradients for W[:,ci] and A[:,ci,*]
                            W.grad[:, ci] = 0.0
                            if p_orders > 0:
                                A.grad[:, ci, :] = 0.0
        # Gradient step
        optimizer.step()
        # No self-loops: enforce W diagonal = 0
        with torch.no_grad():
            W.fill_diagonal_(0.0)
        # Compute current values for logging and checks
        loss_val = float(loss.item())
        h_val = float(h_val_tensor.item())
        loss_history.append(loss_val)
        # Log outer iteration metrics
        logger.info(f"{logger_prefix} Iteration {it}: loss={loss_val:.6f}, h={h_val:.3e}, rho={rho:.1e}, alpha={alpha:.2e}")
        if device.type == "cuda":
            logger.debug(f"{logger_prefix} GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        # Save periodic checkpoint and weights if required
        if save_every is not None and save_every > 0 and (it % save_every == 0):
            # Save weights to CSV
            if save_csv_path:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(save_csv_path, "a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    tag = "periodic"
                    for i in range(d):
                        for j in range(d):
                            writer.writerow([ts_str, tag, it, "W", i, j, 0, float(W[i, j].item())])
                    for lag in range(1, p_orders + 1):
                        for i in range(d):
                            for j in range(d):
                                writer.writerow([ts_str, tag, it, "A", i, j, lag, float(A[i, j, lag-1].item())])
            # Save binary checkpoint (with optimizer state)
            if checkpoint_path:
                ckpt = {
                    "W": W.detach().cpu().numpy(),
                    "A": W.detach().new_tensor(A.detach()).cpu().numpy(),  # A is 3D
                    "alpha": alpha,
                    "rho": rho,
                    "it": it,
                    "optimizer_state": {}
                }
                # Store optimizer state (for Adam: exp_avg, exp_avg_sq, step for each param)
                if W in optimizer.state and "exp_avg" in optimizer.state[W]:
                    ckpt["optimizer_state"]["exp_avg_W"] = optimizer.state[W]["exp_avg"].detach().cpu().numpy()
                    ckpt["optimizer_state"]["exp_avg_sq_W"] = optimizer.state[W]["exp_avg_sq"].detach().cpu().numpy()
                    ckpt["optimizer_state"]["step_W"] = optimizer.state[W].get("step", 0)
                if A in optimizer.state and "exp_avg" in optimizer.state[A]:
                    ckpt["optimizer_state"]["exp_avg_A"] = optimizer.state[A]["exp_avg"].detach().cpu().numpy()
                    ckpt["optimizer_state"]["exp_avg_sq_A"] = optimizer.state[A]["exp_avg_sq"].detach().cpu().numpy()
                    ckpt["optimizer_state"]["step_A"] = optimizer.state[A].get("step", 0)
                with open(checkpoint_path, "wb") as f_ckpt:
                    pickle.dump(ckpt, f_ckpt)
        # Check convergence conditions
        converged = abs(h_val) <= h_tol
        if prev_loss is not None:
            converged |= (abs(loss_val - prev_loss) <= loss_tol)
        if converged or it == max_iter - 1:
            # Final save and exit
            if save_csv_path:
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(save_csv_path, "a", newline="") as f_csv:
                    writer = csv.writer(f_csv)
                    tag = "final"
                    for i in range(d):
                        for j in range(d):
                            writer.writerow([ts_str, tag, it, "W", i, j, 0, float(W[i, j].item())])
                    for lag in range(1, p_orders + 1):
                        for i in range(d):
                            for j in range(d):
                                writer.writerow([ts_str, tag, it, "A", i, j, lag, float(A[i, j, lag-1].item())])
            if checkpoint_path:
                result_state = {
                    "W": W.detach().cpu().numpy(),
                    "A": W.detach().new_tensor(A.detach()).cpu().numpy(),
                    "alpha": alpha,
                    "rho": rho,
                    "it": it,
                    "optimizer_state": {}
                }
                if W in optimizer.state and "exp_avg" in optimizer.state[W]:
                    result_state["optimizer_state"]["exp_avg_W"] = optimizer.state[W]["exp_avg"].detach().cpu().numpy()
                    result_state["optimizer_state"]["exp_avg_sq_W"] = optimizer.state[W]["exp_avg_sq"].detach().cpu().numpy()
                    result_state["optimizer_state"]["step_W"] = optimizer.state[W].get("step", 0)
                if A in optimizer.state and "exp_avg" in optimizer.state[A]:
                    result_state["optimizer_state"]["exp_avg_A"] = optimizer.state[A]["exp_avg"].detach().cpu().numpy()
                    result_state["optimizer_state"]["exp_avg_sq_A"] = optimizer.state[A]["exp_avg_sq"].detach().cpu().numpy()
                    result_state["optimizer_state"]["step_A"] = optimizer.state[A].get("step", 0)
                with open(checkpoint_path, "wb") as f_ckpt:
                    pickle.dump(result_state, f_ckpt)
                logger.info(f"{logger_prefix} Saved final checkpoint to {checkpoint_path} at iteration {it}")
            logger.info(f"{logger_prefix} Converged at iteration {it}: h={h_val:.3e}, loss_change={0.0 if prev_loss is None else abs(loss_val - prev_loss):.3e}")
            return W.detach(), A.detach(), loss_history
        # Update dual variables for acyclicity
        alpha += rho * h_val
        rho = min(rho * 10, 1e6)
        prev_loss = loss_val
    # If loop finishes without convergence (should not normally happen due to return above)
    return W.detach(), A.detach(), loss_history

# --------------------------------------------------------------------------
# Helper functions to extract results and analyze learned weights
# --------------------------------------------------------------------------

def extract_matrices(sm: StructureModel, var_names: List[str], p: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the learned weight matrices from a StructureModel.

    Args:
        sm: StructureModel produced by this module's functions.
        var_names: List of original variable names (for ordering).
        p: Number of lags considered in the model.

    Returns:
        A tuple (W, A) where:
        - W is a torch.Tensor of shape [D, D] containing intra-slice weights.
        - A is a torch.Tensor of shape [D, D, p] containing inter-slice weights.
        (D = len(var_names))
    """
    D = len(var_names)
    dev = device  # use the same device as global (ensures consistency)
    W = torch.zeros((D, D), device=dev)
    A = torch.zeros((D, D, p), device=dev)
    for u, v, w in sm.edges.data("weight"):
        # Nodes u, v are labeled as "{node}_lag{L}"
        src_label, lag_u_str = u.rsplit("_lag", 1)
        tgt_label, lag_v_str = v.rsplit("_lag", 1)
        try:
            lag_u = int(lag_u_str)
            lag_v = int(lag_v_str)
        except ValueError:
            continue
        # Determine indices i, j for src and tgt in var_names
        if src_label in var_names:
            i = var_names.index(src_label)
        else:
            try:
                idx_val = int(src_label)
                i = idx_val if 0 <= idx_val < D else None
            except ValueError:
                i = None
        if tgt_label in var_names:
            j = var_names.index(tgt_label)
        else:
            try:
                idx_val = int(tgt_label)
                j = idx_val if 0 <= idx_val < D else None
            except ValueError:
                j = None
        if i is None or j is None:
            continue
        if lag_u == 0 and lag_v == 0:
            W[i, j] = w
        elif lag_u > 0 and lag_v == 0 and 1 <= lag_u <= p:
            A[i, j, lag_u - 1] = w
    return W, A

def calculate_matrix_distance(W1: torch.Tensor, W2: torch.Tensor) -> float:
    """
    Compute the Frobenius norm (Euclidean distance) between two matrices.

    Args:
        W1: First weight matrix (torch.Tensor).
        W2: Second weight matrix (torch.Tensor) of the same shape as W1.

    Returns:
        The Frobenius norm (scalar float) of (W1 - W2).
    """
    return float(torch.norm(W1 - W2, p="fro").item())

def generate_histogram_and_kde(weights: Union[np.ndarray, torch.Tensor], bin_step: float = 0.02) -> Dict[str, pd.DataFrame]:
    """
    Generate a histogram and kernel density estimate (KDE) for an array of weights.

    This function considers only nonzero weights. It returns two pandas DataFrames:
    one for the histogram (with columns 'bin_center' and 'count'), and one for the KDE (with columns 'x' and 'density').

    Args:
        weights: A numpy array or torch.Tensor of weight values.
        bin_step: Bin width for the histogram.

    Returns:
        A dictionary with keys 'histogram' and 'kde', each mapping to a DataFrame.
    """
    from scipy.stats import gaussian_kde
    # Convert to numpy array and filter zeros
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    weights = weights.astype(float)
    weights_nonzero = weights[weights != 0]
    if weights_nonzero.size == 0:
        hist_df = pd.DataFrame(columns=["bin_center", "count"])
        kde_df = pd.DataFrame(columns=["x", "density"])
        return {"histogram": hist_df, "kde": kde_df}
    logger.info(f"Generating histogram & KDE for {weights_nonzero.size} nonzero weights")
    # Histogram
    min_w, max_w = weights_nonzero.min(), weights_nonzero.max()
    bins = np.arange(min_w, max_w + bin_step, bin_step)
    counts, edges = np.histogram(weights_nonzero, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist_df = pd.DataFrame({"bin_center": centers, "count": counts})
    # KDE (use a fine grid of 2000 points between min and max)
    kde = gaussian_kde(weights_nonzero)
    xs = np.linspace(min_w, max_w, 2000)
    ys = kde(xs)
    kde_df = pd.DataFrame({"x": xs, "density": ys})
    return {"histogram": hist_df, "kde": kde_df}

# --------------------------------------------------------------------------
# Simple baseline main section (when run as Step 5)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple baseline implementation when run as standalone script
    import sys
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    
    # Configuration from environment variables
    DATA_FILE = os.getenv('INPUT_DIFFERENCED_CSV', "differenced_stationary_series.csv")
    OPTIMAL_LAGS_FILE = os.getenv('INPUT_LAGS_CSV', "optimal_lags.csv")
    RESULT_DIR = os.getenv('RESULT_DIR')
    if RESULT_DIR:
        OUTPUT_DIR = os.path.join(RESULT_DIR, 'weights')
    else:
        OUTPUT_DIR = os.getenv('OUTPUT_DIR', "dynotears_results")
    
    # Setup logging for main script
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'dynotears_baseline.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    def load_optimal_lags(path):
        """Load optimal lag value from CSV"""
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            return None
        if df.empty:
            return None
        if 'optimal_lag' in df.columns and df['optimal_lag'].notna().all():
            if df['optimal_lag'].nunique() == 1:
                return int(df['optimal_lag'].iloc[0])
            return df.set_index('variable')['optimal_lag'].to_dict()
        return None
    
    def fit_simple_baseline(data, lag=1, alpha=0.1):
        """Simple VAR baseline using Ridge regression (for Step 5)"""
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        n, d = data.shape
        
        if n <= lag:
            return np.zeros((d, d)), [np.zeros((d, d)) for _ in range(lag)]
        
        # Create design matrix
        X = []
        y = []
        
        for t in range(lag, n):
            y.append(data[t])
            x_row = []
            for l in range(1, lag + 1):
                x_row.extend(data[t - l])
            X.append(x_row)
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit Ridge regression for each variable
        W = np.zeros((d, d))  # No contemporaneous effects in baseline
        A_list = [np.zeros((d, d)) for _ in range(lag)]
        
        for j in range(d):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X, y[:, j])
            
            coef = ridge.coef_
            coef_idx = 0
            for l in range(lag):
                for i in range(d):
                    if coef_idx < len(coef):
                        A_list[l][i, j] = coef[coef_idx]
                    coef_idx += 1
        
        return W, A_list
    
    logger.info("=== DYNOTEARS Simple Baseline (Step 5) ===")
    
    try:
        # Load data
        logger.info(f"Loading data from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.values)
        df_scaled = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)
        
        # Load optimal lag
        opt_lag = load_optimal_lags(OPTIMAL_LAGS_FILE)
        if opt_lag is None:
            lag = 1
        elif isinstance(opt_lag, int):
            lag = opt_lag
        else:
            lag = max(opt_lag.values()) if len(opt_lag) > 0 else 1
        logger.info(f"Using lag p = {lag}")
        
        # Run simple baseline
        logger.info("Running simple VAR baseline...")
        start_time = time.time()
        
        W, A_list = fit_simple_baseline(df_scaled.values, lag, alpha=0.1)
        
        # Count edges
        W_edges = np.sum(np.abs(W) > 1e-4)
        A_edges = sum(np.sum(np.abs(A) > 1e-4) for A in A_list)
        
        elapsed = time.time() - start_time
        logger.info(f"Simple baseline completed in {elapsed:.2f}s")
        logger.info(f"Found {W_edges} contemporaneous edges (W matrix)")
        logger.info(f"Found {A_edges} lagged edges (A matrix)")
        logger.info(f"Total edges: {W_edges + A_edges}")
        
        # Save results
        run_id = time.strftime("%Y%m%d_%H%M%S")
        import json
        results = {
            'method': 'Simple VAR baseline (Ridge regression)',
            'approach': 'Single global analysis (Step 5)',
            'runtime_seconds': elapsed,
            'data_info': {
                'shape': list(df.shape),
                'variables': list(df.columns),
                'lag_order': lag
            },
            'results': {
                'W_edges': int(W_edges),
                'A_edges': int(A_edges),
                'total_edges': int(W_edges + A_edges)
            }
        }
        
        # Save matrices and results
        np.save(os.path.join(OUTPUT_DIR, f"W_baseline_{run_id}.npy"), W)
        np.save(os.path.join(OUTPUT_DIR, f"A_baseline_{run_id}.npy"), np.stack(A_list, axis=2) if A_list else np.zeros((len(df.columns), len(df.columns), 1)))
        
        with open(os.path.join(OUTPUT_DIR, f"baseline_results_{run_id}.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Baseline results saved to: {OUTPUT_DIR}")
        logger.info("=== DYNOTEARS Simple Baseline Complete ===")
        
    except Exception as e:
        logger.error(f"Error in simple baseline: {e}")
        sys.exit(1)
