#!/usr/bin/env python3
"""
Anomaly Reconstruction using Golden Weights

This module reconstructs multivariate time series during anomalous windows
using the causal structure learned from golden (baseline) data.

Approach:
- Phase 1: Load golden weights (W_golden, A_golden) from reference data
- Phase 2: Teacher forcing reconstruction (use actual history, predict current)
- Phase 3: Full reconstruction (all variables)
- Phase 4: Validate by comparing reconstructed weights vs golden
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from scipy.linalg import solve, LinAlgError
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class AnomalyReconstructor:
    """
    Reconstructs anomalous time series using golden (baseline) causal structure.

    Uses teacher forcing: actual history + golden weights -> predicted current values
    """

    def __init__(self,
                 W_golden: np.ndarray,
                 A_golden_list: List[np.ndarray],
                 var_names: List[str],
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize reconstructor with golden weights.

        Args:
            W_golden: Instantaneous weight matrix [d, d] from golden data
            A_golden_list: List of lag matrices [A^(1), ..., A^(p)] from golden data
            var_names: List of variable names
            scaler: Optional scaler for inverse transformation
        """
        self.var_names = var_names
        self.d = len(var_names)
        self.p = len(A_golden_list)
        self.scaler = scaler

        # Store golden weights
        self.W_golden = np.array(W_golden)
        self.A_golden_list = [np.array(A) for A in A_golden_list]

        # Validate dimensions
        self._validate_matrices()

        # Compute S = I - W for solving instantaneous layer
        self.S_golden = np.eye(self.d) - self.W_golden

        # Check invertibility
        self._check_invertibility()

        # Pre-compute inverse for efficiency
        try:
            self.S_golden_inv = np.linalg.inv(self.S_golden)
        except LinAlgError:
            logger.warning("Using pseudo-inverse for S_golden")
            self.S_golden_inv = np.linalg.pinv(self.S_golden)

        logger.info(f"Initialized reconstructor: {self.d} variables, lag order {self.p}")
        self._log_matrix_stats()

    def _validate_matrices(self):
        """Validate matrix dimensions."""
        if self.W_golden.shape != (self.d, self.d):
            raise ValueError(f"W_golden shape {self.W_golden.shape} != expected ({self.d}, {self.d})")

        for i, A in enumerate(self.A_golden_list):
            if A.shape != (self.d, self.d):
                raise ValueError(f"A_golden[{i+1}] shape {A.shape} != expected ({self.d}, {self.d})")

    def _check_invertibility(self, reg_param: float = 1e-6):
        """Check if S_golden is invertible."""
        try:
            cond_num = np.linalg.cond(self.S_golden)
            if cond_num > 1e12:
                logger.warning(f"S_golden is ill-conditioned (cond={cond_num:.2e})")
                # Apply Tikhonov regularization
                self.S_golden += reg_param * np.eye(self.d)
                logger.info(f"Applied Tikhonov regularization (lambda={reg_param})")

            # Test inversion
            _ = np.linalg.inv(self.S_golden)
            logger.info("Invertibility check passed")

        except LinAlgError:
            logger.error("S_golden is not invertible")
            raise ValueError("Golden weights violate acyclicity - cannot reconstruct")

    def _log_matrix_stats(self):
        """Log statistics about golden weights."""
        w_edges = np.sum(np.abs(self.W_golden) > 1e-8)
        logger.info(f"W_golden: {w_edges} edges, sparsity {1 - w_edges/(self.d**2):.3f}")

        for i, A in enumerate(self.A_golden_list):
            a_edges = np.sum(np.abs(A) > 1e-8)
            logger.info(f"A_golden[{i+1}]: {a_edges} edges, sparsity {1 - a_edges/(self.d**2):.3f}")

    def predict_one_step(self, history: np.ndarray) -> np.ndarray:
        """
        Predict one step ahead using golden weights (teacher forcing).

        Args:
            history: Historical values [p, d] for x(t-p), ..., x(t-1)

        Returns:
            Predicted values x_pred(t) of shape [d]
        """
        if history.shape != (self.p, self.d):
            raise ValueError(f"History shape {history.shape} != expected ({self.p}, {self.d})")

        # Compute lagged contribution: h(t) = sum_k A^(k) @ x(t-k)
        h = np.zeros(self.d)
        for k in range(self.p):
            # history[p-1-k] corresponds to x(t-k-1)
            # For k=0: we want A^(1) @ x(t-1) = A_list[0] @ history[p-1]
            h += self.A_golden_list[k] @ history[self.p - 1 - k]

        # Solve instantaneous layer: x(t) = S^(-1) @ h(t)
        x_pred = self.S_golden_inv @ h

        return x_pred

    def reconstruct_window(self,
                          df_anomaly: pd.DataFrame,
                          window_start: int,
                          window_end: int) -> pd.DataFrame:
        """
        Reconstruct a specific anomalous window using teacher forcing.

        Args:
            df_anomaly: Anomalous time series DataFrame [T, d]
            window_start: Start index of anomalous window (inclusive)
            window_end: End index of anomalous window (exclusive)

        Returns:
            Reconstructed DataFrame with same shape as input
        """
        logger.info(f"Reconstructing window [{window_start}, {window_end})")

        # Make a copy to avoid modifying original
        df_reconstructed = df_anomaly.copy()

        # Extract data as numpy array
        X = df_anomaly.values

        # Ensure we have enough history before the window
        if window_start < self.p:
            raise ValueError(f"Window starts at {window_start}, need at least {self.p} previous timesteps")

        # Reconstruct each timestep in the window
        for t in range(window_start, window_end):
            # Get history: x(t-p), ..., x(t-1)
            history = X[t-self.p:t]  # Shape [p, d]

            # Predict current timestep using golden weights
            x_pred = self.predict_one_step(history)

            # Replace with prediction (full reconstruction of all variables)
            X[t] = x_pred

        # Update DataFrame
        df_reconstructed[df_anomaly.columns] = X

        logger.info(f"Reconstruction completed for {window_end - window_start} timesteps")

        return df_reconstructed

    def reconstruct_multiple_windows(self,
                                    df_anomaly: pd.DataFrame,
                                    anomaly_windows: List[Tuple[int, int]]) -> pd.DataFrame:
        """
        Reconstruct multiple anomalous windows.

        Args:
            df_anomaly: Anomalous time series DataFrame
            anomaly_windows: List of (start, end) tuples for anomalous windows

        Returns:
            Fully reconstructed DataFrame
        """
        logger.info(f"Reconstructing {len(anomaly_windows)} anomalous windows")

        df_reconstructed = df_anomaly.copy()

        for i, (start, end) in enumerate(anomaly_windows):
            logger.info(f"Window {i+1}/{len(anomaly_windows)}: [{start}, {end})")
            df_reconstructed = self.reconstruct_window(df_reconstructed, start, end)

        return df_reconstructed

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

        data_original = self.scaler.inverse_transform(df.values)
        df_original = pd.DataFrame(data_original, columns=df.columns, index=df.index)

        logger.info("Applied inverse scaling transformation")
        return df_original


def load_golden_weights_from_csv(weights_csv: str,
                                 var_names: Optional[List[str]] = None,
                                 window_idx: Optional[int] = None) -> Tuple[np.ndarray, List[np.ndarray], List[str], int]:
    """
    Load golden weights from weights_enhanced.csv file.

    Args:
        weights_csv: Path to weights_enhanced.csv
        var_names: Optional list of variable names (extracted if None)
        window_idx: Specific window to load (averages all windows if None)

    Returns:
        Tuple of (W, A_list, var_names, p)
    """
    logger.info(f"Loading golden weights from {weights_csv}")

    df_weights = pd.read_csv(weights_csv)

    # Extract variable names if not provided
    if var_names is None:
        unique_vars = sorted(set(df_weights['parent_name'].unique()) | set(df_weights['child_name'].unique()))
        var_names = unique_vars
        logger.info(f"Extracted {len(var_names)} variables from weights file")

    d = len(var_names)
    var_to_idx = {name: idx for idx, name in enumerate(var_names)}

    # Determine lag order
    p = int(df_weights['lag'].max())
    logger.info(f"Lag order: {p}")

    # Filter by window if specified
    if window_idx is not None:
        df_weights = df_weights[df_weights['window_idx'] == window_idx]
        logger.info(f"Using weights from window {window_idx}")
    else:
        # Average across all windows
        logger.info(f"Averaging weights across {df_weights['window_idx'].nunique()} windows")
        df_weights = df_weights.groupby(['lag', 'parent_name', 'child_name', 'i', 'j'])['weight'].mean().reset_index()

    # Initialize matrices
    W = np.zeros((d, d))
    A_list = [np.zeros((d, d)) for _ in range(p)]

    # Fill matrices
    for _, row in df_weights.iterrows():
        lag = int(row['lag'])
        parent = row['parent_name']
        child = row['child_name']
        weight = float(row['weight'])

        # Get indices
        if parent in var_to_idx and child in var_to_idx:
            i = var_to_idx[child]  # row (child/target)
            j = var_to_idx[parent]  # column (parent/source)

            if lag == 0:
                W[i, j] = weight
            elif 1 <= lag <= p:
                A_list[lag - 1][i, j] = weight

    # Log statistics
    w_edges = np.sum(np.abs(W) > 1e-8)
    logger.info(f"Loaded W: {w_edges} edges")
    for k, A in enumerate(A_list):
        a_edges = np.sum(np.abs(A) > 1e-8)
        logger.info(f"Loaded A[{k+1}]: {a_edges} edges")

    return W, A_list, var_names, p


def reconstruct_from_golden(golden_weights_csv: str,
                            anomaly_data_csv: str,
                            anomaly_windows: List[Tuple[int, int]],
                            output_csv: Optional[str] = None,
                            scaler: Optional[StandardScaler] = None) -> pd.DataFrame:
    """
    High-level function to reconstruct anomalous data using golden weights.

    Args:
        golden_weights_csv: Path to golden weights CSV
        anomaly_data_csv: Path to anomalous time series CSV
        anomaly_windows: List of (start, end) tuples for anomalous windows
        output_csv: Optional path to save reconstructed data
        scaler: Optional scaler for inverse transformation

    Returns:
        Reconstructed DataFrame
    """
    logger.info("=" * 80)
    logger.info("ANOMALY RECONSTRUCTION USING GOLDEN WEIGHTS")
    logger.info("=" * 80)

    # Load golden weights
    W_golden, A_golden_list, var_names, p = load_golden_weights_from_csv(golden_weights_csv)

    # Load anomalous data
    logger.info(f"Loading anomalous data from {anomaly_data_csv}")
    df_anomaly = pd.read_csv(anomaly_data_csv)

    # Handle timestamp column if present
    if 'timestamp' in df_anomaly.columns:
        df_anomaly = df_anomaly.set_index('timestamp')

    # Ensure variable names match
    data_vars = [col for col in df_anomaly.columns]
    if data_vars != var_names:
        logger.warning(f"Variable names mismatch: {len(data_vars)} in data vs {len(var_names)} in weights")
        # Use intersection
        common_vars = [v for v in var_names if v in data_vars]
        if len(common_vars) < len(var_names):
            logger.warning(f"Using {len(common_vars)} common variables")
            var_names = common_vars
            # Refilter weights to common variables
            var_mask = [i for i, v in enumerate(var_names) if v in common_vars]
            W_golden = W_golden[np.ix_(var_mask, var_mask)]
            A_golden_list = [A[np.ix_(var_mask, var_mask)] for A in A_golden_list]

    # Select relevant columns
    df_anomaly = df_anomaly[var_names]

    # Create reconstructor
    reconstructor = AnomalyReconstructor(W_golden, A_golden_list, var_names, scaler)

    # Reconstruct anomalous windows
    df_reconstructed = reconstructor.reconstruct_multiple_windows(df_anomaly, anomaly_windows)

    # Save if requested
    if output_csv:
        df_reconstructed.to_csv(output_csv, index=True)
        logger.info(f"Saved reconstructed data to {output_csv}")

    logger.info("=" * 80)
    logger.info("RECONSTRUCTION COMPLETED")
    logger.info("=" * 80)

    return df_reconstructed


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct anomalous time series using golden weights")
    parser.add_argument("--golden_weights", required=True, help="Path to golden weights CSV")
    parser.add_argument("--anomaly_data", required=True, help="Path to anomalous time series CSV")
    parser.add_argument("--windows", required=True, help="Anomaly windows as 'start1,end1;start2,end2;...'")
    parser.add_argument("--output", help="Output CSV file for reconstructed data")

    args = parser.parse_args()

    # Parse windows
    windows = []
    for window_str in args.windows.split(';'):
        start, end = map(int, window_str.split(','))
        windows.append((start, end))

    # Reconstruct
    df_reconstructed = reconstruct_from_golden(
        golden_weights_csv=args.golden_weights,
        anomaly_data_csv=args.anomaly_data,
        anomaly_windows=windows,
        output_csv=args.output
    )

    print(f"\nReconstructed {len(df_reconstructed)} timesteps")
    print(f"Anomalous windows: {windows}")
    print("\nFirst few rows:")
    print(df_reconstructed.head())
