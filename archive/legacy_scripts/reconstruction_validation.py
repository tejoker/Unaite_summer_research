#!/usr/bin/env python3
"""
Reconstruction Validation

Validates reconstructed time series by:
1. Computing DynoTEARS weights on reconstructed data
2. Comparing reconstructed weights vs golden weights
3. Using same metrics as anomaly detection (Frobenius, SHD, spectral, max edge change)
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy.linalg import eigh

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def compute_frobenius_distance(W1: np.ndarray, W2: np.ndarray,
                               A1_list: list, A2_list: list) -> float:
    """
    Compute Frobenius distance between two sets of weight matrices.

    Args:
        W1, W2: Instantaneous matrices
        A1_list, A2_list: Lists of lag matrices

    Returns:
        Frobenius distance (L2 norm of differences)
    """
    # Distance for W
    dist_w = np.linalg.norm(W1 - W2, ord='fro')

    # Distance for A matrices
    dist_a = 0.0
    p = min(len(A1_list), len(A2_list))
    for k in range(p):
        dist_a += np.linalg.norm(A1_list[k] - A2_list[k], ord='fro')

    total_dist = dist_w + dist_a
    return total_dist


def compute_structural_hamming_distance(W1: np.ndarray, W2: np.ndarray,
                                       A1_list: list, A2_list: list,
                                       threshold: float = 1e-8) -> int:
    """
    Compute Structural Hamming Distance (edge difference count).

    Args:
        W1, W2: Instantaneous matrices
        A1_list, A2_list: Lists of lag matrices
        threshold: Edge existence threshold

    Returns:
        Number of edge differences
    """
    # Binarize matrices
    W1_binary = (np.abs(W1) > threshold).astype(int)
    W2_binary = (np.abs(W2) > threshold).astype(int)

    # Count differences for W
    shd_w = np.sum(W1_binary != W2_binary)

    # Count differences for A
    shd_a = 0
    p = min(len(A1_list), len(A2_list))
    for k in range(p):
        A1_binary = (np.abs(A1_list[k]) > threshold).astype(int)
        A2_binary = (np.abs(A2_list[k]) > threshold).astype(int)
        shd_a += np.sum(A1_binary != A2_binary)

    total_shd = shd_w + shd_a
    return int(total_shd)


def compute_spectral_distance(W1: np.ndarray, W2: np.ndarray) -> float:
    """
    Compute spectral distance (difference in largest eigenvalue magnitude).

    Args:
        W1, W2: Instantaneous matrices

    Returns:
        Difference in spectral radius
    """
    eigenvals_1 = eigh(W1, eigvals_only=True)
    eigenvals_2 = eigh(W2, eigvals_only=True)

    spec_radius_1 = np.max(np.abs(eigenvals_1))
    spec_radius_2 = np.max(np.abs(eigenvals_2))

    dist = abs(spec_radius_1 - spec_radius_2)
    return dist


def compute_max_edge_change(W1: np.ndarray, W2: np.ndarray,
                           A1_list: list, A2_list: list) -> float:
    """
    Compute maximum absolute edge weight change.

    Args:
        W1, W2: Instantaneous matrices
        A1_list, A2_list: Lists of lag matrices

    Returns:
        Maximum absolute weight change
    """
    # Max change in W
    max_change_w = np.max(np.abs(W1 - W2))

    # Max change in A
    max_change_a = 0.0
    p = min(len(A1_list), len(A2_list))
    for k in range(p):
        max_change_a = max(max_change_a, np.max(np.abs(A1_list[k] - A2_list[k])))

    max_change = max(max_change_w, max_change_a)
    return max_change


def validate_reconstruction(W_golden: np.ndarray,
                           A_golden_list: list,
                           W_reconstructed: np.ndarray,
                           A_reconstructed_list: list) -> Dict[str, float]:
    """
    Validate reconstruction quality by comparing weights.

    Args:
        W_golden: Golden instantaneous matrix
        A_golden_list: Golden lag matrices
        W_reconstructed: Reconstructed instantaneous matrix
        A_reconstructed_list: Reconstructed lag matrices

    Returns:
        Dictionary of validation metrics
    """
    logger.info("=" * 80)
    logger.info("RECONSTRUCTION VALIDATION")
    logger.info("=" * 80)

    metrics = {}

    # Frobenius distance
    frobenius = compute_frobenius_distance(W_golden, W_reconstructed,
                                          A_golden_list, A_reconstructed_list)
    metrics['frobenius_distance'] = frobenius
    logger.info(f"Frobenius distance: {frobenius:.6f}")

    # Structural Hamming Distance
    shd = compute_structural_hamming_distance(W_golden, W_reconstructed,
                                             A_golden_list, A_reconstructed_list)
    metrics['structural_hamming_distance'] = shd
    logger.info(f"Structural Hamming Distance: {shd} edges")

    # Spectral distance
    spectral = compute_spectral_distance(W_golden, W_reconstructed)
    metrics['spectral_distance'] = spectral
    logger.info(f"Spectral distance: {spectral:.6f}")

    # Max edge change
    max_edge = compute_max_edge_change(W_golden, W_reconstructed,
                                      A_golden_list, A_reconstructed_list)
    metrics['max_edge_change'] = max_edge
    logger.info(f"Max edge change: {max_edge:.6f}")

    # Quality assessment
    logger.info("=" * 80)
    if frobenius < 0.1 and shd < 10:
        logger.info("VALIDATION: EXCELLENT - Reconstruction very close to golden")
    elif frobenius < 0.5 and shd < 50:
        logger.info("VALIDATION: GOOD - Reconstruction reasonably close to golden")
    elif frobenius < 1.0 and shd < 100:
        logger.info("VALIDATION: MODERATE - Some structural differences remain")
    else:
        logger.info("VALIDATION: POOR - Significant differences from golden")
    logger.info("=" * 80)

    return metrics


def load_weights_from_csv(weights_csv: str,
                          var_names: list,
                          window_idx: Optional[int] = None) -> Tuple[np.ndarray, list]:
    """
    Load weights from CSV file.

    Args:
        weights_csv: Path to weights CSV
        var_names: List of variable names
        window_idx: Optional specific window (averages all if None)

    Returns:
        Tuple of (W, A_list)
    """
    df_weights = pd.read_csv(weights_csv)

    d = len(var_names)
    var_to_idx = {name: idx for idx, name in enumerate(var_names)}
    p = int(df_weights['lag'].max())

    # Filter by window if specified
    if window_idx is not None:
        df_weights = df_weights[df_weights['window_idx'] == window_idx]
    else:
        # Average across all windows
        df_weights = df_weights.groupby(['lag', 'parent_name', 'child_name'])['weight'].mean().reset_index()

    # Initialize matrices
    W = np.zeros((d, d))
    A_list = [np.zeros((d, d)) for _ in range(p)]

    # Fill matrices
    for _, row in df_weights.iterrows():
        lag = int(row['lag'])
        parent = row['parent_name']
        child = row['child_name']
        weight = float(row['weight'])

        if parent in var_to_idx and child in var_to_idx:
            i = var_to_idx[child]
            j = var_to_idx[parent]

            if lag == 0:
                W[i, j] = weight
            elif 1 <= lag <= p:
                A_list[lag - 1][i, j] = weight

    return W, A_list


def compare_golden_vs_reconstructed(golden_weights_csv: str,
                                   reconstructed_weights_csv: str,
                                   var_names: list) -> Dict[str, float]:
    """
    High-level function to compare golden vs reconstructed weights.

    Args:
        golden_weights_csv: Path to golden weights
        reconstructed_weights_csv: Path to reconstructed weights
        var_names: List of variable names

    Returns:
        Dictionary of validation metrics
    """
    logger.info(f"Loading golden weights from {golden_weights_csv}")
    W_golden, A_golden_list = load_weights_from_csv(golden_weights_csv, var_names)

    logger.info(f"Loading reconstructed weights from {reconstructed_weights_csv}")
    W_reconstructed, A_reconstructed_list = load_weights_from_csv(reconstructed_weights_csv, var_names)

    metrics = validate_reconstruction(W_golden, A_golden_list,
                                     W_reconstructed, A_reconstructed_list)

    return metrics


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate reconstruction quality")
    parser.add_argument("--golden_weights", required=True, help="Path to golden weights CSV")
    parser.add_argument("--reconstructed_weights", required=True, help="Path to reconstructed weights CSV")
    parser.add_argument("--variables", nargs="+", required=True, help="Variable names")
    parser.add_argument("--output", help="Optional CSV file to save metrics")

    args = parser.parse_args()

    # Compare
    metrics = compare_golden_vs_reconstructed(
        golden_weights_csv=args.golden_weights,
        reconstructed_weights_csv=args.reconstructed_weights,
        var_names=args.variables
    )

    # Save if requested
    if args.output:
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(args.output, index=False)
        logger.info(f"Saved metrics to {args.output}")

    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")
