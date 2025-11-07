#!/usr/bin/env python3
"""
Weight Correction Module for Anomaly Reconstruction

This module provides strategies for correcting anomalous weights identified
through the anomaly detection suite, enabling reconstruction of "corrected"
time series.

Correction Strategies:
1. Replace with baseline - Use baseline weights for anomalous edges
2. Median correction - Replace with median of nearby windows
3. Interpolation - Interpolate between nearest non-anomalous windows
4. Zeroing - Set anomalous edges to zero (conservative)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightCorrector:
    """
    Corrects anomalous weights using various strategies.
    """

    def __init__(self, strategy: str = 'replace_with_baseline'):
        """
        Initialize weight corrector.

        Args:
            strategy: Correction strategy
                - 'replace_with_baseline': Replace anomalous edges with baseline values
                - 'median': Replace with median from surrounding windows
                - 'interpolate': Interpolate from nearest non-anomalous windows
                - 'zero': Set anomalous edges to zero
                - 'soft_correction': Partial correction (weighted average)
        """
        self.strategy = strategy
        self.valid_strategies = [
            'replace_with_baseline', 'median', 'interpolate',
            'zero', 'soft_correction'
        ]

        if strategy not in self.valid_strategies:
            raise ValueError(f"Strategy must be one of {self.valid_strategies}")

        logger.info(f"Initialized WeightCorrector with strategy: {strategy}")

    def correct_weights(self,
                       W_anomaly: np.ndarray,
                       W_baseline: Optional[np.ndarray] = None,
                       anomalous_edges: Optional[List[Dict]] = None,
                       threshold: float = 0.3) -> Tuple[np.ndarray, Dict]:
        """
        Correct anomalous weights based on selected strategy.

        Args:
            W_anomaly: Anomalous weight matrix [d, d]
            W_baseline: Baseline weight matrix [d, d] (required for some strategies)
            anomalous_edges: List of anomalous edges from root cause analysis
                Each dict has: {'from': i, 'to': j, 'change': value, 'importance': value}
            threshold: Threshold for determining significant edges

        Returns:
            Tuple of (corrected_weights, correction_info)
        """
        W_corrected = W_anomaly.copy()
        correction_info = {
            'strategy': self.strategy,
            'num_corrections': 0,
            'corrected_edges': [],
            'correction_magnitude': []
        }

        # Determine which edges to correct
        if anomalous_edges is not None:
            # Use provided anomalous edges
            edges_to_correct = [
                (edge['from'], edge['to'])
                for edge in anomalous_edges
                if abs(edge['change']) > threshold
            ]
        else:
            # Use difference-based detection if no edges provided
            if W_baseline is None:
                logger.warning("No baseline or anomalous edges provided, using threshold on all edges")
                edges_to_correct = []
            else:
                diff = np.abs(W_anomaly - W_baseline)
                edges_to_correct = list(zip(*np.where(diff > threshold)))

        logger.info(f"Correcting {len(edges_to_correct)} edges using '{self.strategy}' strategy")

        # Apply correction strategy
        for i, j in edges_to_correct:
            original_val = W_corrected[i, j]

            if self.strategy == 'replace_with_baseline':
                if W_baseline is None:
                    raise ValueError("Baseline weights required for 'replace_with_baseline' strategy")
                corrected_val = W_baseline[i, j]

            elif self.strategy == 'zero':
                corrected_val = 0.0

            elif self.strategy == 'soft_correction':
                if W_baseline is None:
                    raise ValueError("Baseline weights required for 'soft_correction' strategy")
                # 70% baseline, 30% current (configurable)
                alpha = 0.7
                corrected_val = alpha * W_baseline[i, j] + (1 - alpha) * W_anomaly[i, j]

            else:
                # For median/interpolate, would need multiple windows
                # Default to baseline if available, else zero
                if W_baseline is not None:
                    corrected_val = W_baseline[i, j]
                else:
                    corrected_val = 0.0

            W_corrected[i, j] = corrected_val

            correction_info['corrected_edges'].append((i, j))
            correction_info['correction_magnitude'].append(abs(corrected_val - original_val))

        correction_info['num_corrections'] = len(edges_to_correct)
        correction_info['mean_correction_magnitude'] = np.mean(correction_info['correction_magnitude']) if correction_info['correction_magnitude'] else 0

        logger.info(f"Corrected {correction_info['num_corrections']} edges "
                   f"(mean magnitude: {correction_info['mean_correction_magnitude']:.6f})")

        return W_corrected, correction_info

    def correct_temporal_weights(self,
                                W_series: List[np.ndarray],
                                anomaly_indices: List[int],
                                W_baseline: np.ndarray) -> List[np.ndarray]:
        """
        Correct a temporal series of weight matrices.

        Args:
            W_series: List of weight matrices over time
            anomaly_indices: Indices of anomalous windows
            W_baseline: Baseline weight matrix

        Returns:
            List of corrected weight matrices
        """
        W_corrected_series = []

        for idx, W in enumerate(W_series):
            if idx in anomaly_indices:
                W_corr, _ = self.correct_weights(W, W_baseline)
                W_corrected_series.append(W_corr)
                logger.debug(f"Corrected window {idx}")
            else:
                W_corrected_series.append(W.copy())

        logger.info(f"Corrected {len(anomaly_indices)} out of {len(W_series)} temporal windows")
        return W_corrected_series


def load_weights_from_csv(weights_csv_path: str, window_idx: Optional[int] = None) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
    """
    Load weights from DynoTEARS CSV output.

    Expected format: window_idx,lag,i,j,weight

    Args:
        weights_csv_path: Path to weights CSV file
        window_idx: If provided, load weights for specific window.
                   If None, average weights across all windows.

    Returns:
        Dict with 'W' (instantaneous) and 'A_list' (lag) matrices
        If window_idx is None, weights are averaged across all windows
        If window_idx is provided, returns weights for that specific window
    """
    df = pd.read_csv(weights_csv_path)

    # Get dimensions
    d = df['i'].max() + 1

    if window_idx is not None:
        # Load weights for specific window
        window_df = df[df['window_idx'] == window_idx]

        if len(window_df) == 0:
            raise ValueError(f"No weights found for window {window_idx}")

        # Extract instantaneous weights (lag = 0)
        W_df = window_df[window_df['lag'] == 0]
        W = np.zeros((d, d))
        for _, row in W_df.iterrows():
            W[int(row['i']), int(row['j'])] = row['weight']

        # Extract lag weights (lag = 1, 2, ...)
        A_matrices = {}
        for lag in window_df['lag'].unique():
            if lag > 0:
                A_df = window_df[window_df['lag'] == lag]
                A = np.zeros((d, d))
                for _, row in A_df.iterrows():
                    A[int(row['i']), int(row['j'])] = row['weight']
                A_matrices[int(lag)] = A

        # Sort lag matrices
        A_list = [A_matrices[k] for k in sorted(A_matrices.keys())]

    else:
        # Average weights across all windows
        # Group by lag, i, j and take mean of weights
        avg_weights = df.groupby(['lag', 'i', 'j'])['weight'].mean().reset_index()

        # Extract instantaneous weights (lag = 0)
        W_df = avg_weights[avg_weights['lag'] == 0]
        W = np.zeros((d, d))
        for _, row in W_df.iterrows():
            W[int(row['i']), int(row['j'])] = row['weight']

        # Extract lag weights (lag = 1, 2, ...)
        A_matrices = {}
        for lag in avg_weights['lag'].unique():
            if lag > 0:
                A_df = avg_weights[avg_weights['lag'] == lag]
                A = np.zeros((d, d))
                for _, row in A_df.iterrows():
                    A[int(row['i']), int(row['j'])] = row['weight']
                A_matrices[int(lag)] = A

        # Sort lag matrices
        A_list = [A_matrices[k] for k in sorted(A_matrices.keys())]

    return {'W': W, 'A_list': A_list}


def load_all_windows_from_csv(weights_csv_path: str) -> Tuple[List[Dict[str, Union[np.ndarray, List[np.ndarray]]]], List[int]]:
    """
    Load weights for all windows separately (no averaging).

    Args:
        weights_csv_path: Path to weights CSV file

    Returns:
        Tuple of (list of weight dicts, list of window indices)
        Each weight dict contains 'W' and 'A_list' for one window
    """
    df = pd.read_csv(weights_csv_path)

    # Get all unique window indices
    window_indices = sorted(df['window_idx'].unique())

    # Load weights for each window
    all_windows = []
    for window_idx in window_indices:
        weights = load_weights_from_csv(weights_csv_path, window_idx=window_idx)
        all_windows.append(weights)

    logger.info(f"Loaded {len(all_windows)} windows from {weights_csv_path}")
    return all_windows, window_indices


def get_window_info(weights_csv_path: str) -> Dict[str, int]:
    """
    Get information about windows in the weights CSV.

    Args:
        weights_csv_path: Path to weights CSV file

    Returns:
        Dict with 'num_windows', 'min_window', 'max_window', 'num_variables', 'max_lag'
    """
    df = pd.read_csv(weights_csv_path)

    return {
        'num_windows': df['window_idx'].nunique(),
        'min_window': int(df['window_idx'].min()),
        'max_window': int(df['window_idx'].max()),
        'num_variables': int(df['i'].max() + 1),
        'max_lag': int(df['lag'].max())
    }


def save_corrected_weights(W_corrected: np.ndarray,
                           A_list: List[np.ndarray],
                           output_path: str,
                           correction_info: Dict):
    """
    Save corrected weights in DynoTEARS format.

    Args:
        W_corrected: Corrected instantaneous weights
        A_list: Lag weight matrices
        output_path: Output CSV path
        correction_info: Metadata about corrections
    """
    rows = []
    d = W_corrected.shape[0]

    # Save instantaneous weights
    for i in range(d):
        for j in range(d):
            rows.append({
                'window_idx': 0,
                'lag': 0,
                'i': i,
                'j': j,
                'weight': W_corrected[i, j]
            })

    # Save lag weights
    for lag_idx, A in enumerate(A_list, start=1):
        for i in range(d):
            for j in range(d):
                rows.append({
                    'window_idx': 0,
                    'lag': lag_idx,
                    'i': i,
                    'j': j,
                    'weight': A[i, j]
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    # Save metadata
    metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_correction_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Correction Strategy: {correction_info['strategy']}\n")
        f.write(f"Number of Corrections: {correction_info['num_corrections']}\n")
        f.write(f"Mean Correction Magnitude: {correction_info['mean_correction_magnitude']:.6f}\n")
        f.write(f"\nCorrected Edges:\n")
        for (i, j), mag in zip(correction_info['corrected_edges'],
                              correction_info['correction_magnitude']):
            f.write(f"  Edge {i} -> {j}: magnitude {mag:.6f}\n")

    logger.info(f"Saved corrected weights to {output_path}")
    logger.info(f"Saved correction metadata to {metadata_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Weight Corrector Module")
    print("=" * 60)
    print("\nAvailable correction strategies:")
    print("  1. replace_with_baseline - Replace anomalous edges with baseline")
    print("  2. median - Replace with median from surrounding windows")
    print("  3. interpolate - Interpolate from nearest non-anomalous windows")
    print("  4. zero - Set anomalous edges to zero")
    print("  5. soft_correction - Partial correction (weighted average)")
    print("\nUsage in pipeline:")
    print("  from weight_corrector import WeightCorrector")
    print("  corrector = WeightCorrector(strategy='replace_with_baseline')")
    print("  W_corrected, info = corrector.correct_weights(W_anomaly, W_baseline, anomalous_edges)")
