#!/usr/bin/env python3
"""
Dual-Metric Anomaly Detection with Cascade Disambiguation

This implements the approach described in todolist.md for detecting multiple anomalies
and distinguishing them from cascade effects and recovery fluctuations.

Approach:
1. Load weights from golden baseline and test timeline
2. Compute three metrics per window:
   - abs_score: distance from golden (how abnormal)
   - change_score: distance from previous window (how much changed)
   - abs_trend: trend in abs_score (getting worse or better)
3. Use adaptive thresholding (mean + 3*std from normal windows)
4. Classify each window into 4 states:
   - NORMAL: Low abs_score
   - NEW_ANOMALY_ONSET: High change + positive trend (degrading)
   - RECOVERY_FLUCTUATION: High change + negative trend (improving)
   - CASCADE_OR_PERSISTENT: High abs but low change (stable-abnormal)

Reference: todolist.md lines 80-153
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.distance import euclidean
from scipy.linalg import norm

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class GraphDistanceMetrics:
    """
    Compute distance metrics between causal graphs.
    """

    @staticmethod
    def frobenius_distance(W1: np.ndarray, W2: np.ndarray) -> float:
        """
        Frobenius norm distance between two weight matrices.

        Args:
            W1: First weight matrix
            W2: Second weight matrix

        Returns:
            Frobenius distance
        """
        return norm(W1 - W2, ord='fro')

    @staticmethod
    def spectral_distance(W1: np.ndarray, W2: np.ndarray) -> float:
        """
        Spectral (operator) norm distance between two weight matrices.

        Args:
            W1: First weight matrix
            W2: Second weight matrix

        Returns:
            Spectral distance
        """
        return norm(W1 - W2, ord=2)

    @staticmethod
    def weighted_edge_distance(W1: np.ndarray, W2: np.ndarray, threshold: float = 1e-3) -> float:
        """
        Distance metric focusing on significant edges.

        Args:
            W1: First weight matrix
            W2: Second weight matrix
            threshold: Edge significance threshold

        Returns:
            Weighted edge distance
        """
        # Mask significant edges
        mask = (np.abs(W1) > threshold) | (np.abs(W2) > threshold)

        # Compute distance only on significant edges
        diff = W1 - W2
        return np.sqrt(np.sum((diff * mask) ** 2))


class WeightsCache:
    """Cache for weight CSV data to avoid repeated file reads. Uses Polars for speed."""

    def __init__(self):
        self.cache = {}

    def load_csv(self, weights_csv: str) -> pl.DataFrame:
        """Load CSV with caching using Polars (much faster than pandas)."""
        if weights_csv not in self.cache:
            logger.info(f"Loading weights CSV with Polars: {weights_csv}")
            self.cache[weights_csv] = pl.read_csv(weights_csv)
            logger.info(f"Loaded {self.cache[weights_csv].height} rows")
        return self.cache[weights_csv]

    def get_matrix(self, weights_csv: str, window_idx: int, lag: int, fixed_dim: int) -> np.ndarray:
        """Extract weight matrix for specific window and lag."""
        df = self.load_csv(weights_csv)

        # Filter by window and lag (Polars syntax)
        df_window = df.filter((pl.col('window_idx') == window_idx) & (pl.col('lag') == lag))

        # Build matrix with consistent dimensions
        W = np.zeros((fixed_dim, fixed_dim))

        # Convert to numpy for fast iteration
        i_arr = df_window['i'].to_numpy()
        j_arr = df_window['j'].to_numpy()
        weight_arr = df_window['weight'].to_numpy()

        # Vectorized assignment (much faster than iterrows)
        for idx in range(len(i_arr)):
            i, j = int(i_arr[idx]), int(j_arr[idx])
            if i < fixed_dim and j < fixed_dim:
                W[i, j] = weight_arr[idx]

        return W


# Global cache instance
_weights_cache = WeightsCache()


def load_weights_from_csv(weights_csv: str, window_idx: int, lag: int = 0, fixed_dim: int = None) -> np.ndarray:
    """
    Load weight matrix for a specific window from weights_enhanced.csv.

    Args:
        weights_csv: Path to weights_enhanced.csv
        window_idx: Window index to load
        lag: Lag to extract (0 for contemporaneous, 1+ for lagged)
        fixed_dim: Fixed matrix dimension (if None, infer from data)

    Returns:
        Weight matrix as numpy array
    """
    # Use cached loader for better performance
    if fixed_dim is not None:
        return _weights_cache.get_matrix(weights_csv, window_idx, lag, fixed_dim)

    # Fallback: load without cache (for dimension detection)
    df = pd.read_csv(weights_csv)

    # Determine consistent matrix size across ALL windows
    max_i = int(df['i'].max()) if 'i' in df.columns else 0
    max_j = int(df['j'].max()) if 'j' in df.columns else 0
    d = max(max_i, max_j) + 1

    # Filter by window and lag
    df_window = df[(df['window_idx'] == window_idx) & (df['lag'] == lag)]

    # Build matrix with consistent dimensions
    W = np.zeros((d, d))
    for _, row in df_window.iterrows():
        i, j = int(row['i']), int(row['j'])
        if i < d and j < d:  # Only add if within bounds
            W[i, j] = float(row['weight'])

    return W


def compute_distance(W1: np.ndarray, W2: np.ndarray, metric: str = 'frobenius') -> float:
    """
    Compute distance between two weight matrices.

    Args:
        W1: First weight matrix
        W2: Second weight matrix
        metric: Distance metric ('frobenius', 'spectral', 'weighted_edge')

    Returns:
        Distance value
    """
    if metric == 'frobenius':
        return GraphDistanceMetrics.frobenius_distance(W1, W2)
    elif metric == 'spectral':
        return GraphDistanceMetrics.spectral_distance(W1, W2)
    elif metric == 'weighted_edge':
        return GraphDistanceMetrics.weighted_edge_distance(W1, W2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class DualMetricAnomalyDetector:
    """
    Dual-metric anomaly detector with cascade disambiguation.

    Implements the approach from todolist.md for detecting multiple anomalies.
    """

    def __init__(
        self,
        golden_weights_csv: str,
        lookback: int = 5,
        metric: str = 'frobenius',
        lag: int = 0,
        ensemble_dir: str = None
    ):
        """
        Initialize detector.

        Args:
            golden_weights_csv: Path to golden baseline weights
            lookback: Lookback window for trend calculation
            metric: Distance metric to use
            lag: Which lag to analyze (0=contemporaneous, 1+=lagged)
            ensemble_dir: Optional directory containing multiple run results to average (Bagging)
        """
        self.golden_weights_csv = golden_weights_csv
        self.lookback = lookback
        self.metric = metric
        self.lag = lag
        self.ensemble_dir = ensemble_dir
        
        # Load ensemble files if directory provided
        self.ensemble_files = []
        if self.ensemble_dir:
            p = Path(self.ensemble_dir)
            if p.exists():
                # Find all CSV files recursively or in top level
                self.ensemble_files = sorted(list(p.glob('**/weights*.csv')))
                logger.info(f"Ensemble mode: Found {len(self.ensemble_files)} weight files in {ensemble_dir}")
            else:
                logger.warning(f"Ensemble directory not found: {ensemble_dir}")

        # Load golden weights using Polars (much faster)
        logger.info(f"Loading golden baseline from {golden_weights_csv}")
        df_golden = pl.read_csv(golden_weights_csv)


        # Use ALL golden windows (no averaging - nearest neighbor approach)
        golden_window_indices = sorted(df_golden['window_idx'].unique().to_list())
        logger.info(f"Using nearest-neighbor approach with {len(golden_window_indices)} golden windows")

        # Determine consistent matrix dimension across ALL data (golden + test)
        df_golden_filtered = df_golden.filter(pl.col('lag') == lag)
        max_i = int(df_golden_filtered['i'].max())
        max_j = int(df_golden_filtered['j'].max())
        d = max(max_i, max_j) + 1
        self.fixed_dim = d  # Store for later use
        logger.info(f"Using fixed dimension: {d}x{d}")

        # Store golden window info for on-demand loading (memory efficient)
        logger.info("Using on-demand loading for golden windows (memory efficient)")
        self.golden_window_indices = golden_window_indices
        self.golden_weights_csv = golden_weights_csv  # Store for on-demand loading
        self.df_golden = df_golden  # Keep Polars DataFrame for fast filtering
        
        # Sample a few golden windows for reference baseline computation
        sample_size = min(50, len(golden_window_indices))
        sample_indices = np.random.choice(golden_window_indices, sample_size, replace=False)
        logger.info(f"Loading {sample_size} sample windows for reference baseline...")
        golden_samples = [load_weights_from_csv(golden_weights_csv, idx, lag, fixed_dim=d) 
                         for idx in sample_indices]
        
        W_golden_sum = sum(golden_samples)
        self.W_golden = W_golden_sum / len(golden_samples)
        logger.info(f"Reference averaged baseline (from {sample_size} samples): Frobenius norm = {norm(self.W_golden, 'fro'):.6e}")

        # Adaptive thresholds (will be computed from initial normal period)
        self.threshold_normal = None
        self.threshold_change = None
        self.threshold_trend = None

        # History for adaptive thresholding
        self.normal_abs_scores = []
        self.normal_change_scores = []

    def compute_abs_score(self, W_current: np.ndarray) -> float:
        """
        Compute absolute deviation from golden baseline using nearest-neighbor.
        Uses vectorized Polars operations for speed.

        Args:
            W_current: Current weight matrix

        Returns:
            Absolute score (minimum distance to any golden window)
        """
        # Nearest-neighbor approach: distance to closest golden window
        # For speed, sample only every 10th golden window (still gives good coverage)
        sample_indices = self.golden_window_indices[::10]  # Sample every 10th window
        
        min_distance = float('inf')
        for window_idx in sample_indices:
            W_g = _weights_cache.get_matrix(self.golden_weights_csv, window_idx, self.lag, self.fixed_dim)
            distance = compute_distance(W_current, W_g, self.metric)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def compute_change_score(self, W_current: np.ndarray, W_previous: np.ndarray) -> float:
        """
        Compute rate of change between consecutive windows.

        Args:
            W_current: Current weight matrix
            W_previous: Previous weight matrix

        Returns:
            Change score (distance from previous)
        """
        return compute_distance(W_current, W_previous, self.metric)

    def compute_trend(self, abs_scores: List[float], current_idx: int) -> float:
        """
        Compute trend in abs_score (getting better or worse).

        Args:
            abs_scores: List of absolute scores
            current_idx: Current index

        Returns:
            Trend value (positive = degrading, negative = improving)
        """
        if current_idx < self.lookback:
            return 0.0

        # Trend = current - past
        trend = abs_scores[current_idx] - abs_scores[current_idx - self.lookback]
        return trend

    def update_adaptive_thresholds(self, abs_score: float, change_score: float, is_normal: bool = True):
        """
        Update adaptive thresholds using rolling statistics from normal periods.

        Args:
            abs_score: Current absolute score
            change_score: Current change score
            is_normal: Whether this window is considered normal
        """
        if is_normal:
            self.normal_abs_scores.append(abs_score)
            self.normal_change_scores.append(change_score)

            # Keep only recent history (sliding window)
            max_history = 100
            if len(self.normal_abs_scores) > max_history:
                self.normal_abs_scores = self.normal_abs_scores[-max_history:]
                self.normal_change_scores = self.normal_change_scores[-max_history:]

    def compute_thresholds_adaptive(self, n_sigma: float = 3.0) -> Tuple[float, float, float]:
        """
        Compute adaptive thresholds using Hundman NPDT approach.

        Uses mean + n*std from recent normal windows.

        Args:
            n_sigma: Number of standard deviations

        Returns:
            Tuple of (threshold_normal, threshold_change, threshold_trend)
        """
        if len(self.normal_abs_scores) < 5:
            # Not enough data, use conservative defaults
            return 0.05, 0.15, 0.1

        # Threshold for abs_score: mean + n*std
        threshold_normal = np.mean(self.normal_abs_scores) + n_sigma * np.std(self.normal_abs_scores)

        # Threshold for change_score: mean + n*std
        threshold_change = np.mean(self.normal_change_scores) + n_sigma * np.std(self.normal_change_scores)

        # Threshold for trend: use fraction of abs threshold
        threshold_trend = threshold_normal * 0.2

        return threshold_normal, threshold_change, threshold_trend

    def classify_window(
        self,
        abs_score: float,
        change_score: float,
        abs_trend: float,
        use_adaptive: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        Classify anomaly status of current window.

        Args:
            abs_score: Absolute deviation from golden
            change_score: Rate of change from previous window
            abs_trend: Trend in abs_score
            use_adaptive: Use adaptive thresholds

        Returns:
            Tuple of (status, metrics_dict)
        """
        # Get thresholds
        if use_adaptive and len(self.normal_abs_scores) >= 5:
            threshold_normal, threshold_change, threshold_trend = self.compute_thresholds_adaptive()
        else:
            # Fixed thresholds (fallback)
            threshold_normal = 0.05
            threshold_change = 0.15
            threshold_trend = 0.1

        # Store for return
        metrics = {
            'abs_score': abs_score,
            'change_score': change_score,
            'abs_trend': abs_trend,
            'threshold_normal': threshold_normal,
            'threshold_change': threshold_change,
            'threshold_trend': threshold_trend
        }

        # Decision logic from todolist.md
        if abs_score < threshold_normal:
            status = "NORMAL"
            # Update adaptive thresholds with this normal sample
            self.update_adaptive_thresholds(abs_score, change_score, is_normal=True)
        elif change_score > threshold_change and abs_trend > threshold_trend:
            status = "NEW_ANOMALY_ONSET"  # Getting worse + changing
        elif change_score > threshold_change and abs_trend < -threshold_trend:
            status = "RECOVERY_FLUCTUATION"  # Getting better + changing
        else:
            status = "CASCADE_OR_PERSISTENT"  # Abnormal but stable

        return status, metrics

    def analyze_timeline(
        self,
        test_weights_csv: str,
        output_csv: Optional[str] = None,
        use_adaptive: bool = True
    ) -> pd.DataFrame:
        """
        Analyze full test timeline and detect anomalies.

        Args:
            test_weights_csv: Path to test timeline weights
            output_csv: Optional path to save results
            use_adaptive: Use adaptive thresholding

        Returns:
            DataFrame with window-by-window analysis
        """
        logger.info("="*80)
        logger.info("DUAL-METRIC ANOMALY DETECTION")
        logger.info("="*80)
        logger.info(f"Test timeline: {test_weights_csv}")
        logger.info(f"Golden baseline: {self.golden_weights_csv}")
        logger.info(f"Distance metric: {self.metric}")
        logger.info(f"Lag: {self.lag}")
        logger.info(f"Lookback: {self.lookback}")
        logger.info(f"Adaptive thresholding: {use_adaptive}")
        logger.info("")

        # Load test weights using Polars (fast)
        df_test = pl.read_csv(test_weights_csv)
        test_window_indices = sorted(df_test['window_idx'].unique().to_list())
        logger.info(f"Analyzing {len(test_window_indices)} test windows")

        # Storage for results
        results = []
        abs_scores = []
        W_previous = None

        for i, window_idx in enumerate(test_window_indices):
            # Load current window with consistent dimensions
            if self.ensemble_files:
                # Ensemble mode: Average weights across all runs
                W_sum = np.zeros((self.fixed_dim, self.fixed_dim))
                count = 0
                for fpath in self.ensemble_files:
                    try:
                        # We use the global cache helper, passing str(fpath)
                        W_run = _weights_cache.get_matrix(str(fpath), window_idx, self.lag, self.fixed_dim)
                        W_sum += W_run
                        count += 1
                    except Exception:
                        pass # Skip missing windows in some runs
                
                if count > 0:
                    W_current = W_sum / count
                else:
                    # Fallback if no runs have this window (shouldn't happen)
                    W_current = load_weights_from_csv(test_weights_csv, window_idx, self.lag, fixed_dim=self.fixed_dim)
            else:
                # Single run mode
                W_current = load_weights_from_csv(test_weights_csv, window_idx, self.lag, fixed_dim=self.fixed_dim)

            # Compute abs_score
            abs_score = self.compute_abs_score(W_current)
            abs_scores.append(abs_score)

            # Compute change_score (skip first window)
            if W_previous is not None:
                change_score = self.compute_change_score(W_current, W_previous)
            else:
                change_score = 0.0

            # Compute trend
            abs_trend = self.compute_trend(abs_scores, i)

            # Classify
            status, metrics = self.classify_window(abs_score, change_score, abs_trend, use_adaptive)

            # Get window metadata from CSV (Polars syntax)
            window_data = df_test.filter(pl.col('window_idx') == window_idx).row(0, named=True)
            t_end = int(window_data.get('t_end', window_idx))
            t_center = int(window_data.get('t_center', window_idx))

            # Store result
            results.append({
                'window_idx': window_idx,
                't_end': t_end,
                't_center': t_center,
                'status': status,
                **metrics
            })

            # Log progress
            if i % 10 == 0 or status != "NORMAL":
                logger.info(f"Window {window_idx:3d} (t={t_end:5d}): {status:25s} "
                          f"abs={abs_score:.4f} change={change_score:.4f} trend={abs_trend:+.4f}")

            # Update previous
            W_previous = W_current.copy()

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # Summary statistics
        logger.info("")
        logger.info("="*80)
        logger.info("DETECTION SUMMARY")
        logger.info("="*80)
        status_counts = df_results['status'].value_counts()
        for status, count in status_counts.items():
            pct = 100 * count / len(df_results)
            logger.info(f"{status:25s}: {count:4d} windows ({pct:5.1f}%)")

        # Anomaly windows (non-NORMAL)
        df_anomalies = df_results[df_results['status'] != 'NORMAL']
        logger.info("")
        logger.info(f"Total anomaly windows: {len(df_anomalies)}")
        if len(df_anomalies) > 0:
            logger.info(f"First anomaly: window {df_anomalies.iloc[0]['window_idx']} (t={df_anomalies.iloc[0]['t_end']})")
            logger.info(f"Last anomaly: window {df_anomalies.iloc[-1]['window_idx']} (t={df_anomalies.iloc[-1]['t_end']})")

        logger.info("="*80)

        # Save results
        if output_csv:
            df_results.to_csv(output_csv, index=False)
            logger.info(f"Results saved to: {output_csv}")

        return df_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Dual-Metric Anomaly Detection with Cascade Disambiguation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This implements the approach from todolist.md for detecting multiple anomalies
and distinguishing them from cascade effects and recovery fluctuations.

Three metrics per window:
  - abs_score: distance from golden baseline (how abnormal)
  - change_score: distance from previous window (how much changed)
  - abs_trend: trend in abs_score (getting worse or better)

Four classification states:
  - NORMAL: Low abs_score
  - NEW_ANOMALY_ONSET: High change + positive trend (degrading)
  - RECOVERY_FLUCTUATION: High change + negative trend (improving)
  - CASCADE_OR_PERSISTENT: High abs but low change (stable-abnormal)

Examples:
  # Basic usage
  python dual_metric_anomaly_detection.py \\
    --golden results/golden_baseline/weights/weights_enhanced.csv \\
    --test results/test_timeline/weights/weights_enhanced.csv \\
    --output anomaly_detection_results.csv

  # With custom parameters
  python dual_metric_anomaly_detection.py \\
    --golden results/golden_baseline/weights/weights_enhanced.csv \\
    --test results/test_timeline/weights/weights_enhanced.csv \\
    --metric spectral \\
    --lookback 10 \\
    --lag 0
        """
    )

    parser.add_argument('--golden', required=True, help='Golden baseline weights CSV')
    parser.add_argument('--test', required=True, help='Test timeline weights CSV')
    parser.add_argument('--output', help='Output CSV file (default: anomaly_detection_results.csv)')
    parser.add_argument('--metric', default='frobenius', choices=['frobenius', 'spectral', 'weighted_edge'],
                       help='Distance metric (default: frobenius)')
    parser.add_argument('--lookback', type=int, default=5, help='Lookback for trend (default: 5)')
    parser.add_argument('--lag', type=int, default=0, help='Which lag to analyze (default: 0)')
    parser.add_argument('--no-adaptive', action='store_true', help='Disable adaptive thresholding')
    parser.add_argument('--ensemble', help='Directory containing multiple run results for bagging (averaging)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.golden).exists():
        logger.error(f"Golden weights file not found: {args.golden}")
        return 1

    if not Path(args.test).exists():
        logger.error(f"Test weights file not found: {args.test}")
        return 1

    # Default output
    output_csv = args.output or 'anomaly_detection_results.csv'

    # Initialize detector
    detector = DualMetricAnomalyDetector(
        golden_weights_csv=args.golden,
        lookback=args.lookback,
        metric=args.metric,
        lag=args.lag,
        ensemble_dir=args.ensemble
    )

    # Analyze timeline
    df_results = detector.analyze_timeline(
        test_weights_csv=args.test,
        output_csv=output_csv,
        use_adaptive=not args.no_adaptive
    )

    logger.info("")
    logger.info("Analysis complete!")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
