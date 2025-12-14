#!/usr/bin/env python3
"""
Benchmark Tucker-CAM Causal Graphs: Golden Baseline vs Test Timeline

This script compares the causal graph structures learned from:
- Golden baseline (normal operation windows)
- Test timeline (contains anomaly windows)

Metrics computed:
1. Structural Hamming Distance (SHD) - edge differences
2. Frobenius distance - weight differences
3. Edge set Jaccard similarity
4. Per-window graph statistics
5. Anomaly detection based on graph distance
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy.linalg import norm
from scipy.stats import ks_2samp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class GraphBenchmark:
    """Benchmark causal graphs from Tucker-CAM."""

    def __init__(self, golden_csv: str, test_csv: str, lag: int = 0):
        """
        Initialize benchmark.

        Args:
            golden_csv: Path to golden baseline weights_enhanced.csv
            test_csv: Path to test timeline weights_enhanced.csv
            lag: Which lag to analyze (0=contemporaneous)
        """
        self.golden_csv = golden_csv
        self.test_csv = test_csv
        self.lag = lag

        # Load data
        logger.info(f"Loading golden baseline: {golden_csv}")
        self.df_golden = pl.read_csv(golden_csv)
        logger.info(f"  Loaded {self.df_golden.height:,} edges")

        logger.info(f"Loading test timeline: {test_csv}")
        self.df_test = pl.read_csv(test_csv)
        logger.info(f"  Loaded {self.df_test.height:,} edges")

        # Get window lists
        self.golden_windows = sorted(self.df_golden['window_idx'].unique().to_list())
        self.test_windows = sorted(self.df_test['window_idx'].unique().to_list())

        logger.info(f"Golden windows: {len(self.golden_windows)} (windows {self.golden_windows[0]}-{self.golden_windows[-1]})")
        logger.info(f"Test windows: {len(self.test_windows)} (windows {self.test_windows[0]}-{self.test_windows[-1]})")

        # Determine matrix dimension
        df_all = pl.concat([
            self.df_golden.filter(pl.col('lag') == lag),
            self.df_test.filter(pl.col('lag') == lag)
        ])
        max_i = int(df_all['i'].max())
        max_j = int(df_all['j'].max())
        self.dim = max(max_i, max_j) + 1
        logger.info(f"Matrix dimension: {self.dim}x{self.dim}")

    def load_window_graph(self, df: pl.DataFrame, window_idx: int) -> np.ndarray:
        """Load weight matrix for a specific window."""
        df_window = df.filter((pl.col('window_idx') == window_idx) & (pl.col('lag') == self.lag))

        W = np.zeros((self.dim, self.dim))
        if df_window.height > 0:
            for row in df_window.iter_rows(named=True):
                i, j = int(row['i']), int(row['j'])
                if i < self.dim and j < self.dim:
                    W[i, j] = row['weight']

        return W

    def compute_shd(self, W1: np.ndarray, W2: np.ndarray, threshold: float = 1e-6) -> int:
        """
        Structural Hamming Distance: number of edge differences.

        Args:
            W1, W2: Weight matrices
            threshold: Edge existence threshold

        Returns:
            SHD (number of different edges)
        """
        edges1 = np.abs(W1) > threshold
        edges2 = np.abs(W2) > threshold

        # Edges in W1 but not W2 + edges in W2 but not W1
        shd = np.sum(edges1 != edges2)
        return int(shd)

    def compute_jaccard(self, W1: np.ndarray, W2: np.ndarray, threshold: float = 1e-6) -> float:
        """
        Jaccard similarity of edge sets.

        Returns:
            Jaccard similarity in [0, 1]
        """
        edges1 = np.abs(W1) > threshold
        edges2 = np.abs(W2) > threshold

        intersection = np.sum(edges1 & edges2)
        union = np.sum(edges1 | edges2)

        if union == 0:
            return 1.0  # Both empty

        return intersection / union

    def compute_frobenius_distance(self, W1: np.ndarray, W2: np.ndarray) -> float:
        """Frobenius norm distance."""
        return norm(W1 - W2, ord='fro')

    def graph_statistics(self, W: np.ndarray, threshold: float = 1e-6) -> Dict:
        """Compute graph statistics."""
        edges = np.abs(W) > threshold
        weights = W[edges]

        return {
            'n_edges': int(np.sum(edges)),
            'density': np.sum(edges) / (self.dim * (self.dim - 1)),  # Exclude diagonal
            'mean_weight': np.abs(W).mean(),
            'max_weight': np.abs(W).max(),
            'nonzero_mean_weight': np.abs(weights).mean() if len(weights) > 0 else 0,
            'frobenius_norm': norm(W, ord='fro')
        }

    def compare_distributions(self) -> Dict:
        """Compare weight distributions between golden and test."""
        logger.info("Comparing weight distributions...")

        # Get all weights
        golden_weights = self.df_golden.filter(pl.col('lag') == self.lag)['weight'].to_numpy()
        test_weights = self.df_test.filter(pl.col('lag') == self.lag)['weight'].to_numpy()

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(np.abs(golden_weights), np.abs(test_weights))

        return {
            'golden_mean': np.abs(golden_weights).mean(),
            'golden_std': np.abs(golden_weights).std(),
            'golden_max': np.abs(golden_weights).max(),
            'test_mean': np.abs(test_weights).mean(),
            'test_std': np.abs(test_weights).std(),
            'test_max': np.abs(test_weights).max(),
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        }

    def benchmark_window_similarity(self, sample_size: int = 50) -> pd.DataFrame:
        """
        Benchmark similarity between test and golden windows.

        For each test window, find the nearest golden window and compute metrics.
        """
        logger.info(f"Benchmarking window similarity (sampling {sample_size} test windows)...")

        # Sample test windows evenly
        step = max(1, len(self.test_windows) // sample_size)
        sampled_test = self.test_windows[::step][:sample_size]

        results = []

        for i, test_idx in enumerate(sampled_test):
            W_test = self.load_window_graph(self.df_test, test_idx)

            # Find nearest golden window
            min_distance = float('inf')
            best_golden_idx = None

            for golden_idx in self.golden_windows:
                W_golden = self.load_window_graph(self.df_golden, golden_idx)
                distance = self.compute_frobenius_distance(W_test, W_golden)

                if distance < min_distance:
                    min_distance = distance
                    best_golden_idx = golden_idx

            # Compute metrics with best match
            W_golden_best = self.load_window_graph(self.df_golden, best_golden_idx)

            shd = self.compute_shd(W_test, W_golden_best)
            jaccard = self.compute_jaccard(W_test, W_golden_best)
            frob_dist = min_distance

            test_stats = self.graph_statistics(W_test)

            results.append({
                'test_window': test_idx,
                'nearest_golden_window': best_golden_idx,
                'frobenius_distance': frob_dist,
                'shd': shd,
                'jaccard_similarity': jaccard,
                'test_n_edges': test_stats['n_edges'],
                'test_mean_weight': test_stats['mean_weight'],
                'test_max_weight': test_stats['max_weight'],
                'test_frobenius_norm': test_stats['frobenius_norm']
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(sampled_test)} windows")

        return pd.DataFrame(results)

    def run_benchmark(self, output_dir: str = 'results/benchmark'):
        """Run complete benchmark and save results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        logger.info("="*80)
        logger.info("TUCKER-CAM CAUSAL GRAPH BENCHMARK")
        logger.info("="*80)
        logger.info("")

        # 1. Weight distribution comparison
        logger.info("1. Weight Distribution Comparison")
        logger.info("-" * 40)
        dist_stats = self.compare_distributions()
        for key, value in dist_stats.items():
            if 'pvalue' in key or 'statistic' in key:
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value:.6e}")
        logger.info("")

        # 2. Window-by-window similarity
        logger.info("2. Window-by-Window Similarity")
        logger.info("-" * 40)
        df_similarity = self.benchmark_window_similarity(sample_size=50)

        logger.info(f"  Mean Frobenius distance: {df_similarity['frobenius_distance'].mean():.6e}")
        logger.info(f"  Mean SHD: {df_similarity['shd'].mean():.1f}")
        logger.info(f"  Mean Jaccard similarity: {df_similarity['jaccard_similarity'].mean():.4f}")
        logger.info("")

        # Save similarity results
        similarity_file = output_path / 'window_similarity.csv'
        df_similarity.to_csv(similarity_file, index=False)
        logger.info(f"  Saved: {similarity_file}")
        logger.info("")

        # 3. Per-window graph statistics
        logger.info("3. Graph Statistics by Window Type")
        logger.info("-" * 40)

        # Sample golden windows
        golden_stats = []
        for w in self.golden_windows[::10][:20]:  # Sample every 10th, max 20
            W = self.load_window_graph(self.df_golden, w)
            stats = self.graph_statistics(W)
            stats['window'] = w
            stats['type'] = 'golden'
            golden_stats.append(stats)

        # Sample test windows
        test_stats = []
        for w in self.test_windows[::10][:20]:  # Sample every 10th, max 20
            W = self.load_window_graph(self.df_test, w)
            stats = self.graph_statistics(W)
            stats['window'] = w
            stats['type'] = 'test'
            test_stats.append(stats)

        df_stats = pd.DataFrame(golden_stats + test_stats)

        # Aggregate by type
        logger.info("  Golden baseline:")
        for col in ['n_edges', 'mean_weight', 'max_weight', 'frobenius_norm']:
            mean_val = df_stats[df_stats['type'] == 'golden'][col].mean()
            logger.info(f"    Mean {col}: {mean_val:.6e}" if 'weight' in col else f"    Mean {col}: {mean_val:.1f}")

        logger.info("  Test timeline:")
        for col in ['n_edges', 'mean_weight', 'max_weight', 'frobenius_norm']:
            mean_val = df_stats[df_stats['type'] == 'test'][col].mean()
            logger.info(f"    Mean {col}: {mean_val:.6e}" if 'weight' in col else f"    Mean {col}: {mean_val:.1f}")

        # Save stats
        stats_file = output_path / 'graph_statistics.csv'
        df_stats.to_csv(stats_file, index=False)
        logger.info(f"  Saved: {stats_file}")
        logger.info("")

        # 4. Summary
        logger.info("="*80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*80)
        logger.info(f"Golden windows analyzed: {len(self.golden_windows)}")
        logger.info(f"Test windows analyzed: {len(self.test_windows)}")
        logger.info(f"Matrix dimension: {self.dim}x{self.dim}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("")

        return {
            'distribution': dist_stats,
            'similarity': df_similarity,
            'statistics': df_stats
        }


def main():
    """Main entry point."""
    golden_csv = 'results/golden_baseline/weights/weights_enhanced.csv'
    test_csv = 'results/test_timeline/weights/weights_enhanced.csv'

    if not Path(golden_csv).exists():
        logger.error(f"Golden baseline not found: {golden_csv}")
        return 1

    if not Path(test_csv).exists():
        logger.error(f"Test timeline not found: {test_csv}")
        return 1

    # Run benchmark
    benchmark = GraphBenchmark(golden_csv, test_csv, lag=0)
    results = benchmark.run_benchmark()

    logger.info("Benchmark complete!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
