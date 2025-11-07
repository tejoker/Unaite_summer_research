#!/usr/bin/env python3
"""
Robust Weight-Based Anomaly Detector
Uses 3-step strategy to eliminate false positives and identify true anomalies:
1. Adaptive thresholding based on baseline noise
2. Epicenter detection to find anomaly source
3. Anomaly characterization and root cause analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Single anomaly detection result with full causal information"""
    window_idx: int
    window_time_range: Tuple[int, int]  # (start_row, end_row) in original time series
    epicenter_sensor: str
    anomaly_type: str  # 'spike', 'level_shift', 'drift', 'trend_change', etc.

    # Magnitude and confidence
    max_weight_change: float
    epicenter_impact_score: float
    n_significant_edges: int

    # Top affected causal relationships
    top_edges: List[Dict]  # [{parent, child, lag, golden_weight, anomaly_weight, change}]

    # Root cause explanation
    root_cause_summary: str


class RobustWeightDetector:
    """
    Robust weight-based anomaly detector with adaptive thresholding
    """

    def __init__(
        self,
        baseline_noise_std: Optional[float] = None,
        baseline_noise_mean: Optional[float] = None,
        sigma_multiplier: float = 5.0,
        min_epicenter_impact: float = 10.0,
        top_edges_to_report: int = 5
    ):
        """
        Args:
            baseline_noise_std: Standard deviation of baseline noise (from Golden-Golden comparison)
            baseline_noise_mean: Mean of baseline noise
            sigma_multiplier: Threshold = mean + sigma_multiplier * std (default: 5.0)
            min_epicenter_impact: Minimum impact score to consider as epicenter
            top_edges_to_report: Number of top changed edges to report
        """
        self.baseline_noise_std = baseline_noise_std
        self.baseline_noise_mean = baseline_noise_mean
        self.sigma_multiplier = sigma_multiplier
        self.min_epicenter_impact = min_epicenter_impact
        self.top_edges_to_report = top_edges_to_report

        self.adaptive_threshold = None
        if baseline_noise_std is not None and baseline_noise_mean is not None:
            self.adaptive_threshold = baseline_noise_mean + sigma_multiplier * baseline_noise_std
            logger.info(f"Adaptive threshold set to: {self.adaptive_threshold:.6f} "
                       f"(μ={baseline_noise_mean:.6f}, σ={baseline_noise_std:.6f})")

    @staticmethod
    def calculate_baseline_noise(
        golden_weights_1: pd.DataFrame,
        golden_weights_2: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Step 0: Calculate baseline noise by comparing two Golden runs
        This establishes the "normal operational noise" of DynoTEARS

        Args:
            golden_weights_1: First Golden run weights
            golden_weights_2: Second Golden run weights

        Returns:
            (mean, std) of weight differences between identical runs
        """
        logger.info("Calculating baseline noise from Golden-Golden comparison...")

        all_diffs = []

        for window_idx in golden_weights_1['window_idx'].unique():
            g1_window = golden_weights_1[golden_weights_1['window_idx'] == window_idx]
            g2_window = golden_weights_2[golden_weights_2['window_idx'] == window_idx]

            # Merge on child-parent-lag
            merged = pd.merge(
                g1_window,
                g2_window,
                on=['child_name', 'parent_name', 'lag'],
                suffixes=('_1', '_2'),
                how='inner'
            )

            # Calculate absolute differences
            merged['abs_diff'] = abs(merged['weight_1'] - merged['weight_2'])
            all_diffs.extend(merged['abs_diff'].values)

        all_diffs = np.array(all_diffs)
        mean = all_diffs.mean()
        std = all_diffs.std()

        logger.info(f"Baseline noise: μ={mean:.6f}, σ={std:.6f}")
        logger.info(f"Recommended threshold (μ + 5σ): {mean + 5*std:.6f}")

        return mean, std

    def detect_anomalies(
        self,
        golden_weights: pd.DataFrame,
        anomaly_weights: pd.DataFrame,
        window_size: int = 100,
        window_overlap: int = 90
    ) -> List[AnomalyEvent]:
        """
        Detect anomalies using the 3-step robust strategy

        Args:
            golden_weights: Baseline weight matrix
            anomaly_weights: Anomaly weight matrix to compare
            window_size: Size of rolling windows in original time series
            window_overlap: Overlap between windows

        Returns:
            List of detected anomaly events
        """
        if self.adaptive_threshold is None:
            logger.warning("No adaptive threshold set - using fallback threshold of 0.5")
            self.adaptive_threshold = 0.5

        detections = []

        # Process each window
        for window_idx in sorted(golden_weights['window_idx'].unique()):
            event = self._analyze_window(
                window_idx=window_idx,
                golden_weights=golden_weights,
                anomaly_weights=anomaly_weights,
                window_size=window_size,
                window_overlap=window_overlap
            )

            if event is not None:
                detections.append(event)

        logger.info(f"Detected {len(detections)} significant anomalies across all windows")
        return detections

    def _analyze_window(
        self,
        window_idx: int,
        golden_weights: pd.DataFrame,
        anomaly_weights: pd.DataFrame,
        window_size: int,
        window_overlap: int
    ) -> Optional[AnomalyEvent]:
        """
        Analyze a single window using 3-step strategy

        Step 1: Adaptive thresholding to filter noise
        Step 2: Epicenter detection to find root cause sensor
        Step 3: Characterization and explanation
        """
        golden_window = golden_weights[golden_weights['window_idx'] == window_idx]
        anomaly_window = anomaly_weights[anomaly_weights['window_idx'] == window_idx]

        if len(anomaly_window) == 0:
            return None

        # Merge weights
        merged = pd.merge(
            golden_window,
            anomaly_window,
            on=['child_name', 'parent_name', 'lag'],
            suffixes=('_golden', '_anomaly'),
            how='outer'
        ).fillna(0)

        # Calculate differences
        merged['abs_diff'] = abs(merged['weight_anomaly'] - merged['weight_golden'])

        # STEP 1: Adaptive Thresholding
        significant = merged[merged['abs_diff'] > self.adaptive_threshold].copy()

        if len(significant) == 0:
            return None  # No significant changes in this window

        logger.debug(f"Window {window_idx}: {len(significant)} edges pass threshold "
                    f"(out of {len(merged)} total)")

        # STEP 2: Epicenter Detection
        epicenter_sensor, epicenter_score, sensor_scores = self._find_epicenter(significant)

        if epicenter_score < self.min_epicenter_impact:
            logger.debug(f"Window {window_idx}: Max impact score {epicenter_score:.2f} "
                        f"below threshold {self.min_epicenter_impact}")
            return None

        # STEP 3: Characterization
        anomaly_type = self._classify_anomaly_type(significant, epicenter_sensor)

        # Get top affected edges
        epicenter_edges = significant[
            (significant['child_name'].str.contains(epicenter_sensor.replace('_diff', ''))) |
            (significant['parent_name'].str.contains(epicenter_sensor.replace('_diff', '')))
        ].copy()

        epicenter_edges['change_ratio'] = np.where(
            abs(epicenter_edges['weight_golden']) > 1e-6,
            epicenter_edges['weight_anomaly'] / epicenter_edges['weight_golden'],
            np.inf
        )

        top_edges_df = epicenter_edges.nlargest(self.top_edges_to_report, 'abs_diff')

        top_edges = []
        for _, row in top_edges_df.iterrows():
            top_edges.append({
                'parent': row['parent_name'],
                'child': row['child_name'],
                'lag': int(row['lag']),
                'golden_weight': float(row['weight_golden']),
                'anomaly_weight': float(row['weight_anomaly']),
                'change': float(row['abs_diff']),
                'ratio': float(row['change_ratio'])
            })

        # Calculate time range in original series
        window_start = window_idx * (window_size - window_overlap)
        window_end = window_start + window_size

        # Generate root cause summary
        root_cause = self._generate_root_cause_summary(
            window_idx=window_idx,
            epicenter_sensor=epicenter_sensor,
            anomaly_type=anomaly_type,
            top_edge=top_edges[0] if top_edges else None,
            n_affected=len(epicenter_edges)
        )

        event = AnomalyEvent(
            window_idx=window_idx,
            window_time_range=(window_start, window_end),
            epicenter_sensor=epicenter_sensor,
            anomaly_type=anomaly_type,
            max_weight_change=float(significant['abs_diff'].max()),
            epicenter_impact_score=float(epicenter_score),
            n_significant_edges=len(epicenter_edges),
            top_edges=top_edges,
            root_cause_summary=root_cause
        )

        return event

    def _find_epicenter(
        self, significant_edges: pd.DataFrame
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Step 2: Find the epicenter sensor with highest impact score

        Impact score = sum of absolute weight changes involving that sensor
        """
        sensor_scores = {}

        for _, row in significant_edges.iterrows():
            child = row['child_name']
            parent = row['parent_name']
            change = row['abs_diff']

            # Add to both child and parent scores
            sensor_scores[child] = sensor_scores.get(child, 0) + change
            sensor_scores[parent] = sensor_scores.get(parent, 0) + change

        # Find sensor with highest score
        epicenter = max(sensor_scores, key=sensor_scores.get)
        epicenter_score = sensor_scores[epicenter]

        logger.debug(f"Epicenter: {epicenter} (score: {epicenter_score:.2f})")

        return epicenter, epicenter_score, sensor_scores

    def _classify_anomaly_type(
        self, significant_edges: pd.DataFrame, epicenter_sensor: str
    ) -> str:
        """
        Step 3: Classify anomaly type based on magnitude pattern
        """
        max_change = significant_edges['abs_diff'].max()
        mean_change = significant_edges['abs_diff'].mean()

        # Classification based on magnitude
        if max_change > 10.0:
            return 'spike'
        elif max_change > 2.0:
            return 'level_shift'
        elif max_change > 0.5:
            return 'drift_or_trend'
        else:
            return 'minor_change'

    def _generate_root_cause_summary(
        self,
        window_idx: int,
        epicenter_sensor: str,
        anomaly_type: str,
        top_edge: Optional[Dict],
        n_affected: int
    ) -> str:
        """
        Step 3: Generate human-readable root cause explanation
        """
        sensor_name = epicenter_sensor.replace('_diff', '')

        if top_edge is None:
            return f"Anomaly detected at window {window_idx}, centered on '{sensor_name}'"

        edge_desc = f"{top_edge['parent'].replace('_diff', '')} → {top_edge['child'].replace('_diff', '')}"
        if top_edge['lag'] > 0:
            edge_desc += f" (lag {top_edge['lag']})"

        change_desc = f"changed from {top_edge['golden_weight']:.4f} to {top_edge['anomaly_weight']:.4f}"

        if abs(top_edge['golden_weight']) > 1e-6:
            ratio = top_edge['ratio']
            if abs(ratio) > 100:
                change_desc += f" ({ratio:.0f}x)"
            elif abs(ratio) > 2:
                change_desc += f" ({ratio:.1f}x)"

        summary = (
            f"A **{anomaly_type}** was detected at **window {window_idx}**, "
            f"centered on **'{sensor_name}'**. "
            f"The root cause is a structural break where the causal link "
            f"**{edge_desc}** {change_desc}. "
            f"Total {n_affected} causal relationships significantly affected."
        )

        return summary

    def save_results(
        self, detections: List[AnomalyEvent], output_path: Path
    ):
        """Save detection results to CSV with full details"""
        if not detections:
            logger.warning("No detections to save")
            return

        records = []
        for event in detections:
            # Main record
            base_record = {
                'window_idx': event.window_idx,
                'time_range_start': event.window_time_range[0],
                'time_range_end': event.window_time_range[1],
                'epicenter_sensor': event.epicenter_sensor,
                'anomaly_type': event.anomaly_type,
                'max_weight_change': event.max_weight_change,
                'epicenter_impact_score': event.epicenter_impact_score,
                'n_significant_edges': event.n_significant_edges,
                'root_cause': event.root_cause_summary
            }

            # Add top edges
            for i, edge in enumerate(event.top_edges[:3]):  # Top 3 for CSV
                base_record[f'edge{i+1}'] = f"{edge['parent']} → {edge['child']}"
                base_record[f'edge{i+1}_change'] = edge['change']
                base_record[f'edge{i+1}_ratio'] = edge['ratio']

            records.append(base_record)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(detections)} detections to {output_path}")

        # Print summary
        self._print_summary(detections)

    def _print_summary(self, detections: List[AnomalyEvent]):
        """Print detection summary"""
        print("\n" + "="*100)
        print("ROBUST WEIGHT-BASED ANOMALY DETECTION RESULTS")
        print("="*100)

        if not detections:
            print("\nNo significant anomalies detected.")
            print("="*100)
            return

        print(f"\nTotal significant anomalies detected: {len(detections)}")
        print(f"Adaptive threshold used: {self.adaptive_threshold:.6f}")
        print("\n" + "-"*100)

        for i, event in enumerate(detections, 1):
            print(f"\n[{i}] Window {event.window_idx} (rows {event.window_time_range[0]}-{event.window_time_range[1]})")
            print(f"    Epicenter: {event.epicenter_sensor}")
            print(f"    Type: {event.anomaly_type}")
            print(f"    Impact Score: {event.epicenter_impact_score:.2f}")
            print(f"    Max Weight Change: {event.max_weight_change:.4f}")
            print(f"    Affected Edges: {event.n_significant_edges}")

            if event.top_edges:
                print(f"\n    Top Causal Changes:")
                for j, edge in enumerate(event.top_edges[:3], 1):
                    ratio_str = f" ({edge['ratio']:.1f}x)" if abs(edge['ratio']) < 1000 else f" ({edge['ratio']:.0f}x)"
                    print(f"      {j}. {edge['parent']} → {edge['child']} (lag {edge['lag']}): "
                          f"{edge['golden_weight']:.4f} → {edge['anomaly_weight']:.4f}{ratio_str}")

            print(f"\n    Root Cause: {event.root_cause_summary}")
            print("    " + "-"*96)

        print("="*100)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Robust Weight-Based Anomaly Detection')
    parser.add_argument('--golden-weights', required=True, help='Golden baseline weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Anomaly weights CSV to analyze')
    parser.add_argument('--golden-weights-2', help='Second golden run for noise calculation (optional)')
    parser.add_argument('--output', required=True, help='Output path for results')
    parser.add_argument('--sigma-multiplier', type=float, default=5.0, help='Threshold sigma multiplier')
    parser.add_argument('--window-size', type=int, default=100, help='Rolling window size')
    parser.add_argument('--window-overlap', type=int, default=90, help='Rolling window overlap')

    args = parser.parse_args()

    # Load weights
    golden_weights = pd.read_csv(args.golden_weights)
    anomaly_weights = pd.read_csv(args.anomaly_weights)

    # Calculate baseline noise if second golden run provided
    baseline_mean = None
    baseline_std = None

    if args.golden_weights_2:
        golden_weights_2 = pd.read_csv(args.golden_weights_2)
        baseline_mean, baseline_std = RobustWeightDetector.calculate_baseline_noise(
            golden_weights, golden_weights_2
        )

    # Create detector
    detector = RobustWeightDetector(
        baseline_noise_mean=baseline_mean,
        baseline_noise_std=baseline_std,
        sigma_multiplier=args.sigma_multiplier
    )

    # Detect anomalies
    detections = detector.detect_anomalies(
        golden_weights=golden_weights,
        anomaly_weights=anomaly_weights,
        window_size=args.window_size,
        window_overlap=args.window_overlap
    )

    # Save results
    detector.save_results(detections, Path(args.output))
