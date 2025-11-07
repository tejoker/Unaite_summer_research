#!/usr/bin/env python3
"""
Unified Anomaly Detection Pipeline
Combines three complementary detection methods:
1. Weight-based (causal structure changes) - for spike, level_shift
2. Double differencing (2nd derivative spikes) - for drift, trend_change
3. Rolling volatility (variance tracking) - for amplitude_change, variance_burst
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetection:
    """Single anomaly detection result"""
    method: str  # 'weight_based', 'double_diff', 'rolling_volatility'
    anomaly_type: str  # 'spike', 'level_shift', 'drift', 'trend_change', 'amplitude_change', 'variance_burst'
    window_idx: Optional[int]  # For weight-based method
    row_idx: Optional[int]  # For statistical methods
    sensor: str
    magnitude: float
    confidence: float  # In standard deviations or ratio
    details: Dict


class UnifiedAnomalyDetector:
    """
    Unified anomaly detector combining three methods for comprehensive coverage
    """

    def __init__(
        self,
        double_diff_threshold: float = 3.0,
        volatility_window: int = 30,
        volatility_spike_threshold: float = 3.0,
        volatility_changepoint_pen: float = 1.5,
        weight_ratio_threshold: float = 2.0,
    ):
        self.double_diff_threshold = double_diff_threshold
        self.volatility_window = volatility_window
        self.volatility_spike_threshold = volatility_spike_threshold
        self.volatility_changepoint_pen = volatility_changepoint_pen
        self.weight_ratio_threshold = weight_ratio_threshold

    def detect_all(
        self,
        raw_data: pd.DataFrame,
        differenced_data: pd.DataFrame,
        golden_weights: Optional[pd.DataFrame] = None,
        anomaly_weights: Optional[pd.DataFrame] = None,
    ) -> List[AnomalyDetection]:
        """
        Run all three detection methods and combine results

        Args:
            raw_data: Original time series data
            differenced_data: First-order differenced data
            golden_weights: Baseline weight matrix (optional, for weight-based detection)
            anomaly_weights: Anomaly weight matrix (optional, for weight-based detection)

        Returns:
            List of all detected anomalies from all methods
        """
        all_detections = []

        # Method 1: Weight-based detection (if weights provided)
        if golden_weights is not None and anomaly_weights is not None:
            logger.info("Running Method 1: Weight-based detection...")
            weight_detections = self._detect_weight_based(golden_weights, anomaly_weights)
            all_detections.extend(weight_detections)
            logger.info(f"  Found {len(weight_detections)} anomalies via weight-based method")

        # Method 2: Double differencing for trend anomalies
        logger.info("Running Method 2: Double differencing detection...")
        double_diff_detections = self._detect_double_diff(raw_data)
        all_detections.extend(double_diff_detections)
        logger.info(f"  Found {len(double_diff_detections)} anomalies via double differencing")

        # Method 3: Rolling volatility for variance anomalies
        logger.info("Running Method 3: Rolling volatility detection...")
        volatility_detections = self._detect_volatility_anomalies(differenced_data)
        all_detections.extend(volatility_detections)
        logger.info(f"  Found {len(volatility_detections)} anomalies via rolling volatility")

        logger.info(f"\nTotal anomalies detected: {len(all_detections)}")
        return all_detections

    def _detect_weight_based(
        self, golden_weights: pd.DataFrame, anomaly_weights: pd.DataFrame
    ) -> List[AnomalyDetection]:
        """
        Method 1: Detect anomalies via causal weight changes
        Detects: spike, level_shift (instantaneous changes)
        """
        detections = []

        # Group by window
        for window_idx in golden_weights['window_idx'].unique():
            golden_window = golden_weights[golden_weights['window_idx'] == window_idx]
            anomaly_window = anomaly_weights[anomaly_weights['window_idx'] == window_idx]

            if len(anomaly_window) == 0:
                continue

            # Merge on child-parent pairs (actual column names in weight files)
            merged = pd.merge(
                golden_window,
                anomaly_window,
                on=['child_name', 'parent_name', 'lag'],
                suffixes=('_golden', '_anomaly'),
                how='outer'
            ).fillna(0)

            # Calculate weight ratios
            merged['abs_diff'] = abs(merged['weight_anomaly'] - merged['weight_golden'])
            merged['ratio'] = np.where(
                abs(merged['weight_golden']) > 1e-6,
                merged['weight_anomaly'] / merged['weight_golden'],
                np.inf
            )

            # Find significant changes
            significant = merged[
                (merged['abs_diff'] > 0.1) &
                (abs(merged['ratio']) > self.weight_ratio_threshold)
            ]

            if len(significant) > 0:
                # Calculate window statistics
                max_diff = merged['abs_diff'].max()
                mean_diff = merged['abs_diff'].mean()
                n_large = len(significant)

                # Classify anomaly type based on pattern
                if max_diff > 10.0:  # Very large sudden change
                    anomaly_type = 'spike'
                elif max_diff > 2.0:
                    anomaly_type = 'level_shift'
                else:
                    anomaly_type = 'unknown'

                # Find most affected sensor
                max_change_row = merged.loc[merged['abs_diff'].idxmax()]
                sensor = max_change_row['child_name']

                detection = AnomalyDetection(
                    method='weight_based',
                    anomaly_type=anomaly_type,
                    window_idx=int(window_idx),
                    row_idx=None,
                    sensor=sensor,
                    magnitude=float(max_diff),
                    confidence=float(abs(max_change_row['ratio'])),
                    details={
                        'max_diff': float(max_diff),
                        'mean_diff': float(mean_diff),
                        'n_significant_edges': int(n_large),
                        'top_edge': f"{max_change_row['parent_name']} → {max_change_row['child_name']} (lag={max_change_row['lag']})"
                    }
                )
                detections.append(detection)

        return detections

    def _detect_double_diff(self, raw_data: pd.DataFrame) -> List[AnomalyDetection]:
        """
        Method 2: Detect trend anomalies via double differencing
        Detects: drift, trend_change (gradual changes with sharp start)
        """
        detections = []

        for col in raw_data.columns:
            series = raw_data[col].values

            # Apply double differencing
            diff1 = np.diff(series)
            diff2 = np.diff(diff1)

            # Calculate z-scores
            mean = diff2.mean()
            std = diff2.std()

            if std < 1e-10:
                continue

            z_scores = np.abs((diff2 - mean) / std)

            # Find spikes above threshold
            spike_indices = np.where(z_scores > self.double_diff_threshold)[0]

            # Cluster nearby spikes (within 5 indices)
            if len(spike_indices) > 0:
                clusters = self._cluster_indices(spike_indices, max_gap=5)

                for cluster in clusters:
                    # Take the strongest spike in cluster
                    cluster_indices = spike_indices[cluster]
                    max_idx_in_cluster = cluster_indices[np.argmax(z_scores[cluster_indices])]

                    # Adjust index for double differencing (add 2)
                    original_idx = int(max_idx_in_cluster + 2)

                    # Classify as drift or trend_change based on magnitude
                    magnitude = float(abs(diff2[max_idx_in_cluster]))
                    if magnitude > 1.0:
                        anomaly_type = 'drift'
                    else:
                        anomaly_type = 'trend_change'

                    detection = AnomalyDetection(
                        method='double_diff',
                        anomaly_type=anomaly_type,
                        window_idx=None,
                        row_idx=original_idx,
                        sensor=col,
                        magnitude=magnitude,
                        confidence=float(z_scores[max_idx_in_cluster]),
                        details={
                            'z_score': float(z_scores[max_idx_in_cluster]),
                            'cluster_size': len(cluster_indices),
                            'diff2_value': float(diff2[max_idx_in_cluster])
                        }
                    )
                    detections.append(detection)

        return detections

    def _detect_volatility_anomalies(self, differenced_data: pd.DataFrame) -> List[AnomalyDetection]:
        """
        Method 3: Detect variance anomalies via rolling volatility
        Detects: amplitude_change (level shift in volatility), variance_burst (spike in volatility)
        """
        detections = []

        for col in differenced_data.columns:
            series = differenced_data[col].values

            # Step 1: Calculate rolling standard deviation (volatility signal)
            volatility_signal = pd.Series(series).rolling(
                window=self.volatility_window,
                min_periods=self.volatility_window // 2
            ).std().values

            # Remove NaN values
            valid_mask = ~np.isnan(volatility_signal)
            volatility_signal_clean = volatility_signal[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            if len(volatility_signal_clean) < 50:
                continue

            # Step 2: Detect variance bursts (spikes in volatility)
            vol_mean = volatility_signal_clean.mean()
            vol_std = volatility_signal_clean.std()

            if vol_std < 1e-10:
                continue

            vol_z_scores = (volatility_signal_clean - vol_mean) / vol_std

            # Find spikes
            spike_indices = np.where(vol_z_scores > self.volatility_spike_threshold)[0]

            if len(spike_indices) > 0:
                clusters = self._cluster_indices(spike_indices, max_gap=5)

                for cluster in clusters:
                    cluster_indices = spike_indices[cluster]
                    max_idx_in_cluster = cluster_indices[np.argmax(vol_z_scores[cluster_indices])]
                    original_idx = int(valid_indices[max_idx_in_cluster])

                    detection = AnomalyDetection(
                        method='rolling_volatility',
                        anomaly_type='variance_burst',
                        window_idx=None,
                        row_idx=original_idx,
                        sensor=col,
                        magnitude=float(volatility_signal_clean[max_idx_in_cluster]),
                        confidence=float(vol_z_scores[max_idx_in_cluster]),
                        details={
                            'volatility_z_score': float(vol_z_scores[max_idx_in_cluster]),
                            'cluster_size': len(cluster_indices),
                            'window_size': self.volatility_window
                        }
                    )
                    detections.append(detection)

            # Step 3: Detect amplitude changes (level shifts in volatility)
            try:
                import ruptures as rpt

                # Use PELT to find change points in volatility signal
                algo = rpt.Pelt(model="l2", min_size=20).fit(volatility_signal_clean)
                pen = np.log(len(volatility_signal_clean)) * self.volatility_changepoint_pen
                change_points = algo.predict(pen=pen)

                # Remove the last point (end of series)
                change_points = [cp for cp in change_points if cp < len(volatility_signal_clean)]

                for cp in change_points:
                    # Check if volatility actually increased
                    before_mean = volatility_signal_clean[max(0, cp-20):cp].mean()
                    after_mean = volatility_signal_clean[cp:min(len(volatility_signal_clean), cp+20)].mean()

                    magnitude = abs(after_mean - before_mean)

                    if magnitude > vol_std * 0.5:  # Significant change
                        original_idx = int(valid_indices[cp])

                        detection = AnomalyDetection(
                            method='rolling_volatility',
                            anomaly_type='amplitude_change',
                            window_idx=None,
                            row_idx=original_idx,
                            sensor=col,
                            magnitude=float(magnitude),
                            confidence=float(magnitude / vol_std),
                            details={
                                'volatility_before': float(before_mean),
                                'volatility_after': float(after_mean),
                                'change_magnitude': float(magnitude),
                                'window_size': self.volatility_window
                            }
                        )
                        detections.append(detection)

            except ImportError:
                logger.warning("ruptures library not available - skipping amplitude_change detection")

        return detections

    @staticmethod
    def _cluster_indices(indices: np.ndarray, max_gap: int = 5) -> List[np.ndarray]:
        """Cluster nearby indices into groups"""
        if len(indices) == 0:
            return []

        clusters = []
        current_cluster = [0]

        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= max_gap:
                current_cluster.append(i)
            else:
                clusters.append(np.array(current_cluster))
                current_cluster = [i]

        clusters.append(np.array(current_cluster))
        return clusters

    def save_results(self, detections: List[AnomalyDetection], output_path: Path):
        """Save detection results to CSV"""
        if not detections:
            logger.warning("No detections to save")
            return

        records = []
        for d in detections:
            record = {
                'method': d.method,
                'anomaly_type': d.anomaly_type,
                'window_idx': d.window_idx if d.window_idx is not None else '',
                'row_idx': d.row_idx if d.row_idx is not None else '',
                'sensor': d.sensor,
                'magnitude': d.magnitude,
                'confidence': d.confidence,
            }
            # Add details as separate columns
            for key, value in d.details.items():
                record[f'detail_{key}'] = value
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(detections)} detections to {output_path}")

        # Print summary
        print("\n" + "="*80)
        print("UNIFIED ANOMALY DETECTION RESULTS")
        print("="*80)

        by_method = df.groupby('method').size()
        print("\nDetections by method:")
        for method, count in by_method.items():
            print(f"  {method}: {count}")

        by_type = df.groupby('anomaly_type').size()
        print("\nDetections by anomaly type:")
        for atype, count in by_type.items():
            print(f"  {atype}: {count}")

        print(f"\nTotal detections: {len(detections)}")
        print("="*80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Unified Anomaly Detection')
    parser.add_argument('--raw-data', required=True, help='Path to raw data CSV')
    parser.add_argument('--diff-data', required=True, help='Path to differenced data CSV')
    parser.add_argument('--golden-weights', help='Path to golden weights CSV (optional)')
    parser.add_argument('--anomaly-weights', help='Path to anomaly weights CSV (optional)')
    parser.add_argument('--output', required=True, help='Output path for results')
    parser.add_argument('--double-diff-threshold', type=float, default=3.0)
    parser.add_argument('--volatility-window', type=int, default=30)
    parser.add_argument('--volatility-spike-threshold', type=float, default=3.0)

    args = parser.parse_args()

    # Load data
    raw_data = pd.read_csv(args.raw_data, index_col=0)
    diff_data = pd.read_csv(args.diff_data, index_col=0)

    golden_weights = None
    anomaly_weights = None
    if args.golden_weights and args.anomaly_weights:
        golden_weights = pd.read_csv(args.golden_weights)
        anomaly_weights = pd.read_csv(args.anomaly_weights)

    # Run detection
    detector = UnifiedAnomalyDetector(
        double_diff_threshold=args.double_diff_threshold,
        volatility_window=args.volatility_window,
        volatility_spike_threshold=args.volatility_spike_threshold
    )

    detections = detector.detect_all(raw_data, diff_data, golden_weights, anomaly_weights)
    detector.save_results(detections, Path(args.output))
