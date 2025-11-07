#!/usr/bin/env python3
"""
Sub-Window Analysis Epicenter Detector

Key Insight: Standard rolling windows (size=100) are too large.
By the time we detect anomalies, propagation has already occurred.

Solution: Analyze at FINER granularity using sub-windows (size=10-20 samples)
to find single-sensor divergence BEFORE propagation spreads to all sensors.

Strategy:
1. Load golden and anomaly time series (not causal graphs)
2. Slide small windows (size=10-20) across both time series
3. For each window, compute statistical divergence per sensor
4. Find FIRST sub-window where SINGLE sensor diverges significantly
5. That sensor = epicenter (detected before propagation)
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

class SubWindowDetector:
    def __init__(self, golden_ts_path: str, anomaly_ts_path: str,
                 ground_truth_path: str = None, window_size: int = 15):
        """
        Args:
            golden_ts_path: Path to golden (baseline) time series CSV
            anomaly_ts_path: Path to anomaly time series CSV
            ground_truth_path: Path to ground truth JSON
            window_size: Sub-window size (default: 15 samples)
        """
        self.golden_df = pd.read_csv(golden_ts_path)
        self.anomaly_df = pd.read_csv(anomaly_ts_path)
        self.window_size = window_size
        self.ground_truth_sensor = None

        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                gt_data = json.load(f)
                self.ground_truth_sensor = gt_data.get('ts_col', gt_data.get('variable', ''))
                # Add _diff suffix if not already present
                if not self.ground_truth_sensor.endswith('_diff'):
                    self.ground_truth_sensor += '_diff'

        # Get sensor columns (exclude timestamp columns)
        self.sensor_cols = [col for col in self.golden_df.columns
                           if col not in ['Unnamed: 0', 'timestamp', 'index']]

    def compute_window_divergence(self, start_idx: int) -> Dict[str, float]:
        """
        Compute statistical divergence for each sensor in a sub-window.

        Uses multiple statistical tests:
        - Mean difference (normalized)
        - Standard deviation difference
        - KS test statistic
        - Combined score

        Returns:
            Dict mapping sensor_name to divergence_score
        """
        end_idx = start_idx + self.window_size

        if end_idx > len(self.golden_df):
            return {}

        divergences = {}

        for sensor in self.sensor_cols:
            golden_window = self.golden_df[sensor].iloc[start_idx:end_idx].values
            anomaly_window = self.anomaly_df[sensor].iloc[start_idx:end_idx].values

            # Remove NaN values
            golden_window = golden_window[~np.isnan(golden_window)]
            anomaly_window = anomaly_window[~np.isnan(anomaly_window)]

            if len(golden_window) < 3 or len(anomaly_window) < 3:
                divergences[sensor] = 0.0
                continue

            # 1. Mean difference (normalized by std)
            mean_diff = abs(np.mean(anomaly_window) - np.mean(golden_window))
            std_pooled = np.sqrt((np.std(golden_window)**2 + np.std(anomaly_window)**2) / 2)
            if std_pooled > 0:
                normalized_mean_diff = mean_diff / std_pooled
            else:
                normalized_mean_diff = 0.0

            # 2. Std difference
            std_diff = abs(np.std(anomaly_window) - np.std(golden_window))

            # 3. Kolmogorov-Smirnov test statistic
            try:
                ks_stat, _ = stats.ks_2samp(golden_window, anomaly_window)
            except:
                ks_stat = 0.0

            # Combined divergence score
            divergence = normalized_mean_diff + std_diff + ks_stat * 10

            divergences[sensor] = float(divergence)

        return divergences

    def find_first_single_sensor_divergence(self, threshold: float = 1.0,
                                           stride: int = 1) -> Dict:
        """
        Slide sub-windows across time series to find first single-sensor divergence.

        Args:
            threshold: Divergence threshold for detection
            stride: Step size between windows (default: 1 for full overlap)

        Returns:
            Dict with detection results
        """
        n_samples = len(self.golden_df)
        detection_timeline = []

        print(f"\nScanning {n_samples} samples with sub-windows (size={self.window_size}, stride={stride})...")

        for start_idx in range(0, n_samples - self.window_size, stride):
            divergences = self.compute_window_divergence(start_idx)

            if not divergences:
                continue

            # Find sensors above threshold
            significant_sensors = {
                sensor: div for sensor, div in divergences.items()
                if div >= threshold
            }

            if significant_sensors:
                n_sensors = len(significant_sensors)
                sorted_sensors = sorted(significant_sensors.items(),
                                      key=lambda x: x[1], reverse=True)

                detection_timeline.append({
                    'start_idx': start_idx,
                    'end_idx': start_idx + self.window_size,
                    'n_sensors': n_sensors,
                    'sensors': significant_sensors,
                    'sorted_sensors': sorted_sensors,
                    'is_single_sensor': n_sensors == 1
                })

                # Print first 20 detections
                if len(detection_timeline) <= 20:
                    print(f"  Sample {start_idx:4d}-{start_idx+self.window_size:4d}: "
                          f"{n_sensors} sensor(s) - ", end='')
                    if n_sensors == 1:
                        sensor = sorted_sensors[0][0]
                        is_gt = "✅ GT" if sensor == self.ground_truth_sensor else ""
                        print(f"🎯 SINGLE SENSOR: {sensor} {is_gt}")
                    else:
                        print(f"Multiple sensors")

        print(f"Total sub-windows detected: {len(detection_timeline)}")

        return self.analyze_detection_timeline(detection_timeline)

    def analyze_detection_timeline(self, timeline: List[Dict]) -> Dict:
        """
        Analyze detection timeline to identify epicenter.

        Strategy:
        1. Find first single-sensor window (pre-propagation)
        2. If no single-sensor window, use first window with any detection
        3. Rank by earliest + highest divergence
        """
        if not timeline:
            return {
                'success': False,
                'message': 'No sub-windows with significant divergence detected',
                'timeline': []
            }

        # Find single-sensor windows
        single_sensor_windows = [w for w in timeline if w['is_single_sensor']]

        epicenter = None
        detection_method = None
        first_detection_idx = None

        if single_sensor_windows:
            # Strategy 1: Use first single-sensor window
            first_single = single_sensor_windows[0]
            epicenter = first_single['sorted_sensors'][0][0]
            detection_method = 'first_single_sensor_window'
            first_detection_idx = first_single['start_idx']

            print(f"\n{'='*80}")
            print(f"🎯 SINGLE-SENSOR DETECTION SUCCESS")
            print(f"{'='*80}")
            print(f"Found {len(single_sensor_windows)} single-sensor window(s)")
            print(f"First single-sensor window: samples {first_single['start_idx']}-{first_single['end_idx']}")
            print(f"Epicenter: {epicenter}")
        else:
            # Strategy 2: Use sensor with highest divergence in first window
            first_window = timeline[0]
            epicenter = first_window['sorted_sensors'][0][0]
            detection_method = 'first_window_highest_divergence'
            first_detection_idx = first_window['start_idx']

            print(f"\n{'='*80}")
            print(f"⚠️  NO SINGLE-SENSOR WINDOW FOUND")
            print(f"{'='*80}")
            print(f"Using highest divergence in first detection window")
            print(f"First window: samples {first_window['start_idx']}-{first_window['end_idx']}")
            print(f"Sensors detected: {first_window['n_sensors']}")
            print(f"Epicenter: {epicenter}")

        matches_gt = epicenter == self.ground_truth_sensor if self.ground_truth_sensor else None

        print(f"Ground truth: {self.ground_truth_sensor}")
        print(f"Matches: {'✅ YES' if matches_gt else '❌ NO'}")
        print(f"{'='*80}\n")

        return {
            'success': True,
            'epicenter': epicenter,
            'detection_method': detection_method,
            'first_detection_idx': first_detection_idx,
            'ground_truth': self.ground_truth_sensor,
            'matches_gt': matches_gt,
            'timeline': timeline,
            'n_single_sensor_windows': len(single_sensor_windows),
            'n_total_detections': len(timeline)
        }

    def generate_report(self, results: Dict, output_dir: str):
        """Generate detailed report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / 'sub_window_results.json'
        with open(json_path, 'w') as f:
            # Convert for JSON serialization
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            json.dump(results, f, indent=2, default=convert)

        # Generate text report
        report_path = output_path / 'sub_window_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUB-WINDOW ANALYSIS EPICENTER DETECTION REPORT\n")
            f.write("="*80 + "\n\n")

            if not results['success']:
                f.write(f"FAILED: {results['message']}\n")
                return

            f.write(f"Window Size: {self.window_size} samples\n")
            f.write(f"Epicenter Detected: {results['epicenter']}\n")
            f.write(f"Detection Method: {results['detection_method']}\n")
            f.write(f"First Detection: Sample {results['first_detection_idx']}\n")
            f.write(f"Ground Truth: {results['ground_truth']}\n")
            f.write(f"Matches: {'✅ YES' if results['matches_gt'] else '❌ NO'}\n\n")

            f.write(f"Total sub-windows detected: {results['n_total_detections']}\n")
            f.write(f"Single-sensor windows: {results['n_single_sensor_windows']}\n\n")

            f.write("-"*80 + "\n")
            f.write("DETECTION TIMELINE (First 50 windows)\n")
            f.write("-"*80 + "\n\n")

            for i, window in enumerate(results['timeline'][:50]):
                f.write(f"Window {i+1}: Samples {window['start_idx']}-{window['end_idx']}\n")
                f.write(f"  Sensors: {window['n_sensors']}\n")

                for sensor, div in window['sorted_sensors'][:5]:
                    is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                    f.write(f"    {sensor:50s} {div:8.4f}{is_gt}\n")

                if window['is_single_sensor']:
                    f.write(f"  🎯 SINGLE SENSOR WINDOW\n")
                f.write("\n")

            if len(results['timeline']) > 50:
                f.write(f"... and {len(results['timeline']) - 50} more windows\n")

        print(f"✅ Report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Sub-Window Analysis Epicenter Detector')
    parser.add_argument('--golden-ts', required=True, help='Path to golden time series CSV')
    parser.add_argument('--anomaly-ts', required=True, help='Path to anomaly time series CSV')
    parser.add_argument('--ground-truth', help='Path to ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--window-size', type=int, default=15, help='Sub-window size (default: 15)')
    parser.add_argument('--threshold', type=float, default=1.0, help='Divergence threshold')
    parser.add_argument('--stride', type=int, default=5, help='Window stride (default: 5)')

    args = parser.parse_args()

    print("="*80)
    print("SUB-WINDOW ANALYSIS EPICENTER DETECTOR")
    print("="*80)
    print(f"Strategy: Analyze at FINE granularity to catch pre-propagation state")
    print(f"Window size: {args.window_size} samples (vs 100 in standard analysis)")
    print(f"Threshold: {args.threshold}")
    print(f"Stride: {args.stride}")
    print()

    detector = SubWindowDetector(
        golden_ts_path=args.golden_ts,
        anomaly_ts_path=args.anomaly_ts,
        ground_truth_path=args.ground_truth,
        window_size=args.window_size
    )

    results = detector.find_first_single_sensor_divergence(
        threshold=args.threshold,
        stride=args.stride
    )

    if results['success']:
        detector.generate_report(results, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
