#!/usr/bin/env python3
"""
Early Single-Sensor Epicenter Detector

Key Insight: The FIRST window(s) where changes appear should show changes
ONLY in the epicenter sensor, before propagation to other sensors.

Strategy:
1. Find the earliest windows with ANY edge changes
2. For each early window, identify which sensors have changing edges
3. Epicenter = sensor that changes FIRST and ALONE
4. Ignore later windows where propagation has occurred
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class EarlySingleSensorDetector:
    def __init__(self, golden_weights_path: str, anomaly_weights_path: str,
                 ground_truth_path: str = None, threshold: float = 0.01):
        self.golden_df = pd.read_csv(golden_weights_path)
        self.anomaly_df = pd.read_csv(anomaly_weights_path)
        self.threshold = threshold
        self.ground_truth_sensor = None

        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                gt_data = json.load(f)
                self.ground_truth_sensor = gt_data.get('variable', '').replace('_diff', '_diff')

        # Merge dataframes
        self.merged_df = self.merge_weights()

    def merge_weights(self) -> pd.DataFrame:
        """Merge golden and anomaly weights, compute differences"""
        merged = self.golden_df.merge(
            self.anomaly_df,
            on=['window_idx', 'parent_name', 'child_name', 'lag'],
            suffixes=('_golden', '_anomaly')
        )

        merged['weight_diff'] = abs(merged['weight_anomaly'] - merged['weight_golden'])
        return merged

    def find_earliest_windows(self, top_n: int = 5) -> List[int]:
        """Find the earliest windows with significant changes"""
        # Group by window and get max change per window
        window_changes = self.merged_df.groupby('window_idx')['weight_diff'].max()

        # Filter windows with changes above threshold
        significant_windows = window_changes[window_changes >= self.threshold]

        if len(significant_windows) == 0:
            return []

        # Sort by window index and take earliest N
        earliest = sorted(significant_windows.index)[:top_n]
        return earliest

    def get_affected_sensors_in_window(self, window_idx: int) -> Dict[str, float]:
        """
        Get sensors affected in a specific window.

        A sensor is "affected" if it has ANY edge (parent or child) that changed significantly.

        Returns dict: {sensor_name: total_change_score}
        """
        window_data = self.merged_df[self.merged_df['window_idx'] == window_idx]
        significant = window_data[window_data['weight_diff'] >= self.threshold]

        sensor_changes = defaultdict(float)

        for _, row in significant.iterrows():
            # Count changes where sensor is parent (outgoing edges - epicenter signal)
            sensor_changes[row['parent_name']] += row['weight_diff']

            # Also track child changes (but weighted less)
            # sensor_changes[row['child_name']] += row['weight_diff'] * 0.3

        return dict(sensor_changes)

    def detect_early_single_sensor(self, top_n_windows: int = 5) -> Dict:
        """
        Main detection algorithm: Find sensor that changes ALONE in earliest window(s)

        Strategy:
        1. Get earliest N windows with changes
        2. For each window, identify affected sensors
        3. Find windows with SINGLE affected sensor (pre-propagation)
        4. Epicenter = sensor that appears alone in earliest window(s)
        """
        earliest_windows = self.find_earliest_windows(top_n=top_n_windows)

        if not earliest_windows:
            return {
                'success': False,
                'message': 'No windows with significant changes found',
                'earliest_windows': []
            }

        print(f"\nAnalyzing {len(earliest_windows)} earliest windows: {earliest_windows}")
        print("=" * 80)

        window_analysis = []
        single_sensor_windows = []

        for window_idx in earliest_windows:
            affected_sensors = self.get_affected_sensors_in_window(window_idx)
            n_sensors = len(affected_sensors)

            # Sort sensors by total change
            sorted_sensors = sorted(affected_sensors.items(), key=lambda x: x[1], reverse=True)

            analysis = {
                'window_idx': window_idx,
                'n_sensors_affected': n_sensors,
                'affected_sensors': affected_sensors,
                'sorted_sensors': sorted_sensors,
                'is_single_sensor': n_sensors == 1
            }

            window_analysis.append(analysis)

            # Print analysis
            print(f"\nWindow {window_idx}:")
            print(f"  Sensors affected: {n_sensors}")
            for sensor, change in sorted_sensors:
                is_gt = " ✅ GT" if sensor == self.ground_truth_sensor else ""
                print(f"    {sensor:50s} {change:8.4f}{is_gt}")

            if n_sensors == 1:
                single_sensor_windows.append(analysis)
                print(f"  ⭐ SINGLE SENSOR WINDOW (pre-propagation)")

        # Detect epicenter
        epicenter = None
        detection_method = None

        if single_sensor_windows:
            # Strategy 1: Take sensor from earliest single-sensor window
            earliest_single = single_sensor_windows[0]
            epicenter = earliest_single['sorted_sensors'][0][0]
            detection_method = 'earliest_single_sensor_window'
            print(f"\n{'=' * 80}")
            print(f"DETECTION METHOD: Earliest single-sensor window")
            print(f"Epicenter: {epicenter}")
            print(f"Detected in window: {earliest_single['window_idx']}")
        else:
            # Strategy 2: Take sensor with highest change in earliest window
            earliest_analysis = window_analysis[0]
            epicenter = earliest_analysis['sorted_sensors'][0][0]
            detection_method = 'earliest_window_strongest_signal'
            print(f"\n{'=' * 80}")
            print(f"DETECTION METHOD: Strongest signal in earliest window (no single-sensor window found)")
            print(f"Epicenter: {epicenter}")
            print(f"Detected in window: {earliest_analysis['window_idx']}")

        # Validate
        matches_gt = epicenter == self.ground_truth_sensor if self.ground_truth_sensor else None

        print(f"Ground truth: {self.ground_truth_sensor}")
        print(f"Matches: {'✅ YES' if matches_gt else '❌ NO'}")
        print("=" * 80)

        return {
            'success': True,
            'epicenter': epicenter,
            'detection_method': detection_method,
            'ground_truth': self.ground_truth_sensor,
            'matches_gt': matches_gt,
            'earliest_windows': earliest_windows,
            'window_analysis': window_analysis,
            'single_sensor_windows': single_sensor_windows,
            'n_single_sensor_windows': len(single_sensor_windows)
        }

    def generate_report(self, results: Dict, output_dir: str):
        """Generate detailed report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = output_path / 'early_single_sensor_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json.dump(results, f, indent=2, default=convert)

        # Generate text report
        report_path = output_path / 'early_single_sensor_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EARLY SINGLE-SENSOR EPICENTER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not results['success']:
                f.write(f"FAILED: {results['message']}\n")
                return

            f.write(f"Detection Method: {results['detection_method']}\n")
            f.write(f"Epicenter Detected: {results['epicenter']}\n")
            f.write(f"Ground Truth: {results['ground_truth']}\n")
            f.write(f"Matches: {'✅ YES' if results['matches_gt'] else '❌ NO'}\n\n")

            f.write(f"Windows Analyzed: {len(results['earliest_windows'])}\n")
            f.write(f"Single-Sensor Windows Found: {results['n_single_sensor_windows']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("WINDOW-BY-WINDOW ANALYSIS\n")
            f.write("-" * 80 + "\n\n")

            for analysis in results['window_analysis']:
                f.write(f"Window {analysis['window_idx']}:\n")
                f.write(f"  Sensors affected: {analysis['n_sensors_affected']}\n")

                for sensor, change in analysis['sorted_sensors']:
                    is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                    f.write(f"    {sensor:50s} {change:8.4f}{is_gt}\n")

                if analysis['is_single_sensor']:
                    f.write(f"  ⭐ SINGLE SENSOR WINDOW\n")
                f.write("\n")

            if results['single_sensor_windows']:
                f.write("-" * 80 + "\n")
                f.write("SINGLE-SENSOR WINDOWS (Pre-Propagation)\n")
                f.write("-" * 80 + "\n\n")

                for analysis in results['single_sensor_windows']:
                    sensor, change = analysis['sorted_sensors'][0]
                    is_gt = " ✅ CORRECT" if sensor == results['ground_truth'] else " ❌ WRONG"
                    f.write(f"Window {analysis['window_idx']}: {sensor} (change: {change:.4f}){is_gt}\n")

        print(f"\n✅ Report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Early Single-Sensor Epicenter Detector')
    parser.add_argument('--golden-weights', required=True, help='Path to golden weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Path to anomaly weights CSV')
    parser.add_argument('--ground-truth', help='Path to ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.01, help='Edge change threshold')
    parser.add_argument('--top-n-windows', type=int, default=5, help='Number of earliest windows to analyze')

    args = parser.parse_args()

    print("=" * 80)
    print("EARLY SINGLE-SENSOR EPICENTER DETECTOR")
    print("=" * 80)
    print(f"Golden weights: {args.golden_weights}")
    print(f"Anomaly weights: {args.anomaly_weights}")
    print(f"Ground truth: {args.ground_truth}")
    print(f"Threshold: {args.threshold}")
    print(f"Top N windows: {args.top_n_windows}")

    detector = EarlySingleSensorDetector(
        golden_weights_path=args.golden_weights,
        anomaly_weights_path=args.anomaly_weights,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold
    )

    results = detector.detect_early_single_sensor(top_n_windows=args.top_n_windows)

    detector.generate_report(results, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
