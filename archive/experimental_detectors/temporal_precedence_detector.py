#!/usr/bin/env python3
"""
Temporal Precedence Epicenter Detector

Key Insight: The sensor that shows significant changes FIRST is likely the epicenter,
regardless of magnitude. Propagation targets change AFTER the epicenter.

Strategy:
1. For each sensor, find the FIRST window where it shows significant edge changes
2. Epicenter = sensor with earliest first-detection window
3. If multiple sensors tie, use magnitude as tiebreaker
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class TemporalPrecedenceDetector:
    def __init__(self, golden_weights_path: str, anomaly_weights_path: str,
                 ground_truth_path: str = None, threshold: float = 0.01):
        self.golden_df = pd.read_csv(golden_weights_path)
        self.anomaly_df = pd.read_csv(anomaly_weights_path)
        self.threshold = threshold
        self.ground_truth_sensor = None

        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                gt_data = json.load(f)
                self.ground_truth_sensor = gt_data.get('ts_col', gt_data.get('variable', '')) + '_diff'

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

    def get_sensor_change_per_window(self, window_idx: int, sensor: str) -> float:
        """
        Get total edge change for a sensor in a specific window.
        Only count OUTGOING edges (where sensor is parent) for epicenter detection.
        """
        window_data = self.merged_df[
            (self.merged_df['window_idx'] == window_idx) &
            (self.merged_df['parent_name'] == sensor)
        ]

        significant = window_data[window_data['weight_diff'] >= self.threshold]
        total_change = significant['weight_diff'].sum()

        return total_change

    def find_first_detection_per_sensor(self) -> Dict[str, Dict]:
        """
        For each sensor, find the FIRST window where it shows significant changes.

        Returns:
            Dict mapping sensor_name to:
                - first_window: First window index with significant change
                - first_change: Magnitude of change in that window
                - detection_sequence: List of (window, change) for all significant windows
        """
        all_windows = sorted(self.merged_df['window_idx'].unique())
        all_sensors = sorted(self.merged_df['parent_name'].unique())

        first_detection = {}

        for sensor in all_sensors:
            detection_sequence = []
            first_window = None
            first_change = 0.0

            for window_idx in all_windows:
                change = self.get_sensor_change_per_window(window_idx, sensor)

                if change > 0:
                    detection_sequence.append((int(window_idx), float(change)))

                    if first_window is None and change >= self.threshold:
                        first_window = int(window_idx)
                        first_change = float(change)

            first_detection[sensor] = {
                'first_window': first_window,
                'first_change': first_change,
                'detection_sequence': detection_sequence,
                'total_windows_detected': len(detection_sequence)
            }

        return first_detection

    def detect_epicenter(self) -> Dict:
        """
        Main detection: Find sensor with earliest first detection.

        Returns dict with detection results and analysis.
        """
        first_detection = self.find_first_detection_per_sensor()

        # Filter sensors that were detected at least once
        detected_sensors = {
            sensor: info for sensor, info in first_detection.items()
            if info['first_window'] is not None
        }

        if not detected_sensors:
            return {
                'success': False,
                'message': 'No sensors with significant changes detected',
                'first_detection': first_detection
            }

        # Find earliest window across all sensors
        earliest_window = min(info['first_window'] for info in detected_sensors.values())

        # Find all sensors detected in earliest window (handle ties)
        earliest_sensors = {
            sensor: info for sensor, info in detected_sensors.items()
            if info['first_window'] == earliest_window
        }

        # If tie, use magnitude as tiebreaker
        if len(earliest_sensors) > 1:
            epicenter = max(earliest_sensors.keys(),
                          key=lambda s: earliest_sensors[s]['first_change'])
            detection_method = 'earliest_window_with_magnitude_tiebreaker'
        else:
            epicenter = list(earliest_sensors.keys())[0]
            detection_method = 'earliest_window_unique'

        matches_gt = epicenter == self.ground_truth_sensor if self.ground_truth_sensor else None

        # Print results
        print("\n" + "=" * 80)
        print("TEMPORAL PRECEDENCE DETECTION RESULTS")
        print("=" * 80)
        print(f"\nEarliest window with significant changes: {earliest_window}")
        print(f"Number of sensors in earliest window: {len(earliest_sensors)}")
        print(f"\nDetected Epicenter: {epicenter}")
        print(f"Detection Method: {detection_method}")
        print(f"Ground Truth: {self.ground_truth_sensor}")
        print(f"Matches: {'✅ YES' if matches_gt else '❌ NO'}")

        print("\n" + "-" * 80)
        print("FIRST DETECTION TIMELINE (All Sensors)")
        print("-" * 80)

        # Sort sensors by first detection window
        sorted_sensors = sorted(detected_sensors.items(),
                              key=lambda x: (x[1]['first_window'], -x[1]['first_change']))

        for sensor, info in sorted_sensors:
            is_gt = " ✅ GT" if sensor == self.ground_truth_sensor else ""
            is_epicenter = " 🎯 EPICENTER" if sensor == epicenter else ""
            print(f"Window {info['first_window']:3d}: {sensor:45s} "
                  f"(change: {info['first_change']:8.4f}){is_gt}{is_epicenter}")

        # Show sensors not detected
        not_detected = {s: info for s, info in first_detection.items()
                       if info['first_window'] is None}
        if not_detected:
            print("\n" + "-" * 80)
            print("SENSORS NEVER DETECTED (No significant changes)")
            print("-" * 80)
            for sensor in not_detected:
                is_gt = " ✅ GT" if sensor == self.ground_truth_sensor else ""
                print(f"  {sensor}{is_gt}")

        print("=" * 80)

        return {
            'success': True,
            'epicenter': epicenter,
            'detection_method': detection_method,
            'earliest_window': earliest_window,
            'ground_truth': self.ground_truth_sensor,
            'matches_gt': matches_gt,
            'first_detection': first_detection,
            'earliest_sensors': list(earliest_sensors.keys()),
            'n_tied_sensors': len(earliest_sensors)
        }

    def generate_report(self, results: Dict, output_dir: str):
        """Generate detailed report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / 'temporal_precedence_results.json'
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            json.dump(results, f, indent=2, default=convert)

        # Generate text report
        report_path = output_path / 'temporal_precedence_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TEMPORAL PRECEDENCE EPICENTER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not results['success']:
                f.write(f"FAILED: {results['message']}\n")
                return

            f.write(f"Epicenter Detected: {results['epicenter']}\n")
            f.write(f"Detection Method: {results['detection_method']}\n")
            f.write(f"Earliest Window: {results['earliest_window']}\n")
            f.write(f"Ground Truth: {results['ground_truth']}\n")
            f.write(f"Matches: {'✅ YES' if results['matches_gt'] else '❌ NO'}\n\n")

            if results['n_tied_sensors'] > 1:
                f.write(f"Note: {results['n_tied_sensors']} sensors tied at window {results['earliest_window']}\n")
                f.write(f"Used magnitude tiebreaker\n\n")

            f.write("-" * 80 + "\n")
            f.write("FIRST DETECTION TIMELINE\n")
            f.write("-" * 80 + "\n\n")

            # Sort by first window
            detected = {s: info for s, info in results['first_detection'].items()
                       if info['first_window'] is not None}
            sorted_sensors = sorted(detected.items(),
                                  key=lambda x: (x[1]['first_window'], -x[1]['first_change']))

            for sensor, info in sorted_sensors:
                is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                is_epicenter = " 🎯 EPICENTER" if sensor == results['epicenter'] else ""
                f.write(f"Window {info['first_window']:3d}: {sensor:45s} "
                       f"(change: {info['first_change']:8.4f}){is_gt}{is_epicenter}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("DETECTION SEQUENCE DETAILS\n")
            f.write("-" * 80 + "\n\n")

            for sensor, info in sorted_sensors:
                f.write(f"{sensor}:\n")
                f.write(f"  First window: {info['first_window']}\n")
                f.write(f"  First change: {info['first_change']:.4f}\n")
                f.write(f"  Total windows detected: {info['total_windows_detected']}\n")

                if len(info['detection_sequence']) > 0:
                    f.write(f"  Detection sequence (window: change):\n")
                    for window, change in info['detection_sequence'][:10]:  # First 10
                        f.write(f"    {window}: {change:.4f}\n")
                    if len(info['detection_sequence']) > 10:
                        f.write(f"    ... and {len(info['detection_sequence']) - 10} more\n")
                f.write("\n")

        print(f"\n✅ Report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Temporal Precedence Epicenter Detector')
    parser.add_argument('--golden-weights', required=True, help='Path to golden weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Path to anomaly weights CSV')
    parser.add_argument('--ground-truth', help='Path to ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.01, help='Edge change threshold')

    args = parser.parse_args()

    print("=" * 80)
    print("TEMPORAL PRECEDENCE EPICENTER DETECTOR")
    print("=" * 80)
    print(f"Strategy: Find sensor that changes FIRST (earliest window)")
    print(f"Threshold: {args.threshold}")
    print()

    detector = TemporalPrecedenceDetector(
        golden_weights_path=args.golden_weights,
        anomaly_weights_path=args.anomaly_weights,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold
    )

    results = detector.detect_epicenter()
    detector.generate_report(results, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
