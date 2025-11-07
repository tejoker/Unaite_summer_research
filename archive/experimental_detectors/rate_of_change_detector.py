#!/usr/bin/env python3
"""
Rate of Change Epicenter Detector

Key Insight: Epicenter shows SUDDEN SPIKE in edge changes,
while propagation targets show GRADUAL INCREASE.

Strategy:
1. Track sensor edge changes over time (window-by-window)
2. Compute rate of change (first derivative)
3. Epicenter = sensor with highest rate of change (steepest spike)
4. Focus on earliest windows where spike occurs
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class RateOfChangeDetector:
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

    def get_sensor_change_timeline(self, sensor: str) -> List[Tuple[int, float]]:
        """
        Get timeline of edge changes for a sensor across all windows.

        Returns list of (window_idx, total_change) tuples.
        """
        all_windows = sorted(self.merged_df['window_idx'].unique())
        timeline = []

        for window_idx in all_windows:
            window_data = self.merged_df[
                (self.merged_df['window_idx'] == window_idx) &
                (self.merged_df['parent_name'] == sensor)  # Outgoing edges only
            ]
            total_change = window_data['weight_diff'].sum()
            timeline.append((int(window_idx), float(total_change)))

        return timeline

    def compute_rate_of_change(self, timeline: List[Tuple[int, float]]) -> Dict:
        """
        Compute rate of change (first derivative) for a sensor's timeline.

        Returns:
            Dict with rate analysis
        """
        if len(timeline) < 2:
            return {
                'rate_timeline': [],
                'max_rate': 0.0,
                'max_rate_window': None,
                'avg_rate': 0.0,
                'spike_score': 0.0
            }

        windows, changes = zip(*timeline)
        changes = np.array(changes)

        # Compute first derivative (rate of change)
        rates = np.diff(changes)

        # Find maximum rate and its window
        if len(rates) > 0:
            max_rate_idx = np.argmax(rates)
            max_rate = float(rates[max_rate_idx])
            max_rate_window = int(windows[max_rate_idx + 1])  # Window after the jump
            avg_rate = float(np.mean(np.abs(rates)))

            # Spike score: how much does max rate exceed average?
            if avg_rate > 0:
                spike_score = max_rate / (avg_rate + 1e-10)
            else:
                spike_score = 0.0
        else:
            max_rate = 0.0
            max_rate_window = None
            avg_rate = 0.0
            spike_score = 0.0

        # Build rate timeline
        rate_timeline = []
        for i, rate in enumerate(rates):
            rate_timeline.append({
                'from_window': int(windows[i]),
                'to_window': int(windows[i + 1]),
                'rate': float(rate)
            })

        return {
            'rate_timeline': rate_timeline,
            'max_rate': float(max_rate),
            'max_rate_window': max_rate_window,
            'avg_rate': float(avg_rate),
            'spike_score': float(spike_score)
        }

    def detect_epicenter(self, use_spike_score: bool = True) -> Dict:
        """
        Main detection: Find sensor with highest rate of change.

        Args:
            use_spike_score: Use spike score (max/avg) instead of max rate

        Returns:
            Dict with detection results
        """
        all_sensors = sorted(self.merged_df['parent_name'].unique())

        sensor_analysis = {}

        for sensor in all_sensors:
            timeline = self.get_sensor_change_timeline(sensor)
            rate_analysis = self.compute_rate_of_change(timeline)

            sensor_analysis[sensor] = {
                'timeline': timeline,
                **rate_analysis
            }

        # Filter sensors with detected changes
        detected = {s: a for s, a in sensor_analysis.items()
                   if a['max_rate'] > 0}

        if not detected:
            return {
                'success': False,
                'message': 'No sensors with rate of change detected'
            }

        # Rank by spike score or max rate
        if use_spike_score:
            sorted_sensors = sorted(detected.items(),
                                  key=lambda x: x[1]['spike_score'],
                                  reverse=True)
            score_type = 'spike_score'
        else:
            sorted_sensors = sorted(detected.items(),
                                  key=lambda x: x[1]['max_rate'],
                                  reverse=True)
            score_type = 'max_rate'

        epicenter = sorted_sensors[0][0]
        epicenter_score = sorted_sensors[0][1][score_type]
        epicenter_window = sorted_sensors[0][1]['max_rate_window']

        matches_gt = epicenter == self.ground_truth_sensor if self.ground_truth_sensor else None

        # Print results
        print("\n" + "=" * 80)
        print("RATE OF CHANGE DETECTION RESULTS")
        print("=" * 80)
        print(f"\nScore type: {score_type}")
        print(f"\nDetected Epicenter: {epicenter}")
        print(f"Score: {epicenter_score:.4f}")
        print(f"Max rate window: {epicenter_window}")
        print(f"Ground Truth: {self.ground_truth_sensor}")
        print(f"Matches: {'✅ YES' if matches_gt else '❌ NO'}")

        print("\n" + "-" * 80)
        print("RATE OF CHANGE RANKING (All Sensors)")
        print("-" * 80)
        print(f"{'Rank':<6} {'Sensor':<50} {'Score':<12} {'Max Rate':<12} {'Window':<8} {'GT':>5}")
        print("-" * 80)

        for rank, (sensor, analysis) in enumerate(sorted_sensors, 1):
            score = analysis[score_type]
            max_rate = analysis['max_rate']
            window = analysis['max_rate_window']
            is_gt = "✅" if sensor == self.ground_truth_sensor else ""

            # Classify spike type
            if analysis['spike_score'] > 5.0:
                spike_type = "🔥 SPIKE"  # Very sharp spike
            elif analysis['spike_score'] > 2.0:
                spike_type = "⬆ JUMP"   # Moderate jump
            else:
                spike_type = "~ GRADUAL"  # Gradual increase

            print(f"{rank:<6} {sensor:<50} {score:>8.4f} {spike_type} "
                  f"{max_rate:>8.4f}  W{window:<6} {is_gt:>5}")

        print("=" * 80)
        print("\nInterpretation:")
        print("  Spike score = max_rate / avg_rate")
        print("  High spike score = Sudden onset (EPICENTER pattern)")
        print("  Low spike score = Gradual increase (PROPAGATION pattern)")
        print("=" * 80)

        return {
            'success': True,
            'epicenter': epicenter,
            'epicenter_score': float(epicenter_score),
            'epicenter_window': epicenter_window,
            'score_type': score_type,
            'ground_truth': self.ground_truth_sensor,
            'matches_gt': matches_gt,
            'sensor_analysis': sensor_analysis,
            'ranking': [(s, a[score_type], a['max_rate_window']) for s, a in sorted_sensors]
        }

    def generate_report(self, results: Dict, output_dir: str):
        """Generate detailed report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / 'rate_of_change_results.json'
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            json.dump(results, f, indent=2, default=convert)

        # Generate text report
        report_path = output_path / 'rate_of_change_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RATE OF CHANGE EPICENTER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not results['success']:
                f.write(f"FAILED: {results['message']}\n")
                return

            f.write(f"Epicenter Detected: {results['epicenter']}\n")
            f.write(f"Score: {results['epicenter_score']:.4f}\n")
            f.write(f"Score Type: {results['score_type']}\n")
            f.write(f"Max Rate Window: {results['epicenter_window']}\n")
            f.write(f"Ground Truth: {results['ground_truth']}\n")
            f.write(f"Matches: {'✅ YES' if results['matches_gt'] else '❌ NO'}\n\n")

            f.write("-" * 80 + "\n")
            f.write("SENSOR RANKING BY RATE OF CHANGE\n")
            f.write("-" * 80 + "\n\n")

            for rank, (sensor, score, window) in enumerate(results['ranking'], 1):
                is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                is_epicenter = " 🎯 EPICENTER" if sensor == results['epicenter'] else ""
                f.write(f"{rank}. {sensor:50s} Score: {score:>8.4f} (Window {window}){is_gt}{is_epicenter}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED SENSOR ANALYSIS\n")
            f.write("-" * 80 + "\n\n")

            for sensor, analysis in results['sensor_analysis'].items():
                is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                f.write(f"{sensor}{is_gt}\n")
                f.write(f"  Max Rate: {analysis['max_rate']:>8.4f}\n")
                f.write(f"  Max Rate Window: {analysis['max_rate_window']}\n")
                f.write(f"  Avg Rate: {analysis['avg_rate']:>8.4f}\n")
                f.write(f"  Spike Score: {analysis['spike_score']:>8.4f}\n")

                if analysis['rate_timeline']:
                    f.write(f"  Rate timeline (first 15 changes):\n")
                    for rt in analysis['rate_timeline'][:15]:
                        f.write(f"    Window {rt['from_window']:3d} → {rt['to_window']:3d}: "
                               f"rate = {rt['rate']:>8.4f}\n")
                    if len(analysis['rate_timeline']) > 15:
                        f.write(f"    ... and {len(analysis['rate_timeline']) - 15} more\n")
                f.write("\n")

        print(f"\n✅ Report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Rate of Change Epicenter Detector')
    parser.add_argument('--golden-weights', required=True, help='Path to golden weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Path to anomaly weights CSV')
    parser.add_argument('--ground-truth', help='Path to ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.01, help='Edge change threshold')
    parser.add_argument('--use-max-rate', action='store_true', help='Use max rate instead of spike score')

    args = parser.parse_args()

    print("=" * 80)
    print("RATE OF CHANGE EPICENTER DETECTOR")
    print("=" * 80)
    print(f"Strategy: Find sensor with steepest change spike (d/dt)")
    print(f"Threshold: {args.threshold}")
    print()

    detector = RateOfChangeDetector(
        golden_weights_path=args.golden_weights,
        anomaly_weights_path=args.anomaly_weights,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold
    )

    results = detector.detect_epicenter(use_spike_score=not args.use_max_rate)

    detector.generate_report(results, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
