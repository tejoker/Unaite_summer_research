#!/usr/bin/env python3
"""
Directional Asymmetry Epicenter Detector

Key Insight: Epicenter has asymmetric causal pattern:
- Strong OUTGOING edges (causes propagation)
- Weak INCOMING edges (not caused by others)

Propagation targets have opposite pattern:
- Strong INCOMING edges (receives propagation)
- Medium OUTGOING edges (re-propagates)

Strategy:
1. For each sensor in each window, compute:
   - Outgoing change: sum(weight_diff) for edges FROM sensor
   - Incoming change: sum(weight_diff) for edges TO sensor
2. Asymmetry score = (outgoing - incoming) / (outgoing + incoming)
3. Epicenter = sensor with highest positive asymmetry in early windows
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

class DirectionalAsymmetryDetector:
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

    def compute_asymmetry_per_window(self, window_idx: int, sensor: str) -> Dict:
        """
        Compute directional asymmetry for a sensor in a specific window.

        Returns:
            Dict with outgoing, incoming, and asymmetry scores
        """
        window_data = self.merged_df[self.merged_df['window_idx'] == window_idx]
        significant = window_data[window_data['weight_diff'] >= self.threshold]

        # Outgoing edges: sensor is parent
        outgoing = significant[significant['parent_name'] == sensor]
        outgoing_change = outgoing['weight_diff'].sum()
        n_outgoing = len(outgoing)

        # Incoming edges: sensor is child
        incoming = significant[significant['child_name'] == sensor]
        incoming_change = incoming['weight_diff'].sum()
        n_incoming = len(incoming)

        # Compute asymmetry score
        total = outgoing_change + incoming_change
        if total > 0:
            asymmetry = (outgoing_change - incoming_change) / total
        else:
            asymmetry = 0.0

        return {
            'outgoing_change': float(outgoing_change),
            'incoming_change': float(incoming_change),
            'n_outgoing': n_outgoing,
            'n_incoming': n_incoming,
            'asymmetry': float(asymmetry),
            'total_change': float(total)
        }

    def compute_asymmetry_scores(self, top_n_windows: int = 10) -> Dict:
        """
        Compute asymmetry scores for all sensors across earliest windows.

        Args:
            top_n_windows: Number of earliest windows to analyze

        Returns:
            Dict with per-sensor asymmetry analysis
        """
        # Find windows with significant changes
        window_changes = self.merged_df.groupby('window_idx')['weight_diff'].max()
        significant_windows = window_changes[window_changes >= self.threshold]

        if len(significant_windows) == 0:
            return {}

        # Get earliest N windows
        earliest_windows = sorted(significant_windows.index)[:top_n_windows]

        all_sensors = sorted(self.merged_df['parent_name'].unique())

        sensor_analysis = {}

        for sensor in all_sensors:
            window_scores = []
            asymmetries = []
            outgoing_total = 0.0
            incoming_total = 0.0

            for window_idx in earliest_windows:
                scores = self.compute_asymmetry_per_window(window_idx, sensor)

                if scores['total_change'] > 0:
                    window_scores.append({
                        'window': int(window_idx),
                        **scores
                    })
                    asymmetries.append(scores['asymmetry'])
                    outgoing_total += scores['outgoing_change']
                    incoming_total += scores['incoming_change']

            # Aggregate statistics
            if asymmetries:
                avg_asymmetry = np.mean(asymmetries)
                max_asymmetry = np.max(asymmetries)
                std_asymmetry = np.std(asymmetries)

                # Compute cumulative asymmetry
                cumulative_total = outgoing_total + incoming_total
                if cumulative_total > 0:
                    cumulative_asymmetry = (outgoing_total - incoming_total) / cumulative_total
                else:
                    cumulative_asymmetry = 0.0
            else:
                avg_asymmetry = 0.0
                max_asymmetry = 0.0
                std_asymmetry = 0.0
                cumulative_asymmetry = 0.0

            sensor_analysis[sensor] = {
                'window_scores': window_scores,
                'avg_asymmetry': float(avg_asymmetry),
                'max_asymmetry': float(max_asymmetry),
                'std_asymmetry': float(std_asymmetry),
                'cumulative_asymmetry': float(cumulative_asymmetry),
                'outgoing_total': float(outgoing_total),
                'incoming_total': float(incoming_total),
                'n_windows_detected': len(window_scores)
            }

        return sensor_analysis

    def detect_epicenter(self, top_n_windows: int = 10,
                        use_cumulative: bool = True) -> Dict:
        """
        Main detection: Find sensor with highest positive asymmetry.

        Args:
            top_n_windows: Number of earliest windows to analyze
            use_cumulative: Use cumulative asymmetry (True) or average (False)

        Returns:
            Dict with detection results
        """
        sensor_analysis = self.compute_asymmetry_scores(top_n_windows)

        if not sensor_analysis:
            return {
                'success': False,
                'message': 'No sensors with significant changes detected'
            }

        # Filter sensors that were detected
        detected = {s: a for s, a in sensor_analysis.items()
                   if a['n_windows_detected'] > 0}

        if not detected:
            return {
                'success': False,
                'message': 'No sensors detected in any window'
            }

        # Rank by asymmetry score
        if use_cumulative:
            sorted_sensors = sorted(detected.items(),
                                  key=lambda x: x[1]['cumulative_asymmetry'],
                                  reverse=True)
            score_type = 'cumulative_asymmetry'
        else:
            sorted_sensors = sorted(detected.items(),
                                  key=lambda x: x[1]['avg_asymmetry'],
                                  reverse=True)
            score_type = 'avg_asymmetry'

        epicenter = sorted_sensors[0][0]
        epicenter_score = sorted_sensors[0][1][score_type]

        matches_gt = epicenter == self.ground_truth_sensor if self.ground_truth_sensor else None

        # Print results
        print("\n" + "=" * 80)
        print("DIRECTIONAL ASYMMETRY DETECTION RESULTS")
        print("=" * 80)
        print(f"\nAnalyzed {top_n_windows} earliest windows")
        print(f"Score type: {score_type}")
        print(f"\nDetected Epicenter: {epicenter}")
        print(f"Asymmetry Score: {epicenter_score:.4f}")
        print(f"Ground Truth: {self.ground_truth_sensor}")
        print(f"Matches: {'✅ YES' if matches_gt else '❌ NO'}")

        print("\n" + "-" * 80)
        print("ASYMMETRY RANKING (All Sensors)")
        print("-" * 80)
        print(f"{'Rank':<6} {'Sensor':<50} {'Asymmetry':<12} {'Out/In':<20} {'GT':>5}")
        print("-" * 80)

        for rank, (sensor, analysis) in enumerate(sorted_sensors, 1):
            asymmetry = analysis[score_type]
            outgoing = analysis['outgoing_total']
            incoming = analysis['incoming_total']
            is_gt = "✅" if sensor == self.ground_truth_sensor else ""

            out_in_str = f"{outgoing:.2f} / {incoming:.2f}"

            # Indicate direction
            if asymmetry > 0.2:
                direction = ">>>>"  # Strong outgoing bias (epicenter pattern)
            elif asymmetry > 0:
                direction = ">>>"
            elif asymmetry < -0.2:
                direction = "<<<<"  # Strong incoming bias (propagation target)
            elif asymmetry < 0:
                direction = "<<<"
            else:
                direction = "===="

            print(f"{rank:<6} {sensor:<50} {asymmetry:>+8.4f} {direction} "
                  f"{out_in_str:<20} {is_gt:>5}")

        print("=" * 80)
        print("\nInterpretation:")
        print("  Positive asymmetry (+) = More outgoing than incoming (EPICENTER pattern)")
        print("  Negative asymmetry (-) = More incoming than outgoing (PROPAGATION target)")
        print("=" * 80)

        return {
            'success': True,
            'epicenter': epicenter,
            'epicenter_score': float(epicenter_score),
            'score_type': score_type,
            'ground_truth': self.ground_truth_sensor,
            'matches_gt': matches_gt,
            'sensor_analysis': sensor_analysis,
            'ranking': [(s, a[score_type]) for s, a in sorted_sensors],
            'top_n_windows': top_n_windows
        }

    def generate_report(self, results: Dict, output_dir: str):
        """Generate detailed report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / 'directional_asymmetry_results.json'
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            json.dump(results, f, indent=2, default=convert)

        # Generate text report
        report_path = output_path / 'directional_asymmetry_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DIRECTIONAL ASYMMETRY EPICENTER DETECTION REPORT\n")
            f.write("=" * 80 + "\n\n")

            if not results['success']:
                f.write(f"FAILED: {results['message']}\n")
                return

            f.write(f"Epicenter Detected: {results['epicenter']}\n")
            f.write(f"Asymmetry Score: {results['epicenter_score']:.4f}\n")
            f.write(f"Score Type: {results['score_type']}\n")
            f.write(f"Ground Truth: {results['ground_truth']}\n")
            f.write(f"Matches: {'✅ YES' if results['matches_gt'] else '❌ NO'}\n\n")

            f.write("-" * 80 + "\n")
            f.write("SENSOR RANKING BY ASYMMETRY\n")
            f.write("-" * 80 + "\n\n")

            for rank, (sensor, score) in enumerate(results['ranking'], 1):
                is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                is_epicenter = " 🎯 EPICENTER" if sensor == results['epicenter'] else ""
                f.write(f"{rank}. {sensor:50s} {score:>+8.4f}{is_gt}{is_epicenter}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("DETAILED SENSOR ANALYSIS\n")
            f.write("-" * 80 + "\n\n")

            for sensor, analysis in results['sensor_analysis'].items():
                is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                f.write(f"{sensor}{is_gt}\n")
                f.write(f"  Cumulative Asymmetry: {analysis['cumulative_asymmetry']:>+8.4f}\n")
                f.write(f"  Average Asymmetry: {analysis['avg_asymmetry']:>+8.4f}\n")
                f.write(f"  Max Asymmetry: {analysis['max_asymmetry']:>+8.4f}\n")
                f.write(f"  Std Asymmetry: {analysis['std_asymmetry']:>8.4f}\n")
                f.write(f"  Total Outgoing: {analysis['outgoing_total']:>8.2f}\n")
                f.write(f"  Total Incoming: {analysis['incoming_total']:>8.2f}\n")
                f.write(f"  Windows Detected: {analysis['n_windows_detected']}\n")

                if analysis['window_scores']:
                    f.write(f"  Per-window scores (first 10):\n")
                    for ws in analysis['window_scores'][:10]:
                        f.write(f"    Window {ws['window']:3d}: asymmetry={ws['asymmetry']:>+7.4f} "
                               f"(out={ws['outgoing_change']:.2f}, in={ws['incoming_change']:.2f})\n")
                    if len(analysis['window_scores']) > 10:
                        f.write(f"    ... and {len(analysis['window_scores']) - 10} more windows\n")
                f.write("\n")

        print(f"\n✅ Report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Directional Asymmetry Epicenter Detector')
    parser.add_argument('--golden-weights', required=True, help='Path to golden weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Path to anomaly weights CSV')
    parser.add_argument('--ground-truth', help='Path to ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.01, help='Edge change threshold')
    parser.add_argument('--top-n-windows', type=int, default=10, help='Number of earliest windows to analyze')
    parser.add_argument('--use-average', action='store_true', help='Use average asymmetry instead of cumulative')

    args = parser.parse_args()

    print("=" * 80)
    print("DIRECTIONAL ASYMMETRY EPICENTER DETECTOR")
    print("=" * 80)
    print(f"Strategy: Find sensor with highest OUT vs IN edge asymmetry")
    print(f"Threshold: {args.threshold}")
    print(f"Top N windows: {args.top_n_windows}")
    print()

    detector = DirectionalAsymmetryDetector(
        golden_weights_path=args.golden_weights,
        anomaly_weights_path=args.anomaly_weights,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold
    )

    results = detector.detect_epicenter(
        top_n_windows=args.top_n_windows,
        use_cumulative=not args.use_average
    )

    detector.generate_report(results, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
