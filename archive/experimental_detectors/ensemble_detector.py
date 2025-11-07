#!/usr/bin/env python3
"""
Ensemble Meta-Detector for Epicenter Detection

Combines multiple detection methods with weighted voting:
1. Temporal Precedence (weight: 3.0) - Best overall
2. Directional Asymmetry (weight: 1.5)
3. Rate of Change (weight: 1.5)
4. Multi-Window Voting (weight: 1.0) - Baseline
5. Sub-Window Analysis (weight: 2.5) - For spike detection

Strategy:
- Run all detectors independently
- Aggregate votes with method-specific weights
- Winner = sensor with highest weighted vote count
- Report confidence based on agreement across methods
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import subprocess
import sys

class EnsembleDetector:
    def __init__(self, golden_weights_path: str, anomaly_weights_path: str,
                 golden_ts_path: str = None, anomaly_ts_path: str = None,
                 ground_truth_path: str = None, threshold: float = 0.01):
        self.golden_weights_path = golden_weights_path
        self.anomaly_weights_path = anomaly_weights_path
        self.golden_ts_path = golden_ts_path
        self.anomaly_ts_path = anomaly_ts_path
        self.ground_truth_path = ground_truth_path
        self.threshold = threshold
        self.ground_truth_sensor = None

        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                gt_data = json.load(f)
                self.ground_truth_sensor = gt_data.get('ts_col', gt_data.get('variable', ''))
                if not self.ground_truth_sensor.endswith('_diff'):
                    self.ground_truth_sensor += '_diff'

        # Method weights (based on observed performance)
        self.method_weights = {
            'temporal_precedence': 3.0,      # Best single method (33.3%)
            'directional_asymmetry': 1.5,    # Equal to baseline
            'rate_of_change': 1.5,           # Equal to baseline
            'voting': 1.0,                   # Baseline
            'sub_window': 2.5                # Expected good for spike
        }

    def run_temporal_precedence(self) -> Dict:
        """Run temporal precedence detector"""
        print("\n" + "="*80)
        print("Running Method 1: Temporal Precedence")
        print("="*80)

        try:
            # Import and run
            from temporal_precedence_detector import TemporalPrecedenceDetector

            detector = TemporalPrecedenceDetector(
                golden_weights_path=self.golden_weights_path,
                anomaly_weights_path=self.anomaly_weights_path,
                ground_truth_path=self.ground_truth_path,
                threshold=self.threshold
            )

            results = detector.detect_epicenter()

            if results['success']:
                return {
                    'success': True,
                    'epicenter': results['epicenter'],
                    'confidence': 1.0  # Temporal precedence is deterministic
                }
            else:
                return {'success': False}

        except Exception as e:
            print(f"ERROR: {e}")
            return {'success': False}

    def run_directional_asymmetry(self) -> Dict:
        """Run directional asymmetry detector"""
        print("\n" + "="*80)
        print("Running Method 2: Directional Asymmetry")
        print("="*80)

        try:
            from directional_asymmetry_detector import DirectionalAsymmetryDetector

            detector = DirectionalAsymmetryDetector(
                golden_weights_path=self.golden_weights_path,
                anomaly_weights_path=self.anomaly_weights_path,
                ground_truth_path=self.ground_truth_path,
                threshold=self.threshold
            )

            results = detector.detect_epicenter(top_n_windows=10)

            if results['success']:
                # Confidence = normalized asymmetry score (0-1)
                confidence = min(abs(results['epicenter_score']), 1.0)

                return {
                    'success': True,
                    'epicenter': results['epicenter'],
                    'confidence': float(confidence)
                }
            else:
                return {'success': False}

        except Exception as e:
            print(f"ERROR: {e}")
            return {'success': False}

    def run_rate_of_change(self) -> Dict:
        """Run rate of change detector"""
        print("\n" + "="*80)
        print("Running Method 3: Rate of Change")
        print("="*80)

        try:
            from rate_of_change_detector import RateOfChangeDetector

            detector = RateOfChangeDetector(
                golden_weights_path=self.golden_weights_path,
                anomaly_weights_path=self.anomaly_weights_path,
                ground_truth_path=self.ground_truth_path,
                threshold=self.threshold
            )

            results = detector.detect_epicenter(use_spike_score=True)

            if results['success']:
                # Confidence = normalized spike score (0-1)
                confidence = min(results['epicenter_score'] / 10.0, 1.0)

                return {
                    'success': True,
                    'epicenter': results['epicenter'],
                    'confidence': float(confidence)
                }
            else:
                return {'success': False}

        except Exception as e:
            print(f"ERROR: {e}")
            return {'success': False}

    def run_voting(self) -> Dict:
        """Run multi-window voting detector"""
        print("\n" + "="*80)
        print("Running Method 4: Multi-Window Voting")
        print("="*80)

        try:
            from multi_window_voting_detector import MultiWindowVotingDetector

            detector = MultiWindowVotingDetector(
                golden_weights_path=self.golden_weights_path,
                anomaly_weights_path=self.anomaly_weights_path,
                ground_truth_path=self.ground_truth_path,
                threshold=self.threshold
            )

            window_detections = detector.detect_epicenter_per_window()
            results = detector.apply_voting_strategy(window_detections, 'hybrid')

            if results['n_windows'] > 0:
                return {
                    'success': True,
                    'epicenter': results['epicenter'],
                    'confidence': results['consensus'] / 100.0  # Convert % to 0-1
                }
            else:
                return {'success': False}

        except Exception as e:
            print(f"ERROR: {e}")
            return {'success': False}

    def run_sub_window(self) -> Dict:
        """Run sub-window analysis detector"""
        print("\n" + "="*80)
        print("Running Method 5: Sub-Window Analysis")
        print("="*80)

        if not self.golden_ts_path or not self.anomaly_ts_path:
            print("WARNING: Time series paths not provided, skipping sub-window analysis")
            return {'success': False}

        try:
            from sub_window_detector import SubWindowDetector

            detector = SubWindowDetector(
                golden_ts_path=self.golden_ts_path,
                anomaly_ts_path=self.anomaly_ts_path,
                ground_truth_path=self.ground_truth_path,
                window_size=15
            )

            results = detector.find_first_single_sensor_divergence(
                threshold=1.0,
                stride=5
            )

            if results['success']:
                # Confidence = 1.0 if single-sensor window found, 0.5 otherwise
                confidence = 1.0 if results['n_single_sensor_windows'] > 0 else 0.5

                return {
                    'success': True,
                    'epicenter': results['epicenter'],
                    'confidence': confidence
                }
            else:
                return {'success': False}

        except Exception as e:
            print(f"ERROR: {e}")
            return {'success': False}

    def run_ensemble(self) -> Dict:
        """Run all detectors and combine with weighted voting"""
        print("\n" + "="*80)
        print("ENSEMBLE META-DETECTOR")
        print("="*80)
        print(f"Ground truth: {self.ground_truth_sensor}\n")

        # Run all methods
        methods = {
            'temporal_precedence': self.run_temporal_precedence(),
            'directional_asymmetry': self.run_directional_asymmetry(),
            'rate_of_change': self.run_rate_of_change(),
            'voting': self.run_voting(),
            'sub_window': self.run_sub_window()
        }

        # Aggregate weighted votes
        weighted_votes = defaultdict(float)
        method_votes = {}

        print("\n" + "="*80)
        print("ENSEMBLE VOTING")
        print("="*80)

        for method_name, result in methods.items():
            if result['success']:
                epicenter = result['epicenter']
                confidence = result.get('confidence', 1.0)
                weight = self.method_weights[method_name]

                # Weighted vote = method_weight × method_confidence
                vote_value = weight * confidence

                weighted_votes[epicenter] += vote_value
                method_votes[method_name] = {
                    'epicenter': epicenter,
                    'confidence': confidence,
                    'weight': weight,
                    'vote_value': vote_value
                }

                is_gt = " ✅ GT" if epicenter == self.ground_truth_sensor else " ❌"
                print(f"{method_name:25s}: {epicenter:45s} "
                      f"(conf={confidence:.2f}, weight={weight:.1f}, vote={vote_value:.2f}){is_gt}")
            else:
                print(f"{method_name:25s}: FAILED")
                method_votes[method_name] = None

        if not weighted_votes:
            return {
                'success': False,
                'message': 'All methods failed',
                'method_results': methods
            }

        # Determine winner
        total_votes = sum(weighted_votes.values())
        sorted_candidates = sorted(weighted_votes.items(),
                                  key=lambda x: x[1], reverse=True)

        epicenter = sorted_candidates[0][0]
        epicenter_votes = sorted_candidates[0][1]
        consensus = (epicenter_votes / total_votes) * 100

        matches_gt = epicenter == self.ground_truth_sensor if self.ground_truth_sensor else None

        # Count agreeing methods
        agreeing_methods = [m for m, v in method_votes.items()
                           if v and v['epicenter'] == epicenter]
        n_agreeing = len(agreeing_methods)
        n_total = len([v for v in method_votes.values() if v])

        print("\n" + "="*80)
        print("ENSEMBLE RESULT")
        print("="*80)
        print(f"Winner: {epicenter}")
        print(f"Weighted votes: {epicenter_votes:.2f} / {total_votes:.2f} ({consensus:.1f}%)")
        print(f"Agreeing methods: {n_agreeing}/{n_total} - {', '.join(agreeing_methods)}")
        print(f"Ground truth: {self.ground_truth_sensor}")
        print(f"Matches: {'✅ YES' if matches_gt else '❌ NO'}")
        print("="*80)

        print("\nFull ranking:")
        for rank, (sensor, votes) in enumerate(sorted_candidates, 1):
            pct = (votes / total_votes) * 100
            is_gt = " ✅ GT" if sensor == self.ground_truth_sensor else ""
            print(f"  {rank}. {sensor:50s} {votes:6.2f} votes ({pct:5.1f}%){is_gt}")

        return {
            'success': True,
            'epicenter': epicenter,
            'epicenter_votes': float(epicenter_votes),
            'total_votes': float(total_votes),
            'consensus': float(consensus),
            'ground_truth': self.ground_truth_sensor,
            'matches_gt': matches_gt,
            'method_results': methods,
            'method_votes': method_votes,
            'ranking': [(s, float(v), (v/total_votes)*100) for s, v in sorted_candidates],
            'n_agreeing_methods': n_agreeing,
            'n_total_methods': n_total,
            'agreeing_methods': agreeing_methods
        }

    def generate_report(self, results: Dict, output_dir: str):
        """Generate detailed report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / 'ensemble_results.json'
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj
            json.dump(results, f, indent=2, default=convert)

        # Generate text report
        report_path = output_path / 'ensemble_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENSEMBLE META-DETECTOR REPORT\n")
            f.write("="*80 + "\n\n")

            if not results['success']:
                f.write(f"FAILED: {results['message']}\n")
                return

            f.write(f"Epicenter Detected: {results['epicenter']}\n")
            f.write(f"Weighted Votes: {results['epicenter_votes']:.2f} / {results['total_votes']:.2f}\n")
            f.write(f"Consensus: {results['consensus']:.1f}%\n")
            f.write(f"Agreeing Methods: {results['n_agreeing_methods']}/{results['n_total_methods']}\n")
            f.write(f"Ground Truth: {results['ground_truth']}\n")
            f.write(f"Matches: {'✅ YES' if results['matches_gt'] else '❌ NO'}\n\n")

            f.write("-"*80 + "\n")
            f.write("METHOD VOTES\n")
            f.write("-"*80 + "\n\n")

            for method, vote_info in results['method_votes'].items():
                if vote_info:
                    f.write(f"{method}:\n")
                    f.write(f"  Epicenter: {vote_info['epicenter']}\n")
                    f.write(f"  Confidence: {vote_info['confidence']:.2f}\n")
                    f.write(f"  Weight: {vote_info['weight']:.1f}\n")
                    f.write(f"  Vote Value: {vote_info['vote_value']:.2f}\n")
                else:
                    f.write(f"{method}: FAILED\n")
                f.write("\n")

            f.write("-"*80 + "\n")
            f.write("FULL RANKING\n")
            f.write("-"*80 + "\n\n")

            for rank, (sensor, votes, pct) in enumerate(results['ranking'], 1):
                is_gt = " ✅ GT" if sensor == results['ground_truth'] else ""
                f.write(f"{rank}. {sensor:50s} {votes:6.2f} votes ({pct:5.1f}%){is_gt}\n")

        print(f"\n✅ Report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Ensemble Meta-Detector')
    parser.add_argument('--golden-weights', required=True, help='Path to golden weights CSV')
    parser.add_argument('--anomaly-weights', required=True, help='Path to anomaly weights CSV')
    parser.add_argument('--golden-ts', help='Path to golden time series CSV (for sub-window)')
    parser.add_argument('--anomaly-ts', help='Path to anomaly time series CSV (for sub-window)')
    parser.add_argument('--ground-truth', help='Path to ground truth JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.01, help='Edge change threshold')

    args = parser.parse_args()

    print("="*80)
    print("ENSEMBLE META-DETECTOR")
    print("="*80)
    print("Combining 5 detection methods with weighted voting")
    print()

    detector = EnsembleDetector(
        golden_weights_path=args.golden_weights,
        anomaly_weights_path=args.anomaly_weights,
        golden_ts_path=args.golden_ts,
        anomaly_ts_path=args.anomaly_ts,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold
    )

    results = detector.run_ensemble()

    if results['success']:
        detector.generate_report(results, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
