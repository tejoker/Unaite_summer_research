#!/usr/bin/env python3
"""
End-to-End Anomaly Detection and Epicenter Identification Pipeline

This script provides a complete pipeline that:
1. Analyzes causal graphs (golden vs anomaly)
2. Detects anomalies (weight changes, structural changes)
3. Identifies epicenter (root cause sensor)
4. Traces causal cascade (propagation paths)
5. Generates comprehensive report

Usage:
    python3 end_to_end_anomaly_pipeline.py \\
        --golden-weights path/to/golden/weights.csv \\
        --anomaly-weights path/to/anomaly/weights.csv \\
        --ground-truth path/to/ground_truth.json \\
        --output-dir results/analysis
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

# Import our detectors
from temporal_precedence_detector import TemporalPrecedenceDetector
from causal_cascade_detector import CausalCascadeDetector


class EndToEndAnomalyPipeline:
    def __init__(self, golden_weights_path: str, anomaly_weights_path: str,
                 ground_truth_path: str = None, threshold: float = 0.01):
        self.golden_weights_path = golden_weights_path
        self.anomaly_weights_path = anomaly_weights_path
        self.ground_truth_path = ground_truth_path
        self.threshold = threshold
        self.ground_truth_sensor = None
        self.ground_truth_data = None

        # Load data
        self.golden_df = pd.read_csv(golden_weights_path)
        self.anomaly_df = pd.read_csv(anomaly_weights_path)

        if ground_truth_path:
            with open(ground_truth_path, 'r') as f:
                self.ground_truth_data = json.load(f)
                self.ground_truth_sensor = self.ground_truth_data.get('ts_col',
                                           self.ground_truth_data.get('variable', ''))
                if not self.ground_truth_sensor.endswith('_diff'):
                    self.ground_truth_sensor += '_diff'

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

    def step1_detect_anomalies(self) -> Dict:
        """
        Step 1: Detect anomalies in causal graph

        Returns dict with:
        - n_windows_total: Total windows analyzed
        - n_windows_anomalous: Windows with significant changes
        - anomaly_windows: List of window indices with anomalies
        - total_changes: Total edge weight changes
        - avg_change_per_window: Average change per window
        """
        print("\n" + "="*80)
        print("STEP 1: ANOMALY DETECTION")
        print("="*80)

        # Group by window
        window_stats = []

        for window_idx in sorted(self.merged_df['window_idx'].unique()):
            window_data = self.merged_df[self.merged_df['window_idx'] == window_idx]

            # Compute statistics
            total_change = window_data['weight_diff'].sum()
            max_change = window_data['weight_diff'].max()
            n_significant = len(window_data[window_data['weight_diff'] >= self.threshold])

            is_anomalous = n_significant > 0

            window_stats.append({
                'window_idx': int(window_idx),
                'total_change': float(total_change),
                'max_change': float(max_change),
                'n_significant_edges': n_significant,
                'is_anomalous': is_anomalous
            })

        anomalous_windows = [w for w in window_stats if w['is_anomalous']]

        print(f"\nTotal windows analyzed: {len(window_stats)}")
        print(f"Anomalous windows detected: {len(anomalous_windows)}")
        print(f"Anomaly detection rate: {len(anomalous_windows)/len(window_stats)*100:.1f}%")

        if anomalous_windows:
            print(f"\nFirst anomalous window: {anomalous_windows[0]['window_idx']}")
            print(f"Last anomalous window: {anomalous_windows[-1]['window_idx']}")

            total_changes = sum(w['total_change'] for w in anomalous_windows)
            avg_change = total_changes / len(anomalous_windows)
            print(f"Average change per anomalous window: {avg_change:.4f}")

        return {
            'n_windows_total': len(window_stats),
            'n_windows_anomalous': len(anomalous_windows),
            'anomaly_detection_rate': len(anomalous_windows)/len(window_stats)*100,
            'anomaly_windows': [w['window_idx'] for w in anomalous_windows],
            'window_stats': window_stats,
            'first_anomaly_window': anomalous_windows[0]['window_idx'] if anomalous_windows else None
        }

    def detect_multiple_anomalies(self, gap_threshold: int = 5) -> List[Dict]:
        """
        Detect multiple anomalies using temporal gap detection.

        Args:
            gap_threshold: Maximum gap (in windows) to consider same anomaly

        Returns:
            List of anomalies, each with:
            - windows: List of window indices
            - first_window: First affected window
            - last_window: Last affected window
            - n_windows: Number of windows affected
            - epicenter: Epicenter identification results
            - cascade: Cascade tracing results
        """
        print("\n" + "="*80)
        print("MULTIPLE ANOMALY DETECTION (Gap-Based Temporal Segmentation)")
        print("="*80)
        print(f"Gap threshold: {gap_threshold} windows\n")

        # Step 1: Get all anomalous windows
        anomaly_detection = self.step1_detect_anomalies()
        all_windows = sorted(anomaly_detection['anomaly_windows'])

        if not all_windows:
            print("No anomalous windows detected.")
            return []

        # Step 2: Segment by gaps
        anomalies = []
        current_segment = [all_windows[0]]

        for i in range(1, len(all_windows)):
            gap = all_windows[i] - all_windows[i-1]

            if gap <= gap_threshold:
                # Same anomaly - add to current segment
                current_segment.append(all_windows[i])
            else:
                # Gap detected - save current segment, start new one
                print(f"\nAnomaly #{len(anomalies) + 1} detected:")
                print(f"  Windows: {min(current_segment)} - {max(current_segment)}")
                print(f"  Number of windows: {len(current_segment)}")
                print(f"  Gap to next anomaly: {gap} windows")

                anomaly = self._analyze_anomaly_segment(current_segment, len(anomalies) + 1)
                anomalies.append(anomaly)

                # Start new segment
                current_segment = [all_windows[i]]

        # Don't forget last segment
        if current_segment:
            print(f"\nAnomaly #{len(anomalies) + 1} detected:")
            print(f"  Windows: {min(current_segment)} - {max(current_segment)}")
            print(f"  Number of windows: {len(current_segment)}")

            anomaly = self._analyze_anomaly_segment(current_segment, len(anomalies) + 1)
            anomalies.append(anomaly)

        print("\n" + "="*80)
        print(f"TOTAL ANOMALIES DETECTED: {len(anomalies)}")
        print("="*80)

        return anomalies

    def _analyze_anomaly_segment(self, windows: List[int], anomaly_id: int) -> Dict:
        """
        Analyze a single anomaly segment (identify epicenter and trace cascade).

        Args:
            windows: List of window indices for this anomaly
            anomaly_id: Anomaly identifier (1, 2, 3, ...)

        Returns:
            Dict with anomaly analysis results
        """
        print(f"\n  Analyzing Anomaly #{anomaly_id}...")

        # Identify epicenter for this window subset
        epicenter_result = self._identify_epicenter_for_windows(windows)

        if not epicenter_result['success']:
            print(f"  ❌ Could not identify epicenter for anomaly #{anomaly_id}")
            return {
                'anomaly_id': anomaly_id,
                'windows': windows,
                'first_window': min(windows),
                'last_window': max(windows),
                'n_windows': len(windows),
                'epicenter': None,
                'cascade': None,
                'success': False
            }

        epicenter = epicenter_result['epicenter']
        print(f"  ✅ Epicenter: {epicenter}")

        # Trace cascade from first window
        cascade_result = self._trace_cascade_for_window(min(windows), epicenter)

        if cascade_result['success']:
            cascade_depth = len(cascade_result['cascade'].get('cascade', {}))
            print(f"  ✅ Cascade traced: {cascade_depth} depth levels")

        return {
            'anomaly_id': anomaly_id,
            'windows': windows,
            'first_window': min(windows),
            'last_window': max(windows),
            'n_windows': len(windows),
            'epicenter_result': epicenter_result,
            'cascade_result': cascade_result,
            'success': True
        }

    def _identify_epicenter_for_windows(self, windows: List[int]) -> Dict:
        """
        Identify epicenter for a specific subset of windows.

        Args:
            windows: List of window indices to analyze

        Returns:
            Epicenter identification results
        """
        # Filter merged_df to only include these windows
        window_subset = self.merged_df[self.merged_df['window_idx'].isin(windows)]

        # Find earliest window with significant changes
        earliest_window = min(windows)
        earliest_data = window_subset[window_subset['window_idx'] == earliest_window]

        # Get significant edges
        significant = earliest_data[earliest_data['weight_diff'] >= self.threshold]

        if len(significant) == 0:
            return {'success': False, 'message': 'No significant changes in earliest window'}

        # Calculate total change per sensor
        sensor_changes = defaultdict(float)
        for _, row in significant.iterrows():
            sensor_changes[row['parent_name']] += row['weight_diff']
            sensor_changes[row['child_name']] += row['weight_diff']

        # Find sensor with maximum change
        epicenter = max(sensor_changes.items(), key=lambda x: x[1])[0]

        # Check against ground truth
        matches_gt = False
        if self.ground_truth_sensor:
            matches_gt = (epicenter == self.ground_truth_sensor)

        return {
            'success': True,
            'epicenter': epicenter,
            'earliest_window': earliest_window,
            'detection_method': 'earliest_window_with_magnitude',
            'matches_gt': matches_gt,
            'sensor_changes': dict(sensor_changes)
        }

    def _trace_cascade_for_window(self, window_idx: int, epicenter: str) -> Dict:
        """
        Trace causal cascade for a specific window and epicenter.

        Args:
            window_idx: Window index to analyze
            epicenter: Epicenter sensor name

        Returns:
            Cascade tracing results
        """
        detector = CausalCascadeDetector(
            golden_weights_csv=self.golden_weights_path,
            anomaly_weights_csv=self.anomaly_weights_path,
            ground_truth_json=self.ground_truth_path,
            threshold=self.threshold
        )

        cascade = detector.trace_cascade_from_epicenter(
            window_idx=window_idx,
            epicenter_var=epicenter,
            max_depth=3
        )

        return {
            'success': True,
            'cascade': cascade
        }

    def step2_identify_epicenter(self) -> Dict:
        """
        Step 2: Identify epicenter (root cause sensor)

        Uses temporal precedence detector to find which sensor changed first.
        """
        print("\n" + "="*80)
        print("STEP 2: EPICENTER IDENTIFICATION")
        print("="*80)

        detector = TemporalPrecedenceDetector(
            golden_weights_path=self.golden_weights_path,
            anomaly_weights_path=self.anomaly_weights_path,
            ground_truth_path=self.ground_truth_path,
            threshold=self.threshold
        )

        results = detector.detect_epicenter()

        if results['success']:
            print(f"\n✅ Epicenter identified: {results['epicenter']}")
            print(f"Detection method: {results['detection_method']}")
            print(f"First detected in window: {results['earliest_window']}")

            if self.ground_truth_sensor:
                print(f"Ground truth: {self.ground_truth_sensor}")
                print(f"Matches ground truth: {'✅ YES' if results['matches_gt'] else '❌ NO'}")

            return results
        else:
            print("\n❌ Epicenter identification failed")
            return {'success': False, 'message': 'No epicenter detected'}

    def step3_trace_cascade(self, epicenter: str) -> Dict:
        """
        Step 3: Trace causal cascade from epicenter

        Shows how anomaly propagates from epicenter to other sensors.
        """
        print("\n" + "="*80)
        print("STEP 3: CAUSAL CASCADE TRACING")
        print("="*80)
        print(f"Tracing propagation from epicenter: {epicenter}\n")

        detector = CausalCascadeDetector(
            golden_weights_csv=self.golden_weights_path,
            anomaly_weights_csv=self.anomaly_weights_path,
            ground_truth_json=self.ground_truth_path,
            threshold=self.threshold
        )

        # Detect epicenter window-by-window
        epicenter_results = detector.detect_epicenter_window_by_window()

        # Get windows where epicenter was detected
        epicenter_windows = epicenter_results[
            epicenter_results['epicenter'] == epicenter
        ]['window_idx'].tolist()

        print(f"Windows where {epicenter} detected as epicenter: {len(epicenter_windows)}")

        if epicenter_windows:
            # Trace cascade from first detection
            first_window = min(epicenter_windows)
            cascade = detector.trace_cascade_from_epicenter(
                window_idx=first_window,
                epicenter_var=epicenter,
                max_depth=3
            )

            print(f"\nCascade from window {first_window}:")
            for depth, edges in cascade['cascade'].items():
                print(f"  Depth {depth}: {len(edges)} edges")
                for edge in edges[:3]:  # Show first 3
                    print(f"    - {edge['from']} → {edge['to']}: {edge['weight_diff']:.4f} change")

            return {
                'success': True,
                'epicenter': epicenter,
                'n_windows_detected': len(epicenter_windows),
                'first_detection_window': first_window,
                'cascade': cascade,
                'epicenter_results': epicenter_results
            }
        else:
            print(f"⚠️  Warning: {epicenter} never detected as epicenter in any window")
            return {
                'success': False,
                'message': f'{epicenter} not detected as epicenter',
                'epicenter_results': epicenter_results
            }

    def generate_comprehensive_report(self, anomaly_results: Dict,
                                     epicenter_results: Dict,
                                     cascade_results: Dict,
                                     output_dir: str):
        """
        Generate comprehensive HTML and text reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Text report
        report_path = output_path / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("END-TO-END ANOMALY ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Golden weights: {self.golden_weights_path}\n")
            f.write(f"Anomaly weights: {self.anomaly_weights_path}\n")
            f.write(f"Threshold: {self.threshold}\n\n")

            # Section 1: Anomaly Detection
            f.write("-"*80 + "\n")
            f.write("1. ANOMALY DETECTION RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total windows analyzed: {anomaly_results['n_windows_total']}\n")
            f.write(f"Anomalous windows detected: {anomaly_results['n_windows_anomalous']}\n")
            f.write(f"Detection rate: {anomaly_results['anomaly_detection_rate']:.1f}%\n")
            f.write(f"First anomaly window: {anomaly_results['first_anomaly_window']}\n\n")

            # Section 2: Epicenter Identification
            f.write("-"*80 + "\n")
            f.write("2. EPICENTER IDENTIFICATION\n")
            f.write("-"*80 + "\n")

            if epicenter_results['success']:
                f.write(f"✅ Epicenter detected: {epicenter_results['epicenter']}\n")
                f.write(f"Detection method: {epicenter_results['detection_method']}\n")
                f.write(f"First detection window: {epicenter_results['earliest_window']}\n")

                if self.ground_truth_sensor:
                    f.write(f"Ground truth: {self.ground_truth_sensor}\n")
                    f.write(f"Matches ground truth: {'✅ YES' if epicenter_results['matches_gt'] else '❌ NO'}\n")

                f.write(f"\nFirst detection timeline:\n")
                for sensor, info in sorted(epicenter_results['first_detection'].items(),
                                          key=lambda x: (x[1]['first_window'] or 999, -x[1].get('first_change', 0))):
                    if info['first_window'] is not None:
                        is_epicenter = " 🎯" if sensor == epicenter_results['epicenter'] else ""
                        is_gt = " ✅ GT" if sensor == self.ground_truth_sensor else ""
                        f.write(f"  Window {info['first_window']:3d}: {sensor:50s}{is_epicenter}{is_gt}\n")
            else:
                f.write(f"❌ Epicenter detection failed: {epicenter_results.get('message', 'Unknown error')}\n")

            f.write("\n")

            # Section 3: Causal Cascade
            f.write("-"*80 + "\n")
            f.write("3. CAUSAL CASCADE ANALYSIS\n")
            f.write("-"*80 + "\n")

            if cascade_results['success']:
                f.write(f"Epicenter: {cascade_results['epicenter']}\n")
                f.write(f"Windows detected as epicenter: {cascade_results['n_windows_detected']}\n")
                f.write(f"First detection window: {cascade_results['first_detection_window']}\n\n")

                f.write(f"Propagation cascade (from window {cascade_results['first_detection_window']}):\n")
                cascade_data = cascade_results['cascade'].get('cascade', {})
                for depth, edges in cascade_data.items():
                    f.write(f"  Depth {depth} ({len(edges)} edges):\n")
                    for edge in edges[:10]:
                        f.write(f"    - {edge['from']:40s} → {edge['to']:40s} (Δ={edge['weight_diff']:.4f})\n")
            else:
                f.write(f"❌ Cascade tracing failed: {cascade_results.get('message', 'Unknown error')}\n")

            f.write("\n")

            # Section 4: Summary
            f.write("-"*80 + "\n")
            f.write("4. SUMMARY\n")
            f.write("-"*80 + "\n")

            if epicenter_results['success']:
                confidence = "HIGH" if epicenter_results['matches_gt'] else "MEDIUM"
                f.write(f"Root cause sensor: {epicenter_results['epicenter']}\n")
                f.write(f"Confidence: {confidence}\n")
                f.write(f"First affected window: {epicenter_results['earliest_window']}\n")

                if self.ground_truth_data:
                    f.write(f"\nGround truth information:\n")
                    f.write(f"  Anomaly type: {self.ground_truth_data.get('anomaly', 'N/A')}\n")
                    f.write(f"  Injected at: row {self.ground_truth_data.get('start', 'N/A')}\n")
                    f.write(f"  Duration: {self.ground_truth_data.get('length', 'N/A')} rows\n")
                    f.write(f"  Magnitude: {self.ground_truth_data.get('magnitude', 'N/A')}\n")
            else:
                f.write("Could not identify root cause sensor.\n")

            f.write("\n" + "="*80 + "\n")

        # JSON report (machine-readable)
        json_path = output_path / 'analysis_results.json'
        with open(json_path, 'w') as f:
            results = {
                'timestamp': timestamp,
                'golden_weights': self.golden_weights_path,
                'anomaly_weights': self.anomaly_weights_path,
                'threshold': self.threshold,
                'anomaly_detection': anomaly_results,
                'epicenter_identification': epicenter_results,
                'cascade_analysis': cascade_results
            }

            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                return obj

            json.dump(results, f, indent=2, default=convert)

        print(f"\n✅ Comprehensive report saved to: {report_path}")
        print(f"✅ JSON results saved to: {json_path}")

        return report_path, json_path

    def run_complete_pipeline(self, output_dir: str) -> Dict:
        """
        Run complete end-to-end pipeline

        Returns dict with all results
        """
        print("\n" + "="*80)
        print("END-TO-END ANOMALY DETECTION AND EPICENTER IDENTIFICATION")
        print("="*80)
        print(f"Golden weights: {self.golden_weights_path}")
        print(f"Anomaly weights: {self.anomaly_weights_path}")
        print(f"Output directory: {output_dir}")
        print(f"Threshold: {self.threshold}")

        if self.ground_truth_sensor:
            print(f"Ground truth sensor: {self.ground_truth_sensor}")

        # Step 1: Detect anomalies
        anomaly_results = self.step1_detect_anomalies()

        if anomaly_results['n_windows_anomalous'] == 0:
            print("\n❌ No anomalies detected. Pipeline stopped.")
            return {
                'success': False,
                'message': 'No anomalies detected',
                'anomaly_detection': anomaly_results
            }

        # Step 2: Identify epicenter
        epicenter_results = self.step2_identify_epicenter()

        if not epicenter_results['success']:
            print("\n❌ Epicenter identification failed. Pipeline stopped.")
            return {
                'success': False,
                'message': 'Epicenter identification failed',
                'anomaly_detection': anomaly_results,
                'epicenter_identification': epicenter_results
            }

        # Step 3: Trace cascade
        cascade_results = self.step3_trace_cascade(epicenter_results['epicenter'])

        # Generate reports
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)

        report_path, json_path = self.generate_comprehensive_report(
            anomaly_results, epicenter_results, cascade_results, output_dir
        )

        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\n✅ Anomalies detected: {anomaly_results['n_windows_anomalous']} windows")
        print(f"✅ Epicenter identified: {epicenter_results['epicenter']}")
        print(f"✅ First detection: Window {epicenter_results['earliest_window']}")

        if self.ground_truth_sensor:
            match_symbol = "✅" if epicenter_results['matches_gt'] else "❌"
            print(f"{match_symbol} Ground truth match: {epicenter_results['matches_gt']}")

        print(f"\n📄 Full report: {report_path}")
        print(f"📄 JSON results: {json_path}")

        return {
            'success': True,
            'anomaly_detection': anomaly_results,
            'epicenter_identification': epicenter_results,
            'cascade_analysis': cascade_results,
            'report_path': str(report_path),
            'json_path': str(json_path)
        }


    def run_multiple_anomaly_pipeline(self, output_dir: str, gap_threshold: int = 5) -> Dict:
        """
        Run pipeline for detecting and analyzing multiple anomalies.

        Args:
            output_dir: Output directory for results
            gap_threshold: Maximum gap (in windows) to consider same anomaly

        Returns:
            Dict with all results
        """
        print("\n" + "="*80)
        print("MULTIPLE ANOMALY DETECTION PIPELINE")
        print("="*80)
        print(f"Golden weights: {self.golden_weights_path}")
        print(f"Anomaly weights: {self.anomaly_weights_path}")
        print(f"Output directory: {output_dir}")
        print(f"Gap threshold: {gap_threshold} windows")

        # Detect and analyze all anomalies
        anomalies = self.detect_multiple_anomalies(gap_threshold=gap_threshold)

        if not anomalies:
            print("\n❌ No anomalies detected.")
            return {'success': False, 'message': 'No anomalies detected'}

        # Generate reports for each anomaly
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate master report
        self._generate_multi_anomaly_report(anomalies, output_dir)

        # Generate individual reports for each anomaly
        for anomaly in anomalies:
            anomaly_dir = output_path / f"anomaly_{anomaly['anomaly_id']}"
            self._generate_single_anomaly_report(anomaly, str(anomaly_dir))

        print("\n" + "="*80)
        print("MULTIPLE ANOMALY PIPELINE COMPLETE")
        print("="*80)
        print(f"\nTotal anomalies detected: {len(anomalies)}")
        for anomaly in anomalies:
            epicenter = anomaly.get('epicenter_result', {}).get('epicenter', 'Unknown')
            windows = f"{anomaly['first_window']}-{anomaly['last_window']}"
            print(f"  Anomaly #{anomaly['anomaly_id']}: Windows {windows}, Epicenter: {epicenter}")

        print(f"\n📄 Master report: {output_path / 'multi_anomaly_summary.txt'}")

        return {
            'success': True,
            'n_anomalies': len(anomalies),
            'anomalies': anomalies,
            'output_dir': output_dir
        }

    def _generate_multi_anomaly_report(self, anomalies: List[Dict], output_dir: str):
        """Generate master report for all detected anomalies."""
        output_path = Path(output_dir)
        report_path = output_path / 'multi_anomaly_summary.txt'

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTIPLE ANOMALY DETECTION - SUMMARY REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Golden weights: {self.golden_weights_path}\n")
            f.write(f"Anomaly weights: {self.anomaly_weights_path}\n")
            f.write(f"Total anomalies detected: {len(anomalies)}\n\n")

            for anomaly in anomalies:
                f.write("-"*80 + "\n")
                f.write(f"ANOMALY #{anomaly['anomaly_id']}\n")
                f.write("-"*80 + "\n")
                f.write(f"Time range: Windows {anomaly['first_window']} - {anomaly['last_window']}\n")
                f.write(f"Number of windows affected: {anomaly['n_windows']}\n")

                epicenter_result = anomaly.get('epicenter_result', {})
                if epicenter_result.get('success'):
                    epicenter = epicenter_result['epicenter']
                    f.write(f"Detected epicenter: {epicenter}\n")

                    if self.ground_truth_sensor:
                        matches = epicenter_result.get('matches_gt', False)
                        f.write(f"Matches ground truth: {'YES' if matches else 'NO'}\n")

                    # Cascade info
                    cascade_result = anomaly.get('cascade_result', {})
                    if cascade_result.get('success'):
                        cascade_depth = len(cascade_result['cascade'].get('cascade', {}))
                        f.write(f"Cascade depth: {cascade_depth} levels\n")
                else:
                    f.write("Could not identify epicenter\n")

                f.write("\n")

            f.write("="*80 + "\n")

        print(f"\n✅ Master report saved to: {report_path}")

        # Also save JSON
        json_path = output_path / 'multi_anomaly_results.json'
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                return obj

            results = {
                'timestamp': timestamp,
                'n_anomalies': len(anomalies),
                'anomalies': anomalies
            }
            json.dump(results, f, indent=2, default=convert)

        print(f"✅ JSON results saved to: {json_path}")

    def _generate_single_anomaly_report(self, anomaly: Dict, output_dir: str):
        """Generate detailed report for a single anomaly."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_path = output_path / 'analysis_report.txt'
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"ANOMALY #{anomaly['anomaly_id']} - DETAILED ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {timestamp}\n\n")

            f.write("TEMPORAL INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"First window: {anomaly['first_window']}\n")
            f.write(f"Last window: {anomaly['last_window']}\n")
            f.write(f"Number of windows affected: {anomaly['n_windows']}\n\n")

            epicenter_result = anomaly.get('epicenter_result', {})
            if epicenter_result.get('success'):
                f.write("EPICENTER IDENTIFICATION\n")
                f.write("-"*80 + "\n")
                f.write(f"Detected epicenter: {epicenter_result['epicenter']}\n")
                f.write(f"Detection method: {epicenter_result.get('detection_method', 'N/A')}\n")

                if self.ground_truth_sensor:
                    matches = epicenter_result.get('matches_gt', False)
                    f.write(f"Ground truth: {self.ground_truth_sensor}\n")
                    f.write(f"Matches ground truth: {'YES' if matches else 'NO'}\n")

                f.write("\n")

            cascade_result = anomaly.get('cascade_result', {})
            if cascade_result.get('success'):
                f.write("CAUSAL CASCADE\n")
                f.write("-"*80 + "\n")
                cascade_data = cascade_result['cascade'].get('cascade', {})
                for depth, edges in cascade_data.items():
                    f.write(f"  Depth {depth} ({len(edges)} edges):\n")
                    for edge in edges[:5]:
                        f.write(f"    - {edge['from']:40s} → {edge['to']:40s} (Δ={edge['weight_diff']:.4f})\n")

            f.write("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End Anomaly Detection and Epicenter Identification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python3 end_to_end_anomaly_pipeline.py \\
        --golden-weights results/golden/weights_enhanced.csv \\
        --anomaly-weights results/anomaly/weights_enhanced.csv \\
        --output-dir results/analysis

    # With ground truth validation
    python3 end_to_end_anomaly_pipeline.py \\
        --golden-weights results/golden/weights_enhanced.csv \\
        --anomaly-weights results/anomaly/weights_enhanced.csv \\
        --ground-truth data/ground_truth.json \\
        --output-dir results/analysis \\
        --threshold 0.01
        """
    )

    parser.add_argument('--golden-weights', required=True,
                       help='Path to golden (baseline) causal graph weights CSV')
    parser.add_argument('--anomaly-weights', required=True,
                       help='Path to anomaly causal graph weights CSV')
    parser.add_argument('--ground-truth',
                       help='Path to ground truth JSON (optional, for validation)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for reports and results')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Edge weight change threshold (default: 0.01)')

    args = parser.parse_args()

    # Run pipeline
    pipeline = EndToEndAnomalyPipeline(
        golden_weights_path=args.golden_weights,
        anomaly_weights_path=args.anomaly_weights,
        ground_truth_path=args.ground_truth,
        threshold=args.threshold
    )

    results = pipeline.run_complete_pipeline(output_dir=args.output_dir)

    # Exit with appropriate code
    if results['success']:
        exit(0)
    else:
        exit(1)


if __name__ == '__main__':
    main()
