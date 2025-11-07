#!/usr/bin/env python3
"""
Telemanom Evaluation: Compare DynoTEARS with Published Baselines

This script evaluates anomaly detection results on the Telemanom dataset
and compares with published SOTA baselines.

Published Baselines:
- Telemanom LSTM: F1 = 0.53
- OmniAnomaly: F1 = 0.62
- USAD: F1 = 0.77
- TranAD: F1 = 0.90

Usage:
    python executable/telemanom_evaluator.py --results results/telemanom/ --output results/telemanom_evaluation.json
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TelemanamEvaluator:
    """Evaluate DynoTEARS detection results against ground truth"""

    def __init__(self, ground_truth_file: str):
        """
        Args:
            ground_truth_file: Path to isolated_anomaly_index.csv
        """
        self.ground_truth = pd.read_csv(ground_truth_file)
        logger.info(f"Loaded ground truth for {len(self.ground_truth)} anomalies")

    def load_detection_results(self, result_dir: Path) -> Dict:
        """
        Load detection results for a single anomaly file.

        Expected structure:
        result_dir/
            anomaly/
                rolling_window/
                    detection_summary.json  # Contains detected windows

        Returns:
            Dict with detected_windows list
        """
        detection_file = result_dir / "anomaly" / "rolling_window" / "detection_summary.json"

        if not detection_file.exists():
            logger.warning(f"Detection file not found: {detection_file}")
            return {'detected_windows': [], 'error': 'file_not_found'}

        with open(detection_file, 'r') as f:
            detection_data = json.load(f)

        # Extract anomalous window indices
        # Assuming format: {"anomalous_windows": [45, 46, 47, ...]}
        detected_windows = detection_data.get('anomalous_windows', [])

        return {
            'detected_windows': detected_windows,
            'num_detected': len(detected_windows)
        }

    def check_overlap(self,
                      detected_windows: List[int],
                      true_start: int,
                      true_end: int,
                      window_size: int = 100,
                      stride: int = 10) -> Tuple[bool, List[int]]:
        """
        Check if detected windows overlap with ground truth anomaly region.

        Args:
            detected_windows: List of window indices flagged as anomalous
            true_start: Ground truth anomaly start index
            true_end: Ground truth anomaly end index
            window_size: Size of rolling window
            stride: Stride between windows

        Returns:
            Tuple of (has_overlap, overlapping_windows)
        """
        overlapping_windows = []

        for window_idx in detected_windows:
            # Convert window index to sample range
            window_start = window_idx * stride
            window_end = window_start + window_size

            # Check overlap with ground truth
            if window_start <= true_end and window_end >= true_start:
                overlapping_windows.append(window_idx)

        has_overlap = len(overlapping_windows) > 0
        return has_overlap, overlapping_windows

    def evaluate_single_file(self,
                             anomaly_id: str,
                             result_dir: Path,
                             window_size: int = 100,
                             stride: int = 10) -> Dict:
        """
        Evaluate detection for a single anomaly file.

        Returns:
            Dict with TP/FP/FN classification
        """
        # Get ground truth
        gt_row = self.ground_truth[self.ground_truth['anomaly_id'] == anomaly_id]

        if len(gt_row) == 0:
            logger.error(f"Ground truth not found for {anomaly_id}")
            return {'error': 'ground_truth_not_found'}

        gt_row = gt_row.iloc[0]
        true_start = int(gt_row['start_idx'])
        true_end = int(gt_row['end_idx'])

        # Load detection results
        detection = self.load_detection_results(result_dir)

        if 'error' in detection:
            return {
                'anomaly_id': anomaly_id,
                'error': detection['error'],
                'tp': False,
                'fp': False,
                'fn': True,
                'channel': gt_row['channel'],
                'spacecraft': gt_row['spacecraft'],
                'anomaly_class': gt_row['anomaly_class']
            }

        detected_windows = detection['detected_windows']

        # Check overlap
        has_overlap, overlapping_windows = self.check_overlap(
            detected_windows, true_start, true_end, window_size, stride
        )

        # Classify detection
        if len(detected_windows) == 0:
            # No detection → False Negative
            tp, fp, fn = False, False, True
        elif has_overlap:
            # Detected AND overlaps → True Positive
            tp, fp, fn = True, False, False
        else:
            # Detected but no overlap → False Positive (missed the actual anomaly)
            # This is still FN for the ground truth anomaly
            tp, fp, fn = False, True, True

        return {
            'anomaly_id': anomaly_id,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_detected_windows': len(detected_windows),
            'num_overlapping_windows': len(overlapping_windows),
            'true_start': true_start,
            'true_end': true_end,
            'channel': gt_row['channel'],
            'spacecraft': gt_row['spacecraft'],
            'anomaly_class': gt_row['anomaly_class'],
            'duration': int(gt_row['duration'])
        }

    def evaluate_all(self,
                     results_dir: Path,
                     window_size: int = 100,
                     stride: int = 10) -> List[Dict]:
        """
        Evaluate all anomaly files.

        Returns:
            List of evaluation results for each file
        """
        all_results = []

        for idx, row in self.ground_truth.iterrows():
            anomaly_id = row['anomaly_id']
            result_dir = results_dir / anomaly_id

            if not result_dir.exists():
                logger.warning(f"Result directory not found for {anomaly_id}")
                all_results.append({
                    'anomaly_id': anomaly_id,
                    'error': 'result_dir_not_found',
                    'tp': False,
                    'fp': False,
                    'fn': True,
                    'channel': row['channel'],
                    'spacecraft': row['spacecraft'],
                    'anomaly_class': row['anomaly_class']
                })
                continue

            logger.info(f"[{idx+1}/{len(self.ground_truth)}] Evaluating {anomaly_id}...")
            result = self.evaluate_single_file(anomaly_id, result_dir, window_size, stride)
            all_results.append(result)

        return all_results

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute aggregate metrics from evaluation results.

        Returns:
            Dict with precision, recall, F1-score
        """
        # Count TP/FP/FN
        tp_count = sum(1 for r in results if r.get('tp', False))
        fp_count = sum(1 for r in results if r.get('fp', False))
        fn_count = sum(1 for r in results if r.get('fn', False))

        # Compute metrics
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Breakdown by anomaly class
        class_metrics = {}
        for anomaly_class in ['point', 'contextual']:
            class_results = [r for r in results if r.get('anomaly_class') == anomaly_class]

            if len(class_results) > 0:
                tp = sum(1 for r in class_results if r.get('tp', False))
                fp = sum(1 for r in class_results if r.get('fp', False))
                fn = sum(1 for r in class_results if r.get('fn', False))

                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

                class_metrics[anomaly_class] = {
                    'precision': p,
                    'recall': r,
                    'f1_score': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'total': len(class_results)
                }

        # Breakdown by spacecraft
        spacecraft_metrics = {}
        for spacecraft in ['SMAP', 'MSL']:
            sc_results = [r for r in results if r.get('spacecraft') == spacecraft]

            if len(sc_results) > 0:
                tp = sum(1 for r in sc_results if r.get('tp', False))
                fp = sum(1 for r in sc_results if r.get('fp', False))
                fn = sum(1 for r in sc_results if r.get('fn', False))

                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

                spacecraft_metrics[spacecraft] = {
                    'precision': p,
                    'recall': r,
                    'f1_score': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'total': len(sc_results)
                }

        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp_count,
                'fp': fp_count,
                'fn': fn_count,
                'total': len(results)
            },
            'by_class': class_metrics,
            'by_spacecraft': spacecraft_metrics
        }

    def compare_with_baselines(self, metrics: Dict) -> Dict:
        """
        Compare results with published baselines.

        Returns:
            Comparison table
        """
        baselines = {
            'Telemanom LSTM': {'f1_score': 0.53, 'source': 'Hundman et al. 2018'},
            'OmniAnomaly': {'f1_score': 0.62, 'source': 'Su et al. 2019'},
            'USAD': {'f1_score': 0.77, 'source': 'Audibert et al. 2020'},
            'TranAD': {'f1_score': 0.90, 'source': 'Tuli et al. 2022'}
        }

        our_f1 = metrics['overall']['f1_score']

        comparison = {
            'DynoTEARS (Ours)': {
                'f1_score': our_f1,
                'precision': metrics['overall']['precision'],
                'recall': metrics['overall']['recall'],
                'source': 'This work'
            }
        }

        for method, data in baselines.items():
            comparison[method] = data
            comparison[method]['improvement'] = f"{((our_f1 - data['f1_score']) / data['f1_score'] * 100):.1f}%"

        return comparison

    def generate_report(self,
                        results: List[Dict],
                        metrics: Dict,
                        output_file: Path):
        """
        Generate comprehensive evaluation report.
        """
        comparison = self.compare_with_baselines(metrics)

        report = {
            'summary': {
                'total_anomalies': len(results),
                'processed': len([r for r in results if 'error' not in r or not r['error']]),
                'errors': len([r for r in results if 'error' in r and r['error']])
            },
            'metrics': metrics,
            'comparison_with_baselines': comparison,
            'individual_results': results
        }

        # Save JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_file}")

        # Print summary
        self._print_summary(metrics, comparison)

        return report

    def _print_summary(self, metrics: Dict, comparison: Dict):
        """Print evaluation summary"""
        logger.info("\n" + "="*80)
        logger.info("TELEMANOM EVALUATION SUMMARY")
        logger.info("="*80)

        overall = metrics['overall']
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Precision: {overall['precision']:.4f}")
        logger.info(f"  Recall:    {overall['recall']:.4f}")
        logger.info(f"  F1-Score:  {overall['f1_score']:.4f}")
        logger.info(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")

        logger.info(f"\n{'Method':<25} {'F1-Score':>12} {'Note':>30}")
        logger.info("-"*80)

        for method, data in sorted(comparison.items(), key=lambda x: x[1]['f1_score'], reverse=True):
            f1 = data['f1_score']
            improvement = data.get('improvement', '-')
            logger.info(f"{method:<25} {f1:>12.4f} {improvement:>30}")

        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DynoTEARS detection on Telemanom dataset"
    )
    parser.add_argument(
        "--results",
        default="./results/telemanom",
        help="Directory containing detection results for each anomaly"
    )
    parser.add_argument(
        "--ground_truth",
        default="./data/Anomaly/telemanom/isolated_anomaly_index.csv",
        help="Path to ground truth CSV"
    )
    parser.add_argument(
        "--output",
        default="./results/telemanom_evaluation.json",
        help="Output file for evaluation report"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size used in detection"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride used in rolling window"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = TelemanamEvaluator(args.ground_truth)

    # Evaluate all files
    logger.info(f"Evaluating results in {args.results}...")
    results = evaluator.evaluate_all(
        Path(args.results),
        window_size=args.window_size,
        stride=args.stride
    )

    # Compute metrics
    metrics = evaluator.compute_metrics(results)

    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = evaluator.generate_report(results, metrics, output_path)

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
