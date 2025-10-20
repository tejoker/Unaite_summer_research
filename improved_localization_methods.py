#!/usr/bin/env python3
"""
Improved Anomaly Localization Methods

This script implements three alternative localization strategies that avoid
the causal propagation problem discovered in progressive zoom analysis:

1. First-Window Detection: Identify onset using the first statistically significant window
2. Edge-Specific Analysis: Focus on edges originating from anomaly variable
3. Change-Point Detection: Apply CUSUM on edge weight time series

These methods are compared against:
- Regular coarse window (160 samples) - baseline
- Medium zoom (96 samples)
- Fine zoom (64 samples)
- Pinpoint zoom (40 samples)
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GroundTruth:
    """Ground truth for injected anomaly"""
    anomaly_type: str
    variable: str
    start_row: int
    length: int
    end_row: int

    def __repr__(self):
        return f"{self.anomaly_type}: rows {self.start_row}-{self.end_row}"


@dataclass
class LocalizationResult:
    """Result from a localization method"""
    method_name: str
    detected_start: int
    detected_end: int
    detected_width: int
    confidence: float
    metadata: Dict = None

    def evaluate(self, ground_truth: GroundTruth) -> Dict:
        """Evaluate localization accuracy against ground truth"""
        gt_start = ground_truth.start_row
        gt_end = ground_truth.end_row
        gt_width = gt_end - gt_start if ground_truth.length > 0 else 1

        # Check if ground truth is contained in detected range
        contains_gt = (self.detected_start <= gt_start) and (self.detected_end >= gt_end)

        # Calculate error metrics
        start_error = abs(self.detected_start - gt_start)
        end_error = abs(self.detected_end - gt_end)
        width_ratio = self.detected_width / gt_width if gt_width > 0 else float('inf')

        # Jaccard similarity (intersection over union)
        intersection_start = max(self.detected_start, gt_start)
        intersection_end = min(self.detected_end, gt_end)
        intersection = max(0, intersection_end - intersection_start)

        union_start = min(self.detected_start, gt_start)
        union_end = max(self.detected_end, gt_end)
        union = union_end - union_start

        jaccard = intersection / union if union > 0 else 0.0

        return {
            'method': self.method_name,
            'contains_ground_truth': contains_gt,
            'start_error': start_error,
            'end_error': end_error,
            'width_ratio': width_ratio,
            'jaccard_similarity': jaccard,
            'confidence': self.confidence,
            'detected_range': f"{self.detected_start}-{self.detected_end}",
            'ground_truth_range': f"{gt_start}-{gt_end}"
        }


class ImprovedLocalizationMethods:
    """Collection of improved localization methods"""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root

    def load_weights(self, weights_file: Path) -> Dict[int, Dict[str, float]]:
        """Load weights from CSV and organize by window"""
        weights = {}

        if not weights_file.exists():
            logger.error(f"Weights file not found: {weights_file}")
            return None

        try:
            df = pd.read_csv(weights_file)

            for window_idx in sorted(df['window_idx'].unique()):
                window_data = df[df['window_idx'] == window_idx]
                weights[window_idx] = {}

                for _, row in window_data.iterrows():
                    edge = f"{row['parent_name']}→{row['child_name']}"
                    weights[window_idx][edge] = row['weight']

        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            return None

        return weights

    def compare_windows_statistical(self, golden_weights: Dict, anomaly_weights: Dict,
                                   alpha: float = 0.05) -> List[Tuple[int, float, float]]:
        """Compare windows using Wilcoxon test - returns (window_idx, mean_abs_diff, p_value)"""
        changed_windows = []

        for w_idx in sorted(golden_weights.keys()):
            if w_idx not in anomaly_weights:
                continue

            g_weights = golden_weights[w_idx]
            a_weights = anomaly_weights[w_idx]

            differences = []
            abs_differences = []

            for edge, g_val in g_weights.items():
                a_val = a_weights.get(edge, 0.0)
                diff = a_val - g_val
                differences.append(diff)
                abs_differences.append(abs(diff))

            if len(differences) < 3:
                max_diff = max(abs_differences) if abs_differences else 0.0
                if max_diff > 0.01:
                    changed_windows.append((w_idx, max_diff, 0.0))
                continue

            try:
                non_zero_diffs = [d for d in differences if abs(d) > 1e-10]

                if len(non_zero_diffs) < 3:
                    continue

                statistic, p_value = stats.wilcoxon(non_zero_diffs, alternative='two-sided')

                if p_value < alpha:
                    mean_abs_diff = np.mean(abs_differences)
                    changed_windows.append((w_idx, mean_abs_diff, p_value))

            except Exception:
                max_diff = max(abs_differences)
                if max_diff > 0.01:
                    changed_windows.append((w_idx, max_diff, 1.0))

        return changed_windows

    def method1_first_window_detection(self, golden_weights: Dict, anomaly_weights: Dict,
                                      window_size: int, stride: int) -> LocalizationResult:
        """
        Method 1: First Window Detection

        Identify the first window with statistically significant changes.
        This represents the anomaly onset.
        """
        logger.info("Method 1: First Window Detection")

        changed_windows = self.compare_windows_statistical(golden_weights, anomaly_weights)

        if not changed_windows:
            return LocalizationResult(
                method_name="FirstWindow",
                detected_start=0,
                detected_end=0,
                detected_width=0,
                confidence=0.0,
                metadata={'num_changed_windows': 0}
            )

        # Sort by window index
        changed_windows.sort(key=lambda x: x[0])

        # First changed window
        first_window_idx, mean_diff, p_value = changed_windows[0]

        # Convert window index to sample range
        start_sample = first_window_idx * stride
        end_sample = start_sample + window_size

        # Confidence = 1 - p_value (higher confidence = lower p-value)
        confidence = 1.0 - min(p_value, 0.99)

        logger.info(f"  First changed window: {first_window_idx} → samples {start_sample}-{end_sample}")
        logger.info(f"  Confidence: {confidence:.4f} (p={p_value:.4e})")

        return LocalizationResult(
            method_name="FirstWindow",
            detected_start=start_sample,
            detected_end=end_sample,
            detected_width=window_size,
            confidence=confidence,
            metadata={
                'first_window_idx': first_window_idx,
                'p_value': p_value,
                'mean_diff': mean_diff,
                'num_changed_windows': len(changed_windows)
            }
        )

    def method2_edge_specific_analysis(self, golden_weights: Dict, anomaly_weights: Dict,
                                      window_size: int, stride: int,
                                      anomaly_variable: str) -> LocalizationResult:
        """
        Method 2: Edge-Specific Analysis

        Focus only on edges ORIGINATING from the anomaly variable.
        This filters out secondary propagation effects.
        """
        logger.info(f"Method 2: Edge-Specific Analysis (variable: {anomaly_variable})")

        changed_windows = []

        for w_idx in sorted(golden_weights.keys()):
            if w_idx not in anomaly_weights:
                continue

            g_weights = golden_weights[w_idx]
            a_weights = anomaly_weights[w_idx]

            # Filter edges: only those FROM anomaly_variable
            specific_diffs = []

            for edge, g_val in g_weights.items():
                parent, child = edge.split('→')

                # Only analyze edges where parent is the anomaly variable
                if anomaly_variable.lower() in parent.lower():
                    a_val = a_weights.get(edge, 0.0)
                    diff = abs(a_val - g_val)
                    specific_diffs.append(diff)

            if not specific_diffs:
                continue

            # Use maximum difference for edge-specific analysis
            max_specific_diff = max(specific_diffs)
            mean_specific_diff = np.mean(specific_diffs)

            # Threshold: edges from source variable should show larger changes
            if max_specific_diff > 0.015:  # Slightly higher threshold than generic
                changed_windows.append((w_idx, mean_specific_diff, max_specific_diff))

        if not changed_windows:
            return LocalizationResult(
                method_name="EdgeSpecific",
                detected_start=0,
                detected_end=0,
                detected_width=0,
                confidence=0.0,
                metadata={'num_specific_edges_changed': 0}
            )

        # Sort by window index
        changed_windows.sort(key=lambda x: x[0])

        # First window with edge-specific changes
        first_window_idx, mean_diff, max_diff = changed_windows[0]

        start_sample = first_window_idx * stride
        end_sample = start_sample + window_size

        # Confidence based on magnitude of edge-specific change
        confidence = min(max_diff * 10, 1.0)  # Scale to 0-1

        logger.info(f"  First edge-specific change: window {first_window_idx} → samples {start_sample}-{end_sample}")
        logger.info(f"  Max edge diff: {max_diff:.4f}, Confidence: {confidence:.4f}")

        return LocalizationResult(
            method_name="EdgeSpecific",
            detected_start=start_sample,
            detected_end=end_sample,
            detected_width=window_size,
            confidence=confidence,
            metadata={
                'first_window_idx': first_window_idx,
                'max_edge_diff': max_diff,
                'mean_edge_diff': mean_diff,
                'num_changed_windows': len(changed_windows)
            }
        )

    def method3_change_point_detection(self, golden_weights: Dict, anomaly_weights: Dict,
                                      window_size: int, stride: int) -> LocalizationResult:
        """
        Method 3: CUSUM Change Point Detection

        Apply CUSUM on the time series of edge weight differences.
        Detect when cumulative sum exceeds threshold.
        """
        logger.info("Method 3: CUSUM Change Point Detection")

        # Compute mean absolute difference for each window
        window_diffs = []
        window_indices = []

        for w_idx in sorted(golden_weights.keys()):
            if w_idx not in anomaly_weights:
                continue

            g_weights = golden_weights[w_idx]
            a_weights = anomaly_weights[w_idx]

            diffs = []
            for edge, g_val in g_weights.items():
                a_val = a_weights.get(edge, 0.0)
                diffs.append(abs(a_val - g_val))

            mean_diff = np.mean(diffs) if diffs else 0.0
            window_diffs.append(mean_diff)
            window_indices.append(w_idx)

        if len(window_diffs) < 3:
            return LocalizationResult(
                method_name="CUSUM",
                detected_start=0,
                detected_end=0,
                detected_width=0,
                confidence=0.0,
                metadata={'error': 'insufficient_windows'}
            )

        # CUSUM algorithm
        # Reference level: median of differences (expected baseline noise)
        reference = np.median(window_diffs)

        # Threshold: 3x median absolute deviation
        mad = np.median(np.abs(np.array(window_diffs) - reference))
        threshold = reference + 3 * mad

        # Cumulative sum of deviations from reference
        cusum = np.zeros(len(window_diffs))
        for i in range(len(window_diffs)):
            if i == 0:
                cusum[i] = max(0, window_diffs[i] - reference)
            else:
                cusum[i] = max(0, cusum[i-1] + window_diffs[i] - reference)

        # Find first point where CUSUM exceeds threshold
        change_points = np.where(cusum > threshold)[0]

        if len(change_points) == 0:
            return LocalizationResult(
                method_name="CUSUM",
                detected_start=0,
                detected_end=0,
                detected_width=0,
                confidence=0.0,
                metadata={'cusum_max': float(np.max(cusum)), 'threshold': threshold}
            )

        # First change point
        cp_idx = change_points[0]
        window_idx = window_indices[cp_idx]

        start_sample = window_idx * stride
        end_sample = start_sample + window_size

        # Confidence: how much CUSUM exceeds threshold
        cusum_value = cusum[cp_idx]
        confidence = min((cusum_value - threshold) / threshold, 1.0)

        logger.info(f"  CUSUM change point at window {window_idx} → samples {start_sample}-{end_sample}")
        logger.info(f"  CUSUM value: {cusum_value:.4f}, Threshold: {threshold:.4f}")

        return LocalizationResult(
            method_name="CUSUM",
            detected_start=start_sample,
            detected_end=end_sample,
            detected_width=window_size,
            confidence=confidence,
            metadata={
                'change_point_window': window_idx,
                'cusum_value': float(cusum_value),
                'threshold': float(threshold),
                'reference_level': float(reference)
            }
        )


def load_ground_truth(anomaly_type: str, data_dir: Path) -> Optional[GroundTruth]:
    """Load ground truth from anomaly metadata JSON"""
    json_file = data_dir / f"output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"

    if not json_file.exists():
        logger.warning(f"Ground truth file not found: {json_file}")
        return None

    try:
        with open(json_file, 'r') as f:
            metadata = json.load(f)

        start = metadata['start']
        length = metadata['length']
        end = start + length if length > 0 else start

        return GroundTruth(
            anomaly_type=anomaly_type,
            variable=metadata['ts_col'],
            start_row=start,
            length=length,
            end_row=end
        )
    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
        return None


def run_comparison_analysis(results_dir: Path, anomaly_type: str, workspace_root: Path):
    """
    Run comprehensive comparison of all localization methods

    Compares:
    1. First Window Detection (new)
    2. Edge-Specific Analysis (new)
    3. CUSUM Change Point (new)
    4. Regular coarse window (baseline - 160 samples)
    5. Medium zoom (96 samples)
    6. Fine zoom (64 samples)
    7. Pinpoint zoom (40 samples)
    """

    logger.info("="*80)
    logger.info(f"LOCALIZATION COMPARISON: {anomaly_type}")
    logger.info("="*80)

    # Load ground truth
    gt = load_ground_truth(anomaly_type, workspace_root / "data" / "Anomaly")
    if not gt:
        logger.error("Cannot proceed without ground truth")
        return None

    logger.info(f"Ground Truth: {gt}")
    logger.info("")

    # Initialize methods
    methods = ImprovedLocalizationMethods(workspace_root)

    results = {}

    # === Test configurations ===
    configs = [
        {'name': 'coarse', 'window': 160, 'stride': 16, 'dir': results_dir / 'golden'},
        {'name': 'medium', 'window': 96, 'stride': 9, 'dir': results_dir / f'zoom_{anomaly_type}' / 'golden' / 'zoom_medium'},
        {'name': 'fine', 'window': 64, 'stride': 6, 'dir': results_dir / f'zoom_{anomaly_type}' / 'golden' / 'zoom_fine'},
        {'name': 'pinpoint', 'window': 40, 'stride': 2, 'dir': results_dir / f'zoom_{anomaly_type}' / 'golden' / 'zoom_pinpoint'},
    ]

    # Run all methods on each configuration
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config['name'].upper()} (window={config['window']}, stride={config['stride']})")
        logger.info(f"{'='*60}")

        # Load weights
        golden_weights_file = config['dir'] / 'weights' / 'weights_enhanced.csv'
        anomaly_weights_file = golden_weights_file.parent.parent.parent / 'anomaly' / f"zoom_{config['name']}" / 'weights' / 'weights_enhanced.csv'

        if not golden_weights_file.exists():
            logger.warning(f"Weights not found for {config['name']}")
            continue

        if not anomaly_weights_file.exists():
            # Try alternative path for coarse level
            anomaly_weights_file = results_dir / f'anomaly_{anomaly_type}' / 'weights' / 'weights_enhanced.csv'

        if not anomaly_weights_file.exists():
            logger.warning(f"Anomaly weights not found for {config['name']}")
            continue

        golden_weights = methods.load_weights(golden_weights_file)
        anomaly_weights = methods.load_weights(anomaly_weights_file)

        if not golden_weights or not anomaly_weights:
            continue

        config_results = {}

        # Method 1: First Window
        result1 = methods.method1_first_window_detection(
            golden_weights, anomaly_weights, config['window'], config['stride']
        )
        config_results['FirstWindow'] = result1.evaluate(gt)

        # Method 2: Edge-Specific
        result2 = methods.method2_edge_specific_analysis(
            golden_weights, anomaly_weights, config['window'], config['stride'],
            gt.variable
        )
        config_results['EdgeSpecific'] = result2.evaluate(gt)

        # Method 3: CUSUM
        result3 = methods.method3_change_point_detection(
            golden_weights, anomaly_weights, config['window'], config['stride']
        )
        config_results['CUSUM'] = result3.evaluate(gt)

        results[config['name']] = config_results

    return results, gt


def print_comparison_table(all_results: Dict, anomaly_type: str, ground_truth: GroundTruth):
    """Print comprehensive comparison table"""

    print("\n" + "="*100)
    print(f"COMPREHENSIVE LOCALIZATION EVALUATION: {anomaly_type}")
    print("="*100)
    print(f"Ground Truth: rows {ground_truth.start_row}-{ground_truth.end_row} ({ground_truth.end_row - ground_truth.start_row} samples)")
    print("="*100)
    print()

    # Table header
    print(f"{'Config':<12} {'Method':<15} {'Detected Range':<20} {'Width':<8} {'GT?':<5} {'Start Err':<10} {'Jaccard':<10} {'Conf':<8}")
    print("-" * 100)

    # Print results
    for config_name in ['coarse', 'medium', 'fine', 'pinpoint']:
        if config_name not in all_results:
            continue

        config_results = all_results[config_name]

        for method_name in ['FirstWindow', 'EdgeSpecific', 'CUSUM']:
            if method_name not in config_results:
                continue

            res = config_results[method_name]

            gt_check = "✅" if res['contains_ground_truth'] else "❌"

            print(f"{config_name:<12} {method_name:<15} {res['detected_range']:<20} "
                  f"{res['width_ratio']:<8.1f}x {gt_check:<5} {res['start_error']:<10} "
                  f"{res['jaccard_similarity']:<10.3f} {res['confidence']:<8.3f}")

    print("\n" + "="*100)


def main():
    if len(sys.argv) < 2:
        print("Usage: python improved_localization_methods.py <results_base_dir> [anomaly_type]")
        print()
        print("Example:")
        print("  python improved_localization_methods.py results/complete_zoom_20251016_143412 drift")
        print()
        print("If anomaly_type is not specified, runs on all 7 anomaly types")
        sys.exit(1)

    results_base = Path(sys.argv[1])
    workspace_root = Path(__file__).parent

    if len(sys.argv) >= 3:
        anomaly_types = [sys.argv[2]]
    else:
        anomaly_types = ['drift', 'spike', 'variance_burst', 'level_shift',
                        'amplitude_change', 'missing_block', 'trend_change']

    # Run analysis for each anomaly type
    all_evaluations = {}

    for anomaly_type in anomaly_types:
        results, gt = run_comparison_analysis(results_base, anomaly_type, workspace_root)

        if results:
            all_evaluations[anomaly_type] = {'results': results, 'ground_truth': gt}
            print_comparison_table(results, anomaly_type, gt)

    # Save comprehensive results
    output_file = results_base / "localization_method_comparison.json"

    # Convert to JSON-serializable format (handle numpy types)
    def convert_to_json_serializable(obj):
        """Convert numpy/python types to JSON serializable"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    json_results = {}
    for anomaly_type, data in all_evaluations.items():
        json_results[anomaly_type] = {
            'ground_truth': {
                'start': int(data['ground_truth'].start_row),
                'end': int(data['ground_truth'].end_row),
                'length': int(data['ground_truth'].length),
                'variable': data['ground_truth'].variable
            },
            'results': convert_to_json_serializable(data['results'])
        }

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"\n✅ Comprehensive results saved to: {output_file}")

    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY: Success Rate Across All Methods and Configurations")
    print("="*100)

    success_stats = {}
    for anomaly_type, data in all_evaluations.items():
        success_stats[anomaly_type] = {'total': 0, 'success': 0, 'best_jaccard': 0.0, 'best_method': None}

        for config_name, config_results in data['results'].items():
            for method_name, method_result in config_results.items():
                success_stats[anomaly_type]['total'] += 1
                if method_result['contains_ground_truth']:
                    success_stats[anomaly_type]['success'] += 1

                if method_result['jaccard_similarity'] > success_stats[anomaly_type]['best_jaccard']:
                    success_stats[anomaly_type]['best_jaccard'] = method_result['jaccard_similarity']
                    success_stats[anomaly_type]['best_method'] = f"{config_name}/{method_name}"

    print(f"\n{'Anomaly Type':<20} {'Success Rate':<15} {'Best Jaccard':<15} {'Best Method':<30}")
    print("-" * 100)
    for anomaly_type, stats in success_stats.items():
        success_rate = f"{stats['success']}/{stats['total']} ({100*stats['success']/stats['total']:.1f}%)"
        best_method = stats['best_method'] if stats['best_method'] else "None"
        print(f"{anomaly_type:<20} {success_rate:<15} {stats['best_jaccard']:<15.3f} {best_method:<30}")

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
