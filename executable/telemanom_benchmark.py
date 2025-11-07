#!/usr/bin/env python3
"""
Telemanom Benchmark: Compare Causal Discovery vs SOTA Methods

Benchmarks the DynoTEARS-based causal discovery approach against SOTA
anomaly detection methods on the NASA Telemanom dataset.

Usage:
    python executable/telemanom_benchmark.py --num_files 10 --output results/telemanom_benchmark/
"""

import sys
import os
import time
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TelemanamBenchmark:
    """Benchmark framework for Telemanom dataset"""

    def __init__(self, telemanom_dir: str, golden_dir: str, output_dir: str):
        self.telemanom_dir = Path(telemanom_dir)
        self.golden_dir = Path(golden_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load ground truth index
        self.index_file = self.telemanom_dir / "isolated_anomaly_index.csv"
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> pd.DataFrame:
        """Load ground truth anomaly labels"""
        if not self.index_file.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_file}")

        df = pd.read_csv(self.index_file)
        logger.info(f"Loaded ground truth for {len(df)} anomaly files")
        return df

    def get_anomaly_files(self, num_files: int = None) -> List[Dict]:
        """Get list of anomaly files with metadata"""
        files = []

        for _, row in self.ground_truth.iterrows():
            file_path = self.telemanom_dir / row['filename']

            if file_path.exists():
                files.append({
                    'anomaly_id': row['anomaly_id'],
                    'filepath': file_path,
                    'channel': row['channel'],
                    'spacecraft': row['spacecraft'],
                    'start_idx': int(row['start_idx']),
                    'end_idx': int(row['end_idx']),
                    'duration': int(row['duration']),
                    'anomaly_class': row['anomaly_class'],
                    'num_features': int(row['num_features']),
                    'dataset_length': int(row['dataset_length'])
                })

                if num_files and len(files) >= num_files:
                    break

        logger.info(f"Found {len(files)} valid anomaly files")
        return files

    def get_golden_baseline(self) -> str:
        """Get golden baseline file path"""
        # Look for golden dataset
        golden_file = self.golden_dir / "golden_period_dataset.csv"
        if not golden_file.exists():
            golden_file = self.golden_dir / "golden_period_dataset_mean_channel.csv"

        if not golden_file.exists():
            raise FileNotFoundError(f"Golden baseline not found in {self.golden_dir}")

        logger.info(f"Using golden baseline: {golden_file}")
        return str(golden_file)

    def run_causal_discovery_method(self, golden_file: str, anomaly_info: Dict) -> Dict:
        """Run DynoTEARS causal discovery method"""
        logger.info(f"Running causal discovery on {anomaly_info['anomaly_id']}...")

        start_time = time.time()

        try:
            # Import launcher
            sys.path.insert(0, str(Path(__file__).parent))
            from launcher import run_pipeline

            # Create output subdirectory
            output_subdir = self.output_dir / "causal_discovery" / anomaly_info['anomaly_id']
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Run pipeline
            results = run_pipeline(
                baseline_file=golden_file,
                test_file=str(anomaly_info['filepath']),
                output_dir=str(output_subdir),
                window_size=100,
                stride=10
            )

            runtime = time.time() - start_time

            # Extract detection results
            detection = self._extract_detection_results(results, anomaly_info)
            detection['runtime'] = runtime
            detection['method'] = 'causal_discovery'

            return detection

        except Exception as e:
            logger.error(f"Causal discovery failed: {e}")
            return {
                'method': 'causal_discovery',
                'error': str(e),
                'runtime': time.time() - start_time
            }

    def run_sota_methods(self, golden_file: str, anomaly_info: Dict, methods: List[str] = None) -> Dict:
        """Run SOTA anomaly detection methods"""
        if methods is None:
            methods = ['isolation_forest', 'one_class_svm', 'lof', 'autoencoder']

        logger.info(f"Running SOTA methods on {anomaly_info['anomaly_id']}...")

        results = {}

        for method in methods:
            start_time = time.time()

            try:
                detection = self._run_single_sota_method(
                    method, golden_file, anomaly_info
                )
                detection['runtime'] = time.time() - start_time
                detection['method'] = method
                results[method] = detection

            except Exception as e:
                logger.error(f"{method} failed: {e}")
                results[method] = {
                    'method': method,
                    'error': str(e),
                    'runtime': time.time() - start_time
                }

        return results

    def _run_single_sota_method(self, method: str, golden_file: str, anomaly_info: Dict) -> Dict:
        """Run a single SOTA method"""
        # Load data
        df_golden = pd.read_csv(golden_file)
        df_anomaly = pd.read_csv(anomaly_info['filepath'])

        # Import SOTA methods
        sys.path.insert(0, str(Path(__file__).parent.parent / "archive" / "old_test_files"))
        from sota_anomaly_methods import SOTAAnomalyDetector

        detector = SOTAAnomalyDetector()

        # Run method
        if method == 'isolation_forest':
            result = detector.isolation_forest(df_golden, df_anomaly)
        elif method == 'one_class_svm':
            result = detector.one_class_svm(df_golden, df_anomaly)
        elif method == 'lof':
            result = detector.local_outlier_factor(df_golden, df_anomaly)
        elif method == 'autoencoder':
            result = detector.autoencoder(df_golden, df_anomaly)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract detection
        return self._extract_sota_detection(result, anomaly_info)

    def _extract_detection_results(self, results: Dict, anomaly_info: Dict) -> Dict:
        """Extract detection results from causal discovery output"""
        # Check if anomaly was detected
        detected = results.get('anomaly_detected', False)

        # Get detected windows
        detected_windows = results.get('anomalous_windows', [])

        # Compute metrics
        true_start = anomaly_info['start_idx']
        true_end = anomaly_info['end_idx']

        metrics = self._compute_metrics(
            detected=detected,
            detected_windows=detected_windows,
            true_start=true_start,
            true_end=true_end,
            dataset_length=anomaly_info['dataset_length']
        )

        return metrics

    def _extract_sota_detection(self, result: Dict, anomaly_info: Dict) -> Dict:
        """Extract detection from SOTA method output"""
        # Simple binary detection
        detected = result.get('is_anomaly', False)

        metrics = {
            'detected': detected,
            'true_positive': detected,  # Simplified
            'false_positive': False,
            'false_negative': not detected,
            'true_negative': False
        }

        return metrics

    def _compute_metrics(self, detected: bool, detected_windows: List[int],
                        true_start: int, true_end: int, dataset_length: int) -> Dict:
        """Compute detection metrics"""
        # True positive: detected and overlaps with ground truth
        true_positive = False
        if detected and detected_windows:
            for window_idx in detected_windows:
                window_start = window_idx * 10  # Assuming stride=10
                window_end = window_start + 100  # Assuming window_size=100

                # Check overlap
                if window_start <= true_end and window_end >= true_start:
                    true_positive = True
                    break

        # False positive: detected but no overlap
        false_positive = detected and not true_positive

        # False negative: not detected when should be
        false_negative = not detected

        # True negative: not anomaly and not detected (N/A for this dataset)
        true_negative = False

        return {
            'detected': detected,
            'true_positive': true_positive,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'true_negative': true_negative,
            'num_detected_windows': len(detected_windows) if detected_windows else 0
        }

    def compute_aggregate_metrics(self, all_results: List[Dict]) -> Dict:
        """Compute aggregate metrics across all files"""
        # Group by method
        method_results = {}

        for result in all_results:
            method = result.get('method', 'unknown')
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)

        # Compute metrics for each method
        aggregate = {}

        for method, results in method_results.items():
            tp = sum(1 for r in results if r.get('true_positive', False))
            fp = sum(1 for r in results if r.get('false_positive', False))
            fn = sum(1 for r in results if r.get('false_negative', False))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            avg_runtime = np.mean([r.get('runtime', 0) for r in results])

            aggregate[method] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'avg_runtime': avg_runtime,
                'num_files': len(results)
            }

        return aggregate

    def run_benchmark(self, num_files: int = 10, methods: List[str] = None) -> Dict:
        """Run full benchmark"""
        logger.info("=" * 80)
        logger.info("TELEMANOM BENCHMARK: Causal Discovery vs SOTA Methods")
        logger.info("=" * 80)

        # Get files
        anomaly_files = self.get_anomaly_files(num_files)
        golden_file = self.get_golden_baseline()

        # Results storage
        all_results = []

        # Run benchmarks
        for i, anomaly_info in enumerate(anomaly_files):
            logger.info(f"\n[{i+1}/{len(anomaly_files)}] Processing {anomaly_info['anomaly_id']}...")
            logger.info(f"  Channel: {anomaly_info['channel']}")
            logger.info(f"  Spacecraft: {anomaly_info['spacecraft']}")
            logger.info(f"  Anomaly: [{anomaly_info['start_idx']}, {anomaly_info['end_idx']}]")
            logger.info(f"  Class: {anomaly_info['anomaly_class']}")

            # Run causal discovery
            causal_result = self.run_causal_discovery_method(golden_file, anomaly_info)
            causal_result['anomaly_info'] = anomaly_info
            all_results.append(causal_result)

            # Run SOTA methods
            sota_results = self.run_sota_methods(golden_file, anomaly_info, methods)
            for method, result in sota_results.items():
                result['anomaly_info'] = anomaly_info
                all_results.append(result)

        # Compute aggregate metrics
        aggregate_metrics = self.compute_aggregate_metrics(all_results)

        # Save results
        output_file = self.output_dir / "benchmark_results.json"
        results_data = {
            'aggregate_metrics': aggregate_metrics,
            'individual_results': all_results,
            'num_files': len(anomaly_files),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"\nResults saved to {output_file}")

        # Print summary
        self._print_summary(aggregate_metrics)

        return results_data

    def _print_summary(self, aggregate_metrics: Dict):
        """Print benchmark summary"""
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)

        # Sort by F1 score
        sorted_methods = sorted(
            aggregate_metrics.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )

        logger.info(f"\n{'Method':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Runtime':>10}")
        logger.info("-" * 80)

        for method, metrics in sorted_methods:
            logger.info(
                f"{method:<25} "
                f"{metrics['precision']:>10.3f} "
                f"{metrics['recall']:>10.3f} "
                f"{metrics['f1_score']:>10.3f} "
                f"{metrics['avg_runtime']:>10.2f}s"
            )

        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DynoTEARS causal discovery vs SOTA methods on Telemanom"
    )
    parser.add_argument(
        "--telemanom_dir",
        default="./data/Anomaly/telemanom",
        help="Directory containing Telemanom anomaly files"
    )
    parser.add_argument(
        "--golden_dir",
        default="./data/Golden",
        help="Directory containing golden baseline data"
    )
    parser.add_argument(
        "--output",
        default="./results/telemanom_benchmark",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=10,
        help="Number of anomaly files to process (default: 10)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=['isolation_forest', 'one_class_svm', 'lof'],
        help="SOTA methods to compare against"
    )

    args = parser.parse_args()

    # Create benchmark
    benchmark = TelemanamBenchmark(
        telemanom_dir=args.telemanom_dir,
        golden_dir=args.golden_dir,
        output_dir=args.output
    )

    # Run benchmark
    results = benchmark.run_benchmark(
        num_files=args.num_files,
        methods=args.methods
    )

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
