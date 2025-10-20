#!/usr/bin/env python3
"""
End-to-End Anomaly Detection and Reconstruction Pipeline

This script orchestrates the complete workflow:
1. Load baseline and anomaly weight matrices
2. Detect anomalies using the anomaly detection suite
3. Classify anomaly type
4. Identify root causes
5. Correct anomalous weights
6. Reconstruct corrected time series

Usage:
    python end_to_end_pipeline.py \
        --baseline-weights results/baseline/weights.csv \
        --anomaly-weights results/anomaly/weights.csv \
        --original-data data/anomaly_period.csv \
        --output-dir results/reconstruction/
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import anomaly detection suite
from test.anomaly_detection_suite.anomaly_detection_suite import UnifiedAnomalyDetectionSuite

# Import reconstruction and correction modules
from final_pipeline.weight_corrector import WeightCorrector, load_weights_from_csv, save_corrected_weights
from final_pipeline.reconstruction import TimeSeriesReconstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """
    Complete anomaly detection and reconstruction pipeline.
    """

    def __init__(self, output_dir: str):
        """Initialize pipeline with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.anomaly_suite = UnifiedAnomalyDetectionSuite()
        self.corrector = None  # Will be initialized based on strategy
        self.reconstructor = None

        # Results storage
        self.results = {}

        # Setup logging to file
        log_file = self.output_dir / "pipeline_execution.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

        logger.info(f"Initialized End-to-End Pipeline, output: {self.output_dir}")

    def load_weights(self, baseline_path: str, anomaly_path: str):
        """
        Load baseline and anomaly weights.

        Args:
            baseline_path: Path to baseline weights CSV
            anomaly_path: Path to anomaly weights CSV
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Loading Weight Matrices")
        logger.info("=" * 80)

        logger.info(f"Loading baseline weights from: {baseline_path}")
        baseline_weights = load_weights_from_csv(baseline_path)
        self.W_baseline = baseline_weights['W']
        self.A_baseline_list = baseline_weights['A_list']

        logger.info(f"Loading anomaly weights from: {anomaly_path}")
        anomaly_weights = load_weights_from_csv(anomaly_path)
        self.W_anomaly = anomaly_weights['W']
        self.A_anomaly_list = anomaly_weights['A_list']

        logger.info(f"Matrix dimensions: {self.W_baseline.shape[0]} variables")
        logger.info(f"Lag order: {len(self.A_baseline_list)}")

        self.results['dimensions'] = self.W_baseline.shape[0]
        self.results['lag_order'] = len(self.A_baseline_list)

    def detect_and_classify(self, variable_names: list):
        """
        Run anomaly detection and classification.

        Args:
            variable_names: List of variable/sensor names
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Anomaly Detection and Classification")
        logger.info("=" * 80)

        # Run unified anomaly detection suite
        detection_result = self.anomaly_suite.analyze_single_comparison(
            self.W_baseline,
            self.W_anomaly,
            variable_names
        )

        # Extract results
        is_anomaly = detection_result['phase1_binary_detection']['binary_detection']['is_anomaly']
        ensemble_score = detection_result['phase1_binary_detection']['binary_detection']['ensemble_score']

        logger.info(f"Anomaly Detected: {is_anomaly}")
        logger.info(f"Ensemble Score: {ensemble_score:.4f}")

        if is_anomaly:
            # Classification
            classification = detection_result.get('phase2_classification', {})
            rule_based = classification.get('rule_based_result', {})
            anomaly_type = rule_based.get('prediction', 'unknown')
            confidence = rule_based.get('confidence', 0.0)

            logger.info(f"Anomaly Type: {anomaly_type}")
            logger.info(f"Classification Confidence: {confidence:.2f}%")

            # Root cause
            root_cause = detection_result.get('phase3_root_cause', {})
            edge_attr = root_cause.get('edge_attribution', {})
            top_edges = edge_attr.get('top_edges', [])

            if top_edges:
                logger.info(f"Top {len(top_edges)} Contributing Edges:")
                for idx, edge in enumerate(top_edges[:5], 1):
                    logger.info(f"  {idx}. {variable_names[edge['from']]} -> {variable_names[edge['to']]}: "
                              f"change={edge['change']:.6f}, importance={edge['importance']:.4f}")

            self.results['anomaly_detected'] = True
            self.results['anomaly_type'] = anomaly_type
            self.results['confidence'] = confidence
            self.results['top_edges'] = top_edges
        else:
            logger.info("No anomaly detected - reconstruction not needed")
            self.results['anomaly_detected'] = False

        self.results['detection_result'] = detection_result

        # Save detection results
        detection_file = self.output_dir / "anomaly_detection_results.json"
        with open(detection_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_result = self._convert_to_json_serializable(detection_result)
            json.dump(json_result, f, indent=2)
        logger.info(f"Saved detection results to: {detection_file}")

        return is_anomaly

    def correct_weights(self, strategy: str = 'replace_with_baseline', threshold: float = 0.1):
        """
        Correct anomalous weights.

        Args:
            strategy: Correction strategy
            threshold: Threshold for edge significance
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Weight Correction")
        logger.info("=" * 80)

        if not self.results.get('anomaly_detected', False):
            logger.info("No anomaly detected - skipping correction")
            return

        # Initialize corrector
        self.corrector = WeightCorrector(strategy=strategy)

        # Get anomalous edges from detection
        anomalous_edges = self.results.get('top_edges', None)

        # Correct weights
        self.W_corrected, correction_info = self.corrector.correct_weights(
            self.W_anomaly,
            self.W_baseline,
            anomalous_edges,
            threshold=threshold
        )

        logger.info(f"Corrected {correction_info['num_corrections']} edges")
        logger.info(f"Mean correction magnitude: {correction_info['mean_correction_magnitude']:.6f}")

        self.results['correction_info'] = correction_info

        # Save corrected weights
        corrected_weights_file = self.output_dir / "corrected_weights.csv"
        save_corrected_weights(
            self.W_corrected,
            self.A_baseline_list,  # Use baseline lag matrices
            str(corrected_weights_file),
            correction_info
        )
        logger.info(f"Saved corrected weights to: {corrected_weights_file}")

    def reconstruct_time_series(self, original_data_path: str, variable_names: list):
        """
        Reconstruct time series using corrected weights.

        Args:
            original_data_path: Path to original time series data
            variable_names: List of variable names
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Time Series Reconstruction")
        logger.info("=" * 80)

        if not self.results.get('anomaly_detected', False):
            logger.info("No anomaly detected - skipping reconstruction")
            return

        # Load original data
        logger.info(f"Loading original data from: {original_data_path}")
        original_data = pd.read_csv(original_data_path, index_col=0, parse_dates=True)

        # Validate column names
        if not all(var in original_data.columns for var in variable_names):
            logger.warning("Variable names don't match data columns, using data columns")
            variable_names = list(original_data.columns)

        # Extract initial conditions (first p rows)
        p = len(self.A_baseline_list)
        initial_conditions = original_data[variable_names].iloc[:p].values

        logger.info(f"Using {p} initial time steps for reconstruction")

        # Initialize reconstructor with corrected weights
        self.reconstructor = TimeSeriesReconstructor(
            W=self.W_corrected,
            A_list=self.A_baseline_list,  # Use baseline lag matrices
            var_names=variable_names,
            scaler=None  # Assume data is already scaled
        )

        # Reconstruct
        T = len(original_data)
        logger.info(f"Reconstructing {T} time steps...")

        reconstructed_df = self.reconstructor.reconstruct_deterministic(
            initial_conditions=initial_conditions,
            T=T
        )

        # Save reconstructed series
        reconstructed_file = self.output_dir / "reconstructed_time_series.csv"
        reconstructed_df.to_csv(reconstructed_file)
        logger.info(f"Saved reconstructed series to: {reconstructed_file}")

        # Compute reconstruction metrics
        original_values = original_data[variable_names].values
        reconstructed_values = reconstructed_df[variable_names].values

        # Skip initial p rows (used as initial conditions)
        mse = np.mean((original_values[p:] - reconstructed_values[p:]) ** 2)
        mae = np.mean(np.abs(original_values[p:] - reconstructed_values[p:]))
        rmse = np.sqrt(mse)

        logger.info(f"Reconstruction Metrics:")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")

        self.results['reconstruction_metrics'] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        }

        # Save comparison
        comparison_df = pd.DataFrame({
            **{f"{var}_original": original_data[var] for var in variable_names},
            **{f"{var}_reconstructed": reconstructed_df[var] for var in variable_names}
        }, index=original_data.index)

        comparison_file = self.output_dir / "original_vs_reconstructed.csv"
        comparison_df.to_csv(comparison_file)
        logger.info(f"Saved comparison to: {comparison_file}")

    def generate_summary_report(self):
        """Generate final summary report."""
        logger.info("=" * 80)
        logger.info("STEP 5: Generating Summary Report")
        logger.info("=" * 80)

        report = []
        report.append("=" * 80)
        report.append("END-TO-END PIPELINE EXECUTION SUMMARY")
        report.append("=" * 80)
        report.append(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Output Directory: {self.output_dir}")
        report.append("")

        report.append("1. DATA SUMMARY")
        report.append("-" * 80)
        report.append(f"   Number of Variables: {self.results.get('dimensions', 'N/A')}")
        report.append(f"   Lag Order: {self.results.get('lag_order', 'N/A')}")
        report.append("")

        report.append("2. ANOMALY DETECTION")
        report.append("-" * 80)
        if self.results.get('anomaly_detected'):
            report.append(f"   Anomaly Detected: YES")
            report.append(f"   Anomaly Type: {self.results.get('anomaly_type', 'N/A')}")
            report.append(f"   Confidence: {self.results.get('confidence', 0):.2f}%")
            report.append(f"   Number of Anomalous Edges: {len(self.results.get('top_edges', []))}")
        else:
            report.append(f"   Anomaly Detected: NO")
        report.append("")

        if self.results.get('anomaly_detected'):
            report.append("3. WEIGHT CORRECTION")
            report.append("-" * 80)
            correction_info = self.results.get('correction_info', {})
            report.append(f"   Strategy: {correction_info.get('strategy', 'N/A')}")
            report.append(f"   Edges Corrected: {correction_info.get('num_corrections', 0)}")
            report.append(f"   Mean Correction Magnitude: {correction_info.get('mean_correction_magnitude', 0):.6f}")
            report.append("")

            report.append("4. RECONSTRUCTION METRICS")
            report.append("-" * 80)
            metrics = self.results.get('reconstruction_metrics', {})
            report.append(f"   RMSE: {metrics.get('rmse', 0):.6f}")
            report.append(f"   MAE: {metrics.get('mae', 0):.6f}")
            report.append("")

        report.append("5. OUTPUT FILES")
        report.append("-" * 80)
        report.append(f"   Detection Results: anomaly_detection_results.json")
        if self.results.get('anomaly_detected'):
            report.append(f"   Corrected Weights: corrected_weights.csv")
            report.append(f"   Reconstructed Series: reconstructed_time_series.csv")
            report.append(f"   Comparison: original_vs_reconstructed.csv")
        report.append(f"   Execution Log: pipeline_execution.log")
        report.append("")
        report.append("=" * 80)

        # Print to console
        for line in report:
            logger.info(line)

        # Save to file
        report_file = self.output_dir / "PIPELINE_SUMMARY.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        logger.info(f"Summary report saved to: {report_file}")

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Anomaly Detection and Reconstruction Pipeline"
    )

    # Required arguments
    parser.add_argument('--baseline-weights', required=True,
                       help="Path to baseline weights CSV")
    parser.add_argument('--anomaly-weights', required=True,
                       help="Path to anomaly weights CSV")
    parser.add_argument('--original-data', required=True,
                       help="Path to original time series CSV (for reconstruction)")
    parser.add_argument('--output-dir', required=True,
                       help="Output directory for results")

    # Optional arguments
    parser.add_argument('--variable-names', nargs='+',
                       help="Variable/sensor names (if not in CSV)")
    parser.add_argument('--correction-strategy', default='replace_with_baseline',
                       choices=['replace_with_baseline', 'median', 'interpolate', 'zero', 'soft_correction'],
                       help="Weight correction strategy")
    parser.add_argument('--correction-threshold', type=float, default=0.1,
                       help="Threshold for determining significant edge changes")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EndToEndPipeline(args.output_dir)

    try:
        # Step 1: Load weights
        pipeline.load_weights(args.baseline_weights, args.anomaly_weights)

        # Get variable names
        if args.variable_names:
            variable_names = args.variable_names
        else:
            # Try to infer from data
            df = pd.read_csv(args.original_data, index_col=0, nrows=1)
            variable_names = list(df.columns)

        logger.info(f"Variable names: {variable_names}")

        # Step 2: Detect and classify
        is_anomaly = pipeline.detect_and_classify(variable_names)

        # Step 3: Correct weights (only if anomaly detected)
        if is_anomaly:
            pipeline.correct_weights(
                strategy=args.correction_strategy,
                threshold=args.correction_threshold
            )

            # Step 4: Reconstruct time series
            pipeline.reconstruct_time_series(args.original_data, variable_names)

        # Step 5: Generate summary
        pipeline.generate_summary_report()

        logger.info("Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
