#!/usr/bin/env python3
"""
Telemanom Single File Pipeline

Demonstrates the proper workflow for processing one Telemanom isolated anomaly file:
1. Split into pre-anomaly (golden) and full dataset
2. Learn golden weights from pre-anomaly portion
3. Run rolling window detection on full dataset
4. Evaluate detection against ground truth

This shows your "split" approach is actually valid - you split internally within each file!

Usage:
    python executable/telemanom_single_file_pipeline.py \
        --anomaly_file data/Anomaly/telemanom/isolated_anomaly_001_P-1_seq1.csv \
        --anomaly_id isolated_anomaly_001_P-1_seq1 \
        --output results/telemanom/isolated_anomaly_001_P-1_seq1
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TelemanamPipeline:
    """Pipeline for processing a single Telemanom file"""

    def __init__(self, ground_truth_file: str):
        self.ground_truth = pd.read_csv(ground_truth_file)

    def get_anomaly_info(self, anomaly_id: str) -> dict:
        """Get ground truth information for an anomaly"""
        row = self.ground_truth[self.ground_truth['anomaly_id'] == anomaly_id]

        if len(row) == 0:
            raise ValueError(f"Anomaly ID not found: {anomaly_id}")

        row = row.iloc[0]
        return {
            'anomaly_id': anomaly_id,
            'start_idx': int(row['start_idx']),
            'end_idx': int(row['end_idx']),
            'duration': int(row['duration']),
            'channel': row['channel'],
            'spacecraft': row['spacecraft'],
            'anomaly_class': row['anomaly_class'],
            'num_features': int(row['num_features']),
            'dataset_length': int(row['dataset_length'])
        }

    def split_golden_and_test(self,
                               df: pd.DataFrame,
                               anomaly_start: int,
                               min_golden_length: int = 2000) -> tuple:
        """
        Split data into golden (pre-anomaly) and test (full dataset).

        Args:
            df: Full dataset
            anomaly_start: Index where anomaly begins
            min_golden_length: Minimum samples needed for golden baseline

        Returns:
            Tuple of (golden_df, test_df)
        """
        # Use pre-anomaly portion as golden baseline
        golden_df = df.iloc[:anomaly_start].copy()

        # Test is the full dataset (for rolling window detection)
        test_df = df.copy()

        if len(golden_df) < min_golden_length:
            logger.warning(
                f"Golden period is short ({len(golden_df)} samples). "
                f"Consider using first {min_golden_length} samples instead."
            )
            # Alternative: use first N samples regardless of anomaly position
            # golden_df = df.iloc[:min_golden_length].copy()

        logger.info(f"Split: Golden = {len(golden_df)} samples, Test = {len(test_df)} samples")
        logger.info(f"Ground truth anomaly at indices [{anomaly_start}, ...] in test data")

        return golden_df, test_df

    def save_split_data(self, golden_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
        """Save split data for pipeline consumption"""
        output_dir.mkdir(parents=True, exist_ok=True)

        golden_file = output_dir / "golden_baseline.csv"
        test_file = output_dir / "test_data.csv"

        golden_df.to_csv(golden_file, index=False)
        test_df.to_csv(test_file, index=False)

        logger.info(f"Saved golden baseline to {golden_file}")
        logger.info(f"Saved test data to {test_file}")

        return str(golden_file), str(test_file)

    def run_pipeline_on_file(self,
                             anomaly_file: str,
                             anomaly_id: str,
                             output_dir: str,
                             window_size: int = 100,
                             stride: int = 10):
        """
        Run complete pipeline on a single Telemanom file.

        This demonstrates the proper workflow:
        1. Load data
        2. Split into golden (pre-anomaly) and test (full)
        3. Save for pipeline consumption
        4. Run your existing launcher on these files

        Then use telemanom_evaluator.py to evaluate results!
        """
        logger.info("="*80)
        logger.info(f"PROCESSING: {anomaly_id}")
        logger.info("="*80)

        # Get ground truth
        anomaly_info = self.get_anomaly_info(anomaly_id)
        logger.info(f"\nGround Truth:")
        logger.info(f"  Anomaly: [{anomaly_info['start_idx']}, {anomaly_info['end_idx']}]")
        logger.info(f"  Duration: {anomaly_info['duration']} samples")
        logger.info(f"  Class: {anomaly_info['anomaly_class']}")
        logger.info(f"  Channel: {anomaly_info['channel']}")

        # Load data
        logger.info(f"\nLoading data from {anomaly_file}...")
        df = pd.read_csv(anomaly_file)
        logger.info(f"Loaded: {len(df)} timesteps, {len(df.columns)} features")

        # Split golden and test
        logger.info(f"\nSplitting data at anomaly start (idx={anomaly_info['start_idx']})...")
        golden_df, test_df = self.split_golden_and_test(
            df,
            anomaly_info['start_idx']
        )

        # Create output structure
        output_path = Path(output_dir)
        split_dir = output_path / "split_data"

        # Save split data
        golden_file, test_file = self.save_split_data(golden_df, test_df, split_dir)

        # Create pipeline commands
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS: Run Your Existing Pipeline")
        logger.info("="*80)

        logger.info(f"\n1. Run preprocessing and DynoTEARS:")
        logger.info(f"\n   python executable/launcher.py \\")
        logger.info(f"       --baseline_file {golden_file} \\")
        logger.info(f"       --test_file {test_file} \\")
        logger.info(f"       --output_dir {output_dir} \\")
        logger.info(f"       --window_size {window_size} \\")
        logger.info(f"       --stride {stride}")

        logger.info(f"\n2. After processing all files, evaluate:")
        logger.info(f"\n   python executable/telemanom_evaluator.py \\")
        logger.info(f"       --results results/telemanom \\")
        logger.info(f"       --output results/telemanom_evaluation.json")

        logger.info("\n" + "="*80)

        return {
            'anomaly_info': anomaly_info,
            'golden_file': golden_file,
            'test_file': test_file,
            'output_dir': output_dir
        }


def main():
    parser = argparse.ArgumentParser(
        description="Process a single Telemanom anomaly file"
    )
    parser.add_argument(
        "--anomaly_file",
        required=True,
        help="Path to anomaly CSV file"
    )
    parser.add_argument(
        "--anomaly_id",
        required=True,
        help="Anomaly ID (e.g., isolated_anomaly_001_P-1_seq1)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for this anomaly"
    )
    parser.add_argument(
        "--ground_truth",
        default="./data/Anomaly/telemanom/isolated_anomaly_index.csv",
        help="Path to ground truth CSV"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size for rolling window detection"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for rolling window"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = TelemanamPipeline(args.ground_truth)

    # Process file
    result = pipeline.run_pipeline_on_file(
        anomaly_file=args.anomaly_file,
        anomaly_id=args.anomaly_id,
        output_dir=args.output,
        window_size=args.window_size,
        stride=args.stride
    )

    logger.info("\nFile preparation complete!")
    logger.info("Run the command above to process with DynoTEARS.")


if __name__ == "__main__":
    main()
