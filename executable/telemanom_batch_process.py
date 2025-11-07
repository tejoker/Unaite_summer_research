#!/usr/bin/env python3
"""
Telemanom Batch Processing

Process all 105 Telemanom anomaly files in batch.
Each file is split internally (pre-anomaly as golden) and processed independently.

Usage:
    # Process all 105 files
    python executable/telemanom_batch_process.py \
        --num_files 105 \
        --output results/telemanom

    # Process first 10 files (for testing)
    python executable/telemanom_batch_process.py \
        --num_files 10 \
        --output results/telemanom_test

    # Process specific anomaly types
    python executable/telemanom_batch_process.py \
        --anomaly_class point \
        --output results/telemanom_point_only
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import argparse
import pandas as pd
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from telemanom_single_file_pipeline import TelemanamPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TelemanamBatchProcessor:
    """Batch process all Telemanom files"""

    def __init__(self,
                 telemanom_dir: str,
                 ground_truth_file: str,
                 output_dir: str):
        self.telemanom_dir = Path(telemanom_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = TelemanamPipeline(ground_truth_file)
        self.ground_truth = pd.read_csv(ground_truth_file)

    def get_files_to_process(self,
                             num_files: int = None,
                             anomaly_class: str = None,
                             spacecraft: str = None) -> list:
        """
        Get list of files to process with filtering options.

        Args:
            num_files: Maximum number of files to process
            anomaly_class: Filter by anomaly class (point/contextual)
            spacecraft: Filter by spacecraft (SMAP/MSL)

        Returns:
            List of dicts with file info
        """
        df = self.ground_truth.copy()

        # Apply filters
        if anomaly_class:
            df = df[df['anomaly_class'] == anomaly_class]
            logger.info(f"Filtered to {len(df)} {anomaly_class} anomalies")

        if spacecraft:
            df = df[df['spacecraft'] == spacecraft]
            logger.info(f"Filtered to {len(df)} {spacecraft} anomalies")

        # Limit number
        if num_files:
            df = df.head(num_files)

        # Build file list
        files = []
        for _, row in df.iterrows():
            file_path = self.telemanom_dir / row['filename']

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            files.append({
                'anomaly_id': row['anomaly_id'],
                'filepath': str(file_path),
                'anomaly_info': {
                    'start_idx': int(row['start_idx']),
                    'end_idx': int(row['end_idx']),
                    'duration': int(row['duration']),
                    'channel': row['channel'],
                    'spacecraft': row['spacecraft'],
                    'anomaly_class': row['anomaly_class']
                }
            })

        logger.info(f"Selected {len(files)} files to process")
        return files

    def process_single_file(self,
                           file_info: dict,
                           window_size: int = 100,
                           stride: int = 10,
                           run_detection: bool = False) -> dict:
        """
        Process a single file.

        Args:
            file_info: Dict with anomaly_id, filepath, anomaly_info
            window_size: Window size for detection
            stride: Stride for rolling window
            run_detection: If True, run DynoTEARS detection immediately

        Returns:
            Dict with processing status
        """
        anomaly_id = file_info['anomaly_id']
        output_dir = self.output_dir / anomaly_id

        try:
            logger.info(f"Processing {anomaly_id}...")

            # Prepare files (split golden/test)
            result = self.pipeline.run_pipeline_on_file(
                anomaly_file=file_info['filepath'],
                anomaly_id=anomaly_id,
                output_dir=str(output_dir),
                window_size=window_size,
                stride=stride
            )

            # Optionally run detection
            if run_detection:
                logger.info(f"  Running DynoTEARS detection...")
                # Import launcher here to avoid circular imports
                try:
                    from launcher import run_pipeline

                    detection_result = run_pipeline(
                        baseline_file=result['golden_file'],
                        test_file=result['test_file'],
                        output_dir=str(output_dir),
                        window_size=window_size,
                        stride=stride
                    )

                    result['detection_completed'] = True
                    result['detection_result'] = detection_result

                except Exception as e:
                    logger.error(f"  Detection failed: {e}")
                    result['detection_completed'] = False
                    result['detection_error'] = str(e)

            result['status'] = 'success'
            return result

        except Exception as e:
            logger.error(f"Failed to process {anomaly_id}: {e}")
            return {
                'anomaly_id': anomaly_id,
                'status': 'error',
                'error': str(e)
            }

    def process_batch(self,
                      num_files: int = None,
                      window_size: int = 100,
                      stride: int = 10,
                      run_detection: bool = False,
                      parallel: bool = False,
                      max_workers: int = 4,
                      anomaly_class: str = None,
                      spacecraft: str = None) -> list:
        """
        Process multiple files in batch.

        Args:
            num_files: Number of files to process
            window_size: Window size for detection
            stride: Stride for rolling window
            run_detection: If True, run DynoTEARS immediately
            parallel: If True, process files in parallel
            max_workers: Number of parallel workers
            anomaly_class: Filter by anomaly class
            spacecraft: Filter by spacecraft

        Returns:
            List of processing results
        """
        logger.info("="*80)
        logger.info("TELEMANOM BATCH PROCESSING")
        logger.info("="*80)

        # Get files to process
        files = self.get_files_to_process(
            num_files=num_files,
            anomaly_class=anomaly_class,
            spacecraft=spacecraft
        )

        if len(files) == 0:
            logger.error("No files to process!")
            return []

        results = []

        if parallel and not run_detection:
            # Parallel processing (only for file preparation, not detection)
            logger.info(f"Processing {len(files)} files in parallel ({max_workers} workers)...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_single_file,
                        file_info,
                        window_size,
                        stride,
                        run_detection
                    ): file_info for file_info in files
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

        else:
            # Sequential processing
            logger.info(f"Processing {len(files)} files sequentially...")

            for i, file_info in enumerate(files):
                logger.info(f"\n[{i+1}/{len(files)}] Processing {file_info['anomaly_id']}...")
                result = self.process_single_file(
                    file_info,
                    window_size,
                    stride,
                    run_detection
                )
                results.append(result)

        # Save batch summary
        summary_file = self.output_dir / "batch_processing_summary.json"
        summary = {
            'total_files': len(files),
            'successful': len([r for r in results if r.get('status') == 'success']),
            'failed': len([r for r in results if r.get('status') == 'error']),
            'results': results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\nBatch summary saved to {summary_file}")
        self._print_batch_summary(summary)

        return results

    def _print_batch_summary(self, summary: dict):
        """Print batch processing summary"""
        logger.info("\n" + "="*80)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total files:  {summary['total_files']}")
        logger.info(f"Successful:   {summary['successful']}")
        logger.info(f"Failed:       {summary['failed']}")

        if summary['failed'] > 0:
            logger.info("\nFailed files:")
            for r in summary['results']:
                if r.get('status') == 'error':
                    logger.info(f"  - {r['anomaly_id']}: {r['error']}")

        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Batch process Telemanom anomaly files"
    )
    parser.add_argument(
        "--telemanom_dir",
        default="./data/Anomaly/telemanom",
        help="Directory containing Telemanom CSV files"
    )
    parser.add_argument(
        "--ground_truth",
        default="./data/Anomaly/telemanom/isolated_anomaly_index.csv",
        help="Path to ground truth CSV"
    )
    parser.add_argument(
        "--output",
        default="./results/telemanom",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to process (default: all 105)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Window size for detection"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for rolling window"
    )
    parser.add_argument(
        "--run_detection",
        action="store_true",
        help="Run DynoTEARS detection immediately (slower)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process files in parallel (file prep only, not detection)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--anomaly_class",
        choices=['point', 'contextual'],
        help="Filter by anomaly class"
    )
    parser.add_argument(
        "--spacecraft",
        choices=['SMAP', 'MSL'],
        help="Filter by spacecraft"
    )

    args = parser.parse_args()

    # Create processor
    processor = TelemanamBatchProcessor(
        telemanom_dir=args.telemanom_dir,
        ground_truth_file=args.ground_truth,
        output_dir=args.output
    )

    # Process batch
    results = processor.process_batch(
        num_files=args.num_files,
        window_size=args.window_size,
        stride=args.stride,
        run_detection=args.run_detection,
        parallel=args.parallel,
        max_workers=args.max_workers,
        anomaly_class=args.anomaly_class,
        spacecraft=args.spacecraft
    )

    logger.info("\nBatch processing complete!")

    if not args.run_detection:
        logger.info("\nTo run detection on prepared files, use:")
        logger.info("  python executable/launcher.py --baseline_file <golden> --test_file <test> ...")
        logger.info("\nOr re-run with --run_detection flag")


if __name__ == "__main__":
    main()
