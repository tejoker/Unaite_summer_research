#!/usr/bin/env python3
"""
Chunked nearest-neighbor anomaly detector.
Processes test windows in batches and saves intermediate results.
Safe for remote execution - won't get killed by OOM.
"""

import numpy as np
import polars as pl
from scipy.linalg import norm
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_weight_matrix(df: pl.DataFrame, window_idx: int, lag: int, d: int) -> np.ndarray:
    """Load weight matrix for specific window from Polars DataFrame."""
    df_window = df.filter((pl.col('window_idx') == window_idx) & (pl.col('lag') == lag))
    
    W = np.zeros((d, d))
    i_arr = df_window['i'].to_numpy()
    j_arr = df_window['j'].to_numpy()
    weight_arr = df_window['weight'].to_numpy()
    
    for idx in range(len(i_arr)):
        i, j = int(i_arr[idx]), int(j_arr[idx])
        if i < d and j < d:
            W[i, j] = weight_arr[idx]
    
    return W


def main():
    parser = argparse.ArgumentParser(description='Chunked nearest-neighbor anomaly detection')
    parser.add_argument('--golden', default='results/golden_baseline/weights/weights_enhanced.csv',
                       help='Golden baseline weights CSV')
    parser.add_argument('--test', default='results/test_timeline/weights/weights_enhanced.csv',
                       help='Test timeline weights CSV')
    parser.add_argument('--output', default='results/nn_detection_chunked.csv',
                       help='Output CSV file')
    parser.add_argument('--chunk-size', type=int, default=50,
                       help='Number of test windows to process per chunk (default: 50)')
    parser.add_argument('--sample-rate', type=int, default=10,
                       help='Use every Nth golden window (default: 10)')
    parser.add_argument('--lag', type=int, default=0,
                       help='Which lag to analyze (default: 0)')
    parser.add_argument('--start-chunk', type=int, default=0,
                       help='Start from chunk N (for resuming, default: 0)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("CHUNKED NEAREST-NEIGHBOR ANOMALY DETECTION")
    logger.info("="*80)
    logger.info(f"Golden: {args.golden}")
    logger.info(f"Test: {args.test}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Sample rate: every {args.sample_rate}th golden window")
    logger.info(f"Lag: {args.lag}")
    
    # Load golden baseline
    logger.info(f"\nLoading golden baseline...")
    df_golden = pl.read_csv(args.golden)
    golden_windows = sorted(df_golden['window_idx'].unique().to_list())
    sampled_golden = golden_windows[::args.sample_rate]
    logger.info(f"Golden windows: {len(golden_windows)} (using {len(sampled_golden)} sampled)")
    
    # Get matrix dimension
    df_golden_lag = df_golden.filter(pl.col('lag') == args.lag)
    d = max(int(df_golden_lag['i'].max()), int(df_golden_lag['j'].max())) + 1
    logger.info(f"Matrix dimension: {d}x{d}")
    
    # Load test timeline
    logger.info(f"\nLoading test timeline...")
    df_test = pl.read_csv(args.test)
    test_windows = sorted(df_test['window_idx'].unique().to_list())
    logger.info(f"Test windows: {len(test_windows)}")
    
    # Calculate chunks
    num_chunks = (len(test_windows) + args.chunk_size - 1) // args.chunk_size
    logger.info(f"Processing in {num_chunks} chunks of {args.chunk_size} windows each")
    
    # Check if output exists (for resuming)
    output_path = Path(args.output)
    if output_path.exists() and args.start_chunk == 0:
        logger.info(f"\nWARNING: Output file exists, will be overwritten!")
        logger.info(f"To resume, use --start-chunk N")
    
    # Process chunks
    all_results = []
    
    for chunk_idx in range(args.start_chunk, num_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min(start_idx + args.chunk_size, len(test_windows))
        chunk_windows = test_windows[start_idx:end_idx]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CHUNK {chunk_idx+1}/{num_chunks}: Windows {chunk_windows[0]}-{chunk_windows[-1]} ({len(chunk_windows)} windows)")
        logger.info(f"{'='*80}")
        
        chunk_results = []
        
        for i, test_idx in enumerate(chunk_windows):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(chunk_windows)} windows in chunk...")
            
            # Load test matrix
            W_test = load_weight_matrix(df_test, test_idx, args.lag, d)
            
            # Find nearest golden window
            min_distance = float('inf')
            nearest_golden = None
            
            for golden_idx in sampled_golden:
                W_golden = load_weight_matrix(df_golden, golden_idx, args.lag, d)
                distance = norm(W_test - W_golden, 'fro')
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_golden = golden_idx
            
            chunk_results.append({
                'window_idx': test_idx,
                'nearest_golden': nearest_golden,
                'distance': min_distance
            })
        
        # Save chunk results
        all_results.extend(chunk_results)
        df_chunk = pl.DataFrame(all_results)
        
        # Save incrementally (append mode for resume capability)
        if chunk_idx == args.start_chunk:
            df_chunk.write_csv(output_path)  # Overwrite on first chunk
            logger.info(f"  Saved first chunk to: {output_path}")
        else:
            # Append to existing file
            with open(output_path, 'a') as f:
                df_new = pl.DataFrame(chunk_results)
                df_new.write_csv(f, include_header=False)
            logger.info(f"  Appended chunk to: {output_path}")
        
        logger.info(f"  Chunk {chunk_idx+1} complete! Total windows processed: {len(all_results)}")
    
    # Final statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL CHUNKS COMPLETE!")
    logger.info(f"{'='*80}")
    
    df_results = pl.read_csv(output_path)
    distances = df_results['distance'].to_numpy()
    
    logger.info(f"\nDistance Statistics (all {len(distances)} windows):")
    logger.info(f"  Min: {distances.min():.6e}")
    logger.info(f"  Max: {distances.max():.6e}")
    logger.info(f"  Mean: {distances.mean():.6e}")
    logger.info(f"  Std: {distances.std():.6e}")
    logger.info(f"  Median: {np.median(distances):.6e}")
    
    # Anomaly detection
    threshold = distances.mean() + 3 * distances.std()
    anomalies = df_results.filter(pl.col('distance') > threshold)
    
    logger.info(f"\nAnomaly Detection (mean + 3*std):")
    logger.info(f"  Threshold: {threshold:.6e}")
    logger.info(f"  Anomalous windows: {anomalies.height}")
    
    if anomalies.height > 0:
        logger.info(f"\nTop 20 anomalous windows:")
        top_anomalies = anomalies.sort('distance', descending=True).head(20)
        for row in top_anomalies.iter_rows(named=True):
            logger.info(f"  Window {row['window_idx']}: distance = {row['distance']:.6e} (nearest golden: {row['nearest_golden']})")
    
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
