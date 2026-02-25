#!/usr/bin/env python3
"""
Generate Bagged Weights Average (Chunked-by-Window Version)

This script aggregates weights from multiple runs by processing them in 
chunks of windows. This avoids:
1. Loading all data into RAM (OOM)
2. Overloading the streaming engine's hash map (TryFromIntError)
"""

import os
import glob
import argparse
import polars as pl
from tqdm import tqdm
import math

def main():
    parser = argparse.ArgumentParser(description='Aggregate bagging runs into mean weights')
    parser.add_argument('--runs-dir', default='results/bagging_experiment/runs', help='Directory containing run subdirectories')
    parser.add_argument('--output', default='results/bagging_experiment/weights_bagged_avg.csv', help='Output CSV file')
    parser.add_argument('--pattern', default='weights/weights_enhanced.csv', help='Relative path to weights file inside each run')
    parser.add_argument('--window-chunk', type=int, default=20, help='Number of windows to process at once')
    args = parser.parse_args()

    # Find all run files
    search_path = os.path.join(args.runs_dir, '*', args.pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        search_path_deep = os.path.join(args.runs_dir, '*', '*', args.pattern)
        files = sorted(glob.glob(search_path_deep))

    # Filter out run_008 as requested
    original_count = len(files)
    files = [f for f in files if "run_008" not in f]
    excluded_count = original_count - len(files)
    
    if not files:
        print("No files to process.")
        return

    num_files = len(files)
    print(f"Found {num_files} runs (Excluded {excluded_count} matching 'run_008').")
    
    # 1. Determine max window index to know how many chunks needed
    print("Scanning for window range...")
    # Just peek at the first file's last rows or scan it quickly
    # Assuming window_idx increases.
    try:
        # Fast way: scan 'window_idx' column of first file
        lf = pl.scan_csv(files[0])
        max_window = lf.select(pl.col("window_idx").max()).collect().item()
        print(f"Max window index found: {max_window}")
    except Exception as e:
        print(f"Error determining max window: {e}")
        # Fallback to a safe upper bound if scan fails, or standard 855
        max_window = 1000 
        print(f"Fallback: assuming max window {max_window}")

    # Prepare output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Remove existing output if present to start fresh
    if os.path.exists(args.output):
        os.remove(args.output)
        
    # 2. Iterate by chunks
    # windows 0..19, 20..39, etc.
    total_chunks = math.ceil((max_window + 1) / args.window_chunk)
    
    schema_overrides = {
        'window_idx': pl.Int32,
        'i': pl.Int32,
        'j': pl.Int32,
        'lag': pl.Int32,
        'weight': pl.Float64
    }

    print(f"Processing in {total_chunks} chunks of {args.window_chunk} windows...")
    
    for chunk_idx in tqdm(range(total_chunks), desc="Aggregating chunks"):
        start_w = chunk_idx * args.window_chunk
        end_w = start_w + args.window_chunk
        
        # Collect this chunk's data from ALL files
        chunk_lfs = []
        for f in files:
            lf = pl.scan_csv(f, schema_overrides=schema_overrides)
            # Filter specifically for this window range
            # This pushes the filter down to the reader!
            lf_chunk = lf.filter(
                (pl.col("window_idx") >= start_w) & 
                (pl.col("window_idx") < end_w)
            )
            chunk_lfs.append(lf_chunk)
            
        # Concat the lazy frames
        # Still lazy!
        combined_lazy = pl.concat(chunk_lfs)
        
        # GroupBy and Agg
        # Since we filtered to a small N windows, the cardinality is manageable.
        # We can materialize (collect) this safely.
        result_df = (
            combined_lazy
            .group_by(["window_idx", "i", "j", "lag"])
            .agg([
                (pl.col("weight").sum() / num_files).alias("weight")
            ])
            .collect() # Executing for this chunk
        )
        
        # Sort for tidiness (optional but good for downstream reading)
        result_df = result_df.sort(["window_idx", "i", "j", "lag"])
        
        # Append to output
        if chunk_idx == 0:
            result_df.write_csv(args.output)
        else:
            with open(args.output, "a") as f:
                result_df.write_csv(f, include_header=False)
                
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
