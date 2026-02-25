#!/usr/bin/env python3
import pandas as pd
import numpy as np
import glob
import os
import argparse
from multiprocessing import Pool

def load_run(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_dir', required=True, help='Directory containing run_XXX folders')
    parser.add_argument('--output', required=True, help='Output averaged CSV path')
    parser.add_argument('--method', default='mean', choices=['mean', 'voting'], help='Aggregation method')
    parser.add_argument('--threshold', type=float, default=0.5, help='Vote threshold (0.0-1.0)')
    args = parser.parse_args()

    # Find all weights files
    pattern = os.path.join(args.runs_dir, "run_*/weights/weights_enhanced.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No weights files found in {args.runs_dir}")
        exit(1)
        
    print(f"Found {len(files)} run files. Aggregating...")
    
    # Check first file for structure
    # Expected: window_idx,i,j,lag,weight
    
    # We can use iterative summation to save memory if needed, 
    # but for 50 runs of ~3MB files, memory is fine (150MB).
    
    # Load all dfs
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Verify columns
        if 'weight' not in df.columns:
            continue
        dfs.append(df)
        
    print(f"Loaded {len(dfs)} valid dataframes.")
    
    # Concatenate
    combined = pd.concat(dfs)
    
    if args.method == 'mean':
        print("Grouping and averaging...")
        result = combined.groupby(['window_idx', 'i', 'j', 'lag'])['weight'].mean().reset_index()
        
    elif args.method == 'voting':
        print(f"Grouping and VOTING (Threshold {args.threshold})...")
        # Binarize weights (assuming any non-zero weight is a vote)
        # Or should we threshold weights first? Usually existence is enough if L1 reg is used.
        combined['vote'] = (combined['weight'].abs() > 0.001).astype(int) # Small epsilon
        
        # Count votes
        counts = combined.groupby(['window_idx', 'i', 'j', 'lag'])['vote'].agg(['sum', 'count']).reset_index()
        
        # Filter
        # Support = sum / count (fraction of runs)
        counts['support'] = counts['sum'] / len(files) # Normalize by total files, not just present rows
        
        # Keep edges with support > threshold
        valid_edges = counts[counts['support'] >= args.threshold]
        
        # For weight, we can use the mean of the valid ones, or just 1.
        # Let's use mean of occurring runs to preserve sign/magnitude info for RCA
        # We need to join back to get weights? 
        # Simpler: Just group by and compute mean, but FILTER by count.
        
        stats = combined.groupby(['window_idx', 'i', 'j', 'lag'])['weight'].agg(['mean', 'count']).reset_index()
        stats = stats[stats['count'] >= (len(files) * args.threshold)]
        
        result = stats.rename(columns={'mean': 'weight'})[['window_idx', 'i', 'j', 'lag', 'weight']]

    print(f"Saving to {args.output}...")
    result.to_csv(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
