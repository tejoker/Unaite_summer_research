#!/usr/bin/env python3
"""
Convert window_edges.npy to weights_enhanced.csv format
This bridges the parallel NPY output with the CSV-based anomaly detection
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def convert_npy_to_enhanced_csv(npy_path: Path, output_csv: Path):
    """
    Convert window_edges.npy to weights_enhanced.csv format
    
    NPY format: List of dicts with keys: window_idx, source, target, weight, lag
    CSV format: Aggregated weights per edge across all windows
    """
    print(f"Loading edges from: {npy_path}")
    
    # Load NPY file
    edges = np.load(npy_path, allow_pickle=True)
    
    if edges.ndim == 0:
        edges = edges.item()  # Handle 0-d array wrapping a list
    
    if not isinstance(edges, (list, np.ndarray)) or len(edges) == 0:
        print("Warning: No edges found in NPY file")
        # Create empty CSV with correct columns
        df = pd.DataFrame(columns=['source', 'target', 'weight', 'lag', 'num_windows'])
        df.to_csv(output_csv, index=False)
        print(f"Created empty CSV: {output_csv}")
        return
    
    print(f"Processing {len(edges)} edges...")
    
    # Aggregate edges across windows
    edge_aggregates = defaultdict(lambda: {'weights': [], 'lags': []})
    
    for edge in edges:
        # Create edge key (source, target)
        key = (edge['source'], edge['target'])
        edge_aggregates[key]['weights'].append(edge['weight'])
        edge_aggregates[key]['lags'].append(edge['lag'])
    
    # Build enhanced CSV format
    rows = []
    for (source, target), data in edge_aggregates.items():
        weights = data['weights']
        lags = data['lags']
        
        # Enhanced weight = mean weight across windows
        mean_weight = np.mean(weights)
        # Most common lag
        most_common_lag = max(set(lags), key=lags.count)
        # Number of windows where this edge appears
        num_windows = len(weights)
        
        rows.append({
            'source': source,
            'target': target,
            'weight': mean_weight,
            'lag': most_common_lag,
            'num_windows': num_windows
        })
    
    # Create DataFrame and sort by weight (descending)
    df = pd.DataFrame(rows)
    df = df.sort_values('weight', ascending=False)
    
    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Converted {len(rows)} unique edges to CSV")
    print(f"  Output: {output_csv}")
    print(f"  File size: {output_csv.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Weight range: [{df['weight'].min():.6f}, {df['weight'].max():.6f}]")
    print(f"  Mean windows per edge: {df['num_windows'].mean():.1f}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_npy_to_enhanced_csv.py <input.npy> <output.csv>")
        sys.exit(1)
    
    npy_path = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])
    
    if not npy_path.exists():
        print(f"Error: Input file not found: {npy_path}")
        sys.exit(1)
    
    convert_npy_to_enhanced_csv(npy_path, output_csv)
