#!/usr/bin/env python3
"""
Prepare Server Machine Dataset (SMD) for Tucker-CAM Pipeline.

Converts raw text files (comma-separated, no header) into:
1. .npy files (data matrix)
2. _columns.npy files (variable names)

This format allows direct loading by 'launcher.py' and 'preprocessing_no_mi.py',
bypassing partial CSV parsing overhead.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def process_smd_file(input_path, output_dir):
    """
    Process a single SMD text file.
    
    Args:
        input_path: Path to raw .txt file
        output_dir: Directory to save .npy files
    """
    filename = input_path.name
    name_stem = input_path.stem  # e.g., 'machine-1-1'
    
    print(f"Processing {filename}...")
    
    # Read raw data (comma-separated, no header)
    try:
        # SMD is numeric only, so we can just read into numpy or pandas
        # Using pandas to easily handle CSV parsing
        df = pd.read_csv(input_path, header=None)
    except Exception as e:
        print(f"  Error reading {filename}: {e}")
        return False
        
    n_rows, n_cols = df.shape
    print(f"  Shape: {n_rows} rows x {n_cols} vars")
    
    # Create variable names: dim_0, dim_1, ...
    var_names = [f"dim_{i}" for i in range(n_cols)]
    
    # Although pipeline expects 'timestamp' usually in CSV, 
    # for .npy input 'preprocessing_no_mi.py' loads data directly into a DataFrame
    # using the provided columns. It does NOT expect a timestamp column in the data matrix
    # strictly speaking if we provide the columns list.
    # HOWEVER, checks in launcher often look for index-like columns.
    # Let's stick to the PURE data matrix for .npy (n_samples x n_features).
    # The pipeline adds an index if missing or uses 0..N.
    
    # Save Data (.npy)
    output_data_path = output_dir / f"{name_stem}.npy"
    np.save(output_data_path, df.values.astype(np.float32))
    
    # Save Columns (.npy)
    output_cols_path = output_dir / f"{name_stem}_columns.npy"
    np.save(output_cols_path, np.array(var_names))
    
    print(f"  Saved -> {output_data_path}")
    return True

def main():
    # Detect workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    
    smd_root = workspace_root / "ServerMachineDataset"
    output_root = workspace_root / "data" / "SMD"
    
    if not smd_root.exists():
        print(f"Error: ServerMachineDataset folder not found at {smd_root}")
        sys.exit(1)
        
    print(f"Source: {smd_root}")
    print(f"Target: {output_root}")
    print("-" * 60)
    
    # Process Train (Golden) and Test
    splits = ['train', 'test']
    
    count = 0
    for split in splits:
        input_split_dir = smd_root / split
        output_split_dir = output_root / split
        
        if not input_split_dir.exists():
            print(f"Warning: {split} directory not found.")
            continue
            
        # Create output directory
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all .txt files
        files = sorted(list(input_split_dir.glob("*.txt")))
        print(f"\n[{split.upper()}] Found {len(files)} files")
        
        for f in files:
            if process_smd_file(f, output_split_dir):
                count += 1
                
    print("-" * 60)
    print(f"Conversion complete. Processed {count} files.")
    print(f"Data ready in: {output_root}")

if __name__ == "__main__":
    main()
