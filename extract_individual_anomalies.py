#!/usr/bin/env python3
"""
Extract individual anomaly sequences from Telemanom test dataset.
Creates isolated anomaly files (Golden + Single Anomaly) for causal discovery.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path

def parse_anomaly_sequences(seq_str):
    """Parse anomaly sequence string from CSV."""
    try:
        return ast.literal_eval(seq_str)
    except:
        return []

def parse_classes(class_str):
    """Parse anomaly class string from CSV."""
    try:
        # The CSV has unquoted strings like [point] or [contextual, contextual]
        # We need to convert to proper Python format ['point'] or ['contextual', 'contextual']
        
        # Remove outer brackets
        inner = class_str.strip('[]')
        
        # Split by comma and clean each item
        items = [item.strip().strip('"').strip("'") for item in inner.split(',')]
        
        return items
    except Exception as e:
        print(f"Warning: Could not parse class string '{class_str}': {e}")
        return ['unknown']

def extract_individual_anomalies(
    merged_test_csv: str,
    golden_csv: str,
    labeled_anomalies_csv: str,
    output_dir: str
):
    """
    Extract individual anomalies from merged test dataset.
    
    For each channel with anomalies:
    1. Load the channel's test data
    2. For each anomaly sequence:
       - Extract Golden period (before anomaly)
       - Extract Anomaly period (anomaly only)
       - Concatenate: [Golden | Anomaly]
       - Save as: anomaly_{channel}_{seq_idx}.csv
    
    Args:
        merged_test_csv: Path to merged test dataset
        golden_csv: Path to Golden period dataset
        labeled_anomalies_csv: Path to labeled_anomalies.csv
        output_dir: Directory to save individual anomaly files
    """
    
    print("="*80)
    print("EXTRACTING INDIVIDUAL ANOMALIES")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load labeled anomalies
    print(f"\n[1/5] Loading labeled anomalies from: {labeled_anomalies_csv}")
    labels_df = pd.read_csv(labeled_anomalies_csv)
    print(f"  ✓ Found {len(labels_df)} channels with anomalies")
    
    # Load merged test dataset
    print(f"\n[2/5] Loading merged test dataset from: {merged_test_csv}")
    test_df = pd.read_csv(merged_test_csv)
    print(f"  ✓ Loaded: {test_df.shape[0]} timesteps × {test_df.shape[1]} channels")
    
    # Load Golden dataset
    print(f"\n[3/5] Loading Golden period dataset from: {golden_csv}")
    golden_df = pd.read_csv(golden_csv)
    print(f"  ✓ Loaded: {golden_df.shape[0]} timesteps × {golden_df.shape[1]} channels")
    golden_length = len(golden_df)
    
    # Process each channel
    print(f"\n[4/5] Extracting individual anomalies...")
    print("-"*80)
    
    total_anomalies = 0
    extracted_files = []
    
    for idx, row in labels_df.iterrows():
        chan_id = row['chan_id']
        spacecraft = row['spacecraft']
        sequences = parse_anomaly_sequences(row['anomaly_sequences'])
        classes = parse_classes(row['class'])
        
        if not sequences:
            continue
        
        print(f"\n{chan_id} ({spacecraft}): {len(sequences)} anomaly sequence(s)")
        
        # Get all columns for this channel
        channel_cols = [col for col in test_df.columns if col.startswith(f"{chan_id}_")]
        
        if not channel_cols:
            print(f"  ⚠ Warning: No columns found for {chan_id}")
            continue
        
        print(f"  Columns: {len(channel_cols)} features")
        
        # Process each anomaly sequence
        for seq_idx, (start, end) in enumerate(sequences):
            # Get anomaly class, handle list wrapping
            if seq_idx < len(classes):
                cls = classes[seq_idx]
                # If classes is a list, extract the element
                anomaly_class = cls if isinstance(cls, str) else str(cls)
            else:
                anomaly_class = 'unknown'
            
            # Extract anomaly window from test dataset
            anomaly_data = test_df.iloc[start:end+1][channel_cols].copy()
            
            # Get Golden data for these columns
            golden_data = golden_df[channel_cols].copy()
            
            # Concatenate: [Golden | Anomaly]
            combined = pd.concat([golden_data, anomaly_data], axis=0, ignore_index=True)
            
            # Create filename
            filename = f"anomaly_{chan_id}_seq{seq_idx+1}_{anomaly_class}.csv"
            filepath = output_path / filename
            
            # Save
            combined.to_csv(filepath, index=False)
            
            print(f"  ✓ Seq {seq_idx+1} [{start}:{end}] ({anomaly_class})")
            print(f"    → {filename}")
            print(f"    → Shape: {combined.shape[0]} timesteps × {combined.shape[1]} features")
            print(f"    → Golden: {len(golden_data)}, Anomaly: {len(anomaly_data)}")
            
            extracted_files.append({
                'filename': filename,
                'channel': chan_id,
                'spacecraft': spacecraft,
                'sequence_idx': seq_idx + 1,
                'anomaly_class': anomaly_class,
                'anomaly_start': start,
                'anomaly_end': end,
                'anomaly_length': end - start + 1,
                'golden_length': len(golden_data),
                'total_length': len(combined),
                'num_features': len(channel_cols)
            })
            
            total_anomalies += 1
    
    # Create catalog
    print(f"\n[5/5] Creating anomaly catalog...")
    catalog_df = pd.DataFrame(extracted_files)
    catalog_path = output_path / "anomaly_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total anomalies extracted: {total_anomalies}")
    print(f"Output directory: {output_dir}")
    print(f"Catalog saved: {catalog_path}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Anomaly classes:")
    for cls, count in catalog_df['anomaly_class'].value_counts().items():
        print(f"  {cls}: {count}")
    print(f"\nSpacecraft:")
    for sc, count in catalog_df['spacecraft'].value_counts().items():
        print(f"  {sc}: {count}")
    print(f"\nAnomaly length statistics:")
    print(f"  Min: {catalog_df['anomaly_length'].min()}")
    print(f"  Max: {catalog_df['anomaly_length'].max()}")
    print(f"  Mean: {catalog_df['anomaly_length'].mean():.1f}")
    print(f"  Median: {catalog_df['anomaly_length'].median():.1f}")
    
    return catalog_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract individual anomalies from Telemanom dataset")
    parser.add_argument(
        "--merged-test",
        default="telemanom/test_dataset_merged_clean.csv",
        help="Path to merged test dataset"
    )
    parser.add_argument(
        "--golden",
        default="telemanom/golden_period_dataset_clean.csv",
        help="Path to Golden period dataset"
    )
    parser.add_argument(
        "--labels",
        default="telemanom/labeled_anomalies.csv",
        help="Path to labeled_anomalies.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="telemanom/individual_anomalies",
        help="Output directory for extracted anomalies"
    )
    
    args = parser.parse_args()
    
    # Extract
    catalog = extract_individual_anomalies(
        merged_test_csv=args.merged_test,
        golden_csv=args.golden,
        labeled_anomalies_csv=args.labels,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Done! Check {args.output_dir}/ for extracted files")
