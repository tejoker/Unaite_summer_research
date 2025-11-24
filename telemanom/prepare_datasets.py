#!/usr/bin/env python3
"""
Prepare Telemanom datasets for Tucker-CAM pipeline.
Applies NaN handling according to todolist.md guidelines.
"""

import pandas as pd
import sys
from pathlib import Path

def prepare_dataset(input_csv, output_csv):
    """
    Apply causality-preserving NaN handling:
    1. Forward fill (preserves temporal order)
    2. Drop remaining NaN (leading values that couldn't be filled)
    """
    print(f"\nProcessing: {input_csv}")
    print("=" * 80)

    # Load data
    df = pd.read_csv(input_csv, index_col=0)
    print(f"Original shape: {df.shape}")
    print(f"Original NaN count: {df.isna().sum().sum()}")

    # Check NaN distribution
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"\nColumns with NaN: {len(nan_cols)}")
        print(f"  Examples: {nan_cols[:5]}")

    # Step 1: Forward fill (causality-preserving)
    df_filled = df.ffill()
    remaining_nan = df_filled.isna().sum().sum()
    print(f"\nAfter ffill:")
    print(f"  Remaining NaN: {remaining_nan}")

    # Step 2: Drop rows with remaining NaN (leading values)
    df_clean = df_filled.dropna()
    rows_dropped = len(df_filled) - len(df_clean)
    print(f"\nAfter dropna:")
    print(f"  Rows dropped: {rows_dropped}")
    print(f"  Final shape: {df_clean.shape}")
    print(f"  Final NaN count: {df_clean.isna().sum().sum()}")

    # Verify no NaN remain
    assert df_clean.isna().sum().sum() == 0, "NaN still present after cleaning!"

    # Save
    df_clean.to_csv(output_csv)
    print(f"\n✓ Saved: {output_csv}")
    print("=" * 80)

    return df_clean

def main():
    """Prepare both golden and test datasets."""

    # Paths
    data_dir = Path(__file__).parent

    datasets = [
        {
            'name': 'Golden (normal operation)',
            'input': data_dir / 'golden_period_dataset.csv',
            'output': data_dir / 'golden_period_dataset_clean.csv'
        },
        {
            'name': 'Test (with anomalies)',
            'input': data_dir / 'test_dataset_merged.csv',
            'output': data_dir / 'test_dataset_merged_clean.csv'
        }
    ]

    print("\n" + "=" * 80)
    print("TELEMANOM DATASET PREPARATION")
    print("=" * 80)
    print("\nApplying NaN handling (todolist.md guidelines):")
    print("  1. Forward fill (causality-preserving)")
    print("  2. Drop leading NaN (cannot be filled)")
    print("")

    results = {}

    for dataset in datasets:
        print(f"\n[{dataset['name']}]")

        if not dataset['input'].exists():
            print(f"✗ ERROR: Input file not found: {dataset['input']}")
            print(f"  Please ensure dataset exists before running this script.")
            continue

        try:
            df_clean = prepare_dataset(dataset['input'], dataset['output'])
            results[dataset['name']] = {
                'shape': df_clean.shape,
                'output': dataset['output']
            }
        except Exception as e:
            print(f"✗ ERROR processing {dataset['name']}: {e}")
            sys.exit(1)

    # Summary
    print("\n" + "=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)

    for name, info in results.items():
        print(f"\n{name}:")
        print(f"  Shape: {info['shape'][0]} timesteps × {info['shape'][1]} variables")
        print(f"  File:  {info['output']}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Run Tucker-CAM on golden_period_dataset_clean.csv")
    print("  2. Run Tucker-CAM on test_dataset_merged_clean.csv")
    print("  3. Apply dual-metric anomaly detection (see todolist.md)")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
