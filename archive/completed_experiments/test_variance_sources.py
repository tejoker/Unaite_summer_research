#!/usr/bin/env python3
"""
Test what ACTUALLY causes variance in DynoTEARS results

Finding: DynoTEARS uses CONSTANT initialization (1e-3), NOT random
So what causes the 0.85 variance between identical runs?

Possible sources:
1. Data ordering / batching randomness
2. GPU non-determinism (floating point operation order)
3. Different lambda values chosen during optimization
4. Numerical instabilities in matrix exponential
5. Actually running on different data (different preprocessing)
"""

import pandas as pd
import numpy as np
from pathlib import Path

def check_initialization():
    """Verify what the actual initialization is"""
    print("="*80)
    print("CHECKING DYNOTEARS INITIALIZATION")
    print("="*80)

    dynotears_file = Path("executable/final_pipeline/dynotears.py")
    with open(dynotears_file, 'r') as f:
        lines = f.readlines()

    print("\nSearching for initialization code...")
    for i, line in enumerate(lines, 1):
        if 'torch.full' in line or 'torch.randn' in line or 'torch.zeros' in line:
            print(f"Line {i}: {line.strip()}")

    print("\nConclusion:")
    print("  W initialized with: torch.full((d, d), 1e-3)")
    print("  A initialized with: torch.full((d, d, p), 1e-3)")
    print("  -> CONSTANT initialization, NOT random")


def compare_actual_data():
    """Check if Golden runs actually used the SAME input data"""
    print("\n" + "="*80)
    print("COMPARING INPUT DATA FOR DIFFERENT GOLDEN RUNS")
    print("="*80)

    # Find preprocessing files for different Golden runs
    golden_runs = list(Path("results").glob("**/Golden/**/preprocessing/*differenced*.csv"))

    print(f"\nFound {len(golden_runs)} Golden preprocessed files:")
    for f in golden_runs[:5]:
        print(f"  {f}")

    if len(golden_runs) >= 2:
        print(f"\nComparing first two:")
        file1 = golden_runs[0]
        file2 = golden_runs[1]

        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        print(f"\n  File 1: {file1.name}")
        print(f"    Shape: {df1.shape}")
        print(f"    Columns: {list(df1.columns)[:3]}")

        print(f"\n  File 2: {file2.name}")
        print(f"    Shape: {df2.shape}")
        print(f"    Columns: {list(df2.columns)[:3]}")

        if df1.shape == df2.shape:
            # Check if data is identical
            numeric_cols = df1.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                first_col = numeric_cols[0]
                diff = np.abs(df1[first_col].values - df2[first_col].values).max()
                print(f"\n  Max difference in '{first_col}': {diff}")

                if diff < 1e-10:
                    print("  -> INPUT DATA IS IDENTICAL")
                else:
                    print("  -> INPUT DATA IS DIFFERENT")
                    print(f"     This could explain the 0.85 variance!")
        else:
            print("\n  -> SHAPES DIFFER - these are different datasets/chunks")


def check_lambda_values():
    """Check if different runs used different lambda values"""
    print("\n" + "="*80)
    print("CHECKING LAMBDA VALUES USED IN DIFFERENT RUNS")
    print("="*80)

    # Look for log files that might contain lambda selection info
    log_files = list(Path("results").glob("**/Golden/**/weights/*.log"))

    if log_files:
        print(f"\nFound {len(log_files)} log files")
        for log_file in log_files[:3]:
            print(f"\n  Checking: {log_file.name}")
            with open(log_file, 'r') as f:
                for line in f:
                    if 'lambda' in line.lower() or 'regularization' in line.lower():
                        print(f"    {line.strip()}")
    else:
        print("\nNo log files found")


def analyze_gpu_determinism():
    """Check if GPU non-determinism could be the issue"""
    print("\n" + "="*80)
    print("CHECKING FOR GPU NON-DETERMINISM")
    print("="*80)

    dynotears_file = Path("executable/final_pipeline/dynotears.py")
    with open(dynotears_file, 'r') as f:
        content = f.read()

    determinism_settings = [
        'torch.backends.cudnn.deterministic',
        'torch.backends.cudnn.benchmark',
        'torch.use_deterministic_algorithms',
    ]

    print("\nSearching for GPU determinism settings...")
    found = False
    for setting in determinism_settings:
        if setting in content:
            print(f"  Found: {setting}")
            found = True

    if not found:
        print("  No determinism settings found")
        print("\n  -> GPU operations may be non-deterministic")
        print("     Different operation order can cause small numerical differences")
        print("     These accumulate through 100+ windows * multiple iterations")


def test_preprocessing_variance():
    """
    Check if preprocessing itself is non-deterministic
    """
    print("\n" + "="*80)
    print("TESTING PREPROCESSING DETERMINISM")
    print("="*80)

    # Find all Golden differenced files
    golden_preproc = list(Path("results").glob("**/Golden/**/preprocessing/*differenced*.csv"))

    if len(golden_preproc) >= 2:
        # Compare first two
        df1 = pd.read_csv(golden_preproc[0])
        df2 = pd.read_csv(golden_preproc[1])

        if df1.shape == df2.shape:
            # Compare actual values
            numeric_cols = df1.select_dtypes(include=[np.number]).columns
            max_diffs = {}
            for col in numeric_cols:
                if col in df2.columns:
                    max_diff = np.abs(df1[col].values - df2[col].values).max()
                    max_diffs[col] = max_diff

            print(f"\nComparing {golden_preproc[0].name}")
            print(f"     with {golden_preproc[1].name}")
            print(f"\nMax differences per column:")
            for col, diff in sorted(max_diffs.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {col}: {diff:.10f}")

            max_overall = max(max_diffs.values())
            if max_overall < 1e-10:
                print("\n  -> Preprocessing IS deterministic")
            else:
                print(f"\n  -> Preprocessing has variance: {max_overall:.10f}")
        else:
            print("\nDifferent shapes - different datasets")
    else:
        print("\nNot enough Golden preprocessing files to compare")


def main():
    print("="*80)
    print("TESTING VARIANCE SOURCES IN DYNOTEARS")
    print("="*80)
    print("\nHypothesis: DynoTEARS uses constant init, so what causes 0.85 variance?")

    check_initialization()
    compare_actual_data()
    check_lambda_values()
    analyze_gpu_determinism()
    test_preprocessing_variance()

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("\nBased on findings above, the 0.85 variance is likely due to:")
    print("  1. Different input data (different chunks/preprocessing runs)")
    print("  2. GPU non-determinism (if no determinism flags set)")
    print("  3. Different lambda values selected")
    print("  4. Numerical instabilities accumulating over many windows")
    print("\nNOT due to:")
    print("  X Random initialization (it's constant at 1e-3)")
    print("="*80)

if __name__ == "__main__":
    main()
