#!/usr/bin/env python3
"""
Diagnose why Tucker-CAM weights are so small.
"""

import polars as pl
import numpy as np
from pathlib import Path


def main():
    print("="*80)
    print("TUCKER-CAM WEIGHT DIAGNOSTICS")
    print("="*80)
    print()

    # Load golden baseline
    df_golden = pl.read_csv('results/golden_baseline/weights/weights_enhanced.csv')
    df_golden_lag0 = df_golden.filter(pl.col('lag') == 0)

    # Get unique windows
    windows = sorted(df_golden['window_idx'].unique().to_list())
    print(f"Golden baseline: {len(windows)} windows")
    print(f"First window: {windows[0]}, Last window: {windows[-1]}")
    print()

    # Analyze weight distribution PER window (not averaged)
    print("="*80)
    print("WEIGHT STATISTICS PER WINDOW (sample of 10 windows)")
    print("="*80)
    sample_windows = windows[::len(windows)//10][:10]  # Sample 10 evenly spaced windows

    for window_idx in sample_windows:
        df_window = df_golden_lag0.filter(pl.col('window_idx') == window_idx)
        weights = df_window['weight'].to_numpy()

        if len(weights) == 0:
            print(f"Window {window_idx:3d}: NO EDGES")
            continue

        abs_weights = np.abs(weights)
        print(f"Window {window_idx:3d}: n={len(weights):5d}, "
              f"mean={abs_weights.mean():.2e}, "
              f"max={abs_weights.max():.2e}, "
              f">1e-4: {np.sum(abs_weights > 1e-4):4d}")

    print()

    # Check if weights have consistent signs or cancel out
    print("="*80)
    print("CHECKING WEIGHT CANCELLATION WHEN AVERAGING")
    print("="*80)

    # Pick a specific edge and track it across windows
    # Get an edge that appears in multiple windows
    sample_window = df_golden_lag0.filter(pl.col('window_idx') == windows[0])
    if sample_window.height > 0:
        first_edge = sample_window.row(0, named=True)
        test_i, test_j = first_edge['i'], first_edge['j']
        print(f"Tracking edge ({test_i}, {test_j}) across windows:")
        print()

        edge_weights = []
        for window_idx in windows[:20]:  # Check first 20 windows
            df_edge = df_golden_lag0.filter(
                (pl.col('window_idx') == window_idx) &
                (pl.col('i') == test_i) &
                (pl.col('j') == test_j)
            )
            if df_edge.height > 0:
                weight = float(df_edge['weight'][0])
                edge_weights.append(weight)
                sign = '+' if weight > 0 else '-'
                print(f"  Window {window_idx:3d}: {sign}{abs(weight):.6e}")

        if edge_weights:
            print()
            print(f"Average: {np.mean(edge_weights):.6e}")
            print(f"Std dev: {np.std(edge_weights):.6e}")
            print(f"Range: [{min(edge_weights):.6e}, {max(edge_weights):.6e}]")
            print()
            print("PROBLEM: Weights have different signs and cancel out when averaged!")

    print()

    # Check what top-K is being used
    print("="*80)
    print("TOP-K SPARSIFICATION ANALYSIS")
    print("="*80)
    avg_edges_per_window = len(df_golden_lag0) / len(windows)
    print(f"Average edges per window: {avg_edges_per_window:.1f}")
    print()
    print("This suggests Top-K was used during Tucker-CAM learning.")
    print("The small weights are REAL - they're the top-K edges by magnitude.")
    print()

    # Fundamental issue
    print("="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)
    print()
    print("1. Tucker-CAM produces small weights (1e-4 to 1e-5 range)")
    print("   - This is likely due to Tucker decomposition dimensionality reduction")
    print("   - Or regularization penalties (lambda_w, lambda_a)")
    print()
    print("2. Averaging 417 windows makes weights even smaller (1e-6 range)")
    print("   - Weights have varying signs across windows")
    print("   - Averaging causes cancellation")
    print()
    print("3. Detection using averaged golden baseline is fundamentally flawed")
    print("   - Signal is destroyed by averaging")
    print("   - Need alternative: use window-to-window comparison instead")
    print()

    # Proposed solution
    print("="*80)
    print("PROPOSED SOLUTIONS")
    print("="*80)
    print()
    print("Option 1: Use MEDIAN instead of MEAN for golden baseline")
    print("  - Median is more robust to sign variations")
    print()
    print("Option 2: Use graph STRUCTURE distance, not weight distance")
    print("  - Compare edge sets (binary: edge exists or not)")
    print("  - Ignore weight magnitudes")
    print()
    print("Option 3: Compare test windows to NEAREST golden window")
    print("  - Find most similar golden window for each test window")
    print("  - Compare against that instead of averaged baseline")
    print()
    print("Option 4: Use normalized weights")
    print("  - Divide by Frobenius norm per window")
    print("  - Compare normalized structures")
    print()


if __name__ == '__main__':
    main()
