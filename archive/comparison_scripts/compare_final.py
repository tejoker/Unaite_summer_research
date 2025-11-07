#!/usr/bin/env python3
"""Compare datasets using reset index"""
import pandas as pd
import numpy as np
import sys

golden = pd.read_csv(sys.argv[1], index_col=0).reset_index(drop=True)
spike = pd.read_csv(sys.argv[2], index_col=0).reset_index(drop=True)

print("="*80)
print("DATASET COMPARISON (using row positions 0-999)")
print("="*80)
print(f"Shape: {golden.shape}")

epsilon = 1e-10
tolerance = 1e-6
diff_values = spike - golden
diff_mask = np.abs(diff_values) > tolerance
rows_with_diff = diff_mask.any(axis=1)
diff_positions = golden.index[rows_with_diff].tolist()

print(f"Rows with differences: {len(diff_positions)}")
print(f"Positions: {diff_positions}")

print("\n" + "="*80)
print("DIFFERENCES")
print("="*80)

for pos in diff_positions[:20]:
    print(f"\nRow {pos}:")
    for col in golden.columns:
        g_val = golden.iloc[pos][col]
        s_val = spike.iloc[pos][col]
        if abs(g_val - s_val) > tolerance:
            print(f"  {col}: Golden={g_val:.4f}, Spike={s_val:.4f}, Diff={s_val-g_val:+.4f}")

print("\n" + "="*80)
print("ROWS 200-204")
print("="*80)
for pos in range(200, min(205, len(golden))):
    print(f"\nRow {pos}:")
    print(f"  Golden: {golden.iloc[pos].values}")
    print(f"  Spike:  {spike.iloc[pos].values}")
    if (np.abs(spike.iloc[pos] - golden.iloc[pos]) > tolerance).any():
        print(f"  DIFF!")

print(f"\n{'='*80}")
print(f"Total rows: {len(golden)}, Differences: {len(diff_positions)} ({100*len(diff_positions)/len(golden):.2f}%)")
