#!/usr/bin/env python3
"""Compare datasets using ratio (Spike/Golden) to find differences"""
import pandas as pd
import numpy as np
import sys

def to_scalar(value):
    """Return a float scalar from a potentially vector-like value."""
    if hasattr(value, "iloc") and not np.isscalar(value):
        try:
            return float(value.iloc[0])
        except Exception:
            pass
    if isinstance(value, (list, tuple, np.ndarray)) and not np.isscalar(value):
        try:
            return float(np.asarray(value).flatten()[0])
        except Exception:
            pass
    return float(value)

golden = pd.read_csv(sys.argv[1], index_col=0)
spike = pd.read_csv(sys.argv[2], index_col=0)

print("="*80)
print("DATASET COMPARISON USING RATIO METHOD")
print("="*80)
print(f"Golden: {golden.shape}")
print(f"Spike:  {spike.shape}")

# Calculate ratio: Spike / Golden
# Add small epsilon to avoid division by zero
epsilon = 1e-10
ratio = spike / (golden + epsilon)

# Find where ratio != 1.0 (with tolerance for floating point)
tolerance = 1e-6
diff_mask = np.abs(ratio - 1.0) > tolerance

# Get rows with any difference
rows_with_diff = diff_mask.any(axis=1)
diff_indices = golden.index[rows_with_diff].tolist()

print(f"\nRows where Spike/Golden != 1.0: {len(diff_indices)}")
if diff_indices:
    print(f"Row indices: {diff_indices}")

print("\n" + "="*80)
print("DETAILED DIFFERENCES")
print("="*80)

# Show first few differences
for idx in diff_indices[:10]:
    print(f"\nIndex: {idx}")

    for col in golden.columns:
        g_val_raw = golden.loc[idx, col]
        s_val_raw = spike.loc[idx, col]
        g_val = to_scalar(g_val_raw)
        s_val = to_scalar(s_val_raw)

        if abs(g_val) < epsilon:
            if abs(s_val) > tolerance:
                print(f"  {col}: Golden=0.0, Spike={s_val:.4f} (added)")
        else:
            r = s_val / g_val
            if abs(r - 1.0) > tolerance:
                diff = s_val - g_val
                print(f"  {col}: Golden={g_val:.4f}, Spike={s_val:.4f}, Ratio={r:.4f}, Diff={diff:+.4f}")

if len(diff_indices) > 10:
    print(f"\n... showing first 10 of {len(diff_indices)} rows with differences")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total rows: {len(golden)}")
print(f"Rows with differences: {len(diff_indices)}")
print(f"Percentage: {100*len(diff_indices)/len(golden):.2f}%")

# Statistics on differences
diff_values = spike - golden
print(f"\nDifference statistics:")
print(f"  Max absolute difference: {np.abs(diff_values).max().max():.4f}")
print(f"  Mean absolute difference: {np.abs(diff_values).mean().mean():.6f}")
print(f"  Total non-zero differences: {(np.abs(diff_values) > tolerance).sum().sum()}")
