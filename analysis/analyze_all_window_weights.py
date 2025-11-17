#!/usr/bin/env python3
"""
Analyze weight changes across ALL windows to see the full pattern.
"""

import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_all_window_weights.py <golden_weights.csv> <anomaly_weights.csv>")
        sys.exit(1)
    
    golden_weights_file = sys.argv[1]
    anomaly_weights_file = sys.argv[2]
    
    print("Loading weight files...")
    df_golden = pd.read_csv(golden_weights_file)
    df_anomaly = pd.read_csv(anomaly_weights_file)
    
    # Group by window (use window_idx which is the actual column name)
    windows_golden = df_golden.groupby('window_idx')
    windows_anomaly = df_anomaly.groupby('window_idx')
    
    print("\n" + "="*80)
    print("WEIGHT CHANGE ANALYSIS - ALL WINDOWS")
    print("="*80)
    print(f"{'Window':<8} {'L2 Norm Diff':<15} {'Max Abs Diff':<15} {'Mean Abs Diff':<15} {'Interpretation'}")
    print("-"*80)
    
    all_diffs = []
    
    for window_id in sorted(df_golden['window_idx'].unique()):
        if window_id not in df_anomaly['window_idx'].values:
            continue
        
        g_weights = windows_golden.get_group(window_id)
        a_weights = windows_anomaly.get_group(window_id)
        
        # Ensure same ordering (use parent_name and child_name instead of from/to)
        g_weights = g_weights.sort_values(['lag', 'parent_name', 'child_name'])
        a_weights = a_weights.sort_values(['lag', 'parent_name', 'child_name'])
        
        # Check if shapes match
        if len(g_weights) != len(a_weights):
            print(f"{window_id:<8} {'SKIPPED':<15} {'SKIPPED':<15} {'SKIPPED':<15} ⚠️  MISMATCH (G:{len(g_weights)}, A:{len(a_weights)})")
            continue
        
        # Calculate differences
        weight_diff = (a_weights['weight'].values - g_weights['weight'].values)
        
        l2_norm = np.linalg.norm(weight_diff)
        max_abs_diff = np.abs(weight_diff).max()
        mean_abs_diff = np.abs(weight_diff).mean()
        
        all_diffs.append({
            'window': window_id,
            'l2_norm': l2_norm,
            'max_abs': max_abs_diff,
            'mean_abs': mean_abs_diff
        })
        
        # Interpretation
        if max_abs_diff > 0.001:
            interp = "🔴 SIGNIFICANT"
        elif max_abs_diff > 0.0:
            interp = "🟡 MINOR"
        else:
            interp = "✅ IDENTICAL"
        
        print(f"{window_id:<8} {l2_norm:<15.6f} {max_abs_diff:<15.6f} {mean_abs_diff:<15.6f} {interp}")
    
    # Summary statistics
    df_diffs = pd.DataFrame(all_diffs)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total windows analyzed: {len(df_diffs)}")
    print(f"\nL2 Norm differences:")
    print(f"  Min:    {df_diffs['l2_norm'].min():.6f}")
    print(f"  Max:    {df_diffs['l2_norm'].max():.6f}")
    print(f"  Mean:   {df_diffs['l2_norm'].mean():.6f}")
    print(f"  Median: {df_diffs['l2_norm'].median():.6f}")
    print(f"  90th percentile: {df_diffs['l2_norm'].quantile(0.90):.6f}")
    
    print(f"\nMax absolute differences:")
    print(f"  Min:    {df_diffs['max_abs'].min():.6f}")
    print(f"  Max:    {df_diffs['max_abs'].max():.6f}")
    print(f"  Mean:   {df_diffs['max_abs'].mean():.6f}")
    print(f"  Median: {df_diffs['max_abs'].median():.6f}")
    print(f"  90th percentile: {df_diffs['max_abs'].quantile(0.90):.6f}")
    
    # Find windows with non-zero differences
    non_zero = df_diffs[df_diffs['max_abs'] > 0.0]
    print(f"\n🔍 Windows with ANY weight changes: {len(non_zero)}")
    if len(non_zero) > 0:
        print(f"   Window IDs: {sorted(non_zero['window'].values)}")
    
    # Find windows with significant differences (> 0.001)
    significant = df_diffs[df_diffs['max_abs'] > 0.001]
    print(f"\n🔴 Windows with SIGNIFICANT changes (>0.001): {len(significant)}")
    if len(significant) > 0:
        print(f"   Window IDs: {sorted(significant['window'].values)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
