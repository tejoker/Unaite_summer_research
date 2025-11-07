#!/usr/bin/env python3
"""
Diagnostic script to test the hypothesis:
"Why do ALL sensors show weight changes from window 0, even when the anomaly is at row 200?"

This script will:
1. Compare raw Golden vs. Anomaly data to confirm the anomaly's true location.
2. Compare the PREPROCESSED (differenced) data to see how the anomaly signal is transformed.
3. Calculate and visualize the magnitude of weight changes for EACH window to find the actual onset of causal changes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

def compare_raw_and_preprocessed_data(golden_csv, anomaly_csv, ground_truth_file):
    """
    Test 1 & 2: Compare raw data to find where it differs, then check how
    differencing transforms that signal.
    """
    print("\n" + "="*80)
    print("STEP 1 & 2: RAW & PREPROCESSED DATA COMPARISON")
    print("="*80)

    df_g = pd.read_csv(golden_csv)
    df_a = pd.read_csv(anomaly_csv)

    # Find the ground truth sensor column from JSON
    with open(ground_truth_file, 'r') as f:
        gt = json.load(f)
    sensor_col = gt['ts_col']
    anomaly_start_row = gt['start']

    print(f"Ground Truth: Anomaly in '{sensor_col}' starting at row {anomaly_start_row}")

    # --- Raw Data Check ---
    raw_diff_mask = (df_g[sensor_col] != df_a[sensor_col])
    raw_diff_indices = raw_diff_mask[raw_diff_mask].index

    if raw_diff_indices.empty:
        print("❌ VERDICT (RAW): Raw data files are identical. Anomaly was not injected correctly.")
        return None

    first_raw_diff, last_raw_diff = raw_diff_indices[0], raw_diff_indices[-1]
    print(f"✅ VERDICT (RAW): Raw data differs ONLY between rows {first_raw_diff} and {last_raw_diff}.")
    if first_raw_diff != anomaly_start_row:
        print(f"  ⚠️ WARNING: First difference at {first_raw_diff} does NOT match ground truth start {anomaly_start_row}.")

    # --- Preprocessed (Differenced) Data Check ---
    g_diff = df_g[sensor_col].diff()
    a_diff = df_a[sensor_col].diff()

    # Compare differenced data, ignoring the first NaN value
    # Use a small tolerance for floating point comparisons
    proc_diff_mask = ~np.isclose(g_diff.iloc[1:], a_diff.iloc[1:])
    # Use the numpy boolean mask to filter the original series' index
    proc_diff_indices = g_diff.iloc[1:][proc_diff_mask].index


    if proc_diff_indices.empty:
        print("❌ VERDICT (PREPROCESSED): Differenced data is identical.")
        return sensor_col

    first_proc_diff, last_proc_diff = proc_diff_indices[0], proc_diff_indices[-1]
    print(f"✅ VERDICT (PREPROCESSED): Differenced data differs between rows {first_proc_diff} and {last_proc_diff}.")
    print("  -> This is the EXPECTED behavior. A spike at row `t` affects the differenced values at `t` and `t+1`.")
    print("  -> This confirms that preprocessing does NOT spread the anomaly signal backwards in time.")

    return sensor_col

def visualize_weight_changes_by_window(golden_weights_file, anomaly_weights_file, anomaly_type, output_dir):
    """
    Test 3: Calculate and plot the magnitude of weight changes per window to find the true onset.
    """
    print("\n" + "="*80)
    print("STEP 3: VISUALIZE WEIGHT CHANGES PER WINDOW")
    print("="*80)

    df_g = pd.read_csv(golden_weights_file)
    df_a = pd.read_csv(anomaly_weights_file)

    merged = pd.merge(
        df_g, df_a,
        on=['window_idx', 'child_name', 'parent_name', 'lag'],
        how='outer', suffixes=('_g', '_a')
    ).fillna(0)

    merged['weight_diff'] = abs(merged['weight_a'] - merged['weight_g'])

    # Group by window and sum the differences (a proxy for the Frobenius norm of the difference matrix)
    window_diffs = merged.groupby('window_idx')['weight_diff'].sum()

    # Find first window with a "significant" change
    # We define "significant" as being above the 90th percentile of all observed window changes.
    threshold = window_diffs.quantile(0.90)
    significant_windows = window_diffs[window_diffs > threshold]

    # Find the first significant window (onset of change)
    first_significant_window = significant_windows.index.min() if not significant_windows.empty else "None"
    # Find the window with the absolute maximum change (likely the anomaly epicenter)
    max_change_window = window_diffs.idxmax() if not window_diffs.empty else "None"

    print(f"Total windows analyzed: {len(window_diffs)}")
    print(f"Significance Threshold (90th percentile of diffs): {threshold:.4f}")
    print(f"Number of significant windows found: {len(significant_windows)}")
    print(f"✅ VERDICT (TEMPORAL ONSET): First significant weight change occurs at WINDOW: {first_significant_window}")
    print(f"✅ VERDICT (MAX IMPACT): Window with the largest weight change is WINDOW: {max_change_window}")

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    window_diffs.plot(kind='bar', ax=ax, color='skyblue', width=0.8, label='Total Weight Change per Window')
    ax.axhline(y=threshold, color='r', linestyle='--', label=f'90th Percentile Threshold ({threshold:.2f})')

    if first_significant_window != "None":
        ax.axvline(x=first_significant_window, color='green', linestyle='-', linewidth=2, label=f'First Significant Change (Window {first_significant_window})')

    if max_change_window != "None":
        ax.axvline(x=max_change_window, color='purple', linestyle=':', linewidth=3, label=f'Maximum Impact (Window {max_change_window})')

    ax.set_title(f'Temporal Onset of Weight Changes ({anomaly_type.upper()})', fontsize=16, weight='bold')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Sum of Absolute Weight Differences')
    ax.legend()
    ax.tick_params(axis='x', rotation=90)

    # Make x-axis labels more readable if there are many windows
    ticks = ax.get_xticks()
    if len(ticks) > 20:
        ax.set_xticks(ticks[::5])

    plt.tight_layout()
    output_path = Path(output_dir) / f"temporal_onset_{anomaly_type}.png"
    plt.savefig(output_path)
    print(f"✅ Plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Diagnose the temporal onset of weight changes.")
    parser.add_argument('--golden-csv', required=True, help="Path to the raw golden dataset CSV.")
    parser.add_argument('--anomaly-csv', required=True, help="Path to the raw anomaly dataset CSV.")
    parser.add_argument('--golden-weights', required=True, help="Path to the golden weights_enhanced.csv file.")
    parser.add_argument('--anomaly-weights', required=True, help="Path to the anomaly weights_enhanced.csv file.")
    parser.add_argument('--ground-truth', required=True, help="Path to the anomaly's ground truth JSON file.")
    parser.add_argument('--anomaly-type', required=True, help="Name of the anomaly (e.g., 'spike') for plotting.")
    parser.add_argument('--output-dir', default='results/diagnostics', help="Directory to save the output plot.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Steps 1 & 2
    compare_raw_and_preprocessed_data(args.golden_csv, args.anomaly_csv, args.ground_truth)

    # Step 3
    visualize_weight_changes_by_window(args.golden_weights, args.anomaly_weights, args.anomaly_type, args.output_dir)

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nCheck the plot and console output. If the first significant change is near window 0,")
    print("it confirms that the model is detecting global changes, likely due to optimization noise")
    print("or high system coupling. If the onset is much later, the temporal strategy should work.")

if __name__ == "__main__":
    main()
