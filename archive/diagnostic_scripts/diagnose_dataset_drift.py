#!/usr/bin/env python3
"""
Diagnostic Script: Dataset Drift Analysis

Purpose:
Tests the hypothesis that the Golden and Anomaly datasets have subtle,
system-wide statistical differences even BEFORE the injected anomaly.

This script compares the data in the pre-anomaly region and reports on:
1. The number of non-identical rows.
2. The Mean Absolute Difference (MAD) for each sensor.
3. The Kolmogorov-Smirnov (KS) test p-value to check if the distributions are statistically different.
"""

import pandas as pd
import numpy as np
import argparse
import json
from scipy.stats import ks_2samp

def diagnose_pre_anomaly_drift(golden_csv: str, anomaly_csv: str, ground_truth_file: str):
    """
    Compares two datasets in the region before the anomaly start time.
    """
    print("\n" + "="*80)
    print("PRE-ANOMALY DATASET DRIFT ANALYSIS")
    print("="*80)

    try:
        df_g = pd.read_csv(golden_csv)
        df_a = pd.read_csv(anomaly_csv)
    except FileNotFoundError as e:
        print(f"❌ Error loading CSV files: {e}")
        return

    # Load ground truth to find the anomaly start row
    with open(ground_truth_file, 'r') as f:
        gt = json.load(f)
    anomaly_start_row = gt.get('start', 0)

    if anomaly_start_row == 0:
        print("❌ Anomaly start row is 0. Cannot analyze pre-anomaly region.")
        return

    print(f"Analyzing data from row 0 to {anomaly_start_row - 1} (before the anomaly at row {anomaly_start_row}).\n")

    # Isolate the pre-anomaly data
    pre_g = df_g.iloc[:anomaly_start_row]
    pre_a = df_a.iloc[:anomaly_start_row]

    # Check for overall equality first
    if pre_g.equals(pre_a):
        print("✅ VERDICT: The datasets are PERFECTLY IDENTICAL before the anomaly.")
        print("  -> This refutes the hypothesis of pre-existing dataset drift.")
        print("  -> The early detection at window 6 is likely due to optimization noise.")
        return

    print("⚠️ VERDICT: The datasets have differences BEFORE the anomaly.\n")
    print("--- Sensor-by-Sensor Analysis ---")

    results = []
    for col in df_g.columns:
        if col == 'timestamp':
            continue

        series_g = pre_g[col]
        series_a = pre_a[col]

        if series_g.equals(series_a):
            results.append({'sensor': col, 'identical_rows': 'YES', 'mad': 0.0, 'ks_p_value': 1.0})
            continue

        # Calculate metrics for differing columns
        non_identical_rows = (series_g != series_a).sum()
        mad = np.mean(np.abs(series_g - series_a))
        ks_stat, ks_p_value = ks_2samp(series_g, series_a)

        results.append({
            'sensor': col,
            'identical_rows': f"NO ({non_identical_rows} diffs)",
            'mad': mad,
            'ks_p_value': ks_p_value
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, formatters={'mad': '{:.6f}'.format, 'ks_p_value': '{:.4f}'.format}))

    low_p_value_sensors = results_df[results_df['ks_p_value'] < 0.05]
    if not low_p_value_sensors.empty:
        print("\nStatistical Insight:")
        print("  The following sensors have statistically different distributions (p < 0.05) before the anomaly:")
        for sensor in low_p_value_sensors['sensor']:
            print(f"  - {sensor}")
        print("\n  -> This VALIDATES the hypothesis that subtle, system-wide differences exist.")
        print("  -> The model is likely detecting this pre-existing drift at window 6.")

def main():
    parser = argparse.ArgumentParser(description="Diagnose pre-anomaly dataset drift.")
    parser.add_argument('--golden-csv', required=True, help="Path to the raw golden dataset CSV.")
    parser.add_argument('--anomaly-csv', required=True, help="Path to the raw anomaly dataset CSV.")
    parser.add_argument('--ground-truth', required=True, help="Path to the anomaly's ground truth JSON file.")
    args = parser.parse_args()

    diagnose_pre_anomaly_drift(args.golden_csv, args.anomaly_csv, args.ground_truth)

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()