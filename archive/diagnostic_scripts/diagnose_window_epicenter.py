#!/usr/bin/env python3
"""
Diagnostic Script: Window Epicenter Analysis

Purpose:
For a given set of windows, this script identifies the "epicenter" sensor—the sensor
contributing most to the weight changes in that specific window.

This helps distinguish between:
- Generalized "optimization noise" (changes spread across many sensors).
- A localized, true signal (changes concentrated on one or two sensors).

Use this to compare the character of an early "noise" window (like window 6)
with a "max impact" window (like window 71).
"""

import pandas as pd
import argparse
from pathlib import Path
import json

def calculate_window_epicenter(
    golden_weights_file: str,
    anomaly_weights_file: str,
    window_indices: list,
    ground_truth_file: str
):
    """
    For each specified window, calculate the epicenter sensor and compare it to ground truth.
    """
    print("\n" + "="*80)
    print("WINDOW EPICENTER ANALYSIS")
    print("="*80)

    try:
        df_g = pd.read_csv(golden_weights_file)
        df_a = pd.read_csv(anomaly_weights_file)
    except FileNotFoundError as e:
        print(f"❌ Error loading weight files: {e}")
        return

    # Merge all weights once for efficiency
    merged = pd.merge(
        df_g, df_a,
        on=['window_idx', 'child_name', 'parent_name', 'lag'],
        how='outer', suffixes=('_g', '_a')
    ).fillna(0)

    merged['weight_diff'] = abs(merged['weight_a'] - merged['weight_g'])

    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        gt = json.load(f)
    gt_sensor = gt['ts_col']
    print(f"Ground Truth Sensor: '{gt_sensor}'\n")

    # Analyze each requested window
    for window_idx in window_indices:
        print(f"--- Analyzing Window {window_idx} ---")

        window_changes = merged[merged['window_idx'] == window_idx]

        if window_changes.empty:
            print("  No data found for this window.")
            continue

        total_change_in_window = window_changes['weight_diff'].sum()
        if total_change_in_window < 1e-6:
            print("  No significant weight changes in this window.")
            continue

        print(f"  Total weight change in window: {total_change_in_window:.4f}")

        # Calculate impact score for each sensor IN THIS WINDOW
        sensor_scores = {}
        involved_sensors = pd.concat([window_changes['child_name'], window_changes['parent_name']]).unique()

        for sensor in involved_sensors:
            mask = (window_changes['child_name'] == sensor) | (window_changes['parent_name'] == sensor)
            sensor_edges = window_changes[mask]
            
            total_impact = sensor_edges['weight_diff'].sum()
            n_edges = len(sensor_edges[sensor_edges['weight_diff'] > 1e-6])
            normalized_impact = total_impact / n_edges if n_edges > 0 else 0.0
            
            sensor_scores[sensor] = {'total_impact': total_impact, 'normalized_impact': normalized_impact, 'n_edges': n_edges}

        if not sensor_scores:
            print("  No sensor impacts calculated.")
            continue

        # Create a DataFrame for nice printing
        scores_df = pd.DataFrame.from_dict(sensor_scores, orient='index').reset_index().rename(columns={'index': 'sensor'})
        scores_df = scores_df.sort_values('normalized_impact', ascending=False).reset_index(drop=True)

        # Contribution is based on total impact, not normalized
        scores_df['contribution_pct'] = (scores_df['total_impact'] / (total_change_in_window * 2)) * 100

        # Identify epicenter
        epicenter_sensor = scores_df.iloc[0]['sensor']
        epicenter_score = scores_df.iloc[0]['normalized_impact']
        epicenter_contribution = scores_df.iloc[0]['contribution_pct']

        print(f"  Top 5 contributing sensors:")
        print(scores_df[['sensor', 'normalized_impact', 'total_impact', 'n_edges', 'contribution_pct']].head(5).to_string(index=False, formatters={'contribution_pct': '{:.1f}%'.format}))

        # Verdict
        is_correct_epicenter = gt_sensor in epicenter_sensor
        verdict_symbol = "✅" if is_correct_epicenter else "❌"

        print(f"\n  {verdict_symbol} VERDICT (Window {window_idx}):")
        print(f"    Epicenter: '{epicenter_sensor}' (Score: {epicenter_score:.4f})")
        print(f"    Contribution: {epicenter_contribution:.1f}% of total change")
        if not is_correct_epicenter:
            print(f"    This does NOT match the ground truth sensor '{gt_sensor}'.")
        else:
            print(f"    This matches the ground truth sensor.")

        if epicenter_contribution < 30:
            print("    Insight: The change is distributed across many sensors, suggesting generalized noise.")
        else:
            print("    Insight: The change is highly concentrated, suggesting a localized signal.")
        print("-" * 25 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze the epicenter of weight changes in specific windows.")
    parser.add_argument('--golden-weights', required=True, help="Path to the golden weights_enhanced.csv file.")
    parser.add_argument('--anomaly-weights', required=True, help="Path to the anomaly weights_enhanced.csv file.")
    parser.add_argument('--ground-truth', required=True, help="Path to the anomaly's ground truth JSON file.")
    parser.add_argument('--windows', required=True, type=int, nargs='+',
                        help="A list of window indices to analyze (e.g., 6 71).")
    args = parser.parse_args()

    calculate_window_epicenter(
        args.golden_weights,
        args.anomaly_weights,
        args.windows,
        args.ground_truth
    )

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nCompare the 'Insight' for each window.")
    print("If an early window shows distributed change and a later one shows concentrated change,")
    print("it confirms the early detection is noise and the later one is the true signal.")


if __name__ == "__main__":
    main()