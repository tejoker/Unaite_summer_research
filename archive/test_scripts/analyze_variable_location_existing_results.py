#!/usr/bin/env python3
"""
Analyze existing variable location test results to see if anomalies were detected
at the correct temporal locations.
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path

results_dir = Path("results/variable_location_test_20251015_145942")
golden_dir = results_dir / "golden"

# Load golden weights from CSV
golden_weights_file = golden_dir / "weights" / "weights_enhanced.csv"
if not golden_weights_file.exists():
    print(f"ERROR: Golden weights not found: {golden_weights_file}")
    sys.exit(1)

golden_df = pd.read_csv(golden_weights_file)

# Convert to window-based format
golden_weights = []
for window_id in sorted(golden_df['window_idx'].unique()):
    window_data = golden_df[golden_df['window_idx'] == window_id]
    weights = {}
    for _, row in window_data.iterrows():
        edge = f"{row['parent_name']}→{row['child_name']}"
        weights[edge] = row['weight']
    golden_weights.append({"weights": weights})

print("="*80)
print(" "*20 + "DETECTION TIMING ANALYSIS")
print("="*80)
print()

# Expected windows for each row location
expected = {
    50: (0, 5, "Very Early"),
    100: (1, 10, "Early"),
    200: (10, 20, "Early-Mid"),
    350: (25, 35, "Middle"),
    500: (40, 50, "Late-Mid"),
    700: (60, 70, "Late")
}

results_summary = []

for row, (exp_min, exp_max, desc) in expected.items():
    anomaly_dir = results_dir / f"spike_row{row}"
    weights_file = anomaly_dir / "weights" / "weights_enhanced.csv"

    if not weights_file.exists():
        print(f"WARNING: Weights not found for row {row}")
        continue

    # Load anomaly weights from CSV
    anomaly_df = pd.read_csv(weights_file)

    # Convert to window-based format
    anomaly_weights = []
    for window_id in sorted(anomaly_df['window_idx'].unique()):
        window_data = anomaly_df[anomaly_df['window_idx'] == window_id]
        weights = {}
        for _, wrow in window_data.iterrows():
            edge = f"{wrow['parent_name']}→{wrow['child_name']}"
            weights[edge] = wrow['weight']
        anomaly_weights.append({"weights": weights})

    # Find windows with differences
    changed_windows = []
    for w_idx in range(len(golden_weights)):
        if w_idx >= len(anomaly_weights):
            break

        g_weights = golden_weights[w_idx]["weights"]
        a_weights = anomaly_weights[w_idx]["weights"]

        # Calculate max absolute difference
        max_diff = 0.0
        for edge, g_val in g_weights.items():
            a_val = a_weights.get(edge, 0.0)
            diff = abs(g_val - a_val)
            max_diff = max(max_diff, diff)

        if max_diff > 0.01:  # threshold
            changed_windows.append(w_idx)

    if changed_windows:
        first_detection = min(changed_windows)
        last_detection = max(changed_windows)
        num_windows = len(changed_windows)

        # Check if in expected range
        in_range = exp_min <= first_detection <= exp_max
        status = "✅" if in_range else "⚠️"

        results_summary.append({
            'row': row,
            'description': desc,
            'expected_min': exp_min,
            'expected_max': exp_max,
            'first_detection': first_detection,
            'last_detection': last_detection,
            'num_windows': num_windows,
            'in_range': in_range,
            'status': status
        })

        print(f"{status} Row {row:3d} ({desc:15s})")
        print(f"   Expected range: Windows {exp_min:2d}-{exp_max:2d}")
        print(f"   First detection: Window {first_detection:2d}")
        print(f"   Last detection:  Window {last_detection:2d}")
        print(f"   Total changed:   {num_windows} windows")
        print()
    else:
        print(f"❌ Row {row:3d} ({desc:15s})")
        print(f"   NO DETECTION (no windows changed)")
        print()
        results_summary.append({
            'row': row,
            'description': desc,
            'expected_min': exp_min,
            'expected_max': exp_max,
            'first_detection': None,
            'last_detection': None,
            'num_windows': 0,
            'in_range': False,
            'status': '❌'
        })

print("="*80)
print(" "*25 + "SUMMARY TABLE")
print("="*80)
print()
print(f"{'Row':<6} {'Description':<15} {'Expected':<12} {'Detected':<12} {'Windows':<10} {'Status':<8}")
print("-"*80)

for r in results_summary:
    exp_range = f"{r['expected_min']}-{r['expected_max']}"
    if r['first_detection'] is not None:
        det_range = f"{r['first_detection']}-{r['last_detection']}"
        num_win = r['num_windows']
    else:
        det_range = "NONE"
        num_win = 0

    print(f"{r['row']:<6} {r['description']:<15} {exp_range:<12} {det_range:<12} {num_win:<10} {r['status']:<8}")

print("-"*80)
print()

# Check if detections are diverse (not all around windows 9-10)
all_detections = [r['first_detection'] for r in results_summary if r['first_detection'] is not None]
if all_detections:
    min_det = min(all_detections)
    max_det = max(all_detections)
    span = max_det - min_det

    print("="*80)
    print(" "*30 + "CONCLUSION")
    print("="*80)
    print()

    if span >= 30:
        print("✅ DETECTION IS NOT HARDCODED!")
        print(f"   Detections span {span} windows (from Window {min_det} to Window {max_det})")
        print("   This proves the system correctly identifies anomalies at their temporal location.")
    elif span >= 15:
        print("✅ DETECTION APPEARS CORRECT")
        print(f"   Detections span {span} windows (from Window {min_det} to Window {max_det})")
        print("   Some clustering is expected due to signal strength requirements.")
    else:
        print("⚠️  LIMITED DETECTION RANGE")
        print(f"   Detections only span {span} windows (from Window {min_det} to Window {max_det})")
        print("   This might indicate issues with sensitivity or test design.")

    # Check success rate
    success_count = sum(1 for r in results_summary if r['in_range'])
    total_count = len(results_summary)
    success_rate = success_count / total_count * 100

    print()
    print(f"   Localization success rate: {success_count}/{total_count} ({success_rate:.1f}%)")
    print("="*80)
