#!/usr/bin/env python3
"""
Analyze variable location anomaly detection results.
Compares golden weights with anomaly weights to find detection windows.
"""

import csv
import os
import sys

def load_weights_csv(filepath):
    """Load weights from CSV and organize by window."""
    windows = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            window_idx = int(row['window_idx'])
            edge = f"{row['parent_name']}→{row['child_name']}"
            weight = float(row['weight'])
            
            if window_idx not in windows:
                windows[window_idx] = {}
            windows[window_idx][edge] = weight
    
    return windows

def compare_weights(golden_windows, anomaly_windows, threshold=0.01):
    """Find windows where weights differ significantly."""
    changed_windows = []
    
    for w_idx in sorted(golden_windows.keys()):
        if w_idx not in anomaly_windows:
            continue
        
        g_weights = golden_windows[w_idx]
        a_weights = anomaly_windows[w_idx]
        
        # Calculate max absolute difference
        max_diff = 0.0
        for edge, g_val in g_weights.items():
            a_val = a_weights.get(edge, 0.0)
            diff = abs(g_val - a_val)
            max_diff = max(max_diff, diff)
        
        if max_diff > threshold:
            changed_windows.append(w_idx)
    
    return changed_windows

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/variable_location_test_20251015_134225"
    
    golden_file = os.path.join(results_dir, "golden", "weights", "weights_enhanced.csv")
    
    if not os.path.exists(golden_file):
        print(f"❌ Golden weights not found: {golden_file}")
        return 1
    
    print("="*80)
    print(" "*20 + "DETECTION TIMING ANALYSIS")
    print("="*80)
    print()
    
    # Load golden weights
    golden_windows = load_weights_csv(golden_file)
    print(f"Loaded golden weights: {len(golden_windows)} windows")
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
    
    for row in sorted(expected.keys()):
        exp_min, exp_max, desc = expected[row]
        anomaly_file = os.path.join(results_dir, f"spike_row{row}", "weights", "weights_enhanced.csv")
        
        if not os.path.exists(anomaly_file):
            print(f"⚠️  Weights not found for row {row}")
            results_summary.append({
                'row': row,
                'description': desc,
                'expected_min': exp_min,
                'expected_max': exp_max,
                'first_detection': None,
                'status': '❌'
            })
            continue
        
        # Load anomaly weights
        anomaly_windows = load_weights_csv(anomaly_file)
        
        # Find changed windows
        changed_windows = compare_weights(golden_windows, anomaly_windows)
        
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
        print()
        
        # Additional statistics
        print("📊 Detection Statistics:")
        print(f"   Number of tests with detection: {len(all_detections)} / {len(results_summary)}")
        print(f"   Detection range: Window {min_det} to Window {max_det}")
        print(f"   Total span: {span} windows")
        
        # Check for hardcoding around windows 9-10
        detections_in_9_10 = [d for d in all_detections if 9 <= d <= 10]
        if detections_in_9_10:
            print(f"   ⚠️ {len(detections_in_9_10)} detection(s) in windows 9-10 (potentially suspicious)")
        else:
            print(f"   ✅ No detections in windows 9-10 (good diversity)")
        print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
