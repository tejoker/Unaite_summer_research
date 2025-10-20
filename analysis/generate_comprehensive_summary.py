#!/usr/bin/env python3
"""
Generate comprehensive anomaly detection summary report
"""

import json
import os
import re

results_dir = "results/master_anomaly_test_20251014_171954"
anomaly_types = ["spike", "drift", "level_shift", "amplitude_change", "trend_change", "variance_burst"]

def calculate_expected_windows(start_row, end_row, window_size=100, stride=10):
    """Calculate which windows should contain the anomaly"""
    if start_row == 'N/A' or not isinstance(start_row, int):
        return set()
    
    # Adjust for differencing (lose first row)
    diff_start = start_row - 1
    diff_end = end_row - 1 if isinstance(end_row, int) and end_row != 'N/A' else diff_start
    
    expected = set()
    for w in range(0, 90):
        w_start = w * stride
        w_end = w_start + window_size - 1
        if w_start <= diff_end and w_end >= diff_start:
            expected.add(w)
    
    return expected

def detect_causality_changes(analysis_text):
    """Detect if there are causality (structure) changes"""
    mismatch_lines = [line for line in analysis_text.split('\n') if 'MISMATCH' in line]
    return len(mismatch_lines), mismatch_lines[:5]  # Return count and first 5

def main():
    print("="*100)
    print(" "*30 + "ANOMALY DETECTION COMPREHENSIVE REPORT")
    print("="*100)
    print()
    print("Test Date: October 14, 2025")
    print("Strategy: Fixed Lambda + Fixed Lags")
    print("Window Size: 100 rows, Stride: 10 rows")
    print()
    print("="*100)
    print()
    
    summary_data = []
    
    for anom_type in anomaly_types:
        print(f"\n{'='*100}")
        print(f"ANOMALY TYPE: {anom_type.upper().replace('_', ' ')}")
        print(f"{'='*100}\n")
        
        # Read ground truth
        gt_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anom_type}.json"
        analysis_file = f"{results_dir}/{anom_type}_run/diagnostics/window_analysis.txt"
        
        if not os.path.exists(gt_file):
            print("⚠️  Ground truth file not found")
            continue
        
        if not os.path.exists(analysis_file):
            print("⚠️  Analysis file not found")
            continue
        
        with open(gt_file, 'r') as f:
            gt = json.load(f)
        
        with open(analysis_file, 'r') as f:
            analysis = f.read()
        
        # Extract ground truth
        start_row = gt.get('start', 'N/A')
        length = gt.get('length', 0)
        magnitude = gt.get('magnitude', gt.get('factor', 'N/A'))
        end_row = start_row + length if isinstance(start_row, int) and length > 0 else start_row
        
        print(f"📋 GROUND TRUTH:")
        print(f"   Start Row: {start_row}")
        print(f"   End Row: {end_row}")
        print(f"   Length: {length} rows")
        print(f"   Magnitude: {magnitude}")
        print()
        
        # Extract detection results
        match = re.search(r'🔍 Windows with ANY weight changes: (\d+)', analysis)
        num_changed = int(match.group(1)) if match else 0
        
        match = re.search(r'Window IDs: \[(.*?)\]', analysis)
        if match:
            window_ids_str = match.group(1)
            changed_windows = [int(x.split('(')[1].split(')')[0]) if '(' in x else int(x) 
                             for x in window_ids_str.split(',') if x.strip()]
        else:
            changed_windows = []
        
        # Calculate expected windows
        expected_windows = calculate_expected_windows(start_row, end_row)
        
        # Detect causality changes
        num_structure_changes, structure_examples = detect_causality_changes(analysis)
        
        # Extract statistics
        match = re.search(r'Max:\s+(\d+\.\d+)', analysis)
        max_l2 = float(match.group(1)) if match else 0.0
        
        match = re.search(r'Mean:\s+(\d+\.\d+)', analysis)
        mean_l2 = float(match.group(1)) if match else 0.0
        
        # Determine detection status
        if num_changed == 0:
            status = "❌ NOT DETECTED"
            timing = "N/A"
        elif len(changed_windows) > 0:
            first_changed = min(changed_windows)
            first_expected = min(expected_windows) if expected_windows else None
            
            if first_expected is None:
                status = "✅ DETECTED"
                timing = "⚠️  UNCERTAIN (no expected windows)"
            elif first_changed < first_expected:
                status = "✅ DETECTED"
                timing = f"⚠️  EARLY (Window {first_changed}, expected Window {first_expected})"
            elif first_changed == first_expected:
                status = "✅ DETECTED"
                timing = f"✅ ON TIME (Window {first_changed})"
            else:
                status = "✅ DETECTED"
                timing = f"⏰ DELAYED (Window {first_changed}, expected Window {first_expected})"
        else:
            status = "⚠️  AMBIGUOUS"
            timing = "N/A"
        
        print(f"🔍 DETECTION STATUS: {status}")
        print(f"   Windows Changed: {num_changed}")
        if changed_windows:
            print(f"   Changed Window IDs: {changed_windows}")
            print(f"   First Changed Window: {min(changed_windows)}")
        if expected_windows:
            print(f"   Expected Window IDs: {sorted(expected_windows)}")
            print(f"   Expected First Window: {min(expected_windows)}")
        print()
        
        print(f"⏱️  TIMING ASSESSMENT: {timing}")
        print()
        
        # Coverage analysis
        if expected_windows and changed_windows:
            detected_set = set(changed_windows)
            overlap_count = len(detected_set & expected_windows)
            coverage = overlap_count / len(expected_windows) if expected_windows else 0
            print(f"📊 COVERAGE:")
            print(f"   Overlap: {overlap_count} / {len(expected_windows)} expected windows")
            print(f"   Coverage: {coverage*100:.1f}%")
            missed = expected_windows - detected_set
            false_pos = detected_set - expected_windows
            if missed:
                print(f"   Missed Windows: {sorted(missed)[:10]}")
            if false_pos:
                print(f"   False Positives: {sorted(false_pos)[:10]}")
            print()
        
        # Causality changes
        if num_structure_changes > 0:
            print(f"🔄 CAUSALITY CHANGES: ✅ YES ({num_structure_changes} windows with structure changes)")
            print(f"   This anomaly changes the causal graph structure, not just edge weights.")
        else:
            print(f"🔄 CAUSALITY CHANGES: ❌ NO")
            print(f"   This anomaly only changes edge weights, preserving graph structure.")
        print()
        
        # Signal strength
        print(f"💪 SIGNAL STRENGTH:")
        print(f"   Max L2 Norm: {max_l2:.4f}")
        print(f"   Mean L2 Norm: {mean_l2:.4f}")
        print()
        
        # Store summary
        summary_data.append({
            'type': anom_type,
            'start': start_row,
            'length': length,
            'detected': num_changed > 0,
            'timing': timing,
            'causality_changes': num_structure_changes > 0,
            'num_changed': num_changed,
            'first_window': min(changed_windows) if changed_windows else None,
            'expected_first': min(expected_windows) if expected_windows else None,
            'max_l2': max_l2
        })
    
    # Overall summary table
    print(f"\n{'='*100}")
    print(" "*35 + "SUMMARY TABLE")
    print(f"{'='*100}\n")
    
    header = f"{'Anomaly Type':<20} {'Start':<8} {'Len':<6} {'Detected':<10} {'Timing':<10} {'Causality':<12} {'Windows':<8} {'Max L2':<10}"
    print(header)
    print("-" * 100)
    
    for d in summary_data:
        detected_str = "✅ YES" if d['detected'] else "❌ NO"
        causality_str = "✅ YES" if d['causality_changes'] else "❌ NO"
        
        # Simplify timing
        if "ON TIME" in d['timing']:
            timing_str = "✅ ON TIME"
        elif "EARLY" in d['timing']:
            timing_str = "⚠️  EARLY"
        elif "DELAYED" in d['timing']:
            timing_str = "⏰ LATE"
        else:
            timing_str = d['timing'][:10]
        
        row = f"{d['type']:<20} {str(d['start']):<8} {str(d['length']):<6} {detected_str:<10} {timing_str:<10} {causality_str:<12} {d['num_changed']:<8} {d['max_l2']:<10.4f}"
        print(row)
    
    print()
    print(f"{'='*100}")
    print()
    
    # Key findings
    print("🎯 KEY FINDINGS:\n")
    
    all_detected = all(d['detected'] for d in summary_data)
    if all_detected:
        print("✅ All anomalies were DETECTED")
    else:
        not_detected = [d['type'] for d in summary_data if not d['detected']]
        print(f"⚠️  Some anomalies NOT detected: {', '.join(not_detected)}")
    
    on_time = [d for d in summary_data if d['detected'] and "ON TIME" in d['timing']]
    print(f"✅ {len(on_time)} anomalies detected ON TIME")
    
    early = [d for d in summary_data if d['detected'] and "EARLY" in d['timing']]
    if early:
        print(f"⚠️  {len(early)} anomalies detected EARLY: {', '.join([d['type'] for d in early])}")
    
    with_causality = [d for d in summary_data if d['causality_changes']]
    without_causality = [d for d in summary_data if not d['causality_changes']]
    print(f"\n🔄 Causality Changes:")
    print(f"   ✅ {len(with_causality)} anomalies change causal structure: {', '.join([d['type'] for d in with_causality])}")
    print(f"   ❌ {len(without_causality)} anomalies preserve causal structure: {', '.join([d['type'] for d in without_causality])}")
    
    print(f"\n💪 Signal Strength:")
    sorted_by_signal = sorted(summary_data, key=lambda x: x['max_l2'], reverse=True)
    print(f"   Strongest: {sorted_by_signal[0]['type']} (L2: {sorted_by_signal[0]['max_l2']:.4f})")
    print(f"   Weakest: {sorted_by_signal[-1]['type']} (L2: {sorted_by_signal[-1]['max_l2']:.4f})")
    
    print()
    print(f"{'='*100}")
    print("Report generated successfully!")
    print(f"Results directory: {results_dir}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()
