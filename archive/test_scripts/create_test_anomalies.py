#!/usr/bin/env python3
"""
Create test anomalies at different time locations to verify
that detection is not hardcoded to windows 9-10
"""

import pandas as pd
import json
import os
import sys

def create_spike_at_location(golden_csv, output_dir, row_location, magnitude=50.0):
    """Create a spike anomaly at a specific row location"""
    
    # Load golden data
    df = pd.read_csv(golden_csv, index_col=0)
    
    # Validate row location
    if row_location >= len(df) or row_location < 0:
        print(f"❌ Error: Row {row_location} is out of bounds (dataset has {len(df)} rows)")
        return None
    
    # Create anomaly
    df_anomaly = df.copy()
    # Add spike to the second column (Temperatur Exzenterlager links)
    anomaly_col = df.columns[1]
    df_anomaly.iloc[row_location, df_anomaly.columns.get_loc(anomaly_col)] += magnitude
    
    # Create filename
    name = f"spike_row{row_location}"
    out_csv = os.path.join(output_dir, f"{name}.csv")
    out_json = os.path.join(output_dir, f"{name}.json")
    
    # Save CSV
    df_anomaly.to_csv(out_csv)
    
    # Create ground truth JSON
    gt = {
        "input": golden_csv,
        "output_csv": out_csv,
        "ts_col": anomaly_col,
        "anomaly": "spike",
        "start": row_location,
        "length": 0,
        "magnitude": magnitude,
        "factor": 1.0,
        "a": 1.0,
        "b": 0.0,
        "mode": "add",
        "seed": 42,
        "rows": len(df)
    }
    
    with open(out_json, 'w') as f:
        json.dump(gt, f, indent=2)
    
    return out_csv, out_json, anomaly_col

def calculate_expected_window(row_location, window_size=100, stride=10):
    """Calculate which window should first detect the anomaly"""
    # After differencing, row N becomes row N-1
    diff_row = row_location - 1
    
    # Find first window containing this row
    for w in range(0, 100):
        w_start = w * stride
        w_end = w_start + window_size - 1
        
        if w_start <= diff_row <= w_end:
            return w, w_start, w_end
    
    return None, None, None

def main():
    # Configuration
    golden_csv = "data/Golden/chunking/output_of_the_1th_chunk.csv"
    output_dir = "data/Test"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test locations - strategically chosen to span different windows
    test_configs = [
        {'row': 50, 'description': 'Very Early (Window 0-5)'},
        {'row': 100, 'description': 'Early (Window 1-10)'},
        {'row': 200, 'description': 'Early-Mid (Window 10-20)'},  # Original spike location
        {'row': 350, 'description': 'Middle (Window 25-35)'},
        {'row': 500, 'description': 'Late-Mid (Window 40-50)'},
        {'row': 700, 'description': 'Late (Window 60-70)'},
    ]
    
    print("="*80)
    print(" "*20 + "CREATING TEST ANOMALIES AT DIFFERENT LOCATIONS")
    print("="*80)
    print()
    
    # Check if golden file exists
    if not os.path.exists(golden_csv):
        print(f"❌ Error: Golden file not found: {golden_csv}")
        return 1
    
    # Load to check size
    df = pd.read_csv(golden_csv, index_col=0)
    print(f"✅ Golden dataset loaded: {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    print()
    
    created_files = []
    
    for config in test_configs:
        row = config['row']
        desc = config['description']
        
        print(f"Creating anomaly at row {row} ({desc})...")
        
        result = create_spike_at_location(golden_csv, output_dir, row)
        if result is None:
            continue
        
        out_csv, out_json, col_name = result
        expected_window, w_start, w_end = calculate_expected_window(row)
        
        created_files.append({
            'row': row,
            'csv': out_csv,
            'json': out_json,
            'expected_window': expected_window,
            'window_range': f"[{w_start}, {w_end}]",
            'description': desc
        })
        
        print(f"  ✅ Created: {os.path.basename(out_csv)}")
        print(f"     Expected detection: Window {expected_window} {w_start}-{w_end}")
        print()
    
    print("="*80)
    print(" "*25 + "TEST ANOMALIES CREATED SUCCESSFULLY")
    print("="*80)
    print()
    
    # Create summary
    print("SUMMARY TABLE:")
    print("-"*80)
    print(f"{'Row':<8} {'Expected Win':<15} {'Window Range':<20} {'Description':<30}")
    print("-"*80)
    for f in created_files:
        print(f"{f['row']:<8} {f['expected_window']:<15} {f['window_range']:<20} {f['description']:<30}")
    print("-"*80)
    print()
    
    print(f"Total test files created: {len(created_files)}")
    print(f"Output directory: {output_dir}")
    print()
    
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print()
    print("Run the test script to verify detection at different time locations:")
    print()
    print("  bash executable/test/test_variable_location_anomalies.sh")
    print()
    print("This will prove that detection is NOT hardcoded to windows 9-10,")
    print("but correctly identifies anomalies based on their actual temporal location.")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
