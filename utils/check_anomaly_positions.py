#!/usr/bin/env python3
"""Check actual positions of all anomalies"""

import json
from pathlib import Path

anomaly_types = ['spike', 'drift', 'level_shift', 'trend_change', 'amplitude_change', 'variance_burst']

print("="*80)
print("ANOMALY POSITIONS IN METADATA")
print("="*80)

for anomaly in anomaly_types:
    metadata_file = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly}.json"

    if Path(metadata_file).exists():
        with open(metadata_file, 'r') as f:
            data = json.load(f)

        start = data.get('start', 'N/A')
        length = data.get('length', 0)
        magnitude = data.get('magnitude', 'N/A')

        if length > 0:
            end = start + length - 1
            print(f"\n{anomaly:20s} Start: {start:3d}  End: {end:3d}  Length: {length:3d}  Magnitude: {magnitude}")
        else:
            print(f"\n{anomaly:20s} Position: {start:3d}  (instantaneous)  Magnitude: {magnitude}")
    else:
        print(f"\n{anomaly:20s} METADATA NOT FOUND")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("""
You're RIGHT to be suspicious!

Looking at the results from compare_all_anomalies.py:
  - spike:           Top window 188  (Expected 91-200)   ✓ Makes sense
  - drift:           Top window 384  (Expected 41-150)   ✗ FAR OFF
  - level_shift:     Top window 139  (Expected 41-150)   ✓ Close!
  - trend_change:    Top window 378  (Expected 41-150)   ✗ FAR OFF
  - amplitude_change: Top window 988 (Expected 41-150)   ✗ VERY FAR OFF
  - variance_burst:  Top window 988  (Expected 41-150)   ✗ VERY FAR OFF

PROBLEM: Most detected windows are NOT near the actual anomaly position!

This suggests:
1. The anomalies at row 150 are NOT causing detectable weight changes there
2. The large weight changes at windows 384, 988 etc. are from OTHER sources
3. Possible causes:
   - Natural variability in the data (not anomaly-related)
   - End-of-series effects (window 988 is near the end)
   - The anomalies are too subtle for this method
   - Need different detection approach for gradual anomalies (drift, trend_change)

RECOMMENDATION:
Check if detected windows (384, 988) actually correspond to anything in the data,
or if they're just random high-variance windows.
""")
