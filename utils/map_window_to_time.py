#!/usr/bin/env python3
"""
Map window indices back to actual time series positions
"""

import pandas as pd
import numpy as np

def explain_window_mapping():
    """
    Explain the relationship between window index and time series position
    """
    print("="*80)
    print("WINDOW INDEX TO TIME SERIES POSITION MAPPING")
    print("="*80)

    # Based on typical rolling window parameters
    window_size = 100
    lag = 10  # from looking at the results, lag goes up to 10

    print(f"\nParameters:")
    print(f"  Window size: {window_size}")
    print(f"  Lag: {lag}")

    print(f"\nWindow calculation:")
    print(f"  Window w uses data from row (w + lag) to (w + lag + window_size)")
    print(f"  Window w spans: [w+{lag}, w+{lag}+{window_size}) = [w+{lag}, w+{lag+window_size})")

    spike_row = 200
    print(f"\n{'='*80}")
    print(f"SPIKE AT ROW {spike_row}")
    print(f"{'='*80}")

    # Calculate which windows contain the spike
    print(f"\nWindows that CONTAIN row {spike_row}:")
    spike_containing_windows = []
    for w in range(max(0, spike_row - window_size - lag), spike_row + 10):
        start = w + lag
        end = start + window_size
        if start <= spike_row < end:
            position_in_window = spike_row - start
            spike_containing_windows.append((w, start, end, position_in_window))
            if w in [91, 100, 150, 187, 188, 189, 190]:
                marker = " <-- SPECIAL" if w in [187, 188] else ""
                print(f"  Window {w:3d}: [{start:3d}, {end:3d}) - spike at position {position_in_window:2d}/100{marker}")

    print(f"\n  First window with spike: {spike_containing_windows[0][0]}")
    print(f"  Last window with spike:  {spike_containing_windows[-1][0]}")

    # Focus on windows 187-188
    print(f"\n{'='*80}")
    print(f"WHY WINDOWS 187-188 SHOW HUGE CHANGES")
    print(f"{'='*80}")

    for w in [187, 188]:
        if any(x[0] == w for x in spike_containing_windows):
            info = [x for x in spike_containing_windows if x[0] == w][0]
            start, end, pos = info[1], info[2], info[3]
            print(f"\nWindow {w}:")
            print(f"  Data range: [{start}, {end})")
            print(f"  Spike at position: {pos}/100")
            print(f"  Spike influence: {100*pos/window_size:.1f}% from start")
            if pos < 10:
                print(f"  -> Spike is at the VERY BEGINNING (first {pos} rows)")
            elif pos > 90:
                print(f"  -> Spike is at the VERY END (last {100-pos} rows)")

    # Check window 189
    print(f"\nWindow 189:")
    w = 189
    start = w + lag
    end = start + window_size
    print(f"  Data range: [{start}, {end})")
    if start <= spike_row < end:
        pos = spike_row - start
        print(f"  Spike at position: {pos}/100")
    else:
        print(f"  NO SPIKE - spike at row {spike_row} is before window start {start}")
        print(f"  -> This is the FIRST window WITHOUT the spike!")


def map_anomalous_windows_to_time():
    """
    For detected anomalous windows, show what time range they represent
    """
    print(f"\n{'='*80}")
    print("MAPPING ANOMALOUS WINDOWS TO TIME SERIES")
    print(f"{'='*80}")

    anomalous_windows = [187, 188]
    window_size = 100
    lag = 10
    spike_row = 200

    print(f"\nDetected anomalous windows: {anomalous_windows}")
    print(f"True spike position: row {spike_row}")

    for w in anomalous_windows:
        start = w + lag
        end = start + window_size

        print(f"\nWindow {w}:")
        print(f"  Time range: [{start}, {end})")

        if start <= spike_row < end:
            print(f"  Contains spike: YES (at position {spike_row - start})")
            print(f"  Distance from spike to window center: {abs((start + end)/2 - spike_row):.1f} rows")
        else:
            print(f"  Contains spike: NO")
            if spike_row < start:
                print(f"  Spike was {start - spike_row} rows BEFORE this window")
            else:
                print(f"  Spike is {spike_row - end + 1} rows AFTER this window")


def localize_anomaly_in_time_series():
    """
    Strategy: How to go from anomalous window to time series position
    """
    print(f"\n{'='*80}")
    print("ANOMALY LOCALIZATION STRATEGY")
    print(f"{'='*80}")

    window_size = 100
    lag = 10

    print(f"\nGiven: Anomalous windows detected at indices [187, 188]")
    print(f"\nStrategy 1: Report the OVERLAP of all anomalous windows")

    anomalous_windows = [187, 188]
    ranges = []
    for w in anomalous_windows:
        start = w + lag
        end = start + window_size
        ranges.append((start, end))
        print(f"  Window {w}: [{start}, {end})")

    # Find intersection
    overlap_start = max(r[0] for r in ranges)
    overlap_end = min(r[1] for r in ranges)

    print(f"\n  Overlap: [{overlap_start}, {overlap_end})")
    print(f"  -> Report: Anomaly detected in rows {overlap_start} to {overlap_end-1}")
    print(f"  -> True spike at row 200: {'CONTAINED' if overlap_start <= 200 < overlap_end else 'NOT CONTAINED'}")

    print(f"\nStrategy 2: Report UNION (all potentially affected rows)")
    union_start = min(r[0] for r in ranges)
    union_end = max(r[1] for r in ranges)
    print(f"  Union: [{union_start}, {union_end})")
    print(f"  -> Report: Anomaly somewhere in rows {union_start} to {union_end-1}")
    print(f"  -> True spike at row 200: {'CONTAINED' if union_start <= 200 < union_end else 'NOT CONTAINED'}")

    print(f"\nStrategy 3: Look at EDGE changes to pinpoint exact location")
    print(f"  - Analyze which SPECIFIC edges changed in windows 187-188")
    print(f"  - Check 'Temperatur Exzenterlager rechts_diff' (showed -642x change)")
    print(f"  - Look at this sensor's time series in the window range")
    print(f"  - Find the exact row with anomalous value")


def main():
    explain_window_mapping()
    map_anomalous_windows_to_time()
    localize_anomaly_in_time_series()

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    print("""
1. Window 188 contains rows [198, 298)
   - Spike at row 200 is at position 2 in this window
   - The spike is at the VERY BEGINNING (2%)

2. Window 187 contains rows [197, 297)
   - Spike at row 200 is at position 3 in this window
   - The spike is at the VERY BEGINNING (3%)

3. Why huge changes?
   - When spike is at START of window, it has maximum leverage
   - DynoTEARS learns relationships WITH the spike
   - The 98% of normal data conflicts with the 2% spike data
   - This creates large weight adjustments

4. Localization accuracy:
   - Detected windows 187-188 overlap at rows [198, 297)
   - True spike at row 200
   - Error: -2 rows (detected range starts 2 rows before spike)
   - Precision: 99 rows wide (detection is NOT precise)

5. To improve precision:
   - Use smaller windows (e.g., 50 instead of 100)
   - Analyze which specific edges changed
   - Look at the actual time series data in the flagged region
    """)

if __name__ == "__main__":
    main()
