#!/usr/bin/env python3
"""
Test Method 1: Change point detection on differenced data
Tests if ruptures can detect the level shift left by a trend change after differencing.
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from pathlib import Path

def test_changepoint_on_synthetic():
    """Test on our known synthetic trend_change anomaly"""

    # Find the most recent differenced data files - try multiple patterns
    golden_patterns = [
        "results/Golden/preprocessing/*_differenced_stationary_series.csv",
        "results/*/Golden/*/preprocessing/*_differenced_stationary_series.csv"
    ]
    trend_patterns = [
        "results/*/Anomaly/*/*trend_change*/preprocessing/*_differenced_stationary_series.csv"
    ]

    golden_files = []
    for pattern in golden_patterns:
        golden_files.extend(list(Path(".").glob(pattern)))

    trend_files = []
    for pattern in trend_patterns:
        trend_files.extend(list(Path(".").glob(pattern)))

    if not golden_files or not trend_files:
        print("ERROR: Could not find differenced data files")
        print(f"Golden files found: {len(golden_files)}")
        print(f"Trend files found: {len(trend_files)}")
        return

    # Get most recent files
    golden_file = max(golden_files, key=lambda p: p.stat().st_mtime)
    trend_file = max(trend_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading Golden data from: {golden_file}")
    print(f"Loading Trend data from: {trend_file}\n")

    # Load the differenced data (already preprocessed)
    golden_diff = pd.read_csv(golden_file, index_col=0)
    trend_diff = pd.read_csv(trend_file, index_col=0)

    # Extract the sensor that has the anomaly
    sensor = 'Temperatur Exzenterlager links'
    sensor_diff = f"{sensor}_diff"

    if sensor_diff not in trend_diff.columns:
        print(f"ERROR: {sensor_diff} not found in differenced data")
        print(f"Available columns: {trend_diff.columns.tolist()}")
        return

    golden_series = golden_diff[sensor_diff].values
    trend_series = trend_diff[sensor_diff].values

    print("=" * 80)
    print("METHOD 1: Change Point Detection on Differenced Data")
    print("=" * 80)

    # Test on the anomaly data (which should have a level shift from trend change)
    print(f"\nTesting on trend_change differenced data...")
    print(f"Length: {len(trend_series)}")
    print(f"Mean: {trend_series.mean():.6f}, Std: {trend_series.std():.6f}")

    # The anomaly is at row 200, happens over 100 rows (200-300)
    # In differenced data, this should create a level shift

    print("\n--- Visual inspection of differenced data around anomaly region ---")
    before = trend_series[150:200].mean()
    during = trend_series[200:300].mean()
    after = trend_series[300:350].mean()

    print(f"Mean before anomaly (150-200): {before:.8f}")
    print(f"Mean during anomaly (200-300): {during:.8f}")
    print(f"Mean after anomaly (300-350):  {after:.8f}")
    print(f"Change at anomaly start: {during - before:.8f}")

    # Apply multiple change point detection algorithms
    results = {}

    # Method 1a: PELT with L2 cost (mean changes)
    print("\n--- Algorithm 1: PELT (L2 cost for mean changes) ---")
    try:
        model = "l2"
        algo = rpt.Pelt(model=model, min_size=10).fit(trend_series)

        # Try multiple penalty values
        for pen_factor in [0.5, 1.0, 1.5, 2.0, 3.0]:
            pen = np.log(len(trend_series)) * pen_factor
            change_points = algo.predict(pen=pen)

            # Remove the last point (which is always the end of the series)
            change_points = [cp for cp in change_points if cp < len(trend_series)]

            print(f"  Penalty factor {pen_factor}: Found {len(change_points)} change points")
            if change_points:
                print(f"    Locations: {change_points}")
                # Check if any are near the expected anomaly (200-300)
                near_anomaly = [cp for cp in change_points if 190 <= cp <= 310]
                if near_anomaly:
                    print(f"    *** Near expected anomaly region (200-300): {near_anomaly}")
                    results[f'PELT_pen{pen_factor}'] = near_anomaly
    except Exception as e:
        print(f"  PELT failed: {e}")

    # Method 1b: Binary Segmentation
    print("\n--- Algorithm 2: Binary Segmentation ---")
    try:
        algo = rpt.Binseg(model="l2", min_size=10).fit(trend_series)

        for n_bkps in [1, 2, 3, 5]:
            change_points = algo.predict(n_bkps=n_bkps)
            change_points = [cp for cp in change_points if cp < len(trend_series)]

            print(f"  Requesting {n_bkps} breakpoints: Found at {change_points}")
            near_anomaly = [cp for cp in change_points if 190 <= cp <= 310]
            if near_anomaly:
                print(f"    *** Near expected anomaly region: {near_anomaly}")
                results[f'BinSeg_n{n_bkps}'] = near_anomaly
    except Exception as e:
        print(f"  Binary Segmentation failed: {e}")

    # Method 1c: Window-based (sliding window to detect changes)
    print("\n--- Algorithm 3: Window-Based Detection ---")
    try:
        algo = rpt.Window(width=50, model="l2").fit(trend_series)

        for pen_factor in [1.0, 2.0, 3.0]:
            pen = np.log(len(trend_series)) * pen_factor
            change_points = algo.predict(pen=pen)
            change_points = [cp for cp in change_points if cp < len(trend_series)]

            print(f"  Penalty factor {pen_factor}: Found {len(change_points)} change points")
            if change_points:
                print(f"    Locations: {change_points}")
                near_anomaly = [cp for cp in change_points if 190 <= cp <= 310]
                if near_anomaly:
                    print(f"    *** Near expected anomaly region: {near_anomaly}")
                    results[f'Window_pen{pen_factor}'] = near_anomaly
    except Exception as e:
        print(f"  Window-based failed: {e}")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Full differenced series
    axes[0].plot(trend_series, label='Trend Change (Differenced)', alpha=0.7)
    axes[0].axvline(x=200, color='red', linestyle='--', label='Anomaly Start (200)')
    axes[0].axvline(x=300, color='orange', linestyle='--', label='Anomaly End (300)')
    axes[0].axhline(y=before, color='blue', linestyle=':', alpha=0.5, label=f'Mean Before: {before:.6f}')
    axes[0].axhline(y=during, color='green', linestyle=':', alpha=0.5, label=f'Mean During: {during:.6f}')
    axes[0].set_title('Differenced Data - Trend Change Should Appear as Level Shift')
    axes[0].set_xlabel('Row')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Zoomed to anomaly region
    zoom_start, zoom_end = 100, 400
    axes[1].plot(range(zoom_start, zoom_end), trend_series[zoom_start:zoom_end],
                 label='Trend Change (Differenced)', alpha=0.7)
    axes[1].axvline(x=200, color='red', linestyle='--', label='Anomaly Start')
    axes[1].axvline(x=300, color='orange', linestyle='--', label='Anomaly End')

    # Add detected change points
    for method, cps in results.items():
        for cp in cps:
            if zoom_start <= cp <= zoom_end:
                axes[1].axvline(x=cp, color='purple', linestyle='-', alpha=0.5, linewidth=2)
                axes[1].text(cp, axes[1].get_ylim()[1] * 0.9, method,
                            rotation=90, fontsize=8, alpha=0.7)

    axes[1].set_title('Zoomed View: Anomaly Region (100-400)')
    axes[1].set_xlabel('Row')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Comparison with Golden
    axes[2].plot(trend_series[:500], label='Trend Change (Differenced)', alpha=0.7)
    axes[2].plot(golden_series[:500], label='Golden (Differenced)', alpha=0.7)
    axes[2].axvline(x=200, color='red', linestyle='--', label='Expected Anomaly')
    axes[2].set_title('Trend Change vs Golden (First 500 points)')
    axes[2].set_xlabel('Row')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('method1_changepoint_results.png', dpi=150, bbox_inches='tight')
    print(f"\n*** Visualization saved to method1_changepoint_results.png ***")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if results:
        print(f"✓ Detected {len(results)} potential change points near anomaly region")
        for method, cps in results.items():
            print(f"  {method}: {cps}")
    else:
        print("✗ NO change points detected near expected anomaly region (200-300)")
        print("  Likely reason: Signal magnitude too small after differencing")
        print(f"  Mean change at anomaly: {during - before:.8f}")

    # Compare to noise level
    noise_std = trend_series[:100].std()
    signal_change = abs(during - before)
    snr = signal_change / noise_std if noise_std > 0 else 0
    print(f"\nSignal-to-Noise Ratio:")
    print(f"  Noise (std of first 100 points): {noise_std:.8f}")
    print(f"  Signal (mean change at anomaly): {signal_change:.8f}")
    print(f"  SNR: {snr:.4f}")
    if snr < 2.0:
        print("  *** SNR < 2.0: Signal likely too weak for reliable detection ***")

if __name__ == '__main__':
    test_changepoint_on_synthetic()
