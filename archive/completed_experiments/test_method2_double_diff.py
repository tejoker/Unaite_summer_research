#!/usr/bin/env python3
"""
Test Method 2: Double differencing to turn trend change into spike
Tests if taking second-order differences creates a detectable spike.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_double_differencing():
    """Test double differencing on trend_change anomaly"""

    # Load raw data (before any differencing)
    try:
        golden_raw = pd.read_csv('data/Golden/chunking/output_of_the_1th_chunk.csv', index_col=0)
        trend_raw = pd.read_csv('data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__trend_change.csv', index_col=0)
        print(f"Loading Golden data from: data/Golden/chunking/output_of_the_1th_chunk.csv")
        print(f"Loading Trend data from: data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__trend_change.csv\n")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find raw data files: {e}")
        print("Expected files:")
        print("  data/Golden/chunking/output_of_the_1th_chunk.csv")
        print("  data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__trend_change.csv")
        return

    sensor = 'Temperatur Exzenterlager links'

    if sensor not in trend_raw.columns:
        print(f"ERROR: {sensor} not found in data")
        return

    golden_series = golden_raw[sensor].values
    trend_series = trend_raw[sensor].values

    print("=" * 80)
    print("METHOD 2: Double Differencing (Trend Change → Spike)")
    print("=" * 80)

    # Apply first-order differencing
    golden_diff1 = np.diff(golden_series)
    trend_diff1 = np.diff(trend_series)

    # Apply second-order differencing
    golden_diff2 = np.diff(golden_diff1)
    trend_diff2 = np.diff(trend_diff1)

    print(f"\nOriginal data length: {len(trend_series)}")
    print(f"After 1st difference: {len(trend_diff1)}")
    print(f"After 2nd difference: {len(trend_diff2)}")

    # Analyze the double-differenced data
    print("\n--- Statistics of Double-Differenced Data ---")
    print(f"Golden - Mean: {golden_diff2.mean():.8f}, Std: {golden_diff2.std():.8f}")
    print(f"Trend  - Mean: {trend_diff2.mean():.8f}, Std: {trend_diff2.std():.8f}")

    # The anomaly starts at row 200 in original data
    # After 2 differences, this becomes row 198
    anomaly_start_shifted = 198
    anomaly_end_shifted = 298

    print(f"\n--- Looking for spike near row {anomaly_start_shifted} (anomaly start) ---")

    # Calculate z-scores to find outliers
    trend_mean = trend_diff2.mean()
    trend_std = trend_diff2.std()

    if trend_std > 0:
        z_scores = np.abs((trend_diff2 - trend_mean) / trend_std)

        # Find extreme values (potential spikes)
        threshold_values = [3.0, 4.0, 5.0, 10.0]
        for threshold in threshold_values:
            spike_indices = np.where(z_scores > threshold)[0]
            print(f"\nZ-score > {threshold}: {len(spike_indices)} points")

            if len(spike_indices) > 0:
                # Check if any are near the expected anomaly region
                near_anomaly = [idx for idx in spike_indices
                               if anomaly_start_shifted - 10 <= idx <= anomaly_end_shifted + 10]

                if near_anomaly:
                    print(f"  *** Near expected anomaly region: {near_anomaly}")
                    for idx in near_anomaly[:5]:  # Show first 5
                        print(f"      Row {idx}: value={trend_diff2[idx]:.8f}, z-score={z_scores[idx]:.2f}")
                else:
                    if len(spike_indices) <= 10:
                        print(f"  Locations: {spike_indices.tolist()}")
                    else:
                        print(f"  First 10 locations: {spike_indices[:10].tolist()}")
    else:
        print("ERROR: Standard deviation is zero - cannot calculate z-scores")

    # Find the maximum absolute value in the expected region
    search_start = max(0, anomaly_start_shifted - 20)
    search_end = min(len(trend_diff2), anomaly_end_shifted + 20)

    region_values = trend_diff2[search_start:search_end]
    max_abs_idx = np.argmax(np.abs(region_values))
    max_abs_value = region_values[max_abs_idx]
    actual_idx = search_start + max_abs_idx

    print(f"\n--- Maximum spike in region [{search_start}, {search_end}] ---")
    print(f"Location: row {actual_idx}")
    print(f"Value: {max_abs_value:.8f}")
    print(f"Distance from anomaly start ({anomaly_start_shifted}): {actual_idx - anomaly_start_shifted} rows")

    # Compare spike magnitude to baseline noise
    baseline_region = trend_diff2[:100]
    baseline_std = baseline_region.std()
    spike_magnitude = abs(max_abs_value) / baseline_std if baseline_std > 0 else 0

    print(f"\n--- Spike Magnitude vs Baseline ---")
    print(f"Baseline std (first 100 points): {baseline_std:.8f}")
    print(f"Spike magnitude (in std units): {spike_magnitude:.2f}")

    if spike_magnitude > 5.0:
        print("  ✓ Strong spike detected (>5σ)")
    elif spike_magnitude > 3.0:
        print("  ~ Moderate spike (3-5σ)")
    else:
        print("  ✗ Weak signal (<3σ) - unlikely to be reliably detected")

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Plot 1: Original data
    axes[0].plot(trend_series[:500], label='Trend Change (Raw)', alpha=0.7)
    axes[0].plot(golden_series[:500], label='Golden (Raw)', alpha=0.7)
    axes[0].axvline(x=200, color='red', linestyle='--', label='Anomaly Start')
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('Row')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: First difference
    axes[1].plot(trend_diff1[:500], label='Trend Change (1st Diff)', alpha=0.7)
    axes[1].plot(golden_diff1[:500], label='Golden (1st Diff)', alpha=0.7)
    axes[1].axvline(x=199, color='red', linestyle='--', label='Anomaly Start (shifted)')
    axes[1].set_title('First-Order Difference (Level Shift)')
    axes[1].set_xlabel('Row')
    axes[1].set_ylabel('Δ Temperature')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Second difference (full view)
    axes[2].plot(trend_diff2, label='Trend Change (2nd Diff)', alpha=0.7)
    axes[2].axvline(x=anomaly_start_shifted, color='red', linestyle='--', label='Expected Spike')
    axes[2].axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # Highlight detected spikes
    if trend_std > 0 and len(np.where(z_scores > 5.0)[0]) > 0:
        spike_indices = np.where(z_scores > 5.0)[0]
        axes[2].scatter(spike_indices, trend_diff2[spike_indices],
                       color='purple', s=50, zorder=5, label='Detected Spikes (>5σ)')

    axes[2].set_title('Second-Order Difference (Should Show Spike at Trend Change)')
    axes[2].set_xlabel('Row')
    axes[2].set_ylabel('Δ² Temperature')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Zoomed view around anomaly
    zoom_start = max(0, anomaly_start_shifted - 50)
    zoom_end = min(len(trend_diff2), anomaly_end_shifted + 50)

    axes[3].plot(range(zoom_start, zoom_end),
                 trend_diff2[zoom_start:zoom_end],
                 label='Trend Change (2nd Diff)', alpha=0.7, marker='o', markersize=3)
    axes[3].axvline(x=anomaly_start_shifted, color='red', linestyle='--',
                   linewidth=2, label=f'Expected Spike (row {anomaly_start_shifted})')
    axes[3].axvline(x=actual_idx, color='purple', linestyle='-',
                   linewidth=2, label=f'Max Spike (row {actual_idx})')
    axes[3].axhline(y=0, color='black', linestyle=':', alpha=0.5)

    # Add ±3σ bands
    if trend_std > 0:
        axes[3].axhline(y=trend_mean + 3 * trend_std, color='orange',
                       linestyle='--', alpha=0.5, label='±3σ threshold')
        axes[3].axhline(y=trend_mean - 3 * trend_std, color='orange',
                       linestyle='--', alpha=0.5)

    axes[3].set_title(f'Zoomed View: Anomaly Region ({zoom_start}-{zoom_end})')
    axes[3].set_xlabel('Row')
    axes[3].set_ylabel('Δ² Temperature')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('method2_double_diff_results.png', dpi=150, bbox_inches='tight')
    print(f"\n*** Visualization saved to method2_double_diff_results.png ***")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    success = False
    if trend_std > 0:
        near_spikes = [idx for idx in np.where(z_scores > 3.0)[0]
                      if anomaly_start_shifted - 10 <= idx <= anomaly_end_shifted + 10]
        if near_spikes:
            print(f"✓ Detected {len(near_spikes)} significant spikes (>3σ) near anomaly region")
            print(f"  Locations: {near_spikes}")
            success = True

    if not success:
        print("✗ NO significant spikes detected near expected anomaly region")
        print(f"  Expected spike at row ~{anomaly_start_shifted}")
        print(f"  Strongest signal: {spike_magnitude:.2f}σ")

        if spike_magnitude < 3.0:
            print("\nLikely reasons:")
            print("  1. Trend change too gradual (slope changes slowly)")
            print("  2. Differencing amplifies noise more than signal")
            print("  3. Synthetic anomaly magnitude too small")

    # Test on drift for comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Testing on Drift Anomaly")
    print("=" * 80)

    try:
        drift_raw = pd.read_csv('data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__drift.csv', index_col=0)
        drift_series = drift_raw[sensor].values
        drift_diff2 = np.diff(np.diff(drift_series))

        drift_mean = drift_diff2.mean()
        drift_std = drift_diff2.std()

        if drift_std > 0:
            drift_z = np.abs((drift_diff2 - drift_mean) / drift_std)
            drift_spikes = np.where(drift_z > 3.0)[0]

            print(f"Drift anomaly - Found {len(drift_spikes)} spikes (>3σ) in double-differenced data")
            near_anomaly_drift = [idx for idx in drift_spikes
                                 if anomaly_start_shifted - 10 <= idx <= anomaly_end_shifted + 10]

            if near_anomaly_drift:
                print(f"  ✓ Near anomaly region: {near_anomaly_drift}")
            else:
                print(f"  ✗ None near expected region")
    except FileNotFoundError:
        print("  (Drift file not found - skipping comparison)")

if __name__ == '__main__':
    test_double_differencing()
