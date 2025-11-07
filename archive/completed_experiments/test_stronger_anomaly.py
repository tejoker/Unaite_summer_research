#!/usr/bin/env python3
"""
Test with artificially strengthened anomalies
Creates stronger versions of trend_change and drift to test if the methods work
when the signal magnitude is increased.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

def create_strong_anomalies():
    """Create artificially strengthened versions of trend_change and drift"""

    # Load golden baseline
    golden_raw = pd.read_csv('data/Golden/chunking/output_of_the_1th_chunk.csv', index_col=0)
    sensor = 'Temperatur Exzenterlager links'

    print(f"Loading Golden data from: data/Golden/chunking/output_of_the_1th_chunk.csv\n")

    if sensor not in golden_raw.columns:
        print(f"ERROR: {sensor} not found")
        print(f"Available columns: {golden_raw.columns.tolist()}")
        return

    golden_series = golden_raw[sensor].values.copy()

    # Create strong trend change (10x stronger than original)
    # Original: 0.25°C slope change over 100 rows
    # Strong: 2.5°C slope change over 100 rows
    trend_strong = golden_series.copy()
    anomaly_start = 200
    anomaly_duration = 100

    original_slope = 0.0  # Assuming roughly constant before
    new_slope = 2.5 / anomaly_duration  # 2.5°C total change over 100 rows = 0.025 per row

    for i in range(anomaly_start, anomaly_start + anomaly_duration):
        if i < len(trend_strong):
            trend_strong[i:] += new_slope

    # Create strong drift (10x stronger than original)
    # Original: 25°C drift over 100 rows
    # Strong: 250°C drift over 100 rows (unrealistic but clear signal)
    drift_strong = golden_series.copy()
    total_drift = 250.0

    for i in range(anomaly_start, anomaly_start + anomaly_duration):
        if i < len(drift_strong):
            progress = (i - anomaly_start) / anomaly_duration
            drift_strong[i] += total_drift * progress

    # After anomaly, maintain the drift
    if anomaly_start + anomaly_duration < len(drift_strong):
        drift_strong[anomaly_start + anomaly_duration:] += total_drift

    print("=" * 80)
    print("TESTING WITH STRONGER SYNTHETIC ANOMALIES")
    print("=" * 80)

    # Test Method 1 on strong trend change
    print("\n" + "=" * 80)
    print("METHOD 1: Change Point Detection on STRONG Trend Change")
    print("=" * 80)

    # Apply first-order differencing
    trend_diff1 = np.diff(trend_strong)

    print(f"\nStrong trend change (after differencing):")
    print(f"  Mean before (100-200): {trend_diff1[100:200].mean():.6f}")
    print(f"  Mean during (200-300): {trend_diff1[200:300].mean():.6f}")
    print(f"  Mean after (300-400):  {trend_diff1[300:400].mean():.6f}")

    change_at_start = trend_diff1[200:300].mean() - trend_diff1[100:200].mean()
    print(f"  Change at anomaly start: {change_at_start:.6f}")

    # Apply PELT change point detection
    algo = rpt.Pelt(model="l2", min_size=10).fit(trend_diff1)
    pen = np.log(len(trend_diff1)) * 1.5
    change_points = algo.predict(pen=pen)
    change_points = [cp for cp in change_points if cp < len(trend_diff1)]

    print(f"\nPELT detected change points: {change_points}")
    near_anomaly = [cp for cp in change_points if 190 <= cp <= 310]

    if near_anomaly:
        print(f"  ✓ SUCCESS: Found change point near anomaly region (200-300)")
        print(f"    Detected at: {near_anomaly}")
    else:
        print(f"  ✗ FAILED: No change point near expected region")

    # Calculate SNR
    noise_std = trend_diff1[:100].std()
    signal_change = abs(change_at_start)
    snr = signal_change / noise_std if noise_std > 0 else 0
    print(f"\nSignal-to-Noise Ratio: {snr:.2f}")

    # Test Method 2 on strong trend change
    print("\n" + "=" * 80)
    print("METHOD 2: Double Differencing on STRONG Trend Change")
    print("=" * 80)

    # Apply second-order differencing
    trend_diff2 = np.diff(trend_diff1)

    trend_mean = trend_diff2.mean()
    trend_std = trend_diff2.std()

    if trend_std > 0:
        z_scores = np.abs((trend_diff2 - trend_mean) / trend_std)
        spike_indices = np.where(z_scores > 3.0)[0]

        print(f"\nDetected {len(spike_indices)} spikes (>3σ) in double-differenced data")

        # Adjust for 2 differences lost
        anomaly_start_shifted = anomaly_start - 2

        near_anomaly_spikes = [idx for idx in spike_indices
                              if anomaly_start_shifted - 10 <= idx <= anomaly_start_shifted + 110]

        if near_anomaly_spikes:
            print(f"  ✓ SUCCESS: Found spikes near anomaly region")
            print(f"    Detected at: {near_anomaly_spikes[:5]}")  # Show first 5
            print(f"    Expected around: {anomaly_start_shifted}")
        else:
            print(f"  ✗ FAILED: No spikes near expected region ({anomaly_start_shifted})")

        # Find max spike
        region_start = max(0, anomaly_start_shifted - 20)
        region_end = min(len(trend_diff2), anomaly_start_shifted + 120)
        region_values = trend_diff2[region_start:region_end]
        max_idx = region_start + np.argmax(np.abs(region_values))
        max_value = trend_diff2[max_idx]
        max_z = z_scores[max_idx]

        print(f"\nStrongest spike in region:")
        print(f"  Location: row {max_idx}")
        print(f"  Z-score: {max_z:.2f}σ")
    else:
        print("ERROR: Cannot calculate z-scores (std=0)")

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Left column: Trend Change
    axes[0, 0].plot(golden_series[:500], label='Golden', alpha=0.7)
    axes[0, 0].plot(trend_strong[:500], label='Strong Trend Change', alpha=0.7)
    axes[0, 0].axvline(x=200, color='red', linestyle='--', label='Anomaly Start')
    axes[0, 0].set_title('Strong Trend Change (Raw Data)')
    axes[0, 0].set_xlabel('Row')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(trend_diff1[:500], label='1st Difference', alpha=0.7)
    axes[1, 0].axvline(x=199, color='red', linestyle='--', label='Anomaly Start')
    if change_points:
        for cp in change_points:
            if cp < 500:
                axes[1, 0].axvline(x=cp, color='purple', linestyle='-',
                                  linewidth=2, alpha=0.6, label='Detected CP')
    axes[1, 0].set_title('Method 1: After 1st Diff (Level Shift)')
    axes[1, 0].set_xlabel('Row')
    axes[1, 0].set_ylabel('Δ Temperature')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[2, 0].plot(trend_diff2[:500], label='2nd Difference', alpha=0.7)
    axes[2, 0].axvline(x=198, color='red', linestyle='--', label='Expected Spike')
    if trend_std > 0 and len(near_anomaly_spikes) > 0:
        for idx in near_anomaly_spikes:
            if idx < 500:
                axes[2, 0].scatter([idx], [trend_diff2[idx]], color='purple',
                                  s=100, zorder=5)
    axes[2, 0].set_title('Method 2: After 2nd Diff (Spike)')
    axes[2, 0].set_xlabel('Row')
    axes[2, 0].set_ylabel('Δ² Temperature')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Right column: Drift
    axes[0, 1].plot(golden_series[:500], label='Golden', alpha=0.7)
    axes[0, 1].plot(drift_strong[:500], label='Strong Drift', alpha=0.7)
    axes[0, 1].axvline(x=200, color='red', linestyle='--', label='Anomaly Start')
    axes[0, 1].set_title('Strong Drift (Raw Data)')
    axes[0, 1].set_xlabel('Row')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    drift_diff1 = np.diff(drift_strong)
    axes[1, 1].plot(drift_diff1[:500], label='1st Difference', alpha=0.7)
    axes[1, 1].axvline(x=199, color='red', linestyle='--', label='Anomaly Region')
    axes[1, 1].axvline(x=299, color='orange', linestyle='--')
    axes[1, 1].set_title('Drift After 1st Diff (Should Show Spike)')
    axes[1, 1].set_xlabel('Row')
    axes[1, 1].set_ylabel('Δ Temperature')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    drift_diff2 = np.diff(drift_diff1)
    axes[2, 1].plot(drift_diff2[:500], label='2nd Difference', alpha=0.7)
    axes[2, 1].axvline(x=198, color='red', linestyle='--', label='Expected Spike')
    axes[2, 1].set_title('Drift After 2nd Diff')
    axes[2, 1].set_xlabel('Row')
    axes[2, 1].set_ylabel('Δ² Temperature')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('strong_anomaly_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n*** Visualization saved to strong_anomaly_test_results.png ***")

    # Summary
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nThis test demonstrates whether the proposed methods CAN work")
    print("when the anomaly signal is strong enough.")
    print("\nIf both methods succeed with strong anomalies but fail with weak ones,")
    print("the conclusion is: the methods are sound, but your synthetic anomalies")
    print("are too subtle to survive differencing preprocessing.")

if __name__ == '__main__':
    create_strong_anomalies()
