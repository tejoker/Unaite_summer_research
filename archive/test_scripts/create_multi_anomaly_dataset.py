#!/usr/bin/env python3
"""
Create synthetic dataset with multiple anomalies at different times and sensors.

This script generates a time series dataset based on Paul Wurth bearing sensors
with 2-3 injected anomalies at different locations and times.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_baseline_data(n_samples=1000, n_sensors=5):
    """
    Generate baseline time series data for Paul Wurth bearing sensors.

    Args:
        n_samples: Number of time samples
        n_sensors: Number of sensors

    Returns:
        DataFrame with baseline sensor readings
    """
    print(f"\nGenerating baseline data: {n_samples} samples, {n_sensors} sensors")

    # Sensor names (Paul Wurth bearings)
    sensor_names = [
        'Temperatur Exzenterlager links',
        'Temperatur Exzenterlager rechts',
        'Temperatur Ständerlager links',
        'Temperatur Ständerlager rechts',
        'Temperatur Druckpfannenlager rechts'
    ][:n_sensors]

    # Generate correlated baseline signals
    # Use simple AR process with coupling
    data = np.zeros((n_samples, n_sensors))

    # Initial conditions
    data[0] = np.random.normal(80, 5, n_sensors)  # ~80°C baseline

    # AR coefficients (slight autocorrelation)
    ar_coef = 0.9

    # Coupling matrix (sensors influence each other)
    coupling = np.array([
        [0.0,  0.1,  0.05, 0.05, 0.02],  # Exzenterlager links
        [0.1,  0.0,  0.05, 0.05, 0.02],  # Exzenterlager rechts
        [0.15, 0.15, 0.0,  0.2,  0.1 ],  # Ständerlager links (main bearing)
        [0.15, 0.15, 0.2,  0.0,  0.1 ],  # Ständerlager rechts (main bearing)
        [0.05, 0.05, 0.1,  0.1,  0.0 ],  # Druckpfannenlager rechts
    ])[:n_sensors, :n_sensors]

    # Generate time series with coupling
    for t in range(1, n_samples):
        # AR component
        ar_component = ar_coef * data[t-1]

        # Coupling component
        coupling_component = coupling @ data[t-1]

        # Noise
        noise = np.random.normal(0, 1, n_sensors)

        # Combined
        data[t] = ar_component + coupling_component + noise

        # Keep temperatures in reasonable range
        data[t] = np.clip(data[t], 60, 120)

    # Create DataFrame
    df = pd.DataFrame(data, columns=sensor_names)
    df.insert(0, 'timestamp', pd.date_range('2024-01-01', periods=n_samples, freq='1min'))

    print(f"Baseline statistics:")
    print(df[sensor_names].describe())

    return df


def inject_anomaly(data, sensor_idx, anomaly_type, start_row, **kwargs):
    """
    Inject anomaly into specific sensor at specific time.

    Args:
        data: DataFrame with sensor readings
        sensor_idx: Index of sensor (0-4)
        anomaly_type: Type of anomaly (spike, drift, level_shift, etc.)
        start_row: Row to start anomaly
        **kwargs: Additional parameters (magnitude, duration, etc.)

    Returns:
        Modified DataFrame and anomaly metadata
    """
    sensor_names = [col for col in data.columns if col != 'timestamp']
    sensor_name = sensor_names[sensor_idx]

    print(f"\nInjecting {anomaly_type} into {sensor_name} at row {start_row}")

    anomaly_metadata = {
        'sensor': sensor_name,
        'sensor_idx': sensor_idx,
        'type': anomaly_type,
        'start_row': start_row,
        **kwargs
    }

    if anomaly_type == 'spike':
        # Brief spike (1 sample)
        magnitude = kwargs.get('magnitude', 30.0)
        data.loc[start_row, sensor_name] += magnitude
        anomaly_metadata['duration'] = 1
        anomaly_metadata['magnitude'] = magnitude
        print(f"  Spike: +{magnitude}°C at row {start_row}")

    elif anomaly_type == 'level_shift':
        # Persistent mean shift
        magnitude = kwargs.get('magnitude', 20.0)
        n_samples = len(data) - start_row
        data.loc[start_row:, sensor_name] += magnitude
        anomaly_metadata['duration'] = n_samples
        anomaly_metadata['magnitude'] = magnitude
        print(f"  Level shift: +{magnitude}°C from row {start_row} to end ({n_samples} samples)")

    elif anomaly_type == 'drift':
        # Gradual drift over time
        duration = kwargs.get('duration', 100)
        magnitude = kwargs.get('magnitude', 15.0)
        end_row = min(start_row + duration, len(data))
        actual_duration = end_row - start_row

        drift_signal = np.linspace(0, magnitude, actual_duration)
        data.loc[start_row:end_row-1, sensor_name] += drift_signal
        anomaly_metadata['duration'] = actual_duration
        anomaly_metadata['magnitude'] = magnitude
        print(f"  Drift: 0 → {magnitude}°C over {actual_duration} samples")

    elif anomaly_type == 'variance_burst':
        # Increased variance
        duration = kwargs.get('duration', 100)
        variance_multiplier = kwargs.get('variance_multiplier', 5.0)
        end_row = min(start_row + duration, len(data))
        actual_duration = end_row - start_row

        noise = np.random.normal(0, variance_multiplier, actual_duration)
        data.loc[start_row:end_row-1, sensor_name] += noise
        anomaly_metadata['duration'] = actual_duration
        anomaly_metadata['variance_multiplier'] = variance_multiplier
        print(f"  Variance burst: {variance_multiplier}x variance over {actual_duration} samples")

    return data, anomaly_metadata


def create_multi_anomaly_dataset(output_dir='data/MultiAnomaly'):
    """
    Create synthetic dataset with multiple anomalies.

    Scenario: Paul Wurth bearing monitoring over 1000 samples
    - Anomaly 1: Spike in Exzenterlager links at row 100
    - Anomaly 2: Level shift in Ständerlager rechts at row 500
    - Anomaly 3: Drift in Druckpfannenlager rechts at row 750
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CREATING MULTI-ANOMALY DATASET")
    print("="*80)

    # Generate baseline
    data = generate_baseline_data(n_samples=1000, n_sensors=5)

    # Save clean baseline
    baseline_path = output_path / 'baseline_clean.csv'
    data.to_csv(baseline_path, index=False)
    print(f"\nBaseline saved: {baseline_path}")

    # Copy for anomaly injection
    anomaly_data = data.copy()
    anomaly_metadata = []

    # Anomaly 1: Spike in Exzenterlager links at row 100
    anomaly_data, meta1 = inject_anomaly(
        anomaly_data,
        sensor_idx=0,  # Exzenterlager links
        anomaly_type='spike',
        start_row=100,
        magnitude=30.0
    )
    anomaly_metadata.append(meta1)

    # Anomaly 2: Level shift in Ständerlager rechts at row 500
    anomaly_data, meta2 = inject_anomaly(
        anomaly_data,
        sensor_idx=3,  # Ständerlager rechts
        anomaly_type='level_shift',
        start_row=500,
        magnitude=20.0
    )
    anomaly_metadata.append(meta2)

    # Anomaly 3: Drift in Druckpfannenlager rechts at row 750
    anomaly_data, meta3 = inject_anomaly(
        anomaly_data,
        sensor_idx=4,  # Druckpfannenlager rechts
        anomaly_type='drift',
        start_row=750,
        duration=150,
        magnitude=15.0
    )
    anomaly_metadata.append(meta3)

    # Save anomaly dataset
    anomaly_path = output_path / 'multi_anomaly_data.csv'
    anomaly_data.to_csv(anomaly_path, index=False)
    print(f"\nAnomaly data saved: {anomaly_path}")

    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'n_samples': len(data),
        'n_sensors': len([c for c in data.columns if c != 'timestamp']),
        'n_anomalies': len(anomaly_metadata),
        'anomalies': anomaly_metadata
    }

    metadata_path = output_path / 'multi_anomaly_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    # Print summary
    print("\n" + "="*80)
    print("MULTI-ANOMALY DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(data)}")
    print(f"Number of sensors: {len([c for c in data.columns if c != 'timestamp'])}")
    print(f"Number of anomalies: {len(anomaly_metadata)}")
    print("\nAnomaly Timeline:")
    for i, meta in enumerate(anomaly_metadata, 1):
        print(f"\n  Anomaly #{i}:")
        print(f"    Type: {meta['type']}")
        print(f"    Sensor: {meta['sensor']}")
        print(f"    Start row: {meta['start_row']}")
        print(f"    Duration: {meta.get('duration', 'N/A')} samples")
        print(f"    Magnitude: {meta.get('magnitude', 'N/A')}°C")

    print("\n" + "="*80)
    print("Expected DynoTEARS Windows (window_size=100, stride=10):")
    print("="*80)
    for i, meta in enumerate(anomaly_metadata, 1):
        start_row = meta['start_row']
        # Window index = (start_row - window_size + stride) / stride
        # Approximate window where anomaly first appears
        first_window = max(1, (start_row - 100) // 10 + 1)
        print(f"  Anomaly #{i} ({meta['type']} in {meta['sensor']})")
        print(f"    Expected first detection: Window ~{first_window}")

    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print(f"1. Run DynoTEARS on baseline: {baseline_path}")
    print(f"2. Run DynoTEARS on anomaly data: {anomaly_path}")
    print(f"3. Compare causal graphs to detect anomalies")
    print(f"4. Test gap-based detection (should find 3 separate anomalies)")
    print("="*80)

    return {
        'baseline_path': str(baseline_path),
        'anomaly_path': str(anomaly_path),
        'metadata_path': str(metadata_path),
        'anomalies': anomaly_metadata
    }


if __name__ == '__main__':
    result = create_multi_anomaly_dataset()

    print("\n\nDataset created successfully!")
    print(f"Baseline: {result['baseline_path']}")
    print(f"Anomaly data: {result['anomaly_path']}")
    print(f"Metadata: {result['metadata_path']}")
