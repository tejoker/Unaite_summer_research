#!/usr/bin/env python3
"""
Create multi-anomaly dataset using REAL Golden data as baseline.

This version uses the actual Golden dataset and injects anomalies,
avoiding the clipping artifacts from synthetic data generation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def inject_anomaly(data, sensor_name, anomaly_type, start_row, **kwargs):
    """
    Inject anomaly into specific sensor at specific time.

    Args:
        data: DataFrame with sensor readings
        sensor_name: Name of sensor column
        anomaly_type: Type of anomaly
        start_row: Row to start anomaly
        **kwargs: Additional parameters

    Returns:
        Modified DataFrame and anomaly metadata
    """
    print(f"\nInjecting {anomaly_type} into {sensor_name} at row {start_row}")

    anomaly_metadata = {
        'sensor': sensor_name,
        'type': anomaly_type,
        'start_row': start_row,
        **kwargs
    }

    if anomaly_type == 'spike':
        magnitude = kwargs.get('magnitude', 30.0)
        data.loc[start_row, sensor_name] += magnitude
        anomaly_metadata['duration'] = 1
        anomaly_metadata['magnitude'] = magnitude
        print(f"  Spike: +{magnitude}°C at row {start_row}")

    elif anomaly_type == 'level_shift':
        magnitude = kwargs.get('magnitude', 20.0)
        n_samples = len(data) - start_row
        data.loc[start_row:, sensor_name] += magnitude
        anomaly_metadata['duration'] = n_samples
        anomaly_metadata['magnitude'] = magnitude
        print(f"  Level shift: +{magnitude}°C from row {start_row} to end ({n_samples} samples)")

    elif anomaly_type == 'drift':
        duration = kwargs.get('duration', 100)
        magnitude = kwargs.get('magnitude', 15.0)
        end_row = min(start_row + duration, len(data))
        actual_duration = end_row - start_row

        drift_signal = np.linspace(0, magnitude, actual_duration)
        data.loc[start_row:end_row-1, sensor_name] += drift_signal
        anomaly_metadata['duration'] = actual_duration
        anomaly_metadata['magnitude'] = magnitude
        print(f"  Drift: 0 → {magnitude}°C over {actual_duration} samples")

    return data, anomaly_metadata


def create_multi_anomaly_dataset_v2(output_dir='data/MultiAnomalyV2'):
    """
    Create multi-anomaly dataset using real Golden data as baseline.

    Scenario:
    - Baseline: Real Golden dataset (output_of_the_1th_chunk.csv)
    - Anomaly 1: Spike in Exzenterlager links at row 100
    - Anomaly 2: Level shift in Ständerlager rechts at row 500
    - Anomaly 3: Drift in Druckpfannenlager rechts at row 750
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CREATING MULTI-ANOMALY DATASET V2 (Real Golden Baseline)")
    print("="*80)

    # Load real Golden dataset
    golden_path = Path('data/Golden/chunking/output_of_the_1th_chunk.csv')
    if not golden_path.exists():
        print(f"ERROR: Golden dataset not found: {golden_path}")
        return None

    baseline_data = pd.read_csv(golden_path)
    print(f"\nLoaded Golden baseline: {len(baseline_data)} samples")

    # Get sensor columns (exclude timestamp)
    sensor_cols = [col for col in baseline_data.columns if 'Temperatur' in col or 'emperatur' in col]
    print(f"Sensors found: {len(sensor_cols)}")
    for col in sensor_cols:
        print(f"  - {col}")

    print("\nBaseline statistics:")
    print(baseline_data[sensor_cols].describe())

    # Copy for anomaly injection
    anomaly_data = baseline_data.copy()
    anomaly_metadata = []

    # Anomaly 1: Spike in Exzenterlager links at row 100
    if 'Temperatur Exzenterlager links' in sensor_cols:
        anomaly_data, meta1 = inject_anomaly(
            anomaly_data,
            sensor_name='Temperatur Exzenterlager links',
            anomaly_type='spike',
            start_row=100,
            magnitude=30.0
        )
        anomaly_metadata.append(meta1)

    # Anomaly 2: Level shift in Ständerlager rechts at row 500
    standerlager_col = [c for c in sensor_cols if 'tänderlager rechts' in c or 'Ständerlager rechts' in c]
    if standerlager_col:
        anomaly_data, meta2 = inject_anomaly(
            anomaly_data,
            sensor_name=standerlager_col[0],
            anomaly_type='level_shift',
            start_row=500,
            magnitude=20.0
        )
        anomaly_metadata.append(meta2)

    # Anomaly 3: Drift in Druckpfannenlager rechts at row 750
    druckpfannen_col = [c for c in sensor_cols if 'Druckpfannenlager rechts' in c]
    if druckpfannen_col:
        anomaly_data, meta3 = inject_anomaly(
            anomaly_data,
            sensor_name=druckpfannen_col[0],
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
        'baseline_source': str(golden_path),
        'n_samples': len(baseline_data),
        'n_sensors': len(sensor_cols),
        'n_anomalies': len(anomaly_metadata),
        'anomalies': anomaly_metadata
    }

    metadata_path = output_path / 'multi_anomaly_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    # Print summary
    print("\n" + "="*80)
    print("MULTI-ANOMALY DATASET V2 SUMMARY")
    print("="*80)
    print(f"Baseline: Real Golden dataset ({len(baseline_data)} samples)")
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
        first_window = max(0, (start_row - 100) // 10 + 1)
        print(f"  Anomaly #{i} ({meta['type']} in {meta['sensor']})")
        print(f"    Expected first detection: Window ~{first_window}")

    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print("1. Run DynoTEARS on anomaly data:")
    print(f"   python3 executable/launcher.py --data {anomaly_path} --output results/multi_anomaly_v2/anomaly_run")
    print()
    print("2. Test gap-based detection:")
    print("   (Use existing Golden weights as baseline)")
    print("="*80)

    return {
        'anomaly_path': str(anomaly_path),
        'metadata_path': str(metadata_path),
        'anomalies': anomaly_metadata
    }


if __name__ == '__main__':
    result = create_multi_anomaly_dataset_v2()

    if result:
        print("\n\nDataset created successfully!")
        print(f"Anomaly data: {result['anomaly_path']}")
        print(f"Metadata: {result['metadata_path']}")
