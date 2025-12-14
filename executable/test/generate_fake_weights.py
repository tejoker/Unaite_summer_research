#!/usr/bin/env python3
"""
Generates fake weight files (window_edges.npy) for testing the
Advanced Root Cause Analysis script.

This script creates two sets of weights:
1. A 'golden' baseline with a stable causal graph.
2. An 'anomalous' timeline where the graph is identical to the golden one,
   except for a specific window where edge weights are deliberately changed
   to simulate an anomaly.
"""

import numpy as np
import os

def generate_fake_weights():
    """Generates and saves fake columns and weight files."""

    output_dir = "fake_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"INFO: Generating fake data in ./{output_dir}")

    # 1. Define columns
    columns = ['SensorA', 'SensorB', 'SensorC', 'SensorD', 'SensorE']
    columns_path = os.path.join(output_dir, "columns.npy")
    np.save(columns_path, np.array(columns))
    print(f"INFO: Saved columns to {columns_path}")

    # 2. Define a base graph structure
    # A -> B -> C
    # D -> E -> C
    base_edges = [
        {'source': 'SensorA', 'target': 'SensorB', 'weight': 0.8, 'lag': 0},
        {'source': 'SensorB', 'target': 'SensorC', 'weight': 0.9, 'lag': 0},
        {'source': 'SensorD', 'target': 'SensorE', 'weight': 0.7, 'lag': 0},
        {'source': 'SensorE', 'target': 'SensorC', 'weight': 0.85, 'lag': 0},
    ]

    # 3. Generate Golden Weights
    golden_edges = []
    for i in range(5): # 5 windows
        for edge in base_edges:
            e = edge.copy()
            # Add slight noise to weights
            e['weight'] += np.random.uniform(-0.05, 0.05)
            e['window_idx'] = i
            golden_edges.append(e)
    
    golden_path = os.path.join(output_dir, "golden_weights.npy")
    np.save(golden_path, golden_edges)
    print(f"INFO: Saved golden weights ({len(golden_edges)} edges) to {golden_path}")

    # 4. Generate Anomalous Weights
    anomalous_window = 3
    anomalous_edges = []
    for i in range(5): # 5 windows
        for edge in base_edges:
            e = edge.copy()
            e['window_idx'] = i

            # Inject the anomaly in the specific window
            if i == anomalous_window:
                # Modify an existing edge significantly
                if e['source'] == 'SensorB' and e['target'] == 'SensorC':
                    e['weight'] = 0.1 # Was ~0.9
                # Add a new, spurious edge
                if e['source'] == 'SensorA' and e['target'] == 'SensorB':
                     anomalous_edges.append({
                         'source': 'SensorD',
                         'target': 'SensorB',
                         'weight': 0.95,
                         'lag': 0,
                         'window_idx': i
                     })
            
            anomalous_edges.append(e)

    anomalous_path = os.path.join(output_dir, "anomalous_weights.npy")
    np.save(anomalous_path, anomalous_edges)
    print(f"INFO: Saved anomalous weights ({len(anomalous_edges)} edges) to {anomalous_path}")
    print(f"INFO: Anomaly injected in window {anomalous_window}")
    print("      - Weight of B->C changed from ~0.9 to 0.1")
    print("      - New edge D->B created with weight 0.95")
    
    return columns_path, golden_path, anomalous_path, anomalous_window

if __name__ == '__main__':
    generate_fake_weights()
    print("\nFake data generation complete.")
    print("You can now run the advanced_rca.py script with this data.")
    print("Example:")
    print("python executable/advanced_rca.py \")
    print("    --anomalous-weights fake_results/anomalous_weights.npy \")
    print("    --golden-weights fake_results/golden_weights.npy \")
    print("    --columns fake_results/columns.npy \")
    print("    --target-node SensorC \")
    print("    --window-idx 3")
