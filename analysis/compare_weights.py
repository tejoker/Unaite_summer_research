#!/usr/bin/env python3
"""Compare weight matrices using ratio (Anomaly/Golden) to detect causal structure changes"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

def load_weight_data(file_path):
    """Load weight data and organize by window"""
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Windows: {df['window_idx'].nunique() if 'window_idx' in df.columns else 'N/A'}")
    return df

def create_weight_matrix(df, window_idx, lag):
    """
    Create weight matrix for specific window and lag
    Returns: DataFrame with parent nodes as rows, child nodes as columns
    """
    # Filter for specific window and lag
    window_data = df[(df['window_idx'] == window_idx) & (df['lag'] == lag)]

    if len(window_data) == 0:
        return None

    # Get unique nodes
    all_nodes = sorted(set(window_data['child_name'].unique()) | set(window_data['parent_name'].unique()))
    n_nodes = len(all_nodes)

    # Create matrix (parent -> child, so parent=row, child=col)
    matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)

    # Fill in weights
    for _, row in window_data.iterrows():
        parent = row['parent_name']
        child = row['child_name']
        weight = row['weight']
        matrix.loc[parent, child] = weight

    return matrix

def compare_weight_matrices(golden_matrix, anomaly_matrix, threshold=1e-6):
    """
    Compare two weight matrices and return differences

    Returns:
        dict with difference statistics
    """
    if golden_matrix is None or anomaly_matrix is None:
        return None

    # Align matrices to have the same index and columns (union of all nodes)
    all_nodes = sorted(set(golden_matrix.index) | set(golden_matrix.columns) |
                      set(anomaly_matrix.index) | set(anomaly_matrix.columns))

    # Reindex both matrices to have the same shape
    golden_matrix = golden_matrix.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)
    anomaly_matrix = anomaly_matrix.reindex(index=all_nodes, columns=all_nodes, fill_value=0.0)

    epsilon = 1e-10

    # Calculate ratio: Anomaly / Golden
    ratio = anomaly_matrix / (golden_matrix.abs() + epsilon)

    # Absolute difference
    diff = anomaly_matrix - golden_matrix

    # Find significant changes
    significant_change_mask = np.abs(diff) > threshold

    # Categorize changes
    new_edges = (np.abs(golden_matrix) < threshold) & (np.abs(anomaly_matrix) > threshold)
    removed_edges = (np.abs(golden_matrix) > threshold) & (np.abs(anomaly_matrix) < threshold)
    changed_edges = significant_change_mask & ~new_edges & ~removed_edges

    results = {
        'n_total': golden_matrix.shape[0] * golden_matrix.shape[1],
        'n_new_edges': new_edges.sum().sum(),
        'n_removed_edges': removed_edges.sum().sum(),
        'n_changed_edges': changed_edges.sum().sum(),
        'max_abs_diff': np.abs(diff).max().max(),
        'mean_abs_diff': np.abs(diff).mean().mean(),
        'new_edges': [(golden_matrix.index[i], golden_matrix.columns[j], anomaly_matrix.iloc[i, j])
                      for i, j in zip(*np.where(new_edges))],
        'removed_edges': [(golden_matrix.index[i], golden_matrix.columns[j], golden_matrix.iloc[i, j])
                          for i, j in zip(*np.where(removed_edges))],
        'changed_edges': [(golden_matrix.index[i], golden_matrix.columns[j],
                          golden_matrix.iloc[i, j],
                          anomaly_matrix.iloc[i, j],
                          diff.iloc[i, j])
                         for i, j in zip(*np.where(changed_edges))]
    }

    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_weights.py <golden_weights.csv> <anomaly_weights.csv> [threshold]")
        print("\nExample:")
        print("  python compare_weights.py results/Golden/weights/weights_enhanced_*.csv \\")
        print("                             results/Anomaly/weights/weights_enhanced_*.csv")
        sys.exit(1)

    golden_file = sys.argv[1]
    anomaly_file = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

    print("="*80)
    print("WEIGHT MATRIX COMPARISON - ANOMALY DETECTION")
    print("="*80)

    # Load data
    golden_df = load_weight_data(golden_file)
    anomaly_df = load_weight_data(anomaly_file)

    # Get all windows and lags
    golden_windows = sorted(golden_df['window_idx'].unique())
    golden_lags = sorted(golden_df['lag'].unique())

    anomaly_windows = sorted(anomaly_df['window_idx'].unique())
    anomaly_lags = sorted(anomaly_df['lag'].unique())

    print(f"\nGolden windows: {len(golden_windows)} (range: {golden_windows[0]}-{golden_windows[-1]})")
    print(f"Anomaly windows: {len(anomaly_windows)} (range: {anomaly_windows[0]}-{anomaly_windows[-1]})")
    print(f"Golden lags: {golden_lags}")
    print(f"Anomaly lags: {anomaly_lags}")
    print(f"Difference threshold: {threshold}")

    # Find common windows
    common_windows = sorted(set(golden_windows) & set(anomaly_windows))
    common_lags = sorted(set(golden_lags) & set(anomaly_lags))

    print(f"\nComparing {len(common_windows)} common windows across {len(common_lags)} lags")

    # Compare each window
    print("\n" + "="*80)
    print("WINDOW-BY-WINDOW COMPARISON")
    print("="*80)

    anomaly_windows_detected = []

    for window_idx in common_windows:
        window_has_anomaly = False
        window_summary = {
            'window': window_idx,
            'lags': {}
        }

        for lag in common_lags:
            golden_matrix = create_weight_matrix(golden_df, window_idx, lag)
            anomaly_matrix = create_weight_matrix(anomaly_df, window_idx, lag)

            comparison = compare_weight_matrices(golden_matrix, anomaly_matrix, threshold)

            if comparison is None:
                continue

            window_summary['lags'][lag] = comparison

            # Check if this lag shows significant differences
            if (comparison['n_new_edges'] > 0 or
                comparison['n_removed_edges'] > 0 or
                comparison['n_changed_edges'] > 0):
                window_has_anomaly = True

        if window_has_anomaly:
            anomaly_windows_detected.append(window_idx)

            print(f"\nWindow {window_idx}: ANOMALY DETECTED")
            for lag, comp in window_summary['lags'].items():
                if comp['n_new_edges'] + comp['n_removed_edges'] + comp['n_changed_edges'] > 0:
                    print(f"  Lag {lag}:")
                    print(f"    New edges:     {comp['n_new_edges']}")
                    print(f"    Removed edges: {comp['n_removed_edges']}")
                    print(f"    Changed edges: {comp['n_changed_edges']}")
                    print(f"    Max diff:      {comp['max_abs_diff']:.6f}")

                    # Show details for first few changes
                    if comp['new_edges']:
                        print(f"    New edges details (first 3):")
                        for parent, child, weight in comp['new_edges'][:3]:
                            print(f"      {parent} -> {child}: weight={weight:.6f}")

                    if comp['removed_edges']:
                        print(f"    Removed edges details (first 3):")
                        for parent, child, weight in comp['removed_edges'][:3]:
                            print(f"      {parent} -> {child}: was={weight:.6f}")

                    if comp['changed_edges']:
                        print(f"    Changed edges details (first 3):")
                        for parent, child, g_weight, a_weight, diff in comp['changed_edges'][:3]:
                            ratio = a_weight / (abs(g_weight) + 1e-10)
                            print(f"      {parent} -> {child}: {g_weight:.6f} -> {a_weight:.6f} (ratio={ratio:.3f}, diff={diff:+.6f})")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total windows compared: {len(common_windows)}")
    print(f"Windows with anomalies: {len(anomaly_windows_detected)}")
    if anomaly_windows_detected:
        print(f"Anomalous window indices: {anomaly_windows_detected}")
        print(f"Percentage anomalous: {100*len(anomaly_windows_detected)/len(common_windows):.2f}%")
    else:
        print("No anomalies detected with current threshold")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("New edges:     Causal relationships that appeared (0 -> non-zero)")
    print("Removed edges: Causal relationships that disappeared (non-zero -> 0)")
    print("Changed edges: Causal relationships with different strength")
    print("\nLarge ratios or differences indicate structural changes in the causal graph")
    print("This can reveal anomalies that manifest as changed dependencies between sensors")

if __name__ == "__main__":
    main()
