#!/usr/bin/env python3
"""
Visualize and compare windows 200 vs 761
"""

import numpy as np
import pandas as pd
import json

def load_window_weights(path, window_idx):
    """Load weight matrix for specific window"""
    df = pd.read_csv(path)
    w_df = df[(df['window_idx'] == window_idx) & (df['lag'] == 0)]
    d = 6
    W = np.zeros((d, d))
    for _, row in w_df.iterrows():
        W[int(row['i']), int(row['j'])] = row['weight']
    return W

# Variable names
var_names = [
    'Temp_Druckpfanne_R',
    'Temp_Exzenter_L',
    'Temp_Exzenter_R',
    'Temp_Ständer_L',
    'Temp_Ständer_R',
    'Temp_Druckpfanne_L'
]

print("="*80)
print("WINDOW-BY-WINDOW COMPARISON: Window 200 vs Window 761")
print("="*80)

# Load weights
baseline_200 = load_window_weights('results/Golden_NoMI/weights/weights_enhanced_20251007_145558.csv', 200)
spike_200 = load_window_weights('results/Spike_NoMI/weights/weights_enhanced_20251007_153044.csv', 200)
baseline_761 = load_window_weights('results/Golden_NoMI/weights/weights_enhanced_20251007_145558.csv', 761)
spike_761 = load_window_weights('results/Spike_NoMI/weights/weights_enhanced_20251007_153044.csv', 761)

# Load results
with open('results/window_by_window_spike_all/window_by_window_results.json') as f:
    data = json.load(f)
    results = data['window_results']
    w200 = [r for r in results if r['window_idx'] == 200][0]
    w761 = [r for r in results if r['window_idx'] == 761][0]

print("\n" + "="*80)
print("WINDOW 200 (Your injected spike at t=200)")
print("="*80)
print(f"Timepoint range: [200, 300)")
print(f"Ensemble score: {w200['phase1_binary_detection']['binary_detection']['ensemble_score']:.2f}")
print(f"Classification: {w200['phase2_classification']['rule_based_result']['prediction']}")
print(f"Confidence: {w200['phase2_classification']['rule_based_result']['confidence']:.2%}")

print("\nMetrics:")
metrics = w200['phase1_binary_detection']['binary_detection']['metrics_raw']
print(f"  Frobenius distance: {metrics['frobenius_distance']:.4f}")
print(f"  Structural Hamming: {metrics['structural_hamming_distance']}")
print(f"  Spectral distance: {metrics['spectral_distance']:.4f}")
print(f"  Max edge change: {metrics['max_edge_change']:.4f}")

print("\nSignature features:")
sig200 = w200['phase2_classification']['signature']
print(f"  Magnitude max change: {sig200['magnitude_max_change']:.4f}")
print(f"  Magnitude mean change: {sig200['magnitude_mean_change']:.4f}")
print(f"  Structural edges added: {sig200['structural_edges_added']}")
print(f"  Structural edges removed: {sig200['structural_edges_removed']}")
print(f"  Structural density change: {sig200['structural_density_change']:.4f}")
print(f"  Sign flips: {sig200['sign_flips']}")

# Edge analysis for window 200
diff_200 = spike_200 - baseline_200
print("\nTop 5 edge changes (absolute value):")
edge_changes_200 = []
for i in range(6):
    for j in range(6):
        if i != j:
            edge_changes_200.append((abs(diff_200[i, j]), i, j, diff_200[i, j]))
edge_changes_200.sort(reverse=True)
for idx, (abs_change, i, j, change) in enumerate(edge_changes_200[:5]):
    print(f"  {idx+1}. {var_names[i]} → {var_names[j]}: {change:+.4f} (baseline={baseline_200[i,j]:.4f}, spike={spike_200[i,j]:.4f})")

print("\n" + "="*80)
print("WINDOW 761 (Highest spike score)")
print("="*80)
print(f"Timepoint range: [761, 861)")
print(f"Ensemble score: {w761['phase1_binary_detection']['binary_detection']['ensemble_score']:.2f}")
print(f"Classification: {w761['phase2_classification']['rule_based_result']['prediction']}")
print(f"Confidence: {w761['phase2_classification']['rule_based_result']['confidence']:.2%}")

print("\nMetrics:")
metrics = w761['phase1_binary_detection']['binary_detection']['metrics_raw']
print(f"  Frobenius distance: {metrics['frobenius_distance']:.4f}")
print(f"  Structural Hamming: {metrics['structural_hamming_distance']}")
print(f"  Spectral distance: {metrics['spectral_distance']:.4f}")
print(f"  Max edge change: {metrics['max_edge_change']:.4f}")

print("\nSignature features:")
sig761 = w761['phase2_classification']['signature']
print(f"  Magnitude max change: {sig761['magnitude_max_change']:.4f}")
print(f"  Magnitude mean change: {sig761['magnitude_mean_change']:.4f}")
print(f"  Structural edges added: {sig761['structural_edges_added']}")
print(f"  Structural edges removed: {sig761['structural_edges_removed']}")
print(f"  Structural density change: {sig761['structural_density_change']:.4f}")
print(f"  Sign flips: {sig761['sign_flips']}")

# Edge analysis for window 761
diff_761 = spike_761 - baseline_761
print("\nTop 5 edge changes (absolute value):")
edge_changes_761 = []
for i in range(6):
    for j in range(6):
        if i != j:
            edge_changes_761.append((abs(diff_761[i, j]), i, j, diff_761[i, j]))
edge_changes_761.sort(reverse=True)
for idx, (abs_change, i, j, change) in enumerate(edge_changes_761[:5]):
    print(f"  {idx+1}. {var_names[i]} → {var_names[j]}: {change:+.4f} (baseline={baseline_761[i,j]:.4f}, spike={spike_761[i,j]:.4f})")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"{'Metric':<30} {'Window 200':>15} {'Window 761':>15}")
print("-"*62)
print(f"{'Ensemble Score':<30} {w200['phase1_binary_detection']['binary_detection']['ensemble_score']:>15.2f} {w761['phase1_binary_detection']['binary_detection']['ensemble_score']:>15.2f}")
print(f"{'Frobenius Distance':<30} {w200['phase1_binary_detection']['binary_detection']['metrics_raw']['frobenius_distance']:>15.4f} {w761['phase1_binary_detection']['binary_detection']['metrics_raw']['frobenius_distance']:>15.4f}")
print(f"{'Structural Hamming':<30} {w200['phase1_binary_detection']['binary_detection']['metrics_raw']['structural_hamming_distance']:>15.0f} {w761['phase1_binary_detection']['binary_detection']['metrics_raw']['structural_hamming_distance']:>15.0f}")
print(f"{'Max Edge Change':<30} {sig200['magnitude_max_change']:>15.4f} {sig761['magnitude_max_change']:>15.4f}")
print(f"{'Edges Added':<30} {sig200['structural_edges_added']:>15.0f} {sig761['structural_edges_added']:>15.0f}")
print(f"{'Edges Removed':<30} {sig200['structural_edges_removed']:>15.0f} {sig761['structural_edges_removed']:>15.0f}")
print(f"{'Density Change':<30} {sig200['structural_density_change']:>15.4f} {sig761['structural_density_change']:>15.4f}")
print(f"{'Sign Flips':<30} {sig200['sign_flips']:>15.0f} {sig761['sign_flips']:>15.0f}")

print("\n" + "="*80)
print("WHY Window 761 Scored Higher Despite Smaller Changes:")
print("="*80)
print("Window 200:")
print("  - LARGER magnitude changes (Frobenius=9.83 vs 3.70)")
print("  - REMOVED 3 edges (30→27) = negative density change")
print("  - Failed 'spike' criteria: needs edge addition, not removal")
print("  - Classified as 'trend_change' instead")
print("")
print("Window 761:")
print("  - Smaller magnitude changes BUT")
print("  - ADDED 4 edges (26→30) = positive density change")
print("  - Passed 'spike' criteria: edges_added=4 > 2 threshold")
print("  - Correctly classified as 'spike'")
print("")
print("RECOMMENDATION:")
print("  Update spike detection rules to allow edge removal")
print("  (spikes can strengthen some edges while suppressing others)")
