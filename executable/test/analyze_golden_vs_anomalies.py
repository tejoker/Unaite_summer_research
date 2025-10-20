#!/usr/bin/env python3
"""
analyze_golden_vs_anomalies.py - Compare Golden baseline structure vs Anomaly structures

This answers the critical question: Do anomalies change graph STRUCTURE or just WEIGHTS?
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def load_weight_matrix(weight_file: Path, edge_threshold: float = 0.01):
    """Load weight matrix and compute adjacency"""
    with open(weight_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Empty CSV file")

    # Detect format
    if 'window_idx' in rows[0]:
        # Rolling window format
        windows = set(int(r['window_idx']) for r in rows)
        last_window = max(windows)

        # Filter to last window and any lag (take max absolute weight per edge)
        edges = {}
        for row in rows:
            if int(row['window_idx']) == last_window:
                i, j = int(row['i']), int(row['j'])
                weight = float(row['weight'])
                key = (i, j)

                if key not in edges or abs(weight) > abs(edges[key]):
                    edges[key] = weight
    elif 'parent' in rows[0] and 'child' in rows[0]:
        # Global format (parent/child names)
        var_names = set()
        for row in rows:
            var_names.add(row['parent'])
            var_names.add(row['child'])

        var_to_idx = {name: i for i, name in enumerate(sorted(var_names))}

        edges = {}
        for row in rows:
            parent_idx = var_to_idx[row['parent']]
            child_idx = var_to_idx[row['child']]
            weight = float(row['weight'])
            key = (parent_idx, child_idx)

            if key not in edges or abs(weight) > abs(edges[key]):
                edges[key] = weight
    else:
        raise ValueError(f"Unknown format: {list(rows[0].keys())}")

    # Determine matrix size
    if edges:
        n_vars = max(max(i for i, j in edges.keys()), max(j for i, j in edges.keys())) + 1
    else:
        n_vars = 6  # Default

    # Build adjacency and metrics
    adjacency = {}
    weights = {}
    for i in range(n_vars):
        for j in range(n_vars):
            key = (i, j)
            weight = edges.get(key, 0.0)
            weights[key] = weight
            adjacency[key] = 1 if abs(weight) > edge_threshold else 0

    edge_list = [(i, j) for (i, j), v in adjacency.items() if v == 1]
    edge_count = len(edge_list)
    density = edge_count / (n_vars * n_vars) if n_vars > 0 else 0.0

    in_degree = [sum(adjacency[(i, j)] for i in range(n_vars)) for j in range(n_vars)]
    out_degree = [sum(adjacency[(i, j)] for j in range(n_vars)) for i in range(n_vars)]

    abs_weights = [abs(w) for w in weights.values()]
    max_weight = max(abs_weights) if abs_weights else 0.0
    mean_weight = sum(abs_weights) / len(abs_weights) if abs_weights else 0.0
    frobenius = sum(w*w for w in weights.values()) ** 0.5

    return {
        'n_vars': n_vars,
        'edge_count': edge_count,
        'edge_list': edge_list,
        'density': density,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'max_weight': max_weight,
        'mean_weight': mean_weight,
        'frobenius_norm': frobenius,
        'adjacency': adjacency,
        'weights': weights
    }


def compare_structures(ref_metrics, target_metrics, ref_name, target_name):
    """Compare two structures"""
    ref_edges = set(ref_metrics['edge_list'])
    target_edges = set(target_metrics['edge_list'])

    added = list(target_edges - ref_edges)
    removed = list(ref_edges - target_edges)
    common = list(ref_edges & target_edges)

    shd = len(added) + len(removed)

    # Weight changes on common edges
    weight_changes = []
    if common:
        for edge in common:
            ref_w = ref_metrics['weights'][edge]
            target_w = target_metrics['weights'][edge]
            weight_changes.append({
                'edge': f"({edge[0]},{edge[1]})",
                'ref_weight': ref_w,
                'target_weight': target_w,
                'abs_change': abs(target_w - ref_w),
                'rel_change': abs(target_w - ref_w) / abs(ref_w) if abs(ref_w) > 1e-10 else 0
            })

    avg_weight_change = sum(w['abs_change'] for w in weight_changes) / len(weight_changes) if weight_changes else 0.0

    return {
        'comparison': f"{ref_name} vs {target_name}",
        'shd': shd,
        'added_edges': added,
        'removed_edges': removed,
        'common_edges': common,
        'num_added': len(added),
        'num_removed': len(removed),
        'num_common': len(common),
        'edge_count_change': target_metrics['edge_count'] - ref_metrics['edge_count'],
        'density_change': target_metrics['density'] - ref_metrics['density'],
        'frobenius_change': target_metrics['frobenius_norm'] - ref_metrics['frobenius_norm'],
        'weight_changes': weight_changes,
        'avg_weight_change': avg_weight_change
    }


def extract_anomaly_type(dir_name: str) -> str:
    """Extract anomaly type from directory name"""
    anomaly_types = ['spike', 'drift', 'level_shift', 'amplitude_change',
                     'trend_change', 'variance_burst', 'missing_block']

    dir_lower = dir_name.lower()
    for atype in anomaly_types:
        if atype in dir_lower:
            return atype

    return 'unknown'


def discover_golden_and_anomalies(results_dir: Path, config_pattern: str = 'full_mi_rolling'):
    """Find Golden baseline and all anomaly results for a specific configuration"""

    golden = None
    anomalies = []

    # Find directories matching configuration pattern
    config_dirs = [d for d in results_dir.iterdir()
                   if d.is_dir() and config_pattern in d.name]

    for config_dir in config_dirs:
        # Check for Golden subdirectory
        golden_dir = config_dir / 'Golden'
        if golden_dir.exists():
            weight_files = list(golden_dir.rglob('*weights*.csv'))
            weight_files = [f for f in weight_files if 'history' not in str(f)]
            if weight_files:
                golden = {
                    'name': f"{config_dir.name}/Golden",
                    'path': golden_dir,
                    'weight_file': weight_files[0],
                    'config_dir': config_dir.name
                }

        # Check for Anomaly subdirectory
        anomaly_dir = config_dir / 'Anomaly'
        if anomaly_dir.exists():
            # Each subdirectory is a different anomaly
            for anomaly_subdir in anomaly_dir.iterdir():
                if anomaly_subdir.is_dir():
                    weight_files = list(anomaly_subdir.rglob('*weights*.csv'))
                    weight_files = [f for f in weight_files if 'history' not in str(f)]
                    if weight_files:
                        anomaly_type = extract_anomaly_type(anomaly_subdir.name)
                        anomalies.append({
                            'name': f"{config_dir.name}/Anomaly/{anomaly_type}",
                            'path': anomaly_subdir,
                            'weight_file': weight_files[0],
                            'anomaly_type': anomaly_type,
                            'config_dir': config_dir.name
                        })

    return golden, anomalies


def main():
    parser = argparse.ArgumentParser(
        description='Compare Golden baseline vs Anomaly structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script answers: Do anomalies change graph STRUCTURE or just WEIGHTS?

Examples:
  python3 analyze_golden_vs_anomalies.py --results-dir ../../results
  python3 analyze_golden_vs_anomalies.py --results-dir ../../results --config no_mi_rolling
  python3 analyze_golden_vs_anomalies.py --results-dir ../../results --edge-threshold 0.05
        """
    )

    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--config', default='full_mi_rolling',
                       help='Configuration pattern to analyze (default: full_mi_rolling)')
    parser.add_argument('--edge-threshold', type=float, default=0.01,
                       help='Edge threshold (default: 0.01)')
    parser.add_argument('--output', default='golden_vs_anomalies.json',
                       help='Output file')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        return 1

    print(f"Analyzing configuration: {args.config}")
    print(f"Edge threshold: {args.edge_threshold}")
    print()

    # Discover Golden and Anomalies
    golden, anomalies = discover_golden_and_anomalies(results_dir, args.config)

    if not golden:
        print(f"ERROR: No Golden baseline found for configuration '{args.config}'")
        print(f"Available directories:")
        for d in results_dir.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return 1

    if not anomalies:
        print(f"WARNING: No anomalies found for configuration '{args.config}'")
        return 1

    print(f"Found Golden baseline: {golden['name']}")
    print(f"Found {len(anomalies)} anomaly results:")
    for a in anomalies:
        print(f"  - {a['anomaly_type']}")
    print()

    # Load Golden structure
    print("Loading Golden baseline structure...")
    try:
        golden_metrics = load_weight_matrix(golden['weight_file'], args.edge_threshold)
        print(f"  Golden: {golden_metrics['edge_count']} edges, density={golden_metrics['density']:.3f}")
    except Exception as e:
        print(f"ERROR loading Golden: {e}")
        return 1

    # Load and compare each anomaly
    print("\nAnalyzing anomalies...")
    comparisons = []

    for anomaly in anomalies:
        print(f"\n  {anomaly['anomaly_type']}...", end=' ')
        try:
            anomaly_metrics = load_weight_matrix(anomaly['weight_file'], args.edge_threshold)
            print(f"edges={anomaly_metrics['edge_count']}, density={anomaly_metrics['density']:.3f}")

            comparison = compare_structures(
                golden_metrics,
                anomaly_metrics,
                'Golden',
                anomaly['anomaly_type']
            )

            comparisons.append({
                'anomaly_type': anomaly['anomaly_type'],
                'anomaly_name': anomaly['name'],
                **comparison
            })

            print(f"    SHD={comparison['shd']} (+{comparison['num_added']} edges, -{comparison['num_removed']} edges)")
            print(f"    Avg weight change on common edges: {comparison['avg_weight_change']:.4f}")

        except Exception as e:
            print(f"FAILED: {e}")

    if not comparisons:
        print("\nNo successful comparisons")
        return 1

    # Compute statistics
    shd_values = [c['shd'] for c in comparisons]
    avg_shd = sum(shd_values) / len(shd_values)
    max_shd = max(shd_values)
    min_shd = min(shd_values)

    structure_stable = avg_shd < 5

    # Results
    results = {
        'golden': {
            'name': golden['name'],
            'config': golden['config_dir'],
            'edge_count': golden_metrics['edge_count'],
            'density': golden_metrics['density'],
            'frobenius_norm': golden_metrics['frobenius_norm']
        },
        'anomalies': comparisons,
        'statistics': {
            'total_anomalies': len(comparisons),
            'avg_shd': avg_shd,
            'min_shd': min_shd,
            'max_shd': max_shd,
            'structure_stable': structure_stable,
            'edge_threshold': args.edge_threshold
        },
        'recommendation': {
            'structure_stable': structure_stable,
            'optimization_strategy': 'fixed_structure' if structure_stable else 'two_stage',
            'expected_speedup': '50-100x' if structure_stable else '5-10x'
        }
    }

    # Save
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("GOLDEN VS ANOMALIES ANALYSIS SUMMARY")
    print(f"{'='*80}")

    print(f"\nConfiguration: {args.config}")
    print(f"Golden Baseline:")
    print(f"  Edges: {golden_metrics['edge_count']}")
    print(f"  Density: {golden_metrics['density']:.3f}")

    print(f"\nStructural Changes Across {len(comparisons)} Anomalies:")
    print(f"  Average SHD: {avg_shd:.1f}")
    print(f"  Min SHD: {min_shd}")
    print(f"  Max SHD: {max_shd}")

    print(f"\n{'Anomaly Type':<20} {'SHD':<6} {'Added':<8} {'Removed':<8} {'Avg ΔWeight':<12}")
    print("-" * 80)
    for comp in sorted(comparisons, key=lambda x: x['shd'], reverse=True):
        print(f"{comp['anomaly_type']:<20} {comp['shd']:<6} {comp['num_added']:<8} "
              f"{comp['num_removed']:<8} {comp['avg_weight_change']:<12.4f}")

    print(f"\n{'='*80}")
    if structure_stable:
        print("CONCLUSION: Structure is STABLE (avg SHD < 5)")
        print("  → Anomalies primarily change WEIGHTS, not STRUCTURE")
        print("  → Recommendation: Use FIXED-STRUCTURE optimization")
        print("  → Expected speedup: 50-100x")
        print("  → Strategy: Skip DYNOTEARS, use linear regression on Golden structure")
    else:
        print("CONCLUSION: Structure CHANGES with anomalies (avg SHD >= 5)")
        print("  → Anomalies alter graph STRUCTURE")
        print("  → Recommendation: Use TWO-STAGE detection")
        print("  → Expected speedup: 5-10x")
        print("  → Strategy: Fast weight check, then full DYNOTEARS only when needed")
    print(f"{'='*80}")

    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())