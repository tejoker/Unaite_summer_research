#!/usr/bin/env python3
"""
analyze_launcher_simple.py - Simple structural analysis without numpy dependency

Analyzes graph structures from launcher results using only Python standard library.
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
        # Build variable name to index mapping
        var_names = set()
        for row in rows:
            var_names.add(row['parent'])
            var_names.add(row['child'])

        var_to_idx = {name: i for i, name in enumerate(sorted(var_names))}

        # Build edge dictionary
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

    edge_count = sum(adjacency.values())
    density = edge_count / (n_vars * n_vars) if n_vars > 0 else 0.0

    # Degree calculations
    in_degree = [sum(adjacency[(i, j)] for i in range(n_vars)) for j in range(n_vars)]
    out_degree = [sum(adjacency[(i, j)] for j in range(n_vars)) for i in range(n_vars)]

    # Weight stats
    abs_weights = [abs(w) for w in weights.values()]
    max_weight = max(abs_weights) if abs_weights else 0.0
    mean_weight = sum(abs_weights) / len(abs_weights) if abs_weights else 0.0

    frobenius = sum(w*w for w in weights.values()) ** 0.5

    return {
        'n_vars': n_vars,
        'edge_count': edge_count,
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
    ref_adj = ref_metrics['adjacency']
    target_adj = target_metrics['adjacency']

    # Find edge differences
    added = [(i, j) for (i, j), v in target_adj.items() if v == 1 and ref_adj.get((i, j), 0) == 0]
    removed = [(i, j) for (i, j), v in ref_adj.items() if v == 1 and target_adj.get((i, j), 0) == 0]

    shd = len(added) + len(removed)

    # Degree changes
    in_change = [target_metrics['in_degree'][i] - ref_metrics['in_degree'][i]
                for i in range(ref_metrics['n_vars'])]
    out_change = [target_metrics['out_degree'][i] - ref_metrics['out_degree'][i]
                 for i in range(ref_metrics['n_vars'])]

    return {
        'comparison': f"{ref_name} vs {target_name}",
        'shd': shd,
        'added_edges': len(added),
        'removed_edges': len(removed),
        'edge_count_change': target_metrics['edge_count'] - ref_metrics['edge_count'],
        'density_change': target_metrics['density'] - ref_metrics['density'],
        'frobenius_change': target_metrics['frobenius_norm'] - ref_metrics['frobenius_norm']
    }


def discover_launcher_dirs(results_dir: Path):
    """Find all launcher result directories"""
    dirs = []

    for item in results_dir.iterdir():
        if item.is_dir() and '_20' in item.name:  # Timestamp pattern
            # Find weight files
            weight_files = list(item.rglob("*weights*.csv"))
            weight_files = [f for f in weight_files if 'history' not in str(f)]

            if weight_files:
                dirs.append({
                    'name': item.name,
                    'path': item,
                    'weight_file': weight_files[0]
                })

    return sorted(dirs, key=lambda x: x['name'])


def main():
    parser = argparse.ArgumentParser(description='Simple launcher structure analysis')
    parser.add_argument('--results-dir', required=True, help='Results directory')
    parser.add_argument('--edge-threshold', type=float, default=0.01, help='Edge threshold')
    parser.add_argument('--reference', help='Reference directory name')
    parser.add_argument('--output', default='launcher_analysis.json', help='Output file')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        return 1

    print("Discovering launcher directories...")
    launcher_dirs = discover_launcher_dirs(results_dir)
    print(f"Found {len(launcher_dirs)} directories")

    if not launcher_dirs:
        print("No launcher directories found")
        return 1

    # Load all structures
    print("\nLoading structures...")
    structures = []
    for dir_info in launcher_dirs:
        print(f"  {dir_info['name'][:50]}...", end=' ')
        try:
            metrics = load_weight_matrix(dir_info['weight_file'], args.edge_threshold)
            structures.append({
                'name': dir_info['name'],
                'metrics': metrics
            })
            print(f"OK (edges={metrics['edge_count']}, density={metrics['density']:.3f})")
        except Exception as e:
            print(f"FAILED: {e}")

    if not structures:
        print("\nNo structures loaded successfully")
        return 1

    # Determine reference
    if args.reference:
        ref_idx = next((i for i, s in enumerate(structures) if args.reference in s['name']), 0)
    else:
        ref_idx = 0

    reference = structures[ref_idx]
    print(f"\nUsing reference: {reference['name']}")

    # Compare all to reference
    print("\nComparing structures...")
    comparisons = []
    for i, structure in enumerate(structures):
        if i == ref_idx:
            continue

        comp = compare_structures(
            reference['metrics'],
            structure['metrics'],
            reference['name'],
            structure['name']
        )
        comparisons.append({
            'target': structure['name'],
            **comp
        })

    # Results
    results = {
        'reference': {
            'name': reference['name'],
            'edge_count': reference['metrics']['edge_count'],
            'density': reference['metrics']['density'],
            'frobenius_norm': reference['metrics']['frobenius_norm']
        },
        'comparisons': comparisons,
        'summary': {
            'total_structures': len(structures),
            'edge_threshold': args.edge_threshold,
            'timestamp': datetime.now().isoformat()
        }
    }

    # Save
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("STRUCTURAL ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"\nReference: {reference['name']}")
    print(f"  Edges: {reference['metrics']['edge_count']}")
    print(f"  Density: {reference['metrics']['density']:.3f}")

    print(f"\n{'Structure':<45} {'SHD':<6} {'+Edges':<8} {'-Edges':<8} {'ΔDensity':<10}")
    print("-" * 70)
    for comp in sorted(comparisons, key=lambda x: x['shd'], reverse=True):
        name = comp['target'][:43]
        print(f"{name:<45} {comp['shd']:<6} {comp['added_edges']:<8} "
              f"{comp['removed_edges']:<8} {comp['density_change']:<+10.3f}")

    print(f"\nResults saved to: {output_file}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())