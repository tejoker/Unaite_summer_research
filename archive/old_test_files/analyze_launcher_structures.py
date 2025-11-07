#!/usr/bin/env python3
"""
analyze_launcher_structures.py - Data-agnostic structural analysis for launcher.py results

This script analyzes structural differences between ANY launcher result directories,
without making assumptions about baseline vs variants. It compares all configurations
pairwise and generates comprehensive reports.

Usage:
    # Analyze all launcher results in directory
    python analyze_launcher_structures.py --results-dir /path/to/results

    # Specify which directory to use as reference (for comparison)
    python analyze_launcher_structures.py --results-dir /path/to/results --reference full_mi_rolling_20250929_133635

    # Custom edge threshold
    python analyze_launcher_structures.py --results-dir /path/to/results --edge-threshold 0.05
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Suppress pandas/numpy version warnings
warnings.filterwarnings('ignore')

def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger('launcher_structure_analyzer')
    logger.setLevel(log_level)

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    log_file = log_dir / 'analysis.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


class LauncherStructureAnalyzer:
    """Data-agnostic analyzer for launcher result structures"""

    def __init__(self, results_dir: Path, edge_threshold: float = 0.01,
                 output_dir: Path = None, reference_dir: str = None,
                 verbose: bool = False):

        self.results_dir = Path(results_dir)
        self.edge_threshold = edge_threshold
        self.reference_dir = reference_dir

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"launcher_structure_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.output_dir, verbose)

    def discover_launcher_directories(self) -> List[Dict[str, Any]]:
        """Discover all launcher result directories"""
        self.logger.info(f"Discovering launcher directories in {self.results_dir}")

        launcher_dirs = []

        # Look for directories with timestamp pattern (any launcher output)
        for item in self.results_dir.iterdir():
            if item.is_dir() and '_20' in item.name:  # Timestamp pattern
                # Find weight files
                weight_files = list(item.rglob("*weights*.csv"))
                weight_files = [f for f in weight_files if 'history' not in str(f)]

                if weight_files:
                    dir_info = {
                        'name': item.name,
                        'path': item,
                        'weight_file': weight_files[0],
                        'timestamp': self._extract_timestamp(item.name)
                    }
                    launcher_dirs.append(dir_info)
                    self.logger.info(f"Found: {item.name}")
                else:
                    self.logger.debug(f"Skipping {item.name} (no weight files)")

        self.logger.info(f"Discovered {len(launcher_dirs)} launcher directories")
        return sorted(launcher_dirs, key=lambda x: x['timestamp'])

    def _extract_timestamp(self, dir_name: str) -> str:
        """Extract timestamp from directory name"""
        parts = dir_name.split('_')
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # Date: YYYYMMDD
                if i + 1 < len(parts) and len(parts[i+1]) == 6:  # Time: HHMMSS
                    return f"{part}_{parts[i+1]}"
        return dir_name

    def load_weight_matrix(self, weight_file: Path):
        """Load weight matrix from CSV file using pure Python"""
        import csv
        import numpy as np

        try:
            with open(weight_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                raise ValueError("Empty CSV file")

            # Check if it's detailed format
            if 'window_idx' in rows[0]:
                # Detailed format - filter to last window, lag 0
                last_window = max(int(r['window_idx']) for r in rows)
                filtered = [r for r in rows if int(r['window_idx']) == last_window]

                if 'lag' in rows[0]:
                    filtered = [r for r in filtered if int(r['lag']) == 0]

                # Build matrix from i,j,weight columns
                n_vars = max(max(int(r['i']) for r in filtered),
                           max(int(r['j']) for r in filtered)) + 1
                matrix = np.zeros((n_vars, n_vars))

                for row in filtered:
                    i, j = int(row['i']), int(row['j'])
                    matrix[i, j] = float(row['weight'])

                return matrix
            else:
                # Direct matrix format - read as 2D array
                with open(weight_file, 'r') as f:
                    lines = f.readlines()

                matrix = []
                for line in lines:
                    values = [float(x) for x in line.strip().split(',')]
                    matrix.append(values)

                return np.array(matrix)

        except Exception as e:
            self.logger.error(f"Failed to load {weight_file}: {e}")
            raise

    def compute_adjacency_matrix(self, weight_matrix):
        """Compute adjacency matrix from weights"""
        import numpy as np
        return (np.abs(weight_matrix) > self.edge_threshold).astype(int)

    def compute_graph_metrics(self, weight_matrix, adjacency_matrix):
        """Compute graph structure metrics"""
        import numpy as np

        edge_count = adjacency_matrix.sum()
        total_possible = adjacency_matrix.shape[0] * adjacency_matrix.shape[1]
        density = edge_count / total_possible if total_possible > 0 else 0.0

        in_degree = adjacency_matrix.sum(axis=0)
        out_degree = adjacency_matrix.sum(axis=1)

        return {
            'edge_count': int(edge_count),
            'density': float(density),
            'in_degree': in_degree.tolist(),
            'out_degree': out_degree.tolist(),
            'max_weight': float(np.max(np.abs(weight_matrix))),
            'mean_weight': float(np.mean(np.abs(weight_matrix))),
            'frobenius_norm': float(np.linalg.norm(weight_matrix, 'fro')),
            'adjacency_matrix': adjacency_matrix.tolist()
        }

    def compare_structures(self, dir1_metrics, dir2_metrics, dir1_name, dir2_name):
        """Compare two graph structures"""
        import numpy as np

        adj1 = np.array(dir1_metrics['adjacency_matrix'])
        adj2 = np.array(dir2_metrics['adjacency_matrix'])

        # Edge differences
        edges1 = set(tuple(np.argwhere(adj1 == 1).tolist()[i]) for i in range((adj1 == 1).sum()))
        edges2 = set(tuple(np.argwhere(adj2 == 1).tolist()[i]) for i in range((adj2 == 1).sum()))

        added_edges = edges2 - edges1
        removed_edges = edges1 - edges2
        shd = len(added_edges) + len(removed_edges)

        # Degree changes
        in_diff = np.array(dir2_metrics['in_degree']) - np.array(dir1_metrics['in_degree'])
        out_diff = np.array(dir2_metrics['out_degree']) - np.array(dir1_metrics['out_degree'])

        return {
            'comparison': f"{dir1_name} vs {dir2_name}",
            'shd': shd,
            'added_edges': list(added_edges),
            'removed_edges': list(removed_edges),
            'in_degree_change': in_diff.tolist(),
            'out_degree_change': out_diff.tolist(),
            'edge_count_change': dir2_metrics['edge_count'] - dir1_metrics['edge_count'],
            'density_change': dir2_metrics['density'] - dir1_metrics['density']
        }

    def run_analysis(self):
        """Run complete structural analysis"""
        self.logger.info("Starting data-agnostic launcher structure analysis")

        # Discover directories
        launcher_dirs = self.discover_launcher_directories()

        if len(launcher_dirs) == 0:
            self.logger.error("No launcher directories found")
            return None

        # Load all structures
        structures = []
        for dir_info in launcher_dirs:
            self.logger.info(f"Loading structure from {dir_info['name']}")
            try:
                weight_matrix = self.load_weight_matrix(dir_info['weight_file'])
                adjacency_matrix = self.compute_adjacency_matrix(weight_matrix)
                metrics = self.compute_graph_metrics(weight_matrix, adjacency_matrix)

                structures.append({
                    'name': dir_info['name'],
                    'path': str(dir_info['path']),
                    'timestamp': dir_info['timestamp'],
                    'weight_file': str(dir_info['weight_file']),
                    'weight_matrix': weight_matrix,
                    'metrics': metrics
                })

                self.logger.info(f"  Edges: {metrics['edge_count']}, Density: {metrics['density']:.3f}")

            except Exception as e:
                self.logger.error(f"Failed to process {dir_info['name']}: {e}")
                continue

        if len(structures) == 0:
            self.logger.error("No structures successfully loaded")
            return None

        # Determine reference structure
        if self.reference_dir:
            reference_idx = next((i for i, s in enumerate(structures)
                                if self.reference_dir in s['name']), 0)
        else:
            reference_idx = 0  # Use first (earliest) as reference

        reference = structures[reference_idx]
        self.logger.info(f"Using reference: {reference['name']}")

        # Compare all structures to reference
        comparisons = []
        for i, structure in enumerate(structures):
            if i == reference_idx:
                continue

            comparison = self.compare_structures(
                reference['metrics'],
                structure['metrics'],
                reference['name'],
                structure['name']
            )
            comparisons.append({
                'target': structure['name'],
                **comparison
            })

            self.logger.info(f"Compared {structure['name']}: SHD={comparison['shd']}")

        # Generate results
        results = {
            'reference': {
                'name': reference['name'],
                'timestamp': reference['timestamp'],
                'metrics': reference['metrics']
            },
            'structures': [
                {
                    'name': s['name'],
                    'timestamp': s['timestamp'],
                    'metrics': s['metrics']
                }
                for s in structures
            ],
            'comparisons': comparisons,
            'summary': {
                'total_structures': len(structures),
                'edge_threshold': self.edge_threshold,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

        # Save results
        self._save_results(results)
        self._print_summary(results)

        return results

    def _save_results(self, results: Dict):
        """Save analysis results"""
        # Save JSON
        json_path = self.output_dir / 'structure_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Saved JSON: {json_path}")

        # Save CSV summary
        import pandas as pd
        csv_path = self.output_dir / 'structure_comparison.csv'

        rows = []
        for comp in results['comparisons']:
            rows.append({
                'target': comp['target'],
                'shd': comp['shd'],
                'added_edges': len(comp['added_edges']),
                'removed_edges': len(comp['removed_edges']),
                'edge_count_change': comp['edge_count_change'],
                'density_change': comp['density_change']
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV: {csv_path}")

    def _print_summary(self, results: Dict):
        """Print analysis summary"""
        print(f"\n{'='*70}")
        print(f"LAUNCHER STRUCTURE ANALYSIS SUMMARY")
        print(f"{'='*70}")

        ref = results['reference']
        print(f"\nReference: {ref['name']}")
        print(f"  Edges: {ref['metrics']['edge_count']}")
        print(f"  Density: {ref['metrics']['density']:.3f}")
        print(f"  Frobenius norm: {ref['metrics']['frobenius_norm']:.3f}")

        print(f"\nCompared {len(results['comparisons'])} other structures:")
        print(f"\n{'Structure':<40} {'SHD':<6} {'Added':<8} {'Removed':<8} {'ΔEdges':<8}")
        print(f"{'-'*70}")

        for comp in sorted(results['comparisons'], key=lambda x: x['shd'], reverse=True):
            name = comp['target'][:38]
            print(f"{name:<40} {comp['shd']:<6} {len(comp['added_edges']):<8} "
                  f"{len(comp['removed_edges']):<8} {comp['edge_count_change']:<+8}")

        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Data-agnostic structural analysis for launcher results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all launcher results
  python analyze_launcher_structures.py --results-dir /path/to/results

  # Use specific directory as reference
  python analyze_launcher_structures.py --results-dir ./results --reference full_mi_rolling_20250929_133635

  # Custom threshold
  python analyze_launcher_structures.py --results-dir ./results --edge-threshold 0.05 --verbose
        """
    )

    parser.add_argument('--results-dir', required=True, type=str,
                       help='Directory containing launcher result directories')
    parser.add_argument('--reference', type=str,
                       help='Reference directory name (default: earliest by timestamp)')
    parser.add_argument('--edge-threshold', type=float, default=0.01,
                       help='Threshold for edge existence (default: 0.01)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: launcher_structure_analysis_TIMESTAMP)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    try:
        analyzer = LauncherStructureAnalyzer(
            results_dir=results_dir,
            edge_threshold=args.edge_threshold,
            output_dir=args.output_dir,
            reference_dir=args.reference,
            verbose=args.verbose
        )

        results = analyzer.run_analysis()

        if results:
            print(f"\nAnalysis completed successfully!")
            return 0
        else:
            print(f"\nAnalysis failed")
            return 1

    except Exception as e:
        print(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())