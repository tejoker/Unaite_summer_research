#!/usr/bin/env python3
"""
Causal Cascade Detector

Purpose:
Enhance test_all_anomalies_master.sh with causal cascade analysis.

This script identifies:
1. ROOT CAUSE (Epicenter): Which sensor/variable is the source of the anomaly
2. CAUSAL CASCADE: How the anomaly propagates through the causal graph
   - Direct effects (1st order): Variables directly affected by root cause
   - Indirect effects (2nd order, 3rd order, ...): Downstream propagation
   - Temporal evolution: How cascade evolves across time

Methodology:
- Compare edge-level weights (not aggregate window changes)
- Focus on edges FROM anomaly variable (outgoing connections)
- Trace propagation paths through the causal graph
- Quantify propagation strength at each step
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class CausalCascadeDetector:
    def __init__(self, golden_weights_csv: str, anomaly_weights_csv: str,
                 ground_truth_json: str = None, threshold: float = 0.01):
        """
        Initialize the causal cascade detector.

        Args:
            golden_weights_csv: Path to golden weights CSV
            anomaly_weights_csv: Path to anomaly weights CSV
            ground_truth_json: Path to ground truth JSON (optional, for validation)
            threshold: Threshold for significant edge weight changes
        """
        self.threshold = threshold

        # Load weights
        self.golden_df = pd.read_csv(golden_weights_csv)
        self.anomaly_df = pd.read_csv(anomaly_weights_csv)

        # Load ground truth if provided
        self.ground_truth = None
        self.ground_truth_sensor = None
        if ground_truth_json and Path(ground_truth_json).exists():
            with open(ground_truth_json, 'r') as f:
                self.ground_truth = json.load(f)
                # Handle both raw and differenced sensor names
                raw_sensor = self.ground_truth['ts_col']
                self.ground_truth_sensor = f"{raw_sensor}_diff"
                print(f"Ground truth sensor: {raw_sensor}")
                print(f"Looking for differenced version: {self.ground_truth_sensor}")

        # Merge and calculate differences
        self.merged_df = pd.merge(
            self.golden_df,
            self.anomaly_df,
            on=['window_idx', 'parent_name', 'child_name', 'lag'],
            how='outer',
            suffixes=('_golden', '_anomaly')
        ).fillna(0)

        self.merged_df['weight_diff'] = abs(
            self.merged_df['weight_anomaly'] - self.merged_df['weight_golden']
        )
        self.merged_df['weight_change'] = (
            self.merged_df['weight_anomaly'] - self.merged_df['weight_golden']
        )

    def detect_epicenter_window_by_window(self) -> pd.DataFrame:
        """
        Detect epicenter (root cause variable) for each window using edge-specific analysis.

        Returns:
            DataFrame with epicenter detection results per window
        """
        print("\n" + "="*80)
        print("STEP 1: EPICENTER DETECTION (Window-by-Window)")
        print("="*80)

        results = []

        for window_idx in sorted(self.merged_df['window_idx'].unique()):
            window_data = self.merged_df[self.merged_df['window_idx'] == window_idx]

            # Filter significant changes
            significant = window_data[window_data['weight_diff'] > self.threshold]

            if len(significant) == 0:
                continue

            # Calculate impact score for each variable as SOURCE (parent in edges)
            # Focus on OUTGOING edges from each variable
            source_scores = defaultdict(lambda: {'total_outgoing_change': 0.0,
                                                 'n_outgoing_edges': 0,
                                                 'max_outgoing_change': 0.0})

            for _, row in significant.iterrows():
                parent = row['parent_name']
                diff = row['weight_diff']

                source_scores[parent]['total_outgoing_change'] += diff
                source_scores[parent]['n_outgoing_edges'] += 1
                source_scores[parent]['max_outgoing_change'] = max(
                    source_scores[parent]['max_outgoing_change'], diff
                )

            # Normalize by number of edges
            for var, scores in source_scores.items():
                if scores['n_outgoing_edges'] > 0:
                    scores['avg_outgoing_change'] = (
                        scores['total_outgoing_change'] / scores['n_outgoing_edges']
                    )
                else:
                    scores['avg_outgoing_change'] = 0.0

            # Find epicenter (variable with highest outgoing change)
            if source_scores:
                epicenter = max(source_scores.items(),
                               key=lambda x: x[1]['total_outgoing_change'])

                epicenter_var = epicenter[0]
                epicenter_score = epicenter[1]

                # Check if matches ground truth
                matches_gt = False
                if self.ground_truth_sensor:
                    matches_gt = self.ground_truth_sensor in epicenter_var

                results.append({
                    'window_idx': window_idx,
                    'epicenter': epicenter_var,
                    'total_outgoing_change': epicenter_score['total_outgoing_change'],
                    'n_outgoing_edges': epicenter_score['n_outgoing_edges'],
                    'avg_outgoing_change': epicenter_score['avg_outgoing_change'],
                    'max_outgoing_change': epicenter_score['max_outgoing_change'],
                    'matches_ground_truth': matches_gt
                })

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            print(f"\nWindows with detected epicenters: {len(results_df)}")
            print(f"\nTop 5 epicenter detections:")
            print(results_df.nlargest(5, 'total_outgoing_change')[
                ['window_idx', 'epicenter', 'total_outgoing_change', 'matches_ground_truth']
            ])

            if self.ground_truth_sensor:
                match_rate = results_df['matches_ground_truth'].sum() / len(results_df) * 100
                print(f"\nGround truth match rate: {match_rate:.1f}%")

        return results_df

    def trace_cascade_from_epicenter(self, window_idx: int, epicenter_var: str,
                                     max_depth: int = 3) -> Dict:
        """
        Trace causal cascade from epicenter variable using BFS through the causal graph.

        Args:
            window_idx: Window to analyze
            epicenter_var: Root cause variable
            max_depth: Maximum propagation depth to trace

        Returns:
            Dictionary with cascade information
        """
        window_data = self.merged_df[
            (self.merged_df['window_idx'] == window_idx) &
            (self.merged_df['weight_diff'] > self.threshold)
        ]

        if len(window_data) == 0:
            return {'epicenter': epicenter_var, 'cascade': {}}

        # Build adjacency list (simplified if networkx not available)
        if not HAS_NETWORKX:
            # Simple adjacency dict
            graph = defaultdict(list)
            for _, row in window_data.iterrows():
                parent = row['parent_name']
                child = row['child_name']
                graph[parent].append({
                    'child': child,
                    'weight_diff': row['weight_diff'],
                    'weight_change': row['weight_change'],
                    'lag': row['lag']
                })
        else:
            # Build directed graph of significant changes
            G = nx.DiGraph()
            for _, row in window_data.iterrows():
                parent = row['parent_name']
                child = row['child_name']
                weight_diff = row['weight_diff']
                weight_change = row['weight_change']
                lag = row['lag']

                G.add_edge(parent, child,
                          weight_diff=weight_diff,
                          weight_change=weight_change,
                          lag=lag)

        # BFS from epicenter
        cascade = defaultdict(list)
        visited = set()
        queue = deque([(epicenter_var, 0)])  # (variable, depth)

        while queue:
            current_var, depth = queue.popleft()

            if current_var in visited or depth > max_depth:
                continue

            visited.add(current_var)

            # Get all outgoing edges from current variable
            if HAS_NETWORKX and current_var in G:
                for child in G.successors(current_var):
                    edge_data = G[current_var][child]

                    cascade[depth].append({
                        'from': current_var,
                        'to': child,
                        'weight_diff': edge_data['weight_diff'],
                        'weight_change': edge_data['weight_change'],
                        'lag': edge_data['lag']
                    })

                    # Add child to queue for next depth
                    queue.append((child, depth + 1))
            elif not HAS_NETWORKX and current_var in graph:
                for edge_info in graph[current_var]:
                    child = edge_info['child']

                    cascade[depth].append({
                        'from': current_var,
                        'to': child,
                        'weight_diff': edge_info['weight_diff'],
                        'weight_change': edge_info['weight_change'],
                        'lag': edge_info['lag']
                    })

                    # Add child to queue for next depth
                    queue.append((child, depth + 1))

        return {
            'epicenter': epicenter_var,
            'window_idx': window_idx,
            'cascade': dict(cascade),
            'max_depth_reached': max(cascade.keys()) if cascade else 0,
            'total_affected_vars': len(visited) - 1  # Exclude epicenter itself
        }

    def analyze_cascade_for_windows(self, epicenter_results: pd.DataFrame,
                                    top_n: int = 5) -> List[Dict]:
        """
        Analyze cascade for top N windows with strongest epicenter signals.

        Args:
            epicenter_results: DataFrame from detect_epicenter_window_by_window()
            top_n: Number of top windows to analyze

        Returns:
            List of cascade analysis results
        """
        print("\n" + "="*80)
        print(f"STEP 2: CAUSAL CASCADE TRACING (Top {top_n} Windows)")
        print("="*80)

        # Select top windows
        top_windows = epicenter_results.nlargest(top_n, 'total_outgoing_change')

        cascade_results = []

        for _, row in top_windows.iterrows():
            window_idx = row['window_idx']
            epicenter = row['epicenter']

            print(f"\n{'─'*80}")
            print(f"Window {window_idx}: Epicenter = '{epicenter}'")
            print(f"{'─'*80}")

            cascade = self.trace_cascade_from_epicenter(window_idx, epicenter)

            # Print cascade
            for depth, edges in cascade['cascade'].items():
                print(f"\n  Depth {depth} (Order {depth+1} effects):")
                for edge in edges[:5]:  # Show top 5 per depth
                    direction = "↑" if edge['weight_change'] > 0 else "↓"
                    print(f"    {edge['from']} → {edge['to']} "
                          f"(Δ={edge['weight_diff']:.4f} {direction}, lag={edge['lag']})")

                if len(edges) > 5:
                    print(f"    ... and {len(edges) - 5} more edges at this depth")

            print(f"\n  Summary:")
            print(f"    Max propagation depth: {cascade['max_depth_reached']}")
            print(f"    Total affected variables: {cascade['total_affected_vars']}")

            cascade_results.append(cascade)

        return cascade_results

    def visualize_cascade(self, cascade: Dict, output_path: str):
        """
        Visualize causal cascade as a directed graph.

        Args:
            cascade: Cascade dictionary from trace_cascade_from_epicenter()
            output_path: Path to save visualization
        """
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            print(f"Skipping visualization (requires matplotlib and networkx)")
            return

        G = nx.DiGraph()

        # Build graph from cascade
        epicenter = cascade['epicenter']
        G.add_node(epicenter, depth=0, node_type='epicenter')

        for depth, edges in cascade['cascade'].items():
            for edge in edges:
                G.add_edge(edge['from'], edge['to'],
                          weight=edge['weight_diff'],
                          label=f"{edge['weight_diff']:.3f}")

                if edge['to'] not in G.nodes:
                    G.add_node(edge['to'], depth=depth+1, node_type='affected')

        # Layout
        pos = nx.spring_layout(G, seed=42, k=2)

        # Draw
        plt.figure(figsize=(16, 12))

        # Node colors by type
        node_colors = []
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data.get('node_type') == 'epicenter':
                node_colors.append('red')
            else:
                depth = node_data.get('depth', 0)
                # Gradient from orange to yellow
                node_colors.append(plt.cm.YlOrRd(0.3 + depth * 0.15))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=1000, alpha=0.9)

        # Draw edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.6,
                              arrowsize=20, arrowstyle='->')

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        plt.title(f"Causal Cascade from Epicenter: {epicenter}\n"
                 f"Window {cascade['window_idx']}", fontsize=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Cascade visualization saved to: {output_path}")

    def generate_summary_report(self, epicenter_results: pd.DataFrame,
                               cascade_results: List[Dict],
                               output_dir: Path):
        """
        Generate comprehensive summary report.

        Args:
            epicenter_results: DataFrame with epicenter detection
            cascade_results: List of cascade analyses
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "causal_cascade_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CAUSAL CASCADE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            # Epicenter summary
            f.write("1. EPICENTER DETECTION SUMMARY\n")
            f.write("-"*80 + "\n\n")

            if len(epicenter_results) > 0:
                f.write(f"Total windows with detected anomalies: {len(epicenter_results)}\n\n")

                f.write("Top 10 epicenter detections:\n")
                top_10 = epicenter_results.nlargest(10, 'total_outgoing_change')
                for _, row in top_10.iterrows():
                    match_str = "✅" if row['matches_ground_truth'] else "❌"
                    f.write(f"  Window {row['window_idx']:3d}: {row['epicenter']:<45s} "
                           f"(Score: {row['total_outgoing_change']:8.4f}) {match_str}\n")

                if self.ground_truth_sensor:
                    match_rate = epicenter_results['matches_ground_truth'].sum() / len(epicenter_results) * 100
                    f.write(f"\nGround truth match rate: {match_rate:.1f}%\n")
            else:
                f.write("No epicenters detected.\n")

            # Cascade summary
            f.write("\n\n2. CAUSAL CASCADE SUMMARY\n")
            f.write("-"*80 + "\n\n")

            for i, cascade in enumerate(cascade_results, 1):
                f.write(f"Cascade {i}:\n")
                f.write(f"  Window: {cascade['window_idx']}\n")
                f.write(f"  Epicenter: {cascade['epicenter']}\n")
                f.write(f"  Max propagation depth: {cascade['max_depth_reached']}\n")
                f.write(f"  Total affected variables: {cascade['total_affected_vars']}\n\n")

                for depth, edges in cascade['cascade'].items():
                    f.write(f"  Depth {depth} ({len(edges)} edges):\n")
                    for edge in edges[:3]:  # Top 3 per depth
                        f.write(f"    {edge['from']} → {edge['to']} "
                               f"(Δ={edge['weight_diff']:.4f}, lag={edge['lag']})\n")
                    if len(edges) > 3:
                        f.write(f"    ... and {len(edges) - 3} more\n")
                    f.write("\n")

        print(f"\n✅ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Causal Cascade Detector")
    parser.add_argument('--golden-weights', required=True,
                       help="Path to golden weights CSV")
    parser.add_argument('--anomaly-weights', required=True,
                       help="Path to anomaly weights CSV")
    parser.add_argument('--ground-truth', default=None,
                       help="Path to ground truth JSON (optional)")
    parser.add_argument('--output-dir', default='results/cascade_analysis',
                       help="Output directory for results")
    parser.add_argument('--threshold', type=float, default=0.01,
                       help="Threshold for significant edge changes")
    parser.add_argument('--top-n', type=int, default=5,
                       help="Number of top windows to analyze for cascade")

    args = parser.parse_args()

    print("="*80)
    print("CAUSAL CASCADE DETECTOR")
    print("="*80)

    # Initialize detector
    detector = CausalCascadeDetector(
        args.golden_weights,
        args.anomaly_weights,
        args.ground_truth,
        args.threshold
    )

    # Step 1: Detect epicenters
    epicenter_results = detector.detect_epicenter_window_by_window()

    # Step 2: Trace cascades for top windows
    cascade_results = detector.analyze_cascade_for_windows(
        epicenter_results,
        args.top_n
    )

    # Step 3: Visualize top cascade
    if cascade_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, cascade in enumerate(cascade_results[:3], 1):  # Top 3 visualizations
            viz_path = output_dir / f"cascade_window_{cascade['window_idx']}.png"
            detector.visualize_cascade(cascade, viz_path)

        # Generate summary report
        detector.generate_summary_report(epicenter_results, cascade_results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
