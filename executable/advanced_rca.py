#!/usr/bin/env python3
"""
Advanced Root Cause Analysis using Causal Path Tracing.

This script traces the root causes of an anomaly in a specified target variable
by traversing the learned causal graph backwards. It identifies the causal
pathways that contribute most significantly to the observed anomaly and can
generate a DOT file for visualization.
"""

import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from graphviz import Digraph

def load_matrices_for_window(path, window_idx, columns, max_lag=1):
    """
    Loads the weight matrices (W, A) for a specific window from the .npy file.
    """
    print(f"INFO: Loading weights for window {window_idx} from {path}")
    all_edges = np.load(path, allow_pickle=True)

    if all_edges.ndim == 0:
        all_edges = all_edges.item()

    num_nodes = len(columns)
    col_to_idx = {name: i for i, name in enumerate(columns)}

    w_matrix = np.zeros((num_nodes, num_nodes))
    a_matrix = np.zeros((num_nodes, num_nodes * max_lag))

    window_edges = [edge for edge in all_edges if edge['window_idx'] == window_idx]

    if not window_edges:
        print(f"WARNING: No edges found for window {window_idx} in {path}")
        return w_matrix, a_matrix

    for edge in window_edges:
        try:
            target_idx = col_to_idx[edge['target']]
            source_idx = col_to_idx[edge['source']]
            lag = edge['lag']
            weight = edge['weight']

            if lag == 0:
                w_matrix[target_idx, source_idx] = weight
            elif lag <= max_lag:
                a_matrix[target_idx, source_idx + (lag - 1) * num_nodes] = weight
        except KeyError as e:
            print(f"WARNING: Node '{e}' not found in columns list. Skipping edge.")
            continue
            
    print(f"INFO: Loaded {len(window_edges)} edges for window {window_idx}. Found {np.count_nonzero(w_matrix)} contemporaneous and {np.count_nonzero(a_matrix)} lagged edges.")
    return w_matrix, a_matrix

def get_golden_graph(path, columns, max_lag=1):
    """
    Computes the average golden graph across all windows in the baseline file.
    """
    print(f"INFO: Computing aggregated golden graph from {path}")
    all_edges = np.load(path, allow_pickle=True)

    if all_edges.ndim == 0:
        all_edges = all_edges.item()
        
    num_nodes = len(columns)
    col_to_idx = {name: i for i, name in enumerate(columns)}

    w_sum = defaultdict(float)
    a_sum = defaultdict(float)
    w_count = defaultdict(int)
    a_count = defaultdict(int)

    for edge in all_edges:
        try:
            target_idx = col_to_idx[edge['target']]
            source_idx = col_to_idx[edge['source']]
            lag = edge['lag']
            weight = edge['weight']

            if lag == 0:
                w_sum[(target_idx, source_idx)] += weight
                w_count[(target_idx, source_idx)] += 1
            elif lag <= max_lag:
                a_idx = (target_idx, source_idx + (lag - 1) * num_nodes)
                a_sum[a_idx] += weight
                a_count[a_idx] += 1
        except KeyError as e:
            print(f"WARNING: Node '{e}' not found in columns list. Skipping edge.")
            continue

    w_avg = np.zeros((num_nodes, num_nodes))
    a_avg = np.zeros((num_nodes, num_nodes * max_lag))
    
    for (r, c), total in w_sum.items():
        w_avg[r, c] = total / w_count[(r, c)]
        
    for (r, c), total in a_sum.items():
        a_avg[r, c] = total / a_count[(r, c)]
        
    print(f"INFO: Aggregated golden graph. Found {np.count_nonzero(w_avg)} contemporaneous and {np.count_nonzero(a_avg)} lagged edges.")
    return w_avg, a_avg

def generate_dot_graph(paths, anomalous_w, golden_w, columns, target_node, output_file):
    """Generates and saves a DOT graph visualization of the causal paths."""
    print(f"\\n--- Generating DOT graph visualization ---")
    dot = Digraph('RootCausePaths', comment='Root Cause Analysis')
    dot.attr('graph', rankdir='LR', splines='true', overlap='false', label=f'Root Cause Analysis for {target_node}', labelloc='t')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    col_to_idx = {name: i for i, name in enumerate(columns)}
    nodes_in_paths = set(node for path, score in paths for node in path)
    
    delta_w = anomalous_w - golden_w
    max_abs_delta = np.max(np.abs(delta_w)) if np.max(np.abs(delta_w)) > 0 else 1.0

    for node_name in nodes_in_paths:
        if node_name == target_node:
            dot.node(node_name, style='rounded,filled', fillcolor='tomato')
        else:
            dot.node(node_name)

    for path, score in paths:
        for i in range(len(path) - 1):
            source_name = path[i]
            target_name = path[i+1]
            
            source_idx = col_to_idx[source_name]
            target_idx = col_to_idx[target_name]

            anom_weight = anomalous_w[target_idx, source_idx]
            delta = delta_w[target_idx, source_idx]
            
            # Style edge based on delta
            penwidth = 1.0 + 4.0 * (abs(delta) / max_abs_delta)
            color = "red" if delta < 0 else "darkgreen"

            dot.edge(
                source_name,
                target_name,
                label=f"w={anom_weight:.2f} (Δ={delta:+.2f})",
                penwidth=str(penwidth),
                color=color,
                fontcolor=color
            )

    try:
        # The render method saves the file and adds a .gv extension if not present
        dot_output_path = output_file if output_file.endswith('.gv') else f"{output_file}.gv"
        dot.render(output_file, view=False, cleanup=True)
        print(f"INFO: Saved DOT graph source to {dot_output_path}")
        print(f"      To render, use: dot -Tpng {dot_output_path} -o output.png")
    except Exception as e:
        print(f"ERROR: Failed to generate DOT graph. Is graphviz installed and in your PATH?")
        print(f"       {e}")
    print("--------------------------------------\\n")


def trace_root_causes(anomalous_w, golden_w, columns, target_node, max_depth=3, top_k=10):
    """
    Traces the causal path backwards from a target node using a recursive DFS.
    """
    print("\\n--- Starting Causal Path Tracing ---")
    print(f"Target Node: {target_node}, Max Depth: {max_depth}, Top K: {top_k}")

    if target_node not in columns:
        raise ValueError(f"Target node '{target_node}' not found in columns.")

    target_idx = columns.index(target_node)
    delta_w = np.abs(anomalous_w - golden_w)
    
    paths = []

    def find_paths_recursive(node_idx, current_path, current_score):
        if len(current_path) >= max_depth:
            paths.append((current_path, current_score))
            return

        parents = np.where(anomalous_w[node_idx, :] != 0)[0]
        if not parents.any():
            paths.append((current_path, current_score))
            return

        for parent_idx in parents:
            if columns[parent_idx] in current_path:
                continue
            
            link_score = delta_w[node_idx, parent_idx]
            new_score = current_score + link_score
            find_paths_recursive(parent_idx, [columns[parent_idx]] + current_path, new_score)

    find_paths_recursive(target_idx, [target_node], 0)

    sorted_paths = sorted(paths, key=lambda x: x[1], reverse=True)
    
    print("\\n--- Top Causal Paths Found ---")
    for i, (path, score) in enumerate(sorted_paths[:top_k]):
        print(f"  {i+1}. Path: {' -> '.join(path)} (Score: {score:.4f})")
    print("--------------------------------\\n")

    return sorted_paths[:top_k]


def main():
    """Main function to run the advanced RCA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anomalous-weights', required=True, help='Path to the .npy file for the test timeline.')
    parser.add_argument('--golden-weights', required=True, help='Path to the .npy file for the golden baseline.')
    parser.add_argument('--columns', required=True, help='Path to the .npy file with column names.')
    parser.add_argument('--target-node', required=True, help='Name of the target variable for RCA.')
    parser.add_argument('--window-idx', required=True, type=int, help='The index of the anomalous window.')
    parser.add_argument('--output-dot', help='Optional: Path to save a .dot file for graph visualization (e.g., "results/rca_graph").')

    args = parser.parse_args()

    print("--- Advanced Root Cause Analysis ---")
    print(f"Anomalous Weights: {args.anomalous_weights}")
    print(f"Golden Weights: {args.golden_weights}")
    print(f"Columns File: {args.columns}")
    print(f"Target Node: {args.target_node}")
    print(f"Window Index: {args.window_idx}")
    print("------------------------------------\\n")

    columns = list(np.load(args.columns, allow_pickle=True))
    anomalous_w, _ = load_matrices_for_window(args.anomalous_weights, args.window_idx, columns)
    golden_w, _ = get_golden_graph(args.golden_weights, columns)

    top_paths = trace_root_causes(
        anomalous_w=anomalous_w,
        golden_w=golden_w,
        columns=columns,
        target_node=args.target_node
    )

    if args.output_dot:
        generate_dot_graph(
            paths=top_paths,
            anomalous_w=anomalous_w,
            golden_w=golden_w,
            columns=columns,
            target_node=args.target_node,
            output_file=args.output_dot
        )

if __name__ == '__main__':
    main()
