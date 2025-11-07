#!/usr/bin/env python3
"""
graph_structure_detector.py - Analyze causal graph structure changes from DYNOTEARS weight matrices

This tool analyzes structural changes in causal graphs learned by DYNOTEARS when different
anomaly types are introduced to bearing monitoring time-series data.

Features:
- Loads weight matrices from timestamped launch directories
- Computes structural signatures (adjacency, edge lists, degrees, density)
- Detects structural changes against golden baseline
- Classifies change patterns (spike-like, stable, degradation, restructuring)
- Generates comprehensive reports and visualizations

Usage:
    python graph_structure_detector.py --results-dir ./results
    python graph_structure_detector.py --results-dir ./results --edge-threshold 0.05 --no-viz
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm

# Try to import networkx for graph visualizations
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Graph visualizations will be limited.")

# Configure logging
def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger('graph_structure_detector')
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_dir / 'analysis.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


class WeightMatrixLoader:
    """Load and validate weight matrices from launch directories."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.variables = [
            'Speed', 'Load', 'Temperatur_Exzenterlager_links',
            'Temperatur_Exzenterlager_rechts', 'Vibration', 'Pressure'
        ]
        self.expected_shape = (6, 6)

    def discover_launches(self, results_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Discover all launch directories and categorize them.

        Returns:
            Dict with 'golden' and 'anomalies' keys containing launch info
        """
        self.logger.info(f"Discovering launches in {results_dir}")

        launches = {'golden': None, 'anomalies': []}

        # Look for launch directories
        launch_pattern = "launch2_*"
        launch_dirs = list(results_dir.glob(launch_pattern))

        self.logger.info(f"Found {len(launch_dirs)} launch directories")

        for launch_dir in sorted(launch_dirs):
            launch_info = self._analyze_launch_directory(launch_dir)
            if launch_info:
                if launch_info['is_golden']:
                    launches['golden'] = launch_info
                    self.logger.info(f"Found golden baseline: {launch_info['launch_name']}")
                else:
                    launches['anomalies'].append(launch_info)
                    self.logger.info(f"Found anomaly launch: {launch_info['launch_name']} "
                                   f"({launch_info['anomaly_type']})")

        if not launches['golden']:
            raise ValueError("No golden baseline launch found")

        self.logger.info(f"Summary: 1 golden baseline, {len(launches['anomalies'])} anomaly launches")
        return launches

    def _analyze_launch_directory(self, launch_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a launch directory to extract launch info and find weight files."""
        launch_name = launch_dir.name

        # Look for weight CSV files
        weight_files = list(launch_dir.rglob("*weights*.csv"))

        if not weight_files:
            self.logger.warning(f"No weight files found in {launch_name}")
            return None

        # Use the first/main weight file found
        weight_file = weight_files[0]

        # Determine if this is golden or anomaly based on directory structure
        # Look for anomaly indicators in subdirectories
        anomaly_type = self._detect_anomaly_type(launch_dir)
        is_golden = anomaly_type is None

        return {
            'launch_name': launch_name,
            'launch_dir': launch_dir,
            'weight_file': weight_file,
            'is_golden': is_golden,
            'anomaly_type': anomaly_type,
            'timestamp': self._extract_timestamp(launch_name)
        }

    def _detect_anomaly_type(self, launch_dir: Path) -> Optional[str]:
        """Detect anomaly type from directory structure or file names."""
        # Look for anomaly type indicators in subdirectories or file names
        anomaly_types = ['spike', 'drift', 'level_shift', 'amplitude_change', 'trend_change', 'variance_burst']

        # Check subdirectories
        for subdir in launch_dir.iterdir():
            if subdir.is_dir():
                subdir_name = subdir.name.lower()
                for anomaly_type in anomaly_types:
                    if anomaly_type.replace('_', '') in subdir_name or anomaly_type in subdir_name:
                        return anomaly_type

        # Check file names
        for file in launch_dir.rglob("*.csv"):
            file_name = file.name.lower()
            for anomaly_type in anomaly_types:
                if anomaly_type.replace('_', '') in file_name or anomaly_type in file_name:
                    return anomaly_type

        # Check if this looks like a golden/baseline directory
        dir_name = launch_dir.name.lower()
        if 'golden' in dir_name or 'baseline' in dir_name or 'normal' in dir_name:
            return None

        # If no anomaly type detected, assume it might be golden
        # You may need to adjust this logic based on your actual directory structure
        return None

    def _extract_timestamp(self, launch_name: str) -> str:
        """Extract timestamp from launch name."""
        # launch2_20250926_131753 -> 20250926_131753
        parts = launch_name.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
        return launch_name

    def load_weight_matrix(self, weight_file: Path) -> np.ndarray:
        """Load and validate a weight matrix from CSV file."""
        try:
            # Try different CSV loading approaches
            if 'window_idx' in open(weight_file).read():
                # This is a detailed weights file with window/lag info
                df = pd.read_csv(weight_file)
                matrix = self._convert_detailed_weights_to_matrix(df)
            else:
                # This is a direct matrix CSV
                matrix = pd.read_csv(weight_file, header=None).values

            # Validate matrix shape
            if matrix.shape != self.expected_shape:
                self.logger.warning(f"Weight matrix shape {matrix.shape} != expected {self.expected_shape}")
                # Try to reshape or extract the relevant part
                matrix = self._fix_matrix_shape(matrix)

            self.logger.debug(f"Loaded weight matrix from {weight_file.name}: shape {matrix.shape}")
            return matrix

        except Exception as e:
            self.logger.error(f"Failed to load weight matrix from {weight_file}: {e}")
            raise

    def _convert_detailed_weights_to_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Convert detailed weights DataFrame to matrix format."""
        # Assume we want the final time window and lag 0 (contemporaneous effects)
        if 'window_idx' in df.columns:
            # Get the last window
            last_window = df['window_idx'].max()
            df_filtered = df[df['window_idx'] == last_window]
        else:
            df_filtered = df

        if 'lag' in df.columns:
            # Get lag 0 (contemporaneous effects)
            df_filtered = df_filtered[df_filtered['lag'] == 0]

        # Create matrix
        matrix = np.zeros(self.expected_shape)
        for _, row in df_filtered.iterrows():
            i, j = int(row['i']), int(row['j'])
            if 0 <= i < self.expected_shape[0] and 0 <= j < self.expected_shape[1]:
                matrix[i, j] = float(row['weight'])

        return matrix

    def _fix_matrix_shape(self, matrix: np.ndarray) -> np.ndarray:
        """Attempt to fix matrix shape to expected dimensions."""
        if matrix.size >= np.prod(self.expected_shape):
            # Take the top-left corner
            return matrix[:self.expected_shape[0], :self.expected_shape[1]]
        else:
            # Pad with zeros
            fixed_matrix = np.zeros(self.expected_shape)
            min_rows = min(matrix.shape[0], self.expected_shape[0])
            min_cols = min(matrix.shape[1], self.expected_shape[1])
            fixed_matrix[:min_rows, :min_cols] = matrix[:min_rows, :min_cols]
            return fixed_matrix


class StructuralSignatureAnalyzer:
    """Compute structural signatures and detect changes in causal graphs."""

    def __init__(self, edge_threshold: float = 0.01, logger: logging.Logger = None):
        self.edge_threshold = edge_threshold
        self.logger = logger or logging.getLogger(__name__)
        self.variables = [
            'Speed', 'Load', 'Temperatur_Exzenterlager_links',
            'Temperatur_Exzenterlager_rechts', 'Vibration', 'Pressure'
        ]

    def compute_structural_signature(self, weight_matrix: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive structural signature of a weight matrix."""
        # Create adjacency matrix
        adjacency = (np.abs(weight_matrix) > self.edge_threshold).astype(int)

        # Extract edge list
        edge_list = []
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[1]):
                if adjacency[i, j] == 1:
                    edge_list.append((i, j))

        # Compute graph metrics
        edge_count = len(edge_list)
        total_possible_edges = adjacency.shape[0] * adjacency.shape[1]
        density = edge_count / total_possible_edges if total_possible_edges > 0 else 0.0

        # Compute degree statistics
        in_degree = adjacency.sum(axis=0)  # Sum over rows (incoming edges)
        out_degree = adjacency.sum(axis=1)  # Sum over columns (outgoing edges)

        signature = {
            'adjacency_matrix': adjacency.tolist(),
            'edge_list': edge_list,
            'edge_count': edge_count,
            'density': float(density),
            'in_degree': in_degree.tolist(),
            'out_degree': out_degree.tolist(),
            'max_weight': float(np.max(np.abs(weight_matrix))),
            'mean_weight': float(np.mean(np.abs(weight_matrix))),
            'frobenius_norm': float(np.linalg.norm(weight_matrix, 'fro'))
        }

        return signature

    def detect_structural_changes(self, baseline_sig: Dict[str, Any],
                                 anomaly_sig: Dict[str, Any]) -> Dict[str, Any]:
        """Detect structural changes between baseline and anomaly signatures."""
        baseline_edges = set(tuple(edge) for edge in baseline_sig['edge_list'])
        anomaly_edges = set(tuple(edge) for edge in anomaly_sig['edge_list'])

        # Find edge differences
        added_edges = list(anomaly_edges - baseline_edges)
        removed_edges = list(baseline_edges - anomaly_edges)

        # Structural Hamming Distance
        shd = len(added_edges) + len(removed_edges)

        # Degree changes
        baseline_in = np.array(baseline_sig['in_degree'])
        baseline_out = np.array(baseline_sig['out_degree'])
        anomaly_in = np.array(anomaly_sig['in_degree'])
        anomaly_out = np.array(anomaly_sig['out_degree'])

        in_degree_changes = (anomaly_in - baseline_in).tolist()
        out_degree_changes = (anomaly_out - baseline_out).tolist()

        # Classify change pattern
        pattern = self._classify_change_pattern(len(added_edges), len(removed_edges), shd)

        change_analysis = {
            'has_change': shd > 0,
            'added_edges': added_edges,
            'removed_edges': removed_edges,
            'shd': shd,
            'num_added_edges': len(added_edges),
            'num_removed_edges': len(removed_edges),
            'pattern': pattern,
            'in_degree_changes': in_degree_changes,
            'out_degree_changes': out_degree_changes,
            'edge_count_change': anomaly_sig['edge_count'] - baseline_sig['edge_count'],
            'density_change': anomaly_sig['density'] - baseline_sig['density']
        }

        return change_analysis

    def _classify_change_pattern(self, num_added: int, num_removed: int, shd: int) -> str:
        """Classify the structural change pattern."""
        if shd == 0:
            return "stable"
        elif num_added > 2 and num_removed == 0:
            return "spike-like"
        elif num_removed > 0 and num_added == 0:
            return "degradation"
        elif num_added > 0 and num_removed > 0:
            return "restructuring"
        elif num_added > 0 and num_removed == 0:
            return "expansion"
        elif num_added == 0 and num_removed > 0:
            return "contraction"
        else:
            return "mixed"

    def compute_weight_summary(self, baseline_matrix: np.ndarray,
                              anomaly_matrix: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics comparing weight matrices."""
        weight_diff = anomaly_matrix - baseline_matrix

        summary = {
            'frobenius_distance': float(np.linalg.norm(weight_diff, 'fro')),
            'max_change': float(np.max(np.abs(weight_diff))),
            'mean_abs_change': float(np.mean(np.abs(weight_diff))),
            'relative_frobenius': float(np.linalg.norm(weight_diff, 'fro') /
                                      (np.linalg.norm(baseline_matrix, 'fro') + 1e-8))
        }

        return summary


class StructuralVisualizationEngine:
    """Create visualizations for structural analysis."""

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.variables = [
            'Speed', 'Load', 'Temp_Exz_L',
            'Temp_Exz_R', 'Vibration', 'Pressure'
        ]

    def create_baseline_structure_plot(self, baseline_signature: Dict[str, Any],
                                     baseline_info: Dict[str, Any]) -> Path:
        """Create visualization of baseline structure."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Adjacency matrix heatmap
        adj_matrix = np.array(baseline_signature['adjacency_matrix'])
        im = axes[0].imshow(adj_matrix, cmap='Blues', interpolation='nearest')
        axes[0].set_title(f'Baseline Structure\n({baseline_info["launch_name"]})')
        axes[0].set_xlabel('Target Variable')
        axes[0].set_ylabel('Source Variable')
        axes[0].set_xticks(range(len(self.variables)))
        axes[0].set_yticks(range(len(self.variables)))
        axes[0].set_xticklabels(self.variables, rotation=45, ha='right')
        axes[0].set_yticklabels(self.variables)

        # Add edge annotations
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] == 1:
                    axes[0].text(j, i, '1', ha='center', va='center',
                               color='white', fontweight='bold')

        plt.colorbar(im, ax=axes[0])

        # Plot 2: Degree distribution
        in_degrees = baseline_signature['in_degree']
        out_degrees = baseline_signature['out_degree']

        x = np.arange(len(self.variables))
        width = 0.35

        axes[1].bar(x - width/2, in_degrees, width, label='In-degree', alpha=0.8)
        axes[1].bar(x + width/2, out_degrees, width, label='Out-degree', alpha=0.8)
        axes[1].set_xlabel('Variables')
        axes[1].set_ylabel('Degree')
        axes[1].set_title('Baseline Degree Distribution')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.variables, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Add summary statistics
        stats_text = f"""Edges: {baseline_signature['edge_count']}
Density: {baseline_signature['density']:.3f}
Max Weight: {baseline_signature['max_weight']:.3f}"""
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        plot_path = self.output_dir / 'baseline_structure.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created baseline structure plot: {plot_path}")
        return plot_path

    def create_structure_diff_plot(self, baseline_sig: Dict[str, Any], anomaly_sig: Dict[str, Any],
                                 change_analysis: Dict[str, Any], anomaly_info: Dict[str, Any]) -> Path:
        """Create visualization showing structural differences."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Baseline adjacency matrix
        baseline_adj = np.array(baseline_sig['adjacency_matrix'])
        im1 = axes[0, 0].imshow(baseline_adj, cmap='Blues', interpolation='nearest')
        axes[0, 0].set_title('Baseline Structure')
        axes[0, 0].set_xlabel('Target Variable')
        axes[0, 0].set_ylabel('Source Variable')
        self._set_variable_labels(axes[0, 0])

        # Plot 2: Anomaly adjacency matrix
        anomaly_adj = np.array(anomaly_sig['adjacency_matrix'])
        im2 = axes[0, 1].imshow(anomaly_adj, cmap='Reds', interpolation='nearest')
        axes[0, 1].set_title(f'Anomaly Structure\n({anomaly_info["anomaly_type"]})')
        axes[0, 1].set_xlabel('Target Variable')
        axes[0, 1].set_ylabel('Source Variable')
        self._set_variable_labels(axes[0, 1])

        # Plot 3: Edge differences
        diff_matrix = anomaly_adj.astype(float) - baseline_adj.astype(float)
        im3 = axes[1, 0].imshow(diff_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        axes[1, 0].set_title('Edge Changes\n(Red=Removed, Blue=Added)')
        axes[1, 0].set_xlabel('Target Variable')
        axes[1, 0].set_ylabel('Source Variable')
        self._set_variable_labels(axes[1, 0])

        # Annotate changes
        for i in range(diff_matrix.shape[0]):
            for j in range(diff_matrix.shape[1]):
                if diff_matrix[i, j] != 0:
                    color = 'white' if abs(diff_matrix[i, j]) > 0.5 else 'black'
                    text = '+' if diff_matrix[i, j] > 0 else '-'
                    axes[1, 0].text(j, i, text, ha='center', va='center',
                                   color=color, fontsize=12, fontweight='bold')

        plt.colorbar(im3, ax=axes[1, 0])

        # Plot 4: Degree changes
        in_changes = change_analysis['in_degree_changes']
        out_changes = change_analysis['out_degree_changes']

        x = np.arange(len(self.variables))
        width = 0.35

        bars1 = axes[1, 1].bar(x - width/2, in_changes, width, label='In-degree change', alpha=0.8)
        bars2 = axes[1, 1].bar(x + width/2, out_changes, width, label='Out-degree change', alpha=0.8)

        # Color bars based on positive/negative changes
        for bar, change in zip(bars1, in_changes):
            bar.set_color('green' if change > 0 else 'red' if change < 0 else 'gray')
        for bar, change in zip(bars2, out_changes):
            bar.set_color('green' if change > 0 else 'red' if change < 0 else 'gray')

        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Variables')
        axes[1, 1].set_ylabel('Degree Change')
        axes[1, 1].set_title('Degree Changes')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.variables, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add summary text
        summary_text = f"""Pattern: {change_analysis['pattern']}
SHD: {change_analysis['shd']}
Added: {change_analysis['num_added_edges']}
Removed: {change_analysis['num_removed_edges']}
ΔDensity: {change_analysis['density_change']:.3f}"""

        axes[1, 1].text(0.02, 0.98, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # Create filename
        filename = f"{anomaly_info['anomaly_type']}_{anomaly_info['timestamp']}_structure_diff.png"
        plot_path = self.output_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created structure diff plot: {plot_path}")
        return plot_path

    def create_summary_visualization(self, analysis_results: Dict[str, Any]) -> Path:
        """Create overall summary visualization."""
        anomalies = analysis_results['anomalies']

        if not anomalies:
            self.logger.warning("No anomalies to visualize")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data for plotting
        anomaly_types = [a['anomaly_type'] for a in anomalies]
        shd_values = [a['structural_change']['shd'] for a in anomalies]
        patterns = [a['structural_change']['pattern'] for a in anomalies]
        frobenius_distances = [a['weight_summary']['frobenius_distance'] for a in anomalies]

        # Plot 1: SHD by anomaly type
        unique_types = list(set(anomaly_types))
        shd_by_type = {t: [shd for i, shd in enumerate(shd_values) if anomaly_types[i] == t]
                       for t in unique_types}

        axes[0, 0].boxplot([shd_by_type[t] for t in unique_types], labels=unique_types)
        axes[0, 0].set_title('Structural Hamming Distance by Anomaly Type')
        axes[0, 0].set_ylabel('SHD')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Pattern distribution
        pattern_counts = pd.Series(patterns).value_counts()
        axes[0, 1].pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Distribution of Change Patterns')

        # Plot 3: SHD vs Frobenius distance
        axes[1, 0].scatter(frobenius_distances, shd_values, alpha=0.7)
        for i, atype in enumerate(anomaly_types):
            axes[1, 0].annotate(atype[:4], (frobenius_distances[i], shd_values[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Frobenius Distance')
        axes[1, 0].set_ylabel('Structural Hamming Distance')
        axes[1, 0].set_title('Weight Changes vs Structural Changes')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Timeline of changes
        timestamps = [a['launch'].split('_')[-1] for a in anomalies]
        has_change = [1 if a['structural_change']['has_change'] else 0 for a in anomalies]

        x_pos = range(len(anomalies))
        colors = ['red' if change else 'gray' for change in has_change]

        bars = axes[1, 1].bar(x_pos, shd_values, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Launch (by timestamp)')
        axes[1, 1].set_ylabel('SHD')
        axes[1, 1].set_title('Structural Changes Over Time')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([f"{t}\\n{a}" for t, a in zip(timestamps, anomaly_types)],
                                  rotation=45, ha='right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Has structural change'),
                          Patch(facecolor='gray', alpha=0.7, label='No structural change')]
        axes[1, 1].legend(handles=legend_elements)

        plt.tight_layout()

        plot_path = self.output_dir / 'structural_analysis_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created summary visualization: {plot_path}")
        return plot_path

    def _set_variable_labels(self, ax):
        """Set variable labels for adjacency matrix plots."""
        ax.set_xticks(range(len(self.variables)))
        ax.set_yticks(range(len(self.variables)))
        ax.set_xticklabels(self.variables, rotation=45, ha='right')
        ax.set_yticklabels(self.variables)


class GraphStructureDetector:
    """Main class orchestrating the graph structure analysis."""

    def __init__(self, results_dir: Path, edge_threshold: float = 0.01,
                 output_dir: Path = None, create_visualizations: bool = True,
                 specific_launches: List[str] = None, verbose: bool = False):

        self.results_dir = Path(results_dir)
        self.edge_threshold = edge_threshold
        self.create_visualizations = create_visualizations
        self.specific_launches = specific_launches

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"structural_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir, verbose)

        # Initialize components
        self.loader = WeightMatrixLoader(self.logger)
        self.analyzer = StructuralSignatureAnalyzer(edge_threshold, self.logger)
        if self.create_visualizations:
            self.viz_engine = StructuralVisualizationEngine(self.output_dir, self.logger)

    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete structural analysis."""
        self.logger.info("Starting graph structure analysis")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Edge threshold: {self.edge_threshold}")
        self.logger.info(f"Output directory: {self.output_dir}")

        try:
            # Step 1: Discover launches
            launches = self.loader.discover_launches(self.results_dir)

            # Filter specific launches if requested
            if self.specific_launches:
                launches['anomalies'] = [
                    a for a in launches['anomalies']
                    if any(launch_id in a['launch_name'] for launch_id in self.specific_launches)
                ]
                self.logger.info(f"Filtered to {len(launches['anomalies'])} specific launches")

            # Step 2: Load baseline
            self.logger.info("Loading baseline weight matrix...")
            baseline_matrix = self.loader.load_weight_matrix(launches['golden']['weight_file'])
            baseline_signature = self.analyzer.compute_structural_signature(baseline_matrix)

            # Step 3: Analyze anomalies
            analysis_results = {
                'baseline': {
                    'launch': launches['golden']['launch_name'],
                    'timestamp': launches['golden']['timestamp'],
                    'weight_file': str(launches['golden']['weight_file']),
                    'edge_count': baseline_signature['edge_count'],
                    'density': baseline_signature['density'],
                    'structure': baseline_signature
                },
                'anomalies': [],
                'summary': {
                    'total_anomalies': len(launches['anomalies']),
                    'edge_threshold': self.edge_threshold,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }

            self.logger.info(f"Analyzing {len(launches['anomalies'])} anomaly launches...")

            for anomaly_info in tqdm(launches['anomalies'], desc="Analyzing anomalies"):
                try:
                    # Load anomaly matrix
                    anomaly_matrix = self.loader.load_weight_matrix(anomaly_info['weight_file'])
                    anomaly_signature = self.analyzer.compute_structural_signature(anomaly_matrix)

                    # Detect changes
                    change_analysis = self.analyzer.detect_structural_changes(baseline_signature, anomaly_signature)
                    weight_summary = self.analyzer.compute_weight_summary(baseline_matrix, anomaly_matrix)

                    # Store results
                    anomaly_result = {
                        'launch': anomaly_info['launch_name'],
                        'timestamp': anomaly_info['timestamp'],
                        'anomaly_type': anomaly_info['anomaly_type'],
                        'weight_file': str(anomaly_info['weight_file']),
                        'structural_change': change_analysis,
                        'weight_summary': weight_summary,
                        'structure': anomaly_signature
                    }

                    analysis_results['anomalies'].append(anomaly_result)

                    # Create individual visualization
                    if self.create_visualizations:
                        self.viz_engine.create_structure_diff_plot(
                            baseline_signature, anomaly_signature, change_analysis, anomaly_info
                        )

                    self.logger.info(f"Analyzed {anomaly_info['launch_name']}: "
                                   f"{anomaly_info['anomaly_type']} - "
                                   f"SHD={change_analysis['shd']}, "
                                   f"Pattern={change_analysis['pattern']}")

                except Exception as e:
                    self.logger.error(f"Failed to analyze {anomaly_info['launch_name']}: {e}")
                    continue

            # Step 4: Create visualizations
            if self.create_visualizations:
                self.logger.info("Creating visualizations...")
                self.viz_engine.create_baseline_structure_plot(baseline_signature, launches['golden'])
                self.viz_engine.create_summary_visualization(analysis_results)

            # Step 5: Generate reports
            self.logger.info("Generating reports...")
            self._generate_json_report(analysis_results)
            self._generate_csv_summary(analysis_results)

            # Step 6: Print summary
            self._print_analysis_summary(analysis_results)

            self.logger.info(f"Analysis complete! Results saved to: {self.output_dir}")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def _generate_json_report(self, results: Dict[str, Any]) -> Path:
        """Generate comprehensive JSON report."""
        json_path = self.output_dir / 'structural_analysis.json'

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Generated JSON report: {json_path}")
        return json_path

    def _generate_csv_summary(self, results: Dict[str, Any]) -> Path:
        """Generate CSV summary table."""
        csv_path = self.output_dir / 'structural_summary.csv'

        rows = []
        for anomaly in results['anomalies']:
            row = {
                'launch': anomaly['launch'],
                'timestamp': anomaly['timestamp'],
                'anomaly_type': anomaly['anomaly_type'],
                'has_structural_change': anomaly['structural_change']['has_change'],
                'num_added_edges': anomaly['structural_change']['num_added_edges'],
                'num_removed_edges': anomaly['structural_change']['num_removed_edges'],
                'shd': anomaly['structural_change']['shd'],
                'pattern_classification': anomaly['structural_change']['pattern'],
                'frobenius_distance': anomaly['weight_summary']['frobenius_distance'],
                'max_weight_change': anomaly['weight_summary']['max_change'],
                'edge_count_change': anomaly['structural_change']['edge_count_change'],
                'density_change': anomaly['structural_change']['density_change']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        self.logger.info(f"Generated CSV summary: {csv_path}")
        return csv_path

    def _print_analysis_summary(self, results: Dict[str, Any]):
        """Print analysis summary to console."""
        print(f"\\n{'='*60}")
        print(f"GRAPH STRUCTURE ANALYSIS SUMMARY")
        print(f"{'='*60}")

        baseline = results['baseline']
        print(f"Baseline: {baseline['launch']}")
        print(f"  Edges: {baseline['edge_count']}")
        print(f"  Density: {baseline['density']:.3f}")

        anomalies = results['anomalies']
        print(f"\\nAnalyzed {len(anomalies)} anomaly launches:")

        # Count by pattern
        pattern_counts = {}
        type_counts = {}
        changes_detected = 0

        for anomaly in anomalies:
            pattern = anomaly['structural_change']['pattern']
            atype = anomaly['anomaly_type']

            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            type_counts[atype] = type_counts.get(atype, 0) + 1

            if anomaly['structural_change']['has_change']:
                changes_detected += 1

        print(f"\\nStructural changes detected: {changes_detected}/{len(anomalies)} ({100*changes_detected/len(anomalies):.1f}%)")

        print(f"\\nChange patterns:")
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count}")

        print(f"\\nAnomaly types:")
        for atype, count in sorted(type_counts.items()):
            print(f"  {atype}: {count}")

        print(f"\\nTop structural changes (by SHD):")
        sorted_anomalies = sorted(anomalies, key=lambda x: x['structural_change']['shd'], reverse=True)
        for i, anomaly in enumerate(sorted_anomalies[:5]):
            shd = anomaly['structural_change']['shd']
            pattern = anomaly['structural_change']['pattern']
            atype = anomaly['anomaly_type']
            print(f"  {i+1}. {atype} (SHD={shd}, {pattern})")

        print(f"\\nResults saved to: {self.output_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze causal graph structure changes from DYNOTEARS weight matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python graph_structure_detector.py --results-dir ./results

  # Custom threshold and no visualizations
  python graph_structure_detector.py --results-dir ./results --edge-threshold 0.05 --no-viz

  # Analyze specific launches only
  python graph_structure_detector.py --results-dir ./results --launches 135848 151412

  # Verbose output with custom output directory
  python graph_structure_detector.py --results-dir ./results --output-dir ./my_analysis --verbose
        """
    )

    parser.add_argument('--results-dir', required=True, type=str,
                       help='Directory containing launch results with weight CSV files')
    parser.add_argument('--edge-threshold', type=float, default=0.01,
                       help='Threshold for determining edge existence (default: 0.01)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (default: structural_analysis_TIMESTAMP)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation (faster)')
    parser.add_argument('--launches', nargs='+', type=str,
                       help='Specific launch IDs to analyze (e.g., 135848 151412)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Validate inputs
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    try:
        # Create and run detector
        detector = GraphStructureDetector(
            results_dir=results_dir,
            edge_threshold=args.edge_threshold,
            output_dir=args.output_dir,
            create_visualizations=not args.no_viz,
            specific_launches=args.launches,
            verbose=args.verbose
        )

        # Run analysis
        results = detector.run_analysis()

        print(f"\\n✅ Analysis completed successfully!")
        print(f"📁 Results available in: {detector.output_dir}")

        return 0

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())