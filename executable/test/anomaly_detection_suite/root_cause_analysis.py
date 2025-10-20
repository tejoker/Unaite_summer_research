#!/usr/bin/env python3
"""
root_cause_analysis.py - Phase 3: Root Cause Analysis

Implements detailed analysis of anomaly propagation:
- Per-edge attribution: Ranked list of changed edges (i→j relationships)
- Node importance ranking: Distinguishes "causes" vs "effects"
- Causal path tracing: Shows how anomaly cascades through system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from itertools import combinations
import networkx as nx

logger = logging.getLogger(__name__)


class PerEdgeAttributor:
    """Per-edge attribution to identify which causal connections changed most."""

    def __init__(self, variable_names: Optional[List[str]] = None):
        """
        Initialize edge attributor.

        Args:
            variable_names: Optional names for variables/nodes
        """
        self.variable_names = variable_names or [f"Var_{i}" for i in range(6)]

    def analyze_edge_changes(self, W_baseline: np.ndarray, W_current: np.ndarray,
                           top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze and rank edge changes between baseline and current matrices.

        Args:
            W_baseline: Baseline weight matrix
            W_current: Current weight matrix
            top_k: Number of top changes to return

        Returns:
            Dict with ranked edge changes and analysis
        """
        logger.debug("Analyzing per-edge changes")

        W_diff = W_current - W_baseline
        W_abs_diff = np.abs(W_diff)

        # Create list of all edge changes
        edge_changes = []
        n = W_baseline.shape[0]

        for i in range(n):
            for j in range(n):
                if W_abs_diff[i, j] > 1e-8:  # Only consider non-zero changes
                    edge_change = {
                        'source_idx': i,
                        'target_idx': j,
                        'source_name': self.variable_names[i] if i < len(self.variable_names) else f"Var_{i}",
                        'target_name': self.variable_names[j] if j < len(self.variable_names) else f"Var_{j}",
                        'baseline_weight': float(W_baseline[i, j]),
                        'current_weight': float(W_current[i, j]),
                        'absolute_change': float(W_abs_diff[i, j]),
                        'relative_change': float(W_diff[i, j]),
                        'percentage_change': float(
                            (W_diff[i, j] / W_baseline[i, j] * 100) if abs(W_baseline[i, j]) > 1e-8 else float('inf')
                        ),
                        'change_type': self._classify_edge_change(W_baseline[i, j], W_current[i, j])
                    }
                    edge_changes.append(edge_change)

        # Sort by absolute change magnitude
        edge_changes.sort(key=lambda x: x['absolute_change'], reverse=True)

        # Get top changes
        top_changes = edge_changes[:top_k]

        # Analyze change patterns
        change_analysis = self._analyze_change_patterns(edge_changes)

        result = {
            'total_changed_edges': len(edge_changes),
            'top_edge_changes': top_changes,
            'change_patterns': change_analysis,
            'all_changes': edge_changes  # For detailed analysis
        }

        logger.info(f"Found {len(edge_changes)} changed edges, top change: {top_changes[0]['absolute_change']:.4f}" if edge_changes else "No edge changes detected")
        return result

    def _classify_edge_change(self, baseline_weight: float, current_weight: float) -> str:
        """Classify the type of edge change."""
        threshold = 1e-6

        if abs(baseline_weight) < threshold and abs(current_weight) > threshold:
            return "edge_added"
        elif abs(baseline_weight) > threshold and abs(current_weight) < threshold:
            return "edge_removed"
        elif abs(baseline_weight) > threshold and abs(current_weight) > threshold:
            if np.sign(baseline_weight) != np.sign(current_weight):
                return "sign_flip"
            elif abs(current_weight) > abs(baseline_weight):
                return "strengthened"
            else:
                return "weakened"
        else:
            return "no_change"

    def _analyze_change_patterns(self, edge_changes: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in edge changes."""
        if not edge_changes:
            return {'total': 0}

        # Count change types
        change_types = {}
        for change in edge_changes:
            change_type = change['change_type']
            change_types[change_type] = change_types.get(change_type, 0) + 1

        # Analyze magnitude distribution
        magnitudes = [change['absolute_change'] for change in edge_changes]

        # Source/target analysis
        source_impacts = {}
        target_impacts = {}

        for change in edge_changes:
            source = change['source_name']
            target = change['target_name']
            magnitude = change['absolute_change']

            source_impacts[source] = source_impacts.get(source, 0) + magnitude
            target_impacts[target] = target_impacts.get(target, 0) + magnitude

        return {
            'total': len(edge_changes),
            'change_types': change_types,
            'magnitude_stats': {
                'max': float(np.max(magnitudes)),
                'mean': float(np.mean(magnitudes)),
                'std': float(np.std(magnitudes)),
                'median': float(np.median(magnitudes))
            },
            'source_impacts': dict(sorted(source_impacts.items(), key=lambda x: x[1], reverse=True)),
            'target_impacts': dict(sorted(target_impacts.items(), key=lambda x: x[1], reverse=True))
        }


class NodeImportanceRanker:
    """Node importance ranking to distinguish "causes" vs "effects"."""

    def __init__(self, variable_names: Optional[List[str]] = None):
        """
        Initialize node importance ranker.

        Args:
            variable_names: Optional names for variables/nodes
        """
        self.variable_names = variable_names or [f"Var_{i}" for i in range(6)]

    def rank_node_importance(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, Any]:
        """
        Rank nodes by their importance in the anomaly.

        Analyzes both outgoing (cause) and incoming (effect) impact changes.

        Args:
            W_baseline: Baseline weight matrix
            W_current: Current weight matrix

        Returns:
            Dict with node importance rankings
        """
        logger.debug("Ranking node importance")

        W_diff = W_current - W_baseline
        W_abs_diff = np.abs(W_diff)
        n = W_baseline.shape[0]

        node_analysis = []

        for i in range(n):
            node_name = self.variable_names[i] if i < len(self.variable_names) else f"Var_{i}"

            # Outgoing impact (this node as a cause)
            outgoing_impact = np.sum(W_abs_diff[i, :])  # Sum of absolute changes in row i
            outgoing_baseline = np.sum(np.abs(W_baseline[i, :]))
            outgoing_relative = outgoing_impact / (outgoing_baseline + 1e-8)

            # Incoming impact (this node as an effect)
            incoming_impact = np.sum(W_abs_diff[:, i])  # Sum of absolute changes in column i
            incoming_baseline = np.sum(np.abs(W_baseline[:, i]))
            incoming_relative = incoming_impact / (incoming_baseline + 1e-8)

            # Combined importance score
            total_impact = outgoing_impact + incoming_impact

            # Classify node role
            if outgoing_impact > 1.5 * incoming_impact:
                role = "primary_cause"
            elif incoming_impact > 1.5 * outgoing_impact:
                role = "primary_effect"
            elif total_impact > np.mean([outgoing_impact, incoming_impact]) * 0.5:
                role = "bidirectional"
            else:
                role = "minimal_impact"

            node_info = {
                'node_idx': i,
                'node_name': node_name,
                'outgoing_impact': float(outgoing_impact),
                'incoming_impact': float(incoming_impact),
                'total_impact': float(total_impact),
                'outgoing_relative': float(outgoing_relative),
                'incoming_relative': float(incoming_relative),
                'role': role,
                'cause_effect_ratio': float(outgoing_impact / (incoming_impact + 1e-8))
            }
            node_analysis.append(node_info)

        # Sort by total impact
        node_analysis.sort(key=lambda x: x['total_impact'], reverse=True)

        # Create separate rankings
        cause_ranking = sorted(node_analysis, key=lambda x: x['outgoing_impact'], reverse=True)
        effect_ranking = sorted(node_analysis, key=lambda x: x['incoming_impact'], reverse=True)

        result = {
            'overall_ranking': node_analysis,
            'cause_ranking': cause_ranking,
            'effect_ranking': effect_ranking,
            'role_summary': self._summarize_roles(node_analysis),
            'network_stats': self._compute_network_stats(W_baseline, W_current)
        }

        logger.info(f"Top cause: {cause_ranking[0]['node_name']} (impact: {cause_ranking[0]['outgoing_impact']:.4f})")
        logger.info(f"Top effect: {effect_ranking[0]['node_name']} (impact: {effect_ranking[0]['incoming_impact']:.4f})")

        return result

    def _summarize_roles(self, node_analysis: List[Dict]) -> Dict[str, Any]:
        """Summarize the roles of nodes in the anomaly."""
        role_counts = {}
        for node in node_analysis:
            role = node['role']
            role_counts[role] = role_counts.get(role, 0) + 1

        primary_causes = [node for node in node_analysis if node['role'] == 'primary_cause']
        primary_effects = [node for node in node_analysis if node['role'] == 'primary_effect']

        return {
            'role_distribution': role_counts,
            'primary_causes': [node['node_name'] for node in primary_causes],
            'primary_effects': [node['node_name'] for node in primary_effects],
            'most_influential': node_analysis[0]['node_name'] if node_analysis else None
        }

    def _compute_network_stats(self, W_baseline: np.ndarray, W_current: np.ndarray) -> Dict[str, float]:
        """Compute network-level statistics."""
        # Overall network strength
        baseline_strength = float(np.sum(np.abs(W_baseline)))
        current_strength = float(np.sum(np.abs(W_current)))
        strength_change = current_strength - baseline_strength

        # Network density (proportion of non-zero edges)
        baseline_density = float(np.sum(np.abs(W_baseline) > 1e-6) / W_baseline.size)
        current_density = float(np.sum(np.abs(W_current) > 1e-6) / W_current.size)
        density_change = current_density - baseline_density

        return {
            'baseline_strength': baseline_strength,
            'current_strength': current_strength,
            'strength_change': strength_change,
            'strength_change_percent': float((strength_change / baseline_strength * 100) if baseline_strength > 0 else 0),
            'baseline_density': baseline_density,
            'current_density': current_density,
            'density_change': density_change
        }


class CausalPathTracer:
    """Causal path tracing to show how anomaly cascades through system."""

    def __init__(self, variable_names: Optional[List[str]] = None, edge_threshold: float = 0.01):
        """
        Initialize causal path tracer.

        Args:
            variable_names: Optional names for variables/nodes
            edge_threshold: Threshold for considering an edge significant
        """
        self.variable_names = variable_names or [f"Var_{i}" for i in range(6)]
        self.edge_threshold = edge_threshold

    def trace_propagation_pathways(self, W_baseline: np.ndarray, W_current: np.ndarray,
                                 max_path_length: int = 3) -> Dict[str, Any]:
        """
        Trace how anomalies propagate through causal pathways.

        Args:
            W_baseline: Baseline weight matrix
            W_current: Current weight matrix
            max_path_length: Maximum length of causal paths to consider

        Returns:
            Dict with propagation pathways and cascade analysis
        """
        logger.debug("Tracing causal propagation pathways")

        # Create adjacency matrices for graph analysis
        adj_baseline = (np.abs(W_baseline) > self.edge_threshold).astype(int)
        adj_current = (np.abs(W_current) > self.edge_threshold).astype(int)

        # Find significant changes
        W_diff = W_current - W_baseline
        significant_changes = np.abs(W_diff) > np.std(np.abs(W_diff)) + np.mean(np.abs(W_diff))

        # Create graphs using networkx if available
        try:
            import networkx as nx

            # Create baseline and current graphs
            G_baseline = self._create_networkx_graph(W_baseline, adj_baseline)
            G_current = self._create_networkx_graph(W_current, adj_current)

            # Find propagation paths
            pathways = self._find_propagation_paths_nx(G_baseline, G_current, W_diff, significant_changes, max_path_length)

        except ImportError:
            logger.warning("NetworkX not available, using simplified path analysis")
            pathways = self._find_propagation_paths_simple(W_baseline, W_current, W_diff, significant_changes, max_path_length)

        # Analyze cascade effects
        cascade_analysis = self._analyze_cascade_effects(W_baseline, W_current, pathways)

        result = {
            'propagation_pathways': pathways,
            'cascade_analysis': cascade_analysis,
            'summary_stats': self._compute_pathway_stats(pathways)
        }

        logger.info(f"Found {len(pathways)} significant propagation pathways")
        return result

    def _create_networkx_graph(self, W: np.ndarray, adj: np.ndarray) -> 'nx.DiGraph':
        """Create NetworkX directed graph from weight matrix."""
        import networkx as nx

        G = nx.DiGraph()
        n = W.shape[0]

        # Add nodes
        for i in range(n):
            node_name = self.variable_names[i] if i < len(self.variable_names) else f"Var_{i}"
            G.add_node(i, name=node_name)

        # Add edges
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    G.add_edge(i, j, weight=float(W[i, j]))

        return G

    def _find_propagation_paths_nx(self, G_baseline: 'nx.DiGraph', G_current: 'nx.DiGraph',
                                  W_diff: np.ndarray, significant_changes: np.ndarray,
                                  max_path_length: int) -> List[Dict]:
        """Find propagation paths using NetworkX."""
        import networkx as nx

        pathways = []

        # Find nodes with significant outgoing changes (potential sources)
        source_candidates = []
        for i in range(W_diff.shape[0]):
            if np.any(significant_changes[i, :]):
                source_candidates.append(i)

        # Find nodes with significant incoming changes (potential targets)
        target_candidates = []
        for j in range(W_diff.shape[1]):
            if np.any(significant_changes[:, j]):
                target_candidates.append(j)

        # Trace paths from sources to targets
        for source in source_candidates:
            for target in target_candidates:
                if source != target:
                    try:
                        # Find paths in both baseline and current graphs
                        paths_baseline = list(nx.all_simple_paths(G_baseline, source, target, cutoff=max_path_length))
                        paths_current = list(nx.all_simple_paths(G_current, source, target, cutoff=max_path_length))

                        # Analyze path changes
                        pathway_info = self._analyze_path_changes(
                            paths_baseline, paths_current, W_diff, source, target
                        )

                        if pathway_info['significance_score'] > 0.1:
                            pathways.append(pathway_info)

                    except nx.NetworkXNoPath:
                        continue

        return pathways

    def _find_propagation_paths_simple(self, W_baseline: np.ndarray, W_current: np.ndarray,
                                     W_diff: np.ndarray, significant_changes: np.ndarray,
                                     max_path_length: int) -> List[Dict]:
        """Simplified path finding without NetworkX."""
        pathways = []

        # Simple two-hop analysis
        n = W_baseline.shape[0]

        for i in range(n):
            for j in range(n):
                if significant_changes[i, j]:
                    # Direct path
                    direct_path = {
                        'path': [i, j],
                        'path_names': [self.variable_names[i], self.variable_names[j]],
                        'path_length': 1,
                        'total_change': float(W_diff[i, j]),
                        'significance_score': float(np.abs(W_diff[i, j])),
                        'pathway_type': 'direct'
                    }
                    pathways.append(direct_path)

                    # Two-hop paths through intermediates
                    for k in range(n):
                        if k != i and k != j:
                            if (np.abs(W_baseline[i, k]) > self.edge_threshold and
                                np.abs(W_baseline[k, j]) > self.edge_threshold):

                                # Calculate pathway strength change
                                baseline_strength = W_baseline[i, k] * W_baseline[k, j]
                                current_strength = W_current[i, k] * W_current[k, j]
                                pathway_change = current_strength - baseline_strength

                                if abs(pathway_change) > 0.01:
                                    indirect_path = {
                                        'path': [i, k, j],
                                        'path_names': [self.variable_names[i], self.variable_names[k], self.variable_names[j]],
                                        'path_length': 2,
                                        'total_change': float(pathway_change),
                                        'significance_score': float(np.abs(pathway_change)),
                                        'pathway_type': 'indirect'
                                    }
                                    pathways.append(indirect_path)

        # Sort by significance
        pathways.sort(key=lambda x: x['significance_score'], reverse=True)
        return pathways[:20]  # Limit to top 20 pathways

    def _analyze_path_changes(self, paths_baseline: List, paths_current: List,
                            W_diff: np.ndarray, source: int, target: int) -> Dict:
        """Analyze changes in causal pathways."""
        # Compute path strength changes
        total_change = 0.0
        path_count_change = len(paths_current) - len(paths_baseline)

        # For now, use direct connection change as proxy
        direct_change = float(W_diff[source, target])

        pathway_info = {
            'source': source,
            'target': target,
            'source_name': self.variable_names[source],
            'target_name': self.variable_names[target],
            'paths_baseline_count': len(paths_baseline),
            'paths_current_count': len(paths_current),
            'path_count_change': path_count_change,
            'direct_change': direct_change,
            'total_change': direct_change,  # Simplified
            'significance_score': abs(direct_change),
            'pathway_type': 'analyzed'
        }

        return pathway_info

    def _analyze_cascade_effects(self, W_baseline: np.ndarray, W_current: np.ndarray,
                               pathways: List[Dict]) -> Dict[str, Any]:
        """Analyze cascade effects in the causal network."""
        if not pathways:
            return {'cascade_detected': False}

        # Count cascade patterns
        direct_effects = [p for p in pathways if p.get('path_length', 1) == 1]
        indirect_effects = [p for p in pathways if p.get('path_length', 1) > 1]

        # Identify amplification vs dampening
        amplifying_paths = [p for p in pathways if p.get('total_change', 0) > 0]
        dampening_paths = [p for p in pathways if p.get('total_change', 0) < 0]

        # Find most affected nodes
        node_involvement = {}
        for pathway in pathways:
            path = pathway.get('path', [])
            for node in path:
                node_name = self.variable_names[node] if node < len(self.variable_names) else f"Var_{node}"
                node_involvement[node_name] = node_involvement.get(node_name, 0) + 1

        most_involved = sorted(node_involvement.items(), key=lambda x: x[1], reverse=True)

        cascade_analysis = {
            'cascade_detected': len(indirect_effects) > 0,
            'direct_effects_count': len(direct_effects),
            'indirect_effects_count': len(indirect_effects),
            'amplifying_paths_count': len(amplifying_paths),
            'dampening_paths_count': len(dampening_paths),
            'most_involved_nodes': most_involved,
            'cascade_complexity': len(indirect_effects) / max(len(pathways), 1),
            'net_amplification': sum(p.get('total_change', 0) for p in amplifying_paths),
            'net_dampening': sum(p.get('total_change', 0) for p in dampening_paths)
        }

        return cascade_analysis

    def _compute_pathway_stats(self, pathways: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics for pathways."""
        if not pathways:
            return {'total_pathways': 0}

        significance_scores = [p.get('significance_score', 0) for p in pathways]
        path_lengths = [p.get('path_length', 1) for p in pathways]

        return {
            'total_pathways': len(pathways),
            'avg_significance': float(np.mean(significance_scores)),
            'max_significance': float(np.max(significance_scores)),
            'avg_path_length': float(np.mean(path_lengths)),
            'max_path_length': float(np.max(path_lengths)),
            'pathway_types': list(set(p.get('pathway_type', 'unknown') for p in pathways))
        }


def perform_root_cause_analysis(W_baseline: np.ndarray, W_current: np.ndarray,
                               variable_names: Optional[List[str]] = None,
                               top_k_edges: int = 10, max_path_length: int = 3) -> Dict[str, Any]:
    """
    Perform comprehensive root cause analysis.

    Args:
        W_baseline: Baseline weight matrix
        W_current: Current weight matrix
        variable_names: Optional variable names
        top_k_edges: Number of top edge changes to analyze
        max_path_length: Maximum causal path length to trace

    Returns:
        Complete root cause analysis results
    """
    logger.info("Performing comprehensive root cause analysis")

    # Use default variable names if not provided
    if variable_names is None:
        variable_names = [
            'Speed', 'Load', 'Temp_Exz_L', 'Temp_Exz_R', 'Vibration', 'Pressure'
        ]

    # Initialize analyzers
    edge_attributor = PerEdgeAttributor(variable_names)
    node_ranker = NodeImportanceRanker(variable_names)
    path_tracer = CausalPathTracer(variable_names)

    # Perform analyses
    edge_analysis = edge_attributor.analyze_edge_changes(W_baseline, W_current, top_k_edges)
    node_analysis = node_ranker.rank_node_importance(W_baseline, W_current)
    pathway_analysis = path_tracer.trace_propagation_pathways(W_baseline, W_current, max_path_length)

    # Combine results
    root_cause_results = {
        'edge_attribution': edge_analysis,
        'node_importance': node_analysis,
        'pathway_tracing': pathway_analysis,
        'summary': _create_root_cause_summary(edge_analysis, node_analysis, pathway_analysis)
    }

    logger.info("Root cause analysis complete")
    return root_cause_results


def _create_root_cause_summary(edge_analysis: Dict, node_analysis: Dict,
                              pathway_analysis: Dict) -> Dict[str, Any]:
    """Create executive summary of root cause analysis."""

    # Primary findings
    top_edge = edge_analysis['top_edge_changes'][0] if edge_analysis['top_edge_changes'] else None
    top_cause = node_analysis['cause_ranking'][0] if node_analysis['cause_ranking'] else None
    top_effect = node_analysis['effect_ranking'][0] if node_analysis['effect_ranking'] else None

    # Key insights
    cascade_detected = pathway_analysis['cascade_analysis'].get('cascade_detected', False)
    total_pathways = pathway_analysis['summary_stats'].get('total_pathways', 0)

    summary = {
        'primary_changed_edge': {
            'connection': f"{top_edge['source_name']} → {top_edge['target_name']}" if top_edge else None,
            'change_magnitude': top_edge['absolute_change'] if top_edge else 0,
            'change_type': top_edge['change_type'] if top_edge else None
        },
        'most_influential_cause': {
            'node': top_cause['node_name'] if top_cause else None,
            'impact_score': top_cause['outgoing_impact'] if top_cause else 0
        },
        'most_affected_target': {
            'node': top_effect['node_name'] if top_effect else None,
            'impact_score': top_effect['incoming_impact'] if top_effect else 0
        },
        'cascade_effects': {
            'detected': cascade_detected,
            'pathway_count': total_pathways,
            'complexity': pathway_analysis['cascade_analysis'].get('cascade_complexity', 0)
        },
        'network_impact': {
            'total_edges_changed': edge_analysis['total_changed_edges'],
            'strength_change_percent': node_analysis['network_stats']['strength_change_percent'],
            'density_change': node_analysis['network_stats']['density_change']
        }
    }

    return summary