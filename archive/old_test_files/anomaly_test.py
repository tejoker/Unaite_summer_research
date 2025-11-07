#!/usr/bin/env python3
"""
Anomaly Detection for Edge Contributions Over Time

This module analyzes when "anomalous" weights actually manifested in the data
by computing contribution series per edge and flagging temporal spikes.

For each edge (j→i, L):
- Computes contribution series c_ij,L(t) = weight × regressor_value(t)
- Uses MAD-based z-score to robustly detect spikes
- Maps flagged times to source/effect timestamps
- Reports anomalous manifestations with proper variable names and timestamps

Based on the mathematical framework:
- L=0: c_ij,0(t) = W[i,j] × x_j(t)
- L≥1: c_ij,L(t) = A^(L)[i,j] × x_j(t-L)
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnomalyEvent:
    """Represents an anomalous edge contribution event."""
    parent_name: str
    child_name: str
    lag: int
    effect_time: pd.Timestamp
    source_time: pd.Timestamp
    contribution_value: float
    z_score: float
    window_idx: int

class EdgeContributionAnalyzer:
    """
    Analyzes edge contributions over time to identify anomalous manifestations.
    """
    
    def __init__(self, 
                 time_series_data: pd.DataFrame,
                 weights_data: pd.DataFrame,
                 var_names: List[str],
                 z_threshold: float = 3.0):
        """
        Initialize the analyzer.
        
        Args:
            time_series_data: Original time series data with datetime index
            weights_data: Edge weights from DynoTears (window_idx, lag, i, j, weight)
            var_names: List of variable names
            z_threshold: Z-score threshold for anomaly detection (default: 3.0)
        """
        self.data = time_series_data
        self.weights = weights_data
        self.var_names = var_names
        self.z_threshold = z_threshold
        self.d = len(var_names)
        
        # Create variable name to index mapping
        self.var_to_idx = {name: idx for idx, name in enumerate(var_names)}
        self.idx_to_var = {idx: name for idx, name in enumerate(var_names)}
        
        logger.info(f"Initialized analyzer: {len(self.data)} time points, "
                   f"{len(self.weights)} edge weights, z_threshold={z_threshold}")
        
        # Group weights by window for efficient processing
        self.weights_by_window = self.weights.groupby('window_idx')
        
        # Store all anomaly events
        self.anomaly_events: List[AnomalyEvent] = []
        
    def compute_edge_contributions(self, 
                                 window_idx: int,
                                 window_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Compute contribution series for all edges in a window.
        
        Args:
            window_idx: Window index
            window_data: Time series data for this window
            
        Returns:
            Dictionary mapping edge_key to contribution series
        """
        contributions = {}
        
        # Get weights for this window
        if window_idx not in self.weights_by_window.groups:
            logger.warning(f"No weights found for window {window_idx}")
            return contributions
        
        window_weights = self.weights_by_window.get_group(window_idx)
        
        for _, weight_row in window_weights.iterrows():
            lag = int(weight_row['lag'])
            i = int(weight_row['i'])  # child index
            j = int(weight_row['j'])  # parent index
            weight = float(weight_row['weight'])
            
            # Skip zero weights
            if abs(weight) < 1e-10:
                continue
            
            parent_name = self.idx_to_var.get(j, f"var_{j}")
            child_name = self.idx_to_var.get(i, f"var_{i}")
            edge_key = f"{parent_name}→{child_name}_lag{lag}"
            
            # Compute contribution series
            if lag == 0:
                # Instantaneous: c_ij,0(t) = W[i,j] × x_j(t)
                if parent_name in window_data.columns:
                    contributions[edge_key] = weight * window_data[parent_name]
                else:
                    logger.warning(f"Variable {parent_name} not found in data")
                    
            else:
                # Lagged: c_ij,L(t) = A^(L)[i,j] × x_j(t-L)
                if parent_name in window_data.columns:
                    # Shift parent values by lag
                    parent_lagged = window_data[parent_name].shift(lag)
                    contributions[edge_key] = weight * parent_lagged
                else:
                    logger.warning(f"Variable {parent_name} not found in data")
        
        logger.info(f"Computed {len(contributions)} contribution series for window {window_idx}")
        return contributions
    
    def detect_anomalous_contributions(self, 
                                     contributions: Dict[str, pd.Series],
                                     window_idx: int,
                                     window_start_time: pd.Timestamp) -> List[AnomalyEvent]:
        """
        Detect anomalous spikes in contribution series using MAD-based z-score.
        
        Args:
            contributions: Dictionary of contribution series
            window_idx: Window index
            window_start_time: Start time of the window
            
        Returns:
            List of anomaly events
        """
        events = []
        
        for edge_key, contrib_series in contributions.items():
            # Skip if series is empty or all NaN
            valid_contrib = contrib_series.dropna()
            if len(valid_contrib) < 5:  # Need minimum points for robust statistics
                continue
            
            # Compute robust statistics using MAD
            median_contrib = np.median(valid_contrib)
            mad_contrib = median_abs_deviation(valid_contrib, scale='normal')  # scale='normal' for z-score compatibility
            
            # Avoid division by zero
            if mad_contrib < 1e-12:
                continue
            
            # Compute MAD-based z-scores
            z_scores = np.abs(contrib_series - median_contrib) / mad_contrib
            
            # Find anomalous time points
            anomalous_mask = z_scores >= self.z_threshold
            anomalous_times = z_scores[anomalous_mask]
            
            if len(anomalous_times) == 0:
                continue
            
            # Parse edge information
            parts = edge_key.split('→')
            parent_name = parts[0]
            child_and_lag = parts[1].split('_lag')
            child_name = child_and_lag[0]
            lag = int(child_and_lag[1])
            
            # Create events for each anomalous time point
            for effect_time, z_score in anomalous_times.items():
                # Effect time is t
                effect_timestamp = effect_time
                
                # Source time is t - L (only meaningful for L >= 1)
                if lag >= 1:
                    source_timestamp = effect_time - pd.Timedelta(seconds=lag * 60)  # Assuming 1-minute intervals
                else:
                    source_timestamp = effect_time  # Instantaneous
                
                contribution_value = float(contrib_series.loc[effect_time])
                
                event = AnomalyEvent(
                    parent_name=parent_name,
                    child_name=child_name,
                    lag=lag,
                    effect_time=effect_timestamp,
                    source_time=source_timestamp,
                    contribution_value=contribution_value,
                    z_score=float(z_score),
                    window_idx=window_idx
                )
                events.append(event)
        
        logger.info(f"Found {len(events)} anomalous contribution events in window {window_idx}")
        return events
    
    def analyze_all_windows(self, 
                           window_size: int,
                           overlap: int = 0) -> pd.DataFrame:
        """
        Analyze all windows to detect anomalous edge contributions.
        
        Args:
            window_size: Size of each window in time steps
            overlap: Number of overlapping time steps between windows
            
        Returns:
            DataFrame with all anomaly events
        """
        logger.info(f"Starting analysis of all windows (size={window_size}, overlap={overlap})")
        
        # Clear previous results
        self.anomaly_events = []
        
        # Get unique window indices from weights data
        window_indices = sorted(self.weights['window_idx'].unique())
        
        step_size = window_size - overlap
        
        for window_idx in window_indices:
            logger.info(f"Processing window {window_idx}")
            
            # Calculate window time range
            start_idx = window_idx * step_size
            end_idx = start_idx + window_size
            
            # Extract window data
            if end_idx <= len(self.data):
                window_data = self.data.iloc[start_idx:end_idx].copy()
            else:
                # Handle last window that might extend beyond data
                window_data = self.data.iloc[start_idx:].copy()
            
            if len(window_data) < 10:  # Skip very small windows
                continue
            
            window_start_time = window_data.index[0]
            
            # Compute contributions for this window
            contributions = self.compute_edge_contributions(window_idx, window_data)
            
            # Detect anomalies
            window_events = self.detect_anomalous_contributions(
                contributions, window_idx, window_start_time
            )
            
            self.anomaly_events.extend(window_events)
        
        # Convert to DataFrame for easier analysis
        if self.anomaly_events:
            events_data = []
            for event in self.anomaly_events:
                events_data.append({
                    'window_idx': event.window_idx,
                    'parent': event.parent_name,
                    'child': event.child_name,
                    'lag': event.lag,
                    'effect_time': event.effect_time,
                    'source_time': event.source_time,
                    'contribution_value': event.contribution_value,
                    'z_score': event.z_score,
                    'edge': f"{event.parent_name}→{event.child_name}_lag{event.lag}"
                })
            
            df_events = pd.DataFrame(events_data)
            logger.info(f"Total anomalous events found: {len(df_events)}")
            return df_events
        else:
            logger.info("No anomalous events found")
            return pd.DataFrame()
    
    def generate_report(self, df_events: pd.DataFrame) -> str:
        """
        Generate a human-readable report of anomalous events.
        
        Args:
            df_events: DataFrame with anomaly events
            
        Returns:
            Formatted report string
        """
        if df_events.empty:
            return "No anomalous edge contributions detected."
        
        report = []
        report.append("ANOMALOUS EDGE CONTRIBUTION REPORT")
        report.append("=" * 50)
        report.append(f"Detection threshold: z >= {self.z_threshold}")
        report.append(f"Total events: {len(df_events)}")
        report.append("")
        
        # Group by edge for summary
        edge_summary = df_events.groupby('edge').agg({
            'z_score': ['count', 'max', 'mean'],
            'contribution_value': ['max', 'min']
        }).round(3)
        
        report.append("SUMMARY BY EDGE:")
        report.append("-" * 30)
        for edge in edge_summary.index:
            count = edge_summary.loc[edge, ('z_score', 'count')]
            max_z = edge_summary.loc[edge, ('z_score', 'max')]
            mean_z = edge_summary.loc[edge, ('z_score', 'mean')]
            max_contrib = edge_summary.loc[edge, ('contribution_value', 'max')]
            min_contrib = edge_summary.loc[edge, ('contribution_value', 'min')]
            
            report.append(f"{edge}:")
            report.append(f"  Events: {count}, Max Z-score: {max_z}, Mean Z-score: {mean_z}")
            report.append(f"  Contribution range: [{min_contrib}, {max_contrib}]")
            report.append("")
        
        report.append("\nDETAILED EVENTS (Top 20 by Z-score):")
        report.append("-" * 40)
        
        # Sort by z-score and show top events
        top_events = df_events.nlargest(20, 'z_score')
        
        for _, event in top_events.iterrows():
            effect_time = event['effect_time'].strftime('%Y-%m-%d %H:%M:%S')
            if event['lag'] >= 1:
                source_time = event['source_time'].strftime('%Y-%m-%d %H:%M:%S')
                time_info = f"effect {effect_time} and source {source_time}"
            else:
                time_info = f"time {effect_time}"
            
            report.append(
                f"Edge {event['parent']} → {event['child']} (lag {event['lag']}) "
                f"had anomalous contribution at {time_info}"
            )
            report.append(
                f"  Z-score: {event['z_score']:.2f}, "
                f"Contribution: {event['contribution_value']:.4f}, "
                f"Window: {event['window_idx']}"
            )
            report.append("")
        
        return "\n".join(report)
    
    def plot_contribution_series(self, 
                               edge_key: str,
                               window_idx: int,
                               save_path: Optional[str] = None):
        """
        Plot contribution series for a specific edge with anomaly highlights.
        
        Args:
            edge_key: Edge identifier (e.g., "var1→var2_lag1")
            window_idx: Window index to plot
            save_path: Optional path to save the plot
        """
        # Get window data
        if window_idx not in self.weights_by_window.groups:
            logger.error(f"Window {window_idx} not found")
            return
        
        # Estimate window time range (this is approximate)
        window_size = 100  # Default assumption
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        window_data = self.data.iloc[start_idx:min(end_idx, len(self.data))]
        
        # Compute contributions
        contributions = self.compute_edge_contributions(window_idx, window_data)
        
        if edge_key not in contributions:
            logger.error(f"Edge {edge_key} not found in window {window_idx}")
            return
        
        contrib_series = contributions[edge_key]
        
        # Compute z-scores for highlighting
        valid_contrib = contrib_series.dropna()
        if len(valid_contrib) < 5:
            logger.error("Insufficient data points for plotting")
            return
        
        median_contrib = np.median(valid_contrib)
        mad_contrib = median_abs_deviation(valid_contrib, scale='normal')
        
        if mad_contrib > 1e-12:
            z_scores = np.abs(contrib_series - median_contrib) / mad_contrib
            anomalous_mask = z_scores >= self.z_threshold
        else:
            anomalous_mask = pd.Series(False, index=contrib_series.index)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot contribution series
        plt.plot(contrib_series.index, contrib_series.values, 
                'b-', alpha=0.7, label='Contribution')
        
        # Highlight anomalous points
        if anomalous_mask.any():
            anomalous_points = contrib_series[anomalous_mask]
            plt.scatter(anomalous_points.index, anomalous_points.values,
                       color='red', s=50, zorder=5, label=f'Anomalies (Z≥{self.z_threshold})')
        
        # Add median and MAD lines
        plt.axhline(y=median_contrib, color='green', linestyle='--', 
                   alpha=0.7, label='Median')
        plt.axhline(y=median_contrib + self.z_threshold*mad_contrib, 
                   color='orange', linestyle=':', alpha=0.7, 
                   label=f'+{self.z_threshold}×MAD')
        plt.axhline(y=median_contrib - self.z_threshold*mad_contrib, 
                   color='orange', linestyle=':', alpha=0.7, 
                   label=f'-{self.z_threshold}×MAD')
        
        plt.title(f'Contribution Series: {edge_key} (Window {window_idx})')
        plt.xlabel('Time')
        plt.ylabel('Contribution Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def load_time_series_data(file_path: str) -> pd.DataFrame:
    """Load time series data with proper datetime index."""
    logger.info(f"Loading time series data from {file_path}")
    
    df = pd.read_csv(file_path, index_col=0)
    
    # Try to parse index as datetime
    try:
        df.index = pd.to_datetime(df.index)
    except:
        # If parsing fails, create a time range
        logger.warning("Could not parse datetime index, creating artificial time range")
        start_time = pd.Timestamp('2023-01-01')
        df.index = pd.date_range(start=start_time, periods=len(df), freq='1min')
    
    logger.info(f"Loaded {len(df)} time points with {len(df.columns)} variables")
    return df


def load_weights_data(file_path: str) -> pd.DataFrame:
    """Load edge weights data."""
    logger.info(f"Loading weights data from {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = ['window_idx', 'lag', 'i', 'j', 'weight']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Weights file must contain columns: {required_cols}")
    
    # Filter out zero weights to focus on active edges
    df = df[df['weight'].abs() > 1e-10]
    
    logger.info(f"Loaded {len(df)} non-zero edge weights")
    return df


def main():
    parser = argparse.ArgumentParser(description="Detect anomalous edge contributions over time")
    parser.add_argument("--data", required=True, help="Time series data CSV file")
    parser.add_argument("--weights", required=True, help="Edge weights CSV file")
    parser.add_argument("--variables", nargs="+", required=True, help="Variable names in order")
    parser.add_argument("--window_size", type=int, default=100, help="Window size in time steps")
    parser.add_argument("--overlap", type=int, default=0, help="Window overlap")
    parser.add_argument("--z_threshold", type=float, default=3.0, help="Z-score threshold for anomaly detection")
    parser.add_argument("--output_dir", default=".", help="Output directory for results")
    parser.add_argument("--plot_top_edges", type=int, default=5, help="Plot top N edges by anomaly count")
    
    args = parser.parse_args()
    
    # Load data
    try:
        data = load_time_series_data(args.data)
        weights = load_weights_data(args.weights)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = EdgeContributionAnalyzer(
        time_series_data=data,
        weights_data=weights,
        var_names=args.variables,
        z_threshold=args.z_threshold
    )
    
    # Run analysis
    try:
        df_events = analyzer.analyze_all_windows(
            window_size=args.window_size,
            overlap=args.overlap
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)
    
    # Generate and save report
    report = analyzer.generate_report(df_events)
    print(report)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    report_path = os.path.join(args.output_dir, "anomaly_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    if not df_events.empty:
        events_path = os.path.join(args.output_dir, "anomaly_events.csv")
        df_events.to_csv(events_path, index=False)
        logger.info(f"Events data saved to {events_path}")
        
        # Plot top edges by anomaly count
        if args.plot_top_edges > 0:
            edge_counts = df_events['edge'].value_counts().head(args.plot_top_edges)
            
            for edge in edge_counts.index:
                # Get a representative window for this edge
                edge_events = df_events[df_events['edge'] == edge]
                window_idx = edge_events['window_idx'].iloc[0]
                
                plot_path = os.path.join(args.output_dir, f"contribution_{edge.replace('→', '_to_')}_win{window_idx}.png")
                try:
                    analyzer.plot_contribution_series(edge, window_idx, plot_path)
                except Exception as e:
                    logger.warning(f"Failed to plot {edge}: {e}")


if __name__ == "__main__":
    main()