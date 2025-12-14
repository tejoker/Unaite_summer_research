#!/usr/bin/env python3
"""
Visualize Anomaly Detection Results

Creates plots showing:
1. Detection timeline (status over time)
2. Metric evolution (abs_score, change_score, abs_trend)
3. Threshold adaptation
4. Comparison with labeled anomalies (if available)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_labeled_anomalies(labels_csv: str) -> List[Tuple[int, int]]:
    """
    Load labeled anomalies from Telemanom labeled_anomalies.csv.

    Args:
        labels_csv: Path to labeled_anomalies.csv

    Returns:
        List of (start, end) tuples
    """
    df = pd.read_csv(labels_csv)

    anomalies = []
    for _, row in df.iterrows():
        sequences_str = row['anomaly_sequences']
        sequences_str = sequences_str.replace('[', '').replace(']', '')
        if not sequences_str.strip():
            continue

        values = [int(x.strip()) for x in sequences_str.split(',') if x.strip()]
        for i in range(0, len(values), 2):
            if i + 1 < len(values):
                start = values[i]
                end = values[i + 1]
                anomalies.append((start, end))

    return anomalies


def visualize_anomaly_detection(
    results_csv: str,
    output_dir: str,
    labels_csv: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Create comprehensive visualization of anomaly detection results.

    Args:
        results_csv: Path to anomaly_detection_results.csv
        output_dir: Output directory for plots
        labels_csv: Optional path to labeled_anomalies.csv
        figsize: Figure size (width, height)
    """
    logger.info("="*80)
    logger.info("ANOMALY DETECTION VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Results: {results_csv}")
    logger.info(f"Output dir: {output_dir}")

    # Load results
    df = pd.read_csv(results_csv)
    logger.info(f"Loaded {len(df)} windows")

    # Load labels if provided
    labeled_ranges = []
    if labels_csv and Path(labels_csv).exists():
        labeled_ranges = load_labeled_anomalies(labels_csv)
        logger.info(f"Loaded {len(labeled_ranges)} labeled anomaly ranges")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, height_ratios=[1.5, 1, 1, 1], hspace=0.3)

    # Color map for status
    status_colors = {
        'NORMAL': '#2ecc71',  # Green
        'NEW_ANOMALY_ONSET': '#e74c3c',  # Red
        'RECOVERY_FLUCTUATION': '#f39c12',  # Orange
        'CASCADE_OR_PERSISTENT': '#9b59b6'  # Purple
    }

    # ========================================================================
    # Plot 1: Detection Timeline with Labeled Anomalies
    # ========================================================================
    ax1 = fig.add_subplot(gs[0])

    # Plot labeled anomalies as background spans
    if labeled_ranges:
        for start, end in labeled_ranges:
            ax1.axvspan(start, end, alpha=0.2, color='red', label='_Labeled Anomaly')

    # Plot detection status
    for status, color in status_colors.items():
        df_status = df[df['status'] == status]
        if len(df_status) > 0:
            ax1.scatter(df_status['t_center'], [status]*len(df_status),
                       c=color, alpha=0.7, s=20, label=status)

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Status')
    ax1.set_title('Anomaly Detection Timeline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # Add labeled anomaly patch to legend if present
    if labeled_ranges:
        labeled_patch = mpatches.Patch(color='red', alpha=0.2, label='Labeled Anomaly')
        handles, labels = ax1.get_legend_handles_labels()
        handles.append(labeled_patch)
        labels.append('Labeled Anomaly')
        ax1.legend(handles, labels, loc='upper right', fontsize=8)

    # ========================================================================
    # Plot 2: Absolute Score Evolution
    # ========================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot abs_score
    ax2.plot(df['t_center'], df['abs_score'], 'b-', linewidth=1.5, label='abs_score', alpha=0.7)

    # Plot threshold
    ax2.plot(df['t_center'], df['threshold_normal'], 'r--', linewidth=1, label='threshold_normal', alpha=0.5)

    # Highlight anomalies
    df_anomalies = df[df['status'] != 'NORMAL']
    if len(df_anomalies) > 0:
        ax2.scatter(df_anomalies['t_center'], df_anomalies['abs_score'],
                   c='red', s=30, alpha=0.6, label='Detected Anomaly', zorder=5)

    ax2.set_ylabel('Absolute Score')
    ax2.set_title('Absolute Deviation from Golden Baseline', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # ========================================================================
    # Plot 3: Change Score Evolution
    # ========================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Plot change_score
    ax3.plot(df['t_center'], df['change_score'], 'g-', linewidth=1.5, label='change_score', alpha=0.7)

    # Plot threshold
    ax3.plot(df['t_center'], df['threshold_change'], 'r--', linewidth=1, label='threshold_change', alpha=0.5)

    # Highlight high change events
    df_high_change = df[df['change_score'] > df['threshold_change']]
    if len(df_high_change) > 0:
        ax3.scatter(df_high_change['t_center'], df_high_change['change_score'],
                   c='orange', s=30, alpha=0.6, label='High Change', zorder=5)

    ax3.set_ylabel('Change Score')
    ax3.set_title('Rate of Change from Previous Window', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)

    # ========================================================================
    # Plot 4: Trend Evolution
    # ========================================================================
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # Plot trend
    ax4.plot(df['t_center'], df['abs_trend'], 'purple', linewidth=1.5, label='abs_trend', alpha=0.7)

    # Plot threshold bands
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax4.fill_between(df['t_center'], -df['threshold_trend'], df['threshold_trend'],
                      alpha=0.2, color='gray', label='Normal Variation')

    # Highlight positive trend (degrading)
    df_degrading = df[df['abs_trend'] > df['threshold_trend']]
    if len(df_degrading) > 0:
        ax4.scatter(df_degrading['t_center'], df_degrading['abs_trend'],
                   c='red', s=30, alpha=0.6, label='Degrading', zorder=5)

    # Highlight negative trend (improving)
    df_improving = df[df['abs_trend'] < -df['threshold_trend']]
    if len(df_improving) > 0:
        ax4.scatter(df_improving['t_center'], df_improving['abs_trend'],
                   c='green', s=30, alpha=0.6, label='Improving', zorder=5)

    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Trend')
    ax4.set_title('Trend in Absolute Score (Positive = Degrading, Negative = Improving)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=8)

    # ========================================================================
    # Save Figure
    # ========================================================================
    plt.tight_layout()

    output_path = Path(output_dir) / 'anomaly_detection_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to: {output_path}")

    plt.close()

    # ========================================================================
    # Create Status Distribution Plot
    # ========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Status distribution
    status_counts = df['status'].value_counts()
    colors = [status_colors.get(status, 'gray') for status in status_counts.index]

    ax1.bar(range(len(status_counts)), status_counts.values, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(status_counts)))
    ax1.set_xticklabels(status_counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Count')
    ax1.set_title('Status Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add percentages
    total = len(df)
    for i, count in enumerate(status_counts.values):
        pct = 100 * count / total
        ax1.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    # Metric statistics
    metrics_data = {
        'abs_score': df['abs_score'].values,
        'change_score': df['change_score'].values,
        'abs_trend': df['abs_trend'].values
    }

    bp = ax2.boxplot(metrics_data.values(), labels=metrics_data.keys(), patch_artist=True)
    for patch, color in zip(bp['boxes'], ['blue', 'green', 'purple']):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax2.set_ylabel('Value')
    ax2.set_title('Metric Distributions', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path2 = Path(output_dir) / 'anomaly_detection_statistics.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    logger.info(f"Saved statistics to: {output_path2}")

    plt.close()

    logger.info("="*80)
    logger.info("Visualization complete!")
    logger.info("="*80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize Anomaly Detection Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates comprehensive visualizations of anomaly detection results including:
- Detection timeline with labeled anomalies overlay
- Metric evolution (abs_score, change_score, abs_trend)
- Adaptive threshold tracking
- Status distribution and metric statistics

Examples:
  # Basic visualization
  python visualize_anomaly_detection.py \\
    --results results/anomaly_detection/anomaly_detection_results.csv \\
    --output results/anomaly_detection/plots

  # With labeled anomalies overlay
  python visualize_anomaly_detection.py \\
    --results results/anomaly_detection/anomaly_detection_results.csv \\
    --labels telemanom/labeled_anomalies.csv \\
    --output results/anomaly_detection/plots
        """
    )

    parser.add_argument('--results', required=True, help='Path to anomaly_detection_results.csv')
    parser.add_argument('--output', required=True, help='Output directory for plots')
    parser.add_argument('--labels', help='Optional path to labeled_anomalies.csv')
    parser.add_argument('--figsize', nargs=2, type=int, default=[16, 12],
                       help='Figure size (width height), default: 16 12')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.results).exists():
        logger.error(f"Results file not found: {args.results}")
        return 1

    if args.labels and not Path(args.labels).exists():
        logger.warning(f"Labels file not found: {args.labels}")
        args.labels = None

    # Create visualization
    visualize_anomaly_detection(
        results_csv=args.results,
        output_dir=args.output,
        labels_csv=args.labels,
        figsize=tuple(args.figsize)
    )

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
