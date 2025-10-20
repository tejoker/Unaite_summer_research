#!/usr/bin/env python3
"""
analyze_launcher_results.py - Apply graph_structure_detector to launcher.py results

This script adapts graph_structure_detector.py to work with the directory structure
created by launcher.py (full_mi_rolling_*, no_mi_rolling_*, etc.) instead of the
launch2_* pattern.

Usage:
    python analyze_launcher_results.py --results-dir /path/to/results
    python analyze_launcher_results.py --results-dir /path/to/results --baseline-dir full_mi_rolling_20250929_133635
    python analyze_launcher_results.py --results-dir /path/to/results --edge-threshold 0.05
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Import the graph structure detector classes
from graph_structure_detector import (
    GraphStructureDetector,
    WeightMatrixLoader,
    StructuralSignatureAnalyzer,
    StructuralVisualizationEngine,
    setup_logging
)

class LauncherResultsLoader(WeightMatrixLoader):
    """Adapted loader for launcher.py result directories"""

    def discover_launches(self, results_dir: Path, baseline_pattern: str = "full_mi_rolling") -> dict:
        """
        Discover launcher result directories and categorize them.

        Args:
            results_dir: Path to results directory
            baseline_pattern: Pattern for baseline directories (default: "full_mi_rolling")

        Returns:
            Dict with 'golden' and 'anomalies' keys containing launch info
        """
        self.logger.info(f"Discovering launcher results in {results_dir}")

        launches = {'golden': None, 'anomalies': []}

        # Pattern mapping for launcher directory types
        config_patterns = {
            'full_mi_rolling': 'full',  # Baseline: MI masking + rolling windows
            'no_mi_rolling': 'no_mi',   # No MI masking
            'mi_no_rolling': 'no_rolling',  # No rolling windows
            'baseline_no_mi_no_rolling': 'baseline'  # Neither MI nor rolling
        }

        # Find all launcher result directories
        launcher_dirs = []
        for item in results_dir.iterdir():
            if item.is_dir():
                # Check if directory name matches any config pattern
                for pattern in config_patterns.keys():
                    if item.name.startswith(pattern):
                        launcher_dirs.append(item)
                        break

        self.logger.info(f"Found {len(launcher_dirs)} launcher result directories")

        # Process each launcher directory
        for launcher_dir in sorted(launcher_dirs):
            launch_info = self._analyze_launcher_directory(launcher_dir, baseline_pattern)

            if launch_info:
                if launch_info['is_golden']:
                    launches['golden'] = launch_info
                    self.logger.info(f"Found golden baseline: {launch_info['launch_name']}")
                else:
                    launches['anomalies'].append(launch_info)
                    self.logger.info(f"Found anomaly launch: {launch_info['launch_name']} "
                                   f"({launch_info['anomaly_type']})")

        if not launches['golden']:
            self.logger.warning("No golden baseline found, using first directory as baseline")
            if launcher_dirs:
                # Use first directory as baseline if no explicit baseline found
                launch_info = self._analyze_launcher_directory(launcher_dirs[0], baseline_pattern)
                if launch_info:
                    launch_info['is_golden'] = True
                    launches['golden'] = launch_info

        self.logger.info(f"Summary: {'1 golden baseline' if launches['golden'] else 'No baseline'}, "
                        f"{len(launches['anomalies'])} anomaly launches")
        return launches

    def _analyze_launcher_directory(self, launcher_dir: Path, baseline_pattern: str) -> dict:
        """Analyze a launcher directory structure"""
        launch_name = launcher_dir.name

        # Find weight files in subdirectories (Anomaly/Golden/etc.)
        weight_files = list(launcher_dir.rglob("*weights*.csv"))

        # Exclude history files, prefer final weight files
        weight_files = [f for f in weight_files if 'history' not in str(f)]

        if not weight_files:
            self.logger.warning(f"No weight files found in {launch_name}")
            return None

        # Use the first main weight file found (usually weights_enhanced_*.csv)
        weight_file = weight_files[0]

        # Determine if this is baseline or variant
        is_golden = launch_name.startswith(baseline_pattern)

        # Extract configuration type from directory name
        config_type = self._extract_config_type(launch_name)

        # Extract timestamp
        timestamp = self._extract_timestamp(launch_name)

        return {
            'launch_name': launch_name,
            'launch_dir': launcher_dir,
            'weight_file': weight_file,
            'is_golden': is_golden,
            'anomaly_type': config_type if not is_golden else None,
            'timestamp': timestamp,
            'config_type': config_type
        }

    def _extract_config_type(self, launch_name: str) -> str:
        """Extract configuration type from launcher directory name"""
        if launch_name.startswith('full_mi_rolling'):
            return 'full_mi_rolling'
        elif launch_name.startswith('no_mi_rolling'):
            return 'no_mi_rolling'
        elif launch_name.startswith('mi_no_rolling'):
            return 'mi_no_rolling'
        elif launch_name.startswith('baseline_no_mi_no_rolling'):
            return 'baseline_no_mi_no_rolling'
        else:
            return 'unknown'

    def _extract_timestamp(self, launch_name: str) -> str:
        """Extract timestamp from launcher directory name"""
        # full_mi_rolling_20250929_133635 -> 20250929_133635
        parts = launch_name.split('_')
        if len(parts) >= 2:
            # Last two parts should be date and time
            return '_'.join(parts[-2:])
        return launch_name


class LauncherGraphStructureDetector(GraphStructureDetector):
    """Graph structure detector adapted for launcher.py results"""

    def __init__(self, results_dir: Path, baseline_pattern: str = "full_mi_rolling",
                 edge_threshold: float = 0.01, output_dir: Path = None,
                 create_visualizations: bool = True, verbose: bool = False):

        self.results_dir = Path(results_dir)
        self.baseline_pattern = baseline_pattern
        self.edge_threshold = edge_threshold
        self.create_visualizations = create_visualizations

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"launcher_structural_analysis_{timestamp}")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.output_dir, verbose)

        # Initialize components with adapted loader
        self.loader = LauncherResultsLoader(self.logger)
        self.analyzer = StructuralSignatureAnalyzer(edge_threshold, self.logger)
        if self.create_visualizations:
            self.viz_engine = StructuralVisualizationEngine(self.output_dir, self.logger)

    def run_analysis(self) -> dict:
        """Run the complete structural analysis on launcher results"""
        self.logger.info("Starting launcher results structural analysis")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Baseline pattern: {self.baseline_pattern}")
        self.logger.info(f"Edge threshold: {self.edge_threshold}")
        self.logger.info(f"Output directory: {self.output_dir}")

        try:
            # Step 1: Discover launcher directories
            launches = self.loader.discover_launches(self.results_dir, self.baseline_pattern)

            if not launches['golden']:
                raise ValueError("No baseline configuration found")

            # Step 2: Load baseline
            self.logger.info("Loading baseline weight matrix...")
            baseline_matrix = self.loader.load_weight_matrix(launches['golden']['weight_file'])
            baseline_signature = self.analyzer.compute_structural_signature(baseline_matrix)

            # Step 3: Analyze variants/anomalies
            analysis_results = {
                'baseline': {
                    'launch': launches['golden']['launch_name'],
                    'timestamp': launches['golden']['timestamp'],
                    'config_type': launches['golden']['config_type'],
                    'weight_file': str(launches['golden']['weight_file']),
                    'edge_count': baseline_signature['edge_count'],
                    'density': baseline_signature['density'],
                    'structure': baseline_signature
                },
                'variants': [],
                'summary': {
                    'total_variants': len(launches['anomalies']),
                    'edge_threshold': self.edge_threshold,
                    'baseline_pattern': self.baseline_pattern,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }

            self.logger.info(f"Analyzing {len(launches['anomalies'])} variant configurations...")

            from tqdm import tqdm
            for variant_info in tqdm(launches['anomalies'], desc="Analyzing variants"):
                try:
                    # Load variant matrix
                    variant_matrix = self.loader.load_weight_matrix(variant_info['weight_file'])
                    variant_signature = self.analyzer.compute_structural_signature(variant_matrix)

                    # Detect changes
                    change_analysis = self.analyzer.detect_structural_changes(
                        baseline_signature, variant_signature
                    )
                    weight_summary = self.analyzer.compute_weight_summary(
                        baseline_matrix, variant_matrix
                    )

                    # Store results
                    variant_result = {
                        'launch': variant_info['launch_name'],
                        'timestamp': variant_info['timestamp'],
                        'config_type': variant_info['config_type'],
                        'weight_file': str(variant_info['weight_file']),
                        'structural_change': change_analysis,
                        'weight_summary': weight_summary,
                        'structure': variant_signature
                    }

                    analysis_results['variants'].append(variant_result)

                    # Create individual visualization
                    if self.create_visualizations:
                        self.viz_engine.create_structure_diff_plot(
                            baseline_signature, variant_signature, change_analysis,
                            {'anomaly_type': variant_info['config_type'],
                             'timestamp': variant_info['timestamp']}
                        )

                    self.logger.info(f"Analyzed {variant_info['launch_name']}: "
                                   f"{variant_info['config_type']} - "
                                   f"SHD={change_analysis['shd']}, "
                                   f"Pattern={change_analysis['pattern']}")

                except Exception as e:
                    self.logger.error(f"Failed to analyze {variant_info['launch_name']}: {e}")
                    continue

            # Step 4: Create visualizations
            if self.create_visualizations:
                self.logger.info("Creating visualizations...")
                self.viz_engine.create_baseline_structure_plot(
                    baseline_signature, launches['golden']
                )
                # Adapt summary visualization for launcher results
                self._create_launcher_summary_visualization(analysis_results)

            # Step 5: Generate reports
            self.logger.info("Generating reports...")
            self._generate_json_report(analysis_results)
            self._generate_csv_summary(analysis_results)

            # Step 6: Print summary
            self._print_launcher_summary(analysis_results)

            self.logger.info(f"Analysis complete! Results saved to: {self.output_dir}")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise

    def _create_launcher_summary_visualization(self, analysis_results: dict):
        """Create summary visualization adapted for launcher results"""
        import matplotlib.pyplot as plt
        import numpy as np

        variants = analysis_results['variants']

        if not variants:
            self.logger.warning("No variants to visualize")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data
        config_types = [v['config_type'] for v in variants]
        shd_values = [v['structural_change']['shd'] for v in variants]
        patterns = [v['structural_change']['pattern'] for v in variants]
        frobenius_distances = [v['weight_summary']['frobenius_distance'] for v in variants]

        # Plot 1: SHD by configuration type
        import pandas as pd
        unique_configs = list(set(config_types))
        shd_by_config = {c: [shd for i, shd in enumerate(shd_values) if config_types[i] == c]
                        for c in unique_configs}

        axes[0, 0].bar(range(len(unique_configs)),
                      [np.mean(shd_by_config[c]) for c in unique_configs])
        axes[0, 0].set_xticks(range(len(unique_configs)))
        axes[0, 0].set_xticklabels(unique_configs, rotation=45, ha='right')
        axes[0, 0].set_title('Average SHD by Configuration Type')
        axes[0, 0].set_ylabel('Structural Hamming Distance')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Pattern distribution
        pattern_counts = pd.Series(patterns).value_counts()
        axes[0, 1].pie(pattern_counts.values, labels=pattern_counts.index,
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Distribution of Change Patterns')

        # Plot 3: SHD vs Frobenius distance
        colors = ['red' if 'no_mi' in ct else 'blue' if 'no_rolling' in ct else 'green'
                 for ct in config_types]
        axes[1, 0].scatter(frobenius_distances, shd_values, c=colors, alpha=0.7, s=100)
        for i, ct in enumerate(config_types):
            axes[1, 0].annotate(ct.split('_')[0][:4],
                               (frobenius_distances[i], shd_values[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Frobenius Distance')
        axes[1, 0].set_ylabel('Structural Hamming Distance')
        axes[1, 0].set_title('Weight Changes vs Structural Changes by Configuration')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Configuration comparison
        x_pos = range(len(variants))
        bars = axes[1, 1].bar(x_pos, shd_values, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Configuration Variant')
        axes[1, 1].set_ylabel('SHD')
        axes[1, 1].set_title('Structural Changes by Configuration Variant')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([ct[:15] for ct in config_types],
                                   rotation=45, ha='right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.output_dir / 'plots' / 'launcher_analysis_summary.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created launcher summary visualization: {plot_path}")
        return plot_path

    def _print_launcher_summary(self, results: dict):
        """Print analysis summary for launcher results"""
        print(f"\n{'='*60}")
        print(f"LAUNCHER RESULTS STRUCTURAL ANALYSIS SUMMARY")
        print(f"{'='*60}")

        baseline = results['baseline']
        print(f"Baseline Configuration: {baseline['config_type']}")
        print(f"  Launch: {baseline['launch']}")
        print(f"  Edges: {baseline['edge_count']}")
        print(f"  Density: {baseline['density']:.3f}")

        variants = results['variants']
        print(f"\nAnalyzed {len(variants)} configuration variants:")

        # Count by configuration type
        config_counts = {}
        pattern_counts = {}
        changes_detected = 0

        for variant in variants:
            config = variant['config_type']
            pattern = variant['structural_change']['pattern']

            config_counts[config] = config_counts.get(config, 0) + 1
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            if variant['structural_change']['has_change']:
                changes_detected += 1

        print(f"\nStructural changes detected: {changes_detected}/{len(variants)} "
              f"({100*changes_detected/max(len(variants), 1):.1f}%)")

        print(f"\nChange patterns:")
        for pattern, count in sorted(pattern_counts.items()):
            print(f"  {pattern}: {count}")

        print(f"\nConfiguration types:")
        for config, count in sorted(config_counts.items()):
            print(f"  {config}: {count}")

        print(f"\nTop structural changes (by SHD):")
        sorted_variants = sorted(variants,
                                key=lambda x: x['structural_change']['shd'],
                                reverse=True)
        for i, variant in enumerate(sorted_variants[:5]):
            shd = variant['structural_change']['shd']
            pattern = variant['structural_change']['pattern']
            config = variant['config_type']
            print(f"  {i+1}. {config} (SHD={shd}, {pattern})")

        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*60}")


def main():
    """Main entry point for launcher results analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze structural changes in launcher.py results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze all launcher results
  python analyze_launcher_results.py --results-dir /path/to/results

  # Specify baseline configuration pattern
  python analyze_launcher_results.py --results-dir ./results --baseline-pattern full_mi_rolling

  # Custom threshold and no visualizations
  python analyze_launcher_results.py --results-dir ./results --edge-threshold 0.05 --no-viz

  # Verbose output with custom output directory
  python analyze_launcher_results.py --results-dir ./results --output-dir ./analysis --verbose
        """
    )

    parser.add_argument('--results-dir', required=True, type=str,
                       help='Directory containing launcher result directories')
    parser.add_argument('--baseline-pattern', type=str, default='full_mi_rolling',
                       help='Pattern for baseline directories (default: full_mi_rolling)')
    parser.add_argument('--edge-threshold', type=float, default=0.01,
                       help='Threshold for edge existence (default: 0.01)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: launcher_structural_analysis_TIMESTAMP)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
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
        detector = LauncherGraphStructureDetector(
            results_dir=results_dir,
            baseline_pattern=args.baseline_pattern,
            edge_threshold=args.edge_threshold,
            output_dir=args.output_dir,
            create_visualizations=not args.no_viz,
            verbose=args.verbose
        )

        # Run analysis
        results = detector.run_analysis()

        print(f"\nAnalysis completed successfully!")
        print(f"Results available in: {detector.output_dir}")

        return 0

    except Exception as e:
        print(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())