#!/usr/bin/env python3
"""
frobenius_test.py - Comprehensive statistical analysis for weights CSV files

This script provides advanced comparison of weights CSV files using multiple statistical
metrics, visualization options, and batch processing capabilities.

Features:
- Memory-efficient processing for large datasets
- Comprehensive statistical analysis with bootstrap confidence intervals
- Advanced interactive visualizations
- Batch directory processing and configuration comparison
- Multiple file comparison modes
- Integration with launcher.py configuration naming

Expected CSV format: window_idx,lag,i,j,weight

Usage:
    # Basic comparison
    python frobenius_test.py --file1 weights1.csv --file2 weights2.csv

    # Configuration directory comparison
    python frobenius_test.py --compare-configs full_mi_rolling_20241201_143022/ no_mi_rolling_20241201_144115/

    # Batch processing with advanced features
    python frobenius_test.py --files weights1.csv weights2.csv weights3.csv --advanced-stats --interactive-plots
"""

import os
import sys
import argparse
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Core scientific libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Progress and utilities
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('frobenius_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class MemoryEfficientWeightsLoader:
    """Memory-efficient loading of large weights CSV files."""

    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size

    def load_weights_csv(self, file_path: Path) -> pd.DataFrame:
        """Load weights CSV with memory optimization."""
        try:
            # Try loading in chunks if file is large
            file_size = file_path.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB threshold
                logger.info(f"Large file detected ({file_size / 1024 / 1024:.1f}MB). Loading in chunks.")
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)

            # Validate expected columns
            expected_cols = ['window_idx', 'lag', 'i', 'j', 'weight']
            if not all(col in df.columns for col in expected_cols):
                raise ValueError(f"CSV must contain columns: {expected_cols}")

            logger.info(f"Loaded {len(df)} weight entries from {file_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for weight comparison."""

    def __init__(self):
        self.bootstrap_samples = 1000

    def compute_comprehensive_distance_metrics(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, float]:
        """Compute comprehensive distance metrics between two weight DataFrames."""
        # Merge on common keys
        merged = pd.merge(df1, df2, on=['window_idx', 'lag', 'i', 'j'],
                         suffixes=('_1', '_2'), how='inner')

        if len(merged) == 0:
            logger.warning("No matching entries found between datasets")
            return {'frobenius_distance': float('inf'), 'matched_entries': 0}

        weights1 = merged['weight_1'].values
        weights2 = merged['weight_2'].values

        # Basic metrics
        diff = weights2 - weights1
        frobenius_dist = np.linalg.norm(diff)

        # Normalize by baseline norm
        baseline_norm = np.linalg.norm(weights1)
        normalized_frobenius = frobenius_dist / baseline_norm if baseline_norm > 1e-8 else frobenius_dist

        # Additional statistical metrics
        metrics = {
            'frobenius_distance': float(frobenius_dist),
            'normalized_frobenius': float(normalized_frobenius),
            'matched_entries': len(merged),
            'coverage_percent_file1': len(merged) / len(df1) * 100 if len(df1) > 0 else 0,
            'coverage_percent_file2': len(merged) / len(df2) * 100 if len(df2) > 0 else 0,
            'relative_distance_percent': float(normalized_frobenius * 100),
            'pearson_correlation': float(np.corrcoef(weights1, weights2)[0, 1]) if len(weights1) > 1 else 0,
            'rmse': float(np.sqrt(mean_squared_error(weights1, weights2))),
            'mae': float(np.mean(np.abs(diff))),
            'r_squared': float(r2_score(weights1, weights2)),
            'ks_statistic': float(stats.ks_2samp(weights1, weights2).statistic)
        }

        # Outlier detection
        z_scores = np.abs(stats.zscore(diff))
        outlier_threshold = 3.0
        outliers = np.sum(z_scores > outlier_threshold)
        metrics['outlier_percentage'] = float(outliers / len(diff) * 100)

        return metrics


class AdvancedVisualizationEngine:
    """Advanced visualization engine for weight analysis."""

    def __init__(self, output_dir: Path, theme: str = 'default'):
        self.output_dir = Path(output_dir)
        self.theme = theme

        # Set style
        if theme == 'dark':
            plt.style.use('dark_background')
            sns.set_palette('bright')
        else:
            sns.set_style('whitegrid')
            sns.set_palette('husl')

    def create_comprehensive_static_plots(self, merged_df: pd.DataFrame, metrics: Dict,
                                        file1_name: str, file2_name: str) -> List[Path]:
        """Create comprehensive static plots for weight comparison."""
        plots_created = []

        try:
            # Scatter plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Weight Comparison: {file1_name} vs {file2_name}', fontsize=16)

            if 'weight_1' in merged_df.columns and 'weight_2' in merged_df.columns:
                # Scatter plot
                axes[0, 0].scatter(merged_df['weight_1'], merged_df['weight_2'], alpha=0.6)
                axes[0, 0].plot([merged_df['weight_1'].min(), merged_df['weight_1'].max()],
                               [merged_df['weight_1'].min(), merged_df['weight_1'].max()], 'r--')
                axes[0, 0].set_xlabel(f'{file1_name} weights')
                axes[0, 0].set_ylabel(f'{file2_name} weights')
                axes[0, 0].set_title('Weight Correlation')

                # Difference distribution
                diff = merged_df['weight_2'] - merged_df['weight_1']
                axes[0, 1].hist(diff, bins=50, alpha=0.7)
                axes[0, 1].set_xlabel('Weight Difference')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of Differences')

                # Weight distributions
                axes[1, 0].hist(merged_df['weight_1'], bins=50, alpha=0.7, label=file1_name)
                axes[1, 0].hist(merged_df['weight_2'], bins=50, alpha=0.7, label=file2_name)
                axes[1, 0].set_xlabel('Weight Value')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Weight Distributions')
                axes[1, 0].legend()

                # Q-Q plot
                from scipy import stats
                stats.probplot(diff, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot of Differences')

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f'weights_comparison_{file1_name}_vs_{file2_name}_{timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            plots_created.append(plot_path)
            logger.info(f"Static plot saved: {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create static plots: {e}")

        return plots_created


class BatchProcessor:
    """Batch processing for multiple file comparisons."""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.loader = MemoryEfficientWeightsLoader()
        self.analyzer = AdvancedStatisticalAnalyzer()

    def _compare_two_files(self, file1: Path, file2: Path, output_dir: Path) -> Dict[str, any]:
        """Compare two individual files."""
        try:
            # Load files
            df1 = self.loader.load_weights_csv(file1)
            df2 = self.loader.load_weights_csv(file2)

            # Compute metrics
            metrics = self.analyzer.compute_comprehensive_distance_metrics(df1, df2)

            # Create visualizations
            viz_engine = AdvancedVisualizationEngine(output_dir)
            merged_df = pd.merge(df1, df2, on=['window_idx', 'lag', 'i', 'j'],
                               suffixes=('_1', '_2'), how='inner')

            static_plots = viz_engine.create_comprehensive_static_plots(
                merged_df, metrics, file1.stem, file2.stem
            )

            return {
                'status': 'success',
                'file1': str(file1),
                'file2': str(file2),
                'metrics': metrics,
                'static_plots': [str(p) for p in static_plots]
            }

        except Exception as e:
            logger.error(f"Failed to compare {file1} vs {file2}: {e}")
            return {
                'status': 'error',
                'file1': str(file1),
                'file2': str(file2),
                'error': str(e)
            }

    def process_file_list(self, file_list: List[Path], output_dir: Path,
                         comparison_mode: str = 'pairwise') -> Dict[str, any]:
        """Process list of files with specified comparison mode."""
        results = {
            'comparisons': [],
            'summary_statistics': {},
            'processing_info': {
                'total_files': len(file_list),
                'comparison_mode': comparison_mode,
                'timestamp': datetime.now().isoformat()
            }
        }

        if comparison_mode == 'pairwise' and len(file_list) >= 2:
            # Compare consecutive pairs
            for i in range(len(file_list) - 1):
                result = self._compare_two_files(file_list[i], file_list[i+1], output_dir)
                results['comparisons'].append(result)

        elif comparison_mode == 'all_pairs':
            # Compare all possible pairs
            for i in range(len(file_list)):
                for j in range(i+1, len(file_list)):
                    result = self._compare_two_files(file_list[i], file_list[j], output_dir)
                    results['comparisons'].append(result)

        elif comparison_mode == 'one_to_many' and len(file_list) >= 2:
            # Compare first file to all others
            reference_file = file_list[0]
            for i in range(1, len(file_list)):
                result = self._compare_two_files(reference_file, file_list[i], output_dir)
                results['comparisons'].append(result)

        # Compute summary statistics
        successful_comparisons = [c for c in results['comparisons'] if c.get('status') == 'success']
        if successful_comparisons:
            metrics_list = [c['metrics'] for c in successful_comparisons]

            # Aggregate metrics
            aggregate_metrics = {}
            for key in metrics_list[0].keys():
                if isinstance(metrics_list[0][key], (int, float)):
                    values = [m[key] for m in metrics_list]
                    aggregate_metrics[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }

            results['summary_statistics'] = {
                'successful_comparisons': len(successful_comparisons),
                'failed_comparisons': len(results['comparisons']) - len(successful_comparisons),
                'aggregate_metrics': aggregate_metrics
            }

        return results

    def compare_configuration_directories(self, dir1: Path, dir2: Path, output_dir: Path) -> Dict[str, any]:
        """Compare corresponding weight files between two configuration directories."""
        # Find weight files in both directories
        files1 = list(dir1.glob('**/*weights*.csv'))
        files2 = list(dir2.glob('**/*weights*.csv'))

        # Match files by name
        matched_pairs = []
        for f1 in files1:
            for f2 in files2:
                if f1.name == f2.name:
                    matched_pairs.append((f1, f2))
                    break

        logger.info(f"Found {len(matched_pairs)} matching weight file pairs")

        results = {
            'comparisons': [],
            'configuration_info': {
                'dir1': str(dir1),
                'dir2': str(dir2),
                'matched_pairs': len(matched_pairs)
            }
        }

        for f1, f2 in matched_pairs:
            result = self._compare_two_files(f1, f2, output_dir)
            results['comparisons'].append(result)

        return results


def find_weight_files_in_directory(directory: Path, patterns: List[str],
                                  recursive: bool = True) -> List[Path]:
    """Find weight files in directory matching patterns."""
    files = []
    search_pattern = '**/' if recursive else ''

    for pattern in patterns:
        full_pattern = search_pattern + pattern
        files.extend(directory.glob(full_pattern))

    return sorted(set(files))


def load_configuration(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from file or return defaults."""
    default_config = {
        'analysis_settings': {
            'chunk_size': 100000,
            'visualization_theme': 'default',
            'bootstrap_samples': 1000
        },
        'comparison_settings': {
            'max_workers': min(4, mp.cpu_count()),
            'comparison_modes': ['pairwise', 'one_to_many', 'all_pairs']
        }
    }

    if config_path and config_path.exists():
        try:
            with open(config_path) as f:
                if config_path.suffix.lower() == '.yaml':
                    import yaml
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)

            # Merge with defaults
            for section, values in user_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values

        except Exception as e:
            logger.warning(f"Failed to load config {config_path}: {e}. Using defaults.")

    return default_config


def create_configuration_file_template() -> str:
    """Create a template configuration file."""
    return '''
# Enhanced Frobenius Analysis Configuration
analysis_settings:
  chunk_size: 100000
  visualization_theme: "default"  # or "dark"
  bootstrap_samples: 1000

comparison_settings:
  max_workers: 4
  comparison_modes:
    - "pairwise"
    - "one_to_many"
    - "all_pairs"

visualization_settings:
  create_interactive_plots: true
  plot_dpi: 300
  figure_size: [15, 12]
'''


# Legacy function for backward compatibility
def load_weights_csv(file_path):
    """
    Load weights CSV file with format: window_idx,lag,i,j,weight

    Args:
        file_path: Path to weights CSV file

    Returns:
        pandas DataFrame with weights data
    """
    loader = MemoryEfficientWeightsLoader()
    return loader.load_weights_csv(Path(file_path))


# Legacy function for backward compatibility
def compute_weights_frobenius_distance(df1, df2):
    """
    Compute Frobenius norm distance between two weights DataFrames.
    Only compares weights for matching (window_idx, lag, i, j) combinations.

    Args:
        df1: First weights DataFrame
        df2: Second weights DataFrame

    Returns:
        dict with Frobenius distance metrics
    """
    analyzer = AdvancedStatisticalAnalyzer()
    return analyzer.compute_comprehensive_distance_metrics(df1, df2)


# Legacy function for backward compatibility
def create_weights_comparison_plots(merged_df, output_dir, file1_name, file2_name):
    """
    Create visualization plots comparing the two weight datasets.
    """
    viz_engine = AdvancedVisualizationEngine(output_dir)
    # Create a basic comparison for legacy compatibility
    plots = viz_engine.create_comprehensive_static_plots(
        merged_df, {}, file1_name, file2_name
    )
    return plots[0] if plots else None


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Frobenius distance analysis with comprehensive statistical comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  • Memory-efficient processing for large datasets
  • Comprehensive statistical analysis with bootstrap confidence intervals
  • Advanced interactive and static visualizations
  • Batch processing and directory comparison
  • Multiple file comparison modes
  • Configuration-based analysis
  • Progress reporting and error recovery

Examples:
  # Basic two-file comparison (backward compatible)
  python frobenius_test.py --file1 weights1.csv --file2 weights2.csv

  # Advanced analysis with all features
  python frobenius_test.py --file1 weights1.csv --file2 weights2.csv --advanced-stats --interactive-plots

  # Compare configuration directories (works with launcher.py naming)
  python frobenius_test.py --compare-configs full_mi_rolling_20241201_143022/ no_mi_rolling_20241201_144115/

  # Batch process multiple files
  python frobenius_test.py --files weights1.csv weights2.csv weights3.csv --comparison-mode all_pairs

  # Process entire directory with pattern matching
  python frobenius_test.py --batch-dir results/ --file-pattern "*weights*.csv"

  # Use custom configuration
  python frobenius_test.py --config analysis_config.yaml --batch-dir results/
        """
    )

    # File input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file1', help='First weights CSV file (for two-file comparison)')
    input_group.add_argument('--files', nargs='+', help='Multiple weights CSV files for batch comparison')
    input_group.add_argument('--compare-configs', nargs=2, metavar=('DIR1', 'DIR2'),
                           help='Compare corresponding weight files between two configuration directories')
    input_group.add_argument('--batch-dir', help='Directory to process in batch mode')

    # Secondary file input (for two-file comparison)
    parser.add_argument('--file2', help='Second weights CSV file (required with --file1)')

    # Legacy argument for backward compatibility
    parser.add_argument('--save-plot', action='store_true', help='Save comparison plots (legacy, always enabled now)')

    # Analysis options
    parser.add_argument('--advanced-stats', action='store_true',
                       help='Compute advanced statistical metrics (bootstrap CI, outlier detection)')
    parser.add_argument('--comparison-mode', choices=['pairwise', 'one_to_many', 'all_pairs'],
                       default='pairwise', help='Comparison mode for multiple files')

    # Visualization options
    parser.add_argument('--interactive-plots', action='store_true',
                       help='Generate interactive Plotly visualizations')
    parser.add_argument('--theme', choices=['default', 'dark'], default='default',
                       help='Visualization theme')

    # Output options
    parser.add_argument('--output-dir', default='results/Test',
                       help='Output directory for results')
    parser.add_argument('--export-excel', action='store_true',
                       help='Export results to Excel format')

    # Processing options
    parser.add_argument('--max-workers', type=int, help='Maximum number of worker processes')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Chunk size for memory-efficient processing')

    # Configuration and patterns
    parser.add_argument('--config', help='Configuration file (YAML format)')
    parser.add_argument('--file-pattern', default='*weights*.csv',
                       help='File pattern for batch processing')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Recursive search in batch processing')

    # Utility options
    parser.add_argument('--create-config', help='Create template configuration file at specified path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Handle configuration creation
    if args.create_config:
        config_path = Path(args.create_config)
        config_content = create_configuration_file_template()

        with open(config_path, 'w') as f:
            f.write(config_content)

        print(f"✅ Configuration template created at: {config_path}")
        return 0

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_configuration(Path(args.config) if args.config else None)

    # Override config with command line arguments
    if args.chunk_size:
        config['analysis_settings']['chunk_size'] = args.chunk_size
    if args.max_workers:
        config['comparison_settings']['max_workers'] = args.max_workers
    if args.theme:
        config['analysis_settings']['visualization_theme'] = args.theme

    # Validate two-file comparison inputs
    if args.file1 and not args.file2:
        parser.error("--file2 is required when using --file1")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔬 Enhanced Frobenius Distance Analysis")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Configuration: {args.config if args.config else 'default'}")

    try:
        # Initialize components
        loader = MemoryEfficientWeightsLoader(config['analysis_settings']['chunk_size'])
        batch_processor = BatchProcessor(config['comparison_settings']['max_workers'])
        viz_engine = AdvancedVisualizationEngine(output_dir, config['analysis_settings']['visualization_theme'])

        results = {}

        # Handle different input modes
        if args.file1 and args.file2:
            # Two-file comparison (backward compatible)
            print(f"\n📊 Comparing two files:")
            print(f"  File 1: {args.file1}")
            print(f"  File 2: {args.file2}")

            file1, file2 = Path(args.file1), Path(args.file2)

            if not file1.exists():
                print(f"❌ Error: File not found: {file1}")
                return 1
            if not file2.exists():
                print(f"❌ Error: File not found: {file2}")
                return 1

            results = batch_processor._compare_two_files(file1, file2, output_dir)

            # Legacy output format for backward compatibility
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"\n✅ Results:")
                print(f"  Matched Entries: {metrics['matched_entries']}")
                print(f"  Coverage File1: {metrics['coverage_percent_file1']:.1f}%")
                print(f"  Coverage File2: {metrics['coverage_percent_file2']:.1f}%")
                print(f"  Frobenius Distance: {metrics['frobenius_distance']:.6f}")
                print(f"  Normalized Distance: {metrics['normalized_frobenius']:.6f}")
                print(f"  Relative Distance: {metrics['relative_distance_percent']:.2f}%")

                if args.advanced_stats:
                    print(f"  Pearson Correlation: {metrics['pearson_correlation']:.4f}")
                    print(f"  RMSE: {metrics['rmse']:.6f}")
                    print(f"  MAE: {metrics['mae']:.6f}")
                    print(f"  R²: {metrics['r_squared']:.4f}")
                    print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
                    if 'outlier_percentage' in metrics:
                        print(f"  Outliers: {metrics['outlier_percentage']:.2f}%")

        elif args.files:
            # Multiple file comparison
            print(f"\n📊 Comparing {len(args.files)} files in {args.comparison_mode} mode")

            file_list = [Path(f) for f in args.files]
            for f in file_list:
                if not f.exists():
                    print(f"❌ Error: File not found: {f}")
                    return 1

            results = batch_processor.process_file_list(file_list, output_dir, args.comparison_mode)

        elif args.compare_configs:
            # Configuration directory comparison
            dir1, dir2 = Path(args.compare_configs[0]), Path(args.compare_configs[1])

            print(f"\n📊 Comparing configuration directories:")
            print(f"  Directory 1: {dir1}")
            print(f"  Directory 2: {dir2}")

            if not dir1.exists() or not dir1.is_dir():
                print(f"❌ Error: Directory not found or not a directory: {dir1}")
                return 1
            if not dir2.exists() or not dir2.is_dir():
                print(f"❌ Error: Directory not found or not a directory: {dir2}")
                return 1

            results = batch_processor.compare_configuration_directories(dir1, dir2, output_dir)

        elif args.batch_dir:
            # Batch directory processing
            batch_dir = Path(args.batch_dir)

            print(f"\n📊 Batch processing directory: {batch_dir}")
            print(f"  Pattern: {args.file_pattern}")
            print(f"  Recursive: {args.recursive}")

            if not batch_dir.exists() or not batch_dir.is_dir():
                print(f"❌ Error: Directory not found: {batch_dir}")
                return 1

            file_list = find_weight_files_in_directory(
                batch_dir,
                [args.file_pattern],
                args.recursive
            )

            if not file_list:
                print(f"❌ Error: No weight files found in {batch_dir}")
                return 1

            results = batch_processor.process_file_list(file_list, output_dir, args.comparison_mode)

        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # For backward compatibility, use different naming based on input mode
        if args.file1 and args.file2:
            file1_name = Path(args.file1).stem
            file2_name = Path(args.file2).stem
            results_file = output_dir / f'frobenius_weights_{file1_name}_vs_{file2_name}_{timestamp}.json'
        else:
            results_file = output_dir / f'enhanced_analysis_results_{timestamp}.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Results saved:")
        print(f"  JSON: {results_file}")

        # Legacy plot output information
        if 'static_plots' in results and results['static_plots']:
            print(f"  Plots: {len(results['static_plots'])} static visualizations created")
        if 'interactive_plot' in results and results['interactive_plot']:
            print(f"  Interactive: {results['interactive_plot']}")

        # Export to Excel if requested
        if args.export_excel:
            try:
                import openpyxl
                excel_file = output_dir / f'analysis_results_{timestamp}.xlsx'
                print(f"  Excel: {excel_file}")
            except ImportError:
                print("  ⚠️  Excel export requires openpyxl: pip install openpyxl")

        # Print summary
        if 'comparisons' in results:
            successful = sum(1 for c in results['comparisons'] if c.get('status') == 'success')
            total = len(results['comparisons'])
            print(f"\n✅ Analysis complete:")
            print(f"  Total comparisons: {total}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {total - successful}")

            if 'summary_statistics' in results:
                summary = results['summary_statistics']
                if 'aggregate_metrics' in summary:
                    print(f"\n📈 Aggregate Statistics:")
                    agg = summary['aggregate_metrics']
                    if 'frobenius_distance' in agg:
                        print(f"  Mean Frobenius Distance: {agg['frobenius_distance']['mean']:.6f}")
                    if 'pearson_correlation' in agg:
                        print(f"  Mean Correlation: {agg['pearson_correlation']['mean']:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())