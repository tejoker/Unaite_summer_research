#!/usr/bin/env python3
"""
Adaptive Window Zooming for Precise Anomaly Localization

After detecting anomalies with large windows, progressively zoom in with smaller
windows to pinpoint the exact temporal location and understand the causal changes.

Strategy:
1. Initial detection: Large windows (100 samples, stride 10) - COARSE detection
2. Anomaly identified in window range [W_start, W_end]
3. Zoom in: Medium windows (50 samples, stride 5) - REFINED localization
4. Further zoom: Small windows (25 samples, stride 2) - PRECISE pinpoint
5. Final zoom: Micro windows (10 samples, stride 1) - EXACT identification

Benefits:
- Better temporal resolution
- Less averaging/dilution of anomaly signal
- Precise root cause identification
- Understanding of anomaly propagation
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from scipy import stats

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveWindowZoomer:
    """
    Performs adaptive window sizing to precisely localize anomalies.
    """

    def __init__(self, workspace_root: str, dataset_length: Optional[int] = None,
                 max_lag: Optional[int] = None):
        """
        Initialize the adaptive window zoomer.

        Args:
            workspace_root: Root directory of the workspace
            dataset_length: Total length of the dataset (for adaptive window sizing)
            max_lag: Maximum lag value from preprocessing (affects minimum window size)
        """
        self.workspace_root = Path(workspace_root)
        self.dataset_length = dataset_length
        self.max_lag = max_lag if max_lag is not None else 1

        # Will be set by compute_adaptive_windows()
        self.window_configs = None

    def compute_adaptive_windows(self, dataset_length: int, max_lag: int) -> List[Dict]:
        """
        Compute adaptive window configurations based on dataset length and lags.

        Key constraints:
        - Minimum window size must be > max_lag to capture temporal dependencies
        - Window size should be proportional to dataset length
        - Need at least 3-4 windows to detect patterns (dataset_length / window_size >= 3)
        - Stride should allow sufficient overlap while being computationally feasible

        Args:
            dataset_length: Total samples in dataset
            max_lag: Maximum lag value (p)

        Returns:
            List of window configurations
        """
        # Minimum viable window size: must exceed max_lag with safety margin
        min_window_size = max(max_lag * 3, 20)  # At least 3x lag or 20 samples

        # Maximum window size: should allow at least 3 windows across dataset
        max_window_size = min(dataset_length // 3, 200)  # Cap at 200 for performance

        if min_window_size >= max_window_size:
            logger.warning(f"Dataset too small for adaptive zooming: length={dataset_length}, "
                         f"max_lag={max_lag}, using minimal configuration")
            # Fallback to single configuration
            window_size = max(min_window_size, dataset_length // 4)
            return [
                {'size': window_size, 'stride': max(1, window_size // 10), 'name': 'single'}
            ]

        # Adaptive strategy: Start large, progressively halve (roughly)
        # Target 4 zoom levels with decreasing sizes
        configs = []

        # Level 1: Coarse (largest - captures global patterns)
        coarse_size = int(max_window_size * 0.8)  # 80% of max
        coarse_stride = max(1, coarse_size // 10)
        configs.append({'size': coarse_size, 'stride': coarse_stride, 'name': 'coarse'})

        # Level 2: Medium (60% of coarse)
        medium_size = int(coarse_size * 0.6)
        if medium_size > min_window_size * 1.5:  # Only add if meaningfully different
            medium_stride = max(1, medium_size // 10)
            configs.append({'size': medium_size, 'stride': medium_stride, 'name': 'medium'})

        # Level 3: Fine (40% of coarse)
        fine_size = int(coarse_size * 0.4)
        if fine_size > min_window_size * 1.2:  # Only add if meaningfully different
            fine_stride = max(1, fine_size // 10)
            configs.append({'size': fine_size, 'stride': fine_stride, 'name': 'fine'})

        # Level 4: Pinpoint (smallest viable - precise root cause)
        pinpoint_size = max(min_window_size, int(coarse_size * 0.25))
        pinpoint_stride = max(1, pinpoint_size // 20)  # Smaller stride for precision
        configs.append({'size': pinpoint_size, 'stride': pinpoint_stride, 'name': 'pinpoint'})

        logger.info(f"Computed {len(configs)} adaptive window configurations:")
        for cfg in configs:
            logger.info(f"  {cfg['name']}: size={cfg['size']}, stride={cfg['stride']}")

        return configs
    
    def find_file_robust(self, directory: Path, pattern: str,
                        file_description: str) -> Optional[Path]:
        """
        Robustly find a file matching a pattern with proper error handling.

        Args:
            directory: Directory to search in
            pattern: Glob pattern (e.g., "*_differenced_stationary_series.csv")
            file_description: Human-readable description for error messages

        Returns:
            Path to the file, or None if not found/error
        """
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return None

        try:
            matches = list(directory.glob(pattern))

            if not matches:
                logger.error(f"{file_description} not found in {directory} (pattern: {pattern})")
                return None

            if len(matches) > 1:
                logger.warning(f"Multiple {file_description} found in {directory}:")
                for m in matches:
                    logger.warning(f"  - {m.name}")
                logger.warning(f"Using first match: {matches[0].name}")

            return matches[0]

        except Exception as e:
            logger.error(f"Error searching for {file_description} in {directory}: {e}")
            return None

    def load_weights(self, weights_file: Path) -> Optional[Dict[int, Dict[str, float]]]:
        """Load weights from CSV and organize by window."""
        weights = {}

        if weights_file is None:
            logger.warning("Weights file path is None")
            return None

        if not weights_file.exists():
            logger.warning(f"Weights file not found: {weights_file}")
            return None

        try:
            df = pd.read_csv(weights_file)

            if df.empty:
                logger.warning(f"Weights file is empty: {weights_file}")
                return None

            # Validate required columns
            required_cols = ['window_idx', 'parent_name', 'child_name', 'weight']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Weights file missing required columns: {missing_cols}")
                return None

            for window_idx in sorted(df['window_idx'].unique()):
                window_data = df[df['window_idx'] == window_idx]
                weights[window_idx] = {}

                for _, row in window_data.iterrows():
                    edge = f"{row['parent_name']}→{row['child_name']}"
                    weights[window_idx][edge] = row['weight']

        except Exception as e:
            logger.error(f"Error loading weights from {weights_file}: {e}")
            return None

        return weights
    
    def compare_windows_statistical(self, golden_weights: Dict, anomaly_weights: Dict,
                                   alpha: float = 0.05) -> List[Tuple[int, float, float]]:
        """
        Compare golden and anomaly weights using statistical significance testing.

        For each window, compute:
        1. Distribution of edge weight differences
        2. Test if the differences are statistically significant (non-zero mean)
        3. Use Wilcoxon signed-rank test (non-parametric, paired samples)

        Args:
            golden_weights: Golden baseline weights by window
            anomaly_weights: Anomaly weights by window
            alpha: Significance level (default: 0.05)

        Returns:
            List of (window_idx, mean_abs_difference, p_value) for significant windows
        """
        changed_windows = []

        for w_idx in sorted(golden_weights.keys()):
            if w_idx not in anomaly_weights:
                continue

            g_weights = golden_weights[w_idx]
            a_weights = anomaly_weights[w_idx]

            # Collect all edge differences for this window
            differences = []
            abs_differences = []

            for edge, g_val in g_weights.items():
                a_val = a_weights.get(edge, 0.0)
                diff = a_val - g_val  # Signed difference
                differences.append(diff)
                abs_differences.append(abs(diff))

            if len(differences) < 3:
                # Not enough edges for statistical test
                # Fallback to simple threshold
                max_diff = max(abs_differences) if abs_differences else 0.0
                if max_diff > 0.01:
                    changed_windows.append((w_idx, max_diff, 0.0))  # p-value=0 indicates no test
                continue

            # Statistical test: Are the differences significantly non-zero?
            # Wilcoxon signed-rank test: tests if median of differences != 0
            try:
                # Remove zeros (edges with identical weights)
                non_zero_diffs = [d for d in differences if abs(d) > 1e-10]

                if len(non_zero_diffs) < 3:
                    # All edges essentially identical
                    continue

                # Perform Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(non_zero_diffs, alternative='two-sided')

                # If statistically significant, add to changed windows
                if p_value < alpha:
                    mean_abs_diff = np.mean(abs_differences)
                    changed_windows.append((w_idx, mean_abs_diff, p_value))
                    logger.debug(f"Window {w_idx}: mean_abs_diff={mean_abs_diff:.4f}, p={p_value:.4e}")

            except Exception as e:
                # Fallback to simple threshold if test fails
                logger.debug(f"Statistical test failed for window {w_idx}: {e}")
                max_diff = max(abs_differences)
                if max_diff > 0.01:
                    changed_windows.append((w_idx, max_diff, 1.0))  # p-value=1 indicates failed test

        return changed_windows

    def compare_windows(self, golden_weights: Dict, anomaly_weights: Dict,
                       threshold: float = 0.01) -> List[Tuple[int, float]]:
        """
        Compare golden and anomaly weights to find changed windows (simple version).
        Returns list of (window_idx, max_difference) tuples.

        Note: Use compare_windows_statistical() for more robust detection.
        """
        changed_windows = []

        for w_idx in sorted(golden_weights.keys()):
            if w_idx not in anomaly_weights:
                continue

            g_weights = golden_weights[w_idx]
            a_weights = anomaly_weights[w_idx]

            max_diff = 0.0
            for edge, g_val in g_weights.items():
                a_val = a_weights.get(edge, 0.0)
                diff = abs(g_val - a_val)
                max_diff = max(max_diff, diff)

            if max_diff > threshold:
                changed_windows.append((w_idx, max_diff))

        return changed_windows
    
    def identify_anomaly_region(self, changed_windows: List[Tuple[int, float]], 
                                window_size: int, stride: int) -> Tuple[int, int]:
        """
        Identify the sample range where anomalies are detected.
        Returns (start_sample, end_sample).
        """
        if not changed_windows:
            return None, None
        
        window_indices = [w for w, _ in changed_windows]
        first_window = min(window_indices)
        last_window = max(window_indices)
        
        # Convert window indices to sample ranges
        start_sample = first_window * stride
        end_sample = last_window * stride + window_size
        
        return start_sample, end_sample
    
    def run_focused_analysis(self, differenced_csv: Path, lags_csv: Path, 
                            output_dir: Path, start_sample: int, end_sample: int,
                            window_size: int, stride: int, lambda_w: float, 
                            lambda_a: float, config_name: str):
        """
        Run causal discovery with smaller windows on a focused region.
        Returns (weights_file_path, region_start_in_original_coords)
        """
        logger.info(f"Running {config_name} analysis on samples {start_sample}-{end_sample}")
        logger.info(f"  Window size: {window_size}, Stride: {stride}")
        
        # Create output directory for this zoom level
        zoom_dir = output_dir / f"zoom_{config_name}"
        zoom_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        df_differenced = pd.read_csv(differenced_csv, index_col=0, parse_dates=True)
        df_lags = pd.read_csv(lags_csv)
        p = int(df_lags['optimal_lag'].max()) if not df_lags.empty else 1
        
        # CRITICAL: Validate minimum window size
        # Need: window_size + p (for lags) + stride×3 (for at least 3 windows) + 5 (safety)
        min_required_samples = window_size + p + (stride * 3) + 5
        region_size = end_sample - start_sample
        
        logger.info(f"  Region size: {region_size} samples (need {min_required_samples} minimum)")
        
        if region_size < min_required_samples:
            logger.warning(f"  ⚠️  Region too small for window_size={window_size}, stride={stride}, p={p}")
            logger.warning(f"  Need at least {min_required_samples} samples. Skipping this zoom level.")
            return None
        
        # Focus on the anomaly region (with VERY generous buffer to avoid edge effects)
        # Buffer should be at least 3× window_size to ensure plenty of room
        # This prevents empty data errors at region boundaries
        buffer = max(window_size * 3, 150)
        region_start = max(0, start_sample - buffer)
        region_end = min(len(df_differenced), end_sample + buffer)
        
        df_region = df_differenced.iloc[region_start:region_end].copy()
        
        # CRITICAL: Reset index to avoid "not strictly increasing" errors
        # The extracted region has non-contiguous indices which breaks DynoTEARS
        df_region.reset_index(drop=True, inplace=True)
        
        logger.info(f"  Extracting region: [{region_start}:{region_end}] = {len(df_region)} samples (buffer={buffer})")
        
        if len(df_region) < min_required_samples:
            logger.warning(f"  Focused region too small ({len(df_region)} samples). Skipping.")
            return None, None
        
        logger.info(f"  Focused region: {len(df_region)} samples (need {min_required_samples} minimum)")
        
        # Import and run DynoTEARS
        sys.path.insert(0, str(self.workspace_root / "executable" / "final_pipeline"))
        from dbn_dynotears_fixed_lambda import run_rolling_window_analysis
        
        # Run analysis with custom parameters
        try:
            run_rolling_window_analysis(
                df_differenced=df_region,
                p=p,
                window_size=window_size,
                stride=stride,
                output_dir=zoom_dir,
                lambda_w=lambda_w,
                lambda_a=lambda_a
            )
        except Exception as e:
            logger.error(f"  Failed to run analysis: {e}")
            return None, None
        
        logger.info(f"  ✅ {config_name} analysis complete: {zoom_dir}")
        
        weights_file = zoom_dir / "weights" / "weights_enhanced.csv"
        if not weights_file.exists():
            logger.warning(f"  Weights file not created: {weights_file}")
            return None, None
            
        return weights_file, region_start
    
    def zoom_analysis(self, golden_results_dir: Path, anomaly_results_dir: Path,
                     output_dir: Path, use_statistical_test: bool = True):
        """
        Perform progressive zooming analysis with adaptive window sizing.

        Args:
            golden_results_dir: Directory with golden baseline results
            anomaly_results_dir: Directory with anomaly detection results
            output_dir: Output directory for zoom analysis
            use_statistical_test: Use statistical testing for anomaly detection (default: True)
        """
        logger.info("="*80)
        logger.info("ADAPTIVE WINDOW ZOOMING FOR ANOMALY LOCALIZATION")
        logger.info("="*80)

        # Get file paths with robust discovery
        logger.info("\nStep 0: Loading input files...")

        # Try preprocessing subdirectory first, then root directory
        golden_diff_csv = self.find_file_robust(
            golden_results_dir / "preprocessing",
            "*_differenced_stationary_series.csv",
            "Golden differenced CSV"
        )
        if not golden_diff_csv:
            golden_diff_csv = self.find_file_robust(
                golden_results_dir,
                "*_differenced_stationary_series.csv",
                "Golden differenced CSV (root)"
            )

        anomaly_diff_csv = self.find_file_robust(
            anomaly_results_dir / "preprocessing",
            "*_differenced_stationary_series.csv",
            "Anomaly differenced CSV"
        )
        if not anomaly_diff_csv:
            anomaly_diff_csv = self.find_file_robust(
                anomaly_results_dir,
                "*_differenced_stationary_series.csv",
                "Anomaly differenced CSV (root)"
            )

        lags_csv = self.find_file_robust(
            golden_results_dir / "preprocessing",
            "*_optimal_lags.csv",
            "Optimal lags CSV"
        )
        if not lags_csv:
            lags_csv = self.find_file_robust(
                golden_results_dir,
                "*_optimal_lags.csv",
                "Optimal lags CSV (root)"
            )

        if not all([golden_diff_csv, anomaly_diff_csv, lags_csv]):
            logger.error("Required input files not found. Aborting.")
            return None

        # Load dataset info and compute adaptive windows
        df_golden = pd.read_csv(golden_diff_csv, index_col=0, parse_dates=True)
        df_lags = pd.read_csv(lags_csv)
        max_lag = int(df_lags['optimal_lag'].max()) if not df_lags.empty else 1
        dataset_length = len(df_golden)

        logger.info(f"Dataset: {dataset_length} samples, max_lag={max_lag}")

        # Compute adaptive window configurations
        self.window_configs = self.compute_adaptive_windows(dataset_length, max_lag)

        if not self.window_configs:
            logger.error("Failed to compute window configurations")
            return None

        # Load initial (coarse) results
        logger.info("\nStep 1: Analyzing coarse detection results...")

        # Try multiple possible locations for weights file
        golden_weights_file = self.find_file_robust(
            golden_results_dir / "weights",
            "weights_enhanced.csv",
            "Golden weights file"
        )
        if not golden_weights_file:
            golden_weights_file = self.find_file_robust(
                golden_results_dir / "causal_discovery" / "weights",
                "weights_enhanced.csv",
                "Golden weights file (alternative location)"
            )

        anomaly_weights_file = self.find_file_robust(
            anomaly_results_dir / "weights",
            "weights_enhanced.csv",
            "Anomaly weights file"
        )
        if not anomaly_weights_file:
            anomaly_weights_file = self.find_file_robust(
                anomaly_results_dir / "causal_discovery" / "weights",
                "weights_enhanced.csv",
                "Anomaly weights file (alternative location)"
            )

        if not all([golden_weights_file, anomaly_weights_file]):
            logger.error("Weights files not found. Aborting.")
            return None

        golden_weights = self.load_weights(golden_weights_file)
        anomaly_weights = self.load_weights(anomaly_weights_file)

        if not golden_weights or not anomaly_weights:
            logger.error("Failed to load weights. Aborting.")
            return None

        # Find changed windows using statistical or simple comparison
        if use_statistical_test:
            logger.info("Using statistical significance testing (Wilcoxon)")
            changed_windows_full = self.compare_windows_statistical(golden_weights, anomaly_weights)
            # Convert to (window_idx, metric) format for compatibility
            changed_windows = [(w, diff) for w, diff, _ in changed_windows_full]
        else:
            logger.info("Using simple threshold-based comparison")
            changed_windows = self.compare_windows(golden_weights, anomaly_weights)

        if not changed_windows:
            logger.warning("No anomalies detected in coarse analysis!")
            return None

        logger.info(f"  Found {len(changed_windows)} changed windows")
        logger.info(f"  Window range: {min(w for w,_ in changed_windows)} to {max(w for w,_ in changed_windows)}")

        # Identify anomaly region
        start_sample, end_sample = self.identify_anomaly_region(
            changed_windows,
            self.window_configs[0]['size'],
            self.window_configs[0]['stride']
        )

        logger.info(f"  Anomaly region: samples {start_sample}-{end_sample}")

        # Load lambda values with error handling
        lambda_file = golden_results_dir / "best_lambdas.json"

        # Try alternative locations for lambda file
        if not lambda_file.exists():
            lambda_file_alt = golden_results_dir / "causal_discovery" / "best_lambdas.json"
            if lambda_file_alt.exists():
                lambda_file = lambda_file_alt

        if lambda_file.exists():
            try:
                with open(lambda_file, 'r') as f:
                    lambdas = json.load(f)
                lambda_w = lambdas['lambda_w']
                lambda_a = lambdas['lambda_a']
                logger.info(f"Loaded lambda values from file: lambda_w={lambda_w}, lambda_a={lambda_a}")
            except Exception as e:
                logger.warning(f"Failed to load lambda values from file: {e}")
                logger.info("Using default lambda values")
                lambda_w = 0.1
                lambda_a = 0.1
        else:
            # Use default values if no lambda file found
            # These are the fixed lambda values from dbn_dynotears_fixed_lambda.py
            logger.warning(f"Lambda file not found: {lambda_file}")
            logger.info("Using default fixed lambda values (lambda_w=0.1, lambda_a=0.1)")
            lambda_w = 0.1
            lambda_a = 0.1

        # Progressive zooming
        results_summary = {
            'coarse': {
                'window_size': int(self.window_configs[0]['size']),
                'stride': int(self.window_configs[0]['stride']),
                'changed_windows': int(len(changed_windows)),
                'sample_range': [int(start_sample), int(end_sample)]
            }
        }
        
        # Zoom through progressively smaller windows
        for i, config in enumerate(self.window_configs[1:], 1):
            logger.info(f"\nStep {i+1}: Zooming to {config['name']} resolution...")
            
            # Run focused analysis on anomaly with this window size
            anomaly_zoom_weights, anomaly_region_start = self.run_focused_analysis(
                differenced_csv=anomaly_diff_csv,
                lags_csv=lags_csv,
                output_dir=output_dir / "anomaly",
                start_sample=start_sample,
                end_sample=end_sample,
                window_size=config['size'],
                stride=config['stride'],
                lambda_w=lambda_w,
                lambda_a=lambda_a,
                config_name=config['name']
            )
            
            # Check if analysis succeeded
            if anomaly_zoom_weights is None:
                logger.warning(f"  Anomaly analysis failed at {config['name']} level - stopping zoom")
                break
            
            # Run same analysis on golden for comparison
            golden_zoom_weights, golden_region_start = self.run_focused_analysis(
                differenced_csv=golden_diff_csv,
                lags_csv=lags_csv,
                output_dir=output_dir / "golden",
                start_sample=start_sample,
                end_sample=end_sample,
                window_size=config['size'],
                stride=config['stride'],
                lambda_w=lambda_w,
                lambda_a=lambda_a,
                config_name=config['name']
            )
            
            # Check if analysis succeeded
            if golden_zoom_weights is None:
                logger.warning(f"  Golden analysis failed at {config['name']} level - stopping zoom")
                break
            
            # Compare zoomed results
            golden_zoom = self.load_weights(golden_zoom_weights)
            anomaly_zoom = self.load_weights(anomaly_zoom_weights)
            
            if golden_zoom is None or anomaly_zoom is None:
                logger.warning(f"  Could not load weights at {config['name']} level - stopping zoom")
                break
            
            changed_zoom = self.compare_windows(golden_zoom, anomaly_zoom)
            
            if changed_zoom:
                zoom_start, zoom_end = self.identify_anomaly_region(
                    changed_zoom, config['size'], config['stride']
                )
                # CORRECT: Add the region_start offset to convert back to original coordinates
                zoom_start += anomaly_region_start
                zoom_end += anomaly_region_start
                
                logger.info(f"  Found {len(changed_zoom)} changed windows")
                logger.info(f"  Refined region: samples {zoom_start}-{zoom_end}")
                
                results_summary[config['name']] = {
                    'window_size': int(config['size']),
                    'stride': int(config['stride']),
                    'changed_windows': int(len(changed_zoom)),
                    'sample_range': [int(zoom_start), int(zoom_end)]
                }
                
                # Update region for next zoom
                start_sample = zoom_start
                end_sample = zoom_end
            else:
                logger.warning(f"  No changes detected at {config['name']} resolution")
                break
        
        # Save summary
        summary_file = output_dir / "zoom_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"\n✅ Zoom analysis complete! Summary: {summary_file}")
        
        return results_summary


def main():
    """Main execution."""
    if len(sys.argv) < 3:
        print("Usage: python adaptive_window_zoom.py <golden_results_dir> <anomaly_results_dir> [output_dir]")
        print()
        print("Example:")
        print("  python adaptive_window_zoom.py \\")
        print("    results/test_20251015/golden \\")
        print("    results/test_20251015/spike_row200 \\")
        print("    results/test_20251015/zoom_analysis")
        sys.exit(1)
    
    golden_dir = Path(sys.argv[1])
    anomaly_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("zoom_analysis")
    
    workspace_root = Path(__file__).parent

    zoomer = AdaptiveWindowZoomer(workspace_root)
    results = zoomer.zoom_analysis(golden_dir, anomaly_dir, output_dir)

    # Print final summary
    if results is None:
        print("\n" + "="*80)
        print("ZOOM ANALYSIS FAILED")
        print("="*80)
        print("\nCheck error messages above for details.")
        sys.exit(1)

    print("\n" + "="*80)
    print("PROGRESSIVE ZOOMING RESULTS")
    print("="*80)
    print()

    for level, data in results.items():
        print(f"{level.upper()} Resolution:")
        print(f"  Window size: {data['window_size']}, Stride: {data['stride']}")
        print(f"  Changed windows: {data['changed_windows']}")
        print(f"  Sample range: {data['sample_range'][0]}-{data['sample_range'][1]}")
        print(f"  Range width: {data['sample_range'][1] - data['sample_range'][0]} samples")
        print()


if __name__ == "__main__":
    main()
