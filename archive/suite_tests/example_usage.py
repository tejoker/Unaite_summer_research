#!/usr/bin/env python3
"""
example_usage.py - Example usage of the Anomaly Detection Suite

This script demonstrates how to use the anomaly detection suite with realistic examples.
Note: This is a demonstration script that will work once dependencies are properly installed.
"""

def create_example_data():
    """Create example data for demonstration."""
    import numpy as np

    # Create a 6x6 baseline weight matrix (typical for Paul Wurth bearing sensors)
    # Variables: [Druckpfannenlager, Exzenterlager, Ständerlager_1, Ständerlager_2, Temperature, Pressure]
    np.random.seed(42)
    W_baseline = np.array([
        [0.0, 0.12, 0.0, 0.0, 0.08, 0.0],      # Druckpfannenlager
        [0.0, 0.0, 0.15, 0.0, 0.0, 0.0],       # Exzenterlager
        [0.0, 0.0, 0.0, 0.11, 0.0, 0.0],       # Ständerlager_1
        [0.0, 0.0, 0.0, 0.0, 0.13, 0.0],       # Ständerlager_2
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.09],       # Temperature
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]         # Pressure
    ])

    # Create normal variation (small random noise)
    W_normal = W_baseline + np.random.randn(6, 6) * 0.005
    np.fill_diagonal(W_normal, 0)  # Ensure no self-loops

    # Create spike anomaly (sudden increase in bearing connection)
    W_spike = W_baseline.copy()
    W_spike[0, 1] += 0.5  # Spike in Druckpfannenlager -> Exzenterlager connection
    W_spike[1, 2] += 0.3  # Secondary spike

    # Create drift anomaly (gradual change)
    W_drift = W_baseline.copy()
    W_drift[2, 3] *= 2.5  # Drift in Ständerlager connection
    W_drift[4, 5] *= 1.8  # Secondary drift

    # Create structural anomaly (topology change)
    W_structural = W_baseline.copy()
    W_structural[0, 3] = 0.2  # New connection: Druckpfannenlager -> Ständerlager_2
    W_structural[1, 2] = 0.0  # Remove existing connection

    variable_names = [
        'Druckpfannenlager',
        'Exzenterlager',
        'Ständerlager_1',
        'Ständerlager_2',
        'Temperature',
        'Pressure'
    ]

    return {
        'baseline': W_baseline,
        'normal': W_normal,
        'spike': W_spike,
        'drift': W_drift,
        'structural': W_structural,
        'variable_names': variable_names
    }

def example_basic_usage():
    """Example 1: Basic usage with the unified suite."""
    print("=== Example 1: Basic Unified Suite Usage ===")

    try:
        from anomaly_detection_suite import UnifiedAnomalyDetectionSuite
        import numpy as np

        # Create example data
        data = create_example_data()

        # Initialize the suite
        suite = UnifiedAnomalyDetectionSuite()

        # Test different scenarios
        scenarios = ['normal', 'spike', 'drift', 'structural']

        for scenario in scenarios:
            print(f"\n--- Testing {scenario} scenario ---")

            result = suite.analyze_single_comparison(
                data['baseline'],
                data[scenario],
                data['variable_names']
            )

            # Extract key results
            is_anomaly = result['phase1_binary_detection']['binary_detection']['is_anomaly']
            ensemble_score = result['phase1_binary_detection']['binary_detection']['ensemble_score']

            print(f"Anomaly detected: {is_anomaly}")
            print(f"Ensemble score: {ensemble_score:.3f}")

            if is_anomaly and 'phase2_classification' in result:
                rule_classification = result['phase2_classification']['rule_based_classification']
                print(f"Classified as: {rule_classification['anomaly_type']} (confidence: {rule_classification['confidence']:.3f})")

                if 'phase3_root_cause' in result:
                    top_edges = result['phase3_root_cause']['edge_attribution']['top_edges'][:2]
                    print("Top contributing edges:")
                    for edge in top_edges:
                        from_var = data['variable_names'][edge['from']]
                        to_var = data['variable_names'][edge['to']]
                        print(f"  {from_var} -> {to_var}: {edge['change']:.4f} ({edge['change_type']})")

        print("\n✅ Basic usage example completed successfully!")

    except Exception as e:
        print(f"❌ Basic usage example failed: {e}")
        print("Make sure all dependencies are installed: pip install numpy pandas scipy scikit-learn")

def example_binary_detection_only():
    """Example 2: Using only Phase 1 (Binary Detection)."""
    print("\n=== Example 2: Binary Detection Only ===")

    try:
        from binary_detection_metrics import compute_binary_detection_suite

        data = create_example_data()

        # Test spike detection
        result = compute_binary_detection_suite(
            data['baseline'],
            data['spike']
        )

        binary_result = result['binary_detection']
        metrics = result['metrics']

        print(f"Anomaly detected: {binary_result['is_anomaly']}")
        print(f"Ensemble score: {binary_result['ensemble_score']:.3f}")
        print(f"Dominant metric: {binary_result['dominant_metric']}")

        print("\nAll metric values:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

        # Show individual contributions
        print("\nMetric contributions:")
        contributions = binary_result['contributions']
        for metric, contrib in contributions.items():
            print(f"  {metric}: {contrib['contribution']:.3f} (weight: {contrib['weight']:.2f})")

        print("\n✅ Binary detection example completed successfully!")

    except Exception as e:
        print(f"❌ Binary detection example failed: {e}")

def example_adaptive_thresholds():
    """Example 3: Using adaptive thresholds with multiple baselines."""
    print("\n=== Example 3: Adaptive Thresholds ===")

    try:
        from binary_detection_metrics import AdaptiveThresholdBootstrap, compute_binary_detection_suite
        import numpy as np

        data = create_example_data()

        # Create multiple golden/baseline matrices (simulating normal variations)
        np.random.seed(123)
        golden_matrices = []
        for i in range(5):
            variation = data['baseline'] + np.random.randn(6, 6) * 0.008
            np.fill_diagonal(variation, 0)
            golden_matrices.append(variation)

        # Compute adaptive thresholds
        threshold_computer = AdaptiveThresholdBootstrap(n_bootstrap=100)  # Reduced for demo
        adaptive_thresholds = threshold_computer.compute_adaptive_thresholds(golden_matrices)

        print("Adaptive thresholds computed:")
        for metric, threshold in adaptive_thresholds.items():
            print(f"  {metric}: {threshold:.4f}")

        # Use adaptive thresholds for detection
        result = compute_binary_detection_suite(
            data['baseline'],
            data['spike'],
            thresholds=adaptive_thresholds
        )

        print(f"\nWith adaptive thresholds:")
        print(f"Anomaly detected: {result['binary_detection']['is_anomaly']}")
        print(f"Ensemble score: {result['binary_detection']['ensemble_score']:.3f}")

        print("\n✅ Adaptive thresholds example completed successfully!")

    except Exception as e:
        print(f"❌ Adaptive thresholds example failed: {e}")

def example_csv_format():
    """Example 4: Working with CSV format (legacy compatibility)."""
    print("\n=== Example 4: CSV Format (Legacy Compatibility) ===")

    try:
        import pandas as pd
        import numpy as np

        # Create synthetic CSV data
        csv_data_1 = []
        csv_data_2 = []

        n_windows = 5
        n_vars = 4
        n_lags = 2

        np.random.seed(42)
        for window in range(n_windows):
            for lag in range(n_lags):
                for i in range(n_vars):
                    for j in range(n_vars):
                        if i != j and np.random.rand() < 0.4:  # Sparse connections

                            # Baseline weights
                            weight1 = np.random.randn() * 0.1
                            csv_data_1.append({
                                'window_idx': window,
                                'lag': lag,
                                'i': i,
                                'j': j,
                                'weight': weight1
                            })

                            # Anomalous weights (with some modifications)
                            weight2 = weight1
                            if i == 1 and j == 2:  # Specific connection gets anomaly
                                weight2 += 0.3 * (window + 1)  # Progressive anomaly

                            csv_data_2.append({
                                'window_idx': window,
                                'lag': lag,
                                'i': i,
                                'j': j,
                                'weight': weight2
                            })

        df1 = pd.DataFrame(csv_data_1)
        df2 = pd.DataFrame(csv_data_2)

        print(f"CSV data created:")
        print(f"  Baseline: {len(df1)} weight entries")
        print(f"  Current: {len(df2)} weight entries")

        # Use legacy function (backward compatible)
        try:
            from frobenius_test import compute_weights_frobenius_distance

            metrics = compute_weights_frobenius_distance(df1, df2)

            print(f"\nLegacy Frobenius analysis results:")
            print(f"  Frobenius distance: {metrics['frobenius_distance']:.6f}")
            print(f"  Normalized distance: {metrics['normalized_frobenius']:.6f}")
            print(f"  Matched entries: {metrics['matched_entries']}")
            print(f"  Pearson correlation: {metrics['pearson_correlation']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")

        except ImportError as e:
            print(f"⚠️  Legacy CSV functions not available: {e}")
            print("This is expected if some dependencies are missing.")

        print("\n✅ CSV format example completed successfully!")

    except Exception as e:
        print(f"❌ CSV format example failed: {e}")

def main():
    """Run all examples."""
    print("🚀 Anomaly Detection Suite - Usage Examples")
    print("=" * 60)

    # Check if basic imports work
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available - install with: pip install numpy")
        return

    try:
        import pandas as pd
        print("✅ Pandas available")
    except ImportError:
        print("❌ Pandas not available - install with: pip install pandas")
        return

    try:
        import scipy
        print("✅ SciPy available")
    except ImportError:
        print("❌ SciPy not available - install with: pip install scipy")
        return

    try:
        import sklearn
        print("✅ Scikit-learn available")
    except ImportError:
        print("❌ Scikit-learn not available - install with: pip install scikit-learn")
        return

    print("✅ All basic dependencies available\n")

    # Run examples
    example_basic_usage()
    example_binary_detection_only()
    example_adaptive_thresholds()
    example_csv_format()

    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\nNext steps:")
    print("1. Adapt these examples to your specific data")
    print("2. Integrate with your Paul Wurth pipeline")
    print("3. Tune thresholds based on your baseline data")
    print("4. Set up continuous monitoring workflow")

if __name__ == "__main__":
    main()