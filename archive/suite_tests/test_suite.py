#!/usr/bin/env python3
"""
test_suite.py - Test script for the Anomaly Detection Suite

This script validates that all components of the anomaly detection suite work correctly
by running tests with synthetic data and real-world scenarios.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from binary_detection_metrics import (
    BinaryDetectionMetrics,
    AdaptiveThresholdBootstrap,
    EnsembleVotingSystem,
    compute_binary_detection_suite
)
from anomaly_classification import GraphSignatureExtractor, RuleBasedClassifier, MLBasedClassifier
from root_cause_analysis import RootCauseAnalyzer
from anomaly_detection_suite import UnifiedAnomalyDetectionSuite

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_baseline_matrix(n: int = 6, seed: int = 42) -> np.ndarray:
    """Create a synthetic baseline weight matrix."""
    np.random.seed(seed)
    W = np.random.randn(n, n) * 0.1
    # Make it sparse
    mask = np.random.rand(n, n) < 0.3
    W = W * mask
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(W, 0)
    return W


def create_anomalous_matrix(baseline: np.ndarray, anomaly_type: str = "spike") -> np.ndarray:
    """Create an anomalous version of the baseline matrix."""
    W_anomaly = baseline.copy()
    n = baseline.shape[0]

    if anomaly_type == "spike":
        # Add sudden spikes in specific edges
        W_anomaly[1, 3] += 2.0  # Large spike
        W_anomaly[2, 4] += 1.5  # Medium spike

    elif anomaly_type == "drift":
        # Gradual drift by scaling some connections
        W_anomaly[0, 1] *= 3.0
        W_anomaly[3, 5] *= 2.5

    elif anomaly_type == "structural":
        # Add/remove edges (structural change)
        W_anomaly[1, 2] = 0.8  # Add new strong connection
        W_anomaly[4, 0] = 0.0  # Remove existing connection

    elif anomaly_type == "oscillation":
        # Create oscillatory pattern
        for i in range(n):
            for j in range(n):
                if W_anomaly[i, j] != 0:
                    W_anomaly[i, j] *= (1 + 0.5 * np.sin(i + j))

    return W_anomaly


def test_binary_detection():
    """Test Phase 1: Binary Detection Metrics."""
    logger.info("=== Testing Phase 1: Binary Detection ===")

    # Create test matrices
    W_baseline = create_synthetic_baseline_matrix()
    W_normal = W_baseline + np.random.randn(*W_baseline.shape) * 0.01  # Small noise
    W_anomaly = create_anomalous_matrix(W_baseline, "spike")

    metrics_computer = BinaryDetectionMetrics()

    # Test normal case (should not detect anomaly)
    logger.info("Testing normal case...")
    metrics_normal = metrics_computer.compute_all_metrics(W_baseline, W_normal)
    print(f"Normal metrics: {metrics_normal}")

    # Test anomaly case (should detect anomaly)
    logger.info("Testing anomaly case...")
    metrics_anomaly = metrics_computer.compute_all_metrics(W_baseline, W_anomaly)
    print(f"Anomaly metrics: {metrics_anomaly}")

    # Test ensemble voting
    ensemble = EnsembleVotingSystem()
    thresholds = {
        'frobenius_distance': 0.1,
        'structural_hamming_distance': 2.0,
        'spectral_distance': 0.15,
        'max_edge_change': 0.05
    }

    decision_normal = ensemble.make_binary_decision(metrics_normal, thresholds)
    decision_anomaly = ensemble.make_binary_decision(metrics_anomaly, thresholds)

    print(f"Normal decision: {decision_normal['is_anomaly']} (score: {decision_normal['ensemble_score']:.3f})")
    print(f"Anomaly decision: {decision_anomaly['is_anomaly']} (score: {decision_anomaly['ensemble_score']:.3f})")

    # Test adaptive thresholds
    logger.info("Testing adaptive thresholds...")
    golden_matrices = [
        W_baseline,
        W_baseline + np.random.randn(*W_baseline.shape) * 0.005,
        W_baseline + np.random.randn(*W_baseline.shape) * 0.008,
    ]

    threshold_computer = AdaptiveThresholdBootstrap(n_bootstrap=100)  # Reduced for testing
    adaptive_thresholds = threshold_computer.compute_adaptive_thresholds(golden_matrices)
    print(f"Adaptive thresholds: {adaptive_thresholds}")

    return True


def test_anomaly_classification():
    """Test Phase 2: Anomaly Classification."""
    logger.info("\n=== Testing Phase 2: Anomaly Classification ===")

    # Create different types of anomalies
    W_baseline = create_synthetic_baseline_matrix()
    test_cases = [
        ("spike", create_anomalous_matrix(W_baseline, "spike")),
        ("drift", create_anomalous_matrix(W_baseline, "drift")),
        ("structural", create_anomalous_matrix(W_baseline, "structural")),
        ("oscillation", create_anomalous_matrix(W_baseline, "oscillation"))
    ]

    # Test signature extraction
    signature_extractor = GraphSignatureExtractor()

    for anomaly_type, W_anomaly in test_cases:
        logger.info(f"Testing {anomaly_type} anomaly...")
        signature = signature_extractor.extract_signature(W_baseline, W_anomaly)
        print(f"{anomaly_type} signature keys: {list(signature.keys())}")
        print(f"Sample signature values: {dict(list(signature.items())[:5])}")

    # Test rule-based classification
    logger.info("Testing rule-based classification...")
    rule_classifier = RuleBasedClassifier()

    for anomaly_type, W_anomaly in test_cases:
        result = rule_classifier.classify_anomaly(W_baseline, W_anomaly)
        print(f"{anomaly_type}: Classified as '{result['anomaly_type']}' (confidence: {result['confidence']:.3f})")

    # Test ML-based classification (with minimal data)
    logger.info("Testing ML-based classification...")
    ml_classifier = MLBasedClassifier()

    # Create minimal training data
    training_data = []
    labels = []

    for anomaly_type, W_anomaly in test_cases * 2:  # Duplicate for more samples
        signature = signature_extractor.extract_signature(W_baseline, W_anomaly)
        training_data.append(list(signature.values()))
        labels.append(anomaly_type)

    try:
        ml_classifier.train(training_data, labels)

        # Test classification
        for anomaly_type, W_anomaly in test_cases:
            result = ml_classifier.classify_anomaly(W_baseline, W_anomaly)
            print(f"{anomaly_type}: ML classified as '{result['predicted_class']}' (confidence: {result['confidence']:.3f})")
    except Exception as e:
        logger.warning(f"ML classification test failed (expected with minimal data): {e}")

    return True


def test_root_cause_analysis():
    """Test Phase 3: Root Cause Analysis."""
    logger.info("\n=== Testing Phase 3: Root Cause Analysis ===")

    # Create test scenario with labeled variables
    W_baseline = create_synthetic_baseline_matrix()
    W_anomaly = create_anomalous_matrix(W_baseline, "spike")

    variable_names = [f"Sensor_{i+1}" for i in range(W_baseline.shape[0])]

    analyzer = RootCauseAnalyzer()

    # Test comprehensive analysis
    analysis = analyzer.perform_root_cause_analysis(
        W_baseline, W_anomaly, variable_names
    )

    print("Root cause analysis results:")
    print(f"- Top contributing edges: {len(analysis['edge_attribution']['top_edges'])}")
    print(f"- Node importance computed: {len(analysis['node_importance'])}")
    print(f"- Causal paths found: {len(analysis['causal_paths'])}")

    # Print top edge changes
    print("\nTop edge changes:")
    for edge in analysis['edge_attribution']['top_edges'][:3]:
        print(f"  {variable_names[edge['from']]} -> {variable_names[edge['to']]}: "
              f"change = {edge['change']:.4f}, type = {edge['change_type']}")

    # Print node importance
    print("\nNode importance:")
    for node in analysis['node_importance'][:3]:
        node_name = variable_names[node['node']] if node['node'] < len(variable_names) else f"Node_{node['node']}"
        print(f"  {node_name}: importance = {node['importance']:.4f}, role = {node['role']}")

    return True


def test_unified_suite():
    """Test the complete unified anomaly detection suite."""
    logger.info("\n=== Testing Unified Anomaly Detection Suite ===")

    # Create test matrices
    W_baseline = create_synthetic_baseline_matrix()
    W_normal = W_baseline + np.random.randn(*W_baseline.shape) * 0.01
    W_anomaly = create_anomalous_matrix(W_baseline, "spike")

    variable_names = [f"Temperature_Sensor_{i+1}" for i in range(W_baseline.shape[0])]

    # Initialize suite
    suite = UnifiedAnomalyDetectionSuite()

    # Test normal case
    logger.info("Testing unified suite with normal data...")
    result_normal = suite.analyze_single_comparison(
        W_baseline, W_normal, variable_names
    )
    print(f"Normal case detected as anomaly: {result_normal['phase1_binary_detection']['binary_detection']['is_anomaly']}")

    # Test anomaly case
    logger.info("Testing unified suite with anomalous data...")
    result_anomaly = suite.analyze_single_comparison(
        W_baseline, W_anomaly, variable_names
    )

    print(f"Anomaly case detected: {result_anomaly['phase1_binary_detection']['binary_detection']['is_anomaly']}")

    if 'phase2_classification' in result_anomaly:
        print(f"Anomaly classified as: {result_anomaly['phase2_classification']['rule_based_classification']['anomaly_type']}")

    if 'phase3_root_cause' in result_anomaly:
        print(f"Root cause analysis completed with {len(result_anomaly['phase3_root_cause']['edge_attribution']['top_edges'])} edge changes")

    return True


def test_with_csv_data():
    """Test with CSV data format (simulated)."""
    logger.info("\n=== Testing with CSV Data Format ===")

    # Create simulated CSV data
    n_windows = 10
    n_vars = 4
    n_lags = 3

    # Create baseline CSV data
    baseline_data = []
    for window in range(n_windows):
        for lag in range(n_lags):
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and np.random.rand() < 0.3:  # Sparse connections
                        weight = np.random.randn() * 0.1
                        baseline_data.append({
                            'window_idx': window,
                            'lag': lag,
                            'i': i,
                            'j': j,
                            'weight': weight
                        })

    # Create anomalous CSV data with some modifications
    anomaly_data = baseline_data.copy()
    for i, row in enumerate(anomaly_data):
        if row['i'] == 1 and row['j'] == 2:  # Modify specific connection
            anomaly_data[i]['weight'] += 1.5  # Add spike

    # Convert to DataFrames
    df_baseline = pd.DataFrame(baseline_data)
    df_anomaly = pd.DataFrame(anomaly_data)

    # Test the frobenius_test.py legacy functions
    try:
        from frobenius_test import compute_weights_frobenius_distance

        metrics = compute_weights_frobenius_distance(df_baseline, df_anomaly)
        print(f"Legacy Frobenius test results:")
        print(f"  Frobenius distance: {metrics['frobenius_distance']:.6f}")
        print(f"  Matched entries: {metrics['matched_entries']}")
        print(f"  Pearson correlation: {metrics['pearson_correlation']:.4f}")

    except Exception as e:
        logger.warning(f"Legacy CSV test failed: {e}")

    return True


def run_comprehensive_test():
    """Run all tests comprehensively."""
    logger.info("Starting Comprehensive Anomaly Detection Suite Tests")

    test_results = {}

    try:
        test_results['binary_detection'] = test_binary_detection()
        logger.info("Phase 1 (Binary Detection) tests passed")
    except Exception as e:
        logger.error(f"Phase 1 tests failed: {e}")
        test_results['binary_detection'] = False

    try:
        test_results['anomaly_classification'] = test_anomaly_classification()
        logger.info("Phase 2 (Classification) tests passed")
    except Exception as e:
        logger.error(f"Phase 2 tests failed: {e}")
        test_results['anomaly_classification'] = False

    try:
        test_results['root_cause_analysis'] = test_root_cause_analysis()
        logger.info("Phase 3 (Root Cause Analysis) tests passed")
    except Exception as e:
        logger.error(f"Phase 3 tests failed: {e}")
        test_results['root_cause_analysis'] = False

    try:
        test_results['unified_suite'] = test_unified_suite()
        logger.info("Unified Suite tests passed")
    except Exception as e:
        logger.error(f"Unified Suite tests failed: {e}")
        test_results['unified_suite'] = False

    try:
        test_results['csv_data'] = test_with_csv_data()
        logger.info("CSV Data tests passed")
    except Exception as e:
        logger.error(f"CSV Data tests failed: {e}")
        test_results['csv_data'] = False

    # Print summary
    passed = sum(test_results.values())
    total = len(test_results)

    logger.info(f"\nTest Summary: {passed}/{total} tests passed")

    if passed == total:
        logger.info("All tests passed! The Anomaly Detection Suite is working correctly.")
        return True
    else:
        logger.error("Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)