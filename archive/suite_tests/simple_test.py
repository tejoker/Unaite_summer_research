#!/usr/bin/env python3
"""
simple_test.py - Minimal test to validate core anomaly detection components

This script tests the basic functionality without heavy dependencies.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        import binary_detection_metrics
        print("binary_detection_metrics imported successfully")
    except Exception as e:
        print(f"Failed to import binary_detection_metrics: {e}")
        return False

    try:
        import anomaly_classification
        print("anomaly_classification imported successfully")
    except Exception as e:
        print(f"Failed to import anomaly_classification: {e}")
        return False

    try:
        import root_cause_analysis
        print("root_cause_analysis imported successfully")
    except Exception as e:
        print(f"Failed to import root_cause_analysis: {e}")
        return False

    try:
        import anomaly_detection_suite
        print("anomaly_detection_suite imported successfully")
    except Exception as e:
        print(f"Failed to import anomaly_detection_suite: {e}")
        return False

    try:
        import frobenius_test
        print("frobenius_test imported successfully")
    except Exception as e:
        print(f"Failed to import frobenius_test: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality with minimal dependencies."""
    print("\nTesting basic functionality...")

    try:
        # Test that we can create basic classes
        from binary_detection_metrics import BinaryDetectionMetrics
        metrics_computer = BinaryDetectionMetrics()
        print("BinaryDetectionMetrics instantiated")

        from anomaly_classification import GraphSignatureExtractor
        signature_extractor = GraphSignatureExtractor()
        print("GraphSignatureExtractor instantiated")

        from root_cause_analysis import RootCauseAnalyzer
        analyzer = RootCauseAnalyzer()
        print("RootCauseAnalyzer instantiated")

        from anomaly_detection_suite import UnifiedAnomalyDetectionSuite
        suite = UnifiedAnomalyDetectionSuite()
        print("UnifiedAnomalyDetectionSuite instantiated")

        return True

    except Exception as e:
        print(f"Basic functionality test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "binary_detection_metrics.py",
        "anomaly_classification.py",
        "root_cause_analysis.py",
        "anomaly_detection_suite.py",
        "frobenius_test.py"
    ]

    current_dir = Path(__file__).parent
    all_exist = True

    for filename in required_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"{filename} exists")
        else:
            print(f"{filename} missing")
            all_exist = False

    return all_exist

def main():
    """Run all basic tests."""
    print("Starting Simple Anomaly Detection Suite Tests\n")

    tests_passed = 0
    total_tests = 3

    # Test file structure
    if test_file_structure():
        tests_passed += 1

    # Test imports
    if test_imports():
        tests_passed += 1

    # Test basic functionality
    if test_basic_functionality():
        tests_passed += 1

    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("All basic tests passed! The anomaly detection suite structure is correct.")
        print("\nNext steps:")
        print("1. Install required dependencies: pip install numpy pandas scipy scikit-learn")
        print("2. Run the full test suite: python3 test_suite.py")
        return True
    else:
        print("Some basic tests failed. Please check the file structure and imports.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)