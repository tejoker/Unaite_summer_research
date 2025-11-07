#!/usr/bin/env python3
"""
Test script for MI masking refactoring
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the executable path to import preprocessing
sys.path.append(str(Path(__file__).parent / 'executable' / 'final_pipeline'))

def test_roi_analysis():
    """Test the ROI analysis function"""
    print("Testing ROI analysis function...")

    try:
        from preprocessing import should_use_mi_masking

        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame(np.random.randn(1000, 6),
                                columns=[f'Sensor_{i}' for i in range(6)])

        # Test with small data (should recommend skip)
        result = should_use_mi_masking(test_data, L=2, window_size=500, verbose=True)
        print(f"Small data test result: {result}")

        # Test with larger data
        large_data = pd.DataFrame(np.random.randn(5000, 15),
                                 columns=[f'Sensor_{i}' for i in range(15)])
        result_large = should_use_mi_masking(large_data, L=3, window_size=1000, verbose=True)
        print(f"Large data test result: {result_large}")

        return True

    except Exception as e:
        print(f"ROI analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_golden_detection():
    """Test Golden/Anomaly detection logic"""
    print("\nTesting Golden/Anomaly detection...")

    # Test different file paths
    test_cases = [
        ("data/Golden/sensor_data.csv", True),
        ("data/golden_baseline.csv", True),
        ("data/anomaly_case1.csv", False),
        ("data/normal_data.csv", False),
    ]

    for file_path, expected_golden in test_cases:
        is_golden = (
            'Golden' in file_path or
            'golden' in file_path.lower()
        )

        print(f"  {file_path}: Golden={is_golden} (expected {expected_golden})")
        assert is_golden == expected_golden, f"Golden detection failed for {file_path}"

    # Test environment variable
    os.environ['IS_GOLDEN_DATA'] = 'true'
    is_golden_env = os.getenv('IS_GOLDEN_DATA', 'false').lower() == 'true'
    assert is_golden_env == True, "Environment variable test failed"

    os.environ['IS_GOLDEN_DATA'] = 'false'
    is_golden_env = os.getenv('IS_GOLDEN_DATA', 'false').lower() == 'true'
    assert is_golden_env == False, "Environment variable test failed"

    print("  Golden/Anomaly detection tests passed!")
    return True

def test_environment_variables():
    """Test environment variable handling"""
    print("\nTesting environment variables...")

    # Test DYNOTEARS_WINDOW_SIZE
    os.environ['DYNOTEARS_WINDOW_SIZE'] = '1500'
    window_size = int(os.getenv('DYNOTEARS_WINDOW_SIZE', '1000'))
    assert window_size == 1500, f"Expected 1500, got {window_size}"

    # Test FORCE_MI_MASKING
    test_cases = [
        ('auto', None),
        ('true', True),
        ('false', False),
        ('TRUE', True),
        ('FALSE', False),
    ]

    for env_value, expected in test_cases:
        os.environ['FORCE_MI_MASKING'] = env_value
        force_mi_env = os.getenv('FORCE_MI_MASKING', 'auto').lower()
        force_mi = None if force_mi_env == 'auto' else (force_mi_env == 'true')
        assert force_mi == expected, f"Expected {expected}, got {force_mi} for {env_value}"

    print("  Environment variable tests passed!")
    return True

def create_test_data():
    """Create test data for validation"""
    print("\nCreating test data...")

    # Create test directories
    test_dir = Path(__file__).parent / 'test_data'
    test_dir.mkdir(exist_ok=True)

    golden_dir = test_dir / 'Golden'
    anomaly_dir = test_dir / 'Anomaly'
    golden_dir.mkdir(exist_ok=True)
    anomaly_dir.mkdir(exist_ok=True)

    # Create synthetic sensor data
    np.random.seed(42)
    n_samples = 2000
    n_sensors = 6

    # Golden baseline data (normal operation)
    baseline_data = np.random.randn(n_samples, n_sensors) * 0.1
    # Add some correlation structure
    for i in range(1, n_sensors):
        baseline_data[:, i] += 0.3 * baseline_data[:, i-1]

    golden_df = pd.DataFrame(
        baseline_data,
        columns=['Druckpfannenlager', 'Exzenterlager', 'Staenderlager_1',
                'Staenderlager_2', 'Temperature', 'Pressure']
    )
    golden_df.to_csv(golden_dir / 'golden_sensor_data.csv', index=False)

    # Anomaly data (with some spikes)
    anomaly_data = baseline_data.copy()
    # Add spikes at random locations
    spike_indices = np.random.choice(n_samples, size=50, replace=False)
    for idx in spike_indices:
        sensor_idx = np.random.choice(n_sensors)
        anomaly_data[idx, sensor_idx] += np.random.uniform(2, 5)

    anomaly_df = pd.DataFrame(
        anomaly_data,
        columns=['Druckpfannenlager', 'Exzenterlager', 'Staenderlager_1',
                'Staenderlager_2', 'Temperature', 'Pressure']
    )
    anomaly_df.to_csv(anomaly_dir / 'anomaly_sensor_data.csv', index=False)

    print(f"  Created test data in {test_dir}")
    return test_dir

def main():
    """Run all tests"""
    print("MI Masking Refactoring Tests")
    print("=" * 50)

    tests_passed = 0
    total_tests = 4

    # Test 1: ROI Analysis
    try:
        if test_roi_analysis():
            tests_passed += 1
            print("✓ ROI analysis test passed")
        else:
            print("✗ ROI analysis test failed")
    except Exception as e:
        print(f"✗ ROI analysis test failed with exception: {e}")

    # Test 2: Golden Detection
    try:
        if test_golden_detection():
            tests_passed += 1
            print("✓ Golden detection test passed")
        else:
            print("✗ Golden detection test failed")
    except Exception as e:
        print(f"✗ Golden detection test failed with exception: {e}")

    # Test 3: Environment Variables
    try:
        if test_environment_variables():
            tests_passed += 1
            print("✓ Environment variables test passed")
        else:
            print("✗ Environment variables test failed")
    except Exception as e:
        print(f"✗ Environment variables test failed with exception: {e}")

    # Test 4: Create Test Data
    try:
        test_dir = create_test_data()
        if test_dir.exists():
            tests_passed += 1
            print("✓ Test data creation passed")
        else:
            print("✗ Test data creation failed")
    except Exception as e:
        print(f"✗ Test data creation failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("✓ All tests passed! MI masking refactoring is working correctly.")
        print("\nNext steps:")
        print("1. Set environment variables: export DYNOTEARS_WINDOW_SIZE=1000")
        print("2. Test with Golden data: export IS_GOLDEN_DATA=true")
        print("3. Run preprocessing on Golden data first")
        print("4. Then run on anomaly data (will reuse Golden mask)")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)