#!/bin/bash
# Run all three hypothesis tests

echo "=========================================="
echo "Testing Proposed Methods for Gradual Anomaly Detection"
echo "=========================================="
echo ""
echo "These tests evaluate two proposed methods:"
echo "  Method 1: Change point detection on differenced data"
echo "  Method 2: Double differencing to create spikes"
echo ""
echo "Testing on your actual synthetic anomalies (trend_change, drift)"
echo "=========================================="
echo ""

# Check if ruptures is installed
if ! python3 -c "import ruptures" 2>/dev/null; then
    echo "Installing required dependency: ruptures"
    pip install ruptures
    echo ""
fi

# Test 1: Method 1 - Change point detection
echo "=========================================="
echo "TEST 1: Change Point Detection (Method 1)"
echo "=========================================="
python3 test_method1_changepoint.py
echo ""
echo "Press Enter to continue to Test 2..."
read

# Test 2: Method 2 - Double differencing
echo "=========================================="
echo "TEST 2: Double Differencing (Method 2)"
echo "=========================================="
python3 test_method2_double_diff.py
echo ""
echo "Press Enter to continue to Test 3..."
read

# Test 3: Stronger anomalies
echo "=========================================="
echo "TEST 3: Testing with STRONGER Anomalies"
echo "=========================================="
python3 test_stronger_anomaly.py
echo ""

echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "Results saved:"
echo "  - method1_changepoint_results.png"
echo "  - method2_double_diff_results.png"
echo "  - strong_anomaly_test_results.png"
echo ""
echo "These tests will show whether the proposed methods:"
echo "  1. Work on your actual synthetic anomalies (likely: NO)"
echo "  2. Would work if anomalies were stronger (likely: YES)"
echo "  3. Confirm the root cause: signal too weak after differencing"
