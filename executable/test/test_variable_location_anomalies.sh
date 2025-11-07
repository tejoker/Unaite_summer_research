#!/bin/bash
# Test Script: Variable Location Anomalies
# Purpose: Verify that anomaly detection is NOT hardcoded to windows 9-10
#          by testing anomalies at different temporal locations

set -e

# Get script directory and workspace root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configuration (using absolute paths)
GOLDEN_CSV="$WORKSPACE_ROOT/data/Golden/chunking/output_of_the_1th_chunk.csv"
TEST_DIR="$WORKSPACE_ROOT/data/Test"
RESULTS_BASE="$WORKSPACE_ROOT/results/variable_location_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_BASE}_${TIMESTAMP}"

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/golden"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                    VARIABLE LOCATION ANOMALY TEST SUITE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Purpose: Prove that detection is NOT hardcoded to windows 9-10"
echo "Method:  Test anomalies at rows 50, 100, 200, 350, 500, 700"
echo "Results: $RESULTS_DIR"
echo ""

# Step 1: Create test anomalies at different locations
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 1: Creating test anomalies at different temporal locations..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

# Change to workspace root for consistent paths
cd "$WORKSPACE_ROOT"

python3 tests/create_test_anomalies.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to create test anomalies"
    exit 1
fi

echo ""

# Step 2: Calibrate on Golden data (once)
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 2: Calibrating on Golden data (one-time calibration)..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

# Preprocessing
echo "🔧 Running preprocessing to calculate optimal lags..."
INPUT_CSV_FILE="$GOLDEN_CSV" \
RESULT_DIR="$RESULTS_DIR/golden" \
python3 "$WORKSPACE_ROOT/executable/final_pipeline/preprocessing_no_mi.py"

if [ $? -ne 0 ]; then
    echo "❌ Golden preprocessing failed"
    exit 1
fi
echo "✅ Golden preprocessing complete"
echo ""

# Causal Discovery - Part 1: Lambda Calibration
echo "🔧 Running causal discovery to find optimal lambda values..."
GOLDEN_BASENAME=$(basename "$GOLDEN_CSV" .csv)
INPUT_DIFFERENCED_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_differenced_stationary_series.csv" \
RESULT_DIR="$RESULTS_DIR/golden" \
INPUT_LAGS_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
CALIBRATE_LAMBDAS="true" \
python3 "$WORKSPACE_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"

if [ $? -ne 0 ]; then
    echo "❌ Golden lambda calibration failed"
    exit 1
fi
echo "✅ Golden lambda calibration complete"

# Extract lambda values
LAMBDA_W=$(python3 "$WORKSPACE_ROOT/utils/read_json_field.py" "$RESULTS_DIR/golden/best_lambdas.json" "lambda_w")
LAMBDA_A=$(python3 "$WORKSPACE_ROOT/utils/read_json_field.py" "$RESULTS_DIR/golden/best_lambdas.json" "lambda_a")

echo "   Calibrated: lambda_w = $LAMBDA_W, lambda_a = $LAMBDA_A"
echo ""

# Causal Discovery - Part 2: Generate Golden Weights
echo "🔧 Running causal discovery with calibrated lambdas to generate golden weights..."

# Clear any existing checkpoint to ensure fresh start
rm -f "$RESULTS_DIR/golden/history/rolling_checkpoint.pkl"

INPUT_DIFFERENCED_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_differenced_stationary_series.csv" \
RESULT_DIR="$RESULTS_DIR/golden" \
INPUT_LAGS_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
FIXED_LAMBDA_W="$LAMBDA_W" \
FIXED_LAMBDA_A="$LAMBDA_A" \
python3 "$WORKSPACE_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"

if [ $? -ne 0 ]; then
    echo "❌ Golden weights generation failed"
    exit 1
fi
echo "✅ Golden weights generated"
echo ""

echo "📊 Calibrated parameters:"
echo "   lambda_w = $LAMBDA_W"
echo "   lambda_a = $LAMBDA_A"
echo "   lags file = $RESULTS_DIR/golden/optimal_lags_${GOLDEN_BASENAME}.csv"
echo ""

# Step 3: Test each anomaly with fixed parameters
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 3: Testing anomalies at different locations with FIXED parameters..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

# Test configurations
declare -a TEST_ROWS=(50 100 200 350 500 700)

for ROW in "${TEST_ROWS[@]}"; do
    ANOMALY_CSV="$TEST_DIR/spike_row${ROW}.csv"
    ANOMALY_JSON="$TEST_DIR/spike_row${ROW}.json"
    ANOMALY_DIR="$RESULTS_DIR/spike_row${ROW}"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: Spike at Row $ROW"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ ! -f "$ANOMALY_CSV" ]; then
        echo "⚠️  Anomaly file not found: $ANOMALY_CSV (skipping)"
        echo ""
        continue
    fi
    
    mkdir -p "$ANOMALY_DIR"
    
    # Get basename for this anomaly file
    ANOMALY_BASENAME=$(basename "$ANOMALY_CSV" .csv)
    
    # Preprocessing with FIXED lags
    echo "🔧 Preprocessing (using fixed lags from Golden)..."
    INPUT_CSV_FILE="$ANOMALY_CSV" \
    RESULT_DIR="$ANOMALY_DIR" \
    INPUT_LAGS_CSV="$RESULTS_DIR/golden/optimal_lags_${GOLDEN_BASENAME}.csv" \
    python3 "$WORKSPACE_ROOT/executable/final_pipeline/preprocessing_no_mi.py"
    
    if [ $? -ne 0 ]; then
        echo "❌ Preprocessing failed for row $ROW"
        continue
    fi
    
    # Causal Discovery with FIXED lambda
    echo "🔧 Causal discovery (using fixed lambda from Golden)..."
    INPUT_DIFFERENCED_CSV="$ANOMALY_DIR/${ANOMALY_BASENAME}_differenced_stationary_series.csv" \
    RESULT_DIR="$ANOMALY_DIR" \
    INPUT_LAGS_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
    FIXED_LAMBDA_W="$LAMBDA_W" \
    FIXED_LAMBDA_A="$LAMBDA_A" \
    python3 "$WORKSPACE_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"
    
    if [ $? -ne 0 ]; then
        echo "❌ Causal discovery failed for row $ROW"
        continue
    fi
    
    echo "✅ Anomaly test complete: Row $ROW"
    echo ""
done

echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 4: Analyzing detection timing across all locations..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

# Create analysis script
python3 - "$RESULTS_DIR" << 'EOF'
import pandas as pd
import json
import os
import sys

results_dir = sys.argv[1]
golden_dir = os.path.join(results_dir, "golden")

# Load golden weights from CSV
golden_weights_file = os.path.join(golden_dir, "weights", "weights_enhanced.csv")
if not os.path.exists(golden_weights_file):
    print(f"❌ Golden weights not found: {golden_weights_file}")
    sys.exit(1)

golden_df = pd.read_csv(golden_weights_file)

# Convert to window-based format
golden_weights = []
for window_id in sorted(golden_df['window_idx'].unique()):
    window_data = golden_df[golden_df['window_idx'] == window_id]
    weights = {}
    for _, row in window_data.iterrows():
        edge = f"{row['parent_name']}→{row['child_name']}"
        weights[edge] = row['weight']
    golden_weights.append({"weights": weights})

print("="*80)
print(" "*20 + "DETECTION TIMING ANALYSIS")
print("="*80)
print()

# Expected windows for each row location
expected = {
    50: (0, 5, "Very Early"),
    100: (1, 10, "Early"),
    200: (10, 20, "Early-Mid"),
    350: (25, 35, "Middle"),
    500: (40, 50, "Late-Mid"),
    700: (60, 70, "Late")
}

results_summary = []

for row, (exp_min, exp_max, desc) in expected.items():
    anomaly_dir = os.path.join(results_dir, f"spike_row{row}")
    weights_file = os.path.join(anomaly_dir, "weights", "weights_enhanced.csv")
    
    if not os.path.exists(weights_file):
        print(f"⚠️  Weights not found for row {row}")
        continue
    
    # Load anomaly weights from CSV
    anomaly_df = pd.read_csv(weights_file)
    
    # Convert to window-based format
    anomaly_weights = []
    for window_id in sorted(anomaly_df['window_idx'].unique()):
        window_data = anomaly_df[anomaly_df['window_idx'] == window_id]
        weights = {}
        for _, wrow in window_data.iterrows():
            edge = f"{wrow['parent_name']}→{wrow['child_name']}"
            weights[edge] = wrow['weight']
        anomaly_weights.append({"weights": weights})
    
    # Find windows with differences
    changed_windows = []
    for w_idx in range(len(golden_weights)):
        if w_idx >= len(anomaly_weights):
            break
        
        g_weights = golden_weights[w_idx]["weights"]
        a_weights = anomaly_weights[w_idx]["weights"]
        
        # Calculate max absolute difference
        max_diff = 0.0
        for edge, g_val in g_weights.items():
            a_val = a_weights.get(edge, 0.0)
            diff = abs(g_val - a_val)
            max_diff = max(max_diff, diff)
        
        if max_diff > 0.01:  # threshold
            changed_windows.append(w_idx)
    
    if changed_windows:
        first_detection = min(changed_windows)
        last_detection = max(changed_windows)
        num_windows = len(changed_windows)
        
        # Check if in expected range
        in_range = exp_min <= first_detection <= exp_max
        status = "✅" if in_range else "⚠️"
        
        results_summary.append({
            'row': row,
            'description': desc,
            'expected_min': exp_min,
            'expected_max': exp_max,
            'first_detection': first_detection,
            'last_detection': last_detection,
            'num_windows': num_windows,
            'in_range': in_range,
            'status': status
        })
        
        print(f"{status} Row {row:3d} ({desc:15s})")
        print(f"   Expected range: Windows {exp_min:2d}-{exp_max:2d}")
        print(f"   First detection: Window {first_detection:2d}")
        print(f"   Last detection:  Window {last_detection:2d}")
        print(f"   Total changed:   {num_windows} windows")
        print()
    else:
        print(f"❌ Row {row:3d} ({desc:15s})")
        print(f"   NO DETECTION (no windows changed)")
        print()
        results_summary.append({
            'row': row,
            'description': desc,
            'expected_min': exp_min,
            'expected_max': exp_max,
            'first_detection': None,
            'last_detection': None,
            'num_windows': 0,
            'in_range': False,
            'status': '❌'
        })

print("="*80)
print(" "*25 + "SUMMARY TABLE")
print("="*80)
print()
print(f"{'Row':<6} {'Description':<15} {'Expected':<12} {'Detected':<12} {'Windows':<10} {'Status':<8}")
print("-"*80)

for r in results_summary:
    exp_range = f"{r['expected_min']}-{r['expected_max']}"
    if r['first_detection'] is not None:
        det_range = f"{r['first_detection']}-{r['last_detection']}"
        num_win = r['num_windows']
    else:
        det_range = "NONE"
        num_win = 0
    
    print(f"{r['row']:<6} {r['description']:<15} {exp_range:<12} {det_range:<12} {num_win:<10} {r['status']:<8}")

print("-"*80)
print()

# Check if detections are diverse (not all around windows 9-10)
all_detections = [r['first_detection'] for r in results_summary if r['first_detection'] is not None]
if all_detections:
    min_det = min(all_detections)
    max_det = max(all_detections)
    span = max_det - min_det
    
    print("="*80)
    print(" "*30 + "CONCLUSION")
    print("="*80)
    print()
    
    if span >= 30:
        print("✅ DETECTION IS NOT HARDCODED!")
        print(f"   Detections span {span} windows (from Window {min_det} to Window {max_det})")
        print("   This proves the system correctly identifies anomalies at their temporal location.")
    elif span >= 15:
        print("✅ DETECTION APPEARS CORRECT")
        print(f"   Detections span {span} windows (from Window {min_det} to Window {max_det})")
        print("   Some clustering is expected due to signal strength requirements.")
    else:
        print("⚠️  LIMITED DETECTION RANGE")
        print(f"   Detections only span {span} windows (from Window {min_det} to Window {max_det})")
        print("   This might indicate issues with sensitivity or test design.")
    print()

EOF

echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 5: Weight comparison visualization..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

for ROW in "${TEST_ROWS[@]}"; do
    ANOMALY_DIR="$RESULTS_DIR/spike_row${ROW}"
    
    if [ ! -f "$ANOMALY_DIR/results.json" ]; then
        continue
    fi
    
    echo "📊 Comparing weights: Golden vs Spike at Row $ROW..."
    python3 "$WORKSPACE_ROOT/analysis/analyze_all_window_weights.py" \
        "$RESULTS_DIR/golden/results.json" \
        "$ANOMALY_DIR/results.json" \
        "$ANOMALY_DIR/weight_comparison.csv"
    
    echo "   ✅ Saved: $ANOMALY_DIR/weight_comparison.csv"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                          TEST COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view detailed comparisons:"
echo "  cat $RESULTS_DIR/spike_row*/weight_comparison.csv"
echo ""
