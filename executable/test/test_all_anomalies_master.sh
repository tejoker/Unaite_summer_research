#!/bin/bash

# ==============================================================================
# Master Test Script: Test All Anomaly Types with Fixed Lambda Strategy
# ==============================================================================
# This script runs the complete fixed lambda pipeline on all anomaly types
# (except missing values) and generates comparative analysis.
# ==============================================================================

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

GOLDEN_CSV="$PROJECT_ROOT/data/Golden/chunking/output_of_the_1th_chunk.csv"
PREPROCESSING_SCRIPT="$PROJECT_ROOT/executable/final_pipeline/preprocessing_no_mi.py"
FIXED_LAMBDA_SCRIPT="$PROJECT_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"
DIAGNOSTIC_SCRIPT="$PROJECT_ROOT/diagnose_global_changes.py"
ANALYSIS_SCRIPT="$PROJECT_ROOT/analyze_all_window_weights.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_RESULTS_DIR="$PROJECT_ROOT/results/master_anomaly_test_$TIMESTAMP"

# Array of anomaly types to test
ANOMALY_TYPES=(
    "spike"
    "drift"
    "level_shift"
    "amplitude_change"
    "trend_change"
    "variance_burst"
)

echo "=============================================================================="
echo "MASTER ANOMALY DETECTION TEST - Fixed Lambda Strategy"
echo "=============================================================================="
echo "Testing ${#ANOMALY_TYPES[@]} anomaly types"
echo "Master results directory: $MASTER_RESULTS_DIR"
echo "Start time: $(date)"
echo "=============================================================================="

mkdir -p "$MASTER_RESULTS_DIR"

# Helper function to run pipeline
run_pipeline() {
    local input_csv=$1
    local output_dir=$2
    local is_golden=$3
    local run_type=$4
    local script_to_run=$5
    local lags_csv_path=${6:-""}
    local calibrate_mode=${7:-"false"}

    echo -e "\n--- Running Pipeline for: $run_type ---"
    mkdir -p "$output_dir"

    # Step 1: Preprocessing
    echo "  [1/2] Running preprocessing..."
    export INPUT_CSV_FILE="$input_csv"
    export RESULT_DIR="$output_dir"
    export IS_GOLDEN_DATA="$is_golden"
    if [ -n "$lags_csv_path" ] && [ -f "$lags_csv_path" ]; then
        export INPUT_LAGS_CSV="$lags_csv_path"
        echo "  Using pre-calculated lags from: $INPUT_LAGS_CSV"
    else
        unset INPUT_LAGS_CSV
        echo "  Will calculate new lags from data"
    fi
    python3 "$PREPROCESSING_SCRIPT" > /dev/null 2>&1

    # Step 2: Causal Discovery
    echo "  [2/2] Running causal discovery..."
    local input_basename=$(basename "$input_csv" .csv)
    export INPUT_DIFFERENCED_CSV="$output_dir/${input_basename}_differenced_stationary_series.csv"
    export INPUT_LAGS_CSV="${lags_csv_path:-$output_dir/${input_basename}_optimal_lags.csv}"

    CALIBRATE_LAMBDAS=$calibrate_mode \
    FIXED_LAMBDA_W=${FIXED_LAMBDA_W:-0.1} \
    FIXED_LAMBDA_A=${FIXED_LAMBDA_A:-0.1} \
    python3 "$script_to_run" > /dev/null 2>&1

    export CALIBRATE_LAMBDAS="false"
    echo "  ✅ Completed: $run_type"
}

# ==============================================================================
# PHASE 1: Calibrate on Golden Data (only once)
# ==============================================================================
echo -e "\n=============================================================================="
echo "PHASE 1: Calibrating Parameters on Golden Data"
echo "=============================================================================="

GOLDEN_CALIBRATION_DIR="$MASTER_RESULTS_DIR/Golden_Calibration"
run_pipeline "$GOLDEN_CSV" "$GOLDEN_CALIBRATION_DIR" "true" "Golden (Calibration)" "$FIXED_LAMBDA_SCRIPT" "" "true"

# Get calibrated parameters
LAMBDA_FILE="$GOLDEN_CALIBRATION_DIR/best_lambdas.json"
GOLDEN_LAGS_FILE="$GOLDEN_CALIBRATION_DIR/output_of_the_1th_chunk_optimal_lags.csv"

if [ ! -f "$LAMBDA_FILE" ] || [ ! -f "$GOLDEN_LAGS_FILE" ]; then
    echo "❌ Error: Calibration failed"
    exit 1
fi

export FIXED_LAMBDA_W=$(jq -r '.lambda_w' "$LAMBDA_FILE")
export FIXED_LAMBDA_A=$(jq -r '.lambda_a' "$LAMBDA_FILE")

echo "✅ Calibration complete!"
echo "   lambda_w = $FIXED_LAMBDA_W"
echo "   lambda_a = $FIXED_LAMBDA_A"
echo "   Lags file = $GOLDEN_LAGS_FILE"

# Run Golden test once (to compare against all anomalies)
GOLDEN_TEST_DIR="$MASTER_RESULTS_DIR/Golden_Test_Run"
run_pipeline "$GOLDEN_CSV" "$GOLDEN_TEST_DIR" "true" "Golden (Test)" "$FIXED_LAMBDA_SCRIPT" "$GOLDEN_LAGS_FILE" "false"

GOLDEN_WEIGHTS="$GOLDEN_TEST_DIR/weights/weights_enhanced.csv"

# ==============================================================================
# PHASE 2: Test Each Anomaly Type
# ==============================================================================
echo -e "\n=============================================================================="
echo "PHASE 2: Testing All Anomaly Types"
echo "=============================================================================="

for anomaly_type in "${ANOMALY_TYPES[@]}"; do
    echo -e "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $anomaly_type"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Find the anomaly files
    ANOMALY_CSV="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__${anomaly_type}.csv"
    ANOMALY_JSON="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__${anomaly_type}.json"
    
    if [ ! -f "$ANOMALY_CSV" ]; then
        echo "⚠️  Skipping $anomaly_type: CSV file not found"
        continue
    fi
    
    if [ ! -f "$ANOMALY_JSON" ]; then
        echo "⚠️  Warning: Ground truth JSON not found for $anomaly_type"
    fi
    
    # Run anomaly pipeline
    ANOMALY_RUN_DIR="$MASTER_RESULTS_DIR/${anomaly_type}_run"
    run_pipeline "$ANOMALY_CSV" "$ANOMALY_RUN_DIR" "false" "$anomaly_type" "$FIXED_LAMBDA_SCRIPT" "$GOLDEN_LAGS_FILE" "false"
    
    # Run diagnostics
    DIAGNOSTICS_DIR="$ANOMALY_RUN_DIR/diagnostics"
    mkdir -p "$DIAGNOSTICS_DIR"
    
    ANOMALY_WEIGHTS="$ANOMALY_RUN_DIR/weights/weights_enhanced.csv"
    
    echo "  Running diagnostics..."
    if [ -f "$ANOMALY_JSON" ]; then
        python3 "$DIAGNOSTIC_SCRIPT" \
          --golden-csv "$GOLDEN_CSV" \
          --anomaly-csv "$ANOMALY_CSV" \
          --golden-weights "$GOLDEN_WEIGHTS" \
          --anomaly-weights "$ANOMALY_WEIGHTS" \
          --ground-truth "$ANOMALY_JSON" \
          --anomaly-type "${anomaly_type}_fixed_lambda" \
          --output-dir "$DIAGNOSTICS_DIR" > /dev/null 2>&1
    fi
    
    echo "  Running detailed weight analysis..."
    python3 "$ANALYSIS_SCRIPT" "$GOLDEN_WEIGHTS" "$ANOMALY_WEIGHTS" > "$DIAGNOSTICS_DIR/window_analysis.txt" 2>&1
    
    echo "✅ Completed: $anomaly_type"
done

# ==============================================================================
# PHASE 3: Generate Comparative Summary
# ==============================================================================
echo -e "\n=============================================================================="
echo "PHASE 3: Generating Comparative Summary"
echo "=============================================================================="

SUMMARY_FILE="$MASTER_RESULTS_DIR/COMPARATIVE_SUMMARY.txt"

cat > "$SUMMARY_FILE" << 'EOF'
================================================================================
COMPARATIVE ANOMALY DETECTION RESULTS
Fixed Lambda + Fixed Lags Strategy
================================================================================

EOF

echo "Test Date: $(date)" >> "$SUMMARY_FILE"
echo "Total Anomaly Types Tested: ${#ANOMALY_TYPES[@]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Parameters Used:" >> "$SUMMARY_FILE"
echo "  lambda_w = $FIXED_LAMBDA_W" >> "$SUMMARY_FILE"
echo "  lambda_a = $FIXED_LAMBDA_A" >> "$SUMMARY_FILE"
echo "  max_lag (p) = 10" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "================================================================================  " >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for anomaly_type in "${ANOMALY_TYPES[@]}"; do
    DIAGNOSTICS_DIR="$MASTER_RESULTS_DIR/${anomaly_type}_run/diagnostics"
    ANALYSIS_FILE="$DIAGNOSTICS_DIR/window_analysis.txt"
    
    if [ ! -f "$ANALYSIS_FILE" ]; then
        continue
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$SUMMARY_FILE"
    echo "ANOMALY TYPE: ${anomaly_type^^}" >> "$SUMMARY_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Extract ground truth
    ANOMALY_JSON="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__${anomaly_type}.json"
    if [ -f "$ANOMALY_JSON" ]; then
        echo "Ground Truth:" >> "$SUMMARY_FILE"
        echo "  Start Row: $(jq -r '.start' "$ANOMALY_JSON")" >> "$SUMMARY_FILE"
        echo "  Length: $(jq -r '.length' "$ANOMALY_JSON")" >> "$SUMMARY_FILE"
        echo "  Magnitude: $(jq -r '.magnitude // .factor // "N/A"' "$ANOMALY_JSON")" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
    fi
    
    # Extract key statistics from analysis
    echo "Detection Results:" >> "$SUMMARY_FILE"
    grep "Windows with ANY weight changes:" "$ANALYSIS_FILE" >> "$SUMMARY_FILE" 2>/dev/null || echo "  Unable to extract statistics" >> "$SUMMARY_FILE"
    grep "Windows with SIGNIFICANT changes" "$ANALYSIS_FILE" >> "$SUMMARY_FILE" 2>/dev/null || true
    echo "" >> "$SUMMARY_FILE"
    
    # Extract window statistics
    echo "Summary Statistics:" >> "$SUMMARY_FILE"
    sed -n '/SUMMARY STATISTICS/,/^$/p' "$ANALYSIS_FILE" | head -20 >> "$SUMMARY_FILE" 2>/dev/null || true
    echo "" >> "$SUMMARY_FILE"
done

echo "================================================================================  " >> "$SUMMARY_FILE"
echo "END OF COMPARATIVE SUMMARY" >> "$SUMMARY_FILE"
echo "================================================================================  " >> "$SUMMARY_FILE"

# Display summary
cat "$SUMMARY_FILE"

# ==============================================================================
# PHASE 4: Create Quick Reference Table
# ==============================================================================
echo -e "\n=============================================================================="
echo "PHASE 4: Creating Quick Reference Table"
echo "=============================================================================="

QUICKREF_FILE="$MASTER_RESULTS_DIR/QUICK_REFERENCE.txt"

cat > "$QUICKREF_FILE" << EOF
================================================================================
QUICK REFERENCE - Anomaly Detection Results
================================================================================

$(printf "%-20s %-15s %-20s %-15s\n" "Anomaly Type" "Start Row" "Windows Changed" "Max L2 Norm")
$(printf "%-20s %-15s %-20s %-15s\n" "----------------" "-----------" "----------------" "-------------")
EOF

for anomaly_type in "${ANOMALY_TYPES[@]}"; do
    ANOMALY_JSON="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__${anomaly_type}.json"
    ANALYSIS_FILE="$MASTER_RESULTS_DIR/${anomaly_type}_run/diagnostics/window_analysis.txt"
    
    if [ ! -f "$ANALYSIS_FILE" ]; then
        continue
    fi
    
    START_ROW=$(jq -r '.start // "N/A"' "$ANOMALY_JSON" 2>/dev/null || echo "N/A")
    CHANGED_WINDOWS=$(grep "Windows with ANY weight changes:" "$ANALYSIS_FILE" 2>/dev/null | grep -oP '\d+' | head -1 || echo "?")
    MAX_L2=$(grep "Max:" "$ANALYSIS_FILE" 2>/dev/null | head -1 | awk '{print $2}' || echo "?")
    
    printf "%-20s %-15s %-20s %-15s\n" "$anomaly_type" "$START_ROW" "$CHANGED_WINDOWS" "$MAX_L2" >> "$QUICKREF_FILE"
done

echo "" >> "$QUICKREF_FILE"
echo "Results directory: $MASTER_RESULTS_DIR" >> "$QUICKREF_FILE"
echo "Generated: $(date)" >> "$QUICKREF_FILE"

cat "$QUICKREF_FILE"

# ==============================================================================
# Completion
# ==============================================================================
echo -e "\n=============================================================================="
echo "MASTER TEST COMPLETE!"
echo "=============================================================================="
echo "Results saved to: $MASTER_RESULTS_DIR"
echo ""
echo "📁 Key Files:"
echo "   📊 $SUMMARY_FILE"
echo "   📋 $QUICKREF_FILE"
echo ""
echo "📂 Individual Results:"
for anomaly_type in "${ANOMALY_TYPES[@]}"; do
    echo "   - $MASTER_RESULTS_DIR/${anomaly_type}_run/diagnostics/"
done
echo ""
echo "End time: $(date)"
echo "=============================================================================="
