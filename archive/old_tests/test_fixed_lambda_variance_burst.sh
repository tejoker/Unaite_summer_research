#!/bin/bash

# ==============================================================================
# Test: Variance Burst Anomaly Detection with Fixed Lambda
# ==============================================================================

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

GOLDEN_CSV="$PROJECT_ROOT/data/Golden/chunking/output_of_the_1th_chunk.csv"
ANOMALY_CSV="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__variance_burst.csv"
GROUND_TRUTH_JSON="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__variance_burst.json"

PREPROCESSING_SCRIPT="$PROJECT_ROOT/executable/final_pipeline/preprocessing_no_mi.py"
FIXED_LAMBDA_SCRIPT="$PROJECT_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"
DIAGNOSTIC_SCRIPT="$PROJECT_ROOT/diagnose_global_changes.py"

RESULTS_BASE_DIR="$PROJECT_ROOT/results/fixed_lambda_variance_burst_test_$(date +%Y%m%d_%H%M%S)"

echo "=============================================================================="
echo "Fixed Lambda Test: VARIANCE BURST Anomaly"
echo "Results: $RESULTS_BASE_DIR"
echo "=============================================================================="

# Helper function
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
    python3 "$PREPROCESSING_SCRIPT"

    # Step 2: Causal Discovery
    echo "  [2/2] Running causal discovery..."
    local input_basename=$(basename "$input_csv" .csv)
    export INPUT_DIFFERENCED_CSV="$output_dir/${input_basename}_differenced_stationary_series.csv"
    export INPUT_LAGS_CSV="${lags_csv_path:-$output_dir/${input_basename}_optimal_lags.csv}"
    echo "  Using lags file: $INPUT_LAGS_CSV"

    CALIBRATE_LAMBDAS=$calibrate_mode \
    FIXED_LAMBDA_W=${FIXED_LAMBDA_W:-0.1} \
    FIXED_LAMBDA_A=${FIXED_LAMBDA_A:-0.1} \
    python3 "$script_to_run"

    export CALIBRATE_LAMBDAS="false"
    echo "--- Finished Pipeline for: $run_type ---"
}

# 1. Calibrate on Golden
GOLDEN_CALIBRATION_DIR="$RESULTS_BASE_DIR/Golden_Calibration"
echo -e "\n--- Step 1: Calibrating on Golden data ---"
run_pipeline "$GOLDEN_CSV" "$GOLDEN_CALIBRATION_DIR" "true" "Golden (Calibration)" "$FIXED_LAMBDA_SCRIPT" "" "true"

# 2. Get calibrated parameters
LAMBDA_FILE="$GOLDEN_CALIBRATION_DIR/best_lambdas.json"
GOLDEN_LAGS_FILE="$GOLDEN_CALIBRATION_DIR/output_of_the_1th_chunk_optimal_lags.csv"

if [ ! -f "$LAMBDA_FILE" ] || [ ! -f "$GOLDEN_LAGS_FILE" ]; then
    echo "Error: Calibration failed"
    exit 1
fi

export FIXED_LAMBDA_W=$(jq -r '.lambda_w' "$LAMBDA_FILE")
export FIXED_LAMBDA_A=$(jq -r '.lambda_a' "$LAMBDA_FILE")
echo -e "\n--- Step 2: Using calibrated parameters ---"
echo "  lambda_w = $FIXED_LAMBDA_W"
echo "  lambda_a = $FIXED_LAMBDA_A"
echo "  Lags file = $GOLDEN_LAGS_FILE"

# 3. Run test pipelines
GOLDEN_RUN_DIR="$RESULTS_BASE_DIR/Golden_Test_Run"
run_pipeline "$GOLDEN_CSV" "$GOLDEN_RUN_DIR" "true" "Golden (Test)" "$FIXED_LAMBDA_SCRIPT" "$GOLDEN_LAGS_FILE" "false"

ANOMALY_RUN_DIR="$RESULTS_BASE_DIR/Anomaly_run"
run_pipeline "$ANOMALY_CSV" "$ANOMALY_RUN_DIR" "false" "Anomaly (Variance Burst)" "$FIXED_LAMBDA_SCRIPT" "$GOLDEN_LAGS_FILE" "false"

# 4. Diagnostics
DIAGNOSTICS_DIR="$RESULTS_BASE_DIR/diagnostics"
mkdir -p "$DIAGNOSTICS_DIR"

GOLDEN_WEIGHTS_FILE="$GOLDEN_RUN_DIR/weights/weights_enhanced.csv"
ANOMALY_WEIGHTS_FILE="$ANOMALY_RUN_DIR/weights/weights_enhanced.csv"

echo -e "\n=============================================================================="
echo "Running Diagnostics: VARIANCE BURST Anomaly"
echo "=============================================================================="

python3 "$DIAGNOSTIC_SCRIPT" \
  --golden-csv "$GOLDEN_CSV" \
  --anomaly-csv "$ANOMALY_CSV" \
  --golden-weights "$GOLDEN_WEIGHTS_FILE" \
  --anomaly-weights "$ANOMALY_WEIGHTS_FILE" \
  --ground-truth "$GROUND_TRUTH_JSON" \
  --anomaly-type "variance_burst_fixed_lambda" \
  --output-dir "$DIAGNOSTICS_DIR"

# 5. Detailed window analysis
echo -e "\n=============================================================================="
echo "Detailed Window Analysis"
echo "=============================================================================="

python3 "$PROJECT_ROOT/analyze_all_window_weights.py" \
  "$GOLDEN_WEIGHTS_FILE" \
  "$ANOMALY_WEIGHTS_FILE"

echo -e "\n=============================================================================="
echo "VARIANCE BURST TEST COMPLETE"
echo "=============================================================================="
