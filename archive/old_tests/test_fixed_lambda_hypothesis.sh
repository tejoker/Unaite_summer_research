#!/bin/bash

# ==============================================================================
# End-to-End Test: Fixed Lambda Hypothesis
#
# Purpose:
# This script tests the "Butterfly Effect" hypothesis by running a full
# analysis pipeline on both a Golden and an Anomaly dataset using a
# fixed-lambda version of DynoTEARS.
#
# It performs the following steps:
# 1. Generates weights for the Golden dataset using fixed lambdas.
# 2. Generates weights for the Anomaly dataset using the same fixed lambdas.
# 3. Runs the diagnostic script to compare the two weight sets.
#
# Expected Outcome:
# If the "Butterfly Effect" was the cause of early detections, the diagnostic
# script should now show the "Temporal Onset" of weight changes much closer
# to the actual anomaly location, not at an early window like 6.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

GOLDEN_CSV="$PROJECT_ROOT/data/Golden/chunking/output_of_the_1th_chunk.csv"
ANOMALY_CSV="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__spike.csv"
GROUND_TRUTH_JSON="$PROJECT_ROOT/data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__spike.json"

PREPROCESSING_SCRIPT="$PROJECT_ROOT/executable/final_pipeline/preprocessing_no_mi.py"
FIXED_LAMBDA_SCRIPT="$PROJECT_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"
DIAGNOSTIC_SCRIPT="$PROJECT_ROOT/diagnose_global_changes.py"

RESULTS_BASE_DIR="$PROJECT_ROOT/results/fixed_lambda_hypothesis_test_$(date +%Y%m%d_%H%M%S)"

echo "=============================================================================="
echo "Starting Fixed Lambda Hypothesis Test"
echo "Results will be stored in: $RESULTS_BASE_DIR"
echo "=============================================================================="

# --- Helper Function to run the pipeline ---
run_pipeline() {
    local input_csv=$1
    local output_dir=$2
    local is_golden=$3
    local run_type=$4      # "Golden" or "Anomaly"
    local script_to_run=$5 # The python script for causal discovery
    local lags_csv_path=${6:-""} # Optional path to a specific lags file to use
    local calibrate_mode=${7:-"false"} # Optional flag for calibration

    echo -e "\n--- Running Pipeline for: $run_type ---"
    mkdir -p "$output_dir"

    # Step 1: Preprocessing
    echo "  [1/2] Running preprocessing..."
    export INPUT_CSV_FILE="$input_csv"
    export RESULT_DIR="$output_dir"
    export IS_GOLDEN_DATA="$is_golden"
    # Pass the lags file to preprocessing if provided (for test runs)
    if [ -n "$lags_csv_path" ] && [ -f "$lags_csv_path" ]; then
        export INPUT_LAGS_CSV="$lags_csv_path"
        echo "  Using pre-calculated lags from: $INPUT_LAGS_CSV"
    else
        unset INPUT_LAGS_CSV  # Ensure it's not set if not provided
        echo "  Will calculate new lags from data"
    fi
    python3 "$PREPROCESSING_SCRIPT"

    # Step 2: Causal Discovery
    echo "  [2/2] Running causal discovery script: $(basename $script_to_run)..."
    local input_basename=$(basename "$input_csv" .csv)
    export INPUT_DIFFERENCED_CSV="$output_dir/${input_basename}_differenced_stationary_series.csv"
    # Use the provided lags file if specified, otherwise use the one from this run's preprocessing
    export INPUT_LAGS_CSV="${lags_csv_path:-$output_dir/${input_basename}_optimal_lags.csv}"
    echo "  Using lags file: $INPUT_LAGS_CSV"

    # Pass environment variables directly to the python command for robustness
    CALIBRATE_LAMBDAS=$calibrate_mode \
    FIXED_LAMBDA_W=${FIXED_LAMBDA_W:-0.1} \
    FIXED_LAMBDA_A=${FIXED_LAMBDA_A:-0.1} \
    python3 "$script_to_run"

    export CALIBRATE_LAMBDAS="false" # Reset for subsequent runs
    echo "--- Finished Pipeline for: $run_type ---"
}

# --- Main Execution ---

# 1. Calibrate all parameters on the Golden dataset
GOLDEN_CALIBRATION_DIR="$RESULTS_BASE_DIR/Golden_Calibration"
echo -e "\n--- Step 1: Calibrating all parameters on Golden data ---"
run_pipeline "$GOLDEN_CSV" "$GOLDEN_CALIBRATION_DIR" "true" "Golden (Calibration)" "$FIXED_LAMBDA_SCRIPT" "" "true" # The last two arguments are for lags_csv and calibrate_mode

# 2. Read the "true golden" parameters (lambdas and lags file path) from the calibration run
LAMBDA_FILE="$GOLDEN_CALIBRATION_DIR/best_lambdas.json"
GOLDEN_LAGS_FILE="$GOLDEN_CALIBRATION_DIR/output_of_the_1th_chunk_optimal_lags.csv"

if [ ! -f "$LAMBDA_FILE" ]; then
    echo "Error: Lambda calibration failed. Could not find $LAMBDA_FILE"
    exit 1
fi
if [ ! -f "$GOLDEN_LAGS_FILE" ]; then
    echo "Error: Lags calibration failed. Could not find $GOLDEN_LAGS_FILE"
    exit 1
fi

export FIXED_LAMBDA_W=$(jq -r '.lambda_w' "$LAMBDA_FILE")
export FIXED_LAMBDA_A=$(jq -r '.lambda_a' "$LAMBDA_FILE")
echo -e "\n--- Step 2: Using calibrated Golden parameters for all subsequent test runs ---"
echo "  lambda_w = $FIXED_LAMBDA_W"
echo "  lambda_a = $FIXED_LAMBDA_A"
echo "  Lags file = $GOLDEN_LAGS_FILE"

# 3. Run final test pipelines with the *same* fixed parameters
GOLDEN_RUN_DIR="$RESULTS_BASE_DIR/Golden_Test_Run"
run_pipeline "$GOLDEN_CSV" "$GOLDEN_RUN_DIR" "true" "Golden (Test)" "$FIXED_LAMBDA_SCRIPT" "$GOLDEN_LAGS_FILE" "false"
ANOMALY_RUN_DIR="$RESULTS_BASE_DIR/Anomaly_run"
run_pipeline "$ANOMALY_CSV" "$ANOMALY_RUN_DIR" "false" "Anomaly" "$FIXED_LAMBDA_SCRIPT" "$GOLDEN_LAGS_FILE" "false"

# 4. Run Diagnostics to Compare the final, controlled test runs
DIAGNOSTICS_DIR="$RESULTS_BASE_DIR/diagnostics"
mkdir -p "$DIAGNOSTICS_DIR"

GOLDEN_WEIGHTS_FILE="$GOLDEN_RUN_DIR/weights/weights_enhanced.csv"
ANOMALY_WEIGHTS_FILE="$ANOMALY_RUN_DIR/weights/weights_enhanced.csv"

echo -e "\n=============================================================================="
echo "Running Diagnostics to Compare Golden vs. Anomaly (both with fixed lambdas)"
echo "=============================================================================="

python3 "$DIAGNOSTIC_SCRIPT" \
  --golden-csv "$GOLDEN_CSV" \
  --anomaly-csv "$ANOMALY_CSV" \
  --golden-weights "$GOLDEN_WEIGHTS_FILE" \
  --anomaly-weights "$ANOMALY_WEIGHTS_FILE" \
  --ground-truth "$GROUND_TRUTH_JSON" \
  --anomaly-type "spike_fixed_lambda" \
  --output-dir "$DIAGNOSTICS_DIR"

echo -e "\n=============================================================================="
echo "TEST COMPLETE"
echo "=============================================================================="
echo "Check the diagnostic output above. If the 'Temporal Onset' is still at an early window,"
echo "it confirms that GPU non-determinism (optimization noise) is the root cause."
echo "=============================================================================="