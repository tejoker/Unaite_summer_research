#!/bin/bash
# run_entity_pipeline.sh
#
# Runs the COMPLETE pipeline for a single entity (e.g. machine-1-6).
# 1. Generates Golden Baseline (from Train data)
# 2. Generates Bagging Runs (from Test data)
# 3. Runs Ensemble Anomaly Detection
# 4. Evaluates Performance (F1, Precision, Recall, AUC-PR)
# 5. Runs Root Cause Analysis (RCA)
#
# Usage: ./scripts/run_entity_pipeline.sh [ENTITY_NAME] [N_RUNS]
# Example: ./scripts/run_entity_pipeline.sh machine-1-6 50

set -e

ENTITY="${1:-machine-1-6}"
N_RUNS="${2:-50}"

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$WORKSPACE_ROOT/data/SMD"

echo "==================================================================="
echo "STARTING FULL PIPELINE FOR: $ENTITY"
echo "==================================================================="

# ------------------------------------------------------------------
# PYTHON ENVIRONMENT (Robust Explicit Path)
# ------------------------------------------------------------------
# Use absolute path to venv python to avoid shell activation issues
PYTHON_EXEC="/home/nicolas_b/.venv/bin/python3"

# Fallback if specific venv missing (unlikely but safe)
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "WARNING: Venv python not found at $PYTHON_EXEC, falling back to python3"
    PYTHON_EXEC="python3"
fi
echo "Using Python: $PYTHON_EXEC"

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Enable Tucker-CAM to prevent OOM
export USE_TUCKER_CAM=true
export USE_PARALLEL=true
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Step 1: Golden Baseline Generation (Train Data)
# ------------------------------------------------------------------
RESULTS_ROOT="${RESULTS_ROOT:-$WORKSPACE_ROOT/results}"
TRAIN_DATA="$DATA_ROOT/train/${ENTITY}.npy"
GOLDEN_OUTPUT_DIR="$RESULTS_ROOT/SMD_${ENTITY}_golden_baseline"
GOLDEN_WEIGHTS_FILE="$GOLDEN_OUTPUT_DIR/run_000/weights/weights_enhanced.csv"

echo ""
echo "[Step 1] Checking/Generating Golden Baseline..."
if [ ! -f "$GOLDEN_WEIGHTS_FILE" ]; then
    echo "  Generating Golden Baseline from $TRAIN_DATA..."
    echo "  (Running 1 robust run on training data)"
    
    $PYTHON_EXEC "$WORKSPACE_ROOT/executable/experiments/generate_bagging_runs.py" \
        --data "$TRAIN_DATA" \
        --output-dir "$GOLDEN_OUTPUT_DIR" \
        --n-runs 1 \
        --workers ${N_WORKERS:-4}
        
    if [ ! -f "$GOLDEN_WEIGHTS_FILE" ]; then
        echo "ERROR: Golden baseline generation failed. File not found: $GOLDEN_WEIGHTS_FILE"
        exit 1
    fi
else
    echo "  Golden Baseline already exists: $GOLDEN_WEIGHTS_FILE"
fi

export GOLDEN_BASELINE="$GOLDEN_WEIGHTS_FILE"

# ------------------------------------------------------------------
# Step 2 & 3: Bagging Runs & Ensemble Detection (Test Data)
# ------------------------------------------------------------------
TEST_DATA="$DATA_ROOT/test/${ENTITY}.npy"
TEST_OUTPUT_DIR="$RESULTS_ROOT/bagging_SMD_${ENTITY}"
DETECTION_FILE="$TEST_OUTPUT_DIR/anomaly_detection_bagged.csv"

echo ""
echo "[Step 2 & 3] Running Bagging Experiment & Detection..."
./scripts/run_bagging_experiment.sh "$TEST_DATA" "$N_RUNS" "$TEST_OUTPUT_DIR"

# ------------------------------------------------------------------
# Step 4: Evaluation
# ------------------------------------------------------------------
LABEL_FILE="$WORKSPACE_ROOT/ServerMachineDataset/test_label/${ENTITY}.txt"

echo ""
echo "[Step 4] Evaluating Results..."
if [ -f "$LABEL_FILE" ]; then
    $PYTHON_EXEC "$WORKSPACE_ROOT/scripts/evaluate_results.py" \
        "$DETECTION_FILE" \
        "$LABEL_FILE"
else
    echo "WARNING: Label file not found: $LABEL_FILE. Skipping evaluation."
fi

# ------------------------------------------------------------------
# Step 5: Root Cause Analysis (RCA)
# ------------------------------------------------------------------
echo ""
echo "[Step 5] Running Root Cause Analysis..."
RCA_REPORT="$TEST_OUTPUT_DIR/rca_report.txt"

# Ensure we use the correct feature names
COLUMNS_FILE="$DATA_ROOT/test/${ENTITY}_columns.npy"

# Run RCA
# Note: RCA script handles the column file logic now
$PYTHON_EXEC "$WORKSPACE_ROOT/scripts/run_rca_on_detections.py" \
    "$TEST_OUTPUT_DIR" \
    "$DETECTION_FILE" \
    "$GOLDEN_BASELINE" \
    > "$RCA_REPORT"

echo "RCA Report saved to: $RCA_REPORT"
cat "$RCA_REPORT"

echo ""
echo "==================================================================="
echo "PIPELINE COMPLETE FOR $ENTITY"
echo "==================================================================="
