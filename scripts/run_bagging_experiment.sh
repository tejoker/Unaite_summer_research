#!/bin/bash
# ------------------------------------------------------------------
# PYTHON ENVIRONMENT (Robust Explicit Path)
# ------------------------------------------------------------------
PYTHON_EXEC="/home/nicolas_b/.venv/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then PYTHON_EXEC="python3"; fi

# run_bagging_experiment.sh
#
# Orchestrates the "Bagging 50 Runs" experiment for robust anomaly detection.
# 1. Generates N independent runs of the Tucker-CAM pipeline on the target data.
# 2. Runs Dual-Metric Anomaly Detection using the ensemble of results.
#
# Usage:
#   ./scripts/run_bagging_experiment.sh [DATA_FILE] [N_RUNS] [OUTPUT_DIR]

set -e

# Default values
DATA_FILE="${1:-telemanom/test_dataset_merged_clean.csv}"
N_RUNS="${2:-50}"
OUTPUT_DIR="${3:-results/bagging_experiment}"

# Paths
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATOR_SCRIPT="$WORKSPACE_ROOT/executable/experiments/generate_bagging_runs.py"
DETECTOR_SCRIPT="$WORKSPACE_ROOT/executable/dual_metric_anomaly_detection.py"
GOLDEN_BASELINE="${GOLDEN_BASELINE:-$WORKSPACE_ROOT/results/golden_baseline/weights/weights_enhanced.csv}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}         TUCKER-CAM BAGGING EXPERIMENT (N=${N_RUNS})            ${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "Data File:      $DATA_FILE"
echo -e "Output Dir:     $OUTPUT_DIR"
echo -e "Golden Baseline: $GOLDEN_BASELINE"
echo -e "Generator:      $GENERATOR_SCRIPT"
echo -e "Detector:       $DETECTOR_SCRIPT"
echo -e "${BLUE}================================================================${NC}"

# Explicitly enable Tucker-CAM and Parallel Mode
export USE_TUCKER_CAM=true
export USE_PARALLEL=true


# Check inputs
if [ ! -f "$DATA_FILE" ]; then
    # Try resolving relative to workspace root
    if [ -f "$WORKSPACE_ROOT/$DATA_FILE" ]; then
        DATA_FILE="$WORKSPACE_ROOT/$DATA_FILE"
    else
        echo -e "${RED}Error: Data file not found: $DATA_FILE${NC}"
        exit 1
    fi
fi

if [ ! -f "$GOLDEN_BASELINE" ]; then
    echo -e "${RED}Error: Golden baseline weights not found at $GOLDEN_BASELINE${NC}"
    echo -e "${RED}Please run the standard benchmark first to generate the baseline.${NC}"
    exit 1
fi

# Step 1: Generate Bagging Runs
RUNS_DIR="$OUTPUT_DIR/runs"
echo -e "${GREEN}[Step 1] Generating ${N_RUNS} bagging runs...${NC}"
$PYTHON_EXEC "$GENERATOR_SCRIPT" \
    --data "$DATA_FILE" \
    --output-dir "$RUNS_DIR" \
    --n-runs "$N_RUNS" \
    --workers ${N_WORKERS:-4} # Use parallel workers for speed

# Step 2: Run Ensemble Detection
echo -e "${GREEN}[Step 2] Running Ensemble Anomaly Detection...${NC}"
DETECTION_OUTPUT="$OUTPUT_DIR/anomaly_detection_bagged.csv"

# Pick the first run as the "Test" reference (for metadata/timeline info)
TEST_REF="$RUNS_DIR/run_000/weights/weights_enhanced.csv"

$PYTHON_EXEC "$DETECTOR_SCRIPT" \
    --golden "$GOLDEN_BASELINE" \
    --test "$TEST_REF" \
    --ensemble "$RUNS_DIR" \
    --output "$DETECTION_OUTPUT" \
    --metric frobenius \
    --lookback 5

echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Experiment Complete!${NC}"
echo -e "Results saved to: $DETECTION_OUTPUT"
echo -e "${BLUE}================================================================${NC}"
