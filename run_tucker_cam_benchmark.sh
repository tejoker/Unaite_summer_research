#!/bin/bash
#
# Tucker-CAM Benchmark Pipeline (Sequential Mode)
#
# Processes two datasets:
#   1. Golden baseline (4,308 timesteps, normal operation) - 421 windows
#   2. Full test timeline (8,640 timesteps, all 105 anomalies) - 842 windows
#
# Uses Tucker-decomposed Fast CAM-DAG for memory-efficient nonlinear causal discovery
# Runtime: ~126 hours (~5.25 days) in sequential mode
#
# Usage: bash run_tucker_cam_benchmark.sh
# Monitor: tail -f full_benchmark.log
#

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# Redirect all output to log file AND console
exec > >(tee -a full_benchmark.log)
exec 2>&1

echo "================================================================================"
echo "STARTING TUCKER-CAM BENCHMARK: $(date)"
echo "================================================================================"
echo "Log file: $WORKSPACE_DIR/full_benchmark.log"
echo "Monitor: tail -f full_benchmark.log"
echo ""

# Use venv Python if available
if [ -f "${HOME}/.venv/bin/python3" ]; then
    PYTHON="${HOME}/.venv/bin/python3"
else
    PYTHON="python3"
fi

# ============================================================================
# Tucker-CAM Configuration
# ============================================================================
export USE_TUCKER_CAM=true
export USE_PARALLEL=true       # Parallel with 1 worker = fresh process per window (clean memory)
export N_WORKERS=1             # Single worker prevents memory issues while keeping clean isolation

# PyTorch memory allocator configuration to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tucker ranks (memory/expressiveness tradeoff)
export TUCKER_RANK_W=20        # Contemporaneous edges (higher = more expressive)
export TUCKER_RANK_A=10        # Lagged edges (can be lower)

# P-spline parameters
export N_KNOTS=5               # Number of B-spline knots
export LAMBDA_SMOOTH=0.01      # Smoothness penalty

# Option D: Zero L1 penalties + post-hoc Top-K sparsification
export LAMBDA_W=0.0            # No L1 penalty on contemporaneous
export LAMBDA_A=0.0            # No L1 penalty on lagged
export TOP_K=10000             # Keep top 10K edges per window

# Rolling window parameters
export WINDOW_SIZE=100
export STRIDE=10

echo "================================================================================"
echo "TUCKER-CAM BENCHMARK PIPELINE"
echo "================================================================================"
echo "Processing two datasets with Tucker-decomposed Fast CAM-DAG:"
echo "  [1/2] Golden baseline (normal operation)"
echo "  [2/2] Full test timeline (all anomalies)"
echo ""
echo "Tucker Configuration:"
echo "  Ranks: r_w=$TUCKER_RANK_W, r_a=$TUCKER_RANK_A"
echo "  P-splines: n_knots=$N_KNOTS, lambda_smooth=$LAMBDA_SMOOTH"
echo "  Option D: lambda_w=$LAMBDA_W, lambda_a=$LAMBDA_A"
echo "  Top-K: $TOP_K edges/window"
echo ""
echo "Estimated runtime: ~3 hours"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# ============================================================================
# STAGE 0: Prepare Datasets (NaN Handling)
# ============================================================================
echo "[STAGE 0/2] Preparing Datasets"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Check if clean datasets already exist
GOLDEN_CLEAN="telemanom/golden_period_dataset_clean.csv"
TEST_CLEAN="telemanom/test_dataset_merged_clean.csv"

if [ -f "$GOLDEN_CLEAN" ] && [ -f "$TEST_CLEAN" ]; then
    echo "Clean datasets already exist:"
    echo "  ✓ $GOLDEN_CLEAN"
    echo "  ✓ $TEST_CLEAN"
    echo ""
    echo "Skipping NaN handling. Delete these files to re-run preparation."
else
    echo "Running NaN handling (ffill + dropna)..."
    $PYTHON telemanom/prepare_datasets.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "✗ Dataset preparation failed!"
        exit 1
    fi
fi

STAGE0_TIME=$(date +%s)
STAGE0_DURATION=$((STAGE0_TIME - START_TIME))
echo "Stage 0 completed in ${STAGE0_DURATION}s"
echo ""

# ============================================================================
# STAGE 1: Process Golden Baseline
# ============================================================================
echo "[STAGE 1/2] Processing Golden Baseline"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

GOLDEN_INPUT="$GOLDEN_CLEAN"
GOLDEN_OUTPUT="results/golden_baseline"
GOLDEN_LOG="logs/golden_baseline.log"

mkdir -p "$GOLDEN_OUTPUT" "$(dirname "$GOLDEN_LOG")"

echo "Input:  $GOLDEN_INPUT"
echo "Output: $GOLDEN_OUTPUT"
echo "Log:    $GOLDEN_LOG"
echo ""

# Preprocessing
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] Preprocessing - STARTING"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Input: $GOLDEN_INPUT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Output: $GOLDEN_OUTPUT"
export INPUT_CSV_FILE="$GOLDEN_INPUT"
export RESULT_DIR="$GOLDEN_OUTPUT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Executing preprocessing script..."
$PYTHON executable/final_pipeline/preprocessing_no_mi.py > "$GOLDEN_LOG" 2>&1
PREPROC_EXIT=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Preprocessing exit code: $PREPROC_EXIT"
if [ $PREPROC_EXIT -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Preprocessing FAILED! Check log: $GOLDEN_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Preprocessing completed successfully"

# Move preprocessing outputs (both CSV and NPY formats)
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Organizing preprocessing outputs..."
mkdir -p "${GOLDEN_OUTPUT}/preprocessing"
mv "${GOLDEN_OUTPUT}"/*_differenced_stationary_series.csv "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${GOLDEN_OUTPUT}"/*_differenced_stationary_series.npy "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${GOLDEN_OUTPUT}"/*_optimal_lags.csv "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${GOLDEN_OUTPUT}"/*_optimal_lags.npy "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${GOLDEN_OUTPUT}"/*_columns.npy "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true

PREPROC_SIZE=$(du -h "${GOLDEN_OUTPUT}/preprocessing/"*_differenced*.npy 2>/dev/null | cut -f1)
if [ -z "$PREPROC_SIZE" ]; then
    PREPROC_SIZE=$(du -h "${GOLDEN_OUTPUT}/preprocessing/"*_differenced*.csv 2>/dev/null | cut -f1)
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Preprocessing complete ($PREPROC_SIZE)"

# Tucker-CAM
if [ "$USE_PARALLEL" = "true" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/2] Running Tucker-CAM (PARALLEL mode with ${N_WORKERS} workers)"
    MODE_MSG="Parallel with ${N_WORKERS} workers"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/2] Running Tucker-CAM (sequential mode)"
    MODE_MSG="Sequential"
fi
TUCKER_START=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Tucker-CAM STARTING"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Mode: $MODE_MSG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Command: $PYTHON executable/launcher.py --skip-steps preprocessing"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Logging to: $GOLDEN_LOG"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Estimated time: ~42 hours for 421 windows (~6 min/window)"
echo ""

$PYTHON executable/launcher.py --skip-steps preprocessing >> "$GOLDEN_LOG" 2>&1
TUCKER_EXIT_CODE=$?
TUCKER_END=$(date +%s)
TUCKER_DURATION=$((TUCKER_END - TUCKER_START))

echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Tucker-CAM FINISHED"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Elapsed time: ${TUCKER_DURATION}s (~$((TUCKER_DURATION / 60)) minutes)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Exit code: $TUCKER_EXIT_CODE"

if [ $TUCKER_EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Tucker-CAM FAILED with exit code $TUCKER_EXIT_CODE!"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Last 30 lines of log:"
    echo "  ──────────────────────────────────────────────────────────────────"
    tail -30 "$GOLDEN_LOG" | sed 's/^/    /'
    echo "  ──────────────────────────────────────────────────────────────────"
    exit 1
fi

# Check what files were created
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Checking output directories..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Results dir: ${GOLDEN_OUTPUT}"

# Verify output - check for NPY (parallel) or CSV (sequential) format
if [ -f "${GOLDEN_OUTPUT}/causal_discovery/window_edges.npy" ]; then
    WEIGHTS_FILE="${GOLDEN_OUTPUT}/causal_discovery/window_edges.npy"
    WEIGHTS_SIZE=$(du -h "$WEIGHTS_FILE" | cut -f1)
    # Count edges from NPY file
    NUM_EDGES=$($PYTHON -c "import numpy as np; edges=np.load('${GOLDEN_OUTPUT}/causal_discovery/window_edges.npy', allow_pickle=True); print(len(edges) if edges.ndim > 0 else 0)" 2>/dev/null || echo "unknown")
    echo "  ✓ Tucker-CAM complete (NPY format)"
    echo "    → Weights: $WEIGHTS_SIZE ($NUM_EDGES edges)"
    if [ -f "${GOLDEN_OUTPUT}/causal_discovery/progress.txt" ]; then
        echo "    → Progress: $(cat ${GOLDEN_OUTPUT}/causal_discovery/progress.txt)"
    fi
    
    # Convert NPY to enhanced CSV for backward compatibility (optional)
    echo "  [3/3] Converting to enhanced CSV format (for legacy tools)..."
    mkdir -p "${GOLDEN_OUTPUT}/weights"
    $PYTHON executable/convert_npy_to_enhanced_csv.py \
        "${GOLDEN_OUTPUT}/causal_discovery/window_edges.npy" \
        "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        ENHANCED_SIZE=$(du -h "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" | cut -f1)
        echo "  ✓ Enhanced CSV created: $ENHANCED_SIZE (legacy format)"
    else
        echo "  ⓘ CSV conversion skipped (NPY format is primary)"
    fi
elif [ -f "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" ]; then
    WEIGHTS_SIZE=$(du -h "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" | cut -f1)
    NUM_EDGES=$(tail -n +2 "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" | wc -l)
    echo "  ✓ Tucker-CAM complete (CSV format)"
    echo "    → Weights: $WEIGHTS_SIZE ($NUM_EDGES edges)"
else
    echo "  ✗ Weights file not found at expected location!"
    echo "    → Expected (NPY): ${GOLDEN_OUTPUT}/causal_discovery/window_edges.npy"
    echo "    → Expected (CSV): ${GOLDEN_OUTPUT}/weights/weights_enhanced.csv"
    echo "    → Last 50 lines of log:"
    echo "  ──────────────────────────────────────────────────────────────────"
    tail -50 "$GOLDEN_LOG" | sed 's/^/    /'
    echo "  ──────────────────────────────────────────────────────────────────"
    exit 1
fi

STAGE1_TIME=$(date +%s)
STAGE1_DURATION=$((STAGE1_TIME - STAGE0_TIME))
echo ""
echo "✓ Golden baseline complete in ${STAGE1_DURATION}s (~$((STAGE1_DURATION / 60)) minutes)"
echo ""

# ============================================================================
# STAGE 2: Process Full Test Timeline
# ============================================================================
echo "[STAGE 2/2] Processing Full Test Timeline"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

TEST_INPUT="$TEST_CLEAN"
TEST_OUTPUT="results/test_timeline"
TEST_LOG="logs/test_timeline.log"

mkdir -p "$TEST_OUTPUT" "$(dirname "$TEST_LOG")"

echo "Input:  $TEST_INPUT"
echo "Output: $TEST_OUTPUT"
echo "Log:    $TEST_LOG"
echo ""

# Preprocessing
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] Preprocessing - STARTING"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Input: $TEST_INPUT"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Output: $TEST_OUTPUT"
export INPUT_CSV_FILE="$TEST_INPUT"
export RESULT_DIR="$TEST_OUTPUT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Executing preprocessing script..."
$PYTHON executable/final_pipeline/preprocessing_no_mi.py > "$TEST_LOG" 2>&1
PREPROC_EXIT=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Preprocessing exit code: $PREPROC_EXIT"
if [ $PREPROC_EXIT -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Preprocessing FAILED! Check log: $TEST_LOG"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Preprocessing completed successfully"

# Move preprocessing outputs (both CSV and NPY formats)
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Organizing preprocessing outputs..."
mkdir -p "${TEST_OUTPUT}/preprocessing"
mv "${TEST_OUTPUT}"/*_differenced_stationary_series.csv "${TEST_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${TEST_OUTPUT}"/*_differenced_stationary_series.npy "${TEST_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${TEST_OUTPUT}"/*_optimal_lags.csv "${TEST_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${TEST_OUTPUT}"/*_optimal_lags.npy "${TEST_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${TEST_OUTPUT}"/*_columns.npy "${TEST_OUTPUT}/preprocessing/" 2>/dev/null || true

PREPROC_SIZE=$(du -h "${TEST_OUTPUT}/preprocessing/"*_differenced*.npy 2>/dev/null | cut -f1)
if [ -z "$PREPROC_SIZE" ]; then
    PREPROC_SIZE=$(du -h "${TEST_OUTPUT}/preprocessing/"*_differenced*.csv 2>/dev/null | cut -f1)
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Preprocessing complete ($PREPROC_SIZE)"

# Tucker-CAM
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/2] Running Tucker-CAM - STARTING"
TUCKER_START=$(date +%s)
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Command: $PYTHON executable/launcher.py --skip-steps preprocessing"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Logging to: $TEST_LOG"

$PYTHON executable/launcher.py --skip-steps preprocessing >> "$TEST_LOG" 2>&1
TUCKER_EXIT_CODE=$?
TUCKER_END=$(date +%s)
TUCKER_DURATION=$((TUCKER_END - TUCKER_START))

echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Tucker-CAM FINISHED"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Elapsed time: ${TUCKER_DURATION}s (~$((TUCKER_DURATION / 60)) minutes)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Exit code: $TUCKER_EXIT_CODE"

if [ $TUCKER_EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Tucker-CAM FAILED with exit code $TUCKER_EXIT_CODE!"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Last 30 lines of log:"
    echo "  ──────────────────────────────────────────────────────────────────"
    tail -30 "$TEST_LOG" | sed 's/^/    /'
    echo "  ──────────────────────────────────────────────────────────────────"
    exit 1
fi

# Check what files were created
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Checking output directories..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Results dir: ${TEST_OUTPUT}"

# Verify output - check for NPY (parallel) or CSV (sequential) format
if [ -f "${TEST_OUTPUT}/causal_discovery/window_edges.npy" ]; then
    WEIGHTS_FILE="${TEST_OUTPUT}/causal_discovery/window_edges.npy"
    WEIGHTS_SIZE=$(du -h "$WEIGHTS_FILE" | cut -f1)
    # Count edges from NPY file
    NUM_EDGES=$($PYTHON -c "import numpy as np; edges=np.load('${TEST_OUTPUT}/causal_discovery/window_edges.npy', allow_pickle=True); print(len(edges) if edges.ndim > 0 else 0)" 2>/dev/null || echo "unknown")
    echo "  ✓ Tucker-CAM complete (NPY format)"
    echo "    → Weights: $WEIGHTS_SIZE ($NUM_EDGES edges)"
    if [ -f "${TEST_OUTPUT}/causal_discovery/progress.txt" ]; then
        echo "    → Progress: $(cat ${TEST_OUTPUT}/causal_discovery/progress.txt)"
    fi
    
    # Convert NPY to enhanced CSV for backward compatibility (optional)
    echo "  [3/3] Converting to enhanced CSV format (for legacy tools)..."
    mkdir -p "${TEST_OUTPUT}/weights"
    $PYTHON executable/convert_npy_to_enhanced_csv.py \
        "${TEST_OUTPUT}/causal_discovery/window_edges.npy" \
        "${TEST_OUTPUT}/weights/weights_enhanced.csv" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        ENHANCED_SIZE=$(du -h "${TEST_OUTPUT}/weights/weights_enhanced.csv" | cut -f1)
        echo "  ✓ Enhanced CSV created: $ENHANCED_SIZE (legacy format)"
    else
        echo "  ⓘ CSV conversion skipped (NPY format is primary)"
    fi
elif [ -f "${TEST_OUTPUT}/weights/weights_enhanced.csv" ]; then
    WEIGHTS_SIZE=$(du -h "${TEST_OUTPUT}/weights/weights_enhanced.csv" | cut -f1)
    NUM_EDGES=$(tail -n +2 "${TEST_OUTPUT}/weights/weights_enhanced.csv" | wc -l)
    echo "  ✓ Tucker-CAM complete (CSV format)"
    echo "    → Weights: $WEIGHTS_SIZE ($NUM_EDGES edges)"
else
    echo "  ✗ Weights file not found at expected location!"
    echo "    → Expected (NPY): ${TEST_OUTPUT}/causal_discovery/window_edges.npy"
    echo "    → Expected (CSV): ${TEST_OUTPUT}/weights/weights_enhanced.csv"
    echo "    → Last 50 lines of log:"
    echo "  ──────────────────────────────────────────────────────────────────"
    tail -50 "$TEST_LOG" | sed 's/^/    /'
    echo "  ──────────────────────────────────────────────────────────────────"
    exit 1
fi

STAGE2_TIME=$(date +%s)
STAGE2_DURATION=$((STAGE2_TIME - STAGE1_TIME))
echo ""
echo "✓ Test timeline complete in ${STAGE2_DURATION}s (~$((STAGE2_DURATION / 60)) minutes)"
echo ""

# ============================================================================
# STAGE 3: Dual-Metric Anomaly Detection
# ============================================================================
echo "[STAGE 3/3] Dual-Metric Anomaly Detection"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

ANOMALY_OUTPUT="results/anomaly_detection"
ANOMALY_LOG="logs/anomaly_detection.log"

mkdir -p "$ANOMALY_OUTPUT" "$(dirname "$ANOMALY_LOG")"

# Use NPY files directly if available (faster), fallback to CSV
if [ -f "results/golden_baseline/causal_discovery/window_edges.npy" ]; then
    GOLDEN_WEIGHTS="results/golden_baseline/causal_discovery/window_edges.npy"
else
    GOLDEN_WEIGHTS="results/golden_baseline/weights/weights_enhanced.csv"
fi

if [ -f "results/test_timeline/causal_discovery/window_edges.npy" ]; then
    TEST_WEIGHTS="results/test_timeline/causal_discovery/window_edges.npy"
else
    TEST_WEIGHTS="results/test_timeline/weights/weights_enhanced.csv"
fi

echo "Golden baseline: $GOLDEN_WEIGHTS"
echo "Test timeline:   $TEST_WEIGHTS"
echo "Output:          $ANOMALY_OUTPUT"
echo "Log:             $ANOMALY_LOG"
echo ""

# Run dual-metric anomaly detection
echo "  Running dual-metric detection..."
$PYTHON executable/dual_metric_anomaly_detection.py \
    --golden "$GOLDEN_WEIGHTS" \
    --test "$TEST_WEIGHTS" \
    --output "$ANOMALY_OUTPUT/anomaly_detection_results.csv" \
    --metric frobenius \
    --lookback 5 \
    --lag 0 > "$ANOMALY_LOG" 2>&1

if [ $? -ne 0 ]; then
    echo "  ✗ Anomaly detection failed! Check log: $ANOMALY_LOG"
    echo ""
    echo "  This is not critical - the weights are still available for manual analysis."
    echo "  You can run anomaly detection separately later."
    echo ""
else
    # Verify output
    if [ -f "$ANOMALY_OUTPUT/anomaly_detection_results.csv" ]; then
        RESULTS_SIZE=$(du -h "$ANOMALY_OUTPUT/anomaly_detection_results.csv" | cut -f1)
        NUM_ANOMALIES=$(grep -v "NORMAL" "$ANOMALY_OUTPUT/anomaly_detection_results.csv" | tail -n +2 | wc -l)
        TOTAL_WINDOWS=$(tail -n +2 "$ANOMALY_OUTPUT/anomaly_detection_results.csv" | wc -l)
        echo "  ✓ Anomaly detection complete"
        echo "    → Results: $RESULTS_SIZE ($NUM_ANOMALIES anomalies in $TOTAL_WINDOWS windows)"

        # Show detection summary
        echo ""
        echo "  Detection Summary:"
        echo "  ──────────────────────────────────────────────────────────────────"
        tail -n 20 "$ANOMALY_LOG" | grep -E "NORMAL|NEW_ANOMALY_ONSET|RECOVERY_FLUCTUATION|CASCADE_OR_PERSISTENT|windows" || true
        echo "  ──────────────────────────────────────────────────────────────────"
    else
        echo "  ✗ Results file not created!"
    fi
fi

STAGE3_TIME=$(date +%s)
STAGE3_DURATION=$((STAGE3_TIME - STAGE2_TIME))
echo ""
echo "✓ Anomaly detection complete in ${STAGE3_DURATION}s"
echo ""

# ============================================================================
# Final Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "================================================================================"
echo "TUCKER-CAM BENCHMARK COMPLETE!"
echo "================================================================================"
echo ""
echo "Stage Summary:"
echo "  [0/3] Dataset preparation:     ${STAGE0_DURATION}s"
echo "  [1/3] Golden baseline:         ${STAGE1_DURATION}s (~$((STAGE1_DURATION / 60)) min)"
echo "  [2/3] Test timeline:           ${STAGE2_DURATION}s (~$((STAGE2_DURATION / 60)) min)"
echo "  [3/3] Anomaly detection:       ${STAGE3_DURATION}s"
echo ""
echo "Total runtime: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "Output locations:"
echo "  Golden weights (NPY):  results/golden_baseline/causal_discovery/window_edges.npy"
echo "  Golden weights (CSV):  results/golden_baseline/weights/weights_enhanced.csv"
echo "  Test weights (NPY):    results/test_timeline/causal_discovery/window_edges.npy"
echo "  Test weights (CSV):    results/test_timeline/weights/weights_enhanced.csv"
echo "  Anomaly results:       results/anomaly_detection/anomaly_detection_results.csv"
echo ""
echo "Logs:"
echo "  Golden baseline:   logs/golden_baseline.log"
echo "  Test timeline:     logs/test_timeline.log"
echo "  Anomaly detection: logs/anomaly_detection.log"
echo ""
echo "================================================================================"
echo "Next steps:"
echo "  1. Review anomaly detection results in results/anomaly_detection/"
echo "  2. Evaluate against labeled anomalies (telemanom/labeled_anomalies.csv)"
echo "  3. Compare with LSTM baseline (Hundman et al. 2018)"
echo "  4. Perform root cause analysis on detected anomalies"
echo "================================================================================"
echo ""

exit 0
