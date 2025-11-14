#!/bin/bash
#
# Complete Telemanom Anomaly Analysis Pipeline
# 
# This script runs Tucker-CAM causal discovery on:
#   1. Re-extracts all 105 anomalies with correct class labels
#   2. Golden baseline (clean 4308 timesteps)
#   3. All 105 individual anomalies (each = Golden + anomaly sequence)
#
# Estimated runtime: 35-40 hours total
#   - Extraction: 30 seconds
#   - Golden baseline: 20 minutes
#   - 105 anomalies: ~20 min each = 35-38 hours
#

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# Use venv Python if available
if [ -f "${HOME}/.venv/bin/python3" ]; then
    PYTHON="${HOME}/.venv/bin/python3"
else
    PYTHON="python3"
fi

echo "================================================================================"
echo "TELEMANOM ANOMALY ANALYSIS PIPELINE"
echo "================================================================================"
echo "Pipeline stages:"
echo "  [1/3] Extract 105 individual anomalies with class labels"
echo "  [2/3] Run Tucker-CAM on Golden baseline"
echo "  [3/3] Run Tucker-CAM on all 105 anomalies"
echo ""
echo "Estimated runtime: 35-40 hours"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# ============================================================================
# STAGE 1: Extract Individual Anomalies
# ============================================================================
echo "[STAGE 1/3] Extracting Individual Anomalies"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Remove old extraction with incorrect labels
if [ -d "telemanom/individual_anomalies" ]; then
    echo "Removing old extraction..."
    rm -rf telemanom/individual_anomalies
fi

echo "Running extraction script..."
$PYTHON extract_individual_anomalies.py \
    --merged-test "telemanom/test_dataset_merged_clean.csv" \
    --golden "telemanom/golden_period_dataset_clean.csv" \
    --labels "telemanom/labeled_anomalies.csv" \
    --output-dir "telemanom/individual_anomalies"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Extraction failed!"
    exit 1
fi

# Verify extraction
TOTAL_FILES=$(ls telemanom/individual_anomalies/anomaly_*.csv 2>/dev/null | wc -l)
echo ""
echo "✓ Extraction complete: $TOTAL_FILES anomaly files created"

if [ $TOTAL_FILES -lt 100 ]; then
    echo "✗ Expected ~105 files, found only $TOTAL_FILES"
    exit 1
fi

STAGE1_TIME=$(date +%s)
STAGE1_DURATION=$((STAGE1_TIME - START_TIME))
echo "Stage 1 completed in ${STAGE1_DURATION}s"
echo ""

# ============================================================================
# STAGE 2: Run Tucker-CAM on Golden Baseline
# ============================================================================
echo "[STAGE 2/3] Running Tucker-CAM on Golden Baseline"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

GOLDEN_INPUT="telemanom/golden_period_dataset_clean.csv"
GOLDEN_OUTPUT="results/telemanom_golden_baseline"
GOLDEN_LOG="logs/telemanom_golden_baseline.log"

mkdir -p "$GOLDEN_OUTPUT" "$(dirname "$GOLDEN_LOG")"

echo "Input:  $GOLDEN_INPUT"
echo "Output: $GOLDEN_OUTPUT"
echo "Log:    $GOLDEN_LOG"
echo ""

# Step 2.1: Preprocessing
echo "  [1/2] Preprocessing Golden dataset..."
export INPUT_CSV_FILE="$GOLDEN_INPUT"
export RESULT_DIR="$GOLDEN_OUTPUT"

$PYTHON executable/final_pipeline/preprocessing_no_mi.py > "$GOLDEN_LOG" 2>&1

if [ $? -ne 0 ]; then
    echo "  ✗ Preprocessing failed! Check log: $GOLDEN_LOG"
    exit 1
fi

# Move preprocessing outputs to subdirectory (required by launcher)
mkdir -p "${GOLDEN_OUTPUT}/preprocessing"
mv "${GOLDEN_OUTPUT}"/*_differenced_stationary_series.csv "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true
mv "${GOLDEN_OUTPUT}"/*_optimal_lags.csv "${GOLDEN_OUTPUT}/preprocessing/" 2>/dev/null || true

PREPROC_SIZE=$(du -h "${GOLDEN_OUTPUT}/preprocessing/"*_differenced*.csv 2>/dev/null | cut -f1)
echo "  ✓ Preprocessing complete ($PREPROC_SIZE)"

# Step 2.2: Tucker-CAM
echo "  [2/2] Running Tucker-CAM (Option D, Top-K=10000)..."
$PYTHON executable/launcher.py --skip-steps preprocessing >> "$GOLDEN_LOG" 2>&1

if [ $? -ne 0 ]; then
    echo "  ✗ Tucker-CAM failed! Check log: $GOLDEN_LOG"
    exit 1
fi

# Verify output
if [ -f "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" ]; then
    WEIGHTS_SIZE=$(du -h "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" | cut -f1)
    NUM_EDGES=$(tail -n +2 "${GOLDEN_OUTPUT}/weights/weights_enhanced.csv" | wc -l)
    echo "  ✓ Tucker-CAM complete"
    echo "    → Weights: $WEIGHTS_SIZE ($NUM_EDGES edges)"
else
    echo "  ✗ Weights file not found!"
    exit 1
fi

STAGE2_TIME=$(date +%s)
STAGE2_DURATION=$((STAGE2_TIME - STAGE1_TIME))
echo ""
echo "✓ Golden baseline complete in ${STAGE2_DURATION}s (~$((STAGE2_DURATION / 60)) minutes)"
echo ""

# ============================================================================
# STAGE 3: Run Tucker-CAM on All 105 Anomalies
# ============================================================================
echo "[STAGE 3/3] Running Tucker-CAM on All 105 Anomalies"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

RESULTS_DIR="results/anomaly_signatures"
LOGS_DIR="logs/anomaly_signatures"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Get all anomaly files
ANOMALY_FILES=($(ls telemanom/individual_anomalies/anomaly_*.csv | sort))
TOTAL_ANOMALIES=${#ANOMALY_FILES[@]}

echo "Found $TOTAL_ANOMALIES anomaly files"
echo "Estimated time: ~$((TOTAL_ANOMALIES * 20 / 60)) hours"
echo ""

# Function to process one anomaly
process_anomaly() {
    local anomaly_file="$1"
    local anomaly_idx="$2"
    local total="$3"
    
    local anomaly_name=$(basename "$anomaly_file" .csv)
    local output_dir="${RESULTS_DIR}/${anomaly_name}"
    local log_file="${LOGS_DIR}/${anomaly_name}.log"
    
    # Check if already completed
    if [ -f "${output_dir}/weights/weights_enhanced.csv" ]; then
        echo "[$anomaly_idx/$total] ✓ SKIP: $anomaly_name (already completed)"
        return 0
    fi
    
    echo "[$anomaly_idx/$total] Processing: $anomaly_name"
    
    mkdir -p "$output_dir"
    
    local anom_start_time=$(date +%s)
    
    # Preprocessing
    export INPUT_CSV_FILE="$anomaly_file"
    export RESULT_DIR="$output_dir"
    
    $PYTHON executable/final_pipeline/preprocessing_no_mi.py > "$log_file" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[$anomaly_idx/$total] ✗ FAIL: $anomaly_name (preprocessing failed)"
        return 1
    fi
    
    # Move preprocessing outputs to subdirectory
    mkdir -p "${output_dir}/preprocessing"
    mv "${output_dir}"/*_differenced_stationary_series.csv "${output_dir}/preprocessing/" 2>/dev/null || true
    mv "${output_dir}"/*_optimal_lags.csv "${output_dir}/preprocessing/" 2>/dev/null || true
    
    # Tucker-CAM
    $PYTHON executable/launcher.py --skip-steps preprocessing >> "$log_file" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[$anomaly_idx/$total] ✗ FAIL: $anomaly_name (Tucker-CAM failed)"
        return 1
    fi
    
    # Verify output
    if [ -f "${output_dir}/weights/weights_enhanced.csv" ]; then
        local anom_end_time=$(date +%s)
        local anom_duration=$((anom_end_time - anom_start_time))
        local weights_size=$(du -h "${output_dir}/weights/weights_enhanced.csv" | cut -f1)
        echo "[$anomaly_idx/$total] ✓ DONE: $anomaly_name ($weights_size, ${anom_duration}s)"
        return 0
    else
        echo "[$anomaly_idx/$total] ✗ FAIL: $anomaly_name (no weights file)"
        return 1
    fi
}

# Process all anomalies sequentially
PASSED=0
FAILED=0
SKIPPED=0

for idx in "${!ANOMALY_FILES[@]}"; do
    anomaly_file="${ANOMALY_FILES[$idx]}"
    anomaly_idx=$((idx + 1))
    
    # Check if already completed before calling function
    anomaly_name=$(basename "$anomaly_file" .csv)
    if [ -f "${RESULTS_DIR}/${anomaly_name}/weights/weights_enhanced.csv" ]; then
        echo "[$anomaly_idx/$TOTAL_ANOMALIES] ✓ SKIP: $anomaly_name (already completed)"
        ((SKIPPED++))
        continue
    fi
    
    if process_anomaly "$anomaly_file" "$anomaly_idx" "$TOTAL_ANOMALIES"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi
    
    # Progress summary every 10 anomalies
    if [ $((anomaly_idx % 10)) -eq 0 ] || [ $anomaly_idx -eq $TOTAL_ANOMALIES ]; then
        echo ""
        echo "Progress: $anomaly_idx/$TOTAL_ANOMALIES processed (✓ $PASSED | ✗ $FAILED | ⊘ $SKIPPED)"
        
        # Time estimate
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - STAGE2_TIME))
        if [ $anomaly_idx -gt 0 ]; then
            AVG_TIME=$((ELAPSED / anomaly_idx))
            REMAINING=$((TOTAL_ANOMALIES - anomaly_idx))
            ETA=$((REMAINING * AVG_TIME))
            echo "Estimated time remaining: ~$((ETA / 3600))h $((ETA % 3600 / 60))m"
        fi
        echo ""
    fi
done

STAGE3_TIME=$(date +%s)
STAGE3_DURATION=$((STAGE3_TIME - STAGE2_TIME))

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# ============================================================================
# Final Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Stage Summary:"
echo "  [1/3] Extraction:       ${STAGE1_DURATION}s"
echo "  [2/3] Golden baseline:  ${STAGE2_DURATION}s (~$((STAGE2_DURATION / 60)) min)"
echo "  [3/3] 105 anomalies:    ${STAGE3_DURATION}s (~$((STAGE3_DURATION / 3600))h $((STAGE3_DURATION % 3600 / 60))m)"
echo ""
echo "Total runtime: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "Results Summary:"
echo "  ✓ Passed:  $PASSED"
echo "  ✗ Failed:  $FAILED"
echo "  ⊘ Skipped: $SKIPPED"
echo "  Total:     $TOTAL_ANOMALIES anomalies"
echo ""
echo "Output locations:"
echo "  Golden:    results/telemanom_golden_baseline/weights/weights_enhanced.csv"
echo "  Anomalies: results/anomaly_signatures/*/weights/weights_enhanced.csv"
echo "  Catalog:   telemanom/individual_anomalies/anomaly_catalog.csv"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All anomalies processed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Build comparison catalog:"
    echo "     $PYTHON build_anomaly_signature_catalog.py"
    echo ""
    echo "  2. Generate analysis:"
    echo "     $PYTHON analyze_anomaly_signatures.py"
else
    echo "⚠️  $FAILED anomalies failed. Check logs in: $LOGS_DIR"
    echo ""
    echo "To retry failed anomalies, simply re-run this script."
    echo "Completed anomalies will be automatically skipped."
fi

echo "================================================================================"

exit $FAILED
