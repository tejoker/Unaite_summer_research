#!/bin/bash

# =============================================================================
# Telemanom Anomaly Processing with Golden Baseline Parameters
# =============================================================================
# This script:
# 1. Preprocesses each anomaly dataset (calculates differenced series)
# 2. REPLACES calculated lags with golden baseline lags
# 3. Runs DynoTEARS with FIXED golden lambdas (0.537)
# 4. Processes 3 datasets in parallel
# =============================================================================

GOLDEN_BASELINE_DIR="results/telemanom_golden"
GOLDEN_LAGS="$GOLDEN_BASELINE_DIR/preprocessing/golden_period_dataset_mean_channel_optimal_lags.csv"
ANOMALY_DIR="data/Anomaly/telemanom"
RESULTS_BASE="results/telemanom"
LOG_FILE="telemanom_processing.log"

# Golden lambda values (calculated from high-dimensional scaling)
LAMBDA_W=0.537
LAMBDA_A=0.537

# Check if golden lags exist
if [ ! -f "$GOLDEN_LAGS" ]; then
    echo "ERROR: Golden lags file not found: $GOLDEN_LAGS"
    echo "Please ensure golden baseline was processed first"
    exit 1
fi

echo "=============================================================================" | tee "$LOG_FILE"
echo "Telemanom Anomaly Processing with Golden Parameters" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "Golden lags: $GOLDEN_LAGS" | tee -a "$LOG_FILE"
echo "Lambda_w: $LAMBDA_W" | tee -a "$LOG_FILE"
echo "Lambda_a: $LAMBDA_A" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Get all anomaly CSV files
ANOMALY_FILES=($ANOMALY_DIR/isolated_anomaly_*.csv)
TOTAL=${#ANOMALY_FILES[@]}

echo "Found $TOTAL anomaly datasets to process" | tee -a "$LOG_FILE"
echo "Processing 3 at a time..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to process a single anomaly dataset
process_anomaly() {
    local anomaly_csv=$1
    local lambda_w=$2
    local lambda_a=$3
    local golden_lags=$4
    local basename=$(basename "$anomaly_csv" .csv)
    local result_dir="$RESULTS_BASE/$basename"
    
    echo "[$(date +%H:%M:%S)] START: $basename" >> "$LOG_FILE"
    
    # Create result directories
    mkdir -p "$result_dir/preprocessing"
    
    # =========================================================================
    # STEP 1: Preprocess anomaly data (get differenced stationary series)
    # =========================================================================
    echo "  [1/3] Preprocessing $basename..." >> "$LOG_FILE"
    
    INPUT_CSV_FILE="$anomaly_csv" \
    RESULT_DIR="$result_dir" \
    python3 executable/final_pipeline/preprocessing_no_mi.py >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ FAILED preprocessing: $basename" >> "$LOG_FILE"
        return 1
    fi
    
    # =========================================================================
    # STEP 2: OVERRIDE with golden baseline lags (DO NOT USE calculated lags)
    # =========================================================================
    echo "  [2/3] Applying golden lags to $basename..." >> "$LOG_FILE"
    
    # Copy golden lags, overwriting the calculated lags
    cp "$golden_lags" "$result_dir/preprocessing/${basename}_optimal_lags.csv"
    
    if [ $? -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ FAILED copying golden lags: $basename" >> "$LOG_FILE"
        return 1
    fi
    
    # =========================================================================
    # STEP 3: Run DynoTEARS with FIXED golden parameters
    # =========================================================================
    echo "  [3/3] Running DynoTEARS on $basename (λw=$lambda_w, λa=$lambda_a)..." >> "$LOG_FILE"
    
    INPUT_DIFFERENCED_CSV="$result_dir/preprocessing/${basename}_differenced_stationary_series.csv" \
    INPUT_LAGS_CSV="$result_dir/preprocessing/${basename}_optimal_lags.csv" \
    RESULT_DIR="$result_dir" \
    FIXED_LAMBDA_W="$lambda_w" \
    FIXED_LAMBDA_A="$lambda_a" \
    python3 executable/final_pipeline/dbn_dynotears_fixed_lambda.py >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ COMPLETED: $basename" >> "$LOG_FILE"
        return 0
    else
        echo "[$(date +%H:%M:%S)] ✗ FAILED DynoTEARS: $basename" >> "$LOG_FILE"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f process_anomaly
export RESULTS_BASE
export LOG_FILE
export GOLDEN_LAGS
export LAMBDA_W
export LAMBDA_A

# =============================================================================
# Process datasets in batches of 3
# =============================================================================
batch_num=0
completed=0
failed=0

for ((i=0; i<$TOTAL; i+=3)); do
    batch_num=$((batch_num + 1))
    echo "=============================================================================" | tee -a "$LOG_FILE"
    echo "Batch $batch_num of $((($TOTAL + 2) / 3)) ($(date +%H:%M:%S))" | tee -a "$LOG_FILE"
    echo "=============================================================================" | tee -a "$LOG_FILE"
    
    # Launch up to 3 jobs in parallel
    pids=()
    for ((j=0; j<3 && i+j<$TOTAL; j++)); do
        idx=$((i + j))
        file="${ANOMALY_FILES[$idx]}"
        basename=$(basename "$file" .csv)
        echo "  [$((idx+1))/$TOTAL] Launching: $basename" | tee -a "$LOG_FILE"
        
        process_anomaly "$file" "$LAMBDA_W" "$LAMBDA_A" "$GOLDEN_LAGS" &
        pids+=($!)
    done
    
    # Wait for all jobs in this batch
    for pid in "${pids[@]}"; do
        wait $pid
        if [ $? -eq 0 ]; then
            ((completed++))
        else
            ((failed++))
        fi
    done
    
    echo "  Batch $batch_num complete! (Completed: $completed, Failed: $failed)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

# =============================================================================
# Final summary
# =============================================================================
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "PROCESSING COMPLETE" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"
echo "Total datasets: $TOTAL" | tee -a "$LOG_FILE"
echo "Successfully completed: $completed" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_BASE/" | tee -a "$LOG_FILE"
echo "=============================================================================" | tee -a "$LOG_FILE"

if [ $failed -gt 0 ]; then
    echo "⚠️  Some datasets failed. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
else
    echo "✅ All datasets processed successfully!" | tee -a "$LOG_FILE"
    exit 0
fi
