#!/bin/bash

# Telemanom Parallel Processing Script
# Runs 3 datasets simultaneously

GOLDEN_BASELINE="data/Golden/golden_period_dataset_mean_channel.csv"
ANOMALY_DIR="data/Anomaly/telemanom"
RESULTS_BASE="results/telemanom"
LOG_FILE="telemanom_processing.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Telemanom Parallel Processing" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Get all anomaly files
ANOMALY_FILES=($ANOMALY_DIR/isolated_anomaly_*.csv)
TOTAL=${#ANOMALY_FILES[@]}

echo "Found $TOTAL anomaly datasets" | tee -a "$LOG_FILE"
echo "Running 3 at a time..." | tee -a "$LOG_FILE"
echo ""

# Function to run single dataset
run_dataset() {
    local csv_file=$1
    local basename=$(basename "$csv_file" .csv)
    local result_dir="$RESULTS_BASE/$basename"
    
    echo "[$(date +%H:%M:%S)] Starting: $basename" >> "$LOG_FILE"
    
    INPUT_CSV_FILE="$csv_file" \
    RESULT_DIR="$result_dir" \
    python3 executable/launcher.py >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ Completed: $basename" >> "$LOG_FILE"
    else
        echo "[$(date +%H:%M:%S)] ✗ FAILED: $basename" >> "$LOG_FILE"
    fi
}

# Process in batches of 3
batch_num=0
for ((i=0; i<$TOTAL; i+=3)); do
    batch_num=$((batch_num + 1))
    echo "=== Batch $batch_num ($(date +%H:%M:%S)) ===" | tee -a "$LOG_FILE"
    
    # Start up to 3 jobs in parallel
    for ((j=0; j<3 && i+j<$TOTAL; j++)); do
        idx=$((i + j))
        file="${ANOMALY_FILES[$idx]}"
        basename=$(basename "$file" .csv)
        echo "  [$((idx+1))/$TOTAL] Launching: $basename" | tee -a "$LOG_FILE"
        run_dataset "$file" &
    done
    
    # Wait for this batch to complete
    wait
    echo "  Batch $batch_num complete!" | tee -a "$LOG_FILE"
    echo ""
done

echo "========================================" | tee -a "$LOG_FILE"
echo "All datasets processed!" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"
echo "Results in: $RESULTS_BASE/" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
