#!/bin/bash
#
# Find Optimal Number of Workers for Tucker-CAM Parallel Processing
#
# Tests different worker counts and measures:
# - Memory usage per worker
# - Time per window
# - CPU utilization
# - Optimal N_WORKERS for fastest completion
#
# Usage: bash find_optimal_workers.sh
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

# Test configuration
# ============================================================================
# PART 1: Prepare Data (Self-Contained Preprocessing)
# ============================================================================

# 1.1 Dataset Preparation (NaN handling)
GOLDEN_CLEAN="telemanom/golden_period_dataset_clean.csv"
if [ ! -f "$GOLDEN_CLEAN" ]; then
    echo "Running NaN handling (ffill + dropna)..."
    $PYTHON telemanom/prepare_datasets.py
    if [ $? -ne 0 ]; then
        echo "✗ Dataset preparation failed!"
        exit 1
    fi
fi

# 1.2 Preprocessing (Stationarity + Lags)
GOLDEN_OUTPUT="results/golden_baseline"
GOLDEN_LOG="logs/worker_test_preprocessing.log"
mkdir -p "$GOLDEN_OUTPUT" "$(dirname "$GOLDEN_LOG")"

# Define expected outputs
DATA_FILE="${GOLDEN_OUTPUT}/preprocessing/golden_period_dataset_clean_differenced_stationary_series.npy"
COLUMNS_FILE="${GOLDEN_OUTPUT}/preprocessing/golden_period_dataset_clean_columns.npy"
LAGS_FILE="${GOLDEN_OUTPUT}/preprocessing/golden_period_dataset_clean_optimal_lags.npy"

if [ -f "$DATA_FILE" ] && [ -f "$COLUMNS_FILE" ] && [ -f "$LAGS_FILE" ]; then
    echo "Preprocessing already complete. Using existing data."
    echo "  Data: $DATA_FILE"
else
    echo "Running preprocessing for Golden Baseline..."
    echo "  Log: $GOLDEN_LOG"
    
    export INPUT_CSV_FILE="$GOLDEN_CLEAN"
    export RESULT_DIR="$GOLDEN_OUTPUT"
    
    $PYTHON executable/final_pipeline/preprocessing_no_mi.py > "$GOLDEN_LOG" 2>&1
    if [ $? -ne 0 ]; then
        echo "✗ Preprocessing FAILED! Check log: $GOLDEN_LOG"
        exit 1
    fi
    
    # Organize outputs (move to preprocessing subdir if needed, though script usually handles it)
    # The script `preprocessing_no_mi.py` creates a `preprocessing` subdir inside RESULT_DIR
    # So files should be in results/golden_baseline/preprocessing/
    
    if [ ! -f "$DATA_FILE" ]; then
        echo "✗ Preprocessing finished but output file missing: $DATA_FILE"
        exit 1
    fi
    echo "✓ Preprocessing complete."
fi

# Test configuration
OUTPUT_DIR="results/worker_test"

mkdir -p "$OUTPUT_DIR"

# Test different worker counts
# Strategy: Start conservative, increase until memory pressure or diminishing returns
WORKER_COUNTS=(1 2 4 8 12 16 20 24)

echo "Test configurations:"
for N in "${WORKER_COUNTS[@]}"; do
    THREADS=$((TOTAL_CORES / N))
    MEM_PER_WORKER=$((TOTAL_RAM_GB / N))
    echo "  N_WORKERS=$N: ${THREADS} threads/worker, ~${MEM_PER_WORKER}GB available/worker"
done
echo ""

# Results file
RESULTS_FILE="${OUTPUT_DIR}/worker_test_results.csv"
echo "n_workers,threads_per_worker,time_per_window_sec,memory_per_worker_gb,total_memory_gb,cpu_percent,status" > "$RESULTS_FILE"

for N_WORKERS in "${WORKER_COUNTS[@]}"; do
    # Calculate threads per worker (divide cores equally)
    THREADS_PER_WORKER=$((TOTAL_CORES / N_WORKERS))
    
    # Minimum 2 threads per worker
    if [ $THREADS_PER_WORKER -lt 2 ]; then
        THREADS_PER_WORKER=2
    fi
    
    echo "────────────────────────────────────────────────────────────────────────────────"
    echo "Testing N_WORKERS=$N_WORKERS (${THREADS_PER_WORKER} threads per worker)"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    # Clean previous test
    rm -rf "${OUTPUT_DIR}/test_${N_WORKERS}_workers"
    mkdir -p "${OUTPUT_DIR}/test_${N_WORKERS}_workers"
    
    # Set environment
    export USE_PARALLEL=true
    export N_WORKERS=$N_WORKERS
    export OMP_NUM_THREADS=$THREADS_PER_WORKER
    export MKL_NUM_THREADS=$THREADS_PER_WORKER
    export OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER
    export VECLIB_MAXIMUM_THREADS=$THREADS_PER_WORKER
    export NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER
    
    # Start time
    START=$(date +%s)
    
    # Run 3 windows only (indices 0, 1, 2)
    # Monitor memory usage during execution
    LOG_FILE="${OUTPUT_DIR}/test_${N_WORKERS}_workers.log"
    
    # Start memory monitor in background
    (
        while true; do
            ps aux | grep "dbn_dynotears_tucker_cam_parallel.py" | grep -v grep | awk '{sum+=$6} END {print systime(), sum/1024/1024}' >> "${OUTPUT_DIR}/test_${N_WORKERS}_workers_memory.log" 2>/dev/null || true
            sleep 1
        done
    ) &
    MONITOR_PID=$!
    
    # Run test (only 3 windows)
    timeout 600 $PYTHON executable/final_pipeline/dbn_dynotears_tucker_cam_parallel.py \
        --data "$DATA_FILE" \
        --columns "$COLUMNS_FILE" \
        --lags "$LAGS_FILE" \
        --output "${OUTPUT_DIR}/test_${N_WORKERS}_workers" \
        --window-size 100 \
        --stride 2100 \
        --workers $N_WORKERS \
        > "$LOG_FILE" 2>&1
    
    STATUS=$?
    
    # Stop memory monitor
    kill $MONITOR_PID 2>/dev/null || true
    
    # End time
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Parse results
    if [ $STATUS -eq 0 ]; then
        # Count completed windows
        COMPLETED=$(grep -c "COMPLETED in" "$LOG_FILE" || echo "0")
        
        # Average time per window
        if [ $COMPLETED -gt 0 ]; then
            TIME_PER_WINDOW=$(echo "scale=2; $ELAPSED / $COMPLETED" | bc)
        else
            TIME_PER_WINDOW="N/A"
        fi
        
        # Peak memory (from monitor log)
        if [ -f "${OUTPUT_DIR}/test_${N_WORKERS}_workers_memory.log" ]; then
            PEAK_MEMORY_GB=$(awk '{if($2>max)max=$2} END {printf "%.2f", max}' "${OUTPUT_DIR}/test_${N_WORKERS}_workers_memory.log")
        else
            PEAK_MEMORY_GB="N/A"
        fi
        
        # Estimate memory per worker
        if [ "$PEAK_MEMORY_GB" != "N/A" ]; then
            MEM_PER_WORKER=$(echo "scale=2; $PEAK_MEMORY_GB / $N_WORKERS" | bc)
        else
            MEM_PER_WORKER="N/A"
        fi
        
        # Estimate CPU usage
        CPU_PERCENT=$((N_WORKERS * THREADS_PER_WORKER * 100 / TOTAL_CORES))
        
        echo "✓ SUCCESS: ${COMPLETED} windows in ${ELAPSED}s (~${TIME_PER_WINDOW}s/window)"
        echo "  Memory: ${PEAK_MEMORY_GB}GB total, ~${MEM_PER_WORKER}GB/worker"
        echo "  CPU: ~${CPU_PERCENT}% utilization"
        
        echo "$N_WORKERS,$THREADS_PER_WORKER,$TIME_PER_WINDOW,$MEM_PER_WORKER,$PEAK_MEMORY_GB,$CPU_PERCENT,success" >> "$RESULTS_FILE"
    else
        echo "✗ FAILED: Return code $STATUS"
        echo "  Check log: $LOG_FILE"
        
        # Check for OOM or timeout
        if grep -q "MemoryError\|Killed" "$LOG_FILE"; then
            FAILURE_REASON="OOM"
        elif [ $STATUS -eq 124 ]; then
            FAILURE_REASON="timeout"
        else
            FAILURE_REASON="error"
        fi
        
        echo "$N_WORKERS,$THREADS_PER_WORKER,N/A,N/A,N/A,N/A,$FAILURE_REASON" >> "$RESULTS_FILE"
    fi
    
    echo ""
done

echo "================================================================================"
echo "TEST COMPLETE - ANALYZING RESULTS"
echo "================================================================================"
echo ""

# Display results table
echo "Results:"
cat "$RESULTS_FILE" | column -t -s ','

echo ""
echo "Recommendation:"
# Find optimal (fastest successful run)
OPTIMAL=$(tail -n +2 "$RESULTS_FILE" | grep "success" | sort -t',' -k3 -n | head -1)
if [ -n "$OPTIMAL" ]; then
    OPTIMAL_N=$(echo "$OPTIMAL" | cut -d',' -f1)
    OPTIMAL_TIME=$(echo "$OPTIMAL" | cut -d',' -f3)
    OPTIMAL_MEM=$(echo "$OPTIMAL" | cut -d',' -f5)
    
    echo "  Optimal: N_WORKERS=$OPTIMAL_N"
    echo "  Time per window: ${OPTIMAL_TIME}s"
    echo "  Memory usage: ${OPTIMAL_MEM}GB"
    echo ""
    echo "For 421 windows:"
    TOTAL_TIME=$(echo "scale=1; 421 * $OPTIMAL_TIME / 3600" | bc)
    echo "  Estimated time: ${TOTAL_TIME} hours"
    
    echo ""
    echo "To apply this configuration:"
    echo "  Edit run_tucker_cam_benchmark.sh"
    echo "  Set: export N_WORKERS=$OPTIMAL_N"
    echo "  Set: export OMP_NUM_THREADS=$(($TOTAL_CORES / $OPTIMAL_N))"
else
    echo "  No successful runs! Try with fewer workers or check for errors."
fi

echo ""
echo "Full results saved to: $RESULTS_FILE"
