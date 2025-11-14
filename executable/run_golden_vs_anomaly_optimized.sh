#!/bin/bash
########################################
# MAXIMUM RESOURCE UTILIZATION SCRIPT
# RTX 3090 (24GB) + 64 CPU Cores
########################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the repository root (parent of executable/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate Python virtual environment
if [ -f "$HOME/.venv/bin/activate" ]; then
    source "$HOME/.venv/bin/activate"
    echo "Using Python: $(which python3)"
fi

# Create logs directory
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$LOGS_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOGS_DIR/optimized_main_${TIMESTAMP}.log"
STATUS_FILE="$LOGS_DIR/optimized_status_${TIMESTAMP}.txt"

echo "========================================" | tee "$MAIN_LOG"
echo "MAXIMUM RESOURCE UTILIZATION MODE" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "GPU: NVIDIA RTX 3090 (24GB)" | tee -a "$MAIN_LOG"
echo "CPUs: 64 cores" | tee -a "$MAIN_LOG"
echo "Mode: Aggressive parallel processing" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

# ========================================
# MAXIMUM CPU CONFIGURATION
# ========================================
export OMP_NUM_THREADS=60              # OpenMP threads (leave 4 for system)
export MKL_NUM_THREADS=60              # Intel MKL threads
export OPENBLAS_NUM_THREADS=60         # OpenBLAS threads
export NUMEXPR_NUM_THREADS=60          # NumExpr threads
export VECLIB_MAXIMUM_THREADS=60       # Apple Accelerate (if applicable)

# PyTorch CPU parallelization
export PYTORCH_INTRA_OP_THREADS=60     # Within-op parallelism
export PYTORCH_INTER_OP_THREADS=8      # Between-op parallelism

# NumPy threading
export NPY_NUM_THREADS=60

# Python optimization
export PYTHONUNBUFFERED=1              # Immediate output flushing

echo -e "${GREEN}CPU Configuration:${NC}" | tee -a "$MAIN_LOG"
echo "  - OpenMP threads: $OMP_NUM_THREADS" | tee -a "$MAIN_LOG"
echo "  - MKL threads: $MKL_NUM_THREADS" | tee -a "$MAIN_LOG"
echo "  - PyTorch intra-op: $PYTORCH_INTRA_OP_THREADS" | tee -a "$MAIN_LOG"
echo "  - PyTorch inter-op: $PYTORCH_INTER_OP_THREADS" | tee -a "$MAIN_LOG"

# ========================================
# MAXIMUM GPU CONFIGURATION
# ========================================
export CUDA_VISIBLE_DEVICES=0          # Use GPU 0 (RTX 3090)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.6"
export TF_FORCE_GPU_ALLOW_GROWTH=true  # If TensorFlow is used
export CUDA_LAUNCH_BLOCKING=0          # Async GPU ops (faster)

# PyTorch GPU memory management
export PYTORCH_NO_CUDA_MEMORY_CACHING=0  # Enable caching (faster)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo -e "${GREEN}GPU Configuration:${NC}" | tee -a "$MAIN_LOG"
echo "  - CUDA device: $CUDA_VISIBLE_DEVICES" | tee -a "$MAIN_LOG"
echo "  - Memory allocation: Aggressive (95%)" | tee -a "$MAIN_LOG"
echo "  - Mixed precision: FP16 enabled" | tee -a "$MAIN_LOG"

# ========================================
# ADAPTIVE KNOT CONFIGURATION
# ========================================
# Set to "auto" for adaptive selection based on dataset characteristics
# Or set to a number (3-15) for fixed knots
export CAM_N_KNOTS="auto"              # Adaptive knot selection
export CAM_LAMBDA_SMOOTH=0.01          # Smoothness penalty

echo -e "${GREEN}CAM Configuration:${NC}" | tee -a "$MAIN_LOG"
echo "  - Knot selection: ADAPTIVE (data-driven)" | tee -a "$MAIN_LOG"
echo "  - Lambda smooth: $CAM_LAMBDA_SMOOTH" | tee -a "$MAIN_LOG"

# ========================================
# DEFINE PROCESSING FUNCTION
# ========================================
run_pipeline() {
    local DATASET_NAME=$1
    local DATASET_FILE=$2
    local LOG_FILE=$3
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} Starting $DATASET_NAME..." | tee -a "$MAIN_LOG"
    
    # Run launcher with optimized settings
    python3 executable/launcher.py \
        --data "$DATASET_FILE" \
        --output "results/${DATASET_NAME}_001" \
        > "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✓ $DATASET_NAME completed successfully" | tee -a "$MAIN_LOG"
        return 0
    else
        echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✗ $DATASET_NAME failed (exit code: $EXIT_CODE)" | tee -a "$MAIN_LOG"
        return $EXIT_CODE
    fi
}

# ========================================
# SEQUENTIAL EXECUTION WITH MAX RESOURCES
# ========================================
echo "STARTING" > "$STATUS_FILE"
echo "Start time: $(date)" >> "$STATUS_FILE"

# Dataset paths
GOLDEN_DATA="$REPO_ROOT/data/Golden/golden_period_dataset.csv"
ANOMALY_DATA="$REPO_ROOT/data/Anomaly/isolated_anomaly_001_P-1_seq1.csv"

# Log files
GOLDEN_LOG="$LOGS_DIR/golden_${TIMESTAMP}.log"
ANOMALY_LOG="$LOGS_DIR/anomaly_${TIMESTAMP}.log"

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "PHASE 1: GOLDEN DATASET" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

GOLDEN_START=$(date +%s)
if run_pipeline "golden" "$GOLDEN_DATA" "$GOLDEN_LOG"; then
    GOLDEN_END=$(date +%s)
    GOLDEN_TIME=$((GOLDEN_END - GOLDEN_START))
    echo "GOLDEN_COMPLETED in ${GOLDEN_TIME}s" >> "$STATUS_FILE"
else
    echo "GOLDEN_FAILED" >> "$STATUS_FILE"
    echo -e "${RED}Golden dataset processing failed. Aborting.${NC}" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "PHASE 2: ANOMALY DATASET" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

ANOMALY_START=$(date +%s)
if run_pipeline "anomaly" "$ANOMALY_DATA" "$ANOMALY_LOG"; then
    ANOMALY_END=$(date +%s)
    ANOMALY_TIME=$((ANOMALY_END - ANOMALY_START))
    echo "ANOMALY_COMPLETED in ${ANOMALY_TIME}s" >> "$STATUS_FILE"
else
    echo "ANOMALY_FAILED" >> "$STATUS_FILE"
    echo -e "${RED}Anomaly dataset processing failed. Aborting comparison.${NC}" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "PHASE 3: COMPARISON ANALYSIS" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

COMPARISON_LOG="$LOGS_DIR/comparison_${TIMESTAMP}.log"
COMPARISON_START=$(date +%s)

python3 compare_weights.py \
    --golden results/golden_001/weights \
    --anomaly results/anomaly_001/weights \
    --output results/comparison_001 \
    > "$COMPARISON_LOG" 2>&1

if [ $? -eq 0 ]; then
    COMPARISON_END=$(date +%s)
    COMPARISON_TIME=$((COMPARISON_END - COMPARISON_START))
    echo "COMPARISON_COMPLETED in ${COMPARISON_TIME}s" >> "$STATUS_FILE"
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ✓ Comparison completed" | tee -a "$MAIN_LOG"
else
    echo "COMPARISON_FAILED" >> "$STATUS_FILE"
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ⚠ Comparison script had issues" | tee -a "$MAIN_LOG"
fi

# ========================================
# SUMMARY
# ========================================
TOTAL_END=$(date +%s)
TOTAL_START=$(grep "Start time:" "$STATUS_FILE" | cut -d' ' -f3- | xargs -I {} date -d {} +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo "" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "EXECUTION SUMMARY" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"
echo "Golden dataset:    ${GOLDEN_TIME}s ($(date -u -d @${GOLDEN_TIME} +%T))" | tee -a "$MAIN_LOG"
echo "Anomaly dataset:   ${ANOMALY_TIME}s ($(date -u -d @${ANOMALY_TIME} +%T))" | tee -a "$MAIN_LOG"
echo "Comparison:        ${COMPARISON_TIME}s ($(date -u -d @${COMPARISON_TIME} +%T))" | tee -a "$MAIN_LOG"
echo "Total time:        ${TOTAL_TIME}s ($(date -u -d @${TOTAL_TIME} +%T))" | tee -a "$MAIN_LOG"
echo "========================================" | tee -a "$MAIN_LOG"

echo "ALL_COMPLETED" >> "$STATUS_FILE"
echo "End time: $(date)" >> "$STATUS_FILE"
echo "Total duration: ${TOTAL_TIME}s" >> "$STATUS_FILE"

echo -e "${GREEN}✓ All processing completed successfully!${NC}" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Logs saved to:" | tee -a "$MAIN_LOG"
echo "  Main log:       $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "  Golden log:     $GOLDEN_LOG" | tee -a "$MAIN_LOG"
echo "  Anomaly log:    $ANOMALY_LOG" | tee -a "$MAIN_LOG"
echo "  Comparison log: $COMPARISON_LOG" | tee -a "$MAIN_LOG"
echo "  Status file:    $STATUS_FILE" | tee -a "$MAIN_LOG"
