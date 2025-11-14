#!/bin/bash
# Test Multiple Anomaly Types in Parallel
# This script creates and tests different anomaly types at various locations

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Configuration
GOLDEN_CSV="$WORKSPACE_ROOT/data/Golden/chunking/output_of_the_1th_chunk.csv"
TEST_DIR="$WORKSPACE_ROOT/data/Test"
RESULTS_BASE="$WORKSPACE_ROOT/results/multi_anomaly_test"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_BASE}_${TIMESTAMP}"

mkdir -p "$TEST_DIR"
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                    MULTI-ANOMALY TYPE TEST SUITE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Testing anomaly types: SPIKE, DRIFT, LEVEL_SHIFT, AMPLITUDE_CHANGE"
echo "At locations: Row 200, 350, 500"
echo "Results: $RESULTS_DIR"
echo ""

# Array of anomaly types and configurations
declare -a ANOMALY_CONFIGS=(
    "spike:200:50.0:add:Spike at row 200"
    "spike:350:50.0:add:Spike at row 350"
    "spike:500:50.0:add:Spike at row 500"
    "drift:200:30:0.3:Drift starting at row 200"
    "drift:350:50:0.5:Drift starting at row 350"
    "drift:500:40:0.4:Drift starting at row 500"
    "level_shift:200:20.0:add:Level shift at row 200"
    "level_shift:350:25.0:add:Level shift at row 350"
    "level_shift:500:30.0:add:Level shift at row 500"
    "amplitude_change:200:2.0:multiply:Amplitude change at row 200"
    "amplitude_change:350:1.5:multiply:Amplitude change at row 350"
    "amplitude_change:500:2.5:multiply:Amplitude change at row 500"
)

# Step 1: Create all test anomalies
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 1: Creating all test anomalies..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

cd "$WORKSPACE_ROOT"

for config in "${ANOMALY_CONFIGS[@]}"; do
    IFS=':' read -r anomaly_type row param1 param2 description <<< "$config"
    
    output_name="${anomaly_type}_row${row}"
    output_csv="$TEST_DIR/${output_name}.csv"
    output_json="$TEST_DIR/${output_name}.json"
    
    echo "Creating: $description"
    
    if [ "$anomaly_type" = "drift" ]; then
        # Drift: param1=length, param2=rate
        python3 "$WORKSPACE_ROOT/tests/create_test_anomalies.py" \
            --input "$GOLDEN_CSV" \
            --output "$output_csv" \
            --ts-col "Temperatur Exzenterlager links" \
            --anomaly "$anomaly_type" \
            --start "$row" \
            --length "$param1" \
            --a "$param2" \
            --mode "add"
    elif [ "$anomaly_type" = "amplitude_change" ]; then
        # Amplitude change: param1=factor, param2=mode
        python3 "$WORKSPACE_ROOT/tests/create_test_anomalies.py" \
            --input "$GOLDEN_CSV" \
            --output "$output_csv" \
            --ts-col "Temperatur Exzenterlager links" \
            --anomaly "$anomaly_type" \
            --start "$row" \
            --factor "$param1" \
            --mode "$param2"
    else
        # Spike or level_shift: param1=magnitude, param2=mode
        python3 "$WORKSPACE_ROOT/tests/create_test_anomalies.py" \
            --input "$GOLDEN_CSV" \
            --output "$output_csv" \
            --ts-col "Temperatur Exzenterlager links" \
            --anomaly "$anomaly_type" \
            --start "$row" \
            --magnitude "$param1" \
            --mode "$param2"
    fi
    
    echo "  ✅ Created: $output_csv"
done

echo ""
echo "✅ All test anomalies created!"
echo ""

# Step 2: Calibrate on Golden data (once)
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 2: Calibrating on Golden data..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

mkdir -p "$RESULTS_DIR/golden"

# Preprocessing
echo "🔧 Running preprocessing..."
INPUT_CSV_FILE="$GOLDEN_CSV" \
RESULT_DIR="$RESULTS_DIR/golden" \
python3 "$WORKSPACE_ROOT/executable/final_pipeline/preprocessing_no_mi.py"

echo "✅ Preprocessing complete"
echo ""

# Calibration
echo "🔧 Finding optimal lambda values..."
GOLDEN_BASENAME=$(basename "$GOLDEN_CSV" .csv)
INPUT_DIFFERENCED_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_differenced_stationary_series.csv" \
RESULT_DIR="$RESULTS_DIR/golden" \
INPUT_LAGS_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
CALIBRATE_LAMBDAS="true" \
python3 "$WORKSPACE_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"

# Extract lambda values
LAMBDA_W=$(python3 "$WORKSPACE_ROOT/utils/read_json_field.py" "$RESULTS_DIR/golden/best_lambdas.json" "lambda_w")
LAMBDA_A=$(python3 "$WORKSPACE_ROOT/utils/read_json_field.py" "$RESULTS_DIR/golden/best_lambdas.json" "lambda_a")

echo "✅ Calibrated: lambda_w = $LAMBDA_W, lambda_a = $LAMBDA_A"
echo ""

# Generate golden weights
echo "🔧 Generating golden weights..."
rm -f "$RESULTS_DIR/golden/history/rolling_checkpoint.pkl"
INPUT_DIFFERENCED_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_differenced_stationary_series.csv" \
RESULT_DIR="$RESULTS_DIR/golden" \
INPUT_LAGS_CSV="$RESULTS_DIR/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
FIXED_LAMBDA_W="$LAMBDA_W" \
FIXED_LAMBDA_A="$LAMBDA_A" \
python3 "$WORKSPACE_ROOT/executable/final_pipeline/dbn_dynotears_fixed_lambda.py"

echo "✅ Golden weights generated"
echo ""

# Step 3: Process anomalies in parallel
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 3: Processing anomalies in PARALLEL..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

# Function to process one anomaly
process_anomaly() {
    local anomaly_csv=$1
    local anomaly_name=$2
    local lambda_w=$3
    local lambda_a=$4
    local results_dir=$5
    local workspace=$6
    
    local anomaly_dir="$results_dir/$anomaly_name"
    mkdir -p "$anomaly_dir"
    
    local basename=$(basename "$anomaly_csv" .csv)
    
    # Preprocessing
    INPUT_CSV_FILE="$anomaly_csv" \
    RESULT_DIR="$anomaly_dir" \
    INPUT_LAGS_CSV="$results_dir/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
    python3 "$workspace/executable/final_pipeline/preprocessing_no_mi.py" > "$anomaly_dir/preprocessing.log" 2>&1
    
    # Causal Discovery
    INPUT_DIFFERENCED_CSV="$anomaly_dir/${basename}_differenced_stationary_series.csv" \
    RESULT_DIR="$anomaly_dir" \
    INPUT_LAGS_CSV="$results_dir/golden/${GOLDEN_BASENAME}_optimal_lags.csv" \
    FIXED_LAMBDA_W="$lambda_w" \
    FIXED_LAMBDA_A="$lambda_a" \
    python3 "$workspace/executable/final_pipeline/dbn_dynotears_fixed_lambda.py" > "$anomaly_dir/causal_discovery.log" 2>&1
    
    echo "✅ Completed: $anomaly_name"
}

export -f process_anomaly
export GOLDEN_BASENAME
export WORKSPACE_ROOT
export RESULTS_DIR
export LAMBDA_W
export LAMBDA_A

# Process all anomalies in parallel (4 at a time to avoid overload)
MAX_PARALLEL=2  # Reduced from 4 to prevent system overload

for config in "${ANOMALY_CONFIGS[@]}"; do
    IFS=':' read -r anomaly_type row param1 param2 description <<< "$config"
    
    output_name="${anomaly_type}_row${row}"
    anomaly_csv="$TEST_DIR/${output_name}.csv"
    
    # Run in background
    (
        echo "🔄 Processing: $description"
        process_anomaly "$anomaly_csv" "$output_name" "$LAMBDA_W" "$LAMBDA_A" "$RESULTS_DIR" "$WORKSPACE_ROOT"
    ) &
    
    # Limit to MAX_PARALLEL processes
    if [[ $(jobs -r -p | wc -l) -ge $MAX_PARALLEL ]]; then
        wait -n
    fi
done

# Wait for all remaining jobs
wait

echo ""
echo "✅ All anomalies processed!"
echo ""

# Step 4: Generate summary report
echo "─────────────────────────────────────────────────────────────────────────────"
echo "STEP 4: Generating summary report..."
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""

python3 "$WORKSPACE_ROOT/analyze_variable_location_results.py" "$RESULTS_DIR" > "$RESULTS_DIR/ANALYSIS_REPORT.txt"

cat "$RESULTS_DIR/ANALYSIS_REPORT.txt"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "                          ALL TESTS COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Analysis report: $RESULTS_DIR/ANALYSIS_REPORT.txt"
echo ""
