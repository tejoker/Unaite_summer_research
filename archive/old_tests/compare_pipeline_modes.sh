#!/bin/bash
# Complete comparison testing script for pipeline modes
# Tests: Full pipeline vs No MI masking vs dynotears_variants

set -e  # Exit on error

# Configuration
EXECUTABLE_DIR="/home/nicolasbigeard/program_internship_paul_wurth/executable"
RESULTS_BASE="/home/nicolasbigeard/program_internship_paul_wurth/results/pipeline_comparison_$(date +%Y%m%d_%H%M%S)"

# Test data files
DATA_FILES=(
    "../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__amplitude_change.csv"
    "../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__drift.csv"
    "../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__level_shift.csv"
)

echo "=========================================="
echo "COMPREHENSIVE PIPELINE COMPARISON TEST"
echo "Results will be stored in: $RESULTS_BASE"
echo "=========================================="

mkdir -p "$RESULTS_BASE"
cd "$EXECUTABLE_DIR"

for data_file in "${DATA_FILES[@]}"; do
    if [ ! -f "$data_file" ]; then
        echo "Warning: File not found: $data_file"
        continue
    fi

    basename_file=$(basename "$data_file" .csv)
    echo ""
    echo "================================================"
    echo "Testing file: $basename_file"
    echo "================================================"

    # Test 1: Full Pipeline (with MI masking + rolling windows)
    echo "1. Running FULL pipeline (MI + rolling windows)..."
    full_result_dir="$RESULTS_BASE/${basename_file}/01_full_pipeline"
    mkdir -p "$full_result_dir"

    RESULT_DIR="$full_result_dir" python launcher.py --folder final_pipeline --csv-file "$data_file" > "$full_result_dir/launcher.log" 2>&1 || echo "Full pipeline failed"

    # Test 2: Pipeline without MI masking
    echo "2. Running pipeline WITHOUT MI masking..."
    no_mi_result_dir="$RESULTS_BASE/${basename_file}/02_no_mi_masking"
    mkdir -p "$no_mi_result_dir"

    # Run preprocessing without MI
    cd final_pipeline
    RESULT_DIR="$no_mi_result_dir" INPUT_CSV_FILE="$data_file" python preprocessing_no_mi.py > "$no_mi_result_dir/preprocessing.log" 2>&1 || echo "No-MI preprocessing failed"

    # Run DBN with the no-MI preprocessing results
    if [ -f "$no_mi_result_dir/preprocessing/${basename_file}_differenced_stationary_series.csv" ]; then
        RESULT_DIR="$no_mi_result_dir" \
        INPUT_DIFFERENCED_CSV="$no_mi_result_dir/preprocessing/${basename_file}_differenced_stationary_series.csv" \
        INPUT_LAGS_CSV="$no_mi_result_dir/preprocessing/${basename_file}_optimal_lags.csv" \
        INPUT_MI_MASK_CSV="$no_mi_result_dir/preprocessing/${basename_file}_mi_mask_edges.csv" \
        python dbn_dynotears.py > "$no_mi_result_dir/dbn.log" 2>&1 || echo "No-MI DBN failed"
    fi
    cd ..

    # Test 3: dynotears_variants (controlled comparison)
    echo "3. Running dynotears_variants comparison..."
    variants_result_dir="$RESULTS_BASE/${basename_file}/03_variants"
    mkdir -p "$variants_result_dir"

    cd final_pipeline
    python dynotears_variants.py --data "$data_file" --output-dir "$variants_result_dir" --variants no_mi no_rolling > "$variants_result_dir/variants.log" 2>&1 || echo "Variants test failed"
    cd ..

    # Test 4: Benchmark comparison
    echo "4. Running benchmark comparison..."
    benchmark_result_dir="$RESULTS_BASE/${basename_file}/04_benchmark"
    mkdir -p "$benchmark_result_dir"

    cd final_pipeline
    python dynotears_benchmark.py --data "$data_file" --output-dir "$benchmark_result_dir" > "$benchmark_result_dir/benchmark.log" 2>&1 || echo "Benchmark failed"
    cd ..

    echo "Completed testing: $basename_file"
done

# Create summary report
echo ""
echo "=========================================="
echo "CREATING SUMMARY REPORT"
echo "=========================================="

cat > "$RESULTS_BASE/README.md" << EOF
# Pipeline Comparison Test Results

Generated: $(date)

## Test Overview

This directory contains comprehensive comparison results for the DynoTEARS pipeline with different configurations:

### Test Configurations:

1. **01_full_pipeline**: Complete pipeline with MI masking + rolling windows
   - Uses standard launcher.py with final_pipeline folder
   - Includes preprocessing → DBN → reconstruction
   - Full MI masking and rolling window analysis

2. **02_no_mi_masking**: Pipeline without MI constraints
   - Uses preprocessing_no_mi.py (generates empty MI mask)
   - Same rolling window analysis
   - Allows ALL causal edges (no MI filtering)

3. **03_variants**: dynotears_variants.py controlled comparison
   - Tests specific algorithm variants:
     - no_mi: VAR without MI mask
     - no_rolling: Single global analysis (no rolling windows)

4. **04_benchmark**: dynotears_benchmark.py comprehensive comparison
   - Compares multiple methods side-by-side
   - Includes enhanced vs baseline methods

## Data Files Tested:

$(for file in "${DATA_FILES[@]}"; do echo "- $(basename "$file")"; done)

## Output Structure:

Each test creates subdirectories:
- \`weights/\`: Weight matrices and results
- \`preprocessing/\`: Preprocessed data
- \`history/\`: Optimization history and checkpoints
- Logs: \`*.log\` files for debugging

## Analysis:

Compare the following metrics across configurations:
- Number of detected edges (W matrix + A matrix)
- Weight distributions and sparsity
- Computational time and convergence
- Reconstruction quality (if available)

## Key Files to Compare:

- \`weights_*.csv\`: Final weight matrices
- \`*_results.json\`: Summary statistics
- \`*.log\`: Execution logs and performance metrics
EOF

echo "Summary report created: $RESULTS_BASE/README.md"
echo ""
echo "=========================================="
echo "PIPELINE COMPARISON TEST COMPLETE!"
echo "Results location: $RESULTS_BASE"
echo "=========================================="
echo ""
echo "To analyze results, check:"
echo "  - Weight matrices in weights/ subdirectories"
echo "  - JSON summary files for edge counts"
echo "  - Log files for performance metrics"
echo "  - README.md for detailed explanation"