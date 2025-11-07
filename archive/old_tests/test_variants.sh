#!/bin/bash
# Test script to run pipeline variants for comparison

DATA_FILES=(
    "../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__amplitude_change.csv"
    "../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__drift.csv"
    "../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__level_shift.csv"
)

cd /home/nicolasbigeard/program_internship_paul_wurth/executable/final_pipeline

echo "=========================================="
echo "Testing DYNOTEARS Variants Comparison"
echo "=========================================="

for data_file in "${DATA_FILES[@]}"; do
    if [ -f "$data_file" ]; then
        echo "Processing: $(basename $data_file)"

        # Test without MI masking
        echo "  -> Running without MI masking..."
        python dynotears_variants.py --data "$data_file" --output-dir "../../results/variant_tests/no_mi_$(date +%Y%m%d_%H%M%S)" --variants no_mi

        # Test without rolling windows
        echo "  -> Running without rolling windows..."
        python dynotears_variants.py --data "$data_file" --output-dir "../../results/variant_tests/no_rolling_$(date +%Y%m%d_%H%M%S)" --variants no_rolling

        # Test both disabled
        echo "  -> Running without MI masking and rolling windows..."
        python dynotears_variants.py --data "$data_file" --output-dir "../../results/variant_tests/no_mi_no_rolling_$(date +%Y%m%d_%H%M%S)" --variants no_mi no_rolling

        echo "  -> Completed $(basename $data_file)"
        echo ""
    else
        echo "Warning: File not found: $data_file"
    fi
done

echo "=========================================="
echo "All variant tests completed!"
echo "Results stored in: ../../results/variant_tests/"
echo "=========================================="