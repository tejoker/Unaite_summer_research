#!/bin/bash

# End-to-End Anomaly Detection and Epicenter Identification Pipeline
# This script runs the complete pipeline on all anomaly types

GOLDEN_WEIGHTS="results/master_anomaly_test_20251014_171954/Golden_Test_Run/weights/weights_enhanced.csv"
RESULTS_BASE="results/master_anomaly_test_20251014_171954"
OUTPUT_BASE="results/end_to_end_analysis"

# Check if golden weights exist
if [ ! -f "$GOLDEN_WEIGHTS" ]; then
    echo "ERROR: Golden weights not found: $GOLDEN_WEIGHTS"
    echo "Please run test_all_anomalies_master.sh first"
    exit 1
fi

echo "================================================================================"
echo "END-TO-END ANOMALY DETECTION AND EPICENTER IDENTIFICATION PIPELINE"
echo "================================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Detect anomalies (causal graph changes)"
echo "  2. Identify epicenter (root cause sensor)"
echo "  3. Trace causal cascade (propagation paths)"
echo "  4. Generate comprehensive reports"
echo ""

# Allow user to specify anomaly type or run all
if [ -n "$1" ]; then
    anomaly_types=("$1")
    echo "Running analysis for: $1"
else
    anomaly_types=("spike" "drift" "level_shift" "amplitude_change" "trend_change" "variance_burst")
    echo "Running analysis for all 6 anomaly types"
fi

echo ""

for anom_type in "${anomaly_types[@]}"; do
    echo "================================================================================"
    echo "ANALYZING: $anom_type"
    echo "================================================================================"

    ANOMALY_WEIGHTS="${RESULTS_BASE}/${anom_type}_run/weights/weights_enhanced.csv"
    GROUND_TRUTH="data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__${anom_type}.json"
    OUTPUT_DIR="${OUTPUT_BASE}/${anom_type}"

    if [ ! -f "$ANOMALY_WEIGHTS" ]; then
        echo "ERROR: Anomaly weights not found: $ANOMALY_WEIGHTS"
        echo "Skipping $anom_type..."
        echo ""
        continue
    fi

    if [ ! -f "$GROUND_TRUTH" ]; then
        echo "WARNING: Ground truth not found: $GROUND_TRUTH"
        GROUND_TRUTH=""
    fi

    # Run end-to-end pipeline
    if [ -n "$GROUND_TRUTH" ]; then
        python3 end_to_end_anomaly_pipeline.py \
            --golden-weights "$GOLDEN_WEIGHTS" \
            --anomaly-weights "$ANOMALY_WEIGHTS" \
            --ground-truth "$GROUND_TRUTH" \
            --output-dir "$OUTPUT_DIR" \
            --threshold 0.01
    else
        python3 end_to_end_anomaly_pipeline.py \
            --golden-weights "$GOLDEN_WEIGHTS" \
            --anomaly-weights "$ANOMALY_WEIGHTS" \
            --output-dir "$OUTPUT_DIR" \
            --threshold 0.01
    fi

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS: Analysis complete for $anom_type"
        echo "   Report: $OUTPUT_DIR/analysis_report.txt"
        echo "   JSON: $OUTPUT_DIR/analysis_results.json"
    else
        echo ""
        echo "❌ ERROR: Analysis failed for $anom_type"
    fi

    echo ""
done

echo "================================================================================"
echo "ALL ANALYSES COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_BASE/"
echo ""
echo "View individual reports:"
echo "  cat results/end_to_end_analysis/*/analysis_report.txt"
echo ""
echo "Generate summary:"
echo "  python3 - <<'EOF'"
echo "import json"
echo "from pathlib import Path"
echo ""
echo "anomalies = ['spike', 'drift', 'level_shift', 'amplitude_change', 'trend_change', 'variance_burst']"
echo "print('\\nEND-TO-END PIPELINE SUMMARY')"
echo "print('=' * 80)"
echo "print(f'{\"Anomaly\":<20} {\"Epicenter\":<40} {\"Correct\":<10}')"
echo "print('-' * 80)"
echo ""
echo "for anom in anomalies:"
echo "    json_path = Path(f'results/end_to_end_analysis/{anom}/analysis_results.json')"
echo "    if json_path.exists():"
echo "        with open(json_path) as f:"
echo "            data = json.load(f)"
echo "            epicenter = data.get('epicenter_identification', {}).get('epicenter', 'N/A')"
echo "            matches = data.get('epicenter_identification', {}).get('matches_gt', False)"
echo "            match_str = '✅ YES' if matches else '❌ NO'"
echo "            print(f'{anom:<20} {epicenter:<40} {match_str:<10}')"
echo "    else:"
echo "        print(f'{anom:<20} NOT RUN')"
echo "EOF"
echo ""
