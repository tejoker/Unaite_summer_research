#!/bin/bash
# verify_cleanup.sh - Verify which files will be archived
# This script shows you exactly what will be moved WITHOUT moving anything

echo "╔════════════════════════════════════════════════════╗"
echo "║  Repository Cleanup - Verification Report         ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Count files
count_experimental=0
count_pipeline=0
count_tests=0
count_exec_tests=0
count_suite=0

echo "🔍 PHASE 1: Experimental Detectors (8 files)"
echo "─────────────────────────────────────────────"
for file in \
    "causal_cascade_detector.py" \
    "directional_asymmetry_detector.py" \
    "rate_of_change_detector.py" \
    "early_single_sensor_detector.py" \
    "temporal_precedence_detector.py" \
    "multi_window_voting_detector.py" \
    "ensemble_detector.py" \
    "sub_window_detector.py"; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $file"
        ((count_experimental++))
    else
        echo "  ✗ $file (not found)"
    fi
done

if [[ -f "executable/final_pipeline/anomaly_detection/robust_weight_detector.py" ]]; then
    echo "  ✓ executable/final_pipeline/anomaly_detection/robust_weight_detector.py"
    ((count_experimental++))
fi

if [[ -f "executable/final_pipeline/anomaly_detection/unified_anomaly_detector.py" ]]; then
    echo "  ✓ executable/final_pipeline/anomaly_detection/unified_anomaly_detector.py"
    ((count_experimental++))
fi

echo ""
echo "🔍 PHASE 2: Old Pipeline Variants (5 files)"
echo "─────────────────────────────────────────────"
for file in \
    "executable/final_pipeline/dbn_dynotears_no_rolling.py" \
    "executable/final_pipeline/dynotears_variants.py" \
    "executable/final_pipeline/reconstruction.py" \
    "executable/final_pipeline/weight_corrector.py" \
    "executable/test/dynotears_variants.py"; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $file"
        ((count_pipeline++))
    else
        echo "  ✗ $file (not found)"
    fi
done

echo ""
echo "🔍 PHASE 3: Test Scripts (10 files)"
echo "─────────────────────────────────────────────"
for file in \
    "test_multiple_anomalies.py" \
    "create_multi_anomaly_dataset.py" \
    "create_multi_anomaly_dataset_v2.py" \
    "analyze_variable_location_results.py" \
    "analyze_variable_location_existing_results.py" \
    "tests/test_unified_detector.py" \
    "tests/test_robust_detector.py" \
    "tests/test_reproducibility.py" \
    "tests/create_test_anomalies.py" \
    "tests/detect_gradual_anomalies.py"; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $file"
        ((count_tests++))
    else
        echo "  ✗ $file (not found)"
    fi
done

echo ""
echo "🔍 PHASE 4: Executable Test Files (11 files)"
echo "─────────────────────────────────────────────"
for file in \
    "executable/test/test.py" \
    "executable/test/anomaly_test.py" \
    "executable/test/chunking.py" \
    "executable/test/tendance.py" \
    "executable/test/fake_data.py" \
    "executable/test/kl_divergence_kde.py" \
    "executable/test/graph_structure_detector.py" \
    "executable/test/analyze_launcher_simple.py" \
    "executable/test/analyze_launcher_structures.py" \
    "executable/test/frobenius_test.py" \
    "executable/test/sota_anomaly_methods.py"; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $file"
        ((count_exec_tests++))
    else
        echo "  ✗ $file (not found)"
    fi
done

echo ""
echo "🔍 PHASE 5: Suite Test Files (4 files)"
echo "─────────────────────────────────────────────"
for file in \
    "executable/test/anomaly_detection_suite/simple_test.py" \
    "executable/test/anomaly_detection_suite/test_suite.py" \
    "executable/test/anomaly_detection_suite/example_usage.py" \
    "executable/test/anomaly_detection_suite/frobenius_test.py"; do
    if [[ -f "$file" ]]; then
        echo "  ✓ $file"
        ((count_suite++))
    else
        echo "  ✗ $file (not found)"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "SUMMARY"
echo "═══════════════════════════════════════════════════════"
echo "  Experimental Detectors:   $count_experimental/10 files found"
echo "  Pipeline Variants:        $count_pipeline/5 files found"
echo "  Test Scripts:             $count_tests/10 files found"
echo "  Executable Tests:         $count_exec_tests/11 files found"
echo "  Suite Tests:              $count_suite/4 files found"
echo "─────────────────────────────────────────────────────"

total=$((count_experimental + count_pipeline + count_tests + count_exec_tests + count_suite))
echo "  TOTAL FILES TO ARCHIVE:   $total/40 files"
echo "═══════════════════════════════════════════════════════"
echo ""

if [[ $total -gt 0 ]]; then
    echo "✅ Ready to run cleanup!"
    echo ""
    echo "Next steps:"
    echo "  1. Review CLEANUP_REPORT.md for full details"
    echo "  2. Run: ./cleanup_repo.sh --dry-run  (to preview)"
    echo "  3. Run: ./cleanup_repo.sh             (to execute)"
else
    echo "⚠️  No files found to archive (already cleaned?)"
fi
echo ""
