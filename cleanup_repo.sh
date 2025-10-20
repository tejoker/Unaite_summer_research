#!/bin/bash
# cleanup_repo.sh - Repository Cleanup Script
# This script archives non-mandatory Python files
# Run with: ./cleanup_repo.sh [--dry-run]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}=== DRY RUN MODE - No files will be moved ===${NC}\n"
fi

# Function to move file
move_file() {
    local src="$1"
    local dest="$2"
    
    if [[ ! -f "$src" ]]; then
        echo -e "${RED}[SKIP]${NC} File not found: $src"
        return
    fi
    
    if $DRY_RUN; then
        echo -e "${BLUE}[DRY-RUN]${NC} Would move: $src -> $dest"
    else
        mkdir -p "$(dirname "$dest")"
        mv "$src" "$dest"
        echo -e "${GREEN}[MOVED]${NC} $src -> $dest"
    fi
}

echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Repository Cleanup - Python Files Archival       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}\n"

# Phase 1: Experimental Detectors
echo -e "${YELLOW}Phase 1: Archiving Experimental Detectors${NC}"
DEST_DIR="archive/experimental_detectors"

move_file "causal_cascade_detector.py" "$DEST_DIR/causal_cascade_detector.py"
move_file "directional_asymmetry_detector.py" "$DEST_DIR/directional_asymmetry_detector.py"
move_file "rate_of_change_detector.py" "$DEST_DIR/rate_of_change_detector.py"
move_file "early_single_sensor_detector.py" "$DEST_DIR/early_single_sensor_detector.py"
move_file "temporal_precedence_detector.py" "$DEST_DIR/temporal_precedence_detector.py"
move_file "multi_window_voting_detector.py" "$DEST_DIR/multi_window_voting_detector.py"
move_file "ensemble_detector.py" "$DEST_DIR/ensemble_detector.py"
move_file "sub_window_detector.py" "$DEST_DIR/sub_window_detector.py"
move_file "executable/final_pipeline/anomaly_detection/robust_weight_detector.py" "$DEST_DIR/robust_weight_detector.py"
move_file "executable/final_pipeline/anomaly_detection/unified_anomaly_detector.py" "$DEST_DIR/unified_anomaly_detector.py"

echo ""

# Phase 2: Old Pipeline Variants
echo -e "${YELLOW}Phase 2: Archiving Old Pipeline Variants${NC}"
DEST_DIR="archive/pipeline_variants"

move_file "executable/final_pipeline/dbn_dynotears_no_rolling.py" "$DEST_DIR/dbn_dynotears_no_rolling.py"
move_file "executable/final_pipeline/dynotears_variants.py" "$DEST_DIR/dynotears_variants.py"
move_file "executable/final_pipeline/reconstruction.py" "$DEST_DIR/reconstruction.py"
move_file "executable/final_pipeline/weight_corrector.py" "$DEST_DIR/weight_corrector.py"
move_file "executable/test/dynotears_variants.py" "$DEST_DIR/dynotears_variants_test.py"

echo ""

# Phase 3: Test/Example Scripts
echo -e "${YELLOW}Phase 3: Archiving Test Scripts${NC}"
DEST_DIR="archive/test_scripts"

move_file "test_multiple_anomalies.py" "$DEST_DIR/test_multiple_anomalies.py"
move_file "create_multi_anomaly_dataset.py" "$DEST_DIR/create_multi_anomaly_dataset.py"
move_file "create_multi_anomaly_dataset_v2.py" "$DEST_DIR/create_multi_anomaly_dataset_v2.py"
move_file "analyze_variable_location_results.py" "$DEST_DIR/analyze_variable_location_results.py"
move_file "analyze_variable_location_existing_results.py" "$DEST_DIR/analyze_variable_location_existing_results.py"
move_file "tests/test_unified_detector.py" "$DEST_DIR/test_unified_detector.py"
move_file "tests/test_robust_detector.py" "$DEST_DIR/test_robust_detector.py"
move_file "tests/test_reproducibility.py" "$DEST_DIR/test_reproducibility.py"
move_file "tests/create_test_anomalies.py" "$DEST_DIR/create_test_anomalies.py"
move_file "tests/detect_gradual_anomalies.py" "$DEST_DIR/detect_gradual_anomalies.py"

echo ""

# Phase 4: Executable Test Files
echo -e "${YELLOW}Phase 4: Archiving Executable Test Files${NC}"
DEST_DIR="archive/old_test_files"

move_file "executable/test/test.py" "$DEST_DIR/test.py"
move_file "executable/test/anomaly_test.py" "$DEST_DIR/anomaly_test.py"
move_file "executable/test/chunking.py" "$DEST_DIR/chunking.py"
move_file "executable/test/tendance.py" "$DEST_DIR/tendance.py"
move_file "executable/test/fake_data.py" "$DEST_DIR/fake_data.py"
move_file "executable/test/kl_divergence_kde.py" "$DEST_DIR/kl_divergence_kde.py"
move_file "executable/test/graph_structure_detector.py" "$DEST_DIR/graph_structure_detector.py"
move_file "executable/test/analyze_launcher_simple.py" "$DEST_DIR/analyze_launcher_simple.py"
move_file "executable/test/analyze_launcher_structures.py" "$DEST_DIR/analyze_launcher_structures.py"
move_file "executable/test/frobenius_test.py" "$DEST_DIR/frobenius_test.py"
move_file "executable/test/sota_anomaly_methods.py" "$DEST_DIR/sota_anomaly_methods.py"

echo ""

# Phase 5: Anomaly Detection Suite Tests
echo -e "${YELLOW}Phase 5: Archiving Suite Test Files${NC}"
DEST_DIR="archive/suite_tests"

move_file "executable/test/anomaly_detection_suite/simple_test.py" "$DEST_DIR/simple_test.py"
move_file "executable/test/anomaly_detection_suite/test_suite.py" "$DEST_DIR/test_suite.py"
move_file "executable/test/anomaly_detection_suite/example_usage.py" "$DEST_DIR/example_usage.py"
move_file "executable/test/anomaly_detection_suite/frobenius_test.py" "$DEST_DIR/frobenius_test.py"

echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             Cleanup Complete!                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════╝${NC}\n"

if $DRY_RUN; then
    echo -e "${YELLOW}This was a DRY RUN. To actually move files, run:${NC}"
    echo -e "  ./cleanup_repo.sh"
else
    echo -e "${GREEN}✓ Files archived successfully${NC}"
    echo -e "${BLUE}Review the archived files in the archive/ directory${NC}"
fi

echo ""
echo -e "${BLUE}Total files to archive: ~42 files${NC}"
echo -e "${BLUE}Active files remaining: ~41 core files${NC}"
echo ""
