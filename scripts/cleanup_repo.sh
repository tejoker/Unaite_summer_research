#!/bin/bash
# cleanup_repo.sh
#
# Cleans up specific non-essential files and folders.
# Keeps: rca_report*.txt, evaluation_metrics.txt, PDFs, stress_test_workers.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Cleaning up repository from: $REPO_ROOT"

# Files to delete
FILES=(
    "debug_out.txt"
    "full_benchmark.log"
    "super_monitor.sh"
    "requirements_old.txt"
    "analysis/explain_spike_detection.py"
    "analysis/compare_weights.py"
    "analysis/visualize_window_comparison.py"
    "analysis/generate_comprehensive_summary.py"
    "analysis/analyze_all_window_weights.py"
)

# Directories to delete
DIRS=(
    "results/bagging_experiment"
    "results/ablations"
    "results/anomaly_detection"
    "results/anomaly_signatures_test"
    "results/complexity"
    "results/golden_baseline_test"
    "results/hyperparameter_search"
    "results/reproducibility"
    "results/statistical_validation"
    "results/theoretical"
    "results/bagging_SMD_machine-1-6"
    "results/SMD_machine-1-6_golden_baseline"
    "logs"
    "benchmarking"
    "executable/test"
)

# Delete files
for file in "${FILES[@]}"; do
    target="$REPO_ROOT/$file"
    if [ -f "$target" ]; then
        echo "Removing file: $file"
        rm "$target"
    else
        echo "File not found (skipped): $file"
    fi
done

# Delete directories
for dir in "${DIRS[@]}"; do
    target="$REPO_ROOT/$dir"
    if [ -d "$target" ]; then
        echo "Removing directory: $dir"
        rm -rf "$target"
    else
        echo "Directory not found (skipped): $dir"
    fi
done

echo "Cleanup complete."
