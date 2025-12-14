#!/bin/bash
#
# Master Script: Run All Experiments for NeurIPS Paper
#
# This script runs all experimental validation in sequence:
# 1. Hyperparameter search (find best config)
# 2. Statistical validation (k-fold CV with confidence intervals)
# 3. Ablation studies (test each component)
# 4. Theoretical analysis (identifiability, sample complexity)
# 5. Complexity benchmarks (time/memory scaling)
#
# Usage: bash run_all_experiments.sh [--quick-test]
#

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_DIR"

# Use venv Python if available
if [ -f "${HOME}/.venv/bin/python3" ]; then
    PYTHON="${HOME}/.venv/bin/python3"
else
    PYTHON="python3"
fi

# Parse arguments
QUICK_TEST=false
if [ "$1" = "--quick-test" ]; then
    QUICK_TEST=true
    echo "Running in QUICK TEST mode (reduced experiments)"
fi

echo "================================================================================"
echo "RUNNING ALL EXPERIMENTS FOR NEURIPS PAPER"
echo "================================================================================"
echo "Workspace: $WORKSPACE_DIR"
echo "Python: $PYTHON"
echo "Quick test: $QUICK_TEST"
echo ""
echo "This will run:"
echo "  1. Hyperparameter search"
echo "  2. Statistical validation (k-fold CV)"
echo "  3. Ablation studies"
echo "  4. Theoretical analysis"
echo "  5. Complexity benchmarks"
echo ""
echo "Estimated time: ~10 days (or ~6 hours in quick test mode)"
echo "================================================================================"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

START_TIME=$(date +%s)

# ============================================================================
# 1. Hyperparameter Search
# ============================================================================
echo ""
echo "[1/5] Hyperparameter Search"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ "$QUICK_TEST" = true ]; then
    $PYTHON executable/experiments/hyperparameter_search.py \
        --dataset smap \
        --quick-test \
        --max-trials 5
else
    $PYTHON executable/experiments/hyperparameter_search.py \
        --dataset smap
fi

echo "✓ Hyperparameter search complete"

# ============================================================================
# 2. Statistical Validation
# ============================================================================
echo ""
echo "[2/5] Statistical Validation (k-fold CV)"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ "$QUICK_TEST" = true ]; then
    $PYTHON executable/experiments/statistical_validation.py \
        --dataset smap \
        --k-folds 2 \
        --n-seeds 2
else
    $PYTHON executable/experiments/statistical_validation.py \
        --dataset smap \
        --k-folds 5 \
        --n-seeds 5
fi

echo "✓ Statistical validation complete"

# ============================================================================
# 3. Ablation Studies
# ============================================================================
echo ""
echo "[3/5] Ablation Studies"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ "$QUICK_TEST" = true ]; then
    # Only test a few variants in quick mode
    $PYTHON executable/experiments/ablation_studies.py \
        --dataset smap \
        --variants full linear single_metric
else
    # Test all variants
    $PYTHON executable/experiments/ablation_studies.py \
        --dataset smap
fi

echo "✓ Ablation studies complete"

# ============================================================================
# 4. Theoretical Analysis
# ============================================================================
echo ""
echo "[4/5] Theoretical Analysis"
echo "────────────────────────────────────────────────────────────────────────────────"

# 4a. Identifiability
echo "  [4a] Identifiability analysis..."
if [ "$QUICK_TEST" = true ]; then
    $PYTHON analysis/theoretical/identifiability_analysis.py \
        --n-vars 20 \
        --n-samples 500
else
    $PYTHON analysis/theoretical/identifiability_analysis.py \
        --n-vars 50 \
        --n-samples 2000 \
        --test-rank-sensitivity
fi

# 4b. Sample Complexity
echo "  [4b] Sample complexity analysis..."
if [ "$QUICK_TEST" = true ]; then
    $PYTHON analysis/theoretical/sample_complexity_analysis.py \
        --dataset smap \
        --sample-sizes 500 1000 \
        --n-seeds 2
else
    $PYTHON analysis/theoretical/sample_complexity_analysis.py \
        --dataset smap \
        --sample-sizes 500 1000 2000 4000 \
        --n-seeds 3
fi

echo "✓ Theoretical analysis complete"

# ============================================================================
# 5. Complexity Benchmarks
# ============================================================================
echo ""
echo "[5/5] Computational Complexity Benchmarks"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ "$QUICK_TEST" = true ]; then
    $PYTHON benchmarking/complexity_benchmark.py \
        --dimensions 100 500 \
        --n-samples 500
else
    $PYTHON benchmarking/complexity_benchmark.py \
        --dimensions 100 500 1000 2000 \
        --n-samples 1000
fi

echo "✓ Complexity benchmarks complete"

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "================================================================================"
echo "Total runtime: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""
echo "Results saved to:"
echo "  - results/hyperparameter_search/"
echo "  - results/statistical_validation/"
echo "  - results/ablations/"
echo "  - results/theoretical/"
echo "  - results/complexity/"
echo ""
echo "Next steps:"
echo "  1. Review results in each directory"
echo "  2. Generate paper figures: python analysis/generate_paper_figures.py"
echo "  3. Update NEURIPS_PAPER_DRAFT.md with results"
echo "================================================================================"

exit 0
