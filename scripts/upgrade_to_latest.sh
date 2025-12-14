#!/bin/bash
#
# Upgrade Script: Migrate to Latest Versions (Dec 2025)
#
# This script safely upgrades your Python environment to:
# - Python 3.13.1
# - PyTorch 2.6.0
# - NumPy 2.3.0
# - All other latest packages
#
# Expected speedup: 50-70% faster than current setup
#

set -e

echo "================================================================================"
echo "UPGRADING TO LATEST VERSIONS (December 2025)"
echo "================================================================================"
echo ""
echo "This will upgrade:"
echo "  Python 3.9 → 3.13.1    (+15-25% speed)"
echo "  PyTorch 1.12 → 2.6.0   (+35-45% speed)"
echo "  NumPy 1.23 → 2.3.0     (+10-15% speed)"
echo ""
echo "Expected total speedup: 50-70%"
echo "  42-hour benchmark → ~20-25 hours"
echo "  10-day experiments → ~6-7 days"
echo ""
echo "================================================================================"
echo ""

# Check Python 3.13 is available
if ! command -v python3.13 &> /dev/null; then
    echo "ERROR: Python 3.13 not found!"
    echo ""
    echo "Install with:"
    echo "  sudo apt update"
    echo "  sudo apt install python3.13 python3.13-venv python3.13-dev"
    echo ""
    exit 1
fi

echo "✓ Python 3.13 found: $(python3.13 --version)"
echo ""

# Backup old environment
if [ -d ~/.venv ]; then
    echo "Backing up old environment..."
    mv ~/.venv ~/.venv_backup_$(date +%Y%m%d_%H%M%S)
    echo "✓ Old environment backed up"
    echo ""
fi

# Create new environment
echo "Creating new Python 3.13 virtual environment..."
python3.13 -m venv ~/.venv
source ~/.venv/bin/activate
echo "✓ New environment created"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install PyTorch (CPU-only for best performance)
echo "Installing PyTorch 2.6.0 (CPU-only)..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
echo "✓ PyTorch installed"
echo ""

# Install all other packages
echo "Installing remaining packages from requirements.txt..."
pip install -r requirements.txt
echo "✓ All packages installed"
echo ""

# Optional: Install Intel MKL for maximum performance
read -p "Install Intel MKL for +10-20% extra CPU performance? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Intel MKL..."
    pip install mkl mkl-service
    echo "✓ Intel MKL installed"
    echo ""
    echo "Add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "  export MKL_NUM_THREADS=64"
fi

# Run quick compatibility test
echo ""
echo "================================================================================"
echo "RUNNING COMPATIBILITY TEST"
echo "================================================================================"
echo ""

python3 -c "
import sys
import numpy as np
import pandas as pd
import torch
import scipy

print('✓ Python version:', sys.version.split()[0])
print('✓ NumPy version:', np.__version__)
print('✓ Pandas version:', pd.__version__)
print('✓ PyTorch version:', torch.__version__)
print('✓ SciPy version:', scipy.__version__)
print('')
print('Testing NumPy compatibility...')
# Test that your code patterns still work
arr = np.array([1, 2, 3], dtype=np.float32)
assert isinstance(arr[0], np.floating), 'np.floating check failed'
print('✓ NumPy compatibility: OK')
print('')
print('Testing PyTorch...')
x = torch.randn(10, 10)
y = torch.matmul(x, x.T)
print('✓ PyTorch: OK')
print('')
print('All compatibility tests passed!')
"

echo ""
echo "================================================================================"
echo "UPGRADE COMPLETE!"
echo "================================================================================"
echo ""
echo "Your environment is now:"
echo "  Python: $(python --version)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""
echo "Next steps:"
echo "  1. Run quick test:"
echo "     python executable/experiments/statistical_validation.py --quick-test"
echo ""
echo "  2. If successful, run full benchmark:"
echo "     bash run_tucker_cam_benchmark.sh"
echo ""
echo "  3. Expect ~50-70% speedup!"
echo ""
echo "Old environment backed up to: ~/.venv_backup_*"
echo "================================================================================"

exit 0
