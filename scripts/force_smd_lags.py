import numpy as np
import os
from pathlib import Path

# Paths
base_dir = Path("results/SMD_machine-1-6_golden_baseline/preprocessing")
columns_file = base_dir / "machine-1-6_columns.npy"
output_file = base_dir / "machine-1-6_optimal_lags.npy"

if not columns_file.exists():
    print(f"Error: Columns file not found: {columns_file}")
    exit(1)

# Load columns
cols = np.load(columns_file, allow_pickle=True)
print(f"Loaded {len(cols)} columns")

# Create default lags (lag=5 is standard for SMD)
default_lag = 5
lags_list = [(col, default_lag) for col in cols]
dtype = [('variable', 'U100'), ('optimal_lag', 'i4')]
lags_array = np.array(lags_list, dtype=dtype)

# Save
np.save(output_file, lags_array)
print(f"Force-generated lags file: {output_file}")
