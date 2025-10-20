#!/usr/bin/env python3
"""
Simple script to run Golden data through pipeline without MI masking
This generates a second Golden run for baseline noise calculation
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Setup
INPUT_CSV = Path("data/Golden/chunking/output_of_the_1th_chunk.csv")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/Golden_no_mi_run2_{TIMESTAMP}")
PREPROC_DIR = OUTPUT_DIR / "preprocessing"
WEIGHTS_DIR = OUTPUT_DIR / "weights"

# Create directories
PREPROC_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Running Golden Data - NO MI Masking")
print("="*80)
print(f"\nInput: {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}\n")

# Step 1: Run preprocessing
print("Step 1: Preprocessing")
print("-" * 40)

os.chdir("executable/final_pipeline")

env = os.environ.copy()
env['INPUT_CSV_FILE'] = f"../../{INPUT_CSV}"
env['RESULT_DIR'] = f"../../{OUTPUT_DIR}"  # Point to base dir, not preproc subdir

result = subprocess.run(
    [sys.executable, "preprocessing_no_mi.py"],  # Use NO MI version
    env=env,
    capture_output=True,
    text=True
)

os.chdir("../..")

print("Preprocessing output:")
print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)

if result.returncode != 0:
    print("ERROR: Preprocessing failed with return code", result.returncode)
    sys.exit(1)

# Check outputs
basename = INPUT_CSV.stem
diff_file = PREPROC_DIR / f"{basename}_differenced_stationary_series.csv"
lags_file = PREPROC_DIR / f"{basename}_optimal_lags.csv"
mi_mask_file = PREPROC_DIR / f"{basename}_mi_mask_edges.csv"

if not diff_file.exists():
    print(f"ERROR: Differenced file not created: {diff_file}")
    sys.exit(1)

if not lags_file.exists():
    print(f"ERROR: Lags file not created: {lags_file}")
    sys.exit(1)

if not mi_mask_file.exists():
    print(f"ERROR: MI mask file not created: {mi_mask_file}")
    sys.exit(1)

print(f"✓ Preprocessing complete (NO MI masking)")
print(f"  - Differenced: {diff_file.name}")
print(f"  - Lags: {lags_file.name}")
print(f"  - MI mask: {mi_mask_file.name} (all edges allowed)")

# Step 2: Run DynoTEARS
print("\nStep 3: Running DynoTEARS")
print("-" * 40)

os.chdir("executable/final_pipeline")

env = os.environ.copy()
env['INPUT_DIFFERENCED_CSV'] = f"../../{diff_file}"
env['INPUT_LAGS_CSV'] = f"../../{lags_file}"
env['INPUT_MI_MASK_CSV'] = f"../../{mi_mask_file}"
env['RESULT_DIR'] = f"../../{WEIGHTS_DIR}"

result = subprocess.run(
    [sys.executable, "dbn_dynotears.py"],
    env=env,
    capture_output=True,
    text=True
)

os.chdir("../..")

print("DynoTEARS output:")
print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

if result.returncode != 0:
    print("ERROR: DynoTEARS failed with return code", result.returncode)
    sys.exit(1)

# Check weights output - look recursively due to nested structure
weights_files = list(WEIGHTS_DIR.glob("**/weights_enhanced*.csv"))

if not weights_files:
    print(f"ERROR: No weights file created in {WEIGHTS_DIR}")
    print(f"Searched pattern: **/weights_enhanced*.csv")
    sys.exit(1)

weights_file = max(weights_files, key=lambda p: p.stat().st_mtime)

print(f"✓ DynoTEARS complete")
print(f"  - Weights: {weights_file.name}")

# Summary
print("\n" + "="*80)
print("SUCCESS: Golden run complete (NO MI masking)")
print("="*80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Weights file: {weights_file}")
print("\nThis can now be used for baseline noise calculation!")
print("="*80)
