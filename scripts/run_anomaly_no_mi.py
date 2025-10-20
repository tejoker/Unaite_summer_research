#!/usr/bin/env python3
"""
Run a single anomaly through the pipeline without MI masking
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

if len(sys.argv) < 2:
    print("Usage: python3 run_anomaly_no_mi.py <anomaly_type>")
    print("Example: python3 run_anomaly_no_mi.py spike")
    sys.exit(1)

ANOMALY_TYPE = sys.argv[1]

# Setup
INPUT_CSV = Path(f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{ANOMALY_TYPE}.csv")

if not INPUT_CSV.exists():
    print(f"ERROR: Anomaly file not found: {INPUT_CSV}")
    sys.exit(1)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"results/Anomaly_no_mi_{ANOMALY_TYPE}_{TIMESTAMP}")
PREPROC_DIR = OUTPUT_DIR / "preprocessing"
WEIGHTS_DIR = OUTPUT_DIR / "weights"

# Create directories
PREPROC_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print(f"Running {ANOMALY_TYPE.upper()} Anomaly - NO MI Masking")
print("="*80)
print(f"\nInput: {INPUT_CSV}")
print(f"Output: {OUTPUT_DIR}\n")

# Step 1: Preprocessing
print("Step 1: Preprocessing")
print("-" * 40)

# Get basename for output files
basename = INPUT_CSV.stem
print(f"Using basename: {basename}")

os.chdir("executable/final_pipeline")

env = os.environ.copy()
env['INPUT_CSV_FILE'] = f"../../{INPUT_CSV}"
env['RESULT_DIR'] = f"../../{PREPROC_DIR}"  # Point to preprocessing subdir, not output root
env['INPUT_BASENAME'] = basename  # Required by preprocessing_no_mi.py

result = subprocess.run(
    [sys.executable, "preprocessing_no_mi.py"],
    env=env,
    capture_output=True,
    text=True
)

os.chdir("../..")

if result.returncode != 0:
    print("ERROR: Preprocessing failed")
    print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    sys.exit(1)
diff_file = PREPROC_DIR / f"{basename}_differenced_stationary_series.csv"
lags_file = PREPROC_DIR / f"{basename}_optimal_lags.csv"
mi_mask_file = PREPROC_DIR / f"{basename}_mi_mask_edges.csv"

if not all([diff_file.exists(), lags_file.exists(), mi_mask_file.exists()]):
    print(f"ERROR: Preprocessing outputs not created")
    sys.exit(1)

print(f"✓ Preprocessing complete")

# Step 2: Run DynoTEARS
print("\nStep 2: Running DynoTEARS")
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

if result.returncode != 0:
    print("ERROR: DynoTEARS failed")
    print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
    sys.exit(1)

# Check weights
weights_files = list(WEIGHTS_DIR.glob("**/weights_enhanced*.csv"))

if not weights_files:
    print(f"ERROR: No weights created")
    sys.exit(1)

weights_file = max(weights_files, key=lambda p: p.stat().st_mtime)

print(f"✓ DynoTEARS complete")
print(f"  - Weights: {weights_file.relative_to(OUTPUT_DIR)}")

print("\n" + "="*80)
print(f"SUCCESS: {ANOMALY_TYPE} anomaly processed")
print("="*80)
print(f"\nOutput: {OUTPUT_DIR}")
print(f"Weights: {weights_file}")
print("="*80)
