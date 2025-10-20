#!/usr/bin/env python3
"""
Generate a SECOND Golden run using '1th_chunk' naming convention
for baseline noise calculation by comparing with the original Golden run.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_golden_1th_chunk():
    """Run Golden dataset with 1th_chunk naming to create second baseline run"""

    # Paths
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "Golden" / "chunking" / "output_of_the_1th_chunk.csv"
    results_base = script_dir / "results"

    # Create output directory with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_base / f"Golden_1th_chunk_run2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preproc_dir = output_dir / "preprocessing"
    preproc_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("GENERATING SECOND GOLDEN RUN WITH '1th_chunk' NAMING")
    print("="*80)
    print(f"Input data: {data_file}")
    print(f"Output directory: {output_dir}")
    print()

    # CRITICAL: Use "output_of_the_1th_chunk" as basename to match original Golden run
    input_basename = "output_of_the_1th_chunk"

    # Step 1: Run preprocessing WITHOUT MI masking
    print("Step 1: Running preprocessing (NO MI masking)...")
    preproc_script = script_dir / "executable" / "final_pipeline" / "preprocessing_no_mi.py"

    env = os.environ.copy()
    env['INPUT_CSV_FILE'] = str(data_file)
    env['RESULT_DIR'] = str(preproc_dir)
    env['INPUT_BASENAME'] = input_basename  # This creates "output_of_the_1th_chunk_*.csv"
    env['PYTHONPATH'] = str(script_dir / "executable" / "final_pipeline")

    # New code - allows you to see real-time output
    result = subprocess.run(
        [sys.executable, str(preproc_script)],
        env=env,
        cwd=script_dir / "executable"
        # capture_output=True has been removed to stream logs
    )

    if result.returncode != 0:
        print(f"ERROR: Preprocessing failed")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False

    print(f"Preprocessing completed")

    # Check preprocessing outputs
    differenced_file = preproc_dir / f"{input_basename}_differenced_stationary_series.csv"
    lags_file = preproc_dir / f"{input_basename}_optimal_lags.csv"

    if not differenced_file.exists():
        print(f"ERROR: Differenced file not created: {differenced_file}")
        print(f"Files in preprocessing dir:")
        for f in preproc_dir.iterdir():
            print(f"  - {f.name}")
        return False

    print(f"Found differenced file: {differenced_file.name}")
    print(f"Found lags file: {lags_file.name}")
    print()

    # Step 2: Run DynoTEARS WITHOUT MI masking
    print("Step 2: Running DynoTEARS (NO MI masking)...")
    dynotears_script = script_dir / "executable" / "final_pipeline" / "dbn_dynotears.py"

    env = os.environ.copy()
    env['INPUT_DIFFERENCED_CSV'] = str(differenced_file)
    env['INPUT_LAGS_CSV'] = str(lags_file)
    env['RESULT_DIR'] = str(output_dir)
    env['EXPERIMENT_NAME'] = input_basename  # Ensures output files have "1th_chunk" in name
    env['PYTHONPATH'] = str(script_dir / "executable" / "final_pipeline")

    result = subprocess.run(
        [sys.executable, str(dynotears_script)],
        env=env,
        capture_output=True,
        text=True,
        cwd=script_dir / "executable"
    )

    if result.returncode != 0:
        print(f"ERROR: DynoTEARS failed")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return False

    print(f"DynoTEARS completed")
    print()

    # Step 3: Verify outputs
    print("Step 3: Verifying outputs...")

    # Find weights file (might be nested in weights/weights/)
    weights_files = list(output_dir.rglob("*weights_enhanced*.csv"))

    if not weights_files:
        print(f"ERROR: No weights file created in {output_dir}")
        print(f"Directory structure:")
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(str(output_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
        return False

    weights_file = weights_files[0]
    print(f"Found weights file: {weights_file}")

    # Count rows to verify
    import pandas as pd
    df = pd.read_csv(weights_file)
    print(f"Weights file contains {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    if 'child_name' in df.columns and 'parent_name' in df.columns:
        print(f"Unique window indices: {df['window_idx'].nunique()}")
        print(f"Unique edges: {df[['child_name', 'parent_name', 'lag']].drop_duplicates().shape[0]}")

    print()
    print("="*80)
    print("SECOND GOLDEN RUN COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print()
    print("This can now be compared with the original Golden run at:")
    print(f"  results/Golden/weights/weights_enhanced_*.csv")
    print()
    print("To calculate baseline noise, compare:")
    print(f"  Golden 1: results/Golden/weights/weights_enhanced_*.csv")
    print(f"  Golden 2: {weights_file}")

    return True

if __name__ == "__main__":
    success = run_golden_1th_chunk()
    sys.exit(0 if success else 1)
