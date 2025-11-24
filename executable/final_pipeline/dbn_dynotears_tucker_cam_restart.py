#!/usr/bin/env python3
"""
Wrapper script that restarts the Tucker-CAM process every N windows to prevent memory accumulation.

This solves the NetworkX garbage collection issue by periodically restarting the Python process.
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
RESTART_INTERVAL = 1  # Restart every 1 window (11GB RAM limit, only 1 window completes per run)
SCRIPT_PATH = Path(__file__).parent / "dbn_dynotears_tucker_cam.py"

def main():
    """Run Tucker-CAM with periodic process restarts."""

    # Get environment variables
    differenced_csv = os.getenv('INPUT_DIFFERENCED_CSV')
    lags_csv = os.getenv('INPUT_LAGS_CSV')
    result_dir = os.getenv('RESULT_DIR')

    if not all([differenced_csv, lags_csv, result_dir]):
        print("ERROR: Required environment variables not set")
        print("  INPUT_DIFFERENCED_CSV, INPUT_LAGS_CSV, RESULT_DIR")
        sys.exit(1)

    result_dir = Path(result_dir)
    checkpoint_file = result_dir / "history" / "rolling_checkpoint_tucker.pkl"

    # Calculate total windows needed (use polars for fast row counting)
    import polars as pl
    df = pl.read_csv(differenced_csv)
    window_size = int(os.getenv('WINDOW_SIZE', 100))
    stride = int(os.getenv('STRIDE', 10))
    n_samples = df.height  # polars uses .height for row count
    total_windows = (n_samples - window_size) // stride + 1

    print(f"Total windows to process: {total_windows}")
    print(f"Restart interval: {RESTART_INTERVAL} windows")
    print(f"Expected restarts: {total_windows // RESTART_INTERVAL}")

    restart_count = 0
    while True:
        # Run the actual script
        print(f"\n{'='*80}")
        print(f"RESTART #{restart_count}: Launching Tucker-CAM subprocess...")
        print(f"{'='*80}\n")

        # Pass all environment variables to subprocess
        env = os.environ.copy()
        env['RESTART_INTERVAL'] = str(RESTART_INTERVAL)

        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            env=env
        )

        if result.returncode != 0:
            print(f"ERROR: Subprocess failed with return code {result.returncode}")
            sys.exit(result.returncode)

        # Check if we're done
        if not checkpoint_file.exists():
            print("No checkpoint found - analysis complete or failed")
            break

        import pickle
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            last_window = checkpoint_data.get('last_completed_window', -1)

        print(f"\nCheckpoint: Last completed window = {last_window}")

        if last_window >= total_windows - 1:
            print("All windows processed!")
            break

        # Check if we should restart
        if (last_window + 1) % RESTART_INTERVAL != 0:
            # Not yet time to restart, but subprocess exited - this is an error
            print(f"ERROR: Subprocess exited early at window {last_window}")
            sys.exit(1)

        restart_count += 1
        print(f"Restarting to clear memory (completed {last_window + 1}/{total_windows} windows)...")

    print(f"\n{'='*80}")
    print("Tucker-CAM analysis completed successfully!")
    print(f"Total restarts: {restart_count}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
