#!/usr/bin/env python3
"""
Generate Bagging Runs for Tucker-CAM Anomaly Detection.

This script executes the Tucker-CAM pipeline N times with different random initializations
to generate an ensemble of causal graphs (bagging).

Optimizations:
- Runs preprocessing only ONCE (in the first run).
- Subsequent runs reuse the preprocessed data via symlinks/copying to save time.
"""

import os
import sys
import argparse
import shutil
import subprocess
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate bagging runs for Tucker-CAM')
    parser.add_argument('--data', required=True, help='Input CSV file')
    parser.add_argument('--output-dir', required=True, help='Output directory for the ensemble')
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs')
    parser.add_argument('--base-seed', type=int, default=42, help='Base random seed (not strictly enforced for subprocesses but useful for logging)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers per run')
    
    args = parser.parse_args()
    
    data_file = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting {args.n_runs} bagging runs...")
    logger.info(f"Input: {data_file}")
    logger.info(f"Output: {output_dir}")
    
    # Path to launcher
    workspace_root = Path(__file__).resolve().parent.parent.parent
    launcher_script = workspace_root / "executable" / "launcher.py"
    
    if not launcher_script.exists():
        logger.error(f"Launcher script not found: {launcher_script}")
        sys.exit(1)

    start_total = time.time()
    
    # Store path to the first run's preprocessing directory
    first_run_preprocessing = None
    
    for i in range(args.n_runs):
        run_name = f"run_{i:03d}"
        run_dir = output_dir / run_name
        
        # RESUME CAPABILITY: Check if run already completed
        expected_weights = run_dir / 'weights' / 'weights_enhanced.csv'
        if expected_weights.exists():
            logger.info(f"RUN {i+1}/{args.n_runs} ({run_name}) SKIPPED (Already exists)")
            
            # Still need to capture preprocessing path for future runs if it's run 0
            if i == 0:
                first_run_preprocessing = run_dir / 'preprocessing'
                if not first_run_preprocessing.exists():
                     logger.warning("  Preprocessing directory missing for Run 0 (strange for completed run).")
                     first_run_preprocessing = None
            continue

        logger.info(f"="*60)
        logger.info(f"RUN {i+1}/{args.n_runs} ({run_name})")
        logger.info(f"="*60)
        
        # Prepare command
        cmd = [sys.executable, str(launcher_script), 
               '--data', str(data_file), 
               '--output', str(run_dir),
               '--no-resume'] # Force fresh run logic for this specific run instance
        
        # Pass worker count if specified
        env = os.environ.copy()
        if args.workers:
            env['N_WORKERS'] = str(args.workers)
        
        # Optimization: Reuse preprocessing from run 0
        if i > 0 and first_run_preprocessing and first_run_preprocessing.exists():
            logger.info("  Reusing preprocessing from run_000...")
            
            # Create run directory
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Symlink preprocessing directory
            dest_preprocessing = run_dir / 'preprocessing'
            
            # Remove empty directory created by mkdir parent or previous failed attempts
            if dest_preprocessing.exists() and not dest_preprocessing.is_symlink():
                 try:
                     # Only remove if empty or force remove? Better safe: shutil.rmtree
                     shutil.rmtree(dest_preprocessing)
                 except Exception as e:
                     logger.warning(f"Could not remove existing preprocessing dir: {e}")
            
            if not dest_preprocessing.exists():
                try:
                    os.symlink(first_run_preprocessing, dest_preprocessing)
                    logger.info(f"  Symlinked preprocessing from {first_run_preprocessing}")
                except OSError:
                    # logger.warning("  Symlink failed, falling back to copy tree...")
                    shutil.copytree(first_run_preprocessing, dest_preprocessing)
                
            # Tell launcher to skip preprocessing
            cmd.extend(['--skip-steps', 'preprocessing'])
            
        elif (run_dir / 'preprocessing').exists():
            # Check if existing preprocessing in this dir is valid (from interrupted run)
            # We look for the main .npy file
            # Assuming standard naming convention from launcher: basename + ...
            # We can't easily guess basename here strictly without parsing data path, 
            # but we can just assume if the dir exists and has .npy files it's likely good or launcher will fail fast.
            # actually launcher checking is robust.
            
            # More robust check:
            if any((run_dir / 'preprocessing').glob("*.npy")):
                 logger.info("  Found existing preprocessing artifacts. Skipping preprocessing step.")
                 cmd.extend(['--skip-steps', 'preprocessing'])
        
        # Execute run
        try:
            start_run = time.time()
            # STREAM OUTPUT IN REAL-TIME
            # We remove capture_output=True so logs flow to stdout/stderr immediately
            result = subprocess.run(cmd, env=env, check=True, text=True) 
            elapsed = time.time() - start_run
            logger.info(f"  Run {i+1} completed in {elapsed:.1f}s")
            
            # Since we stream, we can't print captured stdout again.
            # But the user sees it live, which is better.
            
            # If this was the first run, store the preprocessing path
            if i == 0:
                first_run_preprocessing = run_dir / 'preprocessing'
                if not first_run_preprocessing.exists():
                    logger.warning("  Preprocessing directory not found after first run! Optimization disabled.")
                    first_run_preprocessing = None
                    
        except subprocess.CalledProcessError as e:
            logger.error(f"  Run {i+1} FAILED!")
            logger.error(e.stderr)
            # Decide: continue or abort? For bagging, we might tolerate some failures
            logger.warning("  Continuing with next run...")
            
    total_elapsed = time.time() - start_total
    logger.info(f"="*60)
    logger.info(f"BAGGING COMPLETE")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Average time per run: {total_elapsed/args.n_runs:.1f}s")
    logger.info(f"Results stored in: {output_dir}")
    
    # Verify yield
    weights_files = list(output_dir.glob("**/weights_enhanced.csv"))
    logger.info(f"Verification: Found {len(weights_files)} weight files out of {args.n_runs} expected.")
    
    if len(weights_files) == 0:
        sys.exit(1)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
