#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime

def setup_paths():
    """Setup necessary paths for the launcher - Robust Version"""
    # Robustly find project root by looking for 'executable' directory
    current_path = Path(__file__).resolve().parent
    workspace_root = None
    
    # Climb up finding the root
    for parent in [current_path] + list(current_path.parents):
        if (parent / "executable").exists() and (parent / "run_tucker_cam_benchmark.sh").exists():
            workspace_root = parent
            break
            
    if workspace_root is None:
        # Fallback to current directory if not found (e.g. running from root)
        workspace_root = Path.cwd()
        
    script_dir = workspace_root
    executable_dir = workspace_root / "executable"
    data_dir = workspace_root / "data"
    results_dir = workspace_root / "results"

    return {
        'script_dir': script_dir,
        'executable_dir': executable_dir,
        'data_dir': data_dir,
        'results_dir': results_dir
    }

def check_preprocessing_complete(result_dir, input_basename):
    """Check if preprocessing step has been completed"""
    preprocessing_dir = result_dir / 'preprocessing'

    if not preprocessing_dir.exists():
        return False

    # Check for required output files (.npy format for 109x faster loading)
    differenced_file = preprocessing_dir / f'{input_basename}_differenced_stationary_series.npy'
    columns_file = preprocessing_dir / f'{input_basename}_columns.npy'
    lags_file = preprocessing_dir / f'{input_basename}_optimal_lags.npy'

    return differenced_file.exists() and columns_file.exists() and lags_file.exists()

def check_dynotears_complete(result_dir):
    """Check if DynoTEARS step has been completed"""
    causal_dir = result_dir / 'causal_discovery'

    if not causal_dir.exists():
        return False

    # Check for common output files (weights or summary files)
    # DynoTEARS typically produces weight matrices and/or summary results
    has_results = any(causal_dir.glob('*.csv')) or any(causal_dir.glob('*.npy')) or any(causal_dir.glob('*.pkl'))

    return has_results

def run_preprocessing(data_file, result_dir, input_basename):
    """Run preprocessing step"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Step 1: Preprocessing - STARTING")

    paths = setup_paths()
    workspace_root = paths['script_dir']
    
    # Environment variables for preprocessing - pass through all existing env vars
    env_vars = os.environ.copy()
    env_vars.update({
        'INPUT_CSV_FILE': str(data_file),
        'RESULT_DIR': str(result_dir),  # Pass base dir, preprocessing will create subdirectory
        'PYTHONPATH': str(workspace_root / "executable" / "final_pipeline")
    })

    # Run preprocessing
    preprocessing_script = workspace_root / "executable" / "final_pipeline" / "preprocessing_no_mi.py"
    preprocessing_dir = result_dir / 'preprocessing'

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]   Running: python {preprocessing_script}")
    print(f"[{timestamp}]   Input: {data_file}")
    print(f"[{timestamp}]   Output: {preprocessing_dir}")

    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, str(preprocessing_script)
        ], env=env_vars, capture_output=True, text=True, cwd=workspace_root)
        elapsed = time.time() - start_time

        # Always print subprocess output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if result.returncode != 0:
            print(f"[{timestamp}] Preprocessing FAILED with return code {result.returncode} (elapsed: {elapsed:.2f}s)")
            return False

        print(f"[{timestamp}] Preprocessing completed successfully (elapsed: {elapsed:.2f}s)")
        return True

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Error running preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_dynotears(data_file, result_dir, input_basename):
    """Run DynoTEARS analysis"""
    paths = setup_paths()
    workspace_root = paths['script_dir']

    # Check if Tucker-CAM should be used
    use_tucker = os.getenv('USE_TUCKER_CAM', 'false').lower() in ('true', '1', 'yes', 'on')
    use_parallel = os.getenv('USE_PARALLEL', 'false').lower() in ('true', '1', 'yes', 'on')
    method_name = "Tucker-CAM" if use_tucker else "DynoTEARS"
    if use_parallel and use_tucker:
        method_name = "Tucker-CAM (Parallel)"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Step 2: {method_name} Analysis - STARTING")

    # Preprocessing outputs are always in preprocessing/ subdirectory (standardized)
    preprocessing_dir = result_dir / 'preprocessing'
    
    differenced_file = preprocessing_dir / f'{input_basename}_differenced_stationary_series.npy'
    columns_file = preprocessing_dir / f'{input_basename}_columns.npy'
    lags_file = preprocessing_dir / f'{input_basename}_optimal_lags.npy'
    
    weights_dir = result_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not differenced_file.exists():
        print(f"[{timestamp}] Error: Differenced file not found: {differenced_file}")
        # DEBUG
        print(f"DEBUG: ls of {preprocessing_dir}: {list(preprocessing_dir.glob('*'))}")
        return False
    if not columns_file.exists():
        print(f"[{timestamp}] Error: Columns file not found: {columns_file}")
        return False
    if not lags_file.exists():
        print(f"[{timestamp}] Error: Lags file not found: {lags_file}")
        return False
    
    print(f"[{timestamp}] DEBUG: All input files found at {preprocessing_dir}")

    # Environment variables for DynoTEARS/Tucker-CAM - pass through all existing env vars
    env_vars = os.environ.copy()
    env_vars.update({
        'INPUT_DIFFERENCED_CSV': str(differenced_file),
        'INPUT_LAGS_CSV': str(lags_file),
        'RESULT_DIR': str(result_dir),
        'PYTHONPATH': str(workspace_root / "executable" / "final_pipeline")
    })
    
    print(f"[{timestamp}] DEBUG: Env vars prepared", flush=True)

    display_output_dir = result_dir / 'weights' # Expected final location for display logic
    weights_dir = result_dir # Script adds /weights automatically
    pipeline_dir = workspace_root / "executable" / "final_pipeline"
    
    print(f"[{timestamp}] DEBUG: Checking pipeline config: TUCKER={use_tucker} PARALLEL={use_parallel}", flush=True)
    
    # Select script based on USE_TUCKER_CAM and USE_PARALLEL flags
    if use_tucker and use_parallel:
        print(f"[{timestamp}] DEBUG: Selecting parallel script...", flush=True)
        # Use parallel version for maximum CPU utilization (production)
        dynotears_script = pipeline_dir / "dbn_dynotears_tucker_cam_parallel.py"

    elif use_tucker:
        # Sequential Tucker-CAM not supported (archived)
        # Use parallel with N_WORKERS=1 instead for clean memory isolation
        print("WARNING: Sequential Tucker-CAM is deprecated. Using parallel mode with 1 worker.")
        dynotears_script = pipeline_dir / "dbn_dynotears_tucker_cam_parallel.py"
    else:
        # Linear SVAR baseline for comparison
        dynotears_script = pipeline_dir / "dbn_dynotears_fixed_lambda.py"



    print(f"[{timestamp}] DEBUG: Passed selection logic", flush=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]   Running: python {dynotears_script}", flush=True)
    print(f"[{timestamp}]   Differenced data: {differenced_file}")
    print(f"[{timestamp}]   Lags file: {lags_file}")
    print(f"[{timestamp}]   Output: {weights_dir}")

    # Get workspace root (parent of executable/)
    workspace_root = Path(__file__).parent.parent

    try:
        # Build command based on script type
        # Build command based on script type
        if use_tucker:
            # Parallel script is used for both parallel and sequential (deprecated) modes
            if use_parallel:
                n_workers = int(os.getenv('N_WORKERS', '2'))
            else:
                n_workers = 1

            window_size = int(os.getenv('WINDOW_SIZE', '100'))
            stride = int(os.getenv('STRIDE', '10'))
            
            cmd = [
                sys.executable, str(dynotears_script),
                '--data', str(differenced_file),
                '--columns', str(columns_file),
                '--lags', str(lags_file),
                '--output', str(weights_dir),
                '--window-size', str(window_size),
                '--stride', str(stride),
                '--workers', str(n_workers)
            ]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}]   Parallel workers: {n_workers}")
        else:
            # Sequential version uses environment variables
            cmd = [sys.executable, str(dynotears_script)]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] DEBUG: Executing command: {' '.join(cmd)}")
        print(f"[{timestamp}] DEBUG: CWD: {workspace_root}")
        print(f"[{timestamp}] DEBUG: Env N_WORKERS: {os.environ.get('N_WORKERS')}")

        start_time = time.time()
        result = subprocess.run(cmd, env=env_vars, capture_output=True, text=True, cwd=workspace_root)
        elapsed = time.time() - start_time
        
        print(f"[{timestamp}] DEBUG: Subprocess completed with return code: {result.returncode}")

        # Always print subprocess output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if result.returncode != 0:
            print(f"[{timestamp}] {method_name} FAILED with return code {result.returncode} (elapsed: {elapsed:.2f}s)")
            return False

        print(f"[{timestamp}] {method_name} analysis completed successfully (elapsed: {elapsed:.2f}s)")
        return True

    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Error running {method_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_rca_complete(result_dir):
    """Check if RCA step has been completed"""
    rca_dir = result_dir / 'rca'
    return rca_dir.exists() and any(rca_dir.glob('*.json'))

def run_rca(result_dir, input_basename):
    """Run Root Cause Analysis on identified anomalies"""
    import pandas as pd
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Step 4: Root Cause Analysis - STARTING")
    
    paths = setup_paths()
    workspace_root = paths['script_dir']
    
    # Locate inputs
    anomaly_scores_file = result_dir / "anomaly_detection.csv"
    if not anomaly_scores_file.exists():
        print(f"[{timestamp}] Error: Anomaly scores not found at {anomaly_scores_file}")
        return False
        
    golden_weights = result_dir / "weights" / "golden_baseline.csv" # Hypothetical path, need to match actual
    # Verify accurate path for golden weights based on Stage 1/2 logic in shell script
    # The shell script saves golden weights to a specific location. 
    # Let's assume the user passes the correct golden weights path via ENV or we infer it.
    # Actually, the shell script should pass the arguments. 
    # But for the launcher wrapper, we might need to be smarter.
    
    # Let's rely on finding the 'weights' directory in results
    # and looking for 'weights.csv' or similar.
    # The anomaly detection step produced 'anomaly_detection.csv'.
    
    # Create RCA output dir
    rca_dir = result_dir / "rca"
    rca_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load detected anomalies
        df = pd.read_csv(anomaly_scores_file)
        anomalies = df[df['status'] != 'NORMAL']
        
        if anomalies.empty:
            print(f"[{timestamp}] No anomalies detected. Skipping RCA.")
            return True
            
        # Sort by severity (abs_score) and take top 5
        top_anomalies = anomalies.sort_values('abs_score', ascending=False).head(5)
        
        rca_script = workspace_root / "executable" / "test" / "anomaly_detection_suite" / "root_cause_analysis.py"
        test_weights = result_dir / "weights" / "weights.csv" # Default location from DynoTEARS
        
        # We need the Golden Baseline weights. 
        # In the shell script, these are passed to anomaly detection.
        # We can try to find them or user must have them in a standard place.
        # Assuming standard 'golden_baseline/weights/weights_enhanced.csv' pattern if it exists,
        # or relying on what was used for anomaly detection.
        # FIX: We will scan the result_dir/.. to find the golden baseline or accept an env var.
        golden_weights = os.getenv('GOLDEN_WEIGHTS_FILE')
        if not golden_weights or not Path(golden_weights).exists():
             # Try to guess: ../golden_baseline/weights/weights_enhanced.csv
             candidate = result_dir.parent / "golden_baseline" / "weights" / "weights_enhanced.csv"
             if candidate.exists():
                 golden_weights = str(candidate)
             else:
                 print(f"[{timestamp}] WARNING: Could not locate Golden Weights for RCA. Skipping.")
                 return True

        for _, row in top_anomalies.iterrows():
            win_idx = int(row['window_idx'])
            out_file = rca_dir / f"rca_window_{win_idx}.json"
            
            print(f"[{timestamp}]   Running RCA for Window {win_idx} (Score: {row['abs_score']:.4f})")
            
            cmd = [
                sys.executable, str(rca_script),
                "--baseline", golden_weights,
                "--current", str(test_weights),
                "--output", str(out_file),
                "--window-idx", str(win_idx)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
        print(f"[{timestamp}] RCA completed. Results in {rca_dir}")
        return True

    except Exception as e:
        print(f"[{timestamp}] Error running RCA: {e}")
        return False

def run_pipeline(data_file, output_dir=None, resume=True, skip_steps=None):
    """Run the complete pipeline without MI masking

    Args:
        data_file: Input CSV file path
        output_dir: Output directory for results
        resume: If True, auto-detect and skip completed steps (default: True)
        skip_steps: List of steps to skip (e.g., ['preprocessing']) or None
    """

    paths = setup_paths()

    # Determine input file
    if not os.path.exists(data_file):
        print(f"Error: Data file not found: {data_file}")
        return False

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = paths['results_dir'] / f"no_mi_analysis_{timestamp}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_basename = Path(data_file).stem

    print("="*80)
    print("RUNNING PIPELINE WITHOUT MI MASKING")
    print("="*80)
    print(f"Input file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"Input basename: {input_basename}")
    print(f"Resume mode: {'ENABLED' if resume else 'DISABLED'}")
    if skip_steps:
        print(f"Explicitly skipping steps: {', '.join(skip_steps)}")
    print()

    # Initialize skip tracking
    skip_steps = skip_steps or []

    # Auto-detect completed steps if resume is enabled
    preprocessing_complete = False
    dynotears_complete = False

    if resume:
        preprocessing_complete = check_preprocessing_complete(output_dir, input_basename)
        dynotears_complete = check_dynotears_complete(output_dir)

        if preprocessing_complete:
            print("[CHECKPOINT] Preprocessing already completed - SKIPPING")
        if dynotears_complete:
            print("[CHECKPOINT] DynoTEARS already completed - SKIPPING")
        print()

    # Step 1: Preprocessing
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'preprocessing' in skip_steps:
        print(f"[{timestamp}] Step 1: Preprocessing - SKIPPED (explicit skip)")
    elif preprocessing_complete:
        print(f"[{timestamp}] Step 1: Preprocessing - SKIPPED (already completed)")
    else:
        if not run_preprocessing(data_file, output_dir, input_basename):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Pipeline FAILED at preprocessing step")
            return False

    # Step 2: DynoTEARS Analysis
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'dynotears' in skip_steps:
        print(f"[{timestamp}] Step 2: DynoTEARS Analysis - SKIPPED (explicit skip)")
    elif dynotears_complete:
        print(f"[{timestamp}] Step 2: DynoTEARS Analysis - SKIPPED (already completed)")
    else:
        if not run_dynotears(data_file, output_dir, input_basename):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Pipeline FAILED at DynoTEARS step")
            return False

    print("="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {output_dir}")

    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run pipeline without MI masking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (auto-resume enabled by default)
  python launcher.py --data data/input.csv --output results/my_run

  # Disable auto-resume (always run from scratch)
  python launcher.py --data data/input.csv --no-resume

  # Skip specific steps explicitly
  python launcher.py --data data/input.csv --skip-steps preprocessing
  python launcher.py --data data/input.csv --skip-steps preprocessing,dynotears

  # Resume a failed run in existing directory
  python launcher.py --data data/input.csv --output results/existing_run --resume

  # Using environment variables
  INPUT_CSV_FILE=data/input.csv RESULT_DIR=results/my_run python launcher.py
        """
    )
    parser.add_argument('--mode', default='pipeline', choices=['pipeline', 'rca'],
                        help='Execution mode: "pipeline" (default) or "rca" (Root Cause Analysis)')
    parser.add_argument('--data', help='Input CSV file (can also be set via INPUT_CSV_FILE env var)')
    parser.add_argument('--output', help='Output directory (optional, can also be set via RESULT_DIR env var)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                        help='Auto-detect and skip completed steps (default: enabled)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Disable auto-resume - always run all steps from scratch')
    parser.add_argument('--skip-steps', type=str, default=None,
                        help='Comma-separated list of steps to skip (e.g., "preprocessing" or "preprocessing,dynotears")')

    args = parser.parse_args()

    # Determine input data and output dir from args or environment variables
    data_file = args.data or os.getenv('INPUT_CSV_FILE')
    output_dir = args.output or os.getenv('RESULT_DIR')

    if not data_file:
        parser.error("the following arguments are required: --data (or INPUT_CSV_FILE environment variable)")
        sys.exit(1)

    if not output_dir:
        # If no output dir is specified, create a default one
        output_dir = Path("results") / f"launcher_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle RCA mode
    if args.mode == 'rca':
        input_basename = Path(data_file).stem
        success = run_rca(Path(output_dir), input_basename)
        sys.exit(0 if success else 1)

    # Parse skip_steps if provided
    skip_steps = None
    if args.skip_steps:
        skip_steps = [step.strip() for step in args.skip_steps.split(',')]

    success = run_pipeline(data_file, output_dir, resume=args.resume, skip_steps=skip_steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
