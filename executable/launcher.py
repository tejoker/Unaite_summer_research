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
    """Setup necessary paths for the launcher"""
    script_dir = Path(__file__).parent.parent
    executable_dir = script_dir / "executable"
    data_dir = script_dir / "data"
    results_dir = script_dir / "results"

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

    # Check for required output files
    differenced_file = preprocessing_dir / f'{input_basename}_differenced_stationary_series.csv'
    lags_file = preprocessing_dir / f'{input_basename}_optimal_lags.csv'

    return differenced_file.exists() and lags_file.exists()

def check_dynotears_complete(result_dir):
    """Check if DynoTEARS step has been completed"""
    causal_dir = result_dir / 'causal_discovery'

    if not causal_dir.exists():
        return False

    # Check for common output files (weights or summary files)
    # DynoTEARS typically produces weight matrices and/or summary results
    has_results = any(causal_dir.glob('*.csv')) or any(causal_dir.glob('*.pkl')) or any(causal_dir.glob('*.json'))

    return has_results

def run_preprocessing(data_file, result_dir, input_basename):
    """Run preprocessing step"""
    print(f"Step 1: Preprocessing")
    
    preprocessing_dir = result_dir / 'preprocessing'
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment variables for preprocessing - pass through all existing env vars
    env_vars = os.environ.copy()
    env_vars.update({
        'INPUT_CSV_FILE': str(data_file),
        'RESULT_DIR': str(preprocessing_dir),
        'PYTHONPATH': str(Path(__file__).parent / "final_pipeline")
    })
    
    # Run preprocessing
    preprocessing_script = Path(__file__).parent / "final_pipeline" / "preprocessing_no_mi.py"
    
    print(f"Running: python {preprocessing_script}")
    print(f"Input: {data_file}")
    print(f"Output: {preprocessing_dir}")

    # Get workspace root (parent of executable/)
    workspace_root = Path(__file__).parent.parent
    
    try:
        result = subprocess.run([
            sys.executable, str(preprocessing_script)
        ], env=env_vars, capture_output=True, text=True, cwd=workspace_root)

        # Always print subprocess output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"Preprocessing failed with return code {result.returncode}")
            return False

        print("Preprocessing completed successfully")
        return True

    except Exception as e:
        print(f"Error running preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_dynotears(data_file, result_dir, input_basename):
    """Run DynoTEARS analysis"""
    # Check if Tucker-CAM should be used
    use_tucker = os.getenv('USE_TUCKER_CAM', 'false').lower() in ('true', '1', 'yes', 'on')
    method_name = "Tucker-CAM" if use_tucker else "DynoTEARS"

    print(f"Step 2: {method_name} Analysis")

    preprocessing_dir = result_dir / 'preprocessing'
    weights_dir = result_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Find preprocessing output files
    differenced_file = preprocessing_dir / f'{input_basename}_differenced_stationary_series.csv'
    lags_file = preprocessing_dir / f'{input_basename}_optimal_lags.csv'

    if not differenced_file.exists():
        print(f"Error: Differenced file not found: {differenced_file}")
        return False

    if not lags_file.exists():
        print(f"Error: Lags file not found: {lags_file}")
        return False

    # Environment variables for DynoTEARS/Tucker-CAM - pass through all existing env vars
    env_vars = os.environ.copy()
    env_vars.update({
        'INPUT_DIFFERENCED_CSV': str(differenced_file),
        'INPUT_LAGS_CSV': str(lags_file),
        'RESULT_DIR': str(result_dir),
        'PYTHONPATH': str(Path(__file__).parent / "final_pipeline")
    })

    weights_dir = result_dir / 'causal_discovery'

    # Select script based on USE_TUCKER_CAM flag
    if use_tucker:
        # Use restart wrapper to prevent memory accumulation
        dynotears_script = Path(__file__).parent / "final_pipeline" / "dbn_dynotears_tucker_cam_restart.py"
    else:
        dynotears_script = Path(__file__).parent / "final_pipeline" / "dbn_dynotears_fixed_lambda.py"

    print(f"Running: python {dynotears_script}")
    print(f"Differenced data: {differenced_file}")
    print(f"Lags file: {lags_file}")
    print(f"Output: {weights_dir}")

    # Get workspace root (parent of executable/)
    workspace_root = Path(__file__).parent.parent

    try:
        result = subprocess.run([
            sys.executable, str(dynotears_script)
        ], env=env_vars, capture_output=True, text=True, cwd=workspace_root)

        # Always print subprocess output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print(f"{method_name} failed with return code {result.returncode}")
            return False

        print(f"{method_name} analysis completed successfully")
        return True

    except Exception as e:
        print(f"Error running {method_name}: {e}")
        import traceback
        traceback.print_exc()
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
    if 'preprocessing' in skip_steps:
        print("Step 1: Preprocessing - SKIPPED (explicit skip)")
    elif preprocessing_complete:
        print("Step 1: Preprocessing - SKIPPED (already completed)")
    else:
        if not run_preprocessing(data_file, output_dir, input_basename):
            print("Pipeline failed at preprocessing step")
            return False

    # Step 2: DynoTEARS Analysis
    if 'dynotears' in skip_steps:
        print("Step 2: DynoTEARS Analysis - SKIPPED (explicit skip)")
    elif dynotears_complete:
        print("Step 2: DynoTEARS Analysis - SKIPPED (already completed)")
    else:
        if not run_dynotears(data_file, output_dir, input_basename):
            print("Pipeline failed at DynoTEARS step")
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

    # Parse skip_steps if provided
    skip_steps = None
    if args.skip_steps:
        skip_steps = [step.strip() for step in args.skip_steps.split(',')]

    success = run_pipeline(data_file, output_dir, resume=args.resume, skip_steps=skip_steps)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
