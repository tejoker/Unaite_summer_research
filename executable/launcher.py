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

def get_all_executable_folders(executable_dir):
    """Get all folders in executable directory that contain .py files"""
    folders = []
    for item in executable_dir.iterdir():
        if item.is_dir() and item.name != "__pycache__":
            # Check if folder contains any .py files
            py_files = list(item.glob("*.py"))
            if py_files:
                folders.append(item)
    return folders

def get_all_python_files(folder):
    """Get all .py files in a folder, excluding certain patterns"""
    # Exclude library files but keep executable scripts
    exclude_patterns = {'__init__.py', 'test_', 'setup.py'}
    # Specifically exclude library files (not executable scripts)
    exclude_files = {'structuremodel.py', 'transformers.py', 'resource_manager.py', 'performance_benchmark.py', 'dynotears_benchmark.py'}
    py_files = []
    
    for file in folder.glob("*.py"):
        # Skip if matches exclude patterns or is in exclude files list
        if (not any(pattern in file.name for pattern in exclude_patterns) and 
            file.name not in exclude_files):
            py_files.append(file)
    
    return sorted(py_files)

def find_all_csv_files(data_dir):
    """Find all CSV files in all data subdirectories"""
    csv_files = {}
    
    for data_type_dir in data_dir.iterdir():
        if data_type_dir.is_dir():
            data_type = data_type_dir.name.lower()
            csv_files[data_type] = []
            
            # Look in the main directory
            for csv_file in data_type_dir.glob("*.csv"):
                csv_files[data_type].append(csv_file)
            
            # Look in subdirectories (like chunking)
            for subdir in data_type_dir.iterdir():
                if subdir.is_dir():
                    for csv_file in subdir.glob("*.csv"):
                        csv_files[data_type].append(csv_file)
    
    return csv_files

def find_csv_file_by_path(csv_path, data_dir):
    """Find a specific CSV file and determine its data type"""
    csv_path = Path(csv_path)
    
    # If it's an absolute path and exists, use it directly
    if csv_path.is_absolute() and csv_path.exists():
        # Determine data type from path
        for data_type_dir in data_dir.iterdir():
            if data_type_dir.is_dir():
                try:
                    csv_path.relative_to(data_type_dir)
                    return csv_path, data_type_dir.name.lower()
                except ValueError:
                    continue
        # If not in data directory, try to infer data type from file path or name
        # Look for known data types in the path
        path_str = str(csv_path).lower()
        for data_type_dir in data_dir.iterdir():
            if data_type_dir.is_dir():
                data_type = data_type_dir.name.lower()
                if data_type in path_str:
                    return csv_path, data_type
        
        # Default to 'anomaly' for unknown files to avoid creating 'Data' directory
        return csv_path, 'anomaly'
    
    # If it's a relative path, search for it in data directories
    if not csv_path.is_absolute():
        # Search in all data type directories
        for data_type_dir in data_dir.iterdir():
            if data_type_dir.is_dir():
                # Look in main directory
                potential_file = data_type_dir / csv_path.name
                if potential_file.exists():
                    return potential_file, data_type_dir.name.lower()
                
                # Look in subdirectories
                for subdir in data_type_dir.iterdir():
                    if subdir.is_dir():
                        potential_file = subdir / csv_path.name
                        if potential_file.exists():
                            return potential_file, data_type_dir.name.lower()
    
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

def create_result_path(csv_file, data_type, script_name, results_dir, launch_folder_name=None):
    """Create a recognizable result path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_basename = csv_file.stem
    script_basename = script_name.stem
    
    result_name = f"{script_basename}_{csv_basename}_{timestamp}"
    
    # Create path with launch folder if provided
    if launch_folder_name:
        result_dir = results_dir / launch_folder_name / data_type.title() / result_name
    else:
        result_dir = results_dir / data_type.title() / result_name
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    return result_dir

def set_pipeline_env_vars(step_name, input_file, output_dir, preprocessing_files=None):
    """Set environment variables for pipeline execution"""
    env_vars = {}
    
    # Set RESULT_DIR for all scripts to use the launcher folder structure
    env_vars['RESULT_DIR'] = str(output_dir)
    
    if step_name == 'preprocessing.py':
        env_vars['INPUT_CSV_FILE'] = str(input_file)
        # Preprocessing.py now handles its own output paths based on input filename
        # No need to set OUTPUT_* environment variables as they use defaults with input basename
    elif step_name in ['dbn_dynotears.py', 'dynotears.py']:
        # Use the preprocessing output files if provided
        if preprocessing_files:
            env_vars['INPUT_DIFFERENCED_CSV'] = str(preprocessing_files['differenced'])
            env_vars['INPUT_LAGS_CSV'] = str(preprocessing_files['lags'])
            env_vars['INPUT_MI_MASK_CSV'] = str(preprocessing_files['mi_mask'])
        else:
            # Fallback to old behavior
            env_vars['INPUT_DIFFERENCED_CSV'] = str(input_file)
            env_vars['INPUT_LAGS_CSV'] = 'optimal_lags.csv'
            env_vars['INPUT_MI_MASK_CSV'] = 'mi_mask_edges.csv'
        # Scripts will use RESULT_DIR to determine their output paths
    elif step_name == 'reconstruction.py':
        # Reconstruction uses the DBN results
        env_vars['RESULTS_DIR'] = str(output_dir)
        # Will look for weights CSV files in the results directory
    elif step_name == 'tendance.py':
        env_vars['INPUT_CSV_FILE'] = str(input_file)
        # Set output to data/Anomaly directory
        script_dir = Path(input_file).parent.parent.parent
        anomaly_dir = script_dir / "data" / "Anomaly"
        env_vars['OUTPUT_DIR'] = str(anomaly_dir)
    
    return env_vars

def get_pipeline_output_files(script_dir, exclude_input=None):
    """Get CSV files generated by a pipeline step"""
    output_files = []
    for csv_file in script_dir.glob("*.csv"):
        if exclude_input and csv_file.name == exclude_input:
            continue
        output_files.append(csv_file)
    return output_files

def find_preprocessing_output_files(input_csv_file, result_dir=None):
    """Find preprocessing output files based on input CSV file"""
    input_path = Path(input_csv_file)
    input_basename = input_path.stem
    
    # Use result directory if provided (from launcher)
    if result_dir:
        preprocessing_dir = Path(result_dir) / 'preprocessing'
    else:
        # Fallback to old behavior: look for data directory
        current_dir = input_path.parent
        data_dir = None
        
        for _ in range(5):  # Search up to 5 levels up
            if current_dir.name == 'data':
                data_dir = current_dir
                break
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir
        
        if data_dir is None:
            data_dir = input_path.parent / 'data'
        
        preprocessing_dir = data_dir / 'preprocessing'
    
    # Expected output file names based on preprocessing.py
    differenced_file = preprocessing_dir / f'{input_basename}_differenced_stationary_series.csv'
    lags_file = preprocessing_dir / f'{input_basename}_optimal_lags.csv'  
    mi_mask_file = preprocessing_dir / f'{input_basename}_mi_mask_edges.csv'
    
    return {
        'differenced': differenced_file,
        'lags': lags_file,
        'mi_mask': mi_mask_file,
        'preprocessing_dir': preprocessing_dir
    }

def run_enhanced_pipeline_sequence(script_paths, initial_csv_file, result_dir):
    """Run enhanced pipeline sequence including reconstruction"""
    print(f"\n{'🔗'*60}")
    print(f"Running Enhanced Pipeline Sequence:")
    for i, script in enumerate(script_paths, 1):
        print(f"  {i}. {script.name}")
    print(f"Initial data: {initial_csv_file.name}")
    print(f"Result dir: {result_dir}")
    print(f"{'🔗'*60}")
    
    current_input = initial_csv_file
    preprocessing_files = None
    
    # Save current working directory
    original_cwd = Path.cwd()
    
    try:
        # Change to the script directory (assuming all scripts are in the same dir)
        script_dir = script_paths[0].parent
        os.chdir(script_dir)
        
        for i, script_path in enumerate(script_paths, 1):
            print(f"\n{'='*50}")
            print(f"Pipeline Step {i}/{len(script_paths)}: {script_path.name}")
            print(f"Input: {current_input.name}")
            print(f"{'='*50}")
            
            # Handle preprocessing step
            if script_path.name == 'preprocessing.py':
                # Set environment variables for preprocessing
                env_vars = set_pipeline_env_vars(script_path.name, current_input, result_dir)
                # Get expected preprocessing output file paths
                preprocessing_files = find_preprocessing_output_files(current_input, result_dir)
                print(f"📁 Expected preprocessing outputs:")
                for key, path in preprocessing_files.items():
                    if key != 'preprocessing_dir':
                        print(f"   {key}: {path}")
            elif script_path.name in ['dbn_dynotears.py', 'dynotears.py'] and preprocessing_files:
                # Use preprocessing output files for dbn_dynotears and dynotears
                env_vars = set_pipeline_env_vars(script_path.name, current_input, result_dir, preprocessing_files)
                # Check if preprocessing files exist
                missing_files = []
                for key, path in preprocessing_files.items():
                    if key != 'preprocessing_dir' and not Path(path).exists():
                        missing_files.append(f"{key}: {path}")
                if missing_files:
                    print(f"⚠️  Missing preprocessing files:")
                    for missing in missing_files:
                        print(f"   {missing}")
                else:
                    print(f"✅ All preprocessing files found")
            elif script_path.name == 'reconstruction.py':
                # Reconstruction step - use results from DBN
                env_vars = set_pipeline_env_vars(script_path.name, current_input, result_dir)
                print(f"🔄 Setting up reconstruction with DBN results")
            else:
                # Default environment variables
                env_vars = set_pipeline_env_vars(script_path.name, current_input, result_dir)
            
            # Copy current environment and add our variables
            env = os.environ.copy()
            env.update(env_vars)
            
            print(f"🔧 Environment variables:")
            for key, value in env_vars.items():
                print(f"   {key}={value}")
            
            # Build command with script-specific arguments
            cmd = [sys.executable, str(script_path.name)]
            
            # Add arguments for reconstruction.py
            if script_path.name == 'reconstruction.py':
                cmd.extend([
                    '--results_dir', str(result_dir),
                    '--variables', 
                    'Temperatur Druckpfannenlager links',
                    'Temperatur Druckpfannenlager rechts', 
                    'Temperatur Exzenterlager links',
                    'Temperatur Exzenterlager rechts',
                    'Temperatur Ständerlager links',
                    'Temperatur Ständerlager rechts',
                    '--lag', '1',
                    '--steps', '100'
                ])
            
            # Add arguments for dynotears_variants.py
            elif script_path.name == 'dynotears_variants.py':
                cmd.extend(['--data', str(current_input), '--output-dir', str(result_dir)])
            
            # Run the script with environment variables
            result = subprocess.run(cmd, 
            cwd=script_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"❌ {script_path.name} failed with return code {result.returncode}")
                if result.stdout:
                    print("STDOUT:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
                return False
            
            print(f"✅ {script_path.name} completed successfully")
            if result.stdout.strip():
                print("STDOUT:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            
            # Find output files for next step (if not the last step)
            if i < len(script_paths):
                # For preprocessing -> dbn_dynotears_enhanced, use the differenced file from data/preprocessing
                if script_path.name == 'preprocessing.py' and preprocessing_files:
                    output_file = preprocessing_files['differenced']
                    if Path(output_file).exists():
                        current_input = Path(output_file)
                        print(f"📤 Output for next step: {current_input.name}")
                    else:
                        print(f"❌ Expected preprocessing output file not found: {output_file}")
                        return False
                # For dbn_dynotears -> reconstruction, the reconstruction script will find results automatically
                elif script_path.name == 'dbn_dynotears.py':
                    print(f"📤 DBN analysis completed, reconstruction will use results from {result_dir}")
        
        # Copy all final results to result directory
        copy_generated_files(script_dir, result_dir, None)
        
        # Also copy preprocessing files if they exist
        if preprocessing_files and preprocessing_files['preprocessing_dir']:
            copy_preprocessing_files(preprocessing_files['preprocessing_dir'], result_dir)
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Pipeline timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        return False
    finally:
        # Restore working directory
        os.chdir(original_cwd)

def run_script_with_data(script_path, csv_file, result_dir):
    """Run a Python script with a CSV file as input"""
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    print(f"Data: {csv_file.name}")
    print(f"Result dir: {result_dir}")
    print(f"{'='*60}")
    
    # Set environment variables for this script
    env_vars = set_pipeline_env_vars(script_path.name, csv_file, result_dir)
    
    # Copy current environment and add our variables
    env = os.environ.copy()
    env.update(env_vars)
    
    if env_vars:
        print(f"🔧 Environment variables:")
        for key, value in env_vars.items():
            print(f"   {key}={value}")
    
    # Save current working directory
    original_cwd = Path.cwd()
    
    try:
        # Change to script directory
        os.chdir(script_path.parent)
        
        # Build command based on script requirements
        cmd = [sys.executable, str(script_path.name)]
        
        # Special handling for dynotears_variants.py
        if script_path.name == 'dynotears_variants.py':
            cmd.extend(['--data', str(csv_file), '--output-dir', str(result_dir)])
        
        # Run the script with environment variables
        result = subprocess.run(cmd, 
        cwd=script_path.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour timeout
        )
        
        # Copy all generated files to result directory
        copy_generated_files(script_path.parent, result_dir, csv_file.name)
        
        if result.returncode == 0:
            print(f"✅ {script_path.name} completed successfully")
            if result.stdout.strip():
                print("STDOUT:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ {script_path.name} failed with return code {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_path.name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"❌ Error running {script_path.name}: {e}")
        return False
    finally:
        # Restore working directory
        os.chdir(original_cwd)

def copy_generated_files(source_dir, target_dir, input_csv_name):
    """Copy all files generated by the script to the result directory"""
    # Copy all CSV, log, pkl files and result directories
    patterns = ["*.csv", "*.log", "*.pkl", "*.json", "*.txt", "*.png", "*.jpg", "*.pdf", "*.npy"]
    
    for pattern in patterns:
        for file in source_dir.glob(pattern):
            if file.is_file() and file.name != input_csv_name:
                target_file = target_dir / file.name
                shutil.copy2(file, target_file)
                print(f"  📄 Copied: {file.name}")
    
    # Copy result directories
    for subdir in source_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ["__pycache__", ".git"]:
            # Check if directory contains result files
            if any(subdir.glob("*")):
                target_subdir = target_dir / subdir.name
                if target_subdir.exists():
                    shutil.rmtree(target_subdir)
                shutil.copytree(subdir, target_subdir)
                print(f"  📁 Copied directory: {subdir.name}")

def copy_preprocessing_files(preprocessing_dir, target_dir):
    """Copy preprocessing output files to the result directory"""
    preprocessing_dir = Path(preprocessing_dir)
    if not preprocessing_dir.exists():
        print(f"  ⚠️  Preprocessing directory not found: {preprocessing_dir}")
        return
        
    # Copy all files from preprocessing directory
    patterns = ["*.csv", "*.npy", "*.pkl", "*.json", "*.txt"]
    copied_files = 0
    
    for pattern in patterns:
        for file in preprocessing_dir.glob(pattern):
            if file.is_file():
                target_file = target_dir / file.name
                shutil.copy2(file, target_file)
                print(f"  📄 Copied preprocessing file: {file.name}")
                copied_files += 1
    
    if copied_files == 0:
        print(f"  ⚠️  No preprocessing files found in {preprocessing_dir}")
    else:
        print(f"  ✅ Copied {copied_files} preprocessing files")

def cleanup_stray_analysis_directories(results_dir, launch_folder_name):
    """Move any stray analysis directories into the launch folder"""
    results_dir = Path(results_dir)
    analysis_dirs = ['combined_analysis', 'tendance_analysis']
    
    for analysis_dir_name in analysis_dirs:
        analysis_dir = results_dir / analysis_dir_name
        if analysis_dir.exists() and analysis_dir.is_dir():
            # Create destination in launch folder
            launch_dir = results_dir / launch_folder_name
            launch_dir.mkdir(parents=True, exist_ok=True)
            
            destination = launch_dir / analysis_dir_name
            if destination.exists():
                shutil.rmtree(destination)
            
            # Move the directory
            shutil.move(str(analysis_dir), str(destination))
            print(f"  📁 Moved {analysis_dir_name} to {launch_folder_name}/{analysis_dir_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced launcher that runs dbn_dynotears.py followed by reconstruction.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ENHANCED PIPELINE automatically (preprocessing -> dbn_dynotears -> reconstruction)
  python launcher2.py --folder final_pipeline --data-type golden
  python launcher2.py --folder final_pipeline --csv-file output_of_the_11th_chunk.csv
  
  # Run specific script with specific CSV file
  python launcher2.py --script final_pipeline/dbn_dynotears.py --csv-file output_of_the_11th_chunk.csv
  
  # Run other folders individually (benchmark, test, etc.)
  python launcher2.py --folder benchmark --data-type golden
  
  # Force pipeline mode for other folders
  python launcher2.py --folder benchmark --data-type golden --pipeline
  
  # Run everything
  python launcher2.py --all
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', 
                      help='Specific executable folder to run (e.g., final_pipeline, benchmark)')
    group.add_argument('--script', 
                      help='Specific script to run (e.g., final_pipeline/dbn_dynotears_enhanced.py)')
    group.add_argument('--all', action='store_true',
                      help='Run all scripts in all folders with all data')
    
    parser.add_argument('--data-type', default='all',
                       help='Data type to process (golden, anomaly, reconstruction, test, or all)')
    parser.add_argument('--csv-file', 
                       help='Specific CSV file to process (filename or path)')
    parser.add_argument('--pipeline', action='store_true',
                       help='Run scripts in pipeline mode (output of one becomes input of next)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without actually running')
    
    args = parser.parse_args()
    
    # Setup paths
    paths = setup_paths()
    
    print(f"🚀 Enhanced Universal Script Launcher")
    print(f"📁 Executable directory: {paths['executable_dir']}")
    print(f"📁 Data directory: {paths['data_dir']}")
    print(f"📁 Results directory: {paths['results_dir']}")
    
    # Handle CSV file selection
    if args.csv_file:
        # Process single CSV file
        try:
            csv_file, data_type = find_csv_file_by_path(args.csv_file, paths['data_dir'])
            csv_files_to_process = {data_type: [csv_file]}
            print(f"📄 Processing single CSV: {csv_file} (data type: {data_type})")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
    else:
        # Find all CSV files
        all_csv_files = find_all_csv_files(paths['data_dir'])
        
        # Filter data types
        if args.data_type.lower() == 'all':
            csv_files_to_process = all_csv_files
        else:
            data_type = args.data_type.lower()
            if data_type in all_csv_files:
                csv_files_to_process = {data_type: all_csv_files[data_type]}
            else:
                print(f"❌ Data type '{data_type}' not found. Available: {list(all_csv_files.keys())}")
                return 1
    
    # Determine scripts to run
    scripts_to_run = []
    
    if args.all:
        # Run all scripts in all folders
        for folder in get_all_executable_folders(paths['executable_dir']):
            for script in get_all_python_files(folder):
                scripts_to_run.append(script)
    elif args.folder:
        # Run all scripts in specified folder
        folder_path = paths['executable_dir'] / args.folder
        if not folder_path.exists():
            print(f"❌ Folder '{args.folder}' not found in executable directory")
            return 1
        scripts_to_run = get_all_python_files(folder_path)
    elif args.script:
        # Run specific script
        script_path = paths['executable_dir'] / args.script
        if not script_path.exists():
            print(f"❌ Script '{args.script}' not found")
            return 1
        scripts_to_run = [script_path]
    
    # Show execution plan
    total_executions = sum(len(csv_files) for csv_files in csv_files_to_process.values()) * len(scripts_to_run)
    
    print(f"\n📋 Execution Plan:")
    print(f"   Scripts to run: {len(scripts_to_run)}")
    for script in scripts_to_run:
        print(f"     - {script.relative_to(paths['executable_dir'])}")
    
    print(f"   Data types: {list(csv_files_to_process.keys())}")
    for data_type, csv_files in csv_files_to_process.items():
        print(f"     - {data_type}: {len(csv_files)} files")
    
    print(f"   Total executions: {total_executions}")
    
    if args.dry_run:
        print("\n🔍 Dry run complete - no scripts were executed")
        return 0
    
    # Confirm execution
    if total_executions > 10:
        response = input(f"\n⚠️  This will run {total_executions} script executions. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Execution cancelled")
            return 0
    
    # Create launch timestamp for this execution
    launch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_folder_name = f"launch2_{launch_timestamp}"
    
    # Execute all combinations
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Check if we should run in pipeline mode
    # Auto-enable pipeline mode for final_pipeline folder
    auto_pipeline = False
    if args.folder == 'final_pipeline' and not args.pipeline:
        auto_pipeline = True
        print(f"\n🔗 AUTO-PIPELINE MODE (final_pipeline folder detected)")
    elif args.pipeline and len(scripts_to_run) > 1:
        print(f"\n🔗 Running in PIPELINE MODE")
    
    if (args.pipeline and len(scripts_to_run) > 1) or auto_pipeline:
        # Group scripts by directory for pipeline execution
        script_groups = {}
        for script in scripts_to_run:
            dir_key = script.parent
            if dir_key not in script_groups:
                script_groups[dir_key] = []
            script_groups[dir_key].append(script)
        
        for script_dir, script_group in script_groups.items():
            # Sort scripts in logical order (preprocessing -> dbn_dynotears -> reconstruction -> dynotears -> dynotears_variants)
            pipeline_order = ['preprocessing.py', 'dbn_dynotears.py', 'reconstruction.py', 'dynotears.py', 'dynotears_variants.py']
            ordered_scripts = []
            
            for script_name in pipeline_order:
                for script in script_group:
                    if script.name == script_name:
                        ordered_scripts.append(script)
                        break
            
            # Add any remaining scripts not in the predefined order
            for script in script_group:
                if script not in ordered_scripts:
                    ordered_scripts.append(script)
            
            if not ordered_scripts:
                print(f"❌ No executable scripts found in {script_dir.name}")
                continue
                
            # Run pipeline for each CSV file
            for data_type, csv_files in csv_files_to_process.items():
                for csv_file in csv_files:
                    # Create result directory name based on pipeline
                    if auto_pipeline or args.folder == 'final_pipeline':
                        pipeline_name = "enhanced_pipeline"
                    else:
                        pipeline_name = "_".join([s.stem for s in ordered_scripts])
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_basename = csv_file.stem
                    result_name = f"pipeline_{pipeline_name}_{csv_basename}_{timestamp}"
                    
                    # Use launch folder for organization
                    result_dir = paths['results_dir'] / launch_folder_name / data_type.title() / result_name
                    result_dir.mkdir(parents=True, exist_ok=True)
                    
                    if run_enhanced_pipeline_sequence(ordered_scripts, csv_file, result_dir):
                        successful += 1
                    else:
                        failed += 1
    else:
        # Regular mode - run each script individually
        print(f"\n⚙️ Running in INDIVIDUAL MODE")
        
        for script in scripts_to_run:
                
            for data_type, csv_files in csv_files_to_process.items():
                for csv_file in csv_files:
                    result_dir = create_result_path(csv_file, data_type, script, paths['results_dir'], launch_folder_name)
                    
                    if run_script_with_data(script, csv_file, result_dir):
                        successful += 1
                    else:
                        failed += 1
    
    # Clean up any stray analysis directories created directly in results
    cleanup_stray_analysis_directories(paths['results_dir'], launch_folder_name)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 PIPELINE EXECUTION COMPLETE!")
    print(f"🏁 PIPELINE EXECUTION COMPLETE!")
    print(f"🏁 PIPELINE EXECUTION COMPLETE!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📁 Results saved in: {paths['results_dir']}/{launch_folder_name}")
    
    if failed == 0:
        print(f"✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"✅ CHECK YOUR RESULTS AT: {paths['results_dir']}/{launch_folder_name}")
    else:
        print(f"❌ {failed} STEP(S) FAILED - CHECK LOGS FOR DETAILS")
    
    print(f"{'='*60}")
    print(f"🏁 END OF PIPELINE EXECUTION")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())