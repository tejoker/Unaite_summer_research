#!/usr/bin/env python3
"""
SOTA Anomaly Detection Benchmark Script
Runs state-of-the-art anomaly detection methods on raw time series data.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def find_raw_data_files(data_type, base_dir="/home/nicolasbigeard/program_internship_paul_wurth/data"):
    """Find raw time series CSV files for a given data type"""
    base_path = Path(base_dir)
    
    if data_type.lower() == "golden":
        # Look for golden/baseline data
        golden_dir = base_path / "Golden" / "chunking"
        if golden_dir.exists():
            csv_files = list(golden_dir.glob("*.csv"))
            if csv_files:
                # Return the first chunk as baseline
                csv_files.sort()
                return csv_files[0]
    
    elif data_type.lower() == "anomaly":
        # Look for anomaly data
        anomaly_dir = base_path / "Anomaly"
        if anomaly_dir.exists():
            csv_files = list(anomaly_dir.glob("*.csv"))
            if csv_files:
                # Return the first anomaly file
                csv_files.sort()
                return csv_files[0]
    
    raise FileNotFoundError(f"No CSV files found for {data_type}")

def run_sota_benchmark(baseline_file, anomaly_file, output_dir, methods=None):
    """Run SOTA anomaly detection benchmark"""
    
    if methods is None:
        methods = ['all']
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"sota_benchmark_results_{timestamp}.json"
    
    # Import the SOTA methods
    try:
        from sota_anomaly_methods import main as sota_main
    except ImportError as e:
        print(f"Error importing SOTA methods: {e}")
        return None
    
    # Prepare arguments for SOTA script
    sota_args = [
        '--baseline', str(baseline_file),
        '--anomaly', str(anomaly_file), 
        '--output', str(output_file),
        '--methods'] + methods
    
    print(f"🚀 Starting SOTA Anomaly Detection Benchmark")
    print(f"📁 Baseline: {baseline_file}")
    print(f"📁 Anomaly: {anomaly_file}")
    print(f"📁 Output: {output_file}")
    print(f"🔧 Methods: {methods}")
    print("=" * 60)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Temporarily replace sys.argv to pass arguments to sota_main
        original_argv = sys.argv
        sys.argv = ['sota_anomaly_methods.py'] + sota_args[1:]  # Skip script name, add rest
        
        # Run SOTA methods
        sota_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Load and enhance results with timing
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            # Add timing information
            results['benchmark_info'] = {
                'execution_time_seconds': execution_time,
                'execution_time_formatted': f"{execution_time:.2f}s",
                'timestamp': timestamp,
                'baseline_file': str(baseline_file),
                'anomaly_file': str(anomaly_file)
            }
            
            # Save enhanced results
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print("=" * 60)
            print(f"✅ Benchmark completed successfully!")
            print(f"⏱️  Execution time: {execution_time:.2f} seconds")
            print(f"📄 Results saved to: {output_file}")
            
            # Print summary
            print("\n🎯 BENCHMARK SUMMARY:")
            if 'summary' in results and 'method_performance' in results['summary']:
                for method, perf in results['summary']['method_performance'].items():
                    print(f"  {method.upper()}:")
                    print(f"    Avg anomaly ratio: {perf.get('avg_anomaly_ratio', 0):.4f}")
                    print(f"    W anomalies detected: {perf.get('W_anomalies_detected', 0)}")
                    print(f"    A anomalies detected: {perf.get('A_anomalies_detected', 0)}")
            
            return output_file
            
        else:
            print(f"❌ Output file not created: {output_file}")
            return None
            
    except Exception as e:
        print(f"❌ Error running SOTA benchmark: {e}")
        return None
    finally:
        # Ensure argv is restored
        sys.argv = original_argv

def main():
    parser = argparse.ArgumentParser(description="Run SOTA anomaly detection benchmark")
    parser.add_argument("--baseline", help="Baseline results file (W_matrices CSV)")
    parser.add_argument("--anomaly", help="Anomaly results file (W_matrices CSV)") 
    parser.add_argument("--output-dir", default="./sota_results", help="Output directory for results")
    parser.add_argument("--methods", nargs="+", 
                       choices=['all', 'isolation_forest', 'one_class_svm', 'lof', 'elliptic_envelope', 'dbscan', 'statistical'],
                       default=['all'], help="SOTA methods to run")
    parser.add_argument("--auto-find", action="store_true", help="Automatically find latest baseline and anomaly files")
    
    args = parser.parse_args()
    
    try:
        if args.auto_find:
            print("🔍 Auto-finding latest results...")
            baseline_file = find_latest_results("golden")
            anomaly_file = find_latest_results("anomaly")
            print(f"📁 Found baseline: {baseline_file}")
            print(f"📁 Found anomaly: {anomaly_file}")
        else:
            if not args.baseline or not args.anomaly:
                print("❌ Error: --baseline and --anomaly are required when not using --auto-find")
                return 1
            
            baseline_file = Path(args.baseline)
            anomaly_file = Path(args.anomaly)
            
            if not baseline_file.exists():
                print(f"❌ Baseline file not found: {baseline_file}")
                return 1
            
            if not anomaly_file.exists():
                print(f"❌ Anomaly file not found: {anomaly_file}")
                return 1
        
        # Run benchmark
        result_file = run_sota_benchmark(
            baseline_file=baseline_file,
            anomaly_file=anomaly_file, 
            output_dir=args.output_dir,
            methods=args.methods
        )
        
        if result_file:
            print(f"\n✅ Benchmark completed! Results: {result_file}")
            return 0
        else:
            print(f"\n❌ Benchmark failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())