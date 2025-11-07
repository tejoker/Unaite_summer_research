#!/usr/bin/env python3
"""
Diagnostic Script: Lambda Search Analysis

Purpose:
Tests the hypothesis that the Golden and Anomaly runs use different
regularization lambdas, which causes the "Butterfly Effect" of early
weight changes.

This script will:
1. Parse the log files from a Golden and an Anomaly run.
2. Search for lines indicating the result of a lambda hyperparameter search.
3. Extract and compare the final selected lambda_w and lambda_a values.
4. Report if the lambdas are different, which would validate the hypothesis.
"""

import re
import argparse
from pathlib import Path

def find_and_parse_lambda_from_log(log_dir: Path) -> dict:
    """
    Finds a .log file in the given directory and parses it to find the selected lambda values.
    Looks for patterns like "Best lambdas found: lambda_w=0.05, lambda_a=0.01".
    """
    if not log_dir.is_dir():
        print(f"❌ Directory not found: {log_dir}")
        return None

    # Find the first .log file in the directory
    log_files = list(log_dir.glob('*.log'))
    if not log_files:
        # If not in top-level, search recursively
        log_files = list(log_dir.rglob('*.log'))

    if not log_files:
        print(f"❌ No .log file found in {log_dir}")
        return None

    log_file = log_files[0] # Use the first one found
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return None

    # Regex to find lines with lambda values, tolerating different formats.
    # It captures floating point or integer numbers for lambda_w and lambda_a.
    lambda_pattern = re.compile(
        r"best lambdas found.*lambda_w[=:\s]+([\d\.]+).*lambda_a[=:\s]+([\d\.]+)",
        re.IGNORECASE
    )

    found_lambdas = None
    with open(log_file, 'r') as f:
        for line in f:
            match = lambda_pattern.search(line)
            if match:
                try:
                    lambda_w = float(match.group(1))
                    lambda_a = float(match.group(2))
                    found_lambdas = {'lambda_w': lambda_w, 'lambda_a': lambda_a}
                    # Keep reading to find the last reported "best" values, in case of multiple searches.
                except (ValueError, IndexError):
                    continue # Ignore malformed lines

    return found_lambdas

def main():
    parser = argparse.ArgumentParser(description="Compare selected lambdas from two DynoTEARS runs.")
    parser.add_argument('--golden-dir', required=True, help="Path to the result directory for the Golden run.")
    parser.add_argument('--anomaly-dir', required=True, help="Path to the result directory for the Anomaly run.")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("LAMBDA HYPERPARAMETER ANALYSIS")
    print("="*80)

    golden_dir_path = Path(args.golden_dir)
    anomaly_dir_path = Path(args.anomaly_dir)

    print(f"Searching for log in Golden directory: {golden_dir_path}")
    golden_lambdas = find_and_parse_lambda_from_log(golden_dir_path)

    print(f"Searching for log in Anomaly directory: {anomaly_dir_path}")
    anomaly_lambdas = find_and_parse_lambda_from_log(anomaly_dir_path)

    print("\n--- Comparison ---")
    if golden_lambdas:
        print(f"  Golden Run Lambdas:  lambda_w={golden_lambdas['lambda_w']}, lambda_a={golden_lambdas['lambda_a']}")
    else:
        print("  ❌ Could not find lambda values in Golden log.")

    if anomaly_lambdas:
        print(f"  Anomaly Run Lambdas: lambda_w={anomaly_lambdas['lambda_w']}, lambda_a={anomaly_lambdas['lambda_a']}")
    else:
        print("  ❌ Could not find lambda values in Anomaly log.")

    print("\n--- Verdict ---")
    if golden_lambdas and anomaly_lambdas:
        if golden_lambdas == anomaly_lambdas:
            print("✅ The selected lambdas are IDENTICAL.")
            print("  -> This REFUTES the hypothesis that different hyperparameters are the cause.")
            print("  -> The early weight change must be from another source (e.g., GPU non-determinism).")
        else:
            print("✅ The selected lambdas are DIFFERENT.")
            print("  -> This VALIDATES the 'Butterfly Effect' hypothesis.")
            print("  -> The future anomaly influenced the global hyperparameter search, causing")
            print("     different lambdas to be used, which in turn created the early weight changes.")
    else:
        print("⚠️ Cannot determine verdict as lambda values were not found in one or both logs.")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()