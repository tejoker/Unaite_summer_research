#!/usr/bin/env python3
"""
Test the epicenter detector V2 (multi-strategy) on all 6 anomaly types
"""

import subprocess
import sys
from pathlib import Path
import glob

# Anomaly types to test
ANOMALY_TYPES = [
    'spike',
    'level_shift',
    'drift',
    'trend_change',
    'amplitude_change',
    'variance_burst'
]

def find_weights_file(result_dir):
    """Find weights_enhanced.csv file in result directory"""
    weights_files = list(Path(result_dir).rglob('weights_enhanced.csv'))
    if not weights_files:
        return None
    # Return the one with enhanced format (has 11 columns)
    for f in weights_files:
        with open(f) as file:
            header = file.readline()
            if 't_end' in header and 'i' in header and 'j' in header:
                return f
    return weights_files[0] if weights_files else None

def find_ground_truth(anomaly_type):
    """Find ground truth JSON file for anomaly type"""
    possible_paths = [
        f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json",
        f"data/Anomaly/{anomaly_type}_metadata.json",
        f"data/Anomaly/metadata_{anomaly_type}.json",
        f"data/Anomaly/{anomaly_type}.json",
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    return None

def main():
    print("="*100)
    print("TESTING EPICENTER DETECTOR V2 (Multi-Strategy) ON ALL ANOMALY TYPES")
    print("="*100)

    # Find Golden weights file
    print("\nFinding Golden weights file...")
    golden_file = find_weights_file("results/Golden_1th_chunk_run2_20251013_085952")
    if not golden_file:
        print("ERROR: Could not find Golden weights file")
        return 1
    print(f"✓ Golden: {golden_file}")

    results = []

    for anomaly_type in ANOMALY_TYPES:
        print("\n" + "="*100)
        print(f"TESTING: {anomaly_type.upper()}")
        print("="*100)

        # Find anomaly weights file
        anomaly_patterns = [
            f"results/Anomaly_no_mi_{anomaly_type}_*",
            f"results/{anomaly_type}",
            f"results/Anomaly_{anomaly_type}",
            f"results/{anomaly_type}_1th_chunk",
        ]

        anomaly_file = None
        for pattern in anomaly_patterns:
            matching_dirs = glob.glob(pattern)
            if matching_dirs:
                latest_dir = max(matching_dirs, key=lambda p: Path(p).stat().st_mtime)
                anomaly_file = find_weights_file(latest_dir)
                if anomaly_file:
                    break

        if not anomaly_file:
            print(f"⚠️  WARNING: No weights file found for {anomaly_type}")
            print(f"    Searched patterns: {anomaly_patterns}")
            print(f"    You need to run: python3 run_anomaly_no_mi.py {anomaly_type}")
            results.append({'type': anomaly_type, 'status': 'MISSING_WEIGHTS'})
            continue

        print(f"✓ Anomaly: {anomaly_file}")

        # Find ground truth
        ground_truth = find_ground_truth(anomaly_type)
        if ground_truth:
            print(f"✓ Ground truth: {ground_truth}")
        else:
            print(f"⚠️  No ground truth file found")

        # Run epicenter detector V2
        output_file = f"results/epicenter_detection_v2_{anomaly_type}.csv"

        cmd = [
            sys.executable,
            "epicenter_detector_v2.py",
            "--golden", str(golden_file),
            "--anomaly", str(anomaly_file),
            "--output", output_file
        ]

        if ground_truth:
            cmd.extend(["--ground-truth", ground_truth])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        results.append({
            'type': anomaly_type,
            'status': status,
            'output': output_file if result.returncode == 0 else None
        })

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY - V2 Multi-Strategy Detector")
    print("="*100)

    for r in results:
        status_icon = r['status']
        print(f"{status_icon:<10} {r['type']:<20} {r.get('output', 'N/A')}")

    passed = sum(1 for r in results if '✅' in r['status'])
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
