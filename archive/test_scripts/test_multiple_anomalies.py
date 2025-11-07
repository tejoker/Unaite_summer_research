#!/usr/bin/env python3
"""
Test script to demonstrate gap-based multiple anomaly detection.

This script tests the pipeline on existing single-anomaly datasets to show
how the gap detection works.
"""

import sys
from pathlib import Path
from end_to_end_anomaly_pipeline import EndToEndAnomalyPipeline

def test_single_anomaly_detection(anomaly_type: str):
    """Test gap detection on a single anomaly (should detect 1 anomaly)."""

    print(f"\n{'='*80}")
    print(f"Testing Multiple Anomaly Detection on: {anomaly_type}")
    print(f"{'='*80}\n")

    # Paths
    golden_weights = "results/master_anomaly_test_20251014_171954/Golden_Test_Run/weights/weights_enhanced.csv"
    anomaly_weights = f"results/master_anomaly_test_20251014_171954/{anomaly_type}_run/weights/weights_enhanced.csv"
    ground_truth = f"data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__{anomaly_type}.json"
    output_dir = f"results/multi_anomaly_test/{anomaly_type}"

    # Check if files exist
    if not Path(anomaly_weights).exists():
        print(f"❌ Anomaly weights not found: {anomaly_weights}")
        return False

    # Create pipeline
    pipeline = EndToEndAnomalyPipeline(
        golden_weights_path=golden_weights,
        anomaly_weights_path=anomaly_weights,
        ground_truth_path=ground_truth,
        threshold=0.01
    )

    # Run multiple anomaly detection
    results = pipeline.run_multiple_anomaly_pipeline(
        output_dir=output_dir,
        gap_threshold=5
    )

    if results['success']:
        print(f"\n✅ SUCCESS: {anomaly_type}")
        print(f"   Anomalies detected: {results['n_anomalies']}")
        print(f"   Report: {output_dir}/multi_anomaly_summary.txt")
        return True
    else:
        print(f"\n❌ FAILED: {anomaly_type}")
        return False


def compare_all_anomaly_types():
    """Test gap detection on all 6 anomaly types and compare results."""

    anomaly_types = ['spike', 'drift', 'level_shift', 'amplitude_change', 'trend_change', 'variance_burst']

    print("\n" + "="*80)
    print("MULTIPLE ANOMALY DETECTION - COMPARISON ACROSS ALL TYPES")
    print("="*80)
    print("\nThis demonstrates how gap-based detection separates temporal clusters.")
    print("Each anomaly type has only 1 injected anomaly, so we expect:")
    print("  - spike: May detect 1 anomaly (brief, isolated)")
    print("  - drift: May detect 1 anomaly (gradual, but continuous)")
    print("  - level_shift: Will detect 1 anomaly (persistent)")
    print("  - Others: Similar patterns\n")

    results_summary = []

    for anom_type in anomaly_types:
        success = test_single_anomaly_detection(anom_type)

        if success:
            # Read results
            result_path = Path(f"results/multi_anomaly_test/{anom_type}/multi_anomaly_results.json")
            if result_path.exists():
                import json
                with open(result_path) as f:
                    data = json.load(f)
                    results_summary.append({
                        'type': anom_type,
                        'n_anomalies': data['n_anomalies'],
                        'anomalies': data['anomalies']
                    })

    # Generate comparison table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Anomaly Type':<20} {'# Detected':<12} {'Window Ranges':<40}")
    print("-"*80)

    for result in results_summary:
        anom_type = result['type']
        n_anomalies = result['n_anomalies']

        window_ranges = []
        for anomaly in result['anomalies']:
            first = anomaly['first_window']
            last = anomaly['last_window']
            n_windows = anomaly['n_windows']
            window_ranges.append(f"{first}-{last} ({n_windows}w)")

        ranges_str = ", ".join(window_ranges[:3])  # Show first 3
        if len(window_ranges) > 3:
            ranges_str += f", ... (+{len(window_ranges)-3} more)"

        print(f"{anom_type:<20} {n_anomalies:<12} {ranges_str:<40}")

    print("="*80)

    # Explain interpretation
    print("\nINTERPRETATION:")
    print("  - Single detected anomaly = Good! All windows belong to same injected anomaly")
    print("  - Multiple detected anomalies = Gap detected (could be true or artifact)")
    print("    * True gaps: Windows with no causal changes (signal returned to baseline)")
    print("    * Artifacts: Threshold effects, transient dynamics\n")

    return results_summary


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Test specific anomaly type
        anomaly_type = sys.argv[1]
        test_single_anomaly_detection(anomaly_type)
    else:
        # Test all anomaly types
        compare_all_anomaly_types()
