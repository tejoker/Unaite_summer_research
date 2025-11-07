#!/usr/bin/env python3
"""
Multi-Window Voting Epicenter Detector

Purpose:
Improve epicenter detection accuracy by aggregating votes across multiple windows
instead of relying on a single window's detection.

Methodology:
1. Detect epicenter for each window independently
2. Aggregate votes across all windows
3. Apply voting strategies:
   - Simple majority voting
   - Weighted voting (earlier windows = higher weight)
   - Confidence-based voting (stronger signals = higher weight)
4. Select epicenter with most votes
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter


class MultiWindowVotingDetector:
    def __init__(self, golden_weights_csv: str, anomaly_weights_csv: str,
                 ground_truth_json: str = None, threshold: float = 0.01):
        """
        Initialize the multi-window voting detector.

        Args:
            golden_weights_csv: Path to golden weights CSV
            anomaly_weights_csv: Path to anomaly weights CSV
            ground_truth_json: Path to ground truth JSON (optional)
            threshold: Threshold for significant edge weight changes
        """
        self.threshold = threshold

        # Load weights
        print(f"Loading golden weights from: {golden_weights_csv}")
        self.golden_df = pd.read_csv(golden_weights_csv)
        print(f"Loading anomaly weights from: {anomaly_weights_csv}")
        self.anomaly_df = pd.read_csv(anomaly_weights_csv)

        # Load ground truth
        self.ground_truth = None
        self.ground_truth_sensor = None
        if ground_truth_json and Path(ground_truth_json).exists():
            with open(ground_truth_json, 'r') as f:
                self.ground_truth = json.load(f)
                raw_sensor = self.ground_truth['ts_col']
                self.ground_truth_sensor = f"{raw_sensor}_diff"
                print(f"Ground truth sensor: {self.ground_truth_sensor}")

        # Merge and calculate differences
        self.merged_df = pd.merge(
            self.golden_df,
            self.anomaly_df,
            on=['window_idx', 'parent_name', 'child_name', 'lag'],
            how='outer',
            suffixes=('_golden', '_anomaly')
        ).fillna(0)

        self.merged_df['weight_diff'] = abs(
            self.merged_df['weight_anomaly'] - self.merged_df['weight_golden']
        )

    def detect_epicenter_per_window(self) -> List[Dict]:
        """
        Detect epicenter for each window independently.

        Returns:
            List of dictionaries with epicenter detection per window
        """
        print("\n" + "="*80)
        print("STEP 1: PER-WINDOW EPICENTER DETECTION")
        print("="*80)

        results = []

        for window_idx in sorted(self.merged_df['window_idx'].unique()):
            window_data = self.merged_df[self.merged_df['window_idx'] == window_idx]

            # Filter significant changes
            significant = window_data[window_data['weight_diff'] > self.threshold]

            if len(significant) == 0:
                continue

            # Calculate outgoing change score for each variable
            source_scores = defaultdict(lambda: {'total_outgoing': 0.0, 'n_edges': 0, 'max_change': 0.0})

            for _, row in significant.iterrows():
                parent = row['parent_name']
                diff = row['weight_diff']

                source_scores[parent]['total_outgoing'] += diff
                source_scores[parent]['n_edges'] += 1
                source_scores[parent]['max_change'] = max(source_scores[parent]['max_change'], diff)

            # Find epicenter for this window
            if source_scores:
                epicenter = max(source_scores.items(), key=lambda x: x[1]['total_outgoing'])
                epicenter_var = epicenter[0]
                epicenter_score = epicenter[1]['total_outgoing']

                # Check if matches ground truth
                matches_gt = False
                if self.ground_truth_sensor:
                    matches_gt = self.ground_truth_sensor in epicenter_var

                results.append({
                    'window_idx': window_idx,
                    'epicenter': epicenter_var,
                    'score': epicenter_score,
                    'n_edges': epicenter[1]['n_edges'],
                    'max_change': epicenter[1]['max_change'],
                    'matches_gt': matches_gt
                })

        print(f"\nTotal windows with epicenter detected: {len(results)}")
        return results

    def apply_voting_strategy(self, window_detections: List[Dict],
                              strategy: str = 'simple_majority') -> Dict:
        """
        Apply voting strategy to aggregate epicenter votes across windows.

        Args:
            window_detections: List of per-window epicenter detections
            strategy: Voting strategy ('simple_majority', 'weighted', 'confidence')

        Returns:
            Dictionary with final epicenter and voting details
        """
        print(f"\n" + "="*80)
        print(f"STEP 2: MULTI-WINDOW VOTING (Strategy: {strategy})")
        print("="*80)

        if len(window_detections) == 0:
            print("No windows detected!")
            return {'epicenter': None, 'votes': {}}

        # Strategy 1: Simple Majority Voting
        if strategy == 'simple_majority':
            votes = Counter([w['epicenter'] for w in window_detections])

        # Strategy 2: Weighted by Window Position (earlier = higher weight)
        elif strategy == 'weighted_temporal':
            votes = defaultdict(float)
            for i, w in enumerate(window_detections):
                # Earlier windows get higher weight (exponential decay)
                weight = np.exp(-0.1 * i)
                votes[w['epicenter']] += weight

        # Strategy 3: Confidence-based (score-weighted)
        elif strategy == 'confidence':
            votes = defaultdict(float)
            for w in window_detections:
                # Weight by detection score
                votes[w['epicenter']] += w['score']

        # Strategy 4: Hybrid (temporal + confidence)
        elif strategy == 'hybrid':
            votes = defaultdict(float)
            for i, w in enumerate(window_detections):
                temporal_weight = np.exp(-0.1 * i)
                confidence_weight = w['score']
                votes[w['epicenter']] += temporal_weight * confidence_weight

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert defaultdict to regular dict for JSON serialization
        votes = dict(votes)

        # Find winner
        winner = max(votes.items(), key=lambda x: x[1])
        epicenter = winner[0]
        vote_count = winner[1]

        # Calculate consensus strength
        total_votes = sum(votes.values())
        consensus = (vote_count / total_votes * 100) if total_votes > 0 else 0

        # Check if matches ground truth
        matches_gt = False
        if self.ground_truth_sensor:
            matches_gt = self.ground_truth_sensor in epicenter

        print(f"\nVoting results:")
        print(f"  Total windows: {len(window_detections)}")
        print(f"  Total candidates: {len(votes)}")
        print(f"  Winner: {epicenter}")
        print(f"  Votes: {vote_count:.2f} / {total_votes:.2f} ({consensus:.1f}% consensus)")
        print(f"  Matches ground truth: {'✅ YES' if matches_gt else '❌ NO'}")

        print(f"\nTop 5 candidates:")
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        for i, (candidate, vote) in enumerate(sorted_votes[:5], 1):
            pct = (vote / total_votes * 100) if total_votes > 0 else 0
            match_str = "✅" if self.ground_truth_sensor and self.ground_truth_sensor in candidate else "❌"
            print(f"  {i}. {candidate:<50s} {vote:8.2f} votes ({pct:5.1f}%) {match_str}")

        return {
            'epicenter': epicenter,
            'votes': votes,
            'vote_count': float(vote_count),
            'total_votes': float(total_votes),
            'consensus': float(consensus),
            'matches_gt': matches_gt,
            'n_windows': len(window_detections),
            'n_candidates': len(votes)
        }

    def compare_strategies(self, window_detections: List[Dict]) -> pd.DataFrame:
        """
        Compare all voting strategies.

        Args:
            window_detections: List of per-window epicenter detections

        Returns:
            DataFrame comparing strategies
        """
        print("\n" + "="*80)
        print("STEP 3: STRATEGY COMPARISON")
        print("="*80)

        strategies = ['simple_majority', 'weighted_temporal', 'confidence', 'hybrid']
        results = []

        for strategy in strategies:
            result = self.apply_voting_strategy(window_detections, strategy)
            results.append({
                'strategy': strategy,
                'epicenter': result['epicenter'],
                'consensus': result['consensus'],
                'matches_gt': result['matches_gt'],
                'n_windows': result['n_windows']
            })

        comparison_df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("STRATEGY COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def generate_report(self, window_detections: List[Dict],
                       voting_results: Dict, output_dir: Path):
        """
        Generate comprehensive report.

        Args:
            window_detections: Per-window detections
            voting_results: Final voting results
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "voting_epicenter_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-WINDOW VOTING EPICENTER DETECTION REPORT\n")
            f.write("="*80 + "\n\n")

            # Per-window summary
            f.write("1. PER-WINDOW DETECTIONS\n")
            f.write("-"*80 + "\n\n")

            f.write(f"Total windows analyzed: {len(window_detections)}\n\n")

            if self.ground_truth_sensor:
                correct_windows = [w for w in window_detections if w['matches_gt']]
                window_accuracy = len(correct_windows) / len(window_detections) * 100 if window_detections else 0
                f.write(f"Per-window accuracy: {len(correct_windows)}/{len(window_detections)} "
                       f"({window_accuracy:.1f}%)\n\n")

            f.write("Top 10 window detections:\n")
            sorted_windows = sorted(window_detections, key=lambda x: x['score'], reverse=True)[:10]
            for w in sorted_windows:
                match_str = "✅" if w['matches_gt'] else "❌"
                f.write(f"  Window {w['window_idx']:3d}: {w['epicenter']:<45s} "
                       f"(Score: {w['score']:8.4f}) {match_str}\n")

            # Voting results
            f.write("\n\n2. VOTING RESULTS\n")
            f.write("-"*80 + "\n\n")

            f.write(f"Final epicenter: {voting_results['epicenter']}\n")
            f.write(f"Vote count: {voting_results['vote_count']:.2f} / {voting_results['total_votes']:.2f}\n")
            f.write(f"Consensus: {voting_results['consensus']:.1f}%\n")
            f.write(f"Matches ground truth: {'✅ YES' if voting_results['matches_gt'] else '❌ NO'}\n\n")

            f.write("All candidates with votes:\n")
            sorted_votes = sorted(voting_results['votes'].items(), key=lambda x: x[1], reverse=True)
            for candidate, vote in sorted_votes:
                pct = (vote / voting_results['total_votes'] * 100) if voting_results['total_votes'] > 0 else 0
                match_str = "✅" if self.ground_truth_sensor and self.ground_truth_sensor in candidate else "❌"
                f.write(f"  {candidate:<50s} {vote:8.2f} votes ({pct:5.1f}%) {match_str}\n")

        print(f"\n✅ Report saved to: {report_path}")

        # Save JSON results
        json_path = output_dir / "voting_results.json"
        with open(json_path, 'w') as f:
            json.dump({
                'per_window_detections': window_detections,
                'voting_results': voting_results
            }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)

        print(f"✅ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Window Voting Epicenter Detector")
    parser.add_argument('--golden-weights', required=True,
                       help="Path to golden weights CSV")
    parser.add_argument('--anomaly-weights', required=True,
                       help="Path to anomaly weights CSV")
    parser.add_argument('--ground-truth', default=None,
                       help="Path to ground truth JSON (optional)")
    parser.add_argument('--output-dir', default='results/voting_epicenter',
                       help="Output directory for results")
    parser.add_argument('--threshold', type=float, default=0.01,
                       help="Threshold for significant edge changes")
    parser.add_argument('--strategy', type=str, default='hybrid',
                       choices=['simple_majority', 'weighted_temporal', 'confidence', 'hybrid'],
                       help="Voting strategy to use")
    parser.add_argument('--compare-strategies', action='store_true',
                       help="Compare all voting strategies")

    args = parser.parse_args()

    print("="*80)
    print("MULTI-WINDOW VOTING EPICENTER DETECTOR")
    print("="*80)

    # Initialize detector
    detector = MultiWindowVotingDetector(
        args.golden_weights,
        args.anomaly_weights,
        args.ground_truth,
        args.threshold
    )

    # Step 1: Detect epicenter per window
    window_detections = detector.detect_epicenter_per_window()

    if len(window_detections) == 0:
        print("\n❌ No windows detected!")
        return

    # Step 2: Apply voting
    if args.compare_strategies:
        comparison_df = detector.compare_strategies(window_detections)

        # Save comparison
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_dir / "strategy_comparison.csv", index=False)
        print(f"\n✅ Strategy comparison saved to: {output_dir / 'strategy_comparison.csv'}")

    voting_results = detector.apply_voting_strategy(window_detections, args.strategy)

    # Step 3: Generate report
    detector.generate_report(window_detections, voting_results, args.output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
