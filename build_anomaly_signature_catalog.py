#!/usr/bin/env python3
"""
Build Anomaly Signature Catalog
Analyzes Tucker-CAM outputs for each anomaly to create a comprehensive catalog
showing causal graph changes per anomaly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def load_weights(weights_file):
    """Load Tucker-CAM weights from CSV."""
    if not Path(weights_file).exists():
        return None
    
    df = pd.read_csv(weights_file)
    return df

def analyze_causal_structure(weights_df):
    """
    Analyze causal structure from Tucker-CAM weights.
    Returns key metrics about the causal graph.
    """
    if weights_df is None or len(weights_df) == 0:
        return None
    
    metrics = {
        'total_edges': len(weights_df),
        'unique_windows': weights_df['window'].nunique() if 'window' in weights_df.columns else 0,
        'mean_weight': weights_df['weight'].mean(),
        'max_weight': weights_df['weight'].max(),
        'min_weight': weights_df['weight'].min(),
        'std_weight': weights_df['weight'].std(),
        'num_source_vars': weights_df['source'].nunique() if 'source' in weights_df.columns else 0,
        'num_target_vars': weights_df['target'].nunique() if 'target' in weights_df.columns else 0,
    }
    
    # Analyze lags
    if 'lag' in weights_df.columns:
        metrics['mean_lag'] = weights_df['lag'].mean()
        metrics['max_lag'] = weights_df['lag'].max()
        metrics['contemporaneous_edges'] = (weights_df['lag'] == 0).sum()
        metrics['lagged_edges'] = (weights_df['lag'] > 0).sum()
    
    # Top edges
    top_edges = weights_df.nlargest(10, 'weight')[['source', 'target', 'lag', 'weight']].to_dict('records')
    metrics['top_10_edges'] = top_edges
    
    # Window evolution
    if 'window' in weights_df.columns:
        window_stats = weights_df.groupby('window')['weight'].agg(['count', 'mean', 'max']).reset_index()
        metrics['window_stats'] = window_stats.to_dict('records')
    
    return metrics

def compare_with_baseline(anomaly_weights, baseline_weights):
    """
    Compare anomaly causal structure with baseline (Golden).
    Identifies new edges, removed edges, and weight changes.
    """
    if anomaly_weights is None or baseline_weights is None:
        return None
    
    # Create edge identifiers
    anomaly_weights['edge_id'] = anomaly_weights['source'] + '_' + anomaly_weights['target'] + '_lag' + anomaly_weights['lag'].astype(str)
    baseline_weights['edge_id'] = baseline_weights['source'] + '_' + baseline_weights['target'] + '_lag' + baseline_weights['lag'].astype(str)
    
    # Edge sets
    anomaly_edges = set(anomaly_weights['edge_id'])
    baseline_edges = set(baseline_weights['edge_id'])
    
    # New edges (in anomaly, not in baseline)
    new_edges = anomaly_edges - baseline_edges
    new_edges_df = anomaly_weights[anomaly_weights['edge_id'].isin(new_edges)]
    
    # Removed edges (in baseline, not in anomaly)
    removed_edges = baseline_edges - anomaly_edges
    removed_edges_df = baseline_weights[baseline_weights['edge_id'].isin(removed_edges)]
    
    # Common edges (weight changes)
    common_edges = anomaly_edges & baseline_edges
    common_anomaly = anomaly_weights[anomaly_weights['edge_id'].isin(common_edges)].set_index('edge_id')
    common_baseline = baseline_weights[baseline_weights['edge_id'].isin(common_edges)].set_index('edge_id')
    
    # Calculate weight changes
    weight_changes = common_anomaly['weight'] - common_baseline['weight']
    weight_changes_df = pd.DataFrame({
        'edge_id': weight_changes.index,
        'baseline_weight': common_baseline['weight'],
        'anomaly_weight': common_anomaly['weight'],
        'weight_change': weight_changes,
        'pct_change': (weight_changes / common_baseline['weight']) * 100
    })
    
    comparison = {
        'num_new_edges': len(new_edges),
        'num_removed_edges': len(removed_edges),
        'num_common_edges': len(common_edges),
        'top_new_edges': new_edges_df.nlargest(10, 'weight')[['source', 'target', 'lag', 'weight']].to_dict('records') if len(new_edges_df) > 0 else [],
        'top_removed_edges': removed_edges_df.nlargest(10, 'weight')[['source', 'target', 'lag', 'weight']].to_dict('records') if len(removed_edges_df) > 0 else [],
        'top_weight_increases': weight_changes_df.nlargest(10, 'weight_change').to_dict('records') if len(weight_changes_df) > 0 else [],
        'top_weight_decreases': weight_changes_df.nsmallest(10, 'weight_change').to_dict('records') if len(weight_changes_df) > 0 else [],
    }
    
    return comparison

def build_catalog(results_dir, anomaly_catalog_csv, output_csv, golden_baseline_dir=None):
    """
    Build comprehensive anomaly signature catalog.
    
    Args:
        results_dir: Directory containing Tucker-CAM results for each anomaly
        anomaly_catalog_csv: Path to anomaly catalog (from extraction)
        output_csv: Path to save signature catalog
        golden_baseline_dir: Directory containing Golden baseline Tucker-CAM results
    """
    
    print("="*80)
    print("BUILDING ANOMALY SIGNATURE CATALOG")
    print("="*80)
    
    # Load anomaly catalog
    print(f"\n[1/4] Loading anomaly catalog from: {anomaly_catalog_csv}")
    catalog_df = pd.read_csv(anomaly_catalog_csv)
    print(f"  ✓ Found {len(catalog_df)} anomalies")
    
    # Load baseline (if available)
    baseline_weights = None
    if golden_baseline_dir:
        baseline_weights_file = Path(golden_baseline_dir) / "weights" / "weights_enhanced.csv"
        if baseline_weights_file.exists():
            print(f"\n[2/4] Loading Golden baseline from: {baseline_weights_file}")
            baseline_weights = load_weights(baseline_weights_file)
            print(f"  ✓ Loaded {len(baseline_weights)} baseline edges")
        else:
            print(f"\n[2/4] Golden baseline not found: {baseline_weights_file}")
    else:
        print(f"\n[2/4] Skipping baseline (not provided)")
    
    # Process each anomaly
    print(f"\n[3/4] Analyzing causal signatures...")
    print("-"*80)
    
    signatures = []
    
    for idx, row in catalog_df.iterrows():
        filename = row['filename']
        anomaly_name = Path(filename).stem
        channel = row['channel']
        spacecraft = row['spacecraft']
        sequence_idx = row['sequence_idx']
        anomaly_class = row['anomaly_class']
        
        print(f"\n{idx+1}/{len(catalog_df)}: {anomaly_name}")
        
        # Load Tucker-CAM results
        weights_file = Path(results_dir) / anomaly_name / "weights" / "weights_enhanced.csv"
        
        if not weights_file.exists():
            print(f"  ⚠ Weights file not found: {weights_file}")
            signatures.append({
                'anomaly_name': anomaly_name,
                'channel': channel,
                'spacecraft': spacecraft,
                'sequence_idx': sequence_idx,
                'anomaly_class': anomaly_class,
                'status': 'missing_weights'
            })
            continue
        
        # Load weights
        weights_df = load_weights(weights_file)
        print(f"  ✓ Loaded {len(weights_df)} edges")
        
        # Analyze structure
        metrics = analyze_causal_structure(weights_df)
        
        # Compare with baseline
        comparison = None
        if baseline_weights is not None:
            comparison = compare_with_baseline(weights_df, baseline_weights)
            if comparison:
                print(f"    → New edges: {comparison['num_new_edges']}")
                print(f"    → Removed edges: {comparison['num_removed_edges']}")
                print(f"    → Changed edges: {comparison['num_common_edges']}")
        
        # Build signature entry
        signature = {
            'anomaly_name': anomaly_name,
            'channel': channel,
            'spacecraft': spacecraft,
            'sequence_idx': sequence_idx,
            'anomaly_class': anomaly_class,
            'anomaly_start': row['anomaly_start'],
            'anomaly_end': row['anomaly_end'],
            'anomaly_length': row['anomaly_length'],
            'status': 'success',
            # Causal structure metrics
            'total_edges': metrics['total_edges'],
            'unique_windows': metrics['unique_windows'],
            'mean_weight': metrics['mean_weight'],
            'max_weight': metrics['max_weight'],
            'std_weight': metrics['std_weight'],
            'num_source_vars': metrics['num_source_vars'],
            'num_target_vars': metrics['num_target_vars'],
            'mean_lag': metrics.get('mean_lag', np.nan),
            'max_lag': metrics.get('max_lag', np.nan),
            'contemporaneous_edges': metrics.get('contemporaneous_edges', 0),
            'lagged_edges': metrics.get('lagged_edges', 0),
        }
        
        # Add comparison metrics
        if comparison:
            signature.update({
                'new_edges': comparison['num_new_edges'],
                'removed_edges': comparison['num_removed_edges'],
                'common_edges': comparison['num_common_edges'],
            })
        
        signatures.append(signature)
    
    # Create catalog DataFrame
    print(f"\n[4/4] Creating signature catalog...")
    signatures_df = pd.DataFrame(signatures)
    
    # Save
    signatures_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved: {output_csv}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"CATALOG COMPLETE")
    print(f"{'='*80}")
    print(f"Total signatures: {len(signatures_df)}")
    print(f"Successful: {(signatures_df['status'] == 'success').sum()}")
    print(f"Missing weights: {(signatures_df['status'] == 'missing_weights').sum()}")
    
    if 'anomaly_class' in signatures_df.columns:
        print(f"\nBy anomaly class:")
        for cls, count in signatures_df['anomaly_class'].value_counts().items():
            print(f"  {cls}: {count}")
    
    if 'total_edges' in signatures_df.columns:
        successful = signatures_df[signatures_df['status'] == 'success']
        if len(successful) > 0:
            print(f"\nEdge statistics:")
            print(f"  Mean edges per anomaly: {successful['total_edges'].mean():.1f}")
            print(f"  Min edges: {successful['total_edges'].min()}")
            print(f"  Max edges: {successful['total_edges'].max()}")
    
    return signatures_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build anomaly signature catalog")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing Tucker-CAM results for each anomaly"
    )
    parser.add_argument(
        "--anomaly-catalog",
        required=True,
        help="Path to anomaly catalog CSV (from extraction)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for signature catalog CSV"
    )
    parser.add_argument(
        "--golden-baseline",
        default=None,
        help="Directory containing Golden baseline Tucker-CAM results (optional)"
    )
    
    args = parser.parse_args()
    
    catalog = build_catalog(
        results_dir=args.results_dir,
        anomaly_catalog_csv=args.anomaly_catalog,
        output_csv=args.output,
        golden_baseline_dir=args.golden_baseline
    )
    
    print(f"\n✓ Done! Signature catalog saved to: {args.output}")
