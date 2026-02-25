#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np

def parse_gt_file(file_path):
    anomalies = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(':')
            if len(parts) != 2: continue
            range_str, dims_str = parts
            start, end = map(int, range_str.split('-'))
            dims = [int(d) for d in dims_str.split(',')]
            anomalies.append({'start_time': start, 'end_time': end, 'culprits': dims})
    return anomalies

def process_anomalies(anomalies, df_gold, df_curr, stride):
    results = []
    target_dims = 38 # SMD
    
    # Pre-compute Golden Strengths
    gold_in_strength = df_gold.groupby('j')['weight'].sum().to_dict()
    gold_out_strength = df_gold.groupby('i')['weight'].sum().to_dict()
    
    # Group Current by window
    grouped = df_curr.groupby('window_idx')
    
    found_windows = 0
    
    for anom in anomalies:
        win_idx = anom['start_time'] // stride
        
        if win_idx not in grouped.groups:
            # print(f"Window {win_idx}: No Data")
            continue
            
        found_windows += 1
        df_win = grouped.get_group(win_idx)
        
        # Calculate Strengths
        curr_in = df_win.groupby('j')['weight'].sum().to_dict()
        curr_out = df_win.groupby('i')['weight'].sum().to_dict()
        
        node_scores = {}
        for node in range(target_dims):
            g_in = gold_in_strength.get(node, 0.0)
            c_in = curr_in.get(node, 0.0)
            g_out = gold_out_strength.get(node, 0.0)
            c_out = curr_out.get(node, 0.0)
            
            score = abs(c_in - g_in) + abs(c_out - g_out)
            node_scores[node] = score
            
        ranked_nodes = sorted(node_scores.keys(), key=lambda x: node_scores[x], reverse=True)
        
        ranks = []
        for c in anom['culprits']:
            if c in ranked_nodes:
                ranks.append(ranked_nodes.index(c) + 1)
            else:
                ranks.append(999)
       
        # print(f"Window {win_idx} (Time {anom['start_time']}): Ranks {ranks}")
        results.extend(ranks)

    return results, found_windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--golden', required=True)
    parser.add_argument('--current', required=True)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()
    
    print("Loading data...")
    anomalies = parse_gt_file(args.gt)
    df_gold = pd.read_csv(args.golden)
    df_curr = pd.read_csv(args.current)
    
    print(f"Loaded {len(anomalies)} anomalies and weights for {df_curr['window_idx'].nunique()} windows.")
    
    results, found_count = process_anomalies(anomalies, df_gold, df_curr, args.stride)
    
    if not results:
        print("No valid comparisons made.")
        return

    r1 = sum(1 for r in results if r <= 1) / len(results)
    r3 = sum(1 for r in results if r <= 3) / len(results)
    r5 = sum(1 for r in results if r <= 5) / len(results)
    
    print("\n" + "="*60)
    print("RCA BENCHMARK RESULTS (Averaged Bagging)")
    print("="*60)
    print(f"Windows Evaluated:         {found_count} / {len(anomalies)}")
    print(f"Total Culprits Evaluated:  {len(results)}")
    print(f"Recall@1: {r1:.4f}")
    print(f"Recall@3: {r3:.4f}")
    print(f"Recall@5: {r5:.4f}")
    print("="*60)
    print("Average Rank:", np.mean(results))

if __name__ == "__main__":
    main()
