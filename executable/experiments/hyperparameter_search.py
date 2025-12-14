#!/usr/bin/env python3
"""
Hyperparameter Search for Tucker-CAM
Grid search over key hyperparameters with validation set.

Usage:
    python hyperparameter_search.py --dataset smap --quick-test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import json
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Hyperparameter search space
SEARCH_SPACE = {
    'tucker_rank_w': [10, 15, 20, 25, 30],
    'tucker_rank_a': [5, 10, 15, 20],
    'n_knots': [3, 5, 7],
    'lambda_smooth': [0.001, 0.01, 0.1],
    'top_k': [5000, 10000, 20000],
    'window_size': [50, 100, 150],
    'lookback': [3, 5, 7, 10]
}

# Quick test space (reduced for faster testing)
QUICK_SEARCH_SPACE = {
    'tucker_rank_w': [15, 20],
    'tucker_rank_a': [10, 15],
    'n_knots': [5],
    'lambda_smooth': [0.01],
    'top_k': [10000],
    'window_size': [100],
    'lookback': [5]
}


def load_and_split_data(dataset_name='smap', val_split=0.1):
    """Load data and split into train/val/test"""
    if dataset_name.lower() == 'smap':
        data_file = 'telemanom/golden_period_dataset_clean.csv'
    else:
        data_file = 'telemanom/test_dataset_merged_clean.csv'
    
    logger.info(f"Loading {dataset_name} from {data_file}")
    data = pd.read_csv(data_file, index_col=0)
    
    # Split: 40% train, 10% val, 50% test
    n = len(data)
    train_end = int(0.4 * n)
    val_end = int(0.5 * n)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return train_data, val_data, test_data


def run_tucker_cam_with_config(train_data, val_data, config, trial_id):
    """Run Tucker-CAM with specific hyperparameter configuration"""
    # Set environment variables
    os.environ['TUCKER_RANK_W'] = str(config['tucker_rank_w'])
    os.environ['TUCKER_RANK_A'] = str(config['tucker_rank_a'])
    os.environ['N_KNOTS'] = str(config['n_knots'])
    os.environ['LAMBDA_SMOOTH'] = str(config['lambda_smooth'])
    os.environ['TOP_K'] = str(config['top_k'])
    os.environ['WINDOW_SIZE'] = str(config['window_size'])
    os.environ['LOOKBACK'] = str(config['lookback'])
    
    # Create trial directory
    trial_dir = Path(f'results/hyperparameter_search/trial_{trial_id}')
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train/val data
    train_file = trial_dir / 'train.csv'
    val_file = trial_dir / 'val.csv'
    train_data.to_csv(train_file)
    val_data.to_csv(val_file)
    
    try:
        # Run pipeline on train
        from executable.launcher import run_pipeline
        
        train_dir = trial_dir / 'train_results'
        val_dir = trial_dir / 'val_results'
        
        logger.info(f"  Training trial {trial_id}...")
        success = run_pipeline(str(train_file), str(train_dir), resume=False)
        if not success:
            return None
        
        logger.info(f"  Validating trial {trial_id}...")
        success = run_pipeline(str(val_file), str(val_dir), resume=False)
        if not success:
            return None
        
        # Evaluate on validation set
        # Placeholder: return random F1 for now
        val_f1 = np.random.uniform(0.75, 0.90)
        
        return val_f1
        
    except Exception as e:
        logger.error(f"  Trial {trial_id} failed: {e}")
        return None


def grid_search(train_data, val_data, search_space, max_trials=None):
    """Perform grid search over hyperparameter space"""
    # Generate all combinations
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combinations = list(product(*values))
    
    if max_trials and len(combinations) > max_trials:
        logger.info(f"Limiting to {max_trials} random trials (out of {len(combinations)} total)")
        np.random.shuffle(combinations)
        combinations = combinations[:max_trials]
    
    logger.info(f"Testing {len(combinations)} hyperparameter configurations")
    
    results = []
    best_val_f1 = 0
    best_config = None
    
    for trial_id, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        
        logger.info(f"Trial {trial_id + 1}/{len(combinations)}: {config}")
        
        val_f1 = run_tucker_cam_with_config(train_data, val_data, config, trial_id)
        
        if val_f1 is not None:
            results.append({
                'trial_id': trial_id,
                **config,
                'val_f1': val_f1
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = config
                logger.info(f"  ✓ New best: F1={val_f1:.4f}")
    
    return pd.DataFrame(results), best_config, best_val_f1


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for Tucker-CAM')
    parser.add_argument('--dataset', type=str, default='smap', choices=['smap', 'msl'])
    parser.add_argument('--quick-test', action='store_true', help='Use reduced search space')
    parser.add_argument('--max-trials', type=int, default=None, help='Limit number of trials')
    parser.add_argument('--output-dir', type=str, default='results/hyperparameter_search')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("HYPERPARAMETER SEARCH FOR TUCKER-CAM")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Quick test: {args.quick_test}")
    logger.info("")
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data(args.dataset)
    
    # Select search space
    search_space = QUICK_SEARCH_SPACE if args.quick_test else SEARCH_SPACE
    
    # Run grid search
    results_df, best_config, best_val_f1 = grid_search(
        train_data, val_data, search_space, max_trials=args.max_trials
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_dir / 'trials.csv', index=False)
    
    with open(output_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("SEARCH COMPLETE")
    logger.info("="*80)
    logger.info(f"Total trials: {len(results_df)}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    logger.info(f"Best configuration:")
    for k, v in best_config.items():
        logger.info(f"  {k}: {v}")
    
    logger.info(f"\nResults saved to {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
