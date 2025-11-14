#!/usr/bin/env python3
"""
Adaptive Knot Selection for B-Splines
======================================

Automatically determines optimal number of knots based on:
1. Dataset size (more samples = more knots possible)
2. Number of variables (high-dimensional = fewer knots to avoid overfitting)
3. Data complexity (variance, autocorrelation)
4. Window size (smaller windows = fewer knots)

Rule of thumb:
- Too few knots: Underfitting (can't capture nonlinearity)
- Too many knots: Overfitting, computational cost, numerical instability
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def estimate_data_complexity(df: pd.DataFrame, max_samples: int = 1000) -> float:
    """
    Estimate data complexity based on variance and autocorrelation.
    
    Returns:
        Complexity score (0-1): Higher = more complex = needs more knots
    """
    # Sample if dataset is large
    if len(df) > max_samples:
        df_sample = df.sample(n=max_samples, random_state=42)
    else:
        df_sample = df
    
    # Measure 1: Normalized variance (how much the data varies)
    # Higher variance = more complex patterns
    normalized_var = np.mean(df_sample.var() / (df_sample.std() + 1e-10))
    var_score = min(normalized_var / 2.0, 1.0)  # Normalize to [0, 1]
    
    # Measure 2: Autocorrelation (temporal smoothness)
    # Lower autocorrelation = more erratic = needs more flexibility
    autocorrs = []
    for col in df_sample.columns[:min(10, len(df_sample.columns))]:  # Sample first 10 variables
        series = df_sample[col].values
        if len(series) > 1:
            # Lag-1 autocorrelation
            autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
            if not np.isnan(autocorr):
                autocorrs.append(abs(autocorr))
    
    if autocorrs:
        avg_autocorr = np.mean(autocorrs)
        # High autocorr = smooth = fewer knots needed
        # Low autocorr = erratic = more knots needed
        autocorr_score = 1.0 - avg_autocorr  # Invert: low autocorr = high complexity
    else:
        autocorr_score = 0.5  # Default if can't compute
    
    # Combined complexity score
    complexity = 0.6 * var_score + 0.4 * autocorr_score
    
    return complexity


def select_adaptive_knots(
    n_samples: int,
    n_variables: int,
    window_size: int = None,
    data_complexity: float = None,
    df: pd.DataFrame = None,
    min_knots: int = 5,
    max_knots: int = 15
) -> Tuple[int, str]:
    """
    Automatically select optimal number of B-spline knots.
    
    Args:
        n_samples: Total number of samples in dataset
        n_variables: Number of variables (features)
        window_size: Size of rolling window (if applicable)
        data_complexity: Pre-computed complexity score (0-1), or None to compute
        df: DataFrame to compute complexity from (if data_complexity is None)
        min_knots: Minimum allowed knots (default 3)
        max_knots: Maximum allowed knots (default 15)
    
    Returns:
        n_knots: Recommended number of knots
        reasoning: Human-readable explanation
    """
    reasoning_parts = []
    
    # Base knot selection on effective sample size
    effective_samples = window_size if window_size is not None else n_samples
    
    # Rule 1: Sample size constraint
    # Heuristic: knots ~ sqrt(samples) / 10, bounded
    if effective_samples < 50:
        base_knots = 3
        reasoning_parts.append(f"Very small sample size ({effective_samples})")
    elif effective_samples < 100:
        base_knots = 4
        reasoning_parts.append(f"Small sample size ({effective_samples})")
    elif effective_samples < 200:
        base_knots = 5
        reasoning_parts.append(f"Moderate sample size ({effective_samples})")
    elif effective_samples < 500:
        base_knots = 6
        reasoning_parts.append(f"Good sample size ({effective_samples})")
    elif effective_samples < 1000:
        base_knots = 8
        reasoning_parts.append(f"Large sample size ({effective_samples})")
    else:
        base_knots = 10
        reasoning_parts.append(f"Very large sample size ({effective_samples})")
    
    # Rule 2: Dimensionality adjustment
    # High dimensions = risk of overfitting = reduce knots
    if n_variables < 50:
        dim_adjustment = 0
        reasoning_parts.append("low dimensionality")
    elif n_variables < 500:
        dim_adjustment = -1
        reasoning_parts.append(f"moderate dimensionality ({n_variables} vars)")
    elif n_variables < 2000:
        dim_adjustment = -2
        reasoning_parts.append(f"high dimensionality ({n_variables} vars)")
    else:
        dim_adjustment = -3
        reasoning_parts.append(f"very high dimensionality ({n_variables} vars)")
    
    # Rule 3: Data complexity adjustment
    if data_complexity is None and df is not None:
        data_complexity = estimate_data_complexity(df)
        reasoning_parts.append(f"estimated complexity={data_complexity:.2f}")
    
    if data_complexity is not None:
        if data_complexity > 0.7:
            complexity_adjustment = +2
            reasoning_parts.append("high complexity (needs more flexibility)")
        elif data_complexity > 0.5:
            complexity_adjustment = +1
            reasoning_parts.append("moderate complexity")
        elif data_complexity > 0.3:
            complexity_adjustment = 0
            reasoning_parts.append("low complexity")
        else:
            complexity_adjustment = -1
            reasoning_parts.append("very low complexity (smooth data)")
    else:
        complexity_adjustment = 0
    
    # Combine all factors
    n_knots = base_knots + dim_adjustment + complexity_adjustment
    
    # Enforce bounds
    n_knots = max(min_knots, min(n_knots, max_knots))
    
    # Additional constraint: knots should not exceed samples / 10
    max_knots_from_samples = max(min_knots, effective_samples // 10)
    if n_knots > max_knots_from_samples:
        n_knots = max_knots_from_samples
        reasoning_parts.append(f"limited by sample size constraint ({max_knots_from_samples})")
    
    reasoning = f"Selected {n_knots} knots based on: " + ", ".join(reasoning_parts)
    
    return n_knots, reasoning


def get_knots_for_dataset(
    df: pd.DataFrame,
    window_size: int = None,
    min_knots: int = 5,
    max_knots: int = 15
) -> Tuple[int, str]:
    """
    Convenience function to select knots for a given dataset.
    
    Args:
        df: Input DataFrame
        window_size: Rolling window size (optional)
        min_knots: Minimum knots
        max_knots: Maximum knots
    
    Returns:
        n_knots: Recommended number of knots
        reasoning: Explanation
    """
    n_samples = len(df)
    n_variables = len(df.columns)
    
    # Compute complexity from data
    complexity = estimate_data_complexity(df)
    
    n_knots, reasoning = select_adaptive_knots(
        n_samples=n_samples,
        n_variables=n_variables,
        window_size=window_size,
        data_complexity=complexity,
        min_knots=min_knots,
        max_knots=max_knots
    )
    
    logger.info(f"Adaptive knot selection: {reasoning}")
    
    return n_knots, reasoning


# Quick reference table for manual selection
KNOT_SELECTION_GUIDE = """
Knot Selection Guide:
=====================

Sample Size (Window Size):
  < 50 samples:     3-4 knots
  50-100 samples:   4-5 knots
  100-200 samples:  5-6 knots
  200-500 samples:  6-8 knots
  500-1000 samples: 8-10 knots
  > 1000 samples:   10-12 knots

Dimensionality Adjustment:
  < 50 variables:     +0 knots
  50-500 variables:   -1 knots
  500-2000 variables: -2 knots
  > 2000 variables:   -3 knots

Data Complexity:
  Smooth/stationary:     -1 knots
  Moderate variation:     0 knots
  High variation/noise:  +1 knots
  Very complex/chaotic:  +2 knots

Safety Bounds:
  Minimum: 3 knots (linear + some curvature)
  Maximum: 15 knots (diminishing returns, numerical issues)
  
For your dataset (2890 vars, 100 samples/window):
  Base: 5 knots (100 samples)
  Adjustment: -2 (high dim)
  → Recommended: 3-4 knots
"""


if __name__ == "__main__":
    # Example usage
    print(KNOT_SELECTION_GUIDE)
    
    # Test with synthetic data
    np.random.seed(42)
    
    # Test 1: Small, low-dimensional dataset
    df_small = pd.DataFrame(np.random.randn(100, 10))
    n_knots, reason = get_knots_for_dataset(df_small, window_size=100)
    print(f"\nTest 1 (100 samples, 10 vars): {n_knots} knots")
    print(f"  Reasoning: {reason}")
    
    # Test 2: Large, high-dimensional dataset (like yours)
    df_large = pd.DataFrame(np.random.randn(8640, 2890))
    n_knots, reason = get_knots_for_dataset(df_large, window_size=100)
    print(f"\nTest 2 (8640 samples, 2890 vars, window=100): {n_knots} knots")
    print(f"  Reasoning: {reason}")
    
    # Test 3: Very complex data
    df_complex = pd.DataFrame(np.random.randn(1000, 50) * 10 + np.random.choice([0, 100], (1000, 50)))
    n_knots, reason = get_knots_for_dataset(df_complex, window_size=200)
    print(f"\nTest 3 (1000 samples, 50 vars, high variance): {n_knots} knots")
    print(f"  Reasoning: {reason}")
