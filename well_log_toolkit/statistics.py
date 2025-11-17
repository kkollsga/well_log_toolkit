"""
Statistical functions for well log data with depth-weighted calculations.

This module provides both weighted (by depth intervals) and arithmetic (unweighted)
statistical functions for well log analysis.
"""

import numpy as np
from typing import Tuple, Optional


def compute_intervals(depth: np.ndarray) -> np.ndarray:
    """
    Compute depth intervals (thicknesses) for each sample point.

    Uses midpoint method: each sample represents the interval from halfway
    to the previous sample to halfway to the next sample.

    Parameters
    ----------
    depth : np.ndarray
        Depth values (must be sorted ascending)

    Returns
    -------
    np.ndarray
        Interval thickness for each depth point

    Examples
    --------
    >>> depth = np.array([1500, 1501, 1505])
    >>> compute_intervals(depth)
    array([0.5, 2.5, 2.0])

    The intervals are:
    - 1500: from 1500 to 1500.5 = 0.5m (first point gets half interval to next)
    - 1501: from 1500.5 to 1503 = 2.5m (midpoint to midpoint)
    - 1505: from 1503 to 1505 = 2.0m (last point gets half interval from prev)
    """
    if len(depth) == 0:
        return np.array([])

    if len(depth) == 1:
        return np.array([1.0])  # Default interval for single point

    intervals = np.zeros(len(depth))

    # First point: half interval to next point
    intervals[0] = (depth[1] - depth[0]) / 2.0

    # Middle points: midpoint to midpoint
    for i in range(1, len(depth) - 1):
        lower_mid = (depth[i] + depth[i-1]) / 2.0
        upper_mid = (depth[i+1] + depth[i]) / 2.0
        intervals[i] = upper_mid - lower_mid

    # Last point: half interval from previous point
    intervals[-1] = (depth[-1] - depth[-2]) / 2.0

    return intervals


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute depth-weighted mean.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray
        Weights (depth intervals) for each value

    Returns
    -------
    float
        Weighted mean, or NaN if no valid values

    Examples
    --------
    >>> values = np.array([0, 1, 0])  # NTG values
    >>> weights = np.array([0.5, 2.5, 2.0])  # depth intervals
    >>> weighted_mean(values, weights)
    0.5  # (0*0.5 + 1*2.5 + 0*2.0) / (0.5 + 2.5 + 2.0) = 2.5/5.0
    """
    valid_mask = ~np.isnan(values) & ~np.isnan(weights)
    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    if len(valid_values) == 0 or np.sum(valid_weights) == 0:
        return np.nan

    return float(np.sum(valid_values * valid_weights) / np.sum(valid_weights))


def weighted_sum(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted sum (e.g., for cumulative thickness).

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray
        Weights (depth intervals) for each value

    Returns
    -------
    float
        Weighted sum, or NaN if no valid values

    Examples
    --------
    For NTG flag (0 or 1), this gives net thickness:
    >>> values = np.array([0, 1, 0])  # NTG values
    >>> weights = np.array([0.5, 2.5, 2.0])  # depth intervals
    >>> weighted_sum(values, weights)
    2.5  # Only the 1 contributes: 1 * 2.5 = 2.5m of net
    """
    valid_mask = ~np.isnan(values) & ~np.isnan(weights)
    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    if len(valid_values) == 0:
        return np.nan

    return float(np.sum(valid_values * valid_weights))


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted standard deviation.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray
        Weights (depth intervals) for each value

    Returns
    -------
    float
        Weighted standard deviation, or NaN if insufficient valid values
    """
    valid_mask = ~np.isnan(values) & ~np.isnan(weights)
    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    if len(valid_values) < 2 or np.sum(valid_weights) == 0:
        return np.nan

    w_mean = weighted_mean(values, weights)
    variance = np.sum(valid_weights * (valid_values - w_mean) ** 2) / np.sum(valid_weights)
    return float(np.sqrt(variance))


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float
) -> float:
    """
    Compute weighted percentile.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray
        Weights (depth intervals) for each value
    percentile : float
        Percentile to compute (0-100)

    Returns
    -------
    float
        Weighted percentile value, or NaN if no valid values
    """
    valid_mask = ~np.isnan(values) & ~np.isnan(weights)
    valid_values = values[valid_mask]
    valid_weights = weights[valid_mask]

    if len(valid_values) == 0:
        return np.nan

    # Sort by values
    sort_idx = np.argsort(valid_values)
    sorted_values = valid_values[sort_idx]
    sorted_weights = valid_weights[sort_idx]

    # Compute cumulative weight
    cumulative_weight = np.cumsum(sorted_weights)
    total_weight = cumulative_weight[-1]

    # Find percentile position
    target_weight = (percentile / 100.0) * total_weight

    # Linear interpolation
    idx = np.searchsorted(cumulative_weight, target_weight)

    if idx == 0:
        return float(sorted_values[0])
    elif idx >= len(sorted_values):
        return float(sorted_values[-1])
    else:
        # Interpolate between idx-1 and idx
        w_below = cumulative_weight[idx - 1]
        w_above = cumulative_weight[idx]
        fraction = (target_weight - w_below) / (w_above - w_below)
        return float(sorted_values[idx - 1] + fraction * (sorted_values[idx] - sorted_values[idx - 1]))


def arithmetic_mean(values: np.ndarray) -> float:
    """
    Compute simple arithmetic mean (unweighted).

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)

    Returns
    -------
    float
        Arithmetic mean, or NaN if no valid values
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid))


def arithmetic_sum(values: np.ndarray) -> float:
    """
    Compute simple sum (unweighted).

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)

    Returns
    -------
    float
        Sum of non-NaN values, or NaN if no valid values
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float(np.sum(valid))


def arithmetic_std(values: np.ndarray) -> float:
    """
    Compute simple standard deviation (unweighted).

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)

    Returns
    -------
    float
        Standard deviation, or NaN if insufficient valid values
    """
    valid = values[~np.isnan(values)]
    if len(valid) < 2:
        return np.nan
    return float(np.std(valid))


def compute_all_statistics(
    values: np.ndarray,
    depth: np.ndarray
) -> dict:
    """
    Compute comprehensive statistics including both weighted and arithmetic measures.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    depth : np.ndarray
        Depth values corresponding to values

    Returns
    -------
    dict
        Dictionary containing:
        - weighted_mean: Depth-weighted mean
        - weighted_sum: Depth-weighted sum (useful for cumulative thickness)
        - weighted_std: Depth-weighted standard deviation
        - weighted_p10, weighted_p50, weighted_p90: Depth-weighted percentiles
        - arithmetic_mean: Simple arithmetic mean
        - arithmetic_sum: Simple sum
        - arithmetic_std: Simple standard deviation
        - count: Number of non-NaN values
        - depth_samples: Total number of samples
        - depth_thickness: Total thickness covered
        - min: Minimum value
        - max: Maximum value
    """
    intervals = compute_intervals(depth)
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    valid_intervals = intervals[valid_mask]

    return {
        # Depth-weighted statistics (preferred for well log analysis)
        'weighted_mean': weighted_mean(values, intervals),
        'weighted_sum': weighted_sum(values, intervals),
        'weighted_std': weighted_std(values, intervals),
        'weighted_p10': weighted_percentile(values, intervals, 10),
        'weighted_p50': weighted_percentile(values, intervals, 50),
        'weighted_p90': weighted_percentile(values, intervals, 90),

        # Arithmetic statistics (sample-based)
        'arithmetic_mean': arithmetic_mean(values),
        'arithmetic_sum': arithmetic_sum(values),
        'arithmetic_std': arithmetic_std(values),

        # Counts and ranges
        'count': int(len(valid_values)),
        'depth_samples': int(len(values)),
        'depth_thickness': float(np.sum(valid_intervals)) if len(valid_intervals) > 0 else 0.0,
        'min': float(np.min(valid_values)) if len(valid_values) > 0 else np.nan,
        'max': float(np.max(valid_values)) if len(valid_values) > 0 else np.nan,
    }
