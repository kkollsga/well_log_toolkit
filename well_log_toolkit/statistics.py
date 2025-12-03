"""
Statistical functions for well log data with depth-weighted calculations.

This module provides both weighted (by depth intervals) and arithmetic (unweighted)
statistical functions for well log analysis.
"""

import numpy as np
from typing import Optional, Union


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

    # Middle points: midpoint to midpoint (vectorized)
    # lower_mid[i] = (depth[i] + depth[i-1]) / 2.0
    # upper_mid[i] = (depth[i+1] + depth[i]) / 2.0
    # intervals[i] = upper_mid[i] - lower_mid[i]
    # Simplifies to: intervals[i] = (depth[i+1] - depth[i-1]) / 2.0
    intervals[1:-1] = (depth[2:] - depth[:-2]) / 2.0

    # Last point: half interval from previous point
    intervals[-1] = (depth[-1] - depth[-2]) / 2.0

    return intervals


def mean(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: Optional[str] = None
) -> Union[float, dict]:
    """
    Compute mean with optional method selection.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray, optional
        Weights (depth intervals) for weighted calculation
    method : str, optional
        'weighted' for depth-weighted mean, 'arithmetic' for simple mean.
        If None, returns dict with both methods.

    Returns
    -------
    float or dict
        If method specified: single float value
        If method is None: {'weighted': float, 'arithmetic': float}

    Examples
    --------
    >>> values = np.array([0.1, 0.2, 0.3])
    >>> weights = np.array([1.0, 2.0, 1.0])
    >>> mean(values, weights)
    {'weighted': 0.2, 'arithmetic': 0.2}
    >>> mean(values, weights, method='weighted')
    0.2
    >>> mean(values, weights, method='arithmetic')
    0.2
    """
    # Arithmetic mean computation
    def _arithmetic():
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid))

    # Weighted mean computation
    def _weighted():
        if weights is None:
            raise ValueError("weights required for weighted method")
        valid_mask = ~np.isnan(values) & ~np.isnan(weights)
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        if len(valid_values) == 0 or np.sum(valid_weights) == 0:
            return np.nan
        return float(np.sum(valid_values * valid_weights) / np.sum(valid_weights))

    if method == 'weighted':
        return _weighted()
    elif method == 'arithmetic':
        return _arithmetic()
    elif method is None:
        return {
            'weighted': _weighted(),
            'arithmetic': _arithmetic()
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'weighted' or 'arithmetic'.")


def sum(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: Optional[str] = None
) -> Union[float, dict]:
    """
    Compute sum with optional method selection.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray, optional
        Weights (depth intervals) for weighted calculation
    method : str, optional
        'weighted' for depth-weighted sum, 'arithmetic' for simple sum.
        If None, returns dict with both methods.

    Returns
    -------
    float or dict
        If method specified: single float value
        If method is None: {'weighted': float, 'arithmetic': float}

    Examples
    --------
    >>> values = np.array([0, 1, 0])  # NTG values
    >>> weights = np.array([0.5, 2.5, 2.0])
    >>> sum(values, weights, method='weighted')
    2.5  # Net thickness
    >>> sum(values, weights, method='arithmetic')
    1.0  # Simple count of net samples
    """
    # Arithmetic sum computation
    def _arithmetic():
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid))

    # Weighted sum computation
    def _weighted():
        if weights is None:
            raise ValueError("weights required for weighted method")
        valid_mask = ~np.isnan(values) & ~np.isnan(weights)
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        if len(valid_values) == 0:
            return np.nan
        return float(np.sum(valid_values * valid_weights))

    if method == 'weighted':
        return _weighted()
    elif method == 'arithmetic':
        return _arithmetic()
    elif method is None:
        return {
            'weighted': _weighted(),
            'arithmetic': _arithmetic()
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'weighted' or 'arithmetic'.")


def std(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: Optional[str] = None
) -> Union[float, dict]:
    """
    Compute standard deviation with optional method selection.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray, optional
        Weights (depth intervals) for weighted calculation
    method : str, optional
        'weighted' for depth-weighted std, 'arithmetic' for simple std.
        If None, returns dict with both methods.

    Returns
    -------
    float or dict
        If method specified: single float value
        If method is None: {'weighted': float, 'arithmetic': float}

    Examples
    --------
    >>> values = np.array([0.1, 0.2, 0.3, 0.2])
    >>> weights = np.array([1.0, 1.0, 1.0, 1.0])
    >>> std(values, weights)
    {'weighted': 0.0707..., 'arithmetic': 0.0707...}
    """
    # Arithmetic std computation
    def _arithmetic():
        valid = values[~np.isnan(values)]
        if len(valid) < 2:
            return np.nan
        return float(np.std(valid))

    # Weighted std computation
    def _weighted():
        if weights is None:
            raise ValueError("weights required for weighted method")
        valid_mask = ~np.isnan(values) & ~np.isnan(weights)
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        if len(valid_values) < 2 or np.sum(valid_weights) == 0:
            return np.nan
        w_mean = mean(values, weights, method='weighted')
        variance = np.sum(valid_weights * (valid_values - w_mean) ** 2) / np.sum(valid_weights)
        return float(np.sqrt(variance))

    if method == 'weighted':
        return _weighted()
    elif method == 'arithmetic':
        return _arithmetic()
    elif method is None:
        return {
            'weighted': _weighted(),
            'arithmetic': _arithmetic()
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'weighted' or 'arithmetic'.")


def percentile(
    values: np.ndarray,
    p: float,
    weights: Optional[np.ndarray] = None,
    method: Optional[str] = None
) -> Union[float, dict]:
    """
    Compute percentile with optional method selection.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    p : float
        Percentile to compute (0-100)
    weights : np.ndarray, optional
        Weights (depth intervals) for weighted calculation
    method : str, optional
        'weighted' for depth-weighted percentile, 'arithmetic' for simple percentile.
        If None, returns dict with both methods.

    Returns
    -------
    float or dict
        If method specified: single float value
        If method is None: {'weighted': float, 'arithmetic': float}

    Examples
    --------
    >>> values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    >>> percentile(values, 50, weights)
    {'weighted': 0.3, 'arithmetic': 0.3}
    >>> percentile(values, 50, weights, method='arithmetic')
    0.3
    """
    # Arithmetic percentile computation
    def _arithmetic():
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan
        return float(np.percentile(valid, p))

    # Weighted percentile computation
    def _weighted():
        if weights is None:
            raise ValueError("weights required for weighted method")
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
        target_weight = (p / 100.0) * total_weight

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

    if method == 'weighted':
        return _weighted()
    elif method == 'arithmetic':
        return _arithmetic()
    elif method is None:
        return {
            'weighted': _weighted(),
            'arithmetic': _arithmetic()
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'weighted' or 'arithmetic'.")


def mode(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    bins: int = 50,
    is_discrete: bool = False
) -> Union[float, dict]:
    """
    Compute mode (most frequent value) with optional method selection.

    For continuous data, values are binned before finding the mode.
    For discrete data, bins parameter is ignored.

    Parameters
    ----------
    values : np.ndarray
        Property values (may contain NaN)
    weights : np.ndarray, optional
        Weights (depth intervals) for weighted calculation
    method : str, optional
        'weighted' for depth-weighted mode, 'arithmetic' for simple mode.
        If None, returns dict with both methods.
    bins : int, default 50
        Number of bins for continuous data (ignored if is_discrete=True)
    is_discrete : bool, default False
        If True, treat as discrete data (no binning)

    Returns
    -------
    float or dict
        If method specified: single float value (mode)
        If method is None: {'weighted': float, 'arithmetic': float}

    Examples
    --------
    >>> values = np.array([0.1, 0.2, 0.2, 0.3, 0.2])
    >>> mode(values, method='arithmetic')
    0.2
    >>> discrete_values = np.array([1, 1, 2, 1, 3])
    >>> mode(discrete_values, method='arithmetic', is_discrete=True)
    1.0
    """
    # Arithmetic mode computation
    def _arithmetic():
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return np.nan

        if is_discrete:
            # For discrete: find most common value
            unique_vals, counts = np.unique(valid, return_counts=True)
            mode_idx = np.argmax(counts)
            return float(unique_vals[mode_idx])
        else:
            # For continuous: bin the data first
            hist, bin_edges = np.histogram(valid, bins=bins)
            mode_bin_idx = np.argmax(hist)
            # Return midpoint of the modal bin
            mode_value = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2.0
            return float(mode_value)

    # Weighted mode computation
    def _weighted():
        if weights is None:
            raise ValueError("weights required for weighted method")
        valid_mask = ~np.isnan(values) & ~np.isnan(weights)
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]

        if len(valid_values) == 0:
            return np.nan

        if is_discrete:
            # For discrete: sum weights for each unique value (vectorized)
            unique_vals, inverse_indices = np.unique(valid_values, return_inverse=True)

            # Sum weights for each unique value using bincount
            # bincount is much faster than looping through unique values
            weighted_sums = np.bincount(inverse_indices, weights=valid_weights)

            # Find value with maximum weight
            max_idx = np.argmax(weighted_sums)
            mode_val = unique_vals[max_idx]

            return float(mode_val)
        else:
            # For continuous: bin the data and weight each bin
            hist, bin_edges = np.histogram(valid_values, bins=bins, weights=valid_weights)
            mode_bin_idx = np.argmax(hist)
            # Return midpoint of the modal bin
            mode_value = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2.0
            return float(mode_value)

    if method == 'weighted':
        return _weighted()
    elif method == 'arithmetic':
        return _arithmetic()
    elif method is None:
        return {
            'weighted': _weighted(),
            'arithmetic': _arithmetic()
        }
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'weighted' or 'arithmetic'.")


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
        'weighted_mean': mean(values, intervals, method='weighted'),
        'weighted_sum': sum(values, intervals, method='weighted'),
        'weighted_std': std(values, intervals, method='weighted'),
        'weighted_p10': percentile(values, 10, intervals, method='weighted'),
        'weighted_p50': percentile(values, 50, intervals, method='weighted'),
        'weighted_p90': percentile(values, 90, intervals, method='weighted'),

        # Arithmetic statistics (sample-based)
        'arithmetic_mean': mean(values, method='arithmetic'),
        'arithmetic_sum': sum(values, method='arithmetic'),
        'arithmetic_std': std(values, method='arithmetic'),

        # Counts and ranges
        'count': int(len(valid_values)),
        'depth_samples': int(len(values)),
        'depth_thickness': float(np.sum(valid_intervals)) if len(valid_intervals) > 0 else 0.0,
        'min': float(np.min(valid_values)) if len(valid_values) > 0 else np.nan,
        'max': float(np.max(valid_values)) if len(valid_values) > 0 else np.nan,
    }
