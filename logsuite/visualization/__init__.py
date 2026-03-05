"""
Well log visualization for Jupyter Lab.

Provides Template and WellView classes for creating customizable well log displays.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Rectangle, Patch

if TYPE_CHECKING:
    from ..core.well import Well

# Import regression classes at module level for performance
from ..analysis.regression import (
    LinearRegression,
    LogarithmicRegression,
    ExponentialRegression,
    PolynomialRegression,
    PowerRegression,
    PolynomialExponentialRegression,
)
from ..exceptions import PropertyNotFoundError

# Default color palettes
DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _create_regression(regression_type: str, **kwargs):
    """Factory function to create regression objects efficiently.

    Args:
        regression_type: Type of regression with optional degree suffix
                        Examples: 'linear', 'polynomial', 'polynomial_3', 'exponential-polynomial_4'
        **kwargs: Additional parameters (deprecated, use suffix notation instead)

    Returns:
        Regression object instance
    """
    regression_type = regression_type.lower()

    # Parse degree suffix (e.g., polynomial_3 -> degree=3)
    degree = None
    if "_" in regression_type:
        parts = regression_type.split("_")
        try:
            degree = int(parts[-1])
            regression_type = "_".join(parts[:-1])  # Remove degree suffix
        except ValueError:
            pass  # Not a degree suffix, keep original

    # Simple regression types
    if regression_type == "linear":
        return LinearRegression()
    elif regression_type == "logarithmic":
        return LogarithmicRegression()
    elif regression_type == "exponential":
        return ExponentialRegression()
    elif regression_type == "power":
        return PowerRegression()

    # Polynomial with degree
    elif regression_type == "polynomial":
        # Use suffix degree, then kwargs, then default
        if degree is None:
            degree = kwargs.get("degree", 2)
        return PolynomialRegression(degree=degree)

    # Exponential-polynomial (renamed from polynomial-exponential)
    elif regression_type == "exponential-polynomial":
        if degree is None:
            degree = kwargs.get("degree", 2)
        return PolynomialExponentialRegression(degree=degree)

    # Backward compatibility: old name polynomial-exponential
    elif regression_type == "polynomial-exponential":
        import warnings

        warnings.warn(
            "'polynomial-exponential' is deprecated. Use 'exponential-polynomial' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if degree is None:
            degree = kwargs.get("degree", 2)
        return PolynomialExponentialRegression(degree=degree)

    else:
        # Create detailed error message with equation formats
        error_msg = (
            f"Unknown regression type: '{regression_type}'. "
            f"Available types:\n\n"
            f"  \u2022 linear:                 y = a*x + b\n"
            f"  \u2022 logarithmic:            y = a*ln(x) + b\n"
            f"  \u2022 exponential:            y = a*e^(b*x)\n"
            f"  \u2022 polynomial:             y = a + b*x + c*x\u00b2 (default: degree=2)\n"
            f"    - polynomial_1:         y = a + b*x (1st degree)\n"
            f"    - polynomial_3:         y = a + b*x + c*x\u00b2 + d*x\u00b3 (3rd degree)\n"
            f"  \u2022 exponential-polynomial: y = 10^(a + b*x + c*x\u00b2) (default: degree=2)\n"
            f"    - exponential-polynomial_1: y = 10^(a + b*x) (1st degree)\n"
            f"    - exponential-polynomial_3: y = 10^(a + b*x + c*x\u00b2 + d*x\u00b3) (3rd degree)\n"
            f"  \u2022 power:                  y = a*x^b\n\n"
            f"Did you mean one of these?"
        )

        # Check for common typos
        suggestions = []
        if "expo" in regression_type or "exp" in regression_type:
            suggestions.append("exponential or exponential-polynomial")
        if "poly" in regression_type:
            suggestions.append("polynomial or exponential-polynomial")
        if "log" in regression_type:
            suggestions.append("logarithmic")
        if "lin" in regression_type:
            suggestions.append("linear")
        if "pow" in regression_type:
            suggestions.append("power")

        if suggestions:
            error_msg += f"\n  Possible matches: {', '.join(suggestions)}"

        raise ValueError(error_msg)


def _downsample_for_plotting(
    depth: np.ndarray, values: np.ndarray, max_points: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample depth and values arrays for efficient plotting while preserving curve shape.

    Uses a min-max preservation strategy: for each bin, keeps both the min and max values
    to preserve peaks and troughs that would be visible in the plot.

    Parameters
    ----------
    depth : np.ndarray
        Depth array
    values : np.ndarray
        Values array
    max_points : int, default 2000
        Maximum number of points to return

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Downsampled (depth, values) arrays
    """
    n_points = len(depth)

    # If already small enough, return as-is
    if n_points <= max_points:
        return depth, values

    # Calculate bin size to achieve target point count
    # We'll keep 2 points per bin (min and max), so need max_points/2 bins
    n_bins = max_points // 2
    bin_size = n_points // n_bins

    # Preallocate output arrays
    out_depth = np.empty(n_bins * 2, dtype=depth.dtype)
    out_values = np.empty(n_bins * 2, dtype=values.dtype)

    out_idx = 0
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size if i < n_bins - 1 else n_points

        # Get slice for this bin
        bin_depths = depth[start_idx:end_idx]
        bin_values = values[start_idx:end_idx]

        # Find indices of min and max values (ignore NaN)
        valid_mask = ~np.isnan(bin_values)
        if not np.any(valid_mask):
            # All NaN in this bin, just take first two points
            out_depth[out_idx] = bin_depths[0]
            out_values[out_idx] = bin_values[0]
            out_idx += 1
            if len(bin_depths) > 1:
                out_depth[out_idx] = bin_depths[1]
                out_values[out_idx] = bin_values[1]
                out_idx += 1
            continue

        valid_values = bin_values[valid_mask]
        valid_depths = bin_depths[valid_mask]

        # Get min and max indices in the valid data
        min_idx = np.argmin(valid_values)
        max_idx = np.argmax(valid_values)

        # Store in order they appear in depth (to maintain curve continuity)
        if min_idx < max_idx:
            out_depth[out_idx] = valid_depths[min_idx]
            out_values[out_idx] = valid_values[min_idx]
            out_idx += 1
            out_depth[out_idx] = valid_depths[max_idx]
            out_values[out_idx] = valid_values[max_idx]
            out_idx += 1
        else:
            out_depth[out_idx] = valid_depths[max_idx]
            out_values[out_idx] = valid_values[max_idx]
            out_idx += 1
            out_depth[out_idx] = valid_depths[min_idx]
            out_values[out_idx] = valid_values[min_idx]
            out_idx += 1

    # Trim to actual size
    return out_depth[:out_idx], out_values[:out_idx]


from .template import Template
from .wellview import WellView
from .crossplot import Crossplot

__all__ = ["Template", "WellView", "Crossplot"]
