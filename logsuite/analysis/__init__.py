"""
Statistical analysis and regression for well log data.

Submodules
----------
statistics : Depth-weighted statistical functions
regression : Regression models for crossplot analysis
sums_avg : SumsAvgResult container for multi-well aggregation
"""

from .regression import (
    ExponentialRegression,
    LinearRegression,
    LogarithmicRegression,
    PolynomialExponentialRegression,
    PolynomialRegression,
    PowerRegression,
)
from .statistics import (
    compute_all_statistics,
    compute_intervals,
    geometric_mean,
    harmonic_mean,
    mean,
    mode,
    percentile,
    std,
    sum,
)
from .sums_avg import SumsAvgResult, _flatten_to_dataframe, _sanitize_for_json

__all__ = [
    # Statistics
    "compute_intervals",
    "mean",
    "sum",
    "std",
    "percentile",
    "mode",
    "geometric_mean",
    "harmonic_mean",
    "compute_all_statistics",
    # Regression
    "LinearRegression",
    "LogarithmicRegression",
    "ExponentialRegression",
    "PolynomialRegression",
    "PowerRegression",
    "PolynomialExponentialRegression",
    # Sums/Avg
    "SumsAvgResult",
    "_sanitize_for_json",
    "_flatten_to_dataframe",
]
