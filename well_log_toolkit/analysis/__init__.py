"""
Statistical analysis and regression for well log data.

Submodules
----------
statistics : Depth-weighted statistical functions
regression : Regression models for crossplot analysis
sums_avg : SumsAvgResult container for multi-well aggregation
"""
from .statistics import (
    compute_intervals,
    mean,
    sum,
    std,
    percentile,
    mode,
    compute_all_statistics,
)
from .regression import (
    LinearRegression,
    LogarithmicRegression,
    ExponentialRegression,
    PolynomialRegression,
    PowerRegression,
    PolynomialExponentialRegression,
)
from .sums_avg import SumsAvgResult, _sanitize_for_json, _flatten_to_dataframe

__all__ = [
    # Statistics
    "compute_intervals",
    "mean",
    "sum",
    "std",
    "percentile",
    "mode",
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
