"""
Well Log Toolkit - Fast LAS file processing with lazy loading and filtering.

Main Classes
------------
WellDataManager : Global orchestrator for multi-well analysis
Well : Single well containing multiple properties
Property : Single log property with filtering capabilities
LasFile : LAS file reader with lazy loading

Examples
--------
>>> from logsuite import WellDataManager
>>>
>>> # Load LAS files
>>> manager = WellDataManager()
>>> manager.load_las("well1.las").load_las("well2.las")
>>>
>>> # Access well and properties
>>> well = manager.well_12_3_2_B
>>>
>>> # Mark discrete logs and add labels
>>> zone_prop = well.get_property('Zone')
>>> zone_prop.type = 'discrete'
>>> zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir'}
>>>
>>> ntg_prop = well.get_property('NTG_Flag')
>>> ntg_prop.type = 'discrete'
>>> ntg_prop.labels = {0: 'NonNet', 1: 'Net'}
>>>
>>> # Filter and compute statistics (labels appear in output)
>>> stats = well.phie.filter('Zone').filter('NTG_Flag').sums_avg()
>>> # Returns: {'Reservoir': {'Net': {...}, 'NonNet': {...}}, ...}
>>>
>>> # Create new properties with mathematical expressions
>>> well.HC_Volume = well.PHIE * (1 - well.SW)
>>> well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)
>>>
>>> # Export to LAS format
>>> well.export_to_las('output.las')
"""

from ._version import _get_version

__version__ = _get_version()

from .manager import WellDataManager
from .core import Well, Property
from .io import LasFile
from .utils import sanitize_well_name, sanitize_property_name
from .analysis.statistics import (
    compute_intervals,
    mean,
    sum,
    std,
    percentile,
    mode,
    geometric_mean,
    harmonic_mean,
    compute_all_statistics,
)
from .exceptions import (
    WellLogError,
    LasFileError,
    UnsupportedVersionError,
    PropertyError,
    PropertyNotFoundError,
    PropertyTypeError,
    WellError,
    WellNameMismatchError,
    DepthAlignmentError,
)
from .visualization import Template, WellView, Crossplot
from .analysis.regression import (
    LinearRegression,
    LogarithmicRegression,
    ExponentialRegression,
    PolynomialRegression,
    PowerRegression,
    PolynomialExponentialRegression,
)

__all__ = [
    # Main classes
    "WellDataManager",
    "Well",
    "Property",
    "LasFile",
    # Visualization
    "Template",
    "WellView",
    "Crossplot",
    # Regression
    "LinearRegression",
    "LogarithmicRegression",
    "ExponentialRegression",
    "PolynomialRegression",
    "PowerRegression",
    "PolynomialExponentialRegression",
    # Utilities
    "sanitize_well_name",
    "sanitize_property_name",
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
    # Exceptions
    "WellLogError",
    "LasFileError",
    "UnsupportedVersionError",
    "PropertyError",
    "PropertyNotFoundError",
    "PropertyTypeError",
    "WellError",
    "WellNameMismatchError",
    "DepthAlignmentError",
]