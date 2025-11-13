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
>>> from well_log_toolkit import WellDataManager
>>> 
>>> # Load LAS files
>>> manager = WellDataManager()
>>> manager.load_las("well1.las").load_las("well2.las")
>>> 
>>> # Access well and properties
>>> well = manager.well_36_7_5_B
>>> 
>>> # Mark discrete logs
>>> well.get_property('Zone').type = 'discrete'
>>> well.get_property('NTG_Flag').type = 'discrete'
>>> 
>>> # Filter and compute statistics
>>> stats = well.phie.filter('Zone').filter('NTG_Flag').sums_avg()
"""

__version__ = "0.1.0"

from .manager import WellDataManager
from .well import Well
from .property import Property
from .las_file import LasFile
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

__all__ = [
    # Main classes
    "WellDataManager",
    "Well",
    "Property",
    "LasFile",
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