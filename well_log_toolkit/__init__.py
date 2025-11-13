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
"""

__version__ = "0.1.5"

from .manager import WellDataManager
from .well import Well
from .property import Property
from .las_file import LasFile
from .utils import sanitize_well_name, sanitize_property_name
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
    # Utilities
    "sanitize_well_name",
    "sanitize_property_name",
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