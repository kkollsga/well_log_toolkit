"""
Custom exceptions for well log toolkit.
"""


class WellLogError(Exception):
    """Base exception for all well log toolkit errors."""
    pass


class LasFileError(WellLogError):
    """Raised when LAS file parsing or validation fails."""
    pass


class UnsupportedVersionError(LasFileError):
    """Raised when LAS version is not supported."""
    pass


class PropertyError(WellLogError):
    """Raised when property operations fail."""
    pass


class PropertyNotFoundError(PropertyError):
    """Raised when requested property doesn't exist."""
    pass


class PropertyTypeError(PropertyError):
    """Raised when property type is incorrect for operation."""
    pass


class WellError(WellLogError):
    """Raised when well operations fail."""
    pass


class WellNameMismatchError(WellError):
    """Raised when LAS well name doesn't match target well."""
    pass


class DepthAlignmentError(WellLogError):
    """Raised when depth grids cannot be aligned."""
    pass