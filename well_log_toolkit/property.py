"""
Property class for well log data with filtering support.
"""
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .exceptions import PropertyError, PropertyNotFoundError, PropertyTypeError, DepthAlignmentError
from .statistics import (
    compute_intervals,
    mean as stat_mean, sum as stat_sum, std as stat_std, percentile as stat_percentile
)
from .utils import filter_names
from .operations import PropertyOperationsMixin

if TYPE_CHECKING:
    from .well import Well
    from .las_file import LasFile


class Property(PropertyOperationsMixin):
    """
    Single log property with depth-value pairs and filtering operations.

    A Property can contain secondary properties (filters) that are aligned
    on the same depth grid. This enables chained filtering operations.

    Parameters
    ----------
    name : str
        Property name (sanitized for Python attribute access)
    depth : np.ndarray
        Depth values
    values : np.ndarray
        Log values
    parent_well : Well, optional
        Parent well for property lookup during filtering
    unit : str, default ''
        Unit string
    prop_type : str, default 'continuous'
        Either 'continuous' or 'discrete'
    description : str, default ''
        Property description
    null_value : float, default -999.25
        Value to treat as null/missing
    labels : dict[int, str], optional
        Label mapping for discrete properties (e.g., {0: 'NonNet', 1: 'Net'})
    colors : dict[int, str], optional
        Color mapping for discrete properties (e.g., {0: 'red', 1: 'green'})
    styles : dict[int, str], optional
        Line style mapping for discrete properties (e.g., {0: 'solid', 1: 'dashed'})
    thicknesses : dict[int, float], optional
        Line thickness mapping for discrete properties (e.g., {0: 1.5, 1: 2.0})
    original_name : str, optional
        Original property name with special characters (from LAS file)

    Attributes
    ----------
    name : str
        Property name (sanitized for Python attribute access)
    original_name : str
        Original property name with special characters (from LAS file)
    depth : np.ndarray
        Depth values
    values : np.ndarray
        Log values (nulls converted to np.nan)
    unit : str
        Unit string
    type : str
        'continuous' or 'discrete'
    description : str
        Description
    parent_well : Well | None
        Parent well reference
    labels : dict[int, str] | None
        Label mapping for discrete values
    colors : dict[int, str] | None
        Color mapping for discrete values
    source_las : LasFile | None
        Source LAS file this property came from
    source : str | None
        Source LAS file path (read-only property)
    secondary_properties : list[Property]
        List of aligned filter properties

    Examples
    --------
    >>> phie = well.get_property('PHIE')
    >>> filtered = phie.filter('Zone').filter('NTG_Flag')
    >>> stats = filtered.sums_avg()
    """
    
    def __init__(
        self,
        name: str,
        depth: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        parent_well: Optional['Well'] = None,
        unit: str = '',
        prop_type: str = 'continuous',
        description: str = '',
        null_value: float = -999.25,
        labels: Optional[dict[int, str]] = None,
        colors: Optional[dict[int, str]] = None,
        styles: Optional[dict[int, str]] = None,
        thicknesses: Optional[dict[int, float]] = None,
        source_las: Optional['LasFile'] = None,
        source_name: Optional[str] = None,
        original_name: Optional[str] = None,
        lazy: bool = False
    ):
        self.name = name  # Sanitized name for Python attribute access
        self.original_name = original_name or name  # Original name with special characters
        self.parent_well = parent_well
        self.unit = unit
        self._type = prop_type  # Internal storage for type
        self.description = description
        self._labels = labels  # Internal storage for labels
        self._colors = colors  # Internal storage for colors
        self._styles = styles  # Internal storage for line styles
        self._thicknesses = thicknesses  # Internal storage for line thicknesses
        self.source_las = source_las  # Source LAS file this property came from
        self.source_name = source_name  # Source name (file path or external_df)
        self._null_value = null_value

        # Secondary properties (filters) aligned on same depth grid
        self.secondary_properties: list[Property] = []

        # Lazy loading support
        self._lazy = lazy
        self._depth_cache: Optional[np.ndarray] = None
        self._values_cache: Optional[np.ndarray] = None

        # Filtered copy tracking
        self._is_filtered = False  # True if this is a filtered copy with modified depth grid
        self._boundary_samples_inserted = 0  # Number of boundary samples added during filtering
        self._original_sample_count = 0  # Original sample count before filtering

        # For derived properties (not lazy), store data directly
        if not lazy:
            if depth is not None and values is not None:
                self._depth_cache = np.asarray(depth, dtype=np.float64)
                self._values_cache = np.asarray(values, dtype=np.float64)

                # Replace null values with np.nan
                self._values_cache = np.where(
                    np.abs(self._values_cache - null_value) < 1e-6,
                    np.nan,
                    self._values_cache
                )

                # For discrete properties, round to nearest integer to handle
                # sampling artifacts at zone boundaries (e.g., 0.28 -> 0, 0.67 -> 1)
                # This conversion happens once at initialization for efficiency
                if prop_type == 'discrete':
                    self._values_cache = np.where(
                        np.isnan(self._values_cache),
                        np.nan,  # Keep NaN as NaN
                        np.round(self._values_cache)  # Round valid values to nearest integer
                    )


    def __str__(self) -> str:
        """
        Return user-friendly string representation (numpy-style clipped array).

        Arrays are clipped at 12 elements, showing first 6 and last 6 with '...' separator.
        NaN values at both ends are removed before displaying (like .data() method).

        If filters are active (secondary properties), they are shown below the main values
        with their labels for discrete properties.
        """
        # Get depth and values
        depth = self.depth
        values = self.values

        if len(depth) == 0:
            return f"{self.name}: (empty)"

        # Clip NaN values from both ends (like .data() does)
        valid_mask = ~np.isnan(values)
        if not np.any(valid_mask):
            # All NaN
            return f"[{self.name}] (all NaN values)"

        # Find first and last valid indices
        valid_indices = np.where(valid_mask)[0]
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]

        # Clip to valid range
        depth = depth[first_valid:last_valid + 1]
        values = values[first_valid:last_valid + 1]

        # Also clip secondary properties to same range
        clipped_secondaries = []
        for sec_prop in self.secondary_properties:
            clipped_secondaries.append(sec_prop.values[first_valid:last_valid + 1])

        # Determine if we should clip (show 6 at each end if > 12 total)
        clip_threshold = 12
        should_clip = len(depth) > clip_threshold

        # Format unit string
        unit_str = f" ({self.unit})" if self.unit else ""

        # Build main output
        if not should_clip:
            # Show all values (numpy-style with spaces)
            depth_arr = " ".join(f"{d:.2f}" for d in depth)
            values_arr = " ".join(f"{v:.3f}" if not np.isnan(v) else "   nan" for v in values)

            result = f"[{self.name}]\ndepth: [{depth_arr}]\nvalues{unit_str}: [{values_arr}]"
        else:
            # Clip array (show first 6 ... last 6)
            depth_first = " ".join(f"{d:.2f}" for d in depth[:6])
            depth_last = " ".join(f"{d:.2f}" for d in depth[-6:])
            depth_arr = f"{depth_first} ... {depth_last}"

            values_first = " ".join(f"{v:.3f}" if not np.isnan(v) else "   nan" for v in values[:6])
            values_last = " ".join(f"{v:.3f}" if not np.isnan(v) else "   nan" for v in values[-6:])
            values_arr = f"{values_first} ... {values_last}"

            result = f"[{self.name}] ({len(depth)} samples)\ndepth: [{depth_arr}]\nvalues{unit_str}: [{values_arr}]"

        # Add filter information if secondary properties exist
        if clipped_secondaries:
            result += "\n\nFilters:"
            for i, sec_prop in enumerate(self.secondary_properties):
                # Use clipped values
                sec_values = clipped_secondaries[i]

                if not should_clip:
                    # Show all values
                    sec_arr = " ".join(self._format_discrete_value(v, sec_prop) for v in sec_values)
                else:
                    # Clip array (show first 6 ... last 6)
                    sec_first = " ".join(self._format_discrete_value(v, sec_prop) for v in sec_values[:6])
                    sec_last = " ".join(self._format_discrete_value(v, sec_prop) for v in sec_values[-6:])
                    sec_arr = f"{sec_first} ... {sec_last}"

                # Add unit string if present
                sec_unit_str = f" ({sec_prop.unit})" if sec_prop.unit else ""
                result += f"\n  {sec_prop.name}{sec_unit_str}: [{sec_arr}]"

        return result

    def _format_discrete_value(self, value: float, prop: 'Property') -> str:
        """
        Format a discrete value with its label if available.

        Parameters
        ----------
        value : float
            Numeric value
        prop : Property
            Property containing labels

        Returns
        -------
        str
            Formatted value string (label if available, otherwise numeric)
        """
        if np.isnan(value):
            return "    nan"

        # Try to get label
        if prop.labels and int(value) in prop.labels:
            label = prop.labels[int(value)]
            # Pad to align nicely (use ~7 chars for label width)
            return f"{label:>7s}"
        else:
            return f"{value:7.0f}"

    @property
    def source(self) -> Optional[str]:
        """
        Get the source this property came from.

        Returns
        -------
        Optional[str]
            Source name - either a LAS file path or external DataFrame name
            (e.g., 'path/to/original.las' or 'external_df')

        Examples
        --------
        >>> prop = well.get_property('PHIE')
        >>> print(prop.source)  # 'path/to/original.las' or 'external_df'
        """
        if self.source_name:
            return self.source_name
        return str(self.source_las.filepath) if self.source_las else None

    @property
    def type(self) -> str:
        """
        Get the property type ('continuous' or 'discrete').

        Returns
        -------
        str
            Property type
        """
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """
        Set the property type and mark source as modified.

        When changing to 'discrete', values are automatically rounded to nearest integer
        to handle sampling artifacts at zone boundaries (e.g., 0.28 -> 0, 0.67 -> 1).

        Parameters
        ----------
        value : str
            Property type ('continuous', 'discrete', or 'sampled')
        """
        if value not in ('continuous', 'discrete', 'sampled'):
            raise ValueError(f"type must be 'continuous', 'discrete', or 'sampled', got '{value}'")

        old_type = self._type
        if value != old_type:
            self._type = value

            # If changing TO discrete, round values to integers
            if value == 'discrete' and self._values_cache is not None:
                self._values_cache = np.where(
                    np.isnan(self._values_cache),
                    np.nan,  # Keep NaN as NaN
                    np.round(self._values_cache)  # Round valid values to nearest integer
                )

            # Note: No conversion needed when changing FROM discrete to continuous/sampled
            # Values are already floats (just rounded), which is valid for continuous

            self._mark_source_modified()

    @property
    def labels(self) -> Optional[dict[int, str]]:
        """
        Get the label mapping for discrete property values.

        Returns
        -------
        Optional[dict[int, str]]
            Mapping of numeric values to label strings, or None if not discrete
        """
        return self._labels

    @labels.setter
    def labels(self, value: Optional[dict[int, str]]) -> None:
        """
        Set the label mapping and mark source as modified.

        Parameters
        ----------
        value : Optional[dict[int, str]]
            Mapping of numeric values to label strings
        """
        if value != self._labels:
            self._labels = value
            self._mark_source_modified()

    @property
    def colors(self) -> Optional[dict[int, str]]:
        """
        Get the color mapping for discrete property values.

        Returns
        -------
        Optional[dict[int, str]]
            Mapping of numeric values to color strings, or None if not defined
        """
        return self._colors

    @colors.setter
    def colors(self, value: Optional[dict[int, str]]) -> None:
        """
        Set the color mapping and mark source as modified.

        Parameters
        ----------
        value : Optional[dict[int, str]]
            Mapping of numeric values to color strings (e.g., {0: 'red', 1: 'green'})
        """
        if value != self._colors:
            self._colors = value
            self._mark_source_modified()

    @property
    def styles(self) -> Optional[dict[int, str]]:
        """
        Get the line style mapping for discrete property values.

        Returns
        -------
        Optional[dict[int, str]]
            Mapping of numeric values to line style strings, or None if not defined
        """
        return self._styles

    @styles.setter
    def styles(self, value: Optional[dict[int, str]]) -> None:
        """
        Set the line style mapping and mark source as modified.

        Parameters
        ----------
        value : Optional[dict[int, str]]
            Mapping of numeric values to line style strings (e.g., {0: 'solid', 1: 'dashed'})
        """
        if value != self._styles:
            self._styles = value
            self._mark_source_modified()

    @property
    def thicknesses(self) -> Optional[dict[int, float]]:
        """
        Get the line thickness mapping for discrete property values.

        Returns
        -------
        Optional[dict[int, float]]
            Mapping of numeric values to line thickness floats, or None if not defined
        """
        return self._thicknesses

    @thicknesses.setter
    def thicknesses(self, value: Optional[dict[int, float]]) -> None:
        """
        Set the line thickness mapping and mark source as modified.

        Parameters
        ----------
        value : Optional[dict[int, float]]
            Mapping of numeric values to line thickness floats (e.g., {0: 1.5, 1: 2.0})
        """
        if value != self._thicknesses:
            self._thicknesses = value
            self._mark_source_modified()

    def _mark_source_modified(self) -> None:
        """Mark the parent well's source as modified so it gets re-exported on save."""
        if self.parent_well is not None and self.source_name is not None:
            try:
                self.parent_well.mark_source_modified(self.source_name)
            except (KeyError, AttributeError):
                # Source may not exist in well (e.g., filtered copy) - ignore
                pass

    @property
    def depth(self) -> np.ndarray:
        """
        Get depth array with lazy loading support.

        For lazy properties, data is loaded from source_las on first access.
        For derived properties, data is stored directly.

        Returns
        -------
        np.ndarray
            Depth values
        """
        # Check if we have cached data
        if self._depth_cache is not None:
            return self._depth_cache

        # If lazy and not cached, load from source LAS
        if self._lazy and self.source_las is not None:
            # Load data from LAS file
            las_data = self.source_las.data()

            # Get depth column
            depth_col = self.source_las.depth_column
            if depth_col not in las_data.columns:
                raise PropertyError(
                    f"Depth column '{depth_col}' not found in LAS data. "
                    f"Available columns: {', '.join(las_data.columns)}"
                )

            self._depth_cache = las_data[depth_col].values.astype(np.float64)
            return self._depth_cache

        # No data available
        raise PropertyError(
            f"Property '{self.name}' has no depth data. "
            "Either provide depth during initialization or set source_las for lazy loading."
        )

    @property
    def values(self) -> np.ndarray:
        """
        Get values array with lazy loading support.

        For lazy properties, data is loaded from source_las on first access.
        For derived properties, data is stored directly.

        Returns
        -------
        np.ndarray
            Property values (nulls converted to np.nan)
        """
        # Check if we have cached data
        if self._values_cache is not None:
            return self._values_cache

        # If lazy and not cached, load from source LAS
        if self._lazy and self.source_las is not None:
            # Load data from LAS file
            las_data = self.source_las.data()

            # Get property column using original name
            if self.original_name not in las_data.columns:
                raise PropertyError(
                    f"Property '{self.original_name}' not found in LAS data. "
                    f"Available columns: {', '.join(las_data.columns)}"
                )

            values = las_data[self.original_name].values.astype(np.float64)

            # Replace null values with np.nan
            values = np.where(
                np.abs(values - self._null_value) < 1e-6,
                np.nan,
                values
            )

            # For discrete properties, round to nearest integer to handle
            # sampling artifacts at zone boundaries (e.g., 0.28 -> 0, 0.67 -> 1)
            # This conversion happens once at load time for efficiency
            if self._type == 'discrete':
                values = np.where(
                    np.isnan(values),
                    np.nan,  # Keep NaN as NaN
                    np.round(values)  # Round valid values to nearest integer
                )

            self._values_cache = values
            return self._values_cache

        # No data available
        raise PropertyError(
            f"Property '{self.name}' has no values data. "
            "Either provide values during initialization or set source_las for lazy loading."
        )

    @property
    def MD(self) -> np.ndarray:
        """
        Get Measured Depth array (standardized depth accessor).

        This is an alias for `.depth` that provides a standardized naming convention
        for use in calculations and conditionals across all well log toolkit methods.

        Returns
        -------
        np.ndarray
            Measured depth values (same as `.depth`)

        Examples
        --------
        >>> # Use in conditional calculations
        >>> shallow_phie = np.where(well.PHIE.MD < 2000, well.PHIE, np.nan)
        >>>
        >>> # Combine with Well for multi-well conditionals
        >>> well.calc = np.where((well.PHIE.MD > 2000) & (well.PHIE.MD < 3000),
        ...                      well.PHIE * 2,
        ...                      well.PHIE)
        """
        return self.depth

    @property
    def Well(self) -> np.ndarray:
        """
        Get Well name as an array aligned with depth.

        Returns an array of well names, one for each depth point. This enables
        conditional logic based on well name in calculations and filtering operations.

        Returns
        -------
        np.ndarray
            Array of well names (str), one per depth point

        Examples
        --------
        >>> # Use in multi-well calculations
        >>> plot_data = crossplot.data  # DataFrame with 'well' column
        >>>
        >>> # In property calculations (returns array of well names)
        >>> well_names = well.PHIE.Well
        >>> # array(['36/7-5 ST2', '36/7-5 ST2', ...])
        >>>
        >>> # Conditional based on well
        >>> result = np.where(well.PHIE.Well == "36/7-5", well.PHIE * 1.1, well.PHIE)
        """
        if self.parent_well is None:
            # Return array of empty strings if no parent well
            return np.full(len(self.depth), '', dtype=object)

        # Return array of well names, one for each depth point
        return np.full(len(self.depth), self.parent_well.name, dtype=object)

    @property
    def is_filtered(self) -> bool:
        """
        Check if this property is a filtered copy with modified depth grid.

        Returns
        -------
        bool
            True if this is a filtered copy, False if original
        """
        return self._is_filtered

    def filter_info(self) -> dict:
        """
        Get information about filtering applied to this property.

        Returns
        -------
        dict
            Dictionary with filtering metadata:
            - is_filtered: bool - whether this is a filtered copy
            - filters: list[str] - names of applied filters
            - original_sample_count: int - samples before filtering
            - current_sample_count: int - samples after filtering
            - boundary_samples_inserted: int - synthetic samples added at boundaries
        """
        return {
            'is_filtered': self._is_filtered,
            'filters': [sp.name for sp in self.secondary_properties],
            'original_sample_count': self._original_sample_count if self._is_filtered else len(self.depth),
            'current_sample_count': len(self.depth),
            'boundary_samples_inserted': self._boundary_samples_inserted,
        }

    def resample(self, target_depth: Union[np.ndarray, 'Property']) -> 'Property':
        """
        Resample property to a new depth grid using linear interpolation.

        This method creates a new Property object with values interpolated to match
        the target depth grid. This is required when combining properties with
        different sampling rates.

        Parameters
        ----------
        target_depth : np.ndarray or Property
            Target depth grid to resample to. Can be:
            - numpy array of depth values
            - Property object (will use its .depth attribute)

        Returns
        -------
        Property
            New property with values interpolated to target depth grid

        Notes
        -----
        - Uses linear interpolation for continuous data
        - Uses nearest-neighbor interpolation for discrete data
        - Values outside the original depth range are set to NaN
        - NaN values in original data are excluded from interpolation

        Examples
        --------
        >>> # Resample to another property's depth grid
        >>> phie_resampled = well.CorePHIE.resample(well.PHIE.depth)
        >>> result = well.PHIE + phie_resampled

        >>> # Or pass the property directly
        >>> phie_resampled = well.CorePHIE.resample(well.PHIE)
        >>> result = well.PHIE + phie_resampled

        >>> # Resample to regular 0.5m grid
        >>> target = np.arange(2800, 3500, 0.5)
        >>> phie_regular = well.PHIE.resample(target)
        """
        # Extract depth array if Property object passed
        if hasattr(target_depth, 'depth'):
            target_depth = target_depth.depth

        target_depth = np.asarray(target_depth, dtype=np.float64)

        # Check if already on same grid
        if len(self.depth) == len(target_depth) and np.allclose(self.depth, target_depth, rtol=1e-9, atol=1e-9):
            # Already on same grid - return copy
            return Property(
                name=self.name,
                depth=self.depth.copy(),
                values=self.values.copy(),
                parent_well=self.parent_well,
                unit=self.unit,
                prop_type=self.type,
                description=self.description,
                labels=self.labels.copy() if self.labels else None,
                colors=self.colors.copy() if self.colors else None,
                styles=self.styles.copy() if self.styles else None,
                thicknesses=self.thicknesses.copy() if self.thicknesses else None,
                source_name='computed',
                original_name=self.original_name
            )

        # Handle NaN values - exclude from interpolation
        valid_mask = ~np.isnan(self.values)
        if np.sum(valid_mask) < 2:
            # Not enough valid data to interpolate
            return Property(
                name=self.name,
                depth=target_depth.copy(),
                values=np.full_like(target_depth, np.nan),
                parent_well=self.parent_well,
                unit=self.unit,
                prop_type=self.type,
                description=f"{self.description} (resampled, insufficient data)",
                labels=self.labels.copy() if self.labels else None,
                colors=self.colors.copy() if self.colors else None,
                styles=self.styles.copy() if self.styles else None,
                thicknesses=self.thicknesses.copy() if self.thicknesses else None,
                source_name='computed',
                original_name=self.original_name
            )

        # Choose interpolation method based on type
        if self.type == 'discrete':
            kind = 'nearest'
        else:
            kind = 'linear'

        # Perform interpolation
        interpolator = interp1d(
            self.depth[valid_mask],
            self.values[valid_mask],
            kind=kind,
            bounds_error=False,
            fill_value=np.nan
        )
        resampled_values = interpolator(target_depth)

        # Create new property
        return Property(
            name=self.name,
            depth=target_depth.copy(),
            values=resampled_values,
            parent_well=self.parent_well,
            unit=self.unit,
            prop_type=self.type,
            description=f"{self.description} (resampled)",
            labels=self.labels.copy() if self.labels else None,
            colors=self.colors.copy() if self.colors else None,
            styles=self.styles.copy() if self.styles else None,
            thicknesses=self.thicknesses.copy() if self.thicknesses else None,
            source_name='computed',
            original_name=self.original_name
        )

    def min(self) -> float:
        """
        Return minimum value (ignoring NaN).

        Returns
        -------
        float
            Minimum value, or NaN if all values are NaN

        Examples
        --------
        >>> prop.min()
        0.05
        """
        valid = self.values[~np.isnan(self.values)]
        if len(valid) == 0:
            return np.nan
        return float(np.min(valid))

    def max(self) -> float:
        """
        Return maximum value (ignoring NaN).

        Returns
        -------
        float
            Maximum value, or NaN if all values are NaN

        Examples
        --------
        >>> prop.max()
        0.35
        """
        valid = self.values[~np.isnan(self.values)]
        if len(valid) == 0:
            return np.nan
        return float(np.max(valid))

    def mean(self, weighted: bool = True) -> float:
        """
        Compute mean value.

        Parameters
        ----------
        weighted : bool, default True
            If True, compute depth-weighted mean using interval thicknesses (default for well logs).
            If False, compute simple arithmetic mean (for sampled data like core points).

        Returns
        -------
        float
            Mean value, or NaN if no valid data

        Examples
        --------
        >>> prop.mean()  # Depth-weighted by default
        0.185
        >>> prop.mean(weighted=False)  # Arithmetic mean
        0.182
        """
        if weighted:
            intervals = compute_intervals(self.depth)
            return stat_mean(self.values, intervals, method='weighted')
        else:
            return stat_mean(self.values, method='arithmetic')

    def std(self, weighted: bool = True) -> float:
        """
        Compute standard deviation.

        Parameters
        ----------
        weighted : bool, default True
            If True, compute depth-weighted standard deviation (default for well logs).
            If False, compute simple arithmetic standard deviation (for sampled data).

        Returns
        -------
        float
            Standard deviation, or NaN if insufficient valid data

        Examples
        --------
        >>> prop.std()  # Depth-weighted by default
        0.042
        >>> prop.std(weighted=False)  # Arithmetic std
        0.044
        """
        if weighted:
            intervals = compute_intervals(self.depth)
            return stat_std(self.values, intervals, method='weighted')
        else:
            return stat_std(self.values, method='arithmetic')

    def percentile(self, p: float, weighted: bool = True) -> float:
        """
        Compute percentile value.

        Parameters
        ----------
        p : float
            Percentile to compute (0-100 scale). For example:
            - p=10 gives 10th percentile (P10)
            - p=50 gives median (P50)
            - p=90 gives 90th percentile (P90)
        weighted : bool, default True
            If True, compute depth-weighted percentile (default for well logs).
            If False, compute simple arithmetic percentile (for sampled data).

        Returns
        -------
        float
            Percentile value, or NaN if no valid data

        Examples
        --------
        >>> prop.percentile(50)  # Depth-weighted median by default
        0.18
        >>> prop.percentile(90)  # Depth-weighted P90
        0.24
        >>> prop.percentile(10, weighted=False)  # Arithmetic P10
        0.09
        """
        if weighted:
            intervals = compute_intervals(self.depth)
            return stat_percentile(self.values, p, intervals, method='weighted')
        else:
            return stat_percentile(self.values, p, method='arithmetic')

    def median(self, weighted: bool = True) -> float:
        """
        Compute median value (50th percentile).

        Parameters
        ----------
        weighted : bool, default True
            If True, compute depth-weighted median (default for well logs).
            If False, compute simple arithmetic median (for sampled data).

        Returns
        -------
        float
            Median value, or NaN if no valid data

        Examples
        --------
        >>> prop.median()  # Depth-weighted median by default
        0.18
        >>> prop.median(weighted=False)  # Arithmetic median
        0.175
        """
        return self.percentile(50, weighted=weighted)

    def mode(self, weighted: bool = True, bins: int = 50) -> float:
        """
        Compute mode (most frequent value).

        For continuous data, values are binned before finding the mode.
        For discrete data, bins parameter is ignored.

        Parameters
        ----------
        weighted : bool, default True
            If True, compute depth-weighted mode (default for well logs).
            If False, compute simple arithmetic mode (for sampled data).
        bins : int, default 50
            Number of bins for continuous data (ignored for discrete properties)

        Returns
        -------
        float
            Mode value, or NaN if no valid data

        Examples
        --------
        >>> prop.mode()  # Depth-weighted mode
        0.18
        >>> prop.mode(weighted=False)  # Arithmetic mode
        0.17
        >>> discrete_prop.mode()  # For discrete: most common value
        1.0
        """
        from .statistics import mode as stat_mode

        if weighted:
            intervals = compute_intervals(self.depth)
            return stat_mode(self.values, intervals, method='weighted',
                           bins=bins, is_discrete=(self.type == 'discrete'))
        else:
            return stat_mode(self.values, method='arithmetic',
                           bins=bins, is_discrete=(self.type == 'discrete'))

    def get_value(self, target_depth: float) -> dict:
        """
        Get the value at the closest depth point.

        Finds the nearest sample point to the target depth and returns
        both the actual depth and the corresponding value.

        Parameters
        ----------
        target_depth : float
            Target depth to query

        Returns
        -------
        dict
            Dictionary with keys:
            - 'depth': Actual depth of the nearest sample
            - 'value': Property value at that depth
            - 'distance': Absolute distance from target to actual depth

        Examples
        --------
        >>> # Get porosity at approximately 2850m
        >>> result = well.PHIE.get_value(2850.0)
        >>> print(result)
        {'depth': 2850.5, 'value': 0.18, 'distance': 0.5}

        >>> # Access the values
        >>> actual_depth = result['depth']
        >>> phie_value = result['value']

        >>> # For discrete properties with labels
        >>> zone = well.Zone.get_value(2850.0)
        >>> print(zone)
        {'depth': 2850.0, 'value': 0.0, 'distance': 0.0}
        >>> # To get the label, use the labels dict
        >>> if well.Zone.labels:
        ...     label = well.Zone.labels.get(int(zone['value']))
        """
        if len(self.depth) == 0:
            return {
                'depth': np.nan,
                'value': np.nan,
                'distance': np.nan
            }

        # Find index of closest depth
        distances = np.abs(self.depth - target_depth)
        closest_idx = np.argmin(distances)

        return {
            'depth': float(self.depth[closest_idx]),
            'value': float(self.values[closest_idx]),
            'distance': round(float(distances[closest_idx]), 8)  # 8 decimals to avoid float drift
        }

    def filter(self, property_name: str, insert_boundaries: Optional[bool] = None, source: Optional[str] = None) -> 'Property':
        """
        Add a discrete property from parent well as a filter dimension.

        Returns new Property with the discrete property values interpolated
        to the current depth grid. The depth grid remains unchanged - only
        the discrete values are added as a secondary property.

        Parameters
        ----------
        property_name : str
            Name of discrete property in parent well
        insert_boundaries : bool, optional
            If True, insert synthetic samples at discrete property boundaries.
            Default is True for continuous properties, False for sampled properties.
            Set to False for sampled data (core plugs) to preserve original measurements.
        source : str, optional
            Source name to get filter property from. If None, searches across all sources.
            Use this when the filter property exists in multiple sources to avoid ambiguity.

        Returns
        -------
        Property
            New property instance with secondary property added (same depth grid)

        Raises
        ------
        PropertyNotFoundError
            If parent_well is None or property doesn't exist
        PropertyTypeError
            If property is not discrete type

        Examples
        --------
        >>> filtered = well.phie.filter("Zone").filter("NTG_Flag")
        >>> stats = filtered.sums_avg()
        >>> # Shape remains the same - only discrete values are added
        >>> original.shape  # (29, 2)
        >>> filtered.data().shape  # (29, 3) - added Zone column

        >>> # For sampled data (core plugs), boundaries are not inserted by default
        >>> core_phie.type = 'sampled'
        >>> filtered = core_phie.filter("Zone")  # No boundary insertion

        >>> # When filter property is ambiguous, specify source
        >>> log_phie = well.get_property("PHIE", source="log")
        >>> filtered = log_phie.filter("Zone", source="log")  # Use Zone from log source
        """
        if self.parent_well is None:
            raise PropertyNotFoundError(
                f"Cannot filter property '{self.name}': no parent well reference. "
                "Property must be created from a Well to enable filtering."
            )

        # Lookup in parent well
        try:
            discrete_prop = self.parent_well.get_property(property_name, source=source)
        except KeyError:
            available = ', '.join(self.parent_well.properties)
            raise PropertyNotFoundError(
                f"Property '{property_name}' not found in well '{self.parent_well.name}'. "
                f"Available properties: {available}"
            )

        # Validate it's discrete
        if discrete_prop.type != 'discrete':
            raise PropertyTypeError(
                f"Property '{property_name}' must be discrete type, "
                f"got '{discrete_prop.type}'. Set type with: "
                f"well.get_property('{property_name}').type = 'discrete'"
            )

        # Determine if we should insert boundaries
        # Default: True for continuous, False for sampled
        if insert_boundaries is None:
            insert_boundaries = self.type != 'sampled'

        # Insert synthetic samples at discrete property boundaries
        # This ensures accurate interval weighting when zone boundaries don't align with samples
        # Skip for sampled data (core plugs) to preserve original measurements
        if insert_boundaries:
            new_depth, new_values, new_secondaries = self._insert_boundary_samples(discrete_prop)
        else:
            # No boundary insertion - just copy existing data
            new_depth = self.depth.copy()
            new_values = self.values.copy()
            new_secondaries = [sp for sp in self.secondary_properties]

        # Interpolate discrete property to the NEW depth grid (with boundary samples)
        # Use 'previous' (forward fill) for discrete: value at MD applies from that depth downward
        interpolated_discrete = self._resample_to_grid(
            discrete_prop.depth,
            discrete_prop.values,
            new_depth,  # Use expanded depth grid with boundary samples
            method='previous'  # Forward fill: value applies from depth downward until next marker
        )

        # Mask out discrete values where main property is undefined (NaN)
        # This prevents filtering at depths where the main log doesn't have valid data
        interpolated_discrete = np.where(np.isnan(new_values), np.nan, interpolated_discrete)

        # Add new secondary property (already on same grid as other secondaries)
        new_secondaries.append(Property(
            name=discrete_prop.name,
            depth=new_depth.copy(),  # Same depth grid with boundaries
            values=interpolated_discrete,
            parent_well=self.parent_well,
            unit=discrete_prop.unit,
            prop_type=discrete_prop.type,
            description=discrete_prop.description,
            null_value=-999.25,
            labels=discrete_prop.labels,
            colors=discrete_prop.colors,
            styles=discrete_prop.styles,
            thicknesses=discrete_prop.thicknesses,
            source_las=discrete_prop.source_las,
            source_name=discrete_prop.source_name,
            original_name=discrete_prop.original_name
        ))

        # Create new Property instance with all secondaries
        new_prop = Property(
            name=self.name,
            depth=new_depth,  # Expanded depth grid with boundary samples
            values=new_values,  # Interpolated values at new depths
            parent_well=self.parent_well,
            unit=self.unit,
            prop_type=self.type,
            description=self.description,
            null_value=-999.25,  # Already cleaned
            labels=self.labels,
            colors=self.colors,
            styles=self.styles,
            thicknesses=self.thicknesses,
            source_las=self.source_las,
            source_name=self.source_name,
            original_name=self.original_name
        )
        new_prop.secondary_properties = new_secondaries

        # Track that this is a filtered copy with modified depth grid
        new_prop._is_filtered = True
        new_prop._original_sample_count = len(self.depth)
        new_prop._boundary_samples_inserted = len(new_depth) - len(self.depth)

        return new_prop

    def _insert_boundary_samples(
        self,
        discrete_prop: 'Property'
    ) -> tuple[np.ndarray, np.ndarray, list['Property']]:
        """
        Insert synthetic samples at discrete property boundaries.

        When a discrete property (e.g., zone top) changes value at a depth that
        doesn't align with the log sample grid, this creates synthetic samples
        at those boundary depths to properly partition the intervals.

        Only inserts boundaries within the valid data range of the main property
        (where values are not NaN).

        Parameters
        ----------
        discrete_prop : Property
            Discrete property with boundary depths (e.g., zone tops)

        Returns
        -------
        tuple
            (new_depth, new_values, new_secondaries) with boundary samples inserted
        """
        # Get unique boundary depths from discrete property (handle duplicates)
        # When duplicate depths exist, keep only unique ones for boundary insertion
        valid_discrete_mask = ~np.isnan(discrete_prop.values)
        boundary_depths = np.unique(discrete_prop.depth[valid_discrete_mask])

        # Determine valid depth range of main property (where data exists)
        valid_mask = ~np.isnan(self.values)
        if not np.any(valid_mask):
            # No valid data, return copies
            return (
                self.depth.copy(),
                self.values.copy(),
                [sp for sp in self.secondary_properties]
            )

        valid_depths = self.depth[valid_mask]
        min_valid_depth = valid_depths.min()
        max_valid_depth = valid_depths.max()

        # Find boundaries that fall within the valid data range
        # Filter to boundaries within valid range first
        mask = (boundary_depths > min_valid_depth) & (boundary_depths < max_valid_depth)
        potential_boundaries = boundary_depths[mask]

        if len(potential_boundaries) == 0:
            # No boundaries in range, return copies
            return (
                self.depth.copy(),
                self.values.copy(),
                [sp for sp in self.secondary_properties]
            )

        # Vectorized check if boundaries already exist in depth array
        # Use searchsorted to find insertion points, then check distances to neighbors
        # Cache depth array to avoid repeated property access
        depth_array = self.depth
        depth_len = len(depth_array)
        tolerance = 0.001  # 1mm tolerance

        boundaries_to_insert = []
        for bd in potential_boundaries:
            # Find where this boundary would be inserted
            idx = np.searchsorted(depth_array, bd)

            # Check distance to left and right neighbors
            is_duplicate = False
            if idx > 0 and abs(depth_array[idx - 1] - bd) < tolerance:
                is_duplicate = True
            if idx < depth_len and abs(depth_array[idx] - bd) < tolerance:
                is_duplicate = True

            if not is_duplicate:
                boundaries_to_insert.append(bd)

        if not boundaries_to_insert:
            # No boundaries to insert, return copies
            return (
                depth_array.copy(),
                self.values.copy(),
                [sp for sp in self.secondary_properties]
            )

        boundaries_to_insert = np.array(sorted(boundaries_to_insert))

        # Create new depth grid with boundary samples
        new_depth = np.sort(np.concatenate([depth_array, boundaries_to_insert]))

        # Interpolate main property values at new depths
        new_values = self._resample_to_grid(
            depth_array,
            self.values,
            new_depth,
            method='linear' if self.type == 'continuous' else 'previous'
        )

        # Interpolate all existing secondary properties
        new_secondaries = []
        for sp in self.secondary_properties:
            # Cache sp properties to avoid repeated access
            sp_depth = sp.depth
            sp_values = sp.values
            new_sp_values = self._resample_to_grid(
                sp_depth,
                sp_values,
                new_depth,
                method='previous'  # Secondary properties are discrete
            )
            new_secondaries.append(Property(
                name=sp.name,
                depth=new_depth.copy(),  # Copy to avoid shared references
                values=new_sp_values,
                parent_well=sp.parent_well,
                unit=sp.unit,
                prop_type=sp.type,
                description=sp.description,
                null_value=-999.25,
                labels=sp.labels,
                colors=sp.colors,
                source_las=sp.source_las,
                source_name=sp.source_name,
                original_name=sp.original_name
            ))

        return new_depth, new_values, new_secondaries

    def _align_depths(self, other: 'Property') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Align this property with another on a common depth grid.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (common_depth, self_values_resampled, other_values_resampled)
        """
        # Find common depth range (intersection)
        min_depth = max(self.depth.min(), other.depth.min())
        max_depth = min(self.depth.max(), other.depth.max())
        
        if min_depth >= max_depth:
            raise DepthAlignmentError(
                f"No overlapping depth range between '{self.name}' "
                f"[{self.depth.min():.2f}, {self.depth.max():.2f}] and "
                f"'{other.name}' [{other.depth.min():.2f}, {other.depth.max():.2f}]"
            )
        
        # Use finer grid of the two
        step_self = np.median(np.diff(self.depth)) if len(self.depth) > 1 else 0.1
        step_other = np.median(np.diff(other.depth)) if len(other.depth) > 1 else 0.1
        common_step = min(step_self, step_other)
        
        # Create common depth grid
        common_depth = np.arange(min_depth, max_depth + common_step/2, common_step)
        
        # Resample both properties
        resampled_self = self._resample_to_grid(
            self.depth,
            self.values,
            common_depth,
            method='linear' if self.type == 'continuous' else 'previous'
        )

        resampled_other = self._resample_to_grid(
            other.depth,
            other.values,
            common_depth,
            method='linear' if other.type == 'continuous' else 'previous'
        )
        
        return common_depth, resampled_self, resampled_other
    
    @staticmethod
    def _resample_to_grid(
        old_depth: np.ndarray,
        old_values: np.ndarray,
        new_depth: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Resample values from old depth grid to new depth grid.

        Parameters
        ----------
        old_depth : np.ndarray
            Original depth values
        old_values : np.ndarray
            Original property values
        new_depth : np.ndarray
            Target depth grid
        method : str, default 'linear'
            Interpolation method:
            - 'linear': Linear interpolation (default for continuous)
            - 'previous': Forward fill - value applies from depth downward (use for discrete/tops)
            - 'nearest': Nearest neighbor
            - 'cubic': Cubic spline interpolation

        Returns
        -------
        np.ndarray
            Resampled values on new grid
        """
        # Remove NaN values for interpolation
        mask = ~np.isnan(old_values)
        valid_depth = old_depth[mask]
        valid_values = old_values[mask]

        if len(valid_depth) == 0:
            # All NaN, return NaN array
            return np.full_like(new_depth, np.nan, dtype=np.float64)

        # Handle duplicate depths: keep the LAST value for each depth
        # This is important when tops data has multiple entries at same depth
        if len(valid_depth) > 1:
            unique_depths, unique_indices = np.unique(valid_depth, return_index=True)
            if len(unique_depths) < len(valid_depth):
                # Duplicates exist - for each unique depth, find the last occurrence
                last_indices = []
                for ud in unique_depths:
                    # Find all indices where depth equals this unique depth
                    matches = np.where(valid_depth == ud)[0]
                    # Take the last one
                    last_indices.append(matches[-1])
                valid_depth = valid_depth[last_indices]
                valid_values = valid_values[last_indices]

        if len(valid_depth) == 1:
            # Single point, use nearest neighbor
            method = 'nearest'
        
        # Interpolate
        try:
            # Special handling for 'previous' to forward-fill beyond last point
            if method == 'previous':
                # Use 'previous' for interpolation but forward-fill beyond range
                f = interp1d(
                    valid_depth,
                    valid_values,
                    kind=method,
                    bounds_error=False,
                    fill_value=(np.nan, valid_values[-1])  # NaN before, last value after
                )
            else:
                f = interp1d(
                    valid_depth,
                    valid_values,
                    kind=method,
                    bounds_error=False,
                    fill_value=np.nan
                )
            return f(new_depth)
        except Exception as e:
            raise DepthAlignmentError(
                f"Failed to resample data: {e}"
            )
    
    def sums_avg(self, weighted: Optional[bool] = None, arithmetic: Optional[bool] = None, precision: int = 6) -> dict:
        """
        Compute hierarchical statistics grouped by all secondary properties.

        Parameters
        ----------
        weighted : bool, optional
            Include depth-weighted statistics.
            Default: True for continuous/discrete, False for sampled
        arithmetic : bool, optional
            Include arithmetic (unweighted) statistics.
            Default: False for continuous/discrete, True for sampled
        precision : int, default 6
            Number of decimal places for rounding numeric results

        Returns
        -------
        dict
            Nested dictionary with statistics for each group combination.
            Structure includes:
            - mean: weighted and/or arithmetic mean
            - sum: weighted and/or arithmetic sum
            - std_dev: weighted and/or arithmetic standard deviation
            - percentile: {p10, p50, p90} values
            - range: {min, max} value range
            - depth_range: {min, max} depth range within the zone
            - samples: number of valid samples
            - thickness: depth interval for this group
            - gross_thickness: total depth interval (all groups)
            - thickness_fraction: thickness / gross_thickness
            - calculation: 'weighted', 'arithmetic', or 'both'

        Examples
        --------
        >>> # Simple statistics (no filters)
        >>> phie = well.get_property('PHIE')
        >>> stats = phie.sums_avg()
        >>> # {'mean': 0.18, 'sum': 45.2, 'samples': 251, ...}

        >>> # With arithmetic stats
        >>> stats = phie.sums_avg(arithmetic=True)
        >>> # {'mean': {'weighted': 0.18, 'arithmetic': 0.17}, ...}

        >>> # Grouped statistics
        >>> filtered = phie.filter('Zone').filter('NTG_Flag')
        >>> stats = filtered.sums_avg()
        >>> # {'Zone_1': {'NTG_Flag_0': {...}, 'NTG_Flag_1': {...}}, ...}

        >>> # Sampled data uses arithmetic by default
        >>> core_phie.type = 'sampled'
        >>> stats = core_phie.sums_avg()  # arithmetic=True, weighted=False

        >>> # Custom precision
        >>> stats = phie.sums_avg(precision=3)
        >>> # {'mean': 0.180, 'sum': 45.200, ...}
        """
        # Set defaults based on property type
        # Sampled data: use arithmetic (each sample has equal weight)
        # Continuous/discrete: use weighted (depth-weighted)
        if self.type == 'sampled':
            if weighted is None:
                weighted = False
            if arithmetic is None:
                arithmetic = True
        else:
            if weighted is None:
                weighted = True
            if arithmetic is None:
                arithmetic = False

        # Calculate gross thickness for fraction calculation
        full_intervals = compute_intervals(self.depth)
        valid_mask = ~np.isnan(self.values)
        gross_thickness = float(np.sum(full_intervals[valid_mask]))

        if not self.secondary_properties:
            # No filters, simple statistics
            return self._compute_stats(
                np.ones(len(self.depth), dtype=bool),
                weighted=weighted,
                arithmetic=arithmetic,
                gross_thickness=gross_thickness,
                precision=precision
            )

        # Build hierarchical grouping
        return self._recursive_group(
            0,
            np.ones(len(self.depth), dtype=bool),
            weighted=weighted,
            arithmetic=arithmetic,
            gross_thickness=gross_thickness,
            precision=precision
        )

    def _recursive_group(
        self,
        filter_idx: int,
        mask: np.ndarray,
        weighted: bool,
        arithmetic: bool,
        gross_thickness: float,
        precision: int = 6
    ) -> dict:
        """
        Recursively group by secondary properties.

        Parameters
        ----------
        filter_idx : int
            Index of current secondary property
        mask : np.ndarray
            Boolean mask for current group
        weighted : bool
            Include weighted statistics
        arithmetic : bool
            Include arithmetic statistics
        gross_thickness : float
            Total gross thickness for fraction calculation
        precision : int, default 6
            Number of decimal places for rounding

        Returns
        -------
        dict
            Statistics dict or nested dict of statistics
        """
        if filter_idx >= len(self.secondary_properties):
            # Base case: compute statistics for this group
            return self._compute_stats(mask, weighted, arithmetic, gross_thickness, precision)

        # Get unique values for current filter
        current_filter = self.secondary_properties[filter_idx]
        # Cache filter values to avoid repeated property access
        current_filter_values = current_filter.values
        filter_values = current_filter_values[mask]
        unique_vals = np.unique(filter_values[~np.isnan(filter_values)])

        if len(unique_vals) == 0:
            # No valid values, return stats for current mask
            return self._compute_stats(mask, weighted, arithmetic, gross_thickness, precision)

        # Calculate parent thickness BEFORE subdividing
        # This becomes the gross_thickness for all child groups
        # Cache property access to avoid overhead
        depth_array = self.depth
        values_array = self.values
        parent_intervals = compute_intervals(depth_array)
        parent_valid = mask & ~np.isnan(values_array)
        parent_thickness = float(np.sum(parent_intervals[parent_valid]))

        # Group by each unique value
        result = {}
        for val in unique_vals:
            sub_mask = mask & (current_filter_values == val)

            # Discrete properties are already rounded to integers at load time,
            # so we can directly use integer conversion for label lookup
            if current_filter.type == 'discrete':
                int_val = int(val)  # Already rounded at load time
            else:
                int_val = int(val) if val == int(val) else None

            # Create readable key with label if available
            if current_filter.labels is not None:
                # Try integer lookup first (for discrete properties)
                if int_val is not None and int_val in current_filter.labels:
                    key = current_filter.labels[int_val]
                # Try float lookup (for backward compatibility with float-keyed labels)
                elif val in current_filter.labels:
                    key = current_filter.labels[val]
                # No label found, format as string
                elif int_val is not None:
                    key = f"{current_filter.name}_{int_val}"
                else:
                    key = f"{current_filter.name}_{val:.2f}"
            elif int_val is not None:  # Integer value without label
                key = f"{current_filter.name}_{int_val}"
            else:  # Float value without labels dict
                key = f"{current_filter.name}_{val:.2f}"

            result[key] = self._recursive_group(
                filter_idx + 1, sub_mask, weighted, arithmetic, parent_thickness, precision
            )

        return result

    def _compute_stats(
        self,
        mask: np.ndarray,
        weighted: bool = True,
        arithmetic: bool = False,
        gross_thickness: float = 0.0,
        precision: int = 6
    ) -> dict:
        """
        Compute statistics for values selected by mask.

        Uses depth-weighted statistics by default, which properly accounts for
        varying sample spacing. Also includes arithmetic (unweighted) statistics
        for comparison.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask selecting subset of data
        weighted : bool
            Include weighted statistics
        arithmetic : bool
            Include arithmetic statistics
        gross_thickness : float
            Total gross thickness for fraction calculation
        precision : int, default 6
            Number of decimal places for rounding

        Returns
        -------
        dict
            Statistics dictionary with organized structure:
            - mean, median, mode, sum, std_dev: single value or {weighted, arithmetic} dict
            - percentile: {p10, p50, p90} with single or nested values
            - range: {min, max} value range
            - depth_range: {min, max} depth range within the zone
            - samples, thickness, fraction
            - calculation: method indicator
        """
        # Cache property access to avoid overhead
        values_array = self.values
        depth_array = self.depth

        values = values_array[mask]
        valid = values[~np.isnan(values)]

        # Compute depth intervals on FULL depth array first, then mask
        # This is critical! Intervals must be computed on full grid so that
        # zone boundary samples get correct weights based on their neighbors
        # in the full sequence, not just within their zone.
        full_intervals = compute_intervals(depth_array)
        intervals = full_intervals[mask]
        valid_mask_local = ~np.isnan(values)
        valid_intervals = intervals[valid_mask_local]

        # Total depth thickness (sum of intervals, not just first to last)
        thickness = float(np.sum(valid_intervals)) if len(valid_intervals) > 0 else 0.0

        # Fraction of gross thickness
        fraction = thickness / gross_thickness if gross_thickness > 0 else 0.0

        # Determine calculation method for output
        if weighted and arithmetic:
            calc_method = 'both'
            stat_method = None  # Returns dict with both
        elif weighted:
            calc_method = 'weighted'
            stat_method = 'weighted'
        else:
            calc_method = 'arithmetic'
            stat_method = 'arithmetic'

        # Compute stats using unified functions
        from .statistics import mode as stat_mode

        mean_result = stat_mean(values, intervals, method=stat_method)
        sum_result = stat_sum(values, intervals, method=stat_method)
        std_result = stat_std(values, intervals, method=stat_method)
        p10_result = stat_percentile(values, 10, intervals, method=stat_method)
        p50_result = stat_percentile(values, 50, intervals, method=stat_method)
        p90_result = stat_percentile(values, 90, intervals, method=stat_method)
        median_result = p50_result  # Median is same as P50
        mode_result = stat_mode(values, intervals, method=stat_method,
                               bins=50, is_discrete=(self.type == 'discrete'))

        # Build percentile dict
        percentile_dict = {
            'p10': p10_result,
            'p50': p50_result,
            'p90': p90_result,
        }

        # Build range dict
        range_dict = {
            'min': float(np.min(valid)) if len(valid) > 0 else np.nan,
            'max': float(np.max(valid)) if len(valid) > 0 else np.nan,
        }

        # Build depth_range dict (zone boundaries)
        # For discrete properties: zone extends from first occurrence to next zone start
        # This accounts for forward-fill behavior of discrete properties
        if np.any(mask):
            # Find first and last depth in the zone (including NaN values)
            mask_indices = np.where(mask)[0]
            first_idx = mask_indices[0]
            last_idx = mask_indices[-1]

            zone_top = float(self.depth[first_idx])

            # Bottom: if there are depths below last occurrence, extend to next depth
            # (where the next zone starts). Otherwise, use last occurrence depth.
            if last_idx < len(self.depth) - 1:
                zone_bottom = float(self.depth[last_idx + 1])
            else:
                zone_bottom = float(self.depth[last_idx])

            depth_range_dict = {
                'min': zone_top,
                'max': zone_bottom,
            }
        else:
            depth_range_dict = {
                'min': np.nan,
                'max': np.nan,
            }

        # Helper function to round values recursively
        def _round_value(val):
            """Round a value, handling dicts, floats, and NaN."""
            if isinstance(val, dict):
                return {k: _round_value(v) for k, v in val.items()}
            elif isinstance(val, float):
                if np.isnan(val):
                    return val
                return round(val, precision)
            else:
                return val

        return {
            'mean': _round_value(mean_result),
            'median': _round_value(median_result),
            'mode': _round_value(mode_result),
            'sum': _round_value(sum_result),
            'std_dev': _round_value(std_result),
            'percentile': _round_value(percentile_dict),
            'range': _round_value(range_dict),
            'depth_range': _round_value(depth_range_dict),
            'samples': int(len(valid)),
            'thickness': round(thickness, precision),
            'gross_thickness': round(gross_thickness, precision),
            'thickness_fraction': round(fraction, precision),
            'calculation': calc_method,
        }
    
    def _apply_labels(self, values: np.ndarray) -> np.ndarray:
        """
        Apply label mapping to numeric values.

        Parameters
        ----------
        values : np.ndarray
            Numeric values to map

        Returns
        -------
        np.ndarray
            Array with labels applied (object dtype), preserving NaN values
        """
        if self.labels is None:
            return values

        # Create result array as object type to hold strings
        result = np.empty(len(values), dtype=object)

        for i, val in enumerate(values):
            if np.isnan(val):
                result[i] = np.nan
            else:
                int_val = int(val)
                result[i] = self.labels.get(int_val, int_val)  # Use label or fall back to numeric

        return result

    def data(
        self,
        include: Optional[Union[str, list[str]]] = None,
        exclude: Optional[Union[str, list[str]]] = None,
        discrete_labels: bool = True,
        clip_edges: bool = True,
        clip_to_property: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export property and secondary properties as DataFrame.

        Parameters
        ----------
        include : str or list[str], optional
            Secondary property name(s) to include. If None, includes all.
            Main property is always included. Can be a single string or a list of strings.
        exclude : str or list[str], optional
            Secondary property name(s) to exclude. If both include and
            exclude are specified, exclude overrides (removes from include list).
            Can be a single string or a list of strings.
        discrete_labels : bool, default True
            If True, apply label mappings to discrete properties
        clip_edges : bool, default True
            If True, remove rows at the start and end where all data columns
            (excluding DEPT) contain NaN values. This trims the DataFrame to the
            range where actual data exists.
        clip_to_property : str, optional
            Clip output to the defined range of this specific property. If specified,
            overrides clip_edges behavior.

        Returns
        -------
        pd.DataFrame
            DataFrame with DEPT, main property, and secondary properties

        Examples
        --------
        >>> filtered = well.phie.filter('Zone').filter('NTG_Flag')
        >>> df = filtered.data()
        >>> print(df.head())

        >>> # Include only specific secondary properties
        >>> df = filtered.data(include=['Zone'])

        >>> # Include single secondary property (string or list)
        >>> df = filtered.data(include='Zone')  # Same as include=['Zone']

        >>> # Exclude specific secondary properties
        >>> df = filtered.data(exclude=['NTG_Flag'])

        >>> # Exclude single secondary property
        >>> df = filtered.data(exclude='NTG_Flag')

        >>> # Clip to property range
        >>> df = filtered.data(clip_to_property='Zone')
        """
        # Main property is always included
        data = {
            'DEPT': self.depth,
            self.name: self._apply_labels(self.values) if discrete_labels and self.labels else self.values
        }

        # Determine which secondary properties to include using filter_names helper
        # If both include and exclude are specified, exclude overrides
        secondary_names = [sp.name for sp in self.secondary_properties]
        secondary_filter = filter_names(secondary_names, include, exclude)

        # Add secondary properties
        for sec_prop in self.secondary_properties:
            # Skip if not in filter
            if secondary_filter is not None and sec_prop.name not in secondary_filter:
                continue

            if discrete_labels and sec_prop.labels:
                data[sec_prop.name] = sec_prop._apply_labels(sec_prop.values)
            else:
                data[sec_prop.name] = sec_prop.values

        df = pd.DataFrame(data)

        # Clip to specific property's defined range
        if clip_to_property and clip_to_property in df.columns:
            not_nan = df[clip_to_property].notna()
            if not_nan.any():
                first_valid = not_nan.idxmax()
                last_valid = not_nan[::-1].idxmax()
                df = df.loc[first_valid:last_valid].reset_index(drop=True)
        elif clip_edges and len(df) > 0:
            # Clip edges to remove leading/trailing NaN rows
            # Get data columns (exclude DEPT)
            data_cols = [col for col in df.columns if col != 'DEPT']
            if data_cols:
                # Find first row where at least one data column is not NaN
                not_all_nan = df[data_cols].notna().any(axis=1)
                if not_all_nan.any():
                    first_valid = not_all_nan.idxmax()
                    last_valid = not_all_nan[::-1].idxmax()
                    df = df.loc[first_valid:last_valid].reset_index(drop=True)

        return df

    def head(
        self,
        n: int = 5,
        include: Optional[Union[str, list[str]]] = None,
        exclude: Optional[Union[str, list[str]]] = None
    ) -> pd.DataFrame:
        """
        Return first n rows of property data.

        Convenience method equivalent to `property.data(**kwargs).head(n)`.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return
        include : list[str], optional
            List of secondary property names to include
        exclude : list[str], optional
            List of secondary property names to exclude

        Returns
        -------
        pd.DataFrame
            First n rows of property data

        Examples
        --------
        >>> well.PHIE.head()
        >>> well.PHIE.filter('Zone').head(10)
        >>> well.PHIE.filter('Zone').filter('NTG').head(include=['Zone'])
        """
        return self.data(include=include, exclude=exclude).head(n)

    def export_to_las(
        self,
        filepath: Union[str, Path],
        well_name: Optional[str] = None,
        store_labels: bool = True,
        null_value: float = -999.25
    ) -> None:
        """
        Export property to LAS 2.0 format file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Output LAS file path
        well_name : str, optional
            Well name for the LAS file. If not provided and parent_well exists,
            uses parent well's name. Otherwise uses 'UNKNOWN'.
        store_labels : bool, default True
            If True, store discrete property label mappings in the ~Parameter section.
            The actual data values remain numeric (standard LAS format).
        null_value : float, default -999.25
            Value to use for missing data in LAS file

        Examples
        --------
        >>> prop = well.get_property('PHIE')
        >>> prop.export_to_las('phie_only.las')

        >>> # Export with secondary properties (filters) and labels
        >>> filtered = well.phie.filter('Zone').filter('NTG_Flag')
        >>> filtered.export_to_las('filtered_phie.las', well_name='12/3-2 B')
        """
        # Import here to avoid circular import
        from .las_file import LasFile

        # Determine well name
        if well_name is None:
            if self.parent_well is not None:
                well_name = self.parent_well.name
            else:
                well_name = 'UNKNOWN'

        # Get DataFrame with property and secondary properties (numeric values)
        df = self.data(discrete_labels=False, clip_edges=False)

        # Build column name mapping: sanitized -> original
        column_rename_map = {self.name: self.original_name}
        for sec_prop in self.secondary_properties:
            column_rename_map[sec_prop.name] = sec_prop.original_name

        # Rename DataFrame columns to use original names for LAS export
        df = df.rename(columns=column_rename_map)

        # Build unit mappings (use original names)
        unit_mappings = {
            'DEPT': 'm',  # Default depth unit
            self.original_name: self.unit
        }

        # Add secondary property units
        for sec_prop in self.secondary_properties:
            unit_mappings[sec_prop.original_name] = sec_prop.unit

        # Collect discrete labels, colors, styles, and thicknesses if store_labels is True (use original names)
        label_mappings = None
        color_mappings = None
        style_mappings = None
        thickness_mappings = None
        if store_labels:
            label_mappings = {}
            color_mappings = {}
            style_mappings = {}
            thickness_mappings = {}
            # Check main property
            if self.labels:
                label_mappings[self.original_name] = self.labels
            if self.colors:
                color_mappings[self.original_name] = self.colors
            if self.styles:
                style_mappings[self.original_name] = self.styles
            if self.thicknesses:
                thickness_mappings[self.original_name] = self.thicknesses
            # Check secondary properties
            for sec_prop in self.secondary_properties:
                if sec_prop.labels:
                    label_mappings[sec_prop.original_name] = sec_prop.labels
                if sec_prop.colors:
                    color_mappings[sec_prop.original_name] = sec_prop.colors
                if sec_prop.styles:
                    style_mappings[sec_prop.original_name] = sec_prop.styles
                if sec_prop.thicknesses:
                    thickness_mappings[sec_prop.original_name] = sec_prop.thicknesses

        # Export using LasFile static method
        LasFile.export_las(
            filepath=filepath,
            well_name=well_name,
            df=df,
            unit_mappings=unit_mappings,
            null_value=null_value,
            discrete_labels=label_mappings if label_mappings else None,
            discrete_colors=color_mappings if color_mappings else None,
            discrete_styles=style_mappings if style_mappings else None,
            discrete_thicknesses=thickness_mappings if thickness_mappings else None
        )

    def __repr__(self) -> str:
        """String representation."""
        # Add unit if present
        unit_str = f" ({self.unit})" if self.unit else ""

        # Add filters info if any secondary properties
        filters = f", filters={len(self.secondary_properties)}" if self.secondary_properties else ""

        # Add filtered info if this is a filtered property
        filtered_info = ""
        if self._is_filtered:
            filtered_info = f", filtered=True, boundary_samples={self._boundary_samples_inserted}"

        return (
            f"Property('{self.name}'{unit_str}, "
            f"samples={len(self.depth)}, "
            f"type='{self.type}'"
            f"{filters}"
            f"{filtered_info})"
        )