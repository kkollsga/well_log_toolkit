"""
Proxy objects for manager-level property operations.

Provides _ManagerPropertyProxy and _ManagerMultiPropertyProxy which enable
broadcasting property operations across all wells in a WellDataManager.
"""
from typing import Optional, Union, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

from ..exceptions import PropertyNotFoundError, PropertyTypeError
from ..core.property import Property
from ..analysis.sums_avg import SumsAvgResult, _sanitize_for_json, _flatten_to_dataframe

if TYPE_CHECKING:
    from .data_manager import WellDataManager
    from ..core.well import Well

class _ManagerPropertyProxy:
    """
    Proxy object for manager-level property operations.

    This proxy enables broadcasting property operations across all wells:
        manager.PHIE_scaled = manager.PHIE * 0.01

    The proxy is created when accessing a property name on the manager,
    and operations on the proxy create new proxies that remember the operation.
    When assigned to a manager attribute, the operation is broadcast to all wells.
    """

    def __init__(self, manager: 'WellDataManager', property_name: str, operation=None, filters=None, custom_intervals=None):
        self._manager = manager
        self._property_name = property_name
        self._operation = operation  # Function to apply to each property
        self._filters = filters or []  # List of (filter_name, insert_boundaries) tuples
        self._custom_intervals = custom_intervals  # For filter_intervals: str (saved name) or dict (well-specific)

    def _apply_operation(self, prop: Property):
        """Apply stored operation to a property."""
        if self._operation is None:
            # No operation, just return the property
            return prop
        else:
            # Apply the operation
            return self._operation(prop)

    def _apply_filter_intervals(self, prop: Property, well):
        """
        Apply filter_intervals to a property if custom_intervals is set.

        Returns None if the well doesn't have the required saved intervals.
        """
        if not self._custom_intervals:
            return prop

        intervals_config = self._custom_intervals
        intervals = intervals_config['intervals']
        name = intervals_config['name']
        insert_boundaries = intervals_config['insert_boundaries']
        save = intervals_config['save']

        # Resolve intervals for this well
        if isinstance(intervals, str):
            # Saved filter name - check if this well has it
            if intervals not in well._saved_filter_intervals:
                return None  # Skip wells that don't have this saved filter
            well_intervals = intervals
        elif isinstance(intervals, dict):
            # Well-specific intervals
            # Check original name, sanitized name, and well_-prefixed sanitized name
            well_intervals = None
            prefixed_name = f"well_{well.sanitized_name}"
            if well.name in intervals:
                well_intervals = intervals[well.name]
            elif well.sanitized_name in intervals:
                well_intervals = intervals[well.sanitized_name]
            elif prefixed_name in intervals:
                well_intervals = intervals[prefixed_name]
            if well_intervals is None:
                return None  # Skip wells not in the dict
        else:
            return None

        # Apply filter_intervals
        return prop.filter_intervals(
            well_intervals,
            name=name,
            insert_boundaries=insert_boundaries,
            save=save
        )

    def _create_proxy_with_operation(self, operation):
        """Create a new proxy with an operation."""
        return _ManagerPropertyProxy(self._manager, self._property_name, operation, self._filters, self._custom_intervals)

    def _extract_statistic_from_grouped(self, grouped_result: dict, stat_name: str, **kwargs) -> dict:
        """
        Extract a specific statistic from grouped sums_avg results.

        Recursively walks through nested dict structure and extracts the requested
        statistic (e.g., 'mean', 'median', 'percentile') from each leaf node.

        Parameters
        ----------
        grouped_result : dict
            Nested result from sums_avg (group_val -> {...stats...})
        stat_name : str
            Name of statistic to extract ('mean', 'median', 'min', 'max', etc.)
        **kwargs
            Additional parameters for weighted/arithmetic selection
            - percentile_key: e.g., 'p50' for percentile extraction

        Returns
        -------
        dict
            Nested dict with same structure but only the requested statistic value
        """
        if not isinstance(grouped_result, dict):
            return grouped_result

        # Check if this is a leaf node (contains statistics)
        if 'mean' in grouped_result or 'samples' in grouped_result:
            # This is a stats dict - extract the requested statistic
            if stat_name == 'percentile':
                # Percentile is nested under 'percentile' dict
                percentile_key = kwargs.get('percentile_key', 'p50')
                if 'percentile' in grouped_result and percentile_key in grouped_result['percentile']:
                    value = grouped_result['percentile'][percentile_key]
                    # If value is a dict with 'weighted'/'arithmetic', prefer 'weighted'
                    if isinstance(value, dict):
                        return value.get('weighted', value.get('arithmetic', None))
                    return value
                return None
            elif stat_name == 'range_min':
                if 'range' in grouped_result:
                    value = grouped_result['range']['min']
                    if isinstance(value, dict):
                        return value.get('weighted', value.get('arithmetic', None))
                    return value
                return None
            elif stat_name == 'range_max':
                if 'range' in grouped_result:
                    value = grouped_result['range']['max']
                    if isinstance(value, dict):
                        return value.get('weighted', value.get('arithmetic', None))
                    return value
                return None
            else:
                # Direct statistic (mean, median, mode, std_dev, etc.)
                value = grouped_result.get(stat_name, None)
                # If value is a dict with 'weighted'/'arithmetic', prefer 'weighted'
                if isinstance(value, dict) and ('weighted' in value or 'arithmetic' in value):
                    return value.get('weighted', value.get('arithmetic', None))
                return value
        else:
            # This is a grouping level - recurse
            result = {}
            for key, value in grouped_result.items():
                extracted = self._extract_statistic_from_grouped(value, stat_name, **kwargs)
                if extracted is not None:
                    result[key] = extracted
            return result if result else None

    def _compute_for_well(self, well, stat_func, nested=False):
        """
        Helper to compute a statistic for a property in a well.

        Handles both unique and ambiguous property cases:
        - If property is unique and nested=False: returns single value
        - If property is unique and nested=True: returns dict with source name as key
        - If property is ambiguous: returns dict with source names as keys
        - If property not found: returns None

        Parameters
        ----------
        well : Well
            Well object to compute statistic for
        stat_func : callable
            Function that takes a Property and returns a statistic value
            Example: lambda prop: prop.mean(weighted=True)
        nested : bool, optional
            If True, always return nested dict with source names, even for unique properties
            If False (default), return single value for unique properties

        Returns
        -------
        float or dict or None
            - float: if property is unique and nested=False
            - dict: if property is ambiguous or nested=True (source_name -> value)
            - None: if property not found
        """
        if nested:
            # Force full nesting - always show source names
            source_results = {}

            for source_name in well._sources.keys():
                try:
                    prop = well.get_property(self._property_name, source=source_name)
                    prop = self._apply_operation(prop)
                    source_results[source_name] = stat_func(prop)
                except PropertyNotFoundError:
                    # Property doesn't exist in this source, skip it
                    pass

            return source_results if source_results else None

        # Default behavior (nested=False)
        try:
            # Try to get property without specifying source (unique case)
            prop = well.get_property(self._property_name)
            prop = self._apply_operation(prop)
            return stat_func(prop)

        except PropertyNotFoundError as e:
            # Check if it's ambiguous (exists in multiple sources)
            if "ambiguous" in str(e).lower():
                # Property exists in multiple sources - compute for each
                source_results = {}

                for source_name in well._sources.keys():
                    try:
                        prop = well.get_property(self._property_name, source=source_name)
                        prop = self._apply_operation(prop)
                        source_results[source_name] = stat_func(prop)
                    except PropertyNotFoundError:
                        # Property doesn't exist in this source, skip it
                        pass

                return source_results if source_results else None
            else:
                # Property truly not found in this well
                return None

        except (AttributeError, KeyError):
            return None

    # Arithmetic operations
    def __add__(self, other):
        """manager.PHIE + value"""
        return self._create_proxy_with_operation(lambda p: p + other)

    def __radd__(self, other):
        """value + manager.PHIE"""
        return self._create_proxy_with_operation(lambda p: other + p)

    def __sub__(self, other):
        """manager.PHIE - value"""
        return self._create_proxy_with_operation(lambda p: p - other)

    def __rsub__(self, other):
        """value - manager.PHIE"""
        return self._create_proxy_with_operation(lambda p: other - p)

    def __mul__(self, other):
        """manager.PHIE * value"""
        return self._create_proxy_with_operation(lambda p: p * other)

    def __rmul__(self, other):
        """value * manager.PHIE"""
        return self._create_proxy_with_operation(lambda p: other * p)

    def __truediv__(self, other):
        """manager.PHIE / value"""
        return self._create_proxy_with_operation(lambda p: p / other)

    def __rtruediv__(self, other):
        """value / manager.PHIE"""
        return self._create_proxy_with_operation(lambda p: other / p)

    def __pow__(self, other):
        """manager.PHIE ** value"""
        return self._create_proxy_with_operation(lambda p: p ** other)

    # Comparison operations
    def __gt__(self, other):
        """manager.PHIE > value"""
        return self._create_proxy_with_operation(lambda p: p > other)

    def __ge__(self, other):
        """manager.PHIE >= value"""
        return self._create_proxy_with_operation(lambda p: p >= other)

    def __lt__(self, other):
        """manager.PHIE < value"""
        return self._create_proxy_with_operation(lambda p: p < other)

    def __le__(self, other):
        """manager.PHIE <= value"""
        return self._create_proxy_with_operation(lambda p: p <= other)

    @property
    def type(self):
        """Get type from first well with this property."""
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                return prop.type
            except (AttributeError, PropertyNotFoundError):
                pass
        return None

    @type.setter
    def type(self, value: str):
        """Set type for this property in all wells."""
        count = 0
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.type = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            print(f"✓ Set type='{value}' for property '{self._property_name}' in {count} well(s)")

    @property
    def labels(self):
        """Get labels from first well with this property."""
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                return prop.labels
            except (AttributeError, PropertyNotFoundError):
                pass
        return None

    @labels.setter
    def labels(self, value: dict):
        """Set labels for this property in all wells.

        Also sets property type to 'discrete' if not already set,
        since labels are only meaningful for discrete properties.
        """
        count = 0
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                # Auto-set type to discrete if labels are being set
                if prop.type != 'discrete':
                    prop.type = 'discrete'
                prop.labels = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            print(f"✓ Set labels for property '{self._property_name}' in {count} well(s)")

    @property
    def colors(self):
        """Get colors from first well with this property."""
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                return prop.colors
            except (AttributeError, PropertyNotFoundError):
                pass
        return None

    @colors.setter
    def colors(self, value: dict):
        """Set colors for this property in all wells."""
        count = 0
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.colors = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            print(f"✓ Set colors for property '{self._property_name}' in {count} well(s)")

    @property
    def styles(self):
        """Get styles from first well with this property."""
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                return prop.styles
            except (AttributeError, PropertyNotFoundError):
                pass
        return None

    @styles.setter
    def styles(self, value: dict):
        """Set styles for this property in all wells."""
        count = 0
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.styles = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            print(f"✓ Set styles for property '{self._property_name}' in {count} well(s)")

    @property
    def thicknesses(self):
        """Get thicknesses from first well with this property."""
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                return prop.thicknesses
            except (AttributeError, PropertyNotFoundError):
                pass
        return None

    @thicknesses.setter
    def thicknesses(self, value: dict):
        """Set thicknesses for this property in all wells."""
        count = 0
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.thicknesses = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            print(f"✓ Set thicknesses for property '{self._property_name}' in {count} well(s)")

    def min(self, nested: bool = False, return_df: bool = False):
        """
        Compute minimum value for this property across all wells.

        If filters are applied, returns grouped minimums for each filter value.

        Parameters
        ----------
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single minimum per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> manager.PHIE.min()
        {'well_A': 0.05, 'well_B': 0.08}

        >>> manager.PHIE.filter("Zone").min()
        {'well_A': {'Zone_1': 0.05, 'Zone_2': 0.08}, ...}

        >>> manager.PHIE.filter("Zone").min(return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.05
        1  well_A  Zone_2     0.08
        """
        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=True, arithmetic=None, precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, 'range_min')
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single min per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.min(), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def max(self, nested: bool = False, return_df: bool = False):
        """
        Compute maximum value for this property across all wells.

        If filters are applied, returns grouped maximums for each filter value.

        Parameters
        ----------
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single maximum per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> manager.PHIE.max()
        {'well_A': 0.35, 'well_B': 0.42}

        >>> manager.PHIE.filter("Zone").max()
        {'well_A': {'Zone_1': 0.35, 'Zone_2': 0.42}, ...}

        >>> manager.PHIE.filter("Zone").max(return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.35
        1  well_A  Zone_2     0.42
        """
        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=True, arithmetic=None, precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, 'range_max')
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single max per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.max(), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def mean(self, weighted: bool = True, nested: bool = False, return_df: bool = False):
        """
        Compute mean value for this property across all wells.

        If filters are applied, returns grouped means for each filter value.

        Parameters
        ----------
        weighted : bool, optional
            Whether to use depth-weighted mean (default True)
            If False, uses arithmetic (unweighted) mean
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single mean per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> manager.PHIE.mean()
        {'well_A': 0.185, 'well_B': 0.192}

        >>> manager.PHIE.filter("Zone").mean()
        {'well_A': {'Zone_1': 0.17, 'Zone_2': 0.22}, ...}

        >>> manager.PHIE.filter("Zone").mean(return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.17
        1  well_A  Zone_2     0.22
        """
        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, 'mean')
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single mean per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.mean(weighted=weighted), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            # Only convert to DF if there's nesting (ambiguous properties or nested=True)
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def std(self, weighted: bool = True, nested: bool = False, return_df: bool = False):
        """
        Compute standard deviation for this property across all wells.

        If filters are applied, returns grouped standard deviations for each filter value.

        Parameters
        ----------
        weighted : bool, optional
            Whether to use depth-weighted standard deviation (default True)
            If False, uses arithmetic (unweighted) standard deviation
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single std per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> manager.PHIE.std()
        {'well_A': 0.042, 'well_B': 0.038}

        >>> manager.PHIE.filter("Zone").std()
        {'well_A': {'Zone_1': 0.035, 'Zone_2': 0.048}, ...}

        >>> manager.PHIE.filter("Zone").std(return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.035
        1  well_A  Zone_2     0.048
        """
        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, 'std_dev')
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single std per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.std(weighted=weighted), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def percentile(self, p: float, weighted: bool = True, nested: bool = False, return_df: bool = False):
        """
        Compute percentile for this property across all wells.

        If filters are applied, returns grouped percentiles for each filter value.
        If no filters, returns a single percentile per well.

        Parameters
        ----------
        p : float
            Percentile to compute (0-100)
        weighted : bool, optional
            Whether to use depth-weighted percentile (default True)
            If False, uses arithmetic (unweighted) percentile
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single percentile value per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> # Without filters
        >>> manager.PHIE.percentile(50)
        {'well_A': 0.18, 'well_B': 0.19}

        >>> # With filters - returns grouped percentiles
        >>> manager.PHIE.filter("Zone").percentile(50)
        {'well_A': {'Zone_1': 0.17, 'Zone_2': 0.22}, 'well_B': {...}}

        >>> # Multiple filters
        >>> manager.PHIE.filter("Zone").filter("NTG_Flag").percentile(90)
        {'well_A': {'Zone_1': {'NTG_0': 0.15, 'NTG_1': 0.25}}, ...}

        >>> manager.PHIE.filter("Zone").percentile(50, return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.17
        1  well_A  Zone_2     0.22
        """
        # If filters are applied, use grouped statistics (like sums_avg)
        if self._filters:
            result = {}
            percentile_key = f'p{int(p)}'  # e.g., 'p50', 'p90'

            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6, nested=nested
                )
                if well_result is not None:
                    # Extract just the requested percentile from the grouped results
                    extracted = self._extract_statistic_from_grouped(
                        well_result, 'percentile', percentile_key=percentile_key
                    )
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single percentile per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.percentile(p, weighted=weighted), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def median(self, weighted: bool = True, nested: bool = False, return_df: bool = False):
        """
        Compute median value (50th percentile) for this property across all wells.

        If filters are applied, returns grouped medians for each filter value.

        Parameters
        ----------
        weighted : bool, optional
            Whether to use depth-weighted median (default True)
            If False, uses arithmetic (unweighted) median
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single median per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> manager.PHIE.median()
        {'well_A': 0.18, 'well_B': 0.19}

        >>> manager.PHIE.filter("Zone").median()
        {'well_A': {'Zone_1': 0.17, 'Zone_2': 0.21}, ...}

        >>> manager.PHIE.filter("Zone").median(return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.17
        1  well_A  Zone_2     0.21
        """
        # If filters are applied, use grouped statistics (median = p50)
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, 'median')
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single median per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.median(weighted=weighted), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def mode(self, weighted: bool = True, bins: int = 50, nested: bool = False, return_df: bool = False):
        """
        Compute mode (most frequent value) for this property across all wells.

        For continuous data, values are binned before finding the mode.
        If filters are applied, returns grouped modes for each filter value.

        Parameters
        ----------
        weighted : bool, optional
            Whether to use depth-weighted mode (default True)
            If False, uses arithmetic (unweighted) mode
        bins : int, optional
            Number of bins for continuous data (default 50)
            Ignored for discrete properties
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources
        return_df : bool, optional
            If True, return results as DataFrame instead of dict (default False)
            Only applies when results are nested (filters or ambiguous properties)

        Returns
        -------
        dict or pd.DataFrame
            Nested dictionary with well names as keys, or DataFrame if return_df=True.
            - Without filters: single mode per well
            - With filters: nested dict grouped by filter values

        Examples
        --------
        >>> manager.PHIE.mode()
        {'well_A': 0.18, 'well_B': 0.17}

        >>> manager.PHIE.filter("Zone").mode()
        {'well_A': {'Zone_1': 0.16, 'Zone_2': 0.20}, ...}

        >>> manager.PHIE.filter("Zone").mode(return_df=True)
           Well    Group       PHIE
        0  well_A  Zone_1     0.16
        1  well_A  Zone_2     0.20
        """
        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, 'mode')
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            if return_df and result:
                return _flatten_to_dataframe(result, self._property_name)
            return result

        # No filters - compute single mode per well
        result = {}
        for well_name, well in self._manager._wells.items():
            value = self._compute_for_well(well, lambda prop: prop.mode(weighted=weighted, bins=bins), nested=nested)
            if value is not None:
                result[well_name] = value

        result = _sanitize_for_json(result)
        if return_df and result and any(isinstance(v, dict) for v in result.values()):
            return _flatten_to_dataframe(result, self._property_name)
        return result

    def stats(self, methods=None, weighted: bool = True, return_df: bool = False):
        """
        Compute multiple statistics for this property across all wells.

        Convenient method to get multiple statistics in one call. Returns dict by default,
        or DataFrame with statistics as columns when return_df=True.

        Parameters
        ----------
        methods : str, list of str, or None, optional
            Statistics to compute. Can be:
            - Single stat name: 'mean', 'median', 'std', 'min', 'max', 'percentile_50', etc.
            - List of stat names: ['mean', 'std', 'percentile_10', 'percentile_90']
            - None: returns all common statistics (default)
        weighted : bool, optional
            Whether to use depth-weighted statistics (default True)
            Applies to mean, std, median, and percentiles
        return_df : bool, optional
            If True, return DataFrame with statistics as columns (default False)
            If False, return nested dict with separate keys for each statistic

        Returns
        -------
        dict or pd.DataFrame
            If return_df=False: {'stat_name': {well_results}, ...}
            If return_df=True: DataFrame with columns [Well, Group(s), stat1, stat2, ...]

        Examples
        --------
        >>> # All statistics
        >>> manager.PHIE.filter("Zone").stats()
        {'mean': {...}, 'median': {...}, 'std': {...}, ...}

        >>> # Single statistic
        >>> manager.PHIE.filter("Zone").stats("mean")
        {'mean': {'well_A': {'Zone_1': 0.17, ...}, ...}}

        >>> # Multiple statistics
        >>> manager.PHIE.filter("Zone").stats(["mean", "std", "percentile_50"])
        {'mean': {...}, 'std': {...}, 'percentile_50': {...}}

        >>> # As DataFrame with stats as columns
        >>> manager.PHIE.filter("Zone").stats(return_df=True)
           Well    Group      mean       std       min       max    median       p10       p50       p90
        0  well_A  Zone_1    0.170     0.042     0.05     0.35     0.168     0.09     0.168     0.24
        1  well_A  Zone_2    0.220     0.038     0.08     0.42     0.218     0.12     0.218     0.28
        """
        # Define default statistics
        default_methods = ['mean', 'median', 'std', 'min', 'max', 'percentile_10', 'percentile_50', 'percentile_90']

        # Parse methods argument
        if methods is None:
            stat_methods = default_methods
        elif isinstance(methods, str):
            stat_methods = [methods]
        elif isinstance(methods, list):
            stat_methods = methods
        else:
            raise ValueError("methods must be None, str, or list of str")

        # Compute each statistic
        results = {}
        for method in stat_methods:
            # Handle percentile_XX format
            if method.startswith('percentile_'):
                percentile = int(method.split('_')[1])
                stat_result = self.percentile(percentile, weighted=weighted, return_df=False)
                results[f'p{percentile}'] = stat_result
            elif method == 'mean':
                results['mean'] = self.mean(weighted=weighted, return_df=False)
            elif method == 'median':
                results['median'] = self.median(weighted=weighted, return_df=False)
            elif method == 'std':
                results['std'] = self.std(weighted=weighted, return_df=False)
            elif method == 'min':
                results['min'] = self.min(return_df=False)
            elif method == 'max':
                results['max'] = self.max(return_df=False)
            elif method == 'mode':
                results['mode'] = self.mode(weighted=weighted, return_df=False)
            else:
                raise ValueError(f"Unknown statistic: {method}")

        if not return_df:
            return results

        # Convert to DataFrame with statistics as columns
        # First, flatten each statistic to get rows
        dfs = []
        for stat_name, stat_dict in results.items():
            df = _flatten_to_dataframe(stat_dict, stat_name)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Merge all DataFrames on grouping columns
        # Identify grouping columns (all except the last column which is the stat value)
        first_df = dfs[0]
        grouping_cols = list(first_df.columns[:-1])  # All except last column

        # Start with first DataFrame
        merged = dfs[0]

        # Merge remaining DataFrames
        for df in dfs[1:]:
            merged = pd.merge(merged, df, on=grouping_cols, how='outer')

        return merged

    def filter(self, property_name: str, insert_boundaries: Optional[bool] = None) -> '_ManagerPropertyProxy':
        """
        Add a discrete property filter for grouped statistics across all wells.

        Creates a new proxy with the filter stored. Multiple filters can be chained.
        Use with sums_avg() to compute grouped statistics.

        Parameters
        ----------
        property_name : str
            Name of discrete property to filter by
        insert_boundaries : bool, optional
            If True, insert synthetic samples at discrete property boundaries.
            Default is True for continuous properties, False for sampled properties.

        Returns
        -------
        _ManagerPropertyProxy
            New proxy with filter added

        Examples
        --------
        >>> # Single filter
        >>> manager.PHIE.filter("Zone").sums_avg()
        >>> # Returns statistics grouped by Zone for each well

        >>> # Multiple filters (chained)
        >>> manager.PHIE.filter("Well_Tops").filter("NetSand_2025").sums_avg()
        >>> # Returns statistics grouped by Well_Tops then NetSand_2025
        """
        # Create new filter list with this filter added
        new_filters = self._filters + [(property_name, insert_boundaries)]

        # Return new proxy with filter added
        return _ManagerPropertyProxy(self._manager, self._property_name, self._operation, new_filters, self._custom_intervals)

    def filter_intervals(
        self,
        intervals: Union[str, dict],
        name: str = "Custom_Intervals",
        insert_boundaries: Optional[bool] = None,
        save: Optional[str] = None
    ) -> '_ManagerPropertyProxy':
        """
        Filter by custom depth intervals across all wells.

        Parameters
        ----------
        intervals : str | dict
            - str: Name of saved filter intervals (looks up per-well)
            - dict: Well-specific intervals {well_name: [intervals]}
        name : str, default "Custom_Intervals"
            Name for the filter property (used in output labels)
        insert_boundaries : bool, optional
            If True, insert synthetic samples at interval boundaries.
        save : str, optional
            If provided, save the intervals to the well(s) under this name.

        Returns
        -------
        _ManagerPropertyProxy
            New proxy with intervals filter added

        Examples
        --------
        >>> # Use saved intervals (only wells with saved intervals are included)
        >>> manager.Facies.filter_intervals("Reservoir_Zones").discrete_summary()

        >>> # Well-specific intervals
        >>> manager.Facies.filter_intervals({
        ...     "well_A": [{"name": "Zone1", "top": 2500, "base": 2700}],
        ...     "well_B": [{"name": "Zone1", "top": 2600, "base": 2800}]
        ... }).discrete_summary()
        """
        # Store intervals config for use when computing stats
        intervals_config = {
            'intervals': intervals,
            'name': name,
            'insert_boundaries': insert_boundaries,
            'save': save
        }

        return _ManagerPropertyProxy(
            self._manager, self._property_name, self._operation,
            self._filters, intervals_config
        )

    def discrete_summary(
        self,
        precision: int = 6,
        skip: Optional[list] = None
    ) -> dict:
        """
        Compute discrete summary statistics across all wells.

        Parameters
        ----------
        precision : int, default 6
            Number of decimal places for rounding numeric results
        skip : list[str], optional
            List of field names to exclude from the output.
            Valid fields: 'code', 'count', 'thickness', 'fraction', 'depth_range'

        Returns
        -------
        dict
            Nested dictionary with structure:
            {
                "well_name": {
                    "zone_name": {
                        "depth_range": {...},
                        "thickness": ...,
                        "facies": {...}
                    }
                }
            }

        Examples
        --------
        >>> # Use saved intervals
        >>> manager.Facies.filter_intervals("Reservoir_Zones").discrete_summary()

        >>> # Skip certain fields
        >>> manager.Facies.filter_intervals("Zones").discrete_summary(skip=["code", "count"])
        """
        if not self._custom_intervals:
            raise ValueError(
                "discrete_summary() requires filter_intervals(). "
                "Use .filter_intervals('saved_name') or .filter_intervals({...}) first."
            )

        result = {}

        for well_name, well in self._manager._wells.items():
            well_result = self._compute_discrete_summary_for_well(well, precision, skip)
            if well_result is not None:
                result[well_name] = well_result

        return _sanitize_for_json(result)

    def _compute_discrete_summary_for_well(
        self,
        well,
        precision: int,
        skip: Optional[list]
    ):
        """
        Helper to compute discrete_summary for a property in a well.
        """
        try:
            prop = well.get_property(self._property_name)
            prop = self._apply_operation(prop)

            # Apply filter_intervals
            prop = self._apply_filter_intervals(prop, well)
            if prop is None:
                return None  # Well doesn't have the saved intervals

            # Apply any additional filters
            for filter_name, filter_insert_boundaries in self._filters:
                if filter_insert_boundaries is not None:
                    prop = prop.filter(filter_name, insert_boundaries=filter_insert_boundaries)
                else:
                    prop = prop.filter(filter_name)

            return prop.discrete_summary(precision=precision, skip=skip)

        except (PropertyNotFoundError, PropertyTypeError, AttributeError, KeyError, ValueError):
            return None

    def sums_avg(
        self,
        weighted: Optional[bool] = None,
        arithmetic: Optional[bool] = None,
        precision: int = 6,
        nested: bool = False
    ) -> SumsAvgResult:
        """
        Compute hierarchical statistics grouped by filters across all wells.

        Must be called on a filtered proxy (created via .filter()).
        Returns statistics for each group combination in each well.

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
        nested : bool, optional
            If True, always return nested dict with source names (default False)
            If False, only nest when property exists in multiple sources

        Returns
        -------
        dict
            Nested dictionary with structure:
            {
                "well_name": {
                    "filter_value_1": {
                        "filter_value_2": {
                            "mean": ..., "sum": ..., "std_dev": ..., ...
                        }
                    }
                }
            }

            With nested=True:
            {
                "well_name": {
                    "source_name": {
                        "filter_value_1": {...}
                    }
                }
            }

        Examples
        --------
        >>> # Single filter
        >>> manager.PHIE.filter("Zone").sums_avg()
        >>> # Returns:
        >>> # {
        >>> #     "well_A": {
        >>> #         "Zone_1": {"mean": 0.18, "sum": 45.2, ...},
        >>> #         "Zone_2": {"mean": 0.22, ...}
        >>> #     },
        >>> #     "well_B": {...}
        >>> # }

        >>> # Multiple filters
        >>> manager.PHIE.filter("Zone").filter("NTG_Flag").sums_avg()
        >>> # Returns:
        >>> # {
        >>> #     "well_A": {
        >>> #         "Zone_1": {
        >>> #             "NTG_0": {"mean": 0.15, ...},
        >>> #             "NTG_1": {"mean": 0.21, ...}
        >>> #         }
        >>> #     }
        >>> # }

        >>> # With nested source names
        >>> manager.PHIE.filter("Zone").sums_avg(nested=True)
        >>> # Returns:
        >>> # {
        >>> #     "well_A": {
        >>> #         "log": {
        >>> #             "Zone_1": {"mean": 0.18, ...}
        >>> #         }
        >>> #     },
        >>> #     "well_B": {
        >>> #         "log": {"Zone_1": {...}},
        >>> #         "core": {"Zone_1": {...}}
        >>> #     }
        >>> # }

        >>> # With custom intervals
        >>> manager.PHIE.filter_intervals("Reservoir_Zones").sums_avg()
        >>> # Returns:
        >>> # {
        >>> #     "well_A": {"Zone_1": {"mean": 0.18, ...}},
        >>> #     "well_B": {"Zone_1": {"mean": 0.21, ...}}
        >>> # }
        """
        if not self._filters and not self._custom_intervals:
            raise ValueError(
                "sums_avg() requires at least one filter or filter_intervals(). "
                "Use .filter('property_name') or .filter_intervals(...) before calling sums_avg()"
            )

        result = {}

        for well_name, well in self._manager._wells.items():
            well_result = self._compute_sums_avg_for_well(
                well, weighted, arithmetic, precision, nested
            )
            if well_result is not None:
                result[well_name] = well_result

        return SumsAvgResult(_sanitize_for_json(result))

    def _compute_sums_avg_for_well(
        self,
        well,
        weighted: Optional[bool],
        arithmetic: Optional[bool],
        precision: int,
        nested: bool
    ):
        """
        Helper to compute sums_avg for a property in a well.

        Applies all filters and computes grouped statistics.
        """
        if nested:
            # Force full nesting - compute for each source
            source_results = {}

            for source_name in well._sources.keys():
                try:
                    # Get property from this source
                    prop = well.get_property(self._property_name, source=source_name)
                    prop = self._apply_operation(prop)

                    # Apply filter_intervals if set
                    prop = self._apply_filter_intervals(prop, well)
                    if prop is None:
                        continue  # Well doesn't have the saved intervals

                    # Apply all filters (specify source to avoid ambiguity)
                    # If a filter doesn't exist, PropertyNotFoundError will be raised and caught below
                    for filter_name, insert_boundaries in self._filters:
                        if insert_boundaries is not None:
                            prop = prop.filter(filter_name, insert_boundaries=insert_boundaries, source=source_name)
                        else:
                            prop = prop.filter(filter_name, source=source_name)

                    # Compute sums_avg
                    result = prop.sums_avg(
                        weighted=weighted,
                        arithmetic=arithmetic,
                        precision=precision
                    )

                    # Add well-level thickness for this source if using filter_intervals
                    if self._custom_intervals and result:
                        well_thickness = 0.0
                        for key, value in result.items():
                            if isinstance(value, dict) and 'thickness' in value:
                                well_thickness += value['thickness']
                        if well_thickness > 0:
                            result['thickness'] = round(well_thickness, precision)

                    source_results[source_name] = result

                except (PropertyNotFoundError, PropertyTypeError, AttributeError, KeyError, ValueError):
                    # Property or filter doesn't exist in this source, or filter isn't discrete - skip it
                    pass

            return source_results if source_results else None

        # Default behavior (nested=False)
        try:
            # Try to get property without specifying source (unique case)
            prop = well.get_property(self._property_name)
            prop = self._apply_operation(prop)

            # Apply filter_intervals if set
            prop = self._apply_filter_intervals(prop, well)
            if prop is None:
                return None  # Well doesn't have the saved intervals

            # Apply all filters
            for filter_name, insert_boundaries in self._filters:
                if insert_boundaries is not None:
                    prop = prop.filter(filter_name, insert_boundaries=insert_boundaries)
                else:
                    prop = prop.filter(filter_name)

            # Compute sums_avg
            result = prop.sums_avg(
                weighted=weighted,
                arithmetic=arithmetic,
                precision=precision
            )

            # Add well-level thickness (sum of all zone thicknesses) if using filter_intervals
            if self._custom_intervals and result:
                well_thickness = 0.0
                for key, value in result.items():
                    if isinstance(value, dict) and 'thickness' in value:
                        well_thickness += value['thickness']
                if well_thickness > 0:
                    result['thickness'] = round(well_thickness, precision)

            return result

        except PropertyNotFoundError as e:
            # Check if it's ambiguous (exists in multiple sources)
            if "ambiguous" in str(e).lower():
                # Property exists in multiple sources - compute for each
                source_results = {}

                for source_name in well._sources.keys():
                    try:
                        prop = well.get_property(self._property_name, source=source_name)
                        prop = self._apply_operation(prop)

                        # Apply filter_intervals if set
                        prop = self._apply_filter_intervals(prop, well)
                        if prop is None:
                            continue  # Well doesn't have the saved intervals

                        # Apply all filters (specify source to avoid ambiguity)
                        # If a filter doesn't exist, PropertyNotFoundError will be raised and caught below
                        for filter_name, insert_boundaries in self._filters:
                            if insert_boundaries is not None:
                                prop = prop.filter(filter_name, insert_boundaries=insert_boundaries, source=source_name)
                            else:
                                prop = prop.filter(filter_name, source=source_name)

                        # Compute sums_avg
                        result = prop.sums_avg(
                            weighted=weighted,
                            arithmetic=arithmetic,
                            precision=precision
                        )

                        # Add well-level thickness for this source if using filter_intervals
                        if self._custom_intervals and result:
                            well_thickness = 0.0
                            for key, value in result.items():
                                if isinstance(value, dict) and 'thickness' in value:
                                    well_thickness += value['thickness']
                            if well_thickness > 0:
                                result['thickness'] = round(well_thickness, precision)

                        source_results[source_name] = result

                    except (PropertyNotFoundError, PropertyTypeError, AttributeError, KeyError, ValueError):
                        # Property or filter doesn't exist in this source, or filter isn't discrete - skip it
                        pass

                return source_results if source_results else None
            else:
                # Property truly not found in this well
                return None

        except (AttributeError, KeyError):
            return None

    def __str__(self) -> str:
        """
        Return string representation showing property across all wells.

        Returns
        -------
        str
            Formatted string with property data from each well

        Examples
        --------
        >>> print(manager.PHIE)
        [PHIE] across 3 well(s):

        Well: well_36_7_5_A
        [PHIE] (1001 samples)
        depth: [2800.00, 2801.00, 2802.00, ..., 3798.00, 3799.00, 3800.00]
        values (v/v): [0.180, 0.185, 0.192, ..., 0.215, 0.212, 0.210]

        Well: well_36_7_5_B
        [PHIE] (856 samples)
        ...
        """
        import numpy as np

        # Get all wells that have this property
        wells_with_prop = []
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                wells_with_prop.append((well_name, prop))
            except (AttributeError, PropertyNotFoundError):
                pass

        if not wells_with_prop:
            return f"[{self._property_name}] - No wells have this property"

        # Build output
        lines = [f"[{self._property_name}] across {len(wells_with_prop)} well(s):", ""]

        for well_name, prop in wells_with_prop:
            # Add well name header
            lines.append(f"Well: {well_name}")

            # Use property's __str__ for consistent formatting
            prop_str = str(prop)
            lines.append(prop_str)
            lines.append("")

        return "\n".join(lines)

    def _broadcast_to_manager(self, manager: 'WellDataManager', target_name: str):
        """
        Broadcast the operation to all wells with the source property.

        Parameters
        ----------
        manager : WellDataManager
            Manager to broadcast to
        target_name : str
            Name for the new computed property in each well
        """
        applied_count = 0
        skipped_wells = []

        for well_name, well in manager._wells.items():
            # Check if well has the source property
            try:
                source_prop = well.get_property(self._property_name)

                # Apply operation to create new property
                result_prop = self._apply_operation(source_prop)

                # Assign to well (will be stored as computed property)
                setattr(well, target_name, result_prop)
                applied_count += 1

            except (AttributeError, KeyError, PropertyNotFoundError):
                # Well doesn't have this property, skip it
                skipped_wells.append(well_name)

        # Provide feedback
        if applied_count > 0:
            print(f"✓ Created property '{target_name}' in {applied_count} well(s)")
        if skipped_wells:
            warnings.warn(
                f"Skipped {len(skipped_wells)} well(s) without property '{self._property_name}': "
                f"{', '.join(skipped_wells[:3])}{'...' if len(skipped_wells) > 3 else ''}",
                UserWarning
            )


class _ManagerMultiPropertyProxy:
    """
    Proxy for computing statistics across multiple properties on all wells.

    Supports filter(), filter_intervals(), and sums_avg() methods.
    Multi-property results nest property-specific stats under property names
    while keeping common stats (depth_range, samples, thickness, etc.) at
    the group level.
    """

    # Stats that are specific to each property (nested under property name)
    PROPERTY_STATS = {'mean', 'median', 'mode', 'sum', 'std_dev', 'percentile', 'range'}

    # Stats that are common across properties (stay at group level)
    COMMON_STATS = {'depth_range', 'samples', 'thickness', 'thickness_fraction', 'calculation'}

    def __init__(
        self,
        manager: 'WellDataManager',
        property_names: list[str],
        filters: Optional[list[tuple]] = None,
        custom_intervals: Optional[dict] = None
    ):
        self._manager = manager
        self._property_names = property_names
        self._filters = filters or []
        self._custom_intervals = custom_intervals

    def __getattr__(self, name: str) -> '_ManagerMultiPropertyProxy':
        """
        Attribute access as shorthand for filter().

        Allows: manager.properties(['A', 'B']).Facies.sums_avg()
        Same as: manager.properties(['A', 'B']).filter('Facies').sums_avg()
        """
        # Avoid recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Treat as filter
        return self.filter(name)

    def filter(
        self,
        property_name: str,
        insert_boundaries: Optional[bool] = None
    ) -> '_ManagerMultiPropertyProxy':
        """
        Add a filter (discrete property) to group statistics by.

        Parameters
        ----------
        property_name : str
            Name of discrete property to group by
        insert_boundaries : bool, optional
            Whether to insert boundary values at filter transitions

        Returns
        -------
        _ManagerMultiPropertyProxy
            New proxy with filter added
        """
        new_filters = self._filters + [(property_name, insert_boundaries)]
        return _ManagerMultiPropertyProxy(
            self._manager, self._property_names, new_filters, self._custom_intervals
        )

    def filter_intervals(
        self,
        intervals: Union[str, list, dict],
        name: str = "Custom_Intervals",
        insert_boundaries: Optional[bool] = None,
        save: Optional[str] = None
    ) -> '_ManagerMultiPropertyProxy':
        """
        Filter by custom depth intervals.

        Parameters
        ----------
        intervals : str, list, or dict
            - str: Name of saved intervals to retrieve from each well
            - list: List of interval dicts [{"name": "Zone_A", "top": 2500, "base": 2700}, ...]
            - dict: Well-specific intervals {"well_name": [...], ...}
        name : str, default "Custom_Intervals"
            Name for the interval filter in results
        insert_boundaries : bool, optional
            Whether to insert boundary values at interval edges
        save : str, optional
            If provided, save intervals to wells with this name

        Returns
        -------
        _ManagerMultiPropertyProxy
            New proxy with custom intervals set
        """
        intervals_config = {
            'intervals': intervals,
            'name': name,
            'insert_boundaries': insert_boundaries,
            'save': save
        }
        return _ManagerMultiPropertyProxy(
            self._manager, self._property_names, self._filters, intervals_config
        )

    def sums_avg(
        self,
        weighted: Optional[bool] = None,
        arithmetic: Optional[bool] = None,
        precision: int = 6
    ) -> SumsAvgResult:
        """
        Compute statistics for multiple properties across all wells.

        Multi-property results nest property-specific stats (mean, median, etc.)
        under each property name, while common stats (depth_range, samples,
        thickness, etc.) remain at the group level.

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
            Nested dictionary with structure:
            {
                "well_name": {
                    "interval_name": {  # if using filter_intervals
                        "filter_value": {
                            "PropertyA": {"mean": ..., "median": ..., ...},
                            "PropertyB": {"mean": ..., "median": ..., ...},
                            "depth_range": {...},
                            "samples": ...,
                            "thickness": ...,
                            ...
                        }
                    }
                }
            }

        Examples
        --------
        >>> manager.properties(['PHIE', 'PERM']).filter('Facies').sums_avg()
        >>> # Returns stats for both properties grouped by facies

        >>> manager.properties(['PHIE', 'PERM']).filter_intervals("Zones").sums_avg()
        >>> # Returns stats for both properties grouped by custom intervals

        >>> # No filters - compute stats for full well
        >>> manager.properties(['PHIE', 'PERM']).sums_avg()
        """
        result = {}

        for well_name, well in self._manager._wells.items():
            well_result = self._compute_sums_avg_for_well(
                well, weighted, arithmetic, precision
            )
            if well_result is not None:
                result[well_name] = well_result

        return SumsAvgResult(_sanitize_for_json(result))

    def _compute_sums_avg_for_well(
        self,
        well,
        weighted: Optional[bool],
        arithmetic: Optional[bool],
        precision: int
    ):
        """
        Compute multi-property sums_avg for a single well.
        """
        # Check if this well has the required saved intervals (if using saved name)
        if self._custom_intervals:
            intervals = self._custom_intervals.get('intervals')
            if isinstance(intervals, str):
                # Saved filter name - check if this well has it
                if intervals not in well._saved_filter_intervals:
                    return None  # Skip wells that don't have this saved filter
            elif isinstance(intervals, dict):
                # Well-specific intervals - check if this well is in the dict
                # Check original name, sanitized name, and well_-prefixed sanitized name
                prefixed_name = f"well_{well.sanitized_name}"
                if well.name not in intervals and well.sanitized_name not in intervals and prefixed_name not in intervals:
                    return None  # Skip wells not in the dict

        # Collect results for each property
        property_results = {}

        for prop_name in self._property_names:
            try:
                prop = well.get_property(prop_name)

                # Apply filter_intervals if set
                if self._custom_intervals:
                    prop = self._apply_filter_intervals(prop, well)
                    if prop is None:
                        continue  # Skip this property if intervals can't be applied

                # Apply all filters
                for filter_name, insert_boundaries in self._filters:
                    if insert_boundaries is not None:
                        prop = prop.filter(filter_name, insert_boundaries=insert_boundaries)
                    else:
                        prop = prop.filter(filter_name)

                # Compute sums_avg
                result = prop.sums_avg(
                    weighted=weighted,
                    arithmetic=arithmetic,
                    precision=precision
                )
                property_results[prop_name] = result

            except (PropertyNotFoundError, PropertyTypeError, AttributeError, KeyError):
                # Property doesn't exist in this well or filter error, skip it
                pass

        if not property_results:
            return None

        # If no filters/intervals, return simple merged result (no grouping)
        if not self._filters and not self._custom_intervals:
            return self._merge_flat_results(property_results)

        # Merge results: nest property-specific stats, keep common stats at group level
        merged = self._merge_property_results(property_results)

        # Add well-level thickness (sum of all zone thicknesses)
        if self._custom_intervals and merged:
            well_thickness = 0.0
            for key, value in merged.items():
                if isinstance(value, dict) and 'thickness' in value:
                    well_thickness += value['thickness']
            merged['thickness'] = round(well_thickness, 6)

        return merged

    def _apply_filter_intervals(self, prop, well):
        """
        Apply filter_intervals to a property if custom_intervals is set.

        Returns None if the well doesn't have the required saved intervals.
        """
        if not self._custom_intervals:
            return prop

        intervals_config = self._custom_intervals
        intervals = intervals_config['intervals']
        name = intervals_config['name']
        insert_boundaries = intervals_config['insert_boundaries']
        save = intervals_config['save']

        # Resolve intervals for this well
        if isinstance(intervals, str):
            # Saved filter name - check if this well has it
            if intervals not in well._saved_filter_intervals:
                return None  # Skip wells that don't have this saved filter
            well_intervals = intervals
        elif isinstance(intervals, dict):
            # Well-specific intervals
            # Check original name, sanitized name, and well_-prefixed sanitized name
            well_intervals = None
            prefixed_name = f"well_{well.sanitized_name}"
            if well.name in intervals:
                well_intervals = intervals[well.name]
            elif well.sanitized_name in intervals:
                well_intervals = intervals[well.sanitized_name]
            elif prefixed_name in intervals:
                well_intervals = intervals[prefixed_name]
            if well_intervals is None:
                return None  # Skip wells not in the dict
        elif isinstance(intervals, list):
            # Direct list of intervals
            well_intervals = intervals
        else:
            return None

        # Apply filter_intervals
        return prop.filter_intervals(
            well_intervals,
            name=name,
            insert_boundaries=insert_boundaries,
            save=save
        )

    def _merge_flat_results(self, property_results: dict) -> dict:
        """
        Merge results when no filters are applied (flat structure).

        Returns a single dict with property-specific stats nested under property
        names and common stats at the top level.

        Parameters
        ----------
        property_results : dict
            {property_name: sums_avg_result}

        Returns
        -------
        dict
            {
                "PropertyA": {"mean": ..., "median": ..., ...},
                "PropertyB": {"mean": ..., ...},
                "depth_range": {...},
                "samples": ...,
                ...
            }
        """
        if not property_results:
            return {}

        result = {}

        # Add property-specific stats for each property
        for prop_name, prop_result in property_results.items():
            if isinstance(prop_result, dict):
                # Extract property-specific stats
                prop_stats = {
                    k: v for k, v in prop_result.items()
                    if k in self.PROPERTY_STATS
                }
                if prop_stats:
                    result[prop_name] = prop_stats

        # Add common stats from first property
        first_result = next(iter(property_results.values()))
        if isinstance(first_result, dict):
            for k, v in first_result.items():
                if k in self.COMMON_STATS:
                    result[k] = v

        return result

    def _merge_property_results(self, property_results: dict) -> dict:
        """
        Merge results from multiple properties.

        Nests property-specific stats under property names while keeping
        common stats at the group level.

        Parameters
        ----------
        property_results : dict
            {property_name: sums_avg_result}

        Returns
        -------
        dict
            Merged result with structure:
            {
                "group_value": {
                    "PropertyA": {"mean": ..., ...},
                    "PropertyB": {"mean": ..., ...},
                    "depth_range": {...},
                    "samples": ...,
                    ...
                }
            }
        """
        if not property_results:
            return {}

        # Use first property result as the structure template
        first_prop = next(iter(property_results.keys()))
        first_result = property_results[first_prop]

        return self._merge_recursive(property_results, first_result)

    def _merge_recursive(self, property_results: dict, template: dict) -> dict:
        """
        Recursively merge property results following the template structure.
        """
        result = {}

        for key, value in template.items():
            if isinstance(value, dict):
                # Check if this is a stats dict (has property-specific keys)
                if any(k in value for k in self.PROPERTY_STATS):
                    # This is a leaf stats dict - merge property stats here
                    merged = {}

                    # Add property-specific stats for each property
                    for prop_name, prop_result in property_results.items():
                        # Navigate to the same key in this property's result
                        prop_value = self._get_nested_value(prop_result, key)
                        if prop_value and isinstance(prop_value, dict):
                            # Extract property-specific stats
                            prop_stats = {
                                k: v for k, v in prop_value.items()
                                if k in self.PROPERTY_STATS
                            }
                            if prop_stats:
                                merged[prop_name] = prop_stats

                    # Add common stats from the first property
                    for k, v in value.items():
                        if k in self.COMMON_STATS:
                            merged[k] = v

                    result[key] = merged
                else:
                    # This is an intermediate nesting level - recurse
                    # Collect corresponding sub-dicts from all properties
                    sub_property_results = {}
                    for prop_name, prop_result in property_results.items():
                        prop_value = self._get_nested_value(prop_result, key)
                        if prop_value and isinstance(prop_value, dict):
                            sub_property_results[prop_name] = prop_value

                    if sub_property_results:
                        result[key] = self._merge_recursive(sub_property_results, value)
            else:
                # Non-dict value, just copy from template
                result[key] = value

        return result

    def _get_nested_value(self, d: dict, key: str):
        """Get value from dict, returning None if key doesn't exist."""
        return d.get(key) if isinstance(d, dict) else None

