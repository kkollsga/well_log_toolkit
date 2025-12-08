"""
Global orchestrator for multi-well analysis.
"""
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd

from .exceptions import LasFileError, PropertyNotFoundError, PropertyTypeError
from .las_file import LasFile
from .well import Well
from .property import Property
from .utils import sanitize_well_name, sanitize_property_name

if TYPE_CHECKING:
    from .visualization import Template


def _sanitize_for_json(obj):
    """
    Recursively sanitize data structures for JSON serialization.

    Converts NaN, inf, and -inf values to None to ensure JSON compliance.
    This prevents jupyter-client warnings about non-JSON-compliant floats.

    Parameters
    ----------
    obj : any
        Object to sanitize (dict, list, float, etc.)

    Returns
    -------
    any
        Sanitized object with NaN/inf values replaced by None
    """
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        # Check for NaN, inf, -inf using numpy
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj


def _flatten_to_dataframe(nested_dict: dict, property_name: str) -> pd.DataFrame:
    """
    Flatten nested dictionary results into a DataFrame.

    Converts hierarchical dictionary structure from manager-level statistics
    into a tabular format with columns for each grouping level.

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary with well names as top-level keys
    property_name : str
        Name of the property being analyzed (used for value column name)

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame with columns for each level of nesting

    Examples
    --------
    Input: {'well_A': {'Zone1': 0.2, 'Zone2': 0.3}, 'well_B': 0.25}
    Output:
        Well     Zone    PHIE
        well_A   Zone1   0.2
        well_A   Zone2   0.3
        well_B   NaN     0.25
    """
    rows = []

    def _recurse(value, path):
        """Recursively traverse nested dict and collect rows."""
        if isinstance(value, dict):
            # This is a grouping level, recurse deeper
            for key, sub_value in value.items():
                _recurse(sub_value, path + [key])
        else:
            # This is a leaf value, create a row
            rows.append(path + [value])

    # Start recursion from each well
    for well_name, well_result in nested_dict.items():
        _recurse(well_result, [well_name])

    if not rows:
        # No data, return empty DataFrame
        return pd.DataFrame()

    # Determine number of columns (max depth)
    max_depth = max(len(row) for row in rows)

    # Pad shorter rows with None
    padded_rows = [row + [None] * (max_depth - len(row)) for row in rows]

    # Create column names
    # Last column is the value, others are grouping levels
    if max_depth == 2:
        # Simple case: just well and value
        columns = ['Well', property_name]
    elif max_depth == 3:
        # Well, one grouping level (e.g., Source or Zone), value
        columns = ['Well', 'Group', property_name]
    else:
        # Well, multiple grouping levels, value
        # Use generic names: Group1, Group2, etc.
        columns = ['Well'] + [f'Group{i}' for i in range(1, max_depth - 1)] + [property_name]

    df = pd.DataFrame(padded_rows, columns=columns)

    return df


class _ManagerPropertyProxy:
    """
    Proxy object for manager-level property operations.

    This proxy enables broadcasting property operations across all wells:
        manager.PHIE_scaled = manager.PHIE * 0.01

    The proxy is created when accessing a property name on the manager,
    and operations on the proxy create new proxies that remember the operation.
    When assigned to a manager attribute, the operation is broadcast to all wells.
    """

    def __init__(self, manager: 'WellDataManager', property_name: str, operation=None, filters=None):
        self._manager = manager
        self._property_name = property_name
        self._operation = operation  # Function to apply to each property
        self._filters = filters or []  # List of (filter_name, insert_boundaries) tuples

    def _apply_operation(self, prop: Property):
        """Apply stored operation to a property."""
        if self._operation is None:
            # No operation, just return the property
            return prop
        else:
            # Apply the operation
            return self._operation(prop)

    def _create_proxy_with_operation(self, operation):
        """Create a new proxy with an operation."""
        return _ManagerPropertyProxy(self._manager, self._property_name, operation, self._filters)

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
        """Set labels for this property in all wells."""
        count = 0
        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
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
        return _ManagerPropertyProxy(self._manager, self._property_name, self._operation, new_filters)

    def sums_avg(
        self,
        weighted: Optional[bool] = None,
        arithmetic: Optional[bool] = None,
        precision: int = 6,
        nested: bool = False
    ) -> dict:
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
        """
        if not self._filters:
            raise ValueError(
                "sums_avg() requires at least one filter. "
                "Use .filter('property_name') before calling sums_avg()"
            )

        result = {}

        for well_name, well in self._manager._wells.items():
            well_result = self._compute_sums_avg_for_well(
                well, weighted, arithmetic, precision, nested
            )
            if well_result is not None:
                result[well_name] = well_result

        return _sanitize_for_json(result)

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

                    # Apply all filters (specify source to avoid ambiguity)
                    # If a filter doesn't exist, PropertyNotFoundError will be raised and caught below
                    for filter_name, insert_boundaries in self._filters:
                        if insert_boundaries is not None:
                            prop = prop.filter(filter_name, insert_boundaries=insert_boundaries, source=source_name)
                        else:
                            prop = prop.filter(filter_name, source=source_name)

                    # Compute sums_avg
                    source_results[source_name] = prop.sums_avg(
                        weighted=weighted,
                        arithmetic=arithmetic,
                        precision=precision
                    )

                except (PropertyNotFoundError, PropertyTypeError, AttributeError, KeyError, ValueError):
                    # Property or filter doesn't exist in this source, or filter isn't discrete - skip it
                    pass

            return source_results if source_results else None

        # Default behavior (nested=False)
        try:
            # Try to get property without specifying source (unique case)
            prop = well.get_property(self._property_name)
            prop = self._apply_operation(prop)

            # Apply all filters
            for filter_name, insert_boundaries in self._filters:
                if insert_boundaries is not None:
                    prop = prop.filter(filter_name, insert_boundaries=insert_boundaries)
                else:
                    prop = prop.filter(filter_name)

            # Compute sums_avg
            return prop.sums_avg(
                weighted=weighted,
                arithmetic=arithmetic,
                precision=precision
            )

        except PropertyNotFoundError as e:
            # Check if it's ambiguous (exists in multiple sources)
            if "ambiguous" in str(e).lower():
                # Property exists in multiple sources - compute for each
                source_results = {}

                for source_name in well._sources.keys():
                    try:
                        prop = well.get_property(self._property_name, source=source_name)
                        prop = self._apply_operation(prop)

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


class WellDataManager:
    """
    Global orchestrator for multi-well analysis.
    
    Manages multiple wells, each containing multiple properties.
    Provides attribute-based well access for clean API.
    
    Attributes
    ----------
    wells : list[str]
        List of sanitized well names
    
    Examples
    --------
    >>> manager = WellDataManager()
    >>> manager.load_las("well1.las").load_las("well2.las")
    >>> well = manager.well_12_3_2_B
    >>> stats = well.phie.filter('Zone').sums_avg()

    >>> # Load project directly on initialization
    >>> manager = WellDataManager("Cerisa Project")
    >>> print(manager.wells)  # All wells from project
    """

    def __init__(self, project: Optional[Union[str, Path]] = None):
        """
        Initialize WellDataManager, optionally loading a project.

        Parameters
        ----------
        project : Union[str, Path], optional
            Path to project folder to load. If provided, the project will be
            loaded immediately during initialization.

        Examples
        --------
        >>> manager = WellDataManager()  # Empty manager
        >>> manager = WellDataManager("my_project")  # Load project on init
        """
        self._wells: dict[str, Well] = {}  # {sanitized_name: Well}
        self._name_mapping: dict[str, str] = {}  # {original_name: sanitized_name}
        self._project_path: Optional[Path] = None  # Track project path for save()
        self._templates: dict[str, 'Template'] = {}  # {template_name: Template}

        # Load project if provided
        if project is not None:
            self.load(project)

    def __setattr__(self, name: str, value):
        """
        Intercept attribute assignment for manager-level broadcasting.

        When assigning a ManagerPropertyProxy to a manager attribute, it broadcasts
        the operation to all wells that have the source property.

        Examples
        --------
        >>> manager.PHIE_scaled = manager.PHIE * 0.01  # Applies to all wells with PHIE
        >>> manager.Reservoir = manager.PHIE > 0.15    # Applies to all wells with PHIE
        """
        # Allow setting private attributes normally
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        # Check if this is a ManagerPropertyProxy (result of manager.PROPERTY operation)
        if isinstance(value, _ManagerPropertyProxy):
            # This is a broadcasting operation
            value._broadcast_to_manager(self, name)
        else:
            # Normal attribute assignment
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        """
        Get well or create property proxy for broadcasting.

        Handles both well access (well_XXX) and property broadcasting (PROPERTY_NAME).
        """
        # Check if it's a well access pattern
        if name.startswith('well_'):
            if name in self._wells:
                return self._wells[name]
            raise AttributeError(
                f"Well '{name}' not found in manager. "
                f"Available wells: {', '.join(self._wells.keys()) or 'none'}"
            )

        # Otherwise, treat as property name for broadcasting
        # Return a proxy that can be used for operations across all wells
        return _ManagerPropertyProxy(self, name)

    def load_las(
        self,
        filepath: Union[str, Path, list[Union[str, Path]]],
        sampled: bool = False
    ) -> 'WellDataManager':
        """
        Load LAS file(s), auto-create well if needed.

        Parameters
        ----------
        filepath : Union[str, Path, list[Union[str, Path]]]
            Path to LAS file or list of paths to LAS files
        sampled : bool, default False
            If True, mark all properties from the LAS file(s) as 'sampled' type.
            Use this for core plug data or other point measurements.

        Returns
        -------
        WellDataManager
            Self for method chaining

        Raises
        ------
        LasFileError
            If LAS file has no well name

        Examples
        --------
        >>> manager = WellDataManager()
        >>> manager.load_las("well1.las")
        >>> manager.load_las(["well2.las", "well3.las"])
        >>> # Load core plug data
        >>> manager.load_las("core_data.las", sampled=True)
        >>> well = manager.well_12_3_2_B
        """
        # Handle list of files
        if isinstance(filepath, list):
            for file in filepath:
                self.load_las(file, sampled=sampled)
            return self

        # Handle single file
        las = LasFile(filepath)
        well_name = las.well_name

        if well_name is None:
            raise LasFileError(
                f"LAS file {filepath} has no WELL name in header. "
                "Cannot determine which well to load into."
            )

        sanitized_name = sanitize_well_name(well_name)
        # Use well_ prefix for dictionary key (attribute access)
        well_key = f"well_{sanitized_name}"

        if well_key not in self._wells:
            # Create new well
            self._wells[well_key] = Well(
                name=well_name,
                sanitized_name=sanitized_name,
                parent_manager=self
            )
            self._name_mapping[well_name] = well_key

        # Load into well
        self._wells[well_key].load_las(las, sampled=sampled)

        return self  # Enable chaining

    def load_tops(
        self,
        df: pd.DataFrame,
        property_name: str = "Well_Tops",
        source_name: str = "Imported_Tops",
        well_col: Optional[str] = "Well identifier (Well name)",
        well_name: Optional[str] = None,
        discrete_col: str = "Surface",
        depth_col: str = "MD",
        x_col: Optional[str] = "X",
        y_col: Optional[str] = "Y",
        z_col: Optional[str] = "Z",
        include_coordinates: bool = False
    ) -> 'WellDataManager':
        """
        Load formation tops data from a DataFrame into wells.

        Supports three loading patterns:
        1. Multi-well: well_col specified, groups DataFrame by well column
        2. Single-well named: well_col=None, well_name specified, all data to that well
        3. Single-well default: well_col=None, well_name=None, all data to generic "Well"

        Automatically creates wells if they don't exist, converts discrete values
        to discrete integers with labels, and adds the data as a source to each well.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing tops data with columns for well name (optional), discrete values, and depth
        property_name : str, default "Well_Tops"
            Name for the discrete property (will be sanitized)
        source_name : str, default "Imported_Tops"
            Name for this source group (will be sanitized)
        well_col : str, optional, default "Well identifier (Well name)"
            Column name containing well names. Set to None for single-well loading.
        well_name : str, optional
            Well name to use when well_col=None. If both well_col and well_name are None,
            defaults to generic "Well".
        discrete_col : str, default "Surface"
            Column name containing discrete values (e.g., formation/surface names)
        depth_col : str, default "MD"
            Column name containing measured depth values
        x_col : str, optional, default "X"
            Column name for X coordinate (only used if include_coordinates=True)
        y_col : str, optional, default "Y"
            Column name for Y coordinate (only used if include_coordinates=True)
        z_col : str, optional, default "Z"
            Column name for Z coordinate (only used if include_coordinates=True)
        include_coordinates : bool, default False
            If True, include X, Y, Z coordinates as additional properties

        Returns
        -------
        WellDataManager
            Self for method chaining

        Examples
        --------
        >>> # Pattern 1: Multi-well loading (groups by well column)
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Well identifier (Well name)': ['12/3-4 A', '12/3-4 A', '12/3-4 B'],
        ...     'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Brent'],
        ...     'MD': [2850.0, 3100.0, 2860.0]
        ... })
        >>> manager = WellDataManager()
        >>> manager.load_tops(df)  # Uses default well_col
        >>>
        >>> # Pattern 2: Single-well with explicit name (no well column needed)
        >>> df_single = pd.DataFrame({
        ...     'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Cook'],
        ...     'MD': [2850.0, 3100.0, 3400.0]
        ... })
        >>> manager.load_tops(
        ...     df_single,
        ...     well_col=None,
        ...     well_name='12/3-4 A'  # Load all tops to this well
        ... )
        >>>
        >>> # Pattern 3: Single-well with default name "Well" (simplest)
        >>> manager.load_tops(df_single, well_col=None)
        >>>
        >>> # Access tops
        >>> well = manager.well_12_3_4_A
        >>> print(well.sources)  # ['Imported_Tops']
        >>> well.Imported_Tops.Well_Tops  # Discrete property with formation names
        """

        # Determine loading pattern
        if well_col is None:
            # SINGLE-WELL MODE: Load all data to one well
            # Use well_name if provided, otherwise default to "Well"
            target_well_name = well_name if well_name is not None else "Well"

            # Validate required columns (no well column needed)
            required_cols = [discrete_col, depth_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Required columns missing from DataFrame: {', '.join(missing_cols)}. "
                    f"Available columns: {', '.join(df.columns)}"
                )

            # Build global discrete label mapping (consistent across all wells)
            unique_values = sorted(df[discrete_col].unique())
            value_to_code = {value: idx for idx, value in enumerate(unique_values)}
            code_to_value = {idx: value for value, idx in value_to_code.items()}

            # Create a fake grouped structure for single well
            grouped = [(target_well_name, df)]
        else:
            # MULTI-WELL MODE: Group by well column (existing behavior)
            # Validate required columns exist
            required_cols = [well_col, discrete_col, depth_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Required columns missing from DataFrame: {', '.join(missing_cols)}. "
                    f"Available columns: {', '.join(df.columns)}"
                )

            # Build global discrete label mapping (consistent across all wells)
            unique_values = sorted(df[discrete_col].unique())
            value_to_code = {value: idx for idx, value in enumerate(unique_values)}
            code_to_value = {idx: value for value, idx in value_to_code.items()}

            # Group by well
            grouped = df.groupby(well_col)

        for well_name, well_df in grouped:
            # Get or create well
            sanitized_name = sanitize_well_name(well_name)
            # Use well_ prefix for dictionary key (attribute access)
            well_key = f"well_{sanitized_name}"

            if well_key not in self._wells:
                self._wells[well_key] = Well(
                    name=well_name,
                    sanitized_name=sanitized_name,
                    parent_manager=self
                )
                self._name_mapping[well_name] = well_key

            well = self._wells[well_key]

            # Build DataFrame for this well
            well_data = {
                'DEPT': well_df[depth_col].values,
                property_name: well_df[discrete_col].map(value_to_code).values
            }

            # Add coordinates if requested
            if include_coordinates:
                if x_col and x_col in well_df.columns:
                    well_data[x_col] = well_df[x_col].values
                if y_col and y_col in well_df.columns:
                    well_data[y_col] = well_df[y_col].values
                if z_col and z_col in well_df.columns:
                    well_data[z_col] = well_df[z_col].values

            tops_df = pd.DataFrame(well_data)

            # Build unit mappings
            unit_mappings = {'DEPT': 'm', property_name: ''}
            if include_coordinates:
                if x_col and x_col in well_df.columns:
                    unit_mappings[x_col] = 'm'
                if y_col and y_col in well_df.columns:
                    unit_mappings[y_col] = 'm'
                if z_col and z_col in well_df.columns:
                    unit_mappings[z_col] = 'm'

            # Build type mappings (discrete property, coordinates are continuous)
            type_mappings = {property_name: 'discrete'}
            if include_coordinates:
                if x_col and x_col in well_df.columns:
                    type_mappings[x_col] = 'continuous'
                if y_col and y_col in well_df.columns:
                    type_mappings[y_col] = 'continuous'
                if z_col and z_col in well_df.columns:
                    type_mappings[z_col] = 'continuous'

            # Add to well using add_dataframe with custom source name
            base_source_name = sanitize_property_name(source_name)

            # Check if source already exists and notify user of overwrite
            if base_source_name in well._sources:
                print(f"Overwriting existing source '{base_source_name}' in well '{well.name}'")

            # Create LasFile from DataFrame
            las = LasFile.from_dataframe(
                df=tops_df,
                well_name=well_name,
                source_name=base_source_name,
                unit_mappings=unit_mappings,
                type_mappings=type_mappings,
                label_mappings={property_name: code_to_value}
            )

            # Load it
            well.load_las(las)

        return self

    def load_properties(
        self,
        df: pd.DataFrame,
        source_name: str = "external_df",
        well_col: Optional[str] = "Well",
        well_name: Optional[str] = None,
        depth_col: str = "DEPT",
        unit_mappings: Optional[dict[str, str]] = None,
        type_mappings: Optional[dict[str, str]] = None,
        label_mappings: Optional[dict[str, dict[int, str]]] = None,
        resample_method: Optional[str] = None
    ) -> 'WellDataManager':
        """
        Load properties from a DataFrame into wells.

        Supports three loading patterns:
        1. Multi-well: well_col specified, groups DataFrame by well column
        2. Single-well named: well_col=None, well_name specified, all data to that well
        3. Single-well default: well_col=None, well_name=None, all data to generic "Well"

        IMPORTANT: Depth grids must be compatible. If incompatible, you must specify
        a resampling method explicitly. This prevents accidental data loss.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing properties with columns for well name (optional), depth, and properties
        source_name : str, default "external_df"
            Name for this source group (will be sanitized)
        well_col : str, optional, default "Well"
            Column name containing well names. Set to None for single-well loading.
        well_name : str, optional
            Well name to use when well_col=None. If both well_col and well_name are None,
            defaults to generic "Well".
        depth_col : str, default "DEPT"
            Column name containing measured depth values
        unit_mappings : dict[str, str], optional
            Mapping of property names to units (e.g., {'PHIE': 'v/v', 'SW': 'v/v'})
        type_mappings : dict[str, str], optional
            Mapping of property names to types: 'continuous', 'discrete', or 'sampled'
            (e.g., {'Zone': 'discrete', 'PHIE': 'continuous'})
        label_mappings : dict[str, dict[int, str]], optional
            Label mappings for discrete properties
            (e.g., {'Zone': {0: 'Top_Brent', 1: 'Top_Statfjord'}})
        resample_method : str, optional
            Method to use if depth grids are incompatible:
            - None (default): Raises error if depths incompatible
            - 'linear': Linear interpolation (for continuous properties)
            - 'nearest': Nearest neighbor (for discrete/sampled)
            - 'previous': Forward-fill / previous value (for discrete)
            - 'next': Backward-fill / next value
            Warning: Resampling sampled data (core plugs) may cause data loss.

        Returns
        -------
        WellDataManager
            Self for method chaining

        Raises
        ------
        ValueError
            If required columns are missing or if depths are incompatible and resample_method=None

        Examples
        --------
        >>> # Pattern 1: Multi-well loading (groups by well column)
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 B'],
        ...     'DEPT': [2850.0, 2851.0, 2850.5],
        ...     'CorePHIE': [0.20, 0.22, 0.19],
        ...     'CorePERM': [150, 200, 120]
        ... })
        >>> manager.load_properties(
        ...     df,
        ...     source_name='CoreData',
        ...     well_col='Well',  # Groups by this column
        ...     unit_mappings={'CorePHIE': 'v/v', 'CorePERM': 'mD'},
        ...     type_mappings={'CorePHIE': 'sampled', 'CorePERM': 'sampled'}
        ... )
        ✓ Loaded 2 properties into well '12/3-4 A' from source 'CoreData'
        ✓ Loaded 2 properties into well '12/3-4 B' from source 'CoreData'

        >>> # Pattern 2: Single-well with explicit name (no well column needed)
        >>> df_single = pd.DataFrame({
        ...     'DEPT': [2850.0, 2851.0, 2852.0],
        ...     'PHIE': [0.20, 0.22, 0.19]
        ... })
        >>> manager.load_properties(
        ...     df_single,
        ...     well_col=None,
        ...     well_name='12/3-4 A',  # Load all data to this well
        ...     source_name='Interpreted'
        ... )
        ✓ Loaded 1 properties into well '12/3-4 A' from source 'Interpreted'

        >>> # Pattern 3: Single-well with default name "Well" (simplest)
        >>> manager.load_properties(
        ...     df_single,
        ...     well_col=None,  # No well column
        ...     source_name='Analysis'
        ... )
        ✓ Loaded 1 properties into well 'Well' from source 'Analysis'

        >>> # Load with incompatible depths - requires explicit resampling
        >>> manager.load_properties(
        ...     df,
        ...     source_name='Interpreted',
        ...     resample_method='linear'  # Explicitly allow resampling
        ... )

        >>> # Access the data
        >>> well = manager.well_12_3_4_A
        >>> print(well.sources)  # ['Petrophysics', 'CoreData']
        >>> well.CoreData.CorePHIE  # Sampled property
        """

        # Determine loading pattern
        if well_col is None:
            # SINGLE-WELL MODE: Load all data to one well
            # Use well_name if provided, otherwise default to "Well"
            target_well_name = well_name if well_name is not None else "Well"

            # Validate depth column exists
            if depth_col not in df.columns:
                raise ValueError(
                    f"Required column '{depth_col}' missing from DataFrame. "
                    f"Available columns: {', '.join(df.columns)}"
                )

            # Get property columns (all except depth)
            prop_cols = [col for col in df.columns if col != depth_col]

            if not prop_cols:
                raise ValueError(
                    f"No property columns found in DataFrame. "
                    f"DataFrame must have columns other than '{depth_col}'."
                )

            # Create a fake grouped structure for single well
            grouped = [(target_well_name, df)]
        else:
            # MULTI-WELL MODE: Group by well column (existing behavior)
            # Validate required columns exist
            required_cols = [well_col, depth_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Required columns missing from DataFrame: {', '.join(missing_cols)}. "
                    f"Available columns: {', '.join(df.columns)}"
                )

            # Get property columns (all except well and depth)
            prop_cols = [col for col in df.columns if col not in [well_col, depth_col]]

            if not prop_cols:
                raise ValueError(
                    f"No property columns found in DataFrame. "
                    f"DataFrame must have columns other than '{well_col}' and '{depth_col}'."
                )

            # Group by well
            grouped = df.groupby(well_col)

        # Set defaults for mappings
        unit_mappings = unit_mappings or {}
        type_mappings = type_mappings or {}
        label_mappings = label_mappings or {}

        for well_name, well_df in grouped:
            # Get or create well
            sanitized_name = sanitize_well_name(well_name)
            # Use well_ prefix for dictionary key (attribute access)
            well_key = f"well_{sanitized_name}"

            if well_key not in self._wells:
                self._wells[well_key] = Well(
                    name=well_name,
                    sanitized_name=sanitized_name,
                    parent_manager=self
                )
                self._name_mapping[well_name] = well_key

            well = self._wells[well_key]

            # Build DataFrame for this well (rename depth column to DEPT)
            well_data = {'DEPT': well_df[depth_col].values}
            for prop_col in prop_cols:
                well_data[prop_col] = well_df[prop_col].values

            props_df = pd.DataFrame(well_data)

            # Build unit mappings (include DEPT)
            full_unit_mappings = {'DEPT': unit_mappings.get(depth_col, 'm')}
            for prop_col in prop_cols:
                full_unit_mappings[prop_col] = unit_mappings.get(prop_col, '')

            # Build type mappings
            full_type_mappings = {}
            for prop_col in prop_cols:
                full_type_mappings[prop_col] = type_mappings.get(prop_col, 'continuous')

            # Sanitize source name
            base_source_name = sanitize_property_name(source_name)

            # Check if source already exists and notify user of overwrite
            if base_source_name in well._sources:
                print(f"⚠ Overwriting existing source '{base_source_name}' in well '{well.name}'")

            # Create LasFile from DataFrame
            las = LasFile.from_dataframe(
                df=props_df,
                well_name=well_name,
                source_name=base_source_name,
                unit_mappings=full_unit_mappings,
                type_mappings=full_type_mappings,
                label_mappings=label_mappings
            )

            # Check compatibility if well already has data
            if well._sources:
                # Get an existing LAS file to check compatibility
                existing_source = list(well._sources.values())[0]
                existing_las = existing_source['las_file']
                compatibility = las.check_depth_compatibility(existing_las)

                if not compatibility['compatible']:
                    if resample_method is None:
                        # Strict mode - raise error and suggest resampling method
                        raise ValueError(
                            f"Depth grid incompatible for well '{well.name}': {compatibility['reason']}\n"
                            f"Existing: {compatibility['existing']['samples']} samples "
                            f"({compatibility['existing']['start']:.2f}-{compatibility['existing']['stop']:.2f}m, "
                            f"{compatibility['existing']['spacing']:.4f}m spacing)\n"
                            f"New data: {compatibility['new']['samples']} samples "
                            f"({compatibility['new']['start']:.2f}-{compatibility['new']['stop']:.2f}m, "
                            f"{compatibility['new']['spacing']:.4f}m spacing)\n\n"
                            f"To merge incompatible grids, specify a resampling method:\n"
                            f"  resample_method='linear'    # For continuous properties\n"
                            f"  resample_method='nearest'   # For discrete/sampled properties\n"
                            f"  resample_method='previous'  # Forward-fill for discrete\n"
                            f"  resample_method='next'      # Backward-fill\n\n"
                            f"WARNING: Resampling sampled data (core plugs) may cause data loss."
                        )
                    else:
                        # Resampling method specified - warn and proceed
                        warnings.warn(
                            f"Resampling new data to existing grid using method '{resample_method}' "
                            f"for well '{well.name}'. This may cause data loss for sampled properties.",
                            UserWarning
                        )

            # Load it (with resampling if specified)
            well.load_las(las, resample_method=resample_method)

            print(f"✓ Loaded {len(prop_cols)} properties into well '{well.name}' from source '{base_source_name}'")

        return self

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save all wells and their sources to a project folder structure.

        Creates a folder for each well (well_xxx format) and exports all sources
        as LAS files with well name prefix. Also saves templates to a templates/
        folder at the project root. Also renames LAS files for any sources
        that were renamed using rename_source(), and deletes LAS files for any
        sources that were removed using remove_source(). If path is not provided,
        uses the path from the last load() call.

        Parameters
        ----------
        path : Union[str, Path], optional
            Root directory path for the project. If None, uses path from last load().

        Raises
        ------
        ValueError
            If path is None and no project has been loaded

        Examples
        --------
        >>> manager = WellDataManager()
        >>> manager.load_las(["well1.las", "well2.las"])
        >>> manager.save("my_project")
        # Creates (hyphens preserved in filenames):
        # my_project/
        #   well_36_7_5_A/
        #     36_7-5_A_Log.las
        #     36_7-5_A_CorePor.las
        #   well_36_7_5_B/
        #     36_7-5_B_Log.las
        #   templates/
        #     reservoir.json
        #     qc.json
        >>>
        >>> # After load(), can save without path
        >>> manager = WellDataManager()
        >>> manager.load("my_project")
        >>> # ... make changes ...
        >>> manager.save()  # Saves to "my_project"
        >>>
        >>> # Rename and remove sources, then save
        >>> manager.well_36_7_5_A.rename_source("Log", "Wireline")
        >>> manager.well_36_7_5_A.remove_source("CorePor")
        >>> manager.save()  # Renames 36_7-5_A_Log.las to 36_7-5_A_Wireline.las and deletes 36_7-5_A_CorePor.las
        """
        # Determine path to use
        if path is None:
            if self._project_path is None:
                raise ValueError(
                    "No path provided and no project has been loaded. "
                    "Either provide a path: save('path/to/project') or "
                    "load a project first: load('path/to/project')"
                )
            save_path = self._project_path
        else:
            save_path = Path(path)

        save_path.mkdir(parents=True, exist_ok=True)

        # Save wells
        for well_key, well in self._wells.items():
            # Create well folder (well_key already has well_ prefix)
            well_folder = save_path / well_key
            well_folder.mkdir(exist_ok=True)

            # Export each source (creates files with current names)
            well.export_sources(well_folder)

            # Delete old files from renamed sources
            well.delete_renamed_sources(well_folder)

            # Delete sources marked for deletion
            well.delete_marked_sources(well_folder)

        # Save templates
        if self._templates:
            templates_folder = save_path / "templates"
            templates_folder.mkdir(exist_ok=True)

            for template_name, template in self._templates.items():
                template_file = templates_folder / f"{template_name}.json"
                template.save(template_file)

    def load(self, path: Union[str, Path]) -> 'WellDataManager':
        """
        Load all wells and templates from a project folder structure.

        Automatically discovers and loads:
        - All LAS files from well folders (well_* format)
        - All template JSON files from templates/ folder

        Stores the project path for subsequent save() calls.
        Clears any existing wells and templates before loading.

        Parameters
        ----------
        path : Union[str, Path]
            Root directory path of the project

        Returns
        -------
        WellDataManager
            Self for method chaining

        Examples
        --------
        >>> manager = WellDataManager()
        >>> manager.load("my_project")
        >>> print(manager.wells)  # All wells from project
        >>> print(manager.list_templates())  # All templates from project
        >>> # ... make changes ...
        >>> manager.save()  # Saves back to "my_project"

        >>> # Load clears existing data
        >>> manager.load("other_project")  # Replaces current wells and templates
        """
        base_path = Path(path)

        if not base_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {path}")

        # Clear existing data before loading new project
        self._wells.clear()
        self._name_mapping.clear()
        self._templates.clear()

        # Store project path for save()
        self._project_path = base_path

        # Load templates if templates folder exists
        templates_folder = base_path / "templates"
        if templates_folder.exists() and templates_folder.is_dir():
            from .visualization import Template
            template_files = sorted(templates_folder.glob("*.json"))
            for template_file in template_files:
                try:
                    template = Template.load(template_file)
                    # Use filename (without extension) as template name
                    template_name = template_file.stem
                    self._templates[template_name] = template
                except Exception as e:
                    warnings.warn(f"Could not load template {template_file.name}: {e}")

        # Find all well folders (well_*) - skip templates folder
        well_folders = sorted([
            folder for folder in base_path.glob("well_*")
            if folder.is_dir() and folder.name != "templates"
        ])

        if not well_folders:
            # Try loading all LAS files directly if no well folders
            las_files = list(base_path.glob("*.las"))
            if las_files:
                for las_file in las_files:
                    self.load_las(las_file)
            return self

        # Load from well folders
        for well_folder in well_folders:
            # Find all LAS files in this folder
            las_files = sorted(well_folder.glob("*.las"))
            for las_file in las_files:
                self.load_las(las_file)

        return self

    def add_well(self, well_name: str) -> Well:
        """
        Create or get existing well.

        Parameters
        ----------
        well_name : str
            Original well name

        Returns
        -------
        Well
            New or existing well instance

        Examples
        --------
        >>> well = manager.add_well("12/3-2 B")
        >>> well.load_las("log1.las")
        """
        sanitized_name = sanitize_well_name(well_name)
        # Use well_ prefix for dictionary key (attribute access)
        well_key = f"well_{sanitized_name}"

        if well_key not in self._wells:
            self._wells[well_key] = Well(
                name=well_name,
                sanitized_name=sanitized_name,
                parent_manager=self
            )
            self._name_mapping[well_name] = well_key

        return self._wells[well_key]

    @property
    def wells(self) -> list[str]:
        """
        List of sanitized well names.
        
        Returns
        -------
        list[str]
            List of well names (sanitized for attribute access)
        
        Examples
        --------
        >>> manager.wells
        ['well_12_3_2_B', 'well_12_3_2_A']
        """
        return list(self._wells.keys())
    
    def get_well(self, name: str) -> Well:
        """
        Get well by original or sanitized name.

        Parameters
        ----------
        name : str
            Either original name ("36/7-5 A"), sanitized ("36_7_5_A"),
            or with well_ prefix ("well_36_7_5_A")

        Returns
        -------
        Well
            The requested well

        Raises
        ------
        KeyError
            If well not found

        Examples
        --------
        >>> well = manager.get_well("36/7-5 A")
        >>> well = manager.get_well("36_7_5_A")
        >>> well = manager.get_well("well_36_7_5_A")
        """
        # Try as-is (might be well_xxx format)
        if name in self._wells:
            return self._wells[name]

        # Try adding well_ prefix
        if not name.startswith('well_'):
            well_key = f"well_{name}"
            if well_key in self._wells:
                return self._wells[well_key]

        # Try as original name
        sanitized = sanitize_well_name(name)
        well_key = f"well_{sanitized}"
        if well_key in self._wells:
            return self._wells[well_key]

        # Not found
        available = ', '.join(self._wells.keys())
        raise KeyError(
            f"Well '{name}' not found. "
            f"Available wells: {available or 'none'}"
        )
    
    def remove_well(self, name: str) -> None:
        """
        Remove a well from the manager.

        Parameters
        ----------
        name : str
            Well name (original, sanitized, or with well_ prefix)

        Examples
        --------
        >>> manager.remove_well("36/7-5 A")
        >>> manager.remove_well("well_36_7_5_A")
        """
        # Find the well
        well = self.get_well(name)
        well_key = f"well_{well.sanitized_name}"

        # Remove from mappings
        del self._wells[well_key]
        if well.name in self._name_mapping:
            del self._name_mapping[well.name]

    def set_template(self, name: str, template: Union['Template', dict]) -> None:
        """
        Store a display template in the manager.

        Templates can be referenced by name when creating WellView displays.

        Parameters
        ----------
        name : str
            Template name for reference
        template : Union[Template, dict]
            Template object or dictionary configuration

        Examples
        --------
        >>> from well_log_toolkit.visualization import Template
        >>>
        >>> # Create and store template
        >>> template = Template("reservoir")
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{"name": "GR", "x_range": [0, 150], "color": "green"}]
        ... )
        >>> manager.set_template("reservoir", template)
        >>>
        >>> # Use template in WellView
        >>> from well_log_toolkit.visualization import WellView
        >>> view = manager.well_36_7_5_A.WellView(
        ...     depth_range=[2800, 3000],
        ...     template="reservoir"
        ... )
        """
        from .visualization import Template

        if isinstance(template, dict):
            template = Template.from_dict(template)
        elif not isinstance(template, Template):
            raise TypeError(f"template must be Template or dict, got {type(template).__name__}")

        self._templates[name] = template

    def get_template(self, name: str) -> 'Template':
        """
        Get a stored template by name.

        Parameters
        ----------
        name : str
            Template name

        Returns
        -------
        Template
            The requested template

        Raises
        ------
        KeyError
            If template not found

        Examples
        --------
        >>> template = manager.get_template("reservoir")
        >>> print(template.tracks)
        """
        if name not in self._templates:
            available = ', '.join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. "
                f"Available templates: {available or 'none'}"
            )
        return self._templates[name]

    def list_templates(self) -> list[str]:
        """
        List all stored template names.

        Returns
        -------
        list[str]
            List of template names

        Examples
        --------
        >>> manager.list_templates()
        ['reservoir', 'qc', 'basic']
        """
        return list(self._templates.keys())

    def remove_template(self, name: str) -> None:
        """
        Remove a stored template.

        Parameters
        ----------
        name : str
            Template name to remove

        Examples
        --------
        >>> manager.remove_template("old_template")
        """
        if name in self._templates:
            del self._templates[name]
        else:
            available = ', '.join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. "
                f"Available templates: {available or 'none'}"
            )

    def Crossplot(
        self,
        x: str,
        y: str,
        wells: Optional[list[str]] = None,
        shape: str = "well",
        color: Optional[str] = None,
        size: Optional[str] = None,
        colortemplate: str = "viridis",
        color_range: Optional[tuple[float, float]] = None,
        size_range: tuple[float, float] = (20, 200),
        title: str = "Cross Plot",
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        figsize: tuple[float, float] = (10, 8),
        dpi: int = 100,
        marker: str = "o",
        marker_size: float = 50,
        marker_alpha: float = 0.7,
        edge_color: str = "black",
        edge_width: float = 0.5,
        x_log: bool = False,
        y_log: bool = False,
        grid: bool = True,
        grid_alpha: float = 0.3,
        depth_range: Optional[tuple[float, float]] = None,
        show_colorbar: bool = True,
        show_legend: bool = True,
        regression: Optional[Union[str, dict]] = None,
        regression_by_color: Optional[Union[str, dict]] = None,
        regression_by_group: Optional[Union[str, dict]] = None,
    ) -> 'Crossplot':
        """
        Create a multi-well crossplot.

        Parameters
        ----------
        x : str
            Name of property for x-axis
        y : str
            Name of property for y-axis
        wells : list[str], optional
            List of well names to include. If None, uses all wells.
            Default: None (all wells)
        shape : str, optional
            Property name for shape mapping. Use "well" to map shapes by well name.
            Default: "well" (each well gets different marker)
        color : str, optional
            Property name for color mapping. Use "depth" to color by depth.
            Default: None (color by well if shape="well", else single color)
        size : str, optional
            Property name for size mapping.
            Default: None (constant size)
        colortemplate : str, optional
            Matplotlib colormap name (e.g., "viridis", "plasma", "coolwarm")
            Default: "viridis"
        color_range : tuple[float, float], optional
            Min and max values for color mapping. If None, uses data range.
            Default: None
        size_range : tuple[float, float], optional
            Min and max marker sizes for size mapping.
            Default: (20, 200)
        title : str, optional
            Plot title. Default: "Cross Plot"
        xlabel : str, optional
            X-axis label. If None, uses property name.
        ylabel : str, optional
            Y-axis label. If None, uses property name.
        figsize : tuple[float, float], optional
            Figure size (width, height) in inches. Default: (10, 8)
        dpi : int, optional
            Figure resolution. Default: 100
        marker : str, optional
            Base marker style (used when shape mapping is not "well"). Default: "o"
        marker_size : float, optional
            Base marker size. Default: 50
        marker_alpha : float, optional
            Marker transparency (0-1). Default: 0.7
        edge_color : str, optional
            Marker edge color. Default: "black"
        edge_width : float, optional
            Marker edge width. Default: 0.5
        x_log : bool, optional
            Use logarithmic scale for x-axis. Default: False
        y_log : bool, optional
            Use logarithmic scale for y-axis. Default: False
        grid : bool, optional
            Show grid. Default: True
        grid_alpha : float, optional
            Grid transparency. Default: 0.3
        depth_range : tuple[float, float], optional
            Depth range to filter data. Default: None (all depths)
        show_colorbar : bool, optional
            Show colorbar when using color mapping. Default: True
        show_legend : bool, optional
            Show legend. Default: True
        regression : str or dict, optional
            Regression type to apply to all data points. Can be a string (e.g., "linear") or
            dict with keys: type, line_color, line_width, line_style, line_alpha, x_range.
            Default: None
        regression_by_color : str or dict, optional
            Regression type to apply separately for each color group in the plot. Creates
            separate regression lines based on what determines colors in the visualization:
            explicit color mapping if specified, otherwise shape groups (e.g., wells when
            shape='well'). Accepts string or dict format. Default: None
        regression_by_group : str or dict, optional
            Regression type to apply separately for each well. Creates separate
            regression lines for each well. Accepts string or dict format.
            Default: None

        Returns
        -------
        Crossplot
            Crossplot visualization object

        Examples
        --------
        Multi-well crossplot with each well as different marker:

        >>> plot = manager.Crossplot(x="RHOB", y="NPHI", shape="well")
        >>> plot.show()

        Specific wells with color and size mapping:

        >>> plot = manager.Crossplot(
        ...     x="PHIE_2025",
        ...     y="NetSand_2025",
        ...     wells=["Well_A", "Well_B"],
        ...     color="depth",
        ...     size="Sw_2025",
        ...     colortemplate="viridis",
        ...     color_range=[2000, 2500],
        ...     title="Multi-Well Cross Plot"
        ... )
        >>> plot.show()

        With regression analysis:

        >>> plot = manager.Crossplot(x="RHOB", y="NPHI")
        >>> plot.add_regression("linear", line_color="red")
        >>> plot.add_regression("polynomial", degree=2, line_color="blue")
        >>> plot.show()
        """
        from .visualization import Crossplot as CrossplotClass

        # Get well objects
        if wells is None:
            well_objects = list(self._wells.values())
        else:
            well_objects = []
            for well_name in wells:
                well = self.get_well(well_name)
                if well is None:
                    raise ValueError(f"Well '{well_name}' not found")
                well_objects.append(well)

        if not well_objects:
            raise ValueError("No wells available for crossplot")

        return CrossplotClass(
            wells=well_objects,
            x=x,
            y=y,
            shape=shape,
            color=color,
            size=size,
            colortemplate=colortemplate,
            color_range=color_range,
            size_range=size_range,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            dpi=dpi,
            marker=marker,
            marker_size=marker_size,
            marker_alpha=marker_alpha,
            edge_color=edge_color,
            edge_width=edge_width,
            x_log=x_log,
            y_log=y_log,
            grid=grid,
            grid_alpha=grid_alpha,
            depth_range=depth_range,
            show_colorbar=show_colorbar,
            show_legend=show_legend,
            regression=regression,
            regression_by_color=regression_by_color,
            regression_by_group=regression_by_group,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"WellDataManager(wells={len(self._wells)})"