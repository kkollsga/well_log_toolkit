"""
Proxy objects for manager-level property operations.

Provides _ManagerPropertyProxy and _ManagerMultiPropertyProxy which enable
broadcasting property operations across all wells in a WellDataManager.
"""

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from ..analysis.statistics import compute_intervals
from ..analysis.sums_avg import SumsAvgResult, _flatten_to_dataframe, _sanitize_for_json
from ..core.property import Property
from ..exceptions import PropertyNotFoundError, PropertyTypeError
from ..utils import emit_status, suggest_similar_names

if TYPE_CHECKING:
    from .data_manager import WellDataManager


class _ManagerPropertyProxy:
    """
    Proxy object for manager-level property operations.

    This proxy enables broadcasting property operations across all wells:
        manager.PHIE_scaled = manager.PHIE * 0.01

    The proxy is created when accessing a property name on the manager,
    and operations on the proxy create new proxies that remember the operation.
    When assigned to a manager attribute, the operation is broadcast to all wells.
    """

    def __init__(
        self,
        manager: "WellDataManager",
        property_name: str,
        operation=None,
        filters=None,
        custom_intervals=None,
    ):
        self._manager = manager
        self._property_name = property_name
        self._operation = operation  # Function to apply to each property
        self._filters = filters or []  # List of (filter_name, insert_boundaries) tuples
        self._custom_intervals = (
            custom_intervals  # For filter_intervals: str (saved name) or dict (well-specific)
        )
        self._cache: dict = {}

    def _cached(self, key: tuple, compute):
        """Return cached result or compute, cache, and return."""
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

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
        intervals = intervals_config["intervals"]
        name = intervals_config["name"]
        insert_boundaries = intervals_config["insert_boundaries"]
        save = intervals_config["save"]

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
            well_intervals, name=name, insert_boundaries=insert_boundaries, save=save
        )

    def _create_proxy_with_operation(self, operation):
        """Create a new proxy with an operation."""
        return _ManagerPropertyProxy(
            self._manager, self._property_name, operation, self._filters, self._custom_intervals
        )

    def _extract_statistic_from_grouped(
        self, grouped_result: dict, stat_name: str, **kwargs
    ) -> dict:
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
        if "mean" in grouped_result or "samples" in grouped_result:
            # This is a stats dict - extract the requested statistic
            if stat_name == "percentile":
                # Percentile is nested under 'percentile' dict
                percentile_key = kwargs.get("percentile_key", "p50")
                if (
                    "percentile" in grouped_result
                    and percentile_key in grouped_result["percentile"]
                ):
                    value = grouped_result["percentile"][percentile_key]
                    # If value is a dict with 'weighted'/'arithmetic', prefer 'weighted'
                    if isinstance(value, dict):
                        return value.get("weighted", value.get("arithmetic", None))
                    return value
                return None
            elif stat_name == "range_min":
                if "range" in grouped_result:
                    value = grouped_result["range"]["min"]
                    if isinstance(value, dict):
                        return value.get("weighted", value.get("arithmetic", None))
                    return value
                return None
            elif stat_name == "range_max":
                if "range" in grouped_result:
                    value = grouped_result["range"]["max"]
                    if isinstance(value, dict):
                        return value.get("weighted", value.get("arithmetic", None))
                    return value
                return None
            else:
                # Direct statistic (mean, median, mode, std_dev, etc.)
                value = grouped_result.get(stat_name, None)
                # If value is a dict with 'weighted'/'arithmetic', prefer 'weighted'
                if isinstance(value, dict) and ("weighted" in value or "arithmetic" in value):
                    return value.get("weighted", value.get("arithmetic", None))
                return value
        else:
            # This is a grouping level - recurse
            result = {}
            for key, value in grouped_result.items():
                extracted = self._extract_statistic_from_grouped(value, stat_name, **kwargs)
                if extracted is not None:
                    result[key] = extracted
            return result if result else None

    def _warn_skipped_wells(self, result: dict) -> None:
        """Warn if any wells were skipped because they lack this property."""
        all_wells = set(self._manager._wells.keys())
        included = set(result.keys())
        skipped = all_wells - included
        if skipped:
            names = ", ".join(sorted(skipped))
            warnings.warn(
                f"Skipped {len(skipped)} well(s) without property "
                f"'{self._property_name}': {names}",
                stacklevel=3,
            )

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
        return self._create_proxy_with_operation(lambda p: p**other)

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
        for _well_name, well in self._manager._wells.items():
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
        for _well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.type = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            emit_status(
                f"✓ Set type='{value}' for property '{self._property_name}' in {count} well(s)"
            )

    @property
    def labels(self):
        """Get labels from first well with this property."""
        for _well_name, well in self._manager._wells.items():
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
        for _well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                # Auto-set type to discrete if labels are being set
                if prop.type != "discrete":
                    prop.type = "discrete"
                prop.labels = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            emit_status(f"✓ Set labels for property '{self._property_name}' in {count} well(s)")

    @property
    def colors(self):
        """Get colors from first well with this property."""
        for _well_name, well in self._manager._wells.items():
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
        for _well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.colors = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            emit_status(f"✓ Set colors for property '{self._property_name}' in {count} well(s)")

    @property
    def styles(self):
        """Get styles from first well with this property."""
        for _well_name, well in self._manager._wells.items():
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
        for _well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.styles = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            emit_status(f"✓ Set styles for property '{self._property_name}' in {count} well(s)")

    @property
    def thicknesses(self):
        """Get thicknesses from first well with this property."""
        for _well_name, well in self._manager._wells.items():
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
        for _well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
                prop.thicknesses = value
                count += 1
            except (AttributeError, PropertyNotFoundError):
                pass
        if count > 0:
            emit_status(
                f"✓ Set thicknesses for property '{self._property_name}' in {count} well(s)"
            )

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
        cache_key = ("min", nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=True, arithmetic=None, precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, "range_min")
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single min per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(well, lambda prop: prop.min(), nested=nested)
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
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
        cache_key = ("max", nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well, weighted=True, arithmetic=None, precision=6, nested=nested
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, "range_max")
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single max per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(well, lambda prop: prop.max(), nested=nested)
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
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

        See Also
        --------
        std : Compute standard deviation across wells.
        sums_avg : Compute full grouped statistics.
        """
        cache_key = ("mean", weighted, nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well,
                    weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6,
                    nested=nested,
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, "mean")
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single mean per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(
                    well, lambda prop: prop.mean(weighted=weighted), nested=nested
                )
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
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
        cache_key = ("std", weighted, nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well,
                    weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6,
                    nested=nested,
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, "std_dev")
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single std per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(
                    well, lambda prop: prop.std(weighted=weighted), nested=nested
                )
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
        return result

    def percentile(
        self, p: float, weighted: bool = True, nested: bool = False, return_df: bool = False
    ):
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
        cache_key = ("percentile", p, weighted, nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics (like sums_avg)
        if self._filters:
            result = {}
            percentile_key = f"p{int(p)}"  # e.g., 'p50', 'p90'

            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well,
                    weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6,
                    nested=nested,
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(
                        well_result, "percentile", percentile_key=percentile_key
                    )
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single percentile per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(
                    well, lambda prop: prop.percentile(p, weighted=weighted), nested=nested
                )
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
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
        cache_key = ("median", weighted, nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics (median = p50)
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well,
                    weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6,
                    nested=nested,
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, "median")
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single median per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(
                    well, lambda prop: prop.median(weighted=weighted), nested=nested
                )
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
        return result

    def mode(
        self, weighted: bool = True, bins: int = 50, nested: bool = False, return_df: bool = False
    ):
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
        cache_key = ("mode", weighted, bins, nested, return_df)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If filters are applied, use grouped statistics
        if self._filters:
            result = {}
            for well_name, well in self._manager._wells.items():
                well_result = self._compute_sums_avg_for_well(
                    well,
                    weighted=weighted if weighted else None,
                    arithmetic=not weighted if not weighted else None,
                    precision=6,
                    nested=nested,
                )
                if well_result is not None:
                    extracted = self._extract_statistic_from_grouped(well_result, "mode")
                    if extracted is not None:
                        result[well_name] = extracted

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result:
                result = _flatten_to_dataframe(result, self._property_name)
        else:
            # No filters - compute single mode per well
            result = {}
            for well_name, well in self._manager._wells.items():
                value = self._compute_for_well(
                    well, lambda prop: prop.mode(weighted=weighted, bins=bins), nested=nested
                )
                if value is not None:
                    result[well_name] = value

            result = _sanitize_for_json(result)
            self._warn_skipped_wells(result)
            if return_df and result and any(isinstance(v, dict) for v in result.values()):
                result = _flatten_to_dataframe(result, self._property_name)

        self._cache[cache_key] = result
        return result

    def stats(
        self,
        methods=None,
        weighted: bool = True,
        return_df: bool = False,
        flat_columns: bool = False,
        pool: bool = False,
    ):
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
        # If pooling is requested, take the long-form data path and aggregate
        # across wells with pandas — gives statistically correct cross-well
        # mean / std / min / max / percentiles. Returns a DataFrame with one
        # row per filter-group (instead of per (well, group)).
        if pool:
            return self._pooled_stats(methods, flat_columns)

        # Define default statistics
        default_methods = [
            "mean",
            "median",
            "std",
            "min",
            "max",
            "percentile_10",
            "percentile_50",
            "percentile_90",
        ]

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
            if method.startswith("percentile_"):
                percentile = int(method.split("_")[1])
                stat_result = self.percentile(percentile, weighted=weighted, return_df=False)
                results[f"p{percentile}"] = stat_result
            elif method == "mean":
                results["mean"] = self.mean(weighted=weighted, return_df=False)
            elif method == "median":
                results["median"] = self.median(weighted=weighted, return_df=False)
            elif method == "std":
                results["std"] = self.std(weighted=weighted, return_df=False)
            elif method == "min":
                results["min"] = self.min(return_df=False)
            elif method == "max":
                results["max"] = self.max(return_df=False)
            elif method == "mode":
                results["mode"] = self.mode(weighted=weighted, return_df=False)
            else:
                raise ValueError(f"Unknown statistic: {method}")

        if not return_df:
            return results

        # Convert to DataFrame with statistics as columns
        # First, flatten each statistic to get rows
        group_names = [name for name, _ in self._filters] if flat_columns else None
        dfs = []
        for stat_name, stat_dict in results.items():
            df = _flatten_to_dataframe(stat_dict, stat_name, group_names=group_names)
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
            merged = pd.merge(merged, df, on=grouping_cols, how="outer")

        return merged

    def _pooled_stats(self, methods, flat_columns: bool) -> pd.DataFrame:
        """Cross-well pooled stats from the long-form ``.data()`` output.

        Returns a DataFrame with one row per filter-group combination
        (or a single row if no filters are active). Columns: filter
        properties, then the requested stats. ``flat_columns=True`` uses
        property names; ``False`` uses ``Group``/``Group1``/... like
        the per-well path.

        Currently unweighted — for depth-weighted pooled stats, use
        ``self.data(weighted=True)`` and aggregate manually.
        """
        default_methods = [
            "samples",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "percentile_10",
            "percentile_50",
            "percentile_90",
        ]
        if methods is None:
            stat_methods = default_methods
        elif isinstance(methods, str):
            stat_methods = [methods]
        elif isinstance(methods, list):
            stat_methods = methods
        else:
            raise ValueError("methods must be None, str, or list of str")

        df = self.data(warn_missing=False)
        if df.empty:
            return pd.DataFrame()

        target = self._property_name
        group_cols_raw = [name for name, _ in self._filters]

        # Build the agg spec.
        agg_funcs: dict = {}
        for method in stat_methods:
            if method == "samples":
                agg_funcs["samples"] = (target, "size")
            elif method == "mean":
                agg_funcs["mean"] = (target, "mean")
            elif method == "median":
                agg_funcs["median"] = (target, "median")
            elif method == "std":
                agg_funcs["std"] = (target, "std")
            elif method == "min":
                agg_funcs["min"] = (target, "min")
            elif method == "max":
                agg_funcs["max"] = (target, "max")
            elif method.startswith("percentile_"):
                p = int(method.split("_")[1])
                agg_funcs[f"p{p}"] = (target, lambda v, p=p: v.quantile(p / 100.0))
            elif len(method) > 1 and method[0] in {"p", "P"} and method[1:].isdigit():
                # Short form: "p10", "p50", "p90" — accepted as input AND output.
                p = int(method[1:])
                agg_funcs[f"p{p}"] = (target, lambda v, p=p: v.quantile(p / 100.0))
            else:
                raise ValueError(f"Unknown statistic: {method}")

        if not group_cols_raw:
            # No filters — single overall row.
            single_row = {}
            for col_name, (col, agg) in agg_funcs.items():
                series = df[col]
                if callable(agg):
                    single_row[col_name] = agg(series)
                elif agg == "size":
                    single_row[col_name] = len(series)
                else:
                    single_row[col_name] = getattr(series, agg)()
            return pd.DataFrame([single_row])

        if flat_columns:
            group_cols = group_cols_raw
        elif len(group_cols_raw) == 1:
            group_cols = ["Group"]
            df = df.rename(columns={group_cols_raw[0]: "Group"})
        else:
            group_cols = [f"Group{i}" for i in range(1, len(group_cols_raw) + 1)]
            rename_map = dict(zip(group_cols_raw, group_cols, strict=True))
            df = df.rename(columns=rename_map)

        return df.groupby(group_cols, sort=False).agg(**agg_funcs).reset_index()

    def data(
        self,
        weighted: bool = False,
        warn_missing: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Return long-format data across all wells as a DataFrame.

        Concatenates per-well property values with a leading ``well`` column.
        Filters and ``filter_intervals`` stored on the proxy are applied per
        well; wells that lack the property or any filter property are skipped
        (consistent with stat methods).

        Parameters
        ----------
        weighted : bool, default False
            If True, append a ``Weight`` column with the depth interval each
            row represents (half the interval before plus half after,
            edge-corrected per well). Use these to replicate depth-weighted
            statistics externally, e.g.
            ``np.average(df["PHIE"], weights=df["Weight"])``.
        warn_missing : bool, default True
            Emit a ``UserWarning`` listing wells that lacked the property
            and were skipped. Set ``False`` for batch / scripted workflows
            where missing-well skips are expected.
        **kwargs
            Forwarded to :meth:`Property.data`. Useful options include
            ``discrete_labels`` (default True — emits label strings for
            discrete filters), ``clip_edges`` (default True),
            ``clip_to_property``, ``include``, ``exclude``.

        Returns
        -------
        pd.DataFrame
            Columns: ``well``, ``DEPT``, ``<property_name>``, optional
            ``Weight``, then one column per active filter named after the
            filter property. Rows ordered by (well, DEPT). Empty DataFrame
            if no well has the property.

        Examples
        --------
        >>> manager.PHIE.filter("Zone").data().head()
             well    DEPT   PHIE          Zone
        0  Well_A  1000.0  0.150  NonReservoir
        1  Well_A  1001.0  0.157  NonReservoir
        """
        dfs = []
        included_wells: dict = {}

        for well_name, well in self._manager._wells.items():
            try:
                prop = well.get_property(self._property_name)
            except PropertyNotFoundError:
                continue

            prop = self._apply_operation(prop)
            if not isinstance(prop, Property):
                continue

            if self._custom_intervals:
                prop = self._apply_filter_intervals(prop, well)
                if prop is None:
                    continue

            skip = False
            for filter_name, insert_boundaries in self._filters:
                try:
                    if insert_boundaries is not None:
                        prop = prop.filter(filter_name, insert_boundaries=insert_boundaries)
                    else:
                        prop = prop.filter(filter_name)
                except (PropertyNotFoundError, PropertyTypeError):
                    skip = True
                    break
            if skip:
                continue

            df = prop.data(**kwargs)
            if weighted and len(df) > 1:
                df["Weight"] = compute_intervals(df["DEPT"].to_numpy())
            elif weighted:
                df["Weight"] = 0.0
            df.insert(0, "well", well.name)
            dfs.append(df)
            included_wells[well_name] = 1

        if not dfs:
            return pd.DataFrame()

        if warn_missing:
            self._warn_skipped_wells(included_wells)

        return pd.concat(dfs, ignore_index=True)

    def filter(
        self, property_name: str, insert_boundaries: bool | None = None
    ) -> "_ManagerPropertyProxy":
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
        Raises
        ------
        PropertyNotFoundError
            If no well in the manager has the named property.
        PropertyTypeError
            If the property exists but is not discrete in any well.
        """
        # Validate that the filter property exists and is discrete in at least one well
        found_in_any = False
        discrete_in_any = False
        all_property_names: set[str] = set()

        for well in self._manager._wells.values():
            for source_data in well._sources.values():
                for pname, prop in source_data["properties"].items():
                    all_property_names.add(pname)
                    if pname == property_name:
                        found_in_any = True
                        if prop.type == "discrete":
                            discrete_in_any = True

        if not found_in_any:
            suggestions = suggest_similar_names(property_name, all_property_names)
            msg = f"Filter property '{property_name}' not found in any well."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            raise PropertyNotFoundError(msg)

        if not discrete_in_any:
            raise PropertyTypeError(
                f"Filter property '{property_name}' exists but is not discrete in any well. "
                f"Set the property type first: well.get_property('{property_name}').type = 'discrete'"
            )

        # Create new filter list with this filter added
        new_filters = self._filters + [(property_name, insert_boundaries)]

        # Return new proxy with filter added
        return _ManagerPropertyProxy(
            self._manager, self._property_name, self._operation, new_filters, self._custom_intervals
        )

    def filter_intervals(
        self,
        intervals: str | dict,
        name: str = "Custom_Intervals",
        insert_boundaries: bool | None = None,
        save: str | None = None,
    ) -> "_ManagerPropertyProxy":
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
            "intervals": intervals,
            "name": name,
            "insert_boundaries": insert_boundaries,
            "save": save,
        }

        return _ManagerPropertyProxy(
            self._manager, self._property_name, self._operation, self._filters, intervals_config
        )

    def discrete_summary(self, precision: int = 6, skip: list | None = None) -> dict:
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

    def _compute_discrete_summary_for_well(self, well, precision: int, skip: list | None):
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
        weighted: bool | None = None,
        arithmetic: bool | None = None,
        precision: int = 6,
        nested: bool = False,
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

        cache_key = ("sums_avg", weighted, arithmetic, precision, nested)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {}

        for well_name, well in self._manager._wells.items():
            well_result = self._compute_sums_avg_for_well(
                well, weighted, arithmetic, precision, nested
            )
            if well_result is not None:
                result[well_name] = well_result

        result = SumsAvgResult(_sanitize_for_json(result))
        self._cache[cache_key] = result
        return result

    def _compute_property_sums_avg(
        self,
        well,
        source_name: str | None,
        weighted: bool | None,
        arithmetic: bool | None,
        precision: int,
    ) -> dict | None:
        """
        Compute sums_avg for a property in a single source.

        Parameters
        ----------
        well : Well
            Well to compute statistics for.
        source_name : str or None
            Source name to get property from. None for default (unique) lookup.
        weighted : bool or None
            Whether to use depth-weighted statistics.
        arithmetic : bool or None
            Whether to include arithmetic statistics.
        precision : int
            Decimal precision for results.

        Returns
        -------
        dict or None
            Grouped statistics, or None if property/filter not available.
        """
        prop = well.get_property(self._property_name, source=source_name)
        prop = self._apply_operation(prop)

        prop = self._apply_filter_intervals(prop, well)
        if prop is None:
            return None

        for filter_name, insert_boundaries in self._filters:
            kwargs = {"source": source_name} if source_name else {}
            if insert_boundaries is not None:
                prop = prop.filter(filter_name, insert_boundaries=insert_boundaries, **kwargs)
            else:
                prop = prop.filter(filter_name, **kwargs)

        result = prop.sums_avg(weighted=weighted, arithmetic=arithmetic, precision=precision)

        if self._custom_intervals and result:
            well_thickness = 0.0
            for _key, value in result.items():
                if isinstance(value, dict) and "thickness" in value:
                    well_thickness += value["thickness"]
            if well_thickness > 0:
                result["thickness"] = round(well_thickness, precision)

        return result

    def _compute_per_source(
        self,
        well,
        weighted: bool | None,
        arithmetic: bool | None,
        precision: int,
    ) -> dict | None:
        """Compute sums_avg for each source in a well, skipping failures."""
        source_results = {}
        for source_name in well._sources:
            try:
                result = self._compute_property_sums_avg(
                    well, source_name, weighted, arithmetic, precision
                )
                if result is not None:
                    source_results[source_name] = result
            except (
                PropertyNotFoundError,
                PropertyTypeError,
                AttributeError,
                KeyError,
                ValueError,
            ):
                pass
        return source_results if source_results else None

    def _compute_sums_avg_for_well(
        self,
        well,
        weighted: bool | None,
        arithmetic: bool | None,
        precision: int,
        nested: bool,
    ):
        """
        Helper to compute sums_avg for a property in a well.

        Applies all filters and computes grouped statistics.
        """
        if nested:
            return self._compute_per_source(well, weighted, arithmetic, precision)

        # Default behavior (nested=False) — try unique property first
        try:
            return self._compute_property_sums_avg(well, None, weighted, arithmetic, precision)
        except PropertyNotFoundError as e:
            if "ambiguous" in str(e).lower():
                return self._compute_per_source(well, weighted, arithmetic, precision)
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

    def _broadcast_to_manager(self, manager: "WellDataManager", target_name: str):
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
            emit_status(f"✓ Created property '{target_name}' in {applied_count} well(s)")
        if skipped_wells:
            warnings.warn(
                f"Skipped {len(skipped_wells)} well(s) without property '{self._property_name}': "
                f"{', '.join(skipped_wells[:3])}{'...' if len(skipped_wells) > 3 else ''}",
                UserWarning,
                stacklevel=2,
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
    PROPERTY_STATS = {"mean", "median", "mode", "sum", "std_dev", "percentile", "range"}

    # Stats that are common across properties (stay at group level)
    COMMON_STATS = {"depth_range", "samples", "thickness", "thickness_fraction", "calculation"}

    def __init__(
        self,
        manager: "WellDataManager",
        property_names: list[str],
        filters: list[tuple] | None = None,
        custom_intervals: dict | None = None,
    ):
        self._manager = manager
        self._property_names = property_names
        self._filters = filters or []
        self._custom_intervals = custom_intervals

    def __getattr__(self, name: str) -> "_ManagerMultiPropertyProxy":
        """
        Attribute access as shorthand for filter().

        Allows: manager.properties(['A', 'B']).Facies.sums_avg()
        Same as: manager.properties(['A', 'B']).filter('Facies').sums_avg()
        """
        # Avoid recursion for private attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Treat as filter
        return self.filter(name)

    def filter(
        self, property_name: str, insert_boundaries: bool | None = None
    ) -> "_ManagerMultiPropertyProxy":
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
        intervals: str | list | dict,
        name: str = "Custom_Intervals",
        insert_boundaries: bool | None = None,
        save: str | None = None,
    ) -> "_ManagerMultiPropertyProxy":
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
            "intervals": intervals,
            "name": name,
            "insert_boundaries": insert_boundaries,
            "save": save,
        }
        return _ManagerMultiPropertyProxy(
            self._manager, self._property_names, self._filters, intervals_config
        )

    def data(self, weighted: bool = False, **kwargs) -> pd.DataFrame:
        """
        Return long-format data across all wells as a DataFrame.

        Per well, all listed properties are filtered with the same active
        filters and joined on DEPT, producing one column per property.
        Filter columns are emitted once. Wells with none of the listed
        properties (or that fail the filters) are skipped.

        Parameters
        ----------
        weighted : bool, default False
            If True, append a ``Weight`` column with the depth interval each
            row represents (half the interval before plus half after,
            edge-corrected per well). Use these to replicate depth-weighted
            statistics externally.
        **kwargs
            Forwarded to :meth:`Property.data`.

        Returns
        -------
        pd.DataFrame
            Columns: ``well``, ``DEPT``, one column per property, optional
            ``Weight``, then one column per active filter. Rows ordered by
            (well, DEPT). Empty DataFrame if no well has any of the
            requested properties.

        Examples
        --------
        >>> manager.properties(["PHIE", "PERM"]).filter("Zone").data().head()
             well    DEPT   PHIE   PERM          Zone
        0  Well_A  1000.0  0.150   80.0  NonReservoir
        """
        dfs = []

        for well in self._manager._wells.values():
            filtered_props = []
            skip_well = False

            for prop_name in self._property_names:
                try:
                    prop = well.get_property(prop_name)
                except (PropertyNotFoundError, AttributeError):
                    continue

                if self._custom_intervals:
                    prop = self._apply_filter_intervals(prop, well)
                    if prop is None:
                        continue

                for filter_name, insert_boundaries in self._filters:
                    try:
                        if insert_boundaries is not None:
                            prop = prop.filter(filter_name, insert_boundaries=insert_boundaries)
                        else:
                            prop = prop.filter(filter_name)
                    except (PropertyNotFoundError, PropertyTypeError):
                        skip_well = True
                        break
                if skip_well:
                    break

                filtered_props.append(prop)

            if skip_well or not filtered_props:
                continue

            base_df = filtered_props[0].data(**kwargs)

            for prop in filtered_props[1:]:
                extra = prop.data(**kwargs)[["DEPT", prop.name]]
                base_df = base_df.merge(extra, on="DEPT", how="outer")

            prop_cols = [p.name for p in filtered_props if p.name in base_df.columns]
            filter_cols = [c for c in base_df.columns if c != "DEPT" and c not in prop_cols]
            base_df = base_df[["DEPT"] + prop_cols + filter_cols]
            base_df = base_df.sort_values("DEPT").reset_index(drop=True)

            if weighted and len(base_df) > 1:
                base_df["Weight"] = compute_intervals(base_df["DEPT"].to_numpy())
            elif weighted:
                base_df["Weight"] = 0.0

            base_df.insert(0, "well", well.name)
            dfs.append(base_df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def fit(
        self,
        model,
        name: str | None = None,
        decimals: int = 4,
        equation_format: str = "natural",
        line_color: str | None = None,
        line_width: float = 2.0,
        line_style: str = "-",
        line_alpha: float = 1.0,
    ):
        """
        Fit a regression model to the data and return a :class:`RegressionFit`.

        The proxy must hold exactly two properties; the first is treated as
        the independent variable (``x``), the second as the dependent
        variable (``y``). Pooled across all wells in the (sub-)view.

        Parameters
        ----------
        model : RegressionBase
            An unfitted regression model (e.g. ``ExponentialRegression()``).
            ``model.fit(x, y)`` is called inside.
        name : str, optional
            Display name for the resulting artifact. Defaults to
            ``"{x_property}-{y_property}"``.
        decimals, equation_format, line_color, line_width, line_style, line_alpha :
            Forwarded to :class:`RegressionFit`.

        Returns
        -------
        RegressionFit
            A fitted artifact ready to ``crossplot.add(fit)`` or pass to a
            tabular consumer.

        Raises
        ------
        ValueError
            If the proxy does not hold exactly two properties, or if the
            pooled data is empty.

        Examples
        --------
        >>> fit = manager.properties(["PHIE", "PERM"]).fit(
        ...     ExponentialRegression(), name="all wells", equation_format="petrel"
        ... )
        >>> crossplot.add(fit)
        """
        from ..analysis.regression_fit import RegressionFit

        if len(self._property_names) != 2:
            raise ValueError(
                f"fit() requires exactly 2 properties (x, y); got "
                f"{len(self._property_names)}: {self._property_names}"
            )
        df = self.data()
        if df.empty:
            raise ValueError("No data available to fit; pooled DataFrame is empty.")
        x_col, y_col = self._property_names
        model.fit(df[x_col].to_numpy(), df[y_col].to_numpy())
        return RegressionFit(
            model,
            name=name if name is not None else f"{x_col}-{y_col}",
            decimals=decimals,
            equation_format=equation_format,
            line_color=line_color,
            line_width=line_width,
            line_style=line_style,
            line_alpha=line_alpha,
        )

    def fit_per(
        self,
        group_property: str,
        model,
        min_samples: int = 5,
        decimals: int = 4,
        equation_format: str = "natural",
        line_color: str | None = None,
        line_width: float = 2.0,
        line_style: str = "-",
        line_alpha: float = 1.0,
    ) -> dict:
        """
        Fit one regression per unique value of ``group_property``.

        Pools across wells, partitions the data by ``group_property``, and
        fits a fresh copy of ``model`` on each subset. Returns a dict
        ``{label: RegressionFit}`` keyed by the group label (label string
        when ``Property.labels`` is set, else the raw value).

        Parameters
        ----------
        group_property : str
            Name of a discrete property to group by (e.g. ``"Zone"``,
            ``"Facies"``). The column is added to the proxy's filter
            chain automatically if not already present.
        model : RegressionBase
            An unfitted regression instance (e.g.
            ``ExponentialRegression()``). The instance is deep-copied
            once per group, so any ``locked_params`` / ``degree`` / etc.
            on the original are preserved across groups.
        min_samples : int, default 5
            Subsets smaller than this are skipped with a ``UserWarning``
            (no exception, no entry in the returned dict).
        decimals, equation_format, line_color, line_width, line_style, line_alpha :
            Forwarded to each :class:`RegressionFit`.

        Returns
        -------
        dict[str, RegressionFit]
            Group label → fitted artifact. Iterate to render, print, or
            feed to ``Crossplot.add(fit)`` / ``Table.add(fit)``.

        Examples
        --------
        >>> fits = manager.properties(["PHIE", "PERM"]).fit_per(
        ...     "Zone", ExponentialRegression(), equation_format="petrel"
        ... )
        >>> for label, fit in fits.items():
        ...     print(label, fit.equation())
        """
        import copy as _copy

        from ..analysis.regression_fit import RegressionFit

        if len(self._property_names) != 2:
            raise ValueError(
                f"fit_per() requires exactly 2 properties (x, y); got "
                f"{len(self._property_names)}: {self._property_names}"
            )

        # Ensure the group column ends up in the data DataFrame.
        existing = {name for name, _ in self._filters}
        proxy = self if group_property in existing else self.filter(group_property)
        df = proxy.data()
        if df.empty:
            return {}
        if group_property not in df.columns:
            raise ValueError(
                f"group_property {group_property!r} not in data columns: {list(df.columns)}"
            )

        x_col, y_col = self._property_names
        fits: dict = {}
        for label in df[group_property].dropna().unique():
            sub = df[df[group_property] == label]
            if len(sub) < min_samples:
                warnings.warn(
                    f"Subset for {group_property}={label!r} has {len(sub)} samples, "
                    f"below min_samples={min_samples}. Skipping.",
                    stacklevel=2,
                )
                continue
            sub_model = _copy.deepcopy(model)
            sub_model.fit(sub[x_col].to_numpy(), sub[y_col].to_numpy())
            fits[str(label)] = RegressionFit(
                sub_model,
                name=str(label),
                decimals=decimals,
                equation_format=equation_format,
                line_color=line_color,
                line_width=line_width,
                line_style=line_style,
                line_alpha=line_alpha,
            )
        return fits

    def sums_avg(
        self, weighted: bool | None = None, arithmetic: bool | None = None, precision: int = 6
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
            well_result = self._compute_sums_avg_for_well(well, weighted, arithmetic, precision)
            if well_result is not None:
                result[well_name] = well_result

        return SumsAvgResult(_sanitize_for_json(result))

    def _compute_sums_avg_for_well(
        self, well, weighted: bool | None, arithmetic: bool | None, precision: int
    ):
        """
        Compute multi-property sums_avg for a single well.
        """
        # Check if this well has the required saved intervals (if using saved name)
        if self._custom_intervals:
            intervals = self._custom_intervals.get("intervals")
            if isinstance(intervals, str):
                # Saved filter name - check if this well has it
                if intervals not in well._saved_filter_intervals:
                    return None  # Skip wells that don't have this saved filter
            elif isinstance(intervals, dict):
                # Well-specific intervals - check if this well is in the dict
                # Check original name, sanitized name, and well_-prefixed sanitized name
                prefixed_name = f"well_{well.sanitized_name}"
                if (
                    well.name not in intervals
                    and well.sanitized_name not in intervals
                    and prefixed_name not in intervals
                ):
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
                    weighted=weighted, arithmetic=arithmetic, precision=precision
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
            for _key, value in merged.items():
                if isinstance(value, dict) and "thickness" in value:
                    well_thickness += value["thickness"]
            merged["thickness"] = round(well_thickness, 6)

        return merged

    def _apply_filter_intervals(self, prop, well):
        """
        Apply filter_intervals to a property if custom_intervals is set.

        Returns None if the well doesn't have the required saved intervals.
        """
        if not self._custom_intervals:
            return prop

        intervals_config = self._custom_intervals
        intervals = intervals_config["intervals"]
        name = intervals_config["name"]
        insert_boundaries = intervals_config["insert_boundaries"]
        save = intervals_config["save"]

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
            well_intervals, name=name, insert_boundaries=insert_boundaries, save=save
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
                prop_stats = {k: v for k, v in prop_result.items() if k in self.PROPERTY_STATS}
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
                                k: v for k, v in prop_value.items() if k in self.PROPERTY_STATS
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
