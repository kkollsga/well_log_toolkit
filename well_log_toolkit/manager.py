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


class SumsAvgResult(dict):
    """
    Dictionary subclass for sums_avg results with reporting capabilities.

    Behaves exactly like a regular dict but adds the `.report()` method
    for generating formatted reports with cross-well aggregation.

    Examples
    --------
    >>> results = manager.properties(['PHIE', 'PERM']).filter('Facies').filter_intervals("Zones").sums_avg()
    >>> results['Well_A']  # Normal dict access
    >>> results.report(zones=[...], groups={...}, columns=[...])  # Generate report
    """

    def report(
        self,
        zones: list[str],
        groups: dict[str, list[str]],
        columns: list[dict],
        print_report: bool = True
    ) -> Optional[dict]:
        """
        Generate a structured report with cross-well aggregation.

        Parameters
        ----------
        zones : list[str]
            Zone names to include in the report
        groups : dict[str, list[str]]
            Facies grouping, e.g., {"NonNet": ["NonNet", "Slump"], "Net": ["Channel Sand"]}
        columns : list[dict]
            Column specifications, each with:
            - property (str, required): Property name in results
            - stat (str, required): Statistic to extract (mean, std_dev, p10, etc.)
            - label (str, optional): Display name (defaults to stat)
            - format (str, optional): Number format (e.g., ".4f")
            - unit (str, optional): Display unit for printing
            - factor (float, optional): Multiplier for value conversion (default 1.0)
            - agg (str, optional): Cross-well aggregation method:
                - "arithmetic" (default for mean): thickness-weighted arithmetic mean
                - "geometric": thickness-weighted geometric mean (for permeability)
                - "pooled" (default for std_dev): pooled standard deviation
                - "sum": simple sum (for thickness)
        print_report : bool, default True
            If True, print formatted report. If False, return structured data.

        Returns
        -------
        dict or None
            If print_report=False, returns structured data dict.
            If print_report=True, prints report and returns None.

        Raises
        ------
        ValueError
            If pooled aggregation is requested but corresponding mean column is missing.

        Examples
        --------
        >>> results.report(
        ...     zones=["Sand 3_SST", "Sand 2_SST"],
        ...     groups={"NonNet": ["NonNet", "Slump"], "Net": ["Channel Sand", "LowQ Sand"]},
        ...     columns=[
        ...         {"property": "CPI_PHIE_2025", "stat": "mean", "label": "por", "format": ".4f"},
        ...         {"property": "CPI_PHIE_2025", "stat": "std_dev", "label": "std", "format": ".4f"},
        ...         {"property": "CPI_PERM_CALC_2025", "stat": "mean", "label": "perm", "agg": "geometric", "unit": "mD", "format": ".2f"},
        ...     ]
        ... )
        """
        # Validate columns
        self._validate_columns(columns)

        # Generate structured data
        report_data = self._generate_report_data(zones, groups, columns)

        if print_report:
            self._print_report(report_data, columns)
            return None
        else:
            return report_data

    def _validate_columns(self, columns: list[dict]) -> None:
        """Validate column specifications."""
        # Check required fields
        for i, col in enumerate(columns):
            if 'property' not in col:
                raise ValueError(f"Column {i} missing required 'property' field")
            if 'stat' not in col:
                raise ValueError(f"Column {i} missing required 'stat' field")

        # Check pooled std_dev has corresponding mean
        for col in columns:
            agg = col.get('agg')
            stat = col.get('stat')
            prop = col.get('property')

            # Default agg for std_dev is pooled
            if stat == 'std_dev' and agg is None:
                agg = 'pooled'

            if agg == 'pooled':
                # Find corresponding mean column
                has_mean = any(
                    c.get('property') == prop and c.get('stat') == 'mean'
                    for c in columns
                )
                if not has_mean:
                    raise ValueError(
                        f"Column with property='{prop}' and stat='std_dev' uses pooled aggregation, "
                        f"but no corresponding mean column found. Add a column with "
                        f"property='{prop}' and stat='mean'."
                    )

    def _get_column_defaults(self, col: dict) -> dict:
        """Get column spec with defaults applied."""
        stat = col.get('stat', 'mean')

        # Default aggregation based on stat type
        if stat == 'std_dev':
            default_agg = 'pooled'
        else:
            default_agg = 'arithmetic'

        return {
            'property': col['property'],
            'stat': stat,
            'label': col.get('label', stat),
            'format': col.get('format', '.4f'),
            'unit': col.get('unit', ''),
            'factor': col.get('factor', 1.0),
            'agg': col.get('agg', default_agg),
        }

    def _extract_value(self, facies_data: dict, col: dict) -> Optional[float]:
        """Extract a value from facies data based on column spec."""
        col = self._get_column_defaults(col)
        prop = col['property']
        stat = col['stat']
        factor = col['factor']

        if prop not in facies_data:
            return None

        prop_data = facies_data[prop]
        if not isinstance(prop_data, dict) or stat not in prop_data:
            return None

        value = prop_data[stat]
        if value is None:
            return None

        return value * factor

    def _generate_report_data(
        self,
        zones: list[str],
        groups: dict[str, list[str]],
        columns: list[dict]
    ) -> dict:
        """Generate structured report data from results."""
        report = {}

        # Collect data for aggregation
        # Structure: {zone: {group: {facies: {label: [values], "thick": [thicknesses]}}}}
        aggregation_data = {}

        # Process each well
        for well_name, well_data in self.items():
            if well_name == "Summary":
                continue  # Skip if already has summary

            well_report = {}

            for zone_name in zones:
                if zone_name not in well_data:
                    continue

                zone_data = well_data[zone_name]
                zone_report = {}

                # Calculate total zone thickness from all facies
                zone_thickness = 0.0
                for facies_name, facies_data in zone_data.items():
                    if isinstance(facies_data, dict) and 'thickness' in facies_data:
                        zone_thickness += facies_data['thickness']

                zone_report['thickness'] = zone_thickness

                # Process each group
                for group_name, facies_list in groups.items():
                    existing_facies = [f for f in facies_list if f in zone_data]
                    if not existing_facies:
                        continue

                    group_report = {}
                    group_thickness = sum(
                        zone_data[f]['thickness'] for f in existing_facies
                        if isinstance(zone_data[f], dict) and 'thickness' in zone_data[f]
                    )

                    group_report['thickness'] = group_thickness
                    group_report['fraction'] = group_thickness / zone_thickness if zone_thickness > 0 else 0.0

                    # Process each facies in the group
                    for facies_name in existing_facies:
                        facies_data = zone_data[facies_name]
                        if not isinstance(facies_data, dict):
                            continue

                        facies_thickness = facies_data.get('thickness', 0.0)
                        if facies_thickness <= 0:
                            continue

                        facies_report = {
                            'thickness': facies_thickness,
                            'fraction': facies_thickness / group_thickness if group_thickness > 0 else 0.0,
                        }

                        # Extract column values
                        for col in columns:
                            col_def = self._get_column_defaults(col)
                            label = col_def['label']
                            value = self._extract_value(facies_data, col)
                            facies_report[label] = value

                        group_report[facies_name] = facies_report

                        # Collect for aggregation
                        if zone_name not in aggregation_data:
                            aggregation_data[zone_name] = {}
                        if group_name not in aggregation_data[zone_name]:
                            aggregation_data[zone_name][group_name] = {}
                        if facies_name not in aggregation_data[zone_name][group_name]:
                            aggregation_data[zone_name][group_name][facies_name] = {
                                'thick': [], 'values': {}
                            }

                        agg_facies = aggregation_data[zone_name][group_name][facies_name]
                        agg_facies['thick'].append(facies_thickness)

                        for col in columns:
                            col_def = self._get_column_defaults(col)
                            label = col_def['label']
                            value = self._extract_value(facies_data, col)
                            if label not in agg_facies['values']:
                                agg_facies['values'][label] = []
                            agg_facies['values'][label].append(value)

                    zone_report[group_name] = group_report

                if zone_report:
                    well_report[zone_name] = zone_report

            if well_report:
                report[well_name] = well_report

        # Generate Summary
        summary = self._generate_summary(aggregation_data, columns, zones, groups)
        if summary:
            report['Summary'] = summary

        return report

    def _generate_summary(
        self,
        aggregation_data: dict,
        columns: list[dict],
        zones: list[str],
        groups: dict[str, list[str]]
    ) -> dict:
        """Generate cross-well summary using thickness-weighted aggregation."""
        summary = {}

        # Build lookup for mean values needed by pooled std
        # {(zone, group, facies, property): grand_mean}
        grand_means = {}

        # First pass: compute all arithmetic means (needed for pooled std)
        for zone_name, zone_agg in aggregation_data.items():
            for group_name, group_agg in zone_agg.items():
                for facies_name, facies_agg in group_agg.items():
                    thicks = np.array(facies_agg['thick'])
                    total_thick = np.sum(thicks)
                    if total_thick <= 0:
                        continue

                    for col in columns:
                        col_def = self._get_column_defaults(col)
                        if col_def['stat'] == 'mean':
                            label = col_def['label']
                            prop = col_def['property']
                            values = facies_agg['values'].get(label, [])

                            valid_mask = [v is not None for v in values]
                            if not any(valid_mask):
                                continue

                            valid_vals = np.array([v for v, m in zip(values, valid_mask) if m])
                            valid_thicks = np.array([t for t, m in zip(thicks, valid_mask) if m])
                            valid_total = np.sum(valid_thicks)

                            if valid_total > 0:
                                grand_mean = np.sum(valid_vals * valid_thicks) / valid_total
                                grand_means[(zone_name, group_name, facies_name, prop)] = grand_mean

        # Second pass: compute all aggregated values
        for zone_name in zones:
            if zone_name not in aggregation_data:
                continue

            zone_agg = aggregation_data[zone_name]
            zone_summary = {'thickness': 0.0}

            for group_name, facies_list in groups.items():
                if group_name not in zone_agg:
                    continue

                group_agg = zone_agg[group_name]
                group_summary = {'thickness': 0.0}

                for facies_name in facies_list:
                    if facies_name not in group_agg:
                        continue

                    facies_agg = group_agg[facies_name]
                    thicks = np.array(facies_agg['thick'])
                    total_thick = np.sum(thicks)

                    if total_thick <= 0:
                        continue

                    facies_summary = {'thickness': total_thick, 'fraction': 0.0}

                    for col in columns:
                        col_def = self._get_column_defaults(col)
                        label = col_def['label']
                        prop = col_def['property']
                        agg_method = col_def['agg']

                        values = facies_agg['values'].get(label, [])
                        valid_mask = [v is not None for v in values]

                        if not any(valid_mask):
                            facies_summary[label] = None
                            continue

                        valid_vals = np.array([v for v, m in zip(values, valid_mask) if m])
                        valid_thicks = np.array([t for t, m in zip(thicks, valid_mask) if m])
                        valid_total = np.sum(valid_thicks)

                        if valid_total <= 0:
                            facies_summary[label] = None
                            continue

                        if agg_method == 'arithmetic':
                            # Thickness-weighted arithmetic mean
                            agg_value = np.sum(valid_vals * valid_thicks) / valid_total

                        elif agg_method == 'geometric':
                            # Thickness-weighted geometric mean
                            # Filter out non-positive values for log
                            pos_mask = valid_vals > 0
                            if not np.any(pos_mask):
                                facies_summary[label] = None
                                continue
                            pos_vals = valid_vals[pos_mask]
                            pos_thicks = valid_thicks[pos_mask]
                            pos_total = np.sum(pos_thicks)
                            agg_value = np.exp(np.sum(np.log(pos_vals) * pos_thicks) / pos_total)

                        elif agg_method == 'pooled':
                            # Pooled standard deviation
                            # Requires the grand mean from the corresponding mean column
                            grand_mean = grand_means.get((zone_name, group_name, facies_name, prop))
                            if grand_mean is None:
                                facies_summary[label] = None
                                continue

                            # Get the corresponding std values (these are the per-well stds)
                            # and the per-well means
                            mean_label = None
                            for c in columns:
                                c_def = self._get_column_defaults(c)
                                if c_def['property'] == prop and c_def['stat'] == 'mean':
                                    mean_label = c_def['label']
                                    break

                            if mean_label is None:
                                facies_summary[label] = None
                                continue

                            mean_values = facies_agg['values'].get(mean_label, [])
                            std_values = values  # current column values (stds)

                            # Both must be valid
                            combined_mask = [
                                m is not None and s is not None
                                for m, s in zip(mean_values, std_values)
                            ]
                            if not any(combined_mask):
                                facies_summary[label] = None
                                continue

                            combined_means = np.array([v for v, m in zip(mean_values, combined_mask) if m])
                            combined_stds = np.array([v for v, m in zip(std_values, combined_mask) if m])
                            combined_thicks = np.array([t for t, m in zip(thicks, combined_mask) if m])
                            combined_total = np.sum(combined_thicks)

                            # Pooled variance formula:
                            # var_pooled = sum(thick * (std^2 + (mean - grand_mean)^2)) / total_thick
                            pooled_var = np.sum(
                                combined_thicks * (combined_stds**2 + (combined_means - grand_mean)**2)
                            ) / combined_total
                            agg_value = np.sqrt(pooled_var)

                        elif agg_method == 'sum':
                            agg_value = np.sum(valid_vals)

                        else:
                            # Default to arithmetic
                            agg_value = np.sum(valid_vals * valid_thicks) / valid_total

                        facies_summary[label] = agg_value

                    group_summary[facies_name] = facies_summary
                    group_summary['thickness'] += total_thick

                # Update facies fractions based on group total
                group_thick = group_summary['thickness']
                for facies_name in facies_list:
                    if facies_name in group_summary and isinstance(group_summary[facies_name], dict):
                        f_thick = group_summary[facies_name].get('thickness', 0)
                        group_summary[facies_name]['fraction'] = f_thick / group_thick if group_thick > 0 else 0.0

                if group_summary['thickness'] > 0:
                    zone_summary[group_name] = group_summary
                    zone_summary['thickness'] += group_summary['thickness']

            # Calculate group fractions
            zone_thick = zone_summary['thickness']
            for group_name in groups:
                if group_name in zone_summary and isinstance(zone_summary[group_name], dict):
                    g_thick = zone_summary[group_name].get('thickness', 0)
                    zone_summary[group_name]['fraction'] = g_thick / zone_thick if zone_thick > 0 else 0.0

            if zone_summary['thickness'] > 0:
                summary[zone_name] = zone_summary

        return summary

    def _print_report(self, report_data: dict, columns: list[dict]) -> None:
        """Print formatted report."""
        indent = "  "

        # Prepare column formatting
        col_defs = [self._get_column_defaults(c) for c in columns]

        for well_name, well_data in report_data.items():
            print(f"\n{well_name}")

            for zone_name, zone_data in well_data.items():
                if not isinstance(zone_data, dict):
                    continue

                zone_thick = zone_data.get('thickness', 0)
                print(f"{indent}{zone_name:<13} iso: {zone_thick:>6.2f}m")

                for group_name, group_data in zone_data.items():
                    if group_name == 'thickness' or not isinstance(group_data, dict):
                        continue

                    group_thick = group_data.get('thickness', 0)
                    group_frac = group_data.get('fraction', 0)
                    print(f"{indent*2}-- {group_name:<10} fraction: {group_frac:>7.4f} iso: {group_thick:>6.2f}m")

                    for facies_name, facies_data in group_data.items():
                        if facies_name in ('thickness', 'fraction') or not isinstance(facies_data, dict):
                            continue

                        f_thick = facies_data.get('thickness', 0)
                        f_frac = facies_data.get('fraction', 0)

                        # Build column values string
                        col_strs = []
                        for col_def in col_defs:
                            label = col_def['label']
                            fmt = col_def['format']
                            unit = col_def['unit']
                            value = facies_data.get(label)

                            if value is not None:
                                formatted = f"{value:{fmt}}"
                                if unit:
                                    col_strs.append(f"{label}: {formatted:>8}{unit}")
                                else:
                                    col_strs.append(f"{label}: {formatted:>8}")
                            else:
                                col_strs.append(f"{label}: {'N/A':>8}")

                        col_str = " ".join(col_strs)
                        print(f"{indent*2}| {facies_name:<15} frac: {f_frac:>7.4f} iso: {f_thick:>6.2f}m {col_str}")

                print("")


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

    def properties(self, property_names: list[str]) -> _ManagerMultiPropertyProxy:
        """
        Create a multi-property proxy for computing statistics across multiple properties.

        This allows computing statistics for multiple properties at once, with
        property-specific stats (mean, median, etc.) nested under property names
        and common stats (depth_range, samples, thickness, etc.) at the group level.

        Parameters
        ----------
        property_names : list[str]
            List of property names to include in statistics

        Returns
        -------
        _ManagerMultiPropertyProxy
            Proxy that supports filter(), filter_intervals(), and sums_avg()

        Examples
        --------
        >>> # Compute stats for multiple properties grouped by facies
        >>> manager.properties(['PHIE', 'PERM']).filter('Facies').sums_avg()
        >>> # Returns:
        >>> # {
        >>> #     "well_A": {
        >>> #         "Sand": {
        >>> #             "PHIE": {"mean": 0.18, "median": 0.17, ...},
        >>> #             "PERM": {"mean": 150, "median": 120, ...},
        >>> #             "depth_range": {...},
        >>> #             "samples": 387,
        >>> #             "thickness": 29.4,
        >>> #             ...
        >>> #         }
        >>> #     }
        >>> # }

        >>> # With custom intervals
        >>> manager.properties(['PHIE', 'PERM']).filter('Facies').filter_intervals("Zones").sums_avg()
        >>> # Returns:
        >>> # {
        >>> #     "well_A": {
        >>> #         "Zone_1": {
        >>> #             "Sand": {
        >>> #                 "PHIE": {"mean": 0.18, ...},
        >>> #                 "PERM": {"mean": 150, ...},
        >>> #                 "depth_range": {...},
        >>> #                 ...
        >>> #             }
        >>> #         }
        >>> #     }
        >>> # }
        """
        return _ManagerMultiPropertyProxy(self, property_names)

    def load_las(
        self,
        filepath: Union[str, Path, list[Union[str, Path]]],
        path: Optional[Union[str, Path]] = None,
        sampled: bool = False,
        combine: Optional[str] = None,
        source_name: Optional[str] = None,
        silent: bool = False
    ) -> 'WellDataManager':
        """
        Load LAS file(s), auto-create well if needed.

        Parameters
        ----------
        filepath : Union[str, Path, list[Union[str, Path]]]
            Path to LAS file or list of paths to LAS files.
            When providing a list, filenames can be relative to the path parameter.
        path : Union[str, Path], optional
            Directory path to prepend to all filenames. Useful when loading multiple
            files from the same directory. If None, filenames are used as-is.
        sampled : bool, default False
            If True, mark all properties from the LAS file(s) as 'sampled' type.
            Use this for core plug data or other point measurements.
        combine : str, optional
            When loading multiple files, combine files from the same well into a single source:
            - None (default): Load files as separate sources, no combining
            - 'match': Combine using match method (safest, errors on mismatch)
            - 'resample': Combine using resample method (interpolates to first file)
            - 'concat': Combine using concat method (merges all unique depths)
            Files are automatically grouped by well name. If 4 files from 2 wells are loaded,
            2 combined sources are created (one per well).
        source_name : str, optional
            Name for combined source when combine is specified. If not specified,
            uses 'combined_match', 'combined_resample', or 'combined_concat'.
            When files span multiple wells, the well name is prepended automatically.
        silent : bool, default False
            If True, suppress debug output showing which sources were loaded.
            Useful when loading many files programmatically.

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

        >>> # Load multiple files from same directory
        >>> manager.load_las(
        ...     ["file1.las", "file2.las", "file3.las"],
        ...     path="data/well_logs"
        ... )

        >>> # Load and combine files (automatically groups by well)
        >>> manager.load_las(
        ...     ["36_7-5_B_CorePerm.las", "36_7-5_B_CorePor.las",
        ...      "36_7-4_CorePerm.las", "36_7-4_CorePor.las"],
        ...     path="data/",
        ...     combine="match",
        ...     source_name="CorePlugs"
        ... )
        Loaded sources:
        - Well 36/7-5 B: CorePlugs (2 files combined)
        - Well 36/7-4: CorePlugs (2 files combined)
        """
        # Handle list of files
        if isinstance(filepath, list):
            # Prepend path to all filenames if provided
            if path is not None:
                base_path = Path(path)
                file_paths = [base_path / file for file in filepath]
            else:
                file_paths = filepath

            # If combine is specified, group files by well and combine each group
            if combine is not None:
                # Group files by well name
                from collections import defaultdict
                well_groups = defaultdict(list)

                for file_path in file_paths:
                    las = LasFile(file_path)
                    if las.well_name is None:
                        raise LasFileError(
                            f"LAS file {file_path} has no WELL name in header. "
                            "Cannot determine which well to load into."
                        )
                    well_groups[las.well_name].append(file_path)

                # Track loaded sources for debug output
                loaded_sources = []

                # Process each well group
                for well_name, files_for_well in well_groups.items():
                    sanitized_name = sanitize_well_name(well_name)
                    well_key = f"well_{sanitized_name}"

                    # Ensure well exists
                    if well_key not in self._wells:
                        self._wells[well_key] = Well(
                            name=well_name,
                            sanitized_name=sanitized_name,
                            parent_manager=self
                        )
                        self._name_mapping[well_name] = well_key

                    # Load files into well with combine
                    self._wells[well_key].load_las(
                        files_for_well,
                        path=None,  # Path already prepended
                        sampled=sampled,
                        combine=combine,
                        source_name=source_name
                    )

                    # Track what was loaded
                    actual_source_name = source_name if source_name else f"combined_{combine}"
                    loaded_sources.append(
                        (well_name, actual_source_name, len(files_for_well))
                    )

                # Print debug output
                if not silent:
                    print("Loaded sources:")
                    for well_name, src_name, file_count in loaded_sources:
                        print(f"  - Well {well_name}: {src_name} ({file_count} file{'s' if file_count > 1 else ''} combined)")

                return self

            # No combine - load each file separately
            loaded_sources = []
            for file_path in file_paths:
                # Read well name before loading
                las = LasFile(file_path)
                if las.well_name is None:
                    raise LasFileError(
                        f"LAS file {file_path} has no WELL name in header. "
                        "Cannot determine which well to load into."
                    )
                well_name = las.well_name
                sanitized_name = sanitize_well_name(well_name)
                well_key = f"well_{sanitized_name}"

                # Track existing sources before loading
                existing_sources = set()
                if well_key in self._wells:
                    existing_sources = set(self._wells[well_key].sources)

                # Load the file
                self.load_las(file_path, path=None, sampled=sampled, combine=None, source_name=None)

                # Find new sources that were added
                if well_key in self._wells:
                    new_sources = set(self._wells[well_key].sources) - existing_sources
                    for src_name in new_sources:
                        loaded_sources.append((well_name, src_name))

            # Print debug output
            if not silent and loaded_sources:
                print("Loaded sources:")
                for well_name, src_name in loaded_sources:
                    print(f"  - Well {well_name}: {src_name}")

            return self

        # Handle single file
        # Prepend path if provided
        if path is not None:
            file_path = Path(path) / filepath
        else:
            file_path = filepath

        las = LasFile(file_path)
        well_name = las.well_name

        if well_name is None:
            raise LasFileError(
                f"LAS file {file_path} has no WELL name in header. "
                "Cannot determine which well to load into."
            )

        sanitized_name = sanitize_well_name(well_name)
        # Use well_ prefix for dictionary key (attribute access)
        well_key = f"well_{sanitized_name}"

        # Track existing sources before loading
        existing_sources = set()
        if well_key in self._wells:
            existing_sources = set(self._wells[well_key].sources)
        else:
            # Create new well
            self._wells[well_key] = Well(
                name=well_name,
                sanitized_name=sanitized_name,
                parent_manager=self
            )
            self._name_mapping[well_name] = well_key

        # Load into well
        self._wells[well_key].load_las(las, sampled=sampled)

        # Find new sources that were added
        new_sources = set(self._wells[well_key].sources) - existing_sources

        # Print debug output
        if not silent and new_sources:
            print("Loaded sources:")
            for src_name in new_sources:
                print(f"  - Well {well_name}: {src_name}")

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

            # Save filter intervals if any exist
            if hasattr(well, '_saved_filter_intervals') and well._saved_filter_intervals:
                import json
                intervals_file = well_folder / "intervals.json"
                with open(intervals_file, 'w') as f:
                    json.dump(well._saved_filter_intervals, f, indent=2)
            else:
                # Remove intervals file if no intervals (in case they were deleted)
                intervals_file = well_folder / "intervals.json"
                if intervals_file.exists():
                    intervals_file.unlink()

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
                    self.load_las(las_file, silent=True)
            return self

        # Load from well folders
        for well_folder in well_folders:
            # Find all LAS files in this folder
            las_files = sorted(well_folder.glob("*.las"))
            for las_file in las_files:
                self.load_las(las_file, silent=True)

            # Load saved filter intervals if they exist
            intervals_file = well_folder / "intervals.json"
            if intervals_file.exists():
                import json
                try:
                    with open(intervals_file, 'r') as f:
                        saved_intervals = json.load(f)
                    # Find the well for this folder and set its intervals
                    well_key = well_folder.name  # e.g., "well_35_9_16_A"
                    if well_key in self._wells:
                        self._wells[well_key]._saved_filter_intervals = saved_intervals
                except Exception as e:
                    warnings.warn(f"Could not load intervals from {intervals_file}: {e}")

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

    @property
    def saved_intervals(self) -> dict[str, list[str]]:
        """
        List saved interval names for all wells.

        Returns
        -------
        dict[str, list[str]]
            Dictionary mapping well names to their saved interval names

        Examples
        --------
        >>> manager.saved_intervals
        {'well_A': ['Reservoir_Zones', 'Slump_Zones'], 'well_B': ['Reservoir_Zones']}
        """
        result = {}
        for well_name, well in self._wells.items():
            if well.saved_intervals:
                result[well_name] = well.saved_intervals
        return result

    def get_intervals(self, name: str) -> dict[str, list[dict]]:
        """
        Get saved filter intervals by name from all wells that have them.

        Parameters
        ----------
        name : str
            Name of the saved filter intervals

        Returns
        -------
        dict[str, list[dict]]
            Dictionary mapping well names to their interval definitions

        Raises
        ------
        KeyError
            If no wells have intervals with the given name

        Examples
        --------
        >>> manager.get_intervals("Slump_Zones")
        {'well_A': [{'name': 'Zone_A', 'top': 2500, 'base': 2650}],
         'well_B': [{'name': 'Zone_A', 'top': 2600, 'base': 2750}]}
        """
        result = {}
        for well_name, well in self._wells.items():
            if name in well.saved_intervals:
                result[well_name] = well.get_intervals(name)

        if not result:
            # Collect all available interval names for error message
            all_names = set()
            for well in self._wells.values():
                all_names.update(well.saved_intervals)
            raise KeyError(
                f"No wells have saved intervals named '{name}'. "
                f"Available: {sorted(all_names) if all_names else 'none'}"
            )

        return result

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

    def add_template(self, template: 'Template') -> None:
        """
        Store a template using its built-in name.

        Parameters
        ----------
        template : Template
            Template object (uses template.name as the key)

        Examples
        --------
        >>> from well_log_toolkit import Template
        >>>
        >>> template = Template("reservoir")
        >>> template.add_track(track_type="continuous", logs=[...])
        >>> manager.add_template(template)  # Stored as "reservoir"
        >>>
        >>> # Use in WellView
        >>> view = well.WellView(template="reservoir")
        """
        from .visualization import Template

        if not isinstance(template, Template):
            raise TypeError(f"template must be Template, got {type(template).__name__}")

        self._templates[template.name] = template

    def set_template(self, name: str, template: Union['Template', dict]) -> None:
        """
        Store a template with a custom name (overrides template.name).

        Use add_template() for the simpler case where the template's
        built-in name should be used.

        Parameters
        ----------
        name : str
            Template name for reference (overrides template.name)
        template : Union[Template, dict]
            Template object or dictionary configuration

        Examples
        --------
        >>> # Store with a different name than the template's built-in name
        >>> template = Template("reservoir")
        >>> manager.set_template("reservoir_v2", template)
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
        x: Optional[str] = None,
        y: Optional[str] = None,
        wells: Optional[list[str]] = None,
        layers: Optional[dict[str, list[str]]] = None,
        shape: Optional[str] = None,
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
        show_regression_legend: bool = True,
        show_regression_equation: bool = True,
        show_regression_r2: bool = True,
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
        show_regression_legend : bool, optional
            Show separate legend for regression lines in lower right corner. Default: True
        show_regression_equation : bool, optional
            Include regression equation in regression legend labels. Default: True
        show_regression_r2 : bool, optional
            Include R² value in regression legend labels. Default: True
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

        # Set default shape: "well" when no layers, "label" when layers provided
        if shape is None and layers is None:
            shape = "well"

        # Set default color: "well" when shape defaults to "label" (i.e., when layers provided)
        if color is None and layers is not None and shape is None:
            color = "well"

        return CrossplotClass(
            wells=well_objects,
            x=x,
            y=y,
            layers=layers,
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
            show_regression_legend=show_regression_legend,
            show_regression_equation=show_regression_equation,
            show_regression_r2=show_regression_r2,
            regression=regression,
            regression_by_color=regression_by_color,
            regression_by_group=regression_by_group,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"WellDataManager(wells={len(self._wells)})"