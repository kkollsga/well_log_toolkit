"""
SumsAvgResult container and helper functions for multi-well statistical aggregation.

Provides the SumsAvgResult dictionary subclass with cross-well reporting capabilities,
along with helper functions for JSON sanitization and DataFrame flattening.
"""

from typing import Optional

import numpy as np
import pandas as pd


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
        print_report: bool = True,
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
            if "property" not in col:
                raise ValueError(f"Column {i} missing required 'property' field")
            if "stat" not in col:
                raise ValueError(f"Column {i} missing required 'stat' field")

        # Check pooled std_dev has corresponding mean
        for col in columns:
            agg = col.get("agg")
            stat = col.get("stat")
            prop = col.get("property")

            # Default agg for std_dev is pooled
            if stat == "std_dev" and agg is None:
                agg = "pooled"

            if agg == "pooled":
                # Find corresponding mean column
                has_mean = any(
                    c.get("property") == prop and c.get("stat") == "mean" for c in columns
                )
                if not has_mean:
                    raise ValueError(
                        f"Column with property='{prop}' and stat='std_dev' uses pooled aggregation, "
                        f"but no corresponding mean column found. Add a column with "
                        f"property='{prop}' and stat='mean'."
                    )

    def _get_column_defaults(self, col: dict) -> dict:
        """Get column spec with defaults applied."""
        stat = col.get("stat", "mean")

        # Default aggregation based on stat type
        if stat == "std_dev":
            default_agg = "pooled"
        else:
            default_agg = "arithmetic"

        return {
            "property": col["property"],
            "stat": stat,
            "label": col.get("label", stat),
            "format": col.get("format", ".4f"),
            "unit": col.get("unit", ""),
            "factor": col.get("factor", 1.0),
            "agg": col.get("agg", default_agg),
        }

    def _extract_value(self, facies_data: dict, col: dict) -> Optional[float]:
        """Extract a value from facies data based on column spec."""
        col = self._get_column_defaults(col)
        prop = col["property"]
        stat = col["stat"]
        factor = col["factor"]

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
        self, zones: list[str], groups: dict[str, list[str]], columns: list[dict]
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
                    if isinstance(facies_data, dict) and "thickness" in facies_data:
                        zone_thickness += facies_data["thickness"]

                zone_report["thickness"] = zone_thickness

                # Process each group
                for group_name, facies_list in groups.items():
                    existing_facies = [f for f in facies_list if f in zone_data]
                    if not existing_facies:
                        continue

                    group_report = {}
                    group_thickness = sum(
                        zone_data[f]["thickness"]
                        for f in existing_facies
                        if isinstance(zone_data[f], dict) and "thickness" in zone_data[f]
                    )

                    group_report["thickness"] = group_thickness
                    group_report["fraction"] = (
                        group_thickness / zone_thickness if zone_thickness > 0 else 0.0
                    )

                    # Process each facies in the group
                    for facies_name in existing_facies:
                        facies_data = zone_data[facies_name]
                        if not isinstance(facies_data, dict):
                            continue

                        facies_thickness = facies_data.get("thickness", 0.0)
                        if facies_thickness <= 0:
                            continue

                        facies_report = {
                            "thickness": facies_thickness,
                            "fraction": (
                                facies_thickness / group_thickness if group_thickness > 0 else 0.0
                            ),
                        }

                        # Extract column values
                        for col in columns:
                            col_def = self._get_column_defaults(col)
                            label = col_def["label"]
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
                                "thick": [],
                                "values": {},
                            }

                        agg_facies = aggregation_data[zone_name][group_name][facies_name]
                        agg_facies["thick"].append(facies_thickness)

                        for col in columns:
                            col_def = self._get_column_defaults(col)
                            label = col_def["label"]
                            value = self._extract_value(facies_data, col)
                            if label not in agg_facies["values"]:
                                agg_facies["values"][label] = []
                            agg_facies["values"][label].append(value)

                    zone_report[group_name] = group_report

                if zone_report:
                    well_report[zone_name] = zone_report

            if well_report:
                report[well_name] = well_report

        # Generate Summary
        summary = self._generate_summary(aggregation_data, columns, zones, groups)
        if summary:
            report["Summary"] = summary

        return report

    def _generate_summary(
        self,
        aggregation_data: dict,
        columns: list[dict],
        zones: list[str],
        groups: dict[str, list[str]],
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
                    thicks = np.array(facies_agg["thick"])
                    total_thick = np.sum(thicks)
                    if total_thick <= 0:
                        continue

                    for col in columns:
                        col_def = self._get_column_defaults(col)
                        if col_def["stat"] == "mean":
                            label = col_def["label"]
                            prop = col_def["property"]
                            values = facies_agg["values"].get(label, [])

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
            zone_summary = {"thickness": 0.0}

            for group_name, facies_list in groups.items():
                if group_name not in zone_agg:
                    continue

                group_agg = zone_agg[group_name]
                group_summary = {"thickness": 0.0}

                for facies_name in facies_list:
                    if facies_name not in group_agg:
                        continue

                    facies_agg = group_agg[facies_name]
                    thicks = np.array(facies_agg["thick"])
                    total_thick = np.sum(thicks)

                    if total_thick <= 0:
                        continue

                    facies_summary = {"thickness": total_thick, "fraction": 0.0}

                    for col in columns:
                        col_def = self._get_column_defaults(col)
                        label = col_def["label"]
                        prop = col_def["property"]
                        agg_method = col_def["agg"]

                        values = facies_agg["values"].get(label, [])
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

                        if agg_method == "arithmetic":
                            # Thickness-weighted arithmetic mean
                            agg_value = np.sum(valid_vals * valid_thicks) / valid_total

                        elif agg_method == "geometric":
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

                        elif agg_method == "pooled":
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
                                if c_def["property"] == prop and c_def["stat"] == "mean":
                                    mean_label = c_def["label"]
                                    break

                            if mean_label is None:
                                facies_summary[label] = None
                                continue

                            mean_values = facies_agg["values"].get(mean_label, [])
                            std_values = values  # current column values (stds)

                            # Both must be valid
                            combined_mask = [
                                m is not None and s is not None
                                for m, s in zip(mean_values, std_values)
                            ]
                            if not any(combined_mask):
                                facies_summary[label] = None
                                continue

                            combined_means = np.array(
                                [v for v, m in zip(mean_values, combined_mask) if m]
                            )
                            combined_stds = np.array(
                                [v for v, m in zip(std_values, combined_mask) if m]
                            )
                            combined_thicks = np.array(
                                [t for t, m in zip(thicks, combined_mask) if m]
                            )
                            combined_total = np.sum(combined_thicks)

                            # Pooled variance formula:
                            # var_pooled = sum(thick * (std^2 + (mean - grand_mean)^2)) / total_thick
                            pooled_var = (
                                np.sum(
                                    combined_thicks
                                    * (combined_stds**2 + (combined_means - grand_mean) ** 2)
                                )
                                / combined_total
                            )
                            agg_value = np.sqrt(pooled_var)

                        elif agg_method == "sum":
                            agg_value = np.sum(valid_vals)

                        else:
                            # Default to arithmetic
                            agg_value = np.sum(valid_vals * valid_thicks) / valid_total

                        facies_summary[label] = agg_value

                    group_summary[facies_name] = facies_summary
                    group_summary["thickness"] += total_thick

                # Update facies fractions based on group total
                group_thick = group_summary["thickness"]
                for facies_name in facies_list:
                    if facies_name in group_summary and isinstance(
                        group_summary[facies_name], dict
                    ):
                        f_thick = group_summary[facies_name].get("thickness", 0)
                        group_summary[facies_name]["fraction"] = (
                            f_thick / group_thick if group_thick > 0 else 0.0
                        )

                if group_summary["thickness"] > 0:
                    zone_summary[group_name] = group_summary
                    zone_summary["thickness"] += group_summary["thickness"]

            # Calculate group fractions
            zone_thick = zone_summary["thickness"]
            for group_name in groups:
                if group_name in zone_summary and isinstance(zone_summary[group_name], dict):
                    g_thick = zone_summary[group_name].get("thickness", 0)
                    zone_summary[group_name]["fraction"] = (
                        g_thick / zone_thick if zone_thick > 0 else 0.0
                    )

            if zone_summary["thickness"] > 0:
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

                zone_thick = zone_data.get("thickness", 0)
                print(f"{indent}{zone_name:<13} iso: {zone_thick:>6.2f}m")

                for group_name, group_data in zone_data.items():
                    if group_name == "thickness" or not isinstance(group_data, dict):
                        continue

                    group_thick = group_data.get("thickness", 0)
                    group_frac = group_data.get("fraction", 0)
                    print(
                        f"{indent*2}-- {group_name:<10} fraction: {group_frac:>7.4f} iso: {group_thick:>6.2f}m"
                    )

                    for facies_name, facies_data in group_data.items():
                        if facies_name in ("thickness", "fraction") or not isinstance(
                            facies_data, dict
                        ):
                            continue

                        f_thick = facies_data.get("thickness", 0)
                        f_frac = facies_data.get("fraction", 0)

                        # Build column values string
                        col_strs = []
                        for col_def in col_defs:
                            label = col_def["label"]
                            fmt = col_def["format"]
                            unit = col_def["unit"]
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
                        print(
                            f"{indent*2}| {facies_name:<15} frac: {f_frac:>7.4f} iso: {f_thick:>6.2f}m {col_str}"
                        )

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
        columns = ["Well", property_name]
    elif max_depth == 3:
        # Well, one grouping level (e.g., Source or Zone), value
        columns = ["Well", "Group", property_name]
    else:
        # Well, multiple grouping levels, value
        # Use generic names: Group1, Group2, etc.
        columns = ["Well"] + [f"Group{i}" for i in range(1, max_depth - 1)] + [property_name]

    df = pd.DataFrame(padded_rows, columns=columns)

    return df
