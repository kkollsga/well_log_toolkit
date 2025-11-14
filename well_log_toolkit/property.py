"""
Property class for well log data with filtering support.
"""
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .exceptions import PropertyError, PropertyNotFoundError, PropertyTypeError, DepthAlignmentError

if TYPE_CHECKING:
    from .well import Well
    from .las_file import LasFile


class Property:
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
        depth: np.ndarray,
        values: np.ndarray,
        parent_well: Optional['Well'] = None,
        unit: str = '',
        prop_type: str = 'continuous',
        description: str = '',
        null_value: float = -999.25,
        labels: Optional[dict[int, str]] = None,
        source_las: Optional['LasFile'] = None,
        source_name: Optional[str] = None,
        original_name: Optional[str] = None
    ):
        self.name = name  # Sanitized name for Python attribute access
        self.original_name = original_name or name  # Original name with special characters
        self.depth = np.asarray(depth, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.parent_well = parent_well
        self.unit = unit
        self.type = prop_type
        self.description = description
        self.labels = labels  # Mapping for discrete property values {0: 'NonNet', 1: 'Net'}
        self.source_las = source_las  # Source LAS file this property came from
        self.source_name = source_name  # Source name (file path or external_df)

        # Secondary properties (filters) aligned on same depth grid
        self.secondary_properties: list[Property] = []

        # Replace null values with np.nan
        self.values = np.where(
            np.abs(self.values - null_value) < 1e-6,
            np.nan,
            self.values
        )

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

    def filter(self, property_name: str) -> 'Property':
        """
        Add a discrete property from parent well as a filter dimension.

        Returns new Property with the discrete property values interpolated
        to the current depth grid. The depth grid remains unchanged - only
        the discrete values are added as a secondary property.

        Parameters
        ----------
        property_name : str
            Name of discrete property in parent well

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
        >>> filtered.to_dataframe().shape  # (29, 3) - added Zone column
        """
        if self.parent_well is None:
            raise PropertyNotFoundError(
                f"Cannot filter property '{self.name}': no parent well reference. "
                "Property must be created from a Well to enable filtering."
            )

        # Lookup in parent well
        try:
            discrete_prop = self.parent_well.get_property(property_name)
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

        # Interpolate discrete property to THIS property's depth grid (no resampling of main property)
        interpolated_discrete = self._resample_to_grid(
            discrete_prop.depth,
            discrete_prop.values,
            self.depth,  # Use current depth grid
            method='nearest'  # Always nearest for discrete
        )

        # Copy existing secondary properties (they're already on the same depth grid)
        new_secondaries = list(self.secondary_properties)

        # Add new secondary property
        new_secondaries.append(Property(
            name=discrete_prop.name,
            depth=self.depth.copy(),  # Same depth grid
            values=interpolated_discrete,
            parent_well=self.parent_well,
            unit=discrete_prop.unit,
            prop_type=discrete_prop.type,
            description=discrete_prop.description,
            null_value=-999.25,
            labels=discrete_prop.labels,
            source_las=discrete_prop.source_las,
            source_name=discrete_prop.source_name,
            original_name=discrete_prop.original_name
        ))

        # Create new Property instance with all secondaries
        new_prop = Property(
            name=self.name,
            depth=self.depth.copy(),  # Keep same depth grid
            values=self.values.copy(),  # Keep same values
            parent_well=self.parent_well,
            unit=self.unit,
            prop_type=self.type,
            description=self.description,
            null_value=-999.25,  # Already cleaned
            labels=self.labels,
            source_las=self.source_las,
            source_name=self.source_name,
            original_name=self.original_name
        )
        new_prop.secondary_properties = new_secondaries

        return new_prop
    
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
            method='linear' if self.type == 'continuous' else 'nearest'
        )
        
        resampled_other = self._resample_to_grid(
            other.depth,
            other.values,
            common_depth,
            method='nearest'  # Always nearest for discrete
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
            Interpolation method: 'linear', 'nearest', 'cubic'
        
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
        
        if len(valid_depth) == 1:
            # Single point, use nearest neighbor
            method = 'nearest'
        
        # Interpolate
        try:
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
    
    def sums_avg(self) -> dict:
        """
        Compute hierarchical statistics grouped by all secondary properties.
        
        Returns
        -------
        dict
            Nested dictionary with statistics for each group combination.
            If no secondary properties, returns simple statistics dict.
        
        Examples
        --------
        >>> # Simple statistics (no filters)
        >>> phie = well.get_property('PHIE')
        >>> stats = phie.sums_avg()
        >>> # {'mean': 0.18, 'sum': 45.2, 'count': 251, ...}
        
        >>> # Grouped statistics
        >>> filtered = phie.filter('Zone').filter('NTG_Flag')
        >>> stats = filtered.sums_avg()
        >>> # {'Zone_1': {'NTG_Flag_0': {...}, 'NTG_Flag_1': {...}}, ...}
        """
        if not self.secondary_properties:
            # No filters, simple statistics
            return self._compute_stats(np.ones(len(self.depth), dtype=bool))
        
        # Build hierarchical grouping
        return self._recursive_group(0, np.ones(len(self.depth), dtype=bool))
    
    def _recursive_group(self, filter_idx: int, mask: np.ndarray) -> dict:
        """
        Recursively group by secondary properties.
        
        Parameters
        ----------
        filter_idx : int
            Index of current secondary property
        mask : np.ndarray
            Boolean mask for current group
        
        Returns
        -------
        dict
            Statistics dict or nested dict of statistics
        """
        if filter_idx >= len(self.secondary_properties):
            # Base case: compute statistics for this group
            return self._compute_stats(mask)
        
        # Get unique values for current filter
        current_filter = self.secondary_properties[filter_idx]
        filter_values = current_filter.values[mask]
        unique_vals = np.unique(filter_values[~np.isnan(filter_values)])
        
        if len(unique_vals) == 0:
            # No valid values, return stats for current mask
            return self._compute_stats(mask)
        
        # Group by each unique value
        result = {}
        for val in unique_vals:
            sub_mask = mask & (current_filter.values == val)

            # Create readable key with label if available
            if current_filter.labels is not None and val == int(val) and int(val) in current_filter.labels:
                # Use label from mapping
                key = current_filter.labels[int(val)]
            elif val == int(val):  # Integer value without label
                key = f"{current_filter.name}_{int(val)}"
            else:  # Float value
                key = f"{current_filter.name}_{val:.2f}"

            result[key] = self._recursive_group(filter_idx + 1, sub_mask)

        return result
    
    def _compute_stats(self, mask: np.ndarray) -> dict:
        """
        Compute statistics for values selected by mask.
        
        Parameters
        ----------
        mask : np.ndarray
            Boolean mask selecting subset of data
        
        Returns
        -------
        dict
            Statistics dictionary with mean, sum, count, etc.
        """
        values = self.values[mask]
        valid = values[~np.isnan(values)]
        
        depth_interval = self.depth[mask]
        depth_thickness = 0.0
        if len(depth_interval) > 1:
            depth_thickness = depth_interval[-1] - depth_interval[0]
        
        return {
            'mean': float(np.mean(valid)) if len(valid) > 0 else np.nan,
            'sum': float(np.sum(valid)) if len(valid) > 0 else np.nan,
            'count': int(len(valid)),
            'depth_samples': int(np.sum(mask)),
            'depth_thickness': float(depth_thickness),
            'min': float(np.min(valid)) if len(valid) > 0 else np.nan,
            'max': float(np.max(valid)) if len(valid) > 0 else np.nan,
            'std': float(np.std(valid)) if len(valid) > 0 else np.nan,
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

    def to_dataframe(self, discrete_labels: bool = True) -> pd.DataFrame:
        """
        Export property and secondary properties as DataFrame.

        Parameters
        ----------
        discrete_labels : bool, default True
            If True, apply label mappings to discrete properties

        Returns
        -------
        pd.DataFrame
            DataFrame with DEPT, main property, and secondary properties

        Examples
        --------
        >>> filtered = well.phie.filter('Zone').filter('NTG_Flag')
        >>> df = filtered.to_dataframe()
        >>> print(df.head())
        """
        data = {
            'DEPT': self.depth,
            self.name: self._apply_labels(self.values) if discrete_labels and self.labels else self.values
        }

        for sec_prop in self.secondary_properties:
            if discrete_labels and sec_prop.labels:
                data[sec_prop.name] = sec_prop._apply_labels(sec_prop.values)
            else:
                data[sec_prop.name] = sec_prop.values

        return pd.DataFrame(data)

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
        df = self.to_dataframe(discrete_labels=False)

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

        # Collect discrete labels if store_labels is True (use original names)
        label_mappings = None
        if store_labels:
            label_mappings = {}
            # Check main property
            if self.labels:
                label_mappings[self.original_name] = self.labels
            # Check secondary properties
            for sec_prop in self.secondary_properties:
                if sec_prop.labels:
                    label_mappings[sec_prop.original_name] = sec_prop.labels

        # Export using LasFile static method
        LasFile.export_las(
            filepath=filepath,
            well_name=well_name,
            df=df,
            unit_mappings=unit_mappings,
            null_value=null_value,
            discrete_labels=label_mappings if label_mappings else None
        )

    def __repr__(self) -> str:
        """String representation."""
        filters = f", filters={len(self.secondary_properties)}" if self.secondary_properties else ""
        return (
            f"Property('{self.name}', "
            f"samples={len(self.depth)}, "
            f"type='{self.type}'"
            f"{filters})"
        )