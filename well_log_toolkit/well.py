"""
Well class for managing log properties from a single well.
"""
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from .exceptions import WellError, WellNameMismatchError, PropertyNotFoundError
from .property import Property
from .las_file import LasFile
from .utils import sanitize_property_name

if TYPE_CHECKING:
    from .manager import WellDataManager


class Well:
    """
    Single well containing multiple log properties.
    
    Parameters
    ----------
    name : str
        Original well name (from LAS file)
    sanitized_name : str
        Pythonic attribute name for parent manager access
    parent_manager : WellDataManager, optional
        Parent manager reference
    
    Attributes
    ----------
    name : str
        Original well name
    sanitized_name : str
        Sanitized name for attribute access
    parent_manager : Optional[WellDataManager]
        Parent manager
    properties : list[str]
        List of property names
    sources : list[str]
        List of source LAS file paths loaded into this well
    original_las : LasFile | None
        First LAS file loaded (for template-based export)
    
    Examples
    --------
    >>> well = manager.well_12_3_2_B
    >>> well.load_las("log1.las").load_las("log2.las")
    >>> print(well.properties)
    ['PHIE', 'PERM', 'Zone', 'NTG_Flag']
    >>> print(well.sources)  # ['log1.las', 'log2.las']
    >>>
    >>> # Add DataFrame as source
    >>> df = pd.DataFrame({'DEPT': [2800, 2801], 'SW': [0.3, 0.32]})
    >>> well.add_dataframe(df, unit_mappings={'SW': 'v/v'})
    >>> print(well.sources)  # ['log1.las', 'log2.las', 'external_df']
    >>>
    >>> stats = well.phie.filter('Zone').sums_avg()
    """
    
    def __init__(
        self,
        name: str,
        sanitized_name: str,
        parent_manager: Optional['WellDataManager'] = None
    ):
        self.name = name
        self.sanitized_name = sanitized_name
        self.parent_manager = parent_manager
        self._properties: dict[str, Property] = {}  # {sanitized_name: Property}
        self._property_name_mapping: dict[str, str] = {}  # {sanitized_name: original_name}
        self._depth_grid: Optional[np.ndarray] = None
        self._source_las_files: list[LasFile] = []  # Track original LAS files for metadata preservation
        self._external_df_count: int = 0  # Counter for external DataFrame sources
    
    def load_las(self, las: Union[LasFile, str, Path]) -> 'Well':
        """
        Load LAS file into this well.
        Validates well name matches.

        Parameters
        ----------
        las : Union[LasFile, str, Path]
            Either a LasFile instance or path to LAS file
        
        Returns
        -------
        Well
            Self for method chaining
        
        Raises
        ------
        WellNameMismatchError
            If LAS well name doesn't match this well
        
        Examples
        --------
        >>> well = manager.well_12_3_2_B
        >>> well.load_las("log1.las").load_las("log2.las")
        """
        # Parse if path provided
        if isinstance(las, (str, Path)):
            las = LasFile(las)

        # Validate well name
        if las.well_name != self.name:
            raise WellNameMismatchError(
                f"Well name mismatch: attempting to load '{las.well_name}' "
                f"into well '{self.name}'. Create a new well or use "
                f"manager.load_las() for automatic well creation."
            )

        # Store reference to source LAS file for metadata preservation
        self._source_las_files.append(las)

        # Load data
        data = las.data
        depth_col = las.depth_column
        
        if depth_col is None:
            raise WellError(f"No depth column found in LAS file")
        
        depth_values = data[depth_col].values
        
        # Get discrete property information from LAS file
        discrete_props = las.discrete_properties

        # Load each curve as a property
        for curve_name in las.curves.keys():
            if curve_name == depth_col:
                continue  # Skip depth itself

            curve_meta = las.curves[curve_name]
            prop_name = curve_meta.get('alias') or curve_name

            # Get values
            values = data.get(curve_meta.get('alias') or curve_name, data.get(curve_name))
            if values is None:
                continue

            # Check if this property is marked as discrete
            is_discrete = prop_name in discrete_props
            prop_type = 'discrete' if is_discrete else curve_meta['type']

            # Get labels if property is discrete
            labels = None
            if is_discrete:
                labels = las.get_discrete_labels(prop_name)

            # Sanitize property name for Python attribute access
            sanitized_prop_name = sanitize_property_name(prop_name)

            # Create property with sanitized name
            prop = Property(
                name=sanitized_prop_name,
                depth=depth_values,
                values=values.values,
                parent_well=self,
                unit=curve_meta['unit'],
                prop_type=prop_type,
                description=curve_meta['description'],
                null_value=las.null_value,
                labels=labels,
                source_las=las,
                source_name=str(las.filepath),
                original_name=prop_name
            )

            if sanitized_prop_name in self._properties:
                # Merge with existing property
                self._merge_property(sanitized_prop_name, prop)
            else:
                # Store with sanitized name as key
                self._properties[sanitized_prop_name] = prop
                self._property_name_mapping[sanitized_prop_name] = prop_name

        return self  # Enable chaining

    def add_dataframe(
        self,
        df: pd.DataFrame,
        unit_mappings: Optional[dict[str, str]] = None,
        type_mappings: Optional[dict[str, str]] = None,
        label_mappings: Optional[dict[str, dict[int, str]]] = None
    ) -> 'Well':
        """
        Add properties from a DataFrame to this well.

        The DataFrame must contain a DEPT column. All other columns will be
        added as properties. The DataFrame is converted to a LasFile object
        with source name 'external_df', 'external_df1', etc.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing DEPT and property columns
        unit_mappings : dict[str, str], optional
            Mapping of property names to units (e.g., {'PHIE': 'v/v'})
        type_mappings : dict[str, str], optional
            Mapping of property names to types: 'continuous' or 'discrete'
            Default is 'continuous' for all properties
        label_mappings : dict[str, dict[int, str]], optional
            Label mappings for discrete properties
            Format: {'PropertyName': {0: 'Label0', 1: 'Label1'}}

        Returns
        -------
        Well
            Self for method chaining

        Examples
        --------
        >>> # Create DataFrame with properties
        >>> df = pd.DataFrame({
        ...     'DEPT': [2800, 2800.5, 2801],
        ...     'PHIE': [0.2, 0.25, 0.22],
        ...     'SW': [0.3, 0.35, 0.32],
        ...     'Zone': [0, 1, 1]
        ... })
        >>>
        >>> # Add to well with metadata
        >>> well.add_dataframe(
        ...     df,
        ...     unit_mappings={'PHIE': 'v/v', 'SW': 'v/v', 'Zone': ''},
        ...     type_mappings={'Zone': 'discrete'},
        ...     label_mappings={'Zone': {0: 'NonReservoir', 1: 'Reservoir'}}
        ... )
        >>>
        >>> # Check sources
        >>> print(well.sources)  # ['original.las', 'external_df']
        >>> print(well.Zone.source)  # 'external_df'
        """
        # Generate source name
        if self._external_df_count == 0:
            source_name = 'external_df'
        else:
            source_name = f'external_df{self._external_df_count}'
        self._external_df_count += 1

        # Create LasFile from DataFrame
        las = LasFile.from_dataframe(
            df=df,
            well_name=self.name,
            source_name=source_name,
            unit_mappings=unit_mappings,
            type_mappings=type_mappings,
            label_mappings=label_mappings
        )

        # Load it like any other LAS file
        return self.load_las(las)

    def _merge_property(self, name: str, new_prop: Property) -> None:
        """
        Merge new property data with existing property.
        
        For now, this concatenates depth/value arrays.
        Future: implement smart merging with interpolation.
        """
        existing = self._properties[name]
        
        # Simple concatenation for now
        combined_depth = np.concatenate([existing.depth, new_prop.depth])
        combined_values = np.concatenate([existing.values, new_prop.values])
        
        # Sort by depth
        sort_idx = np.argsort(combined_depth)
        combined_depth = combined_depth[sort_idx]
        combined_values = combined_values[sort_idx]
        
        # Remove duplicates (keep first occurrence)
        unique_mask = np.concatenate([[True], np.diff(combined_depth) > 1e-6])
        combined_depth = combined_depth[unique_mask]
        combined_values = combined_values[unique_mask]
        
        # Update existing property
        existing.depth = combined_depth
        existing.values = combined_values
    
    def __getattr__(self, name: str) -> Property:
        """
        Enable property access via attributes: well.phie

        This is called when normal attribute lookup fails.
        Supports both original and sanitized property names.
        """
        # Don't intercept private attributes, methods, or class attributes
        if name.startswith('_') or name in [
            'name', 'sanitized_name', 'parent_manager', 'properties',
            'load_las', 'get_property', 'resample', 'to_dataframe'
        ]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Try as-is (sanitized name)
        if name in self._properties:
            return self._properties[name]

        # Try sanitizing the name (in case original name was somehow used)
        sanitized_name = sanitize_property_name(name)
        if sanitized_name in self._properties:
            return self._properties[sanitized_name]

        # Not found - provide helpful error with available names
        available = ', '.join(self._properties.keys())
        raise AttributeError(
            f"Well '{self.name}' has no property '{name}'. "
            f"Available properties: {available or 'none'}"
        )
    
    @property
    def properties(self) -> list[str]:
        """List of property names in this well."""
        return list(self._properties.keys())

    @property
    def sources(self) -> list[str]:
        """
        List of all data sources loaded into this well.

        Includes both LAS file paths and external DataFrame sources
        (labeled as 'external_df', 'external_df1', etc.)

        Returns
        -------
        list[str]
            List of source names in the order they were loaded

        Examples
        --------
        >>> well.load_las("log1.las")
        >>> well.add_dataframe(df)
        >>> print(well.sources)  # ['log1.las', 'external_df']
        """
        # Collect all unique sources from properties in order
        seen_sources = set()
        sources_list = []

        # Add LAS file sources first (in order they were loaded)
        for las in self._source_las_files:
            source = str(las.filepath)
            if source not in seen_sources:
                sources_list.append(source)
                seen_sources.add(source)

        # Then add any external DataFrame sources found in properties
        for i in range(self._external_df_count):
            if i == 0:
                source = 'external_df'
            else:
                source = f'external_df{i}'
            if source not in seen_sources:
                sources_list.append(source)
                seen_sources.add(source)

        return sources_list

    @property
    def original_las(self) -> Optional[LasFile]:
        """
        Get the first (original) LAS file loaded into this well.

        Returns None if no LAS files have been loaded yet.

        Returns
        -------
        Optional[LasFile]
            First LasFile object loaded, or None

        Examples
        --------
        >>> well.load_las("log1.las")
        >>> original = well.original_las
        >>> well.export_to_las("output.las", use_template=original)
        """
        return self._source_las_files[0] if self._source_las_files else None
    
    def get_property(self, name: str) -> Property:
        """
        Explicit property getter.

        Supports both original and sanitized property names.

        Parameters
        ----------
        name : str
            Property name (original or sanitized)

        Returns
        -------
        Property
            The requested property

        Raises
        ------
        PropertyNotFoundError
            If property not found

        Examples
        --------
        >>> prop = well.get_property("Zoneloglinkedto'CerisaTops'")  # Original
        >>> prop = well.get_property("Zoneloglinkedto_CerisaTops_")  # Sanitized
        """
        # Try as-is (sanitized name)
        if name in self._properties:
            return self._properties[name]

        # Try sanitizing the name (in case original name was passed)
        sanitized_name = sanitize_property_name(name)
        if sanitized_name in self._properties:
            return self._properties[sanitized_name]

        # Not found
        available = ', '.join(self._properties.keys())
        raise PropertyNotFoundError(
            f"Property '{name}' not found in well '{self.name}'. "
            f"Available: {available or 'none'}"
        )
    
    def resample(
        self,
        depth_grid: Optional[np.ndarray] = None,
        depth_step: Optional[float] = None,
        depth_range: Optional[tuple[float, float]] = None
    ) -> 'Well':
        """
        Resample all properties to common depth grid.
        
        Parameters
        ----------
        depth_grid : np.ndarray, optional
            Explicit depth grid to use
        depth_step : float, optional
            Step size for regular grid (default 0.1)
        depth_range : tuple[float, float], optional
            (min_depth, max_depth) for grid
        
        Returns
        -------
        Well
            Self for method chaining
        
        Examples
        --------
        >>> well.resample(depth_step=0.1, depth_range=(2800, 3000))
        >>> well.resample(depth_grid=np.arange(2800, 3000, 0.05))
        """
        if depth_grid is None:
            # Create regular grid
            if depth_step is None:
                depth_step = 0.1
            
            if depth_range is None:
                # Use min/max across all properties
                all_depths = [p.depth for p in self._properties.values()]
                if not all_depths:
                    raise WellError("No properties to resample")
                depth_range = (
                    min(d.min() for d in all_depths),
                    max(d.max() for d in all_depths)
                )
            
            depth_grid = np.arange(
                depth_range[0],
                depth_range[1] + depth_step/2,
                depth_step
            )
        
        self._depth_grid = depth_grid
        
        # Resample each property
        for prop in self._properties.values():
            resampled_values = Property._resample_to_grid(
                prop.depth,
                prop.values,
                depth_grid,
                method='linear' if prop.type == 'continuous' else 'nearest'
            )
            prop.depth = depth_grid
            prop.values = resampled_values
        
        return self
    
    def to_dataframe(
        self,
        reference_property: Optional[str] = None,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        auto_resample: bool = True,
        discrete_labels: bool = True
    ) -> pd.DataFrame:
        """
        Export properties as DataFrame with optional resampling and filtering.

        Parameters
        ----------
        reference_property : str, optional
            Property to use as depth reference for resampling. If auto_resample
            is True, all properties will be resampled to this property's depth
            grid. If not specified, defaults to the first property that was added
            (typically the first property from the first LAS file loaded).
        include : list[str], optional
            List of property names to include. If None, includes all properties.
            Cannot be used with exclude.
        exclude : list[str], optional
            List of property names to exclude. If None, no properties are excluded.
            Cannot be used with include. Useful when you want all properties except a few.
        auto_resample : bool, default True
            If True, automatically resample all properties to the reference
            property's depth grid (uses first property if not specified).
            Set to False only if properties are already aligned.
        discrete_labels : bool, default True
            If True, apply label mappings to discrete properties with labels defined.

        Returns
        -------
        pd.DataFrame
            DataFrame with DEPT and selected properties

        Raises
        ------
        WellError
            If properties have different depth grids and auto_resample is False
        ValueError
            If both include and exclude are specified
        PropertyNotFoundError
            If reference_property is not found

        Examples
        --------
        >>> # Export all properties (auto-resamples to first property's grid)
        >>> df = well.to_dataframe()

        >>> # Export with specific reference property
        >>> df = well.to_dataframe(reference_property='DEPT')

        >>> # Include only specific properties
        >>> df = well.to_dataframe(include=['PHIE', 'SW', 'PERM'])

        >>> # Exclude specific properties
        >>> df = well.to_dataframe(exclude=['QC_Flag', 'Temp_Data'])

        >>> # Combine reference and filtering
        >>> df = well.to_dataframe(reference_property='PHIE', exclude=['Zone'])

        >>> # Disable label mapping for discrete properties
        >>> df = well.to_dataframe(discrete_labels=False)
        """
        if not self._properties:
            return pd.DataFrame()

        # Validate include/exclude
        if include is not None and exclude is not None:
            raise ValueError(
                "Cannot specify both 'include' and 'exclude'. "
                "Use either include to specify properties to include, "
                "or exclude to specify properties to skip."
            )

        # Determine reference property
        if reference_property is None:
            # Default: use first property added (first from first LAS file)
            ref_prop = next(iter(self._properties.values()))
        else:
            # Get specified reference property
            if reference_property not in self._properties:
                available = ', '.join(self._properties.keys())
                raise PropertyNotFoundError(
                    f"Reference property '{reference_property}' not found. "
                    f"Available: {available}"
                )
            ref_prop = self._properties[reference_property]

        depth = ref_prop.depth

        # Auto-resample if requested
        if auto_resample:
            # Resample all properties to reference property's depth grid
            self.resample(depth_grid=depth)

        # Determine which properties to include
        if include is not None:
            # Only include specified properties
            props_to_export = {
                name: prop for name, prop in self._properties.items()
                if name in include
            }
            # Warn if some requested properties are missing
            missing = set(include) - set(props_to_export.keys())
            if missing:
                available = ', '.join(self._properties.keys())
                raise PropertyNotFoundError(
                    f"Properties not found: {', '.join(missing)}. "
                    f"Available: {available}"
                )
        elif exclude is not None:
            # Include all except excluded properties
            props_to_export = {
                name: prop for name, prop in self._properties.items()
                if name not in exclude
            }
        else:
            # Include all properties
            props_to_export = self._properties

        # Verify all properties on same grid if auto_resample is False
        if not auto_resample:
            for name, prop in props_to_export.items():
                if not np.array_equal(prop.depth, depth):
                    raise WellError(
                        f"Cannot export to DataFrame: property '{name}' has different "
                        f"depth grid than reference. Either set auto_resample=True or "
                        f"call well.resample() first to align all properties."
                    )

        # Build DataFrame
        data = {'DEPT': depth}
        for name, prop in props_to_export.items():
            # Apply labels to discrete properties if requested
            if discrete_labels and prop.labels:
                data[name] = prop._apply_labels(prop.values)
            else:
                data[name] = prop.values

        return pd.DataFrame(data)

    def export_to_las(
        self,
        filepath: Union[str, Path],
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        store_labels: bool = True,
        null_value: float = -999.25,
        use_template: Union[bool, LasFile, None] = None
    ) -> None:
        """
        Export well data to LAS 2.0 format file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Output LAS file path
        include : list[str], optional
            List of property names to include. If None, includes all properties.
        exclude : list[str], optional
            List of property names to exclude. If None, no properties are excluded.
        store_labels : bool, default True
            If True, store discrete property label mappings in the ~Parameter section.
            The actual data values remain numeric (standard LAS format).
        null_value : float, default -999.25
            Value to use for missing data in LAS file
        use_template : Union[bool, LasFile, None], optional
            If True, uses the primary (first) LAS file as template to preserve original
            metadata. If a LasFile object is provided, uses that specific file as template.
            If None or False, creates a new LAS file without template (default).
            Template mode preserves: ~Version info, ~Well parameters, and ~Parameter entries
            not related to discrete labels.

        Raises
        ------
        WellError
            If properties have different depth grids (call resample() first)
            If use_template=True but no source LAS files exist
        ValueError
            If both include and exclude are specified

        Examples
        --------
        >>> # Export all properties with labels stored in parameter section
        >>> well.export_to_las('output.las')

        >>> # Export specific properties
        >>> well.export_to_las('output.las', include=['PHIE', 'SW', 'PERM'])

        >>> # Export without storing discrete labels
        >>> well.export_to_las('output.las', store_labels=False)

        >>> # Export using original LAS as template (preserves metadata)
        >>> well.export_to_las('updated.las', use_template=True)

        >>> # Export using specific LAS file as template
        >>> template = well.original_las
        >>> well.export_to_las('output.las', use_template=template)
        """
        # Determine template LAS file if requested
        template_las = None
        if use_template is True:
            # Use original (first) LAS file as template
            if not self.original_las:
                raise WellError(
                    "Cannot use template: no source LAS files have been loaded. "
                    "Either load a LAS file first or set use_template=False."
                )
            template_las = self.original_las
        elif isinstance(use_template, LasFile):
            # Use specific LAS file as template
            template_las = use_template

        # Get DataFrame with properties (always numeric values for LAS)
        df = self.to_dataframe(
            include=include,
            exclude=exclude,
            auto_resample=True,
            discrete_labels=False  # Always export numeric values
        )

        # Build column name mapping: sanitized -> original
        column_rename_map = {}
        for prop_name, prop in self._properties.items():
            if prop_name in df.columns:
                column_rename_map[prop_name] = prop.original_name

        # Rename DataFrame columns to use original names for LAS export
        df = df.rename(columns=column_rename_map)

        # Build unit mappings from properties (use original names for export)
        unit_mappings = {'DEPT': 'm'}  # Default depth unit
        for prop_name, prop in self._properties.items():
            if prop_name in column_rename_map:
                unit_mappings[prop.original_name] = prop.unit

        # Collect discrete labels if store_labels is True (use original names)
        label_mappings = None
        if store_labels:
            label_mappings = {}
            for prop_name, prop in self._properties.items():
                if prop_name in column_rename_map and prop.labels:
                    label_mappings[prop.original_name] = prop.labels

        # Export using LasFile static method
        LasFile.export_las(
            filepath=filepath,
            well_name=self.name,
            df=df,
            unit_mappings=unit_mappings,
            null_value=null_value,
            discrete_labels=label_mappings if label_mappings else None,
            template_las=template_las
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Well('{self.name}', "
            f"properties={len(self._properties)})"
        )