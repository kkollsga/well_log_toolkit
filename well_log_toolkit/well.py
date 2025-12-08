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
from .utils import sanitize_property_name, sanitize_well_name, filter_names

if TYPE_CHECKING:
    from .manager import WellDataManager
    from .visualization import Template, WellView


class SourceView:
    """
    View into a specific source's properties within a well.

    Enables access pattern: well.log.PHIE

    Parameters
    ----------
    source_name : str
        Name of the source
    properties : dict[str, Property]
        Dictionary of properties from this source
    """

    def __init__(self, source_name: str, properties: dict[str, Property]):
        self._source_name = source_name
        self._properties = properties

    def __getattr__(self, name: str) -> Property:
        """Enable property access: source.PHIE"""
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Try as-is (sanitized name)
        if name in self._properties:
            return self._properties[name]

        # Try sanitizing the name
        sanitized_name = sanitize_property_name(name)
        if sanitized_name in self._properties:
            return self._properties[sanitized_name]

        # Not found
        available = ', '.join(self._properties.keys())
        raise AttributeError(
            f"Source '{self._source_name}' has no property '{name}'. "
            f"Available properties: {available or 'none'}"
        )

    def __repr__(self) -> str:
        return f"SourceView('{self._source_name}', properties={len(self._properties)})"

    def data(
        self,
        include: Optional[Union[str, list[str]]] = None,
        discrete_labels: bool = True,
        clip_edges: bool = True,
        clip_to_property: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export properties from this source to DataFrame.

        Parameters
        ----------
        include : list[str], optional
            List of property names to include. If None, includes all properties.
        discrete_labels : bool, default True
            If True, apply label mappings to discrete properties
        clip_edges : bool, default True
            If True, remove rows at start/end where all data columns contain NaN
        clip_to_property : str, optional
            Clip output to the defined range of this specific property

        Returns
        -------
        pd.DataFrame
            DataFrame with DEPT column and property values

        Examples
        --------
        >>> df = well.CompLogs.data()
        >>> df = well.CompLogs.data(include=['PHIE', 'SW'])
        >>> df = well.CompLogs.data(clip_to_property='PHIE')
        """
        if not self._properties:
            return pd.DataFrame()

        # Filter properties if requested
        if include:
            props = {k: v for k, v in self._properties.items() if k in include}
        else:
            props = self._properties

        if not props:
            return pd.DataFrame()

        # Get first property for depth reference
        first_prop = next(iter(props.values()))
        result = {'DEPT': first_prop.depth.copy()}

        # Add each property
        for name, prop in props.items():
            if discrete_labels and prop.type == 'discrete' and prop.labels:
                # Apply labels
                labeled_values = prop._apply_labels(prop.values)
                result[name] = labeled_values
            else:
                result[name] = prop.values.copy()

        df = pd.DataFrame(result)

        # Clip to specific property's defined range
        if clip_to_property and clip_to_property in df.columns:
            not_nan = df[clip_to_property].notna()
            if not_nan.any():
                first_valid = not_nan.idxmax()
                last_valid = not_nan[::-1].idxmax()
                df = df.loc[first_valid:last_valid].reset_index(drop=True)
        elif clip_edges and len(df) > 0:
            # Clip edges to remove leading/trailing NaN rows
            data_cols = [col for col in df.columns if col != 'DEPT']
            if data_cols:
                not_all_nan = df[data_cols].notna().any(axis=1)
                if not_all_nan.any():
                    first_valid = not_all_nan.idxmax()
                    last_valid = not_all_nan[::-1].idxmax()
                    df = df.loc[first_valid:last_valid].reset_index(drop=True)

        return df

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return first n rows of data from this source.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        pd.DataFrame
            First n rows of data

        Examples
        --------
        >>> well.CompLogs.head()
        >>> well.CompLogs.head(10)
        """
        return self.data().head(n)


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
        List of unique property names across all sources
    sources : list[str]
        List of source names (sanitized from LAS file names)
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
        # New source-aware storage structure
        self._sources: dict[str, dict] = {}  # {source_name: {'path': Path, 'las_file': LasFile, 'properties': {name: Property}}}
        # Track sources marked for deletion (to delete files on save)
        self._deleted_sources: list[str] = []  # List of source names to delete
        # Track sources marked for rename (to rename files on save)
        self._renamed_sources: dict[str, str] = {}  # {old_name: new_name}

    def __setattr__(self, name: str, value):
        """
        Intercept attribute assignment to handle property operations.

        Behavior:
        - If assigning to NEW property name: creates computed property
        - If assigning to EXISTING property: overwrites its data in-place

        Examples
        --------
        >>> well.new_phie = well.PHIE * 0.01  # Creates NEW computed property
        >>> well.PHIE = well.PHIE * 0.01      # OVERWRITES existing PHIE data
        >>> well.Reservoir = well.PHIE > 0.15 # Creates NEW discrete property
        """
        # Check if this is a Property being assigned
        if isinstance(value, Property) and not name.startswith('_'):
            # Check if property already exists
            try:
                existing_prop = self.get_property(name)
                # Property exists - overwrite its data
                self._overwrite_property(name, value)
            except (AttributeError, PropertyNotFoundError):
                # Property doesn't exist - create new computed property
                self._add_computed_property(name, value)
        else:
            # Normal attribute assignment
            object.__setattr__(self, name, value)

    def _add_computed_property(self, name: str, prop: Property):
        """
        Add a computed property to the well.

        Computed properties are stored in a special 'computed' source and
        can be exported like any other property.

        Parameters
        ----------
        name : str
            Name for the new property
        prop : Property
            Property object to add
        """
        # Sanitize the name
        from .utils import sanitize_property_name
        sanitized_name = sanitize_property_name(name)

        # Update property metadata
        prop.name = sanitized_name
        prop.original_name = name
        prop.parent_well = self
        prop.source_name = 'computed'

        # Create or update 'computed' source
        if 'computed' not in self._sources:
            self._sources['computed'] = {
                'path': None,
                'las_file': None,
                'properties': {},
                'modified': True
            }

        # Add property to computed source
        self._sources['computed']['properties'][sanitized_name] = prop
        self._sources['computed']['modified'] = True

    def _overwrite_property(self, name: str, new_prop: Property):
        """
        Overwrite an existing property's data in-place.

        This maintains the property's source location but updates its depth and values.
        If the new property has different dimensions, the entire property is replaced.

        Parameters
        ----------
        name : str
            Name of existing property to overwrite
        new_prop : Property
            Property object with new data
        """
        from .utils import sanitize_property_name
        sanitized_name = sanitize_property_name(name)

        # Find which source contains this property
        source_name = None
        for src_name, src_data in self._sources.items():
            if sanitized_name in src_data['properties']:
                source_name = src_name
                break

        if source_name is None:
            # Property doesn't exist - shouldn't reach here, but handle gracefully
            self._add_computed_property(name, new_prop)
            return

        # Get the existing property
        existing_prop = self._sources[source_name]['properties'][sanitized_name]

        # Update the existing property's data
        existing_prop._depth_cache = new_prop.depth.copy()
        existing_prop._values_cache = new_prop.values.copy()

        # Update metadata if provided
        if new_prop.unit:
            existing_prop.unit = new_prop.unit
        if new_prop.description:
            existing_prop.description = new_prop.description
        if new_prop.labels:
            existing_prop._labels = new_prop.labels.copy()
        if new_prop.type:
            existing_prop._type = new_prop.type

        # Mark source as modified
        self._sources[source_name]['modified'] = True

    def load_las(
        self,
        las: Union[LasFile, str, Path, list[Union[str, Path]]],
        sampled: bool = False,
        resample_method: Optional[str] = None,
        merge: bool = False,
        combine: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> 'Well':
        """
        Load LAS file(s) into this well, organized by source.

        Properties are grouped by source (LAS file). The source name is derived
        from the filename stem (without extension), sanitized for Python attribute
        access. If the filename starts with the well name, that prefix is removed
        to avoid redundancy (e.g., "36_7_5_B_CorePor.las" becomes "CorePor").

        Parameters
        ----------
        las : Union[LasFile, str, Path, list[Union[str, Path]]]
            Either a LasFile instance, path to LAS file, or list of LAS file paths
        sampled : bool, default False
            If True, mark all properties from this source as 'sampled' type.
            Use this for core plug data or other point measurements where
            boundary insertion during filtering should be disabled.
        resample_method : str, optional
            Method to use if depth grids are incompatible. Only used when loading
            data with different depth grids than existing data:
            - None (default): Will raise error if grids incompatible
            - 'linear': Linear interpolation (for continuous properties)
            - 'nearest': Nearest neighbor (for discrete/sampled)
            - 'previous': Forward-fill / previous value (for discrete)
            - 'next': Backward-fill / next value
        merge : bool, default False
            If True and a source with the same name already exists, merge the new
            properties into the existing source instead of overwriting it.
            When merging with incompatible depth grids, resample_method must be specified.
            If False (default), existing source is overwritten with warning.
        combine : str, optional
            When loading multiple files (list), combine them into a single source:
            - None (default): Load files as separate sources, no combining
            - 'match': Combine using match method (safest, errors on mismatch)
            - 'resample': Combine using resample method (interpolates to first file)
            - 'concat': Combine using concat method (merges all unique depths)
        source_name : str, optional
            Name for combined source when combine is specified. If not specified,
            uses 'combined_match', 'combined_resample', or 'combined_concat'

        Returns
        -------
        Well
            Self for method chaining

        Raises
        ------
        WellNameMismatchError
            If LAS well name doesn't match this well
        WellError
            If no depth column found in LAS file
            If merging with incompatible depths and no resample_method specified

        Examples
        --------
        >>> # Load single file
        >>> well.load_las("36_7-5_B_CorePor.las")  # Source name: "CorePor"

        >>> # Load multiple files as separate sources
        >>> well.load_las(["file1.las", "file2.las", "file3.las"])
        >>> print(well.sources)  # ['file1', 'file2', 'file3']

        >>> # Load and combine multiple files
        >>> well.load_las(
        ...     ["file1.las", "file2.las", "file3.las"],
        ...     combine="match",
        ...     source_name="CombinedLogs"
        ... )
        >>> print(well.sources)  # ['CombinedLogs']

        >>> # Load core plug data as sampled
        >>> well.load_las("36_7-5_B_Core.las", sampled=True)

        >>> # Load with resample combining
        >>> well.load_las(
        ...     ["log1.las", "log2.las"],
        ...     combine="resample",
        ...     source_name="CombinedLogs"
        ... )

        >>> # Merge into existing source (legacy behavior)
        >>> well.load_las("CorePor.las")  # Load initial properties
        >>> well.load_las("CorePor_Extra.las", merge=True)  # Merge new properties
        """
        # Handle list of files
        if isinstance(las, list):
            # Load all files as separate sources
            loaded_sources = []
            for las_file in las:
                # Recursively call load_las for each file (without merge or combine)
                self.load_las(las_file, sampled=sampled, resample_method=resample_method, merge=False, combine=None)
                # Get the source name that was just created
                loaded_sources.append(self.sources[-1])

            # Combine if requested
            if combine in {'match', 'resample', 'concat'}:
                # Determine combined source name
                combined_source_name = source_name or f'combined_{combine}'

                # Merge all loaded sources
                self.merge(
                    method=combine,
                    sources=loaded_sources,
                    source_name=combined_source_name
                )

                # Remove original sources
                self.remove_source(loaded_sources)

            return self

        # Single file logic below
        # Validate combine parameter for single file
        if combine is not None:
            raise ValueError(
                f"combine='{combine}' is only valid when loading multiple files (list). "
                f"For single file loading, use merge=True to merge into existing source."
            )

        # Parse if path provided
        filepath = None
        if isinstance(las, (str, Path)):
            filepath = Path(las)
            las = LasFile(las)
        elif hasattr(las, 'filepath') and las.filepath:
            filepath = Path(las.filepath)

        # Validate well name
        if las.well_name != self.name:
            raise WellNameMismatchError(
                f"Well name mismatch: attempting to load '{las.well_name}' "
                f"into well '{self.name}'. Create a new well or use "
                f"manager.load_las() for automatic well creation."
            )

        # Generate source name from filename stem
        if filepath:
            base_source_name = filepath.stem  # Filename without extension

            # Remove well name prefix BEFORE sanitizing to avoid prop_ prefix issue
            # Try multiple formats of well name (with hyphens and without)
            well_name_variants = [
                sanitize_well_name(self.name, keep_hyphens=True),  # "36_7-5_ST2"
                self.sanitized_name  # "36_7_5_ST2"
            ]

            # Check each variant
            suffix_found = False
            for well_variant in well_name_variants:
                # Check if filename starts with this variant (case-insensitive for robustness)
                if base_source_name.lower().startswith(well_variant.lower()):
                    # Remove the well name prefix and any connecting underscores or hyphens
                    suffix = base_source_name[len(well_variant):]
                    suffix = suffix.lstrip('_-')
                    if suffix:
                        base_source_name = suffix
                        suffix_found = True
                        break

            # Now sanitize for Python attribute access
            base_source_name = sanitize_property_name(base_source_name)
        else:
            # Fallback for LasFile objects without filepath
            base_source_name = 'unknown_source'

        # Handle source name and merge logic
        source_name = base_source_name

        # Check if source already exists
        if source_name in self._sources:
            if merge:
                # MERGE MODE: Add new properties to existing source
                existing_source = self._sources[source_name]
                existing_las = existing_source['las_file']

                # Check depth compatibility
                if existing_las:
                    compatibility = existing_las.check_depth_compatibility(las)

                    if not compatibility['compatible'] and compatibility['requires_resampling']:
                        # Incompatible depths - need resampling
                        if resample_method is None:
                            raise WellError(
                                f"Cannot merge into source '{source_name}': incompatible depth grids.\n"
                                f"Reason: {compatibility['reason']}\n"
                                f"Existing depth: {compatibility['existing']}\n"
                                f"New depth: {compatibility['new']}\n"
                                f"To merge, specify a resample_method: 'linear', 'nearest', 'previous', or 'next'"
                            )
                        else:
                            print(f"Merging into source '{source_name}' with resampling (method={resample_method})")
                    elif compatibility['compatible'] and compatibility['reason'] != 'Identical depth grids':
                        # Compatible but not identical (different coverage or NaN tails)
                        print(f"⚠ Merging into source '{source_name}': {compatibility['reason']}")
                    else:
                        # Identical depth grids
                        print(f"Merging into source '{source_name}' (identical depth grids)")
                else:
                    # No existing LAS file (synthetic source) - just warn
                    print(f"Merging into synthetic source '{source_name}'")
            else:
                # OVERWRITE MODE: Replace existing source
                print(f"Overwriting existing source '{source_name}' in well '{self.name}'")

        # OPTIMIZATION: Don't load data yet - use lazy loading
        # Data will be loaded from las.data() when property.depth or property.values is first accessed
        depth_col = las.depth_column

        if depth_col is None:
            raise WellError(f"No depth column found in LAS file")

        # Get discrete property information from LAS file (from header, no data loading)
        discrete_props = las.discrete_properties

        # Create source entry
        source_properties = {}

        # Create lazy property shells for each curve
        for curve_name in las.curves.keys():
            if curve_name == depth_col:
                continue  # Skip depth itself

            curve_meta = las.curves[curve_name]
            prop_name = curve_meta.get('alias') or curve_name

            # Check if this property is marked as discrete
            is_discrete = prop_name in discrete_props

            # Determine property type
            if sampled:
                # Override to sampled for all properties if loading sampled data
                prop_type = 'sampled'
            elif is_discrete:
                prop_type = 'discrete'
            else:
                prop_type = curve_meta['type']

            # Get labels, colors, styles, and thicknesses if property is discrete
            labels = None
            colors = None
            styles = None
            thicknesses = None
            if is_discrete:
                labels = las.get_discrete_labels(prop_name)
                colors = las.get_discrete_colors(prop_name)
                styles = las.get_discrete_styles(prop_name)
                thicknesses = las.get_discrete_thicknesses(prop_name)

            # Sanitize property name for Python attribute access
            sanitized_prop_name = sanitize_property_name(prop_name)

            # Create LAZY property shell - no data loaded yet
            # Data will be loaded from source_las when first accessed
            prop = Property(
                name=sanitized_prop_name,
                parent_well=self,
                unit=curve_meta['unit'],
                prop_type=prop_type,
                description=curve_meta['description'],
                null_value=las.null_value,
                labels=labels,
                colors=colors,
                styles=styles,
                thicknesses=thicknesses,
                source_las=las,
                source_name=source_name,
                original_name=prop_name,
                lazy=True  # Enable lazy loading
            )

            source_properties[sanitized_prop_name] = prop

        # Store or merge source with its properties
        if merge and source_name in self._sources:
            # MERGE MODE: Add/update properties in existing source
            existing_source = self._sources[source_name]

            # Add new properties or overwrite existing ones with same name
            for prop_name, prop in source_properties.items():
                if prop_name in existing_source['properties']:
                    # Property exists - overwrite (smart merge)
                    print(f"  Overwriting property '{prop_name}' in source '{source_name}'")
                else:
                    # New property - add it
                    print(f"  Adding property '{prop_name}' to source '{source_name}'")

                existing_source['properties'][prop_name] = prop

            # Keep the original LAS file as reference (don't replace with new one)
            # Update resample_method if specified
            if resample_method is not None:
                existing_source['resample_method'] = resample_method

            # Mark as modified since we added/updated properties
            existing_source['modified'] = True
        else:
            # OVERWRITE MODE: Create new source
            # Mark as unmodified if loaded from file, modified if synthetic (no filepath)
            self._sources[source_name] = {
                'path': filepath,
                'las_file': las,
                'properties': source_properties,
                'modified': filepath is None,  # True if synthetic (no file), False if from disk
                'resample_method': resample_method  # Store for lazy loading
            }

        return self  # Enable chaining

    def add_dataframe(
        self,
        df: pd.DataFrame,
        unit_mappings: Optional[dict[str, str]] = None,
        type_mappings: Optional[dict[str, str]] = None,
        label_mappings: Optional[dict[str, dict[int, str]]] = None
    ) -> 'Well':
        """
        Add properties from a DataFrame to this well as a new source.

        The DataFrame must contain a DEPT column. All other columns will be
        added as properties grouped under a source named 'external_df', 'external_df_1', etc.

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
        >>> print(well.sources)  # ['log', 'external_df']
        >>> # Access properties
        >>> well.external_df.Zone
        """
        # Generate base source name
        base_source_name = 'external_df'

        # Check if source already exists and notify user of overwrite
        source_name = base_source_name
        if source_name in self._sources:
            print(f"Overwriting existing source '{source_name}' in well '{self.name}'")

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

    def rename_source(self, old_name: str, new_name: str) -> 'Well':
        """
        Rename a source to resolve conflicts or improve clarity.

        The source is marked for rename. When the project is saved,
        the corresponding LAS file will be renamed on disk.

        Parameters
        ----------
        old_name : str
            Current source name
        new_name : str
            New source name (will be sanitized)

        Returns
        -------
        Well
            Self for method chaining

        Raises
        ------
        KeyError
            If old source doesn't exist
        ValueError
            If new name already exists or conflicts

        Examples
        --------
        >>> well.load_las("log.las")  # Creates source named 'log'
        >>> well.load_las("core.las")  # Creates source named 'core'
        >>> # Both have PHIE - rename one for clarity
        >>> well.rename_source("log", "wireline")
        >>> well.wireline.PHIE  # Access renamed source
        >>> manager.save()  # Will rename log.las to wireline.las on disk
        """
        # Validate old source exists
        if old_name not in self._sources:
            available = ', '.join(self._sources.keys())
            raise KeyError(
                f"Source '{old_name}' not found. "
                f"Available sources: {available or 'none'}"
            )

        # Sanitize new name
        sanitized_new_name = sanitize_property_name(new_name)

        # Validate new name doesn't already exist
        if sanitized_new_name in self._sources and sanitized_new_name != old_name:
            raise ValueError(
                f"Source '{sanitized_new_name}' already exists. "
                "Choose a different name or remove the existing source first."
            )

        # If old and new are the same after sanitization, nothing to do
        if sanitized_new_name == old_name:
            return self

        # Mark source for rename (to rename file on save)
        self._renamed_sources[old_name] = sanitized_new_name

        # Get source data
        source_data = self._sources[old_name]

        # Update all properties in this source to have new source_name
        for prop in source_data['properties'].values():
            prop.source_name = sanitized_new_name

        # Mark source as modified so it gets exported with new name
        source_data['modified'] = True

        # Move source to new key
        self._sources[sanitized_new_name] = source_data
        del self._sources[old_name]

        return self

    def mark_source_modified(self, name: str) -> 'Well':
        """
        Mark a source as modified so it will be re-exported on save.

        This is useful if you manually modify property values and want
        to ensure the source is saved to disk.

        Parameters
        ----------
        name : str
            Source name to mark as modified

        Returns
        -------
        Well
            Self for method chaining

        Raises
        ------
        KeyError
            If source doesn't exist

        Examples
        --------
        >>> # Modify property values directly
        >>> well.log.PHIE.values *= 1.1  # Scale up by 10%
        >>> well.mark_source_modified("log")
        >>> manager.save()  # Will re-export the modified source
        """
        if name not in self._sources:
            available = ', '.join(self._sources.keys())
            raise KeyError(
                f"Source '{name}' not found. "
                f"Available sources: {available or 'none'}"
            )

        self._sources[name]['modified'] = True
        return self

    def remove_source(self, name: Union[str, list[str]]) -> 'Well':
        """
        Remove one or more sources and all their properties from the well.

        The sources are marked for deletion. When the project is saved,
        the corresponding LAS files will be deleted from disk.

        Parameters
        ----------
        name : str or list[str]
            Source name(s) to remove. Can be a single string or list of strings.

        Returns
        -------
        Well
            Self for method chaining

        Raises
        ------
        KeyError
            If any source doesn't exist

        Examples
        --------
        >>> # Remove single source
        >>> well.load_las("log.las")
        >>> well.load_las("core.las")
        >>> well.sources  # ['log', 'core']
        >>> well.remove_source("log")
        >>> well.sources  # ['core']
        >>> manager.save()  # Will delete log.las from project folder
        >>>
        >>> # Remove multiple sources
        >>> well.remove_source(["log", "core"])
        >>> well.sources  # []
        """
        # Convert to list if single string
        names = [name] if isinstance(name, str) else name

        # Validate all sources exist first
        for source_name in names:
            if source_name not in self._sources:
                available = ', '.join(self._sources.keys())
                raise KeyError(
                    f"Source '{source_name}' not found. "
                    f"Available sources: {available or 'none'}"
                )

        # Remove all sources
        for source_name in names:
            # Mark source for deletion (to delete file on save)
            self._deleted_sources.append(source_name)

            # Remove from active sources
            del self._sources[source_name]

        return self

    def __getattr__(self, name: str) -> Union[SourceView, Property]:
        """
        Enable source and property access via attributes.

        Priority order:
        1. Check if name is a source name: well.log -> SourceView
        2. Check if property is unique across all sources: well.PHIE -> Property
        3. If property exists in multiple sources, raise error

        This is called when normal attribute lookup fails.
        Supports both original and sanitized property names.

        Examples
        --------
        >>> well.log  # Returns SourceView for 'log' source
        >>> well.log.PHIE  # Returns PHIE property from log source
        >>> well.SW  # Returns SW if unique across all sources
        >>> well.PHIE  # Error if PHIE exists in multiple sources
        """
        # Don't intercept private attributes, methods, or class attributes
        if name.startswith('_') or name in [
            'name', 'sanitized_name', 'parent_manager', 'properties', 'sources',
            'load_las', 'get_property', 'merge', 'data', 'original_las',
            'add_dataframe', 'to_las', 'export_to_las', 'rename_source', 'remove_source',
            'export_sources', 'delete_renamed_sources', 'delete_marked_sources', 'mark_source_modified'
        ]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Priority 1: Check if name is a source name
        if name in self._sources:
            return SourceView(name, self._sources[name]['properties'])

        # Priority 2: Check if property exists across sources
        # Collect all properties with this name (try both as-is and sanitized)
        matching_sources = []
        matching_property = None

        for source_name, source_data in self._sources.items():
            properties = source_data['properties']

            # Try as-is (sanitized name)
            if name in properties:
                matching_sources.append(source_name)
                matching_property = properties[name]

            # Try sanitizing the name (in case original name was somehow used)
            elif not matching_sources:  # Only try sanitizing if we haven't found anything yet
                sanitized_name = sanitize_property_name(name)
                if sanitized_name in properties:
                    matching_sources.append(source_name)
                    matching_property = properties[sanitized_name]

        # If property found in exactly one source, return it
        if len(matching_sources) == 1:
            return matching_property

        # If property found in multiple sources, raise error
        if len(matching_sources) > 1:
            sources_str = ', '.join(matching_sources)
            raise AttributeError(
                f"Property '{name}' is ambiguous - exists in multiple sources: {sources_str}. "
                f"Use explicit source access: well.{matching_sources[0]}.{name}"
            )

        # Not found - provide helpful error with available sources and properties
        available_sources = ', '.join(self._sources.keys())
        all_properties = set()
        for source_data in self._sources.values():
            all_properties.update(source_data['properties'].keys())
        available_properties = ', '.join(sorted(all_properties))

        raise AttributeError(
            f"Well '{self.name}' has no source or property named '{name}'. "
            f"Available sources: {available_sources or 'none'}. "
            f"Available properties: {available_properties or 'none'}"
        )
    
    @property
    def properties(self) -> list[str]:
        """
        List of all accessible property names with their access patterns.

        If a property is unique across all sources, it's listed as-is.
        If a property exists in multiple sources, each is listed with source prefix.
        If a source name conflicts with a property name, the property needs source prefix.

        Returns
        -------
        list[str]
            Sorted list of property access patterns

        Examples
        --------
        >>> well.properties
        ['PHIE', 'SW', 'log.PERM', 'core.PERM']  # PERM exists in both sources
        >>> # If source named 'PHIE' exists with property 'PHIE':
        >>> well.properties
        ['PHIE.PHIE', 'SW', 'PERM']  # PHIE property needs prefix due to source conflict
        """
        # Count occurrences of each property across sources
        property_sources = {}  # {prop_name: [source_name1, source_name2, ...]}
        for source_name, source_data in self._sources.items():
            for prop_name in source_data['properties'].keys():
                if prop_name not in property_sources:
                    property_sources[prop_name] = []
                property_sources[prop_name].append(source_name)

        # Build access patterns
        access_patterns = []
        for prop_name, sources in property_sources.items():
            if len(sources) == 1:
                # Property exists in only one source
                # Check if there's a source with the same name as the property
                if prop_name in self._sources:
                    # Source name conflicts - need to use source.property syntax
                    access_patterns.append(f"{sources[0]}.{prop_name}")
                else:
                    # No conflict - can access directly
                    access_patterns.append(prop_name)
            else:
                # Property exists in multiple sources - list each with source prefix
                for source_name in sources:
                    access_patterns.append(f"{source_name}.{prop_name}")

        return sorted(access_patterns)

    @property
    def sources(self) -> list[str]:
        """
        List of all source names loaded into this well.

        Source names are sanitized from LAS file names (without extension)
        or 'external_df' for DataFrames.

        Returns
        -------
        list[str]
            List of source names (sanitized, usable as attributes)

        Examples
        --------
        >>> well.load_las("log1.las")
        >>> well.add_dataframe(df)
        >>> print(well.sources)  # ['log1', 'external_df']
        >>> # Access sources
        >>> well.log1.PHIE
        >>> well.external_df.Zone
        """
        return list(self._sources.keys())

    @property
    def original_las(self) -> Optional[LasFile]:
        """
        Get the first (original) LAS file loaded into this well.

        Returns None if no sources have been loaded yet.

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
        if not self._sources:
            return None
        # Return the first source's LAS file
        first_source = next(iter(self._sources.values()))
        return first_source['las_file']
    
    def get_property(self, name: str, source: Optional[str] = None) -> Property:
        """
        Explicit property getter.

        If source is specified, gets property from that source only.
        If source is None, gets property if it's unique across all sources,
        raises error if ambiguous.

        Parameters
        ----------
        name : str
            Property name (original or sanitized)
        source : str, optional
            Source name to get property from. If None, property must be unique.

        Returns
        -------
        Property
            The requested property

        Raises
        ------
        PropertyNotFoundError
            If property not found or is ambiguous (exists in multiple sources)

        Examples
        --------
        >>> prop = well.get_property("PHIE")  # Gets PHIE if unique
        >>> prop = well.get_property("PHIE", source="log")  # Gets PHIE from log source
        """
        # If source specified, get from that source only
        if source is not None:
            if source not in self._sources:
                available_sources = ', '.join(self._sources.keys())
                raise PropertyNotFoundError(
                    f"Source '{source}' not found. Available sources: {available_sources or 'none'}"
                )

            properties = self._sources[source]['properties']

            # Try as-is
            if name in properties:
                return properties[name]

            # Try sanitizing
            sanitized_name = sanitize_property_name(name)
            if sanitized_name in properties:
                return properties[sanitized_name]

            # Not found in this source
            available = ', '.join(properties.keys())
            raise PropertyNotFoundError(
                f"Property '{name}' not found in source '{source}'. "
                f"Available properties: {available or 'none'}"
            )

        # No source specified - find property across all sources
        matching_sources = []
        matching_property = None

        for source_name, source_data in self._sources.items():
            properties = source_data['properties']

            # Try as-is
            if name in properties:
                matching_sources.append(source_name)
                matching_property = properties[name]
            else:
                # Try sanitizing
                sanitized_name = sanitize_property_name(name)
                if sanitized_name in properties:
                    matching_sources.append(source_name)
                    matching_property = properties[sanitized_name]

        # If found in exactly one source, return it
        if len(matching_sources) == 1:
            return matching_property

        # If found in multiple sources, raise error
        if len(matching_sources) > 1:
            sources_str = ', '.join(matching_sources)
            raise PropertyNotFoundError(
                f"Property '{name}' is ambiguous - exists in multiple sources: {sources_str}. "
                f"Use source parameter: well.get_property('{name}', source='{matching_sources[0]}')"
            )

        # Not found
        all_properties = set()
        for source_data in self._sources.values():
            all_properties.update(source_data['properties'].keys())
        available = ', '.join(sorted(all_properties))

        raise PropertyNotFoundError(
            f"Property '{name}' not found in well '{self.name}'. "
            f"Available properties: {available or 'none'}"
        )
    
    @staticmethod
    def _is_regular_grid(depth: np.ndarray, tolerance: float = 1e-6) -> tuple[bool, Optional[float]]:
        """
        Check if a depth grid has regular spacing.

        Parameters
        ----------
        depth : np.ndarray
            Depth values to check
        tolerance : float, default 1e-6
            Maximum allowed deviation from regular spacing

        Returns
        -------
        tuple[bool, Optional[float]]
            (is_regular, step_size)
            - is_regular: True if grid is regular
            - step_size: The step size if regular, None otherwise
        """
        if len(depth) < 2:
            return False, None

        # Calculate differences between consecutive depths
        diffs = np.diff(depth)

        # Check if all differences are approximately equal
        mean_diff = np.mean(diffs)
        max_deviation = np.max(np.abs(diffs - mean_diff))

        is_regular = max_deviation <= tolerance
        step_size = mean_diff if is_regular else None

        return is_regular, step_size

    def _merge_properties(
        self,
        method: str = 'match',
        sources: Optional[list[str]] = None,
        properties: Optional[list[str]] = None,
        depth_step: Optional[float] = None,
        depth_range: Optional[tuple[float, float]] = None,
        depth_grid: Optional[np.ndarray] = None,
        source_name: Optional[str] = None
    ) -> dict[str, Property]:
        """
        Internal method to merge properties without modifying originals.

        Returns a dictionary of new merged Property objects.

        Parameters
        ----------
        method : {'match', 'resample', 'concat'}, default 'match'
            Merge method:
            - 'match': Use first source's depth grid. Continuous properties must have
              exact depth match (errors otherwise). Discrete properties are resampled
              using interval logic (since they define depth intervals).
            - 'resample': Use first source's depth grid (or depth_grid if provided),
              interpolate other sources to this grid. If first source has irregular
              spacing, errors when other sources extend beyond its range.
            - 'concat': Merge all unique depths, fill NaN where depth doesn't exist
        sources : list[str], optional
            List of source names to include (required for 'match' and 'resample' methods)
        properties : list[str], optional
            List of property names to include
        depth_step : float, optional
            Not used (deprecated - kept for backward compatibility)
        depth_range : tuple[float, float], optional
            Not used (deprecated - kept for backward compatibility)
        depth_grid : np.ndarray, optional
            For 'resample' method: explicit depth grid to use. If None, uses first
            source's depth grid.
        source_name : str, optional
            Source name for merged properties. If None, generates 'merged_{method}'

        Returns
        -------
        dict[str, Property]
            Dictionary of merged properties {name: Property}

        Raises
        ------
        ValueError
            If invalid method specified
        WellError
            If no properties match the filters, if 'match' method detects
            incompatible depths, or if 'resample' with irregular grid detects
            extrapolation requirements
        """
        if method not in {'resample', 'concat', 'match'}:
            raise ValueError(
                f"method must be 'resample', 'concat', or 'match', got '{method}'"
            )

        # Filter properties by sources and/or names
        props_to_merge = {}

        # Iterate through all sources
        for source_name, source_data in self._sources.items():
            # Check source filter
            if sources is not None and source_name not in sources:
                continue

            # Add properties from this source
            for prop_name, prop in source_data['properties'].items():
                # Check property name filter
                if properties is not None and prop_name not in properties:
                    continue

                # If property already exists (from another source), we need to handle it
                # For now, we keep the first one encountered
                if prop_name not in props_to_merge:
                    props_to_merge[prop_name] = prop

        if not props_to_merge:
            available_sources = ', '.join(self._sources.keys())
            all_properties = set()
            for source_data in self._sources.values():
                all_properties.update(source_data['properties'].keys())
            available_properties = ', '.join(sorted(all_properties))

            raise WellError(
                "No properties match the specified filters. "
                f"Available sources: {available_sources or 'none'}. "
                f"Available properties: {available_properties or 'none'}"
            )

        # Generate source name for merged properties
        if source_name is None:
            source_name = f'merged_{method}'

        merged_properties = {}

        if method == 'resample':
            # Determine the reference depth grid
            if depth_grid is None:
                # Use first source's depth grid as reference
                if sources is None or len(sources) == 0:
                    raise WellError(
                        "For 'resample' method, you must specify sources list. "
                        "The first source will be used as the reference depth grid."
                    )

                first_source_name = sources[0]
                if first_source_name not in self._sources:
                    available_sources = ', '.join(self._sources.keys())
                    raise WellError(
                        f"First source '{first_source_name}' not found. "
                        f"Available sources: {available_sources}"
                    )

                # Get reference depth from first property in first source
                first_source_props = self._sources[first_source_name]['properties']
                if not first_source_props:
                    raise WellError(f"First source '{first_source_name}' has no properties")

                reference_depth = next(iter(first_source_props.values())).depth

                # Check if the reference depth grid is regular
                is_regular, _ = self._is_regular_grid(reference_depth)

                # Check other sources for values outside reference range
                ref_min, ref_max = reference_depth.min(), reference_depth.max()

                for source in sources[1:]:
                    if source not in self._sources:
                        continue

                    source_props = self._sources[source]['properties']
                    for prop_name, prop in source_props.items():
                        # Check property name filter
                        if properties is not None and prop_name not in properties:
                            continue

                        # Check if this source has depth values outside reference range
                        prop_min, prop_max = prop.depth.min(), prop.depth.max()

                        if prop_min < ref_min or prop_max > ref_max:
                            if not is_regular:
                                raise WellError(
                                    f"Cannot resample sources: source '{source}' has depth values "
                                    f"outside the range of first source '{first_source_name}' "
                                    f"[{ref_min:.2f}, {ref_max:.2f}], but the first source has an "
                                    f"irregular depth grid. Extrapolation is only allowed for regular grids. "
                                    f"Source '{source}' range: [{prop_min:.2f}, {prop_max:.2f}]. "
                                    f"Use method='concat' to merge all depths, or trim '{source}' to fit."
                                )
            else:
                # Use explicitly provided depth grid (for backward compatibility)
                reference_depth = depth_grid

            # Resample all properties to reference depth grid
            for name, prop in props_to_merge.items():
                resampled_values = Property._resample_to_grid(
                    prop.depth,
                    prop.values,
                    reference_depth,
                    method='linear' if prop.type == 'continuous' else 'previous'
                )

                # Create new property with merged source
                merged_prop = Property(
                    name=name,
                    depth=reference_depth.copy(),
                    values=resampled_values,
                    parent_well=self,
                    unit=prop.unit,
                    prop_type=prop.type,
                    description=prop.description,
                    null_value=-999.25,
                    labels=prop.labels,
                    source_las=None,
                    source_name=source_name,
                    original_name=prop.original_name
                )

                merged_properties[name] = merged_prop

        elif method == 'concat':
            # Collect all unique depths from selected properties
            all_depths = []
            for prop in props_to_merge.values():
                all_depths.extend(prop.depth)

            # Get unique sorted depths
            unique_depths = np.unique(np.array(all_depths))

            # Create new concatenated properties
            for name, prop in props_to_merge.items():
                # Create a mapping from original depth to values
                depth_to_value = dict(zip(prop.depth, prop.values))

                # Fill values for unique depths (NaN where depth doesn't exist)
                concat_values = np.array([
                    depth_to_value.get(d, np.nan) for d in unique_depths
                ])

                # Create new property with merged source
                merged_prop = Property(
                    name=name,
                    depth=unique_depths.copy(),
                    values=concat_values,
                    parent_well=self,
                    unit=prop.unit,
                    prop_type=prop.type,
                    description=prop.description,
                    null_value=-999.25,
                    labels=prop.labels,
                    source_las=None,
                    source_name=source_name,
                    original_name=prop.original_name
                )

                merged_properties[name] = merged_prop

        else:  # method == 'match'
            # Get the first source's depth grid as reference
            if sources is None or len(sources) == 0:
                raise WellError(
                    "For 'match' method, you must specify sources list. "
                    "The first source will be used as the reference depth grid."
                )

            first_source_name = sources[0]
            if first_source_name not in self._sources:
                available_sources = ', '.join(self._sources.keys())
                raise WellError(
                    f"First source '{first_source_name}' not found. "
                    f"Available sources: {available_sources}"
                )

            # Get reference depth from first property in first source
            first_source_props = self._sources[first_source_name]['properties']
            if not first_source_props:
                raise WellError(f"First source '{first_source_name}' has no properties")

            reference_depth = next(iter(first_source_props.values())).depth
            reference_depth_set = set(reference_depth)

            # Check other sources for incompatible depths
            # Continuous properties must have exact depth match
            # Discrete properties can be resampled (they define intervals)
            for source in sources[1:]:
                if source not in self._sources:
                    continue

                source_props = self._sources[source]['properties']
                for prop_name, prop in source_props.items():
                    # Check property name filter
                    if properties is not None and prop_name not in properties:
                        continue

                    # Only check continuous properties for exact depth match
                    if prop.type == 'continuous':
                        # Check if this source has depth values not in reference
                        prop_depth_set = set(prop.depth)
                        extra_depths = prop_depth_set - reference_depth_set

                        if extra_depths:
                            raise WellError(
                                f"Cannot match sources: continuous property '{prop_name}' from source '{source}' "
                                f"has depth values that don't exist in first source '{first_source_name}'. "
                                f"Found {len(extra_depths)} incompatible depth values. "
                                f"Use method='resample' or method='concat' instead."
                            )

            # Create matched properties using reference depth
            for name, prop in props_to_merge.items():
                if prop.type == 'continuous':
                    # Continuous properties: require exact depth match
                    depth_to_value = dict(zip(prop.depth, prop.values))

                    # Match values to reference depth (NaN where depth doesn't exist in this property)
                    matched_values = np.array([
                        depth_to_value.get(d, np.nan) for d in reference_depth
                    ])
                else:
                    # Discrete properties: resample using 'previous' method (interval-based)
                    # This works because discrete properties define intervals
                    matched_values = Property._resample_to_grid(
                        prop.depth,
                        prop.values,
                        reference_depth,
                        method='previous'
                    )

                # Create new property with merged source
                merged_prop = Property(
                    name=name,
                    depth=reference_depth.copy(),
                    values=matched_values,
                    parent_well=self,
                    unit=prop.unit,
                    prop_type=prop.type,
                    description=prop.description,
                    null_value=-999.25,
                    labels=prop.labels,
                    source_las=None,
                    source_name=source_name,
                    original_name=prop.original_name
                )

                merged_properties[name] = merged_prop

        return merged_properties

    def merge(
        self,
        method: str = 'match',
        sources: Optional[list[str]] = None,
        properties: Optional[list[str]] = None,
        depth_step: Optional[float] = None,
        depth_range: Optional[tuple[float, float]] = None,
        depth_grid: Optional[np.ndarray] = None,
        source_name: Optional[str] = None
    ) -> 'Well':
        """
        Merge properties from multiple sources into a new "merged" source.

        This operation replaces the selected properties with new merged versions.
        The original LAS files are not modified, but the in-memory properties
        will be replaced with the merged versions.

        Parameters
        ----------
        method : {'match', 'resample', 'concat'}, default 'match'
            Merge method:
            - 'match': Use first source's depth grid as reference. Continuous properties
              must have exact depth match (errors otherwise). Discrete properties are
              automatically resampled using interval logic, since they define depth
              intervals. This is the safest option when depths should already align.
            - 'resample': Use first source's depth grid, interpolate other sources
              to match. If first source has regular spacing (e.g., every 0.1m),
              allows extrapolation for other sources. If irregular spacing, raises
              error when other sources extend beyond first source's range.
            - 'concat': Merge all unique depths, fill NaN where depth doesn't exist
        sources : list[str], optional
            List of source names to include (e.g., ['CorePerm', 'CorePor'])
            Required for 'match' and 'resample' methods. The first source is used
            as the reference depth grid.
        properties : list[str], optional
            List of property names to include. If None, includes all properties
        depth_step : float, optional
            Not used (deprecated - kept for backward compatibility)
        depth_range : tuple[float, float], optional
            Not used (deprecated - kept for backward compatibility)
        depth_grid : np.ndarray, optional
            Not used (deprecated - kept for backward compatibility)
        source_name : str, optional
            Custom source name for merged properties. If None, uses 'merged_{method}'

        Returns
        -------
        Well
            Self for method chaining

        Raises
        ------
        ValueError
            If invalid method specified
        WellError
            If no properties match the filters, if 'match' method detects
            incompatible depths, or if 'resample' with irregular grid detects
            extrapolation requirements

        Examples
        --------
        >>> # Match sources with compatible depths (default, safest)
        >>> well.merge(
        ...     sources=['CorePerm', 'CorePor'],
        ...     source_name='CorePlugs'
        ... )

        >>> # Resample to first source's grid (allows interpolation)
        >>> well.merge(
        ...     method='resample',
        ...     sources=['CompLogs', 'CorePerm'],
        ...     source_name='Resampled'
        ... )

        >>> # Concatenate specific sources with custom name
        >>> well.merge(
        ...     method='concat',
        ...     sources=['log1', 'log2'],
        ...     source_name='combined_logs'
        ... )

        >>> # Match only specific properties
        >>> well.merge(
        ...     sources=['source1', 'source2'],
        ...     properties=['PHIE', 'SW'],
        ...     source_name='selected_props'
        ... )
        """
        # Get merged properties using internal method
        merged_properties = self._merge_properties(
            method=method,
            sources=sources,
            properties=properties,
            depth_step=depth_step,
            depth_range=depth_range,
            depth_grid=depth_grid,
            source_name=source_name
        )

        # Determine the merge source name
        merge_source_name = source_name if source_name else f'merged_{method}'

        # Create a new source with merged properties
        self._sources[merge_source_name] = {
            'path': None,  # Merged sources don't have a file path
            'las_file': None,  # No LAS file for merged source
            'properties': merged_properties
        }

        return self
    
    def data(
        self,
        reference_property: Optional[str] = None,
        include: Optional[Union[str, list[str]]] = None,
        exclude: Optional[Union[str, list[str]]] = None,
        merge_method: str = 'match',
        discrete_labels: bool = True,
        clip_edges: bool = True,
        clip_to_property: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export properties as DataFrame with optional merging and filtering.

        This method does NOT modify the original property depth grids. Properties
        are temporarily aligned using the specified merge method for DataFrame
        output only.

        Parameters
        ----------
        reference_property : str, optional
            Property to use as depth reference. All properties will be aligned
            to this property's depth grid using the merge_method. If not specified,
            defaults to the first property that was added (typically the first
            property from the first LAS file loaded).
        include : str or list[str], optional
            Property name(s) to include. If None, includes all properties.
            Can be a single string or a list of strings.
        exclude : str or list[str], optional
            Property name(s) to exclude. If both include and exclude are
            specified, exclude overrides (removes properties from include list).
            Can be a single string or a list of strings.
        merge_method : {'match', 'resample', 'concat'}, default 'match'
            Method to align properties to reference depth grid:
            - 'match': Require exact depth match for continuous properties (errors if
              not aligned). Discrete properties are automatically resampled using
              interval logic. Safest option.
            - 'resample': Interpolate properties to reference depth grid
            - 'concat': Merge unique depths, fill NaN where missing
        discrete_labels : bool, default True
            If True, apply label mappings to discrete properties with labels defined.
        clip_edges : bool, default True
            If True, remove rows at the start and end where all data columns
            (excluding DEPT) contain NaN values. This trims the DataFrame to the
            range where actual data exists.
        clip_to_property : str, optional
            Clip output to the defined range of this specific property. If specified,
            overrides clip_edges behavior and clips to where this property has valid data.

        Returns
        -------
        pd.DataFrame
            DataFrame with DEPT and selected properties

        Raises
        ------
        WellError
            If properties have different depth grids and merge_method is 'match',
            or if merge requirements fail
        PropertyNotFoundError
            If reference_property or included properties are not found

        Examples
        --------
        >>> # Export all properties (errors if depths don't match exactly)
        >>> df = well.data()

        >>> # Export with interpolation if depths don't align
        >>> df = well.data(merge_method='resample')

        >>> # Export with specific reference property
        >>> df = well.data(reference_property='PHIE')

        >>> # Include only specific properties
        >>> df = well.data(include=['PHIE', 'SW', 'PERM'])

        >>> # Include single property (string or list)
        >>> df = well.data(include='PHIE')  # Same as include=['PHIE']

        >>> # Exclude specific properties
        >>> df = well.data(exclude=['QC_Flag', 'Temp_Data'])

        >>> # Exclude single property
        >>> df = well.data(exclude='QC_Flag')

        >>> # Include with exclusions (exclude overrides)
        >>> df = well.data(include=['PHIE', 'SW', 'PERM', 'Zone'], exclude=['Zone'])

        >>> # Use concat merge method to include all unique depths
        >>> df = well.data(merge_method='concat')

        >>> # Disable label mapping for discrete properties
        >>> df = well.data(discrete_labels=False)

        >>> # Disable edge clipping to keep all NaN rows
        >>> df = well.data(clip_edges=False)

        >>> # Clip to specific property's defined range
        >>> df = well.data(clip_to_property='PHIE')
        """
        # Convert strings to lists for convenience
        if isinstance(include, str):
            include = [include]
        if isinstance(exclude, str):
            exclude = [exclude]

        if not self._sources:
            return pd.DataFrame()

        # Collect all properties across all sources for validation
        all_properties = {}  # {prop_name: (source_name, property)}
        for source_name, source_data in self._sources.items():
            for prop_name, prop in source_data['properties'].items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = (source_name, prop)

        if not all_properties:
            return pd.DataFrame()

        # Validate include list if provided
        if include is not None:
            missing = set(include) - set(all_properties.keys())
            if missing:
                available = ', '.join(all_properties.keys())
                raise PropertyNotFoundError(
                    f"Properties not found: {', '.join(missing)}. "
                    f"Available: {available}"
                )

        # Filter properties using helper function
        properties_filter = filter_names(all_properties.keys(), include, exclude)

        # Determine reference property
        if reference_property is None:
            # Default: use first property added (first from first source)
            ref_prop_name = next(iter(all_properties.keys()))
            _, ref_prop = all_properties[ref_prop_name]
        else:
            # Get specified reference property
            if reference_property not in all_properties:
                available = ', '.join(all_properties.keys())
                raise PropertyNotFoundError(
                    f"Reference property '{reference_property}' not found. "
                    f"Available: {available}"
                )
            ref_prop_name = reference_property
            _, ref_prop = all_properties[reference_property]

        depth = ref_prop.depth

        # Always use merge method to align properties
        # The merge_method parameter controls the behavior:
        # - 'match': errors if grids don't align (safe default)
        # - 'resample': interpolates to reference grid
        # - 'concat': merges all unique depths
        props_to_export = self._merge_properties(
            method=merge_method,
            properties=properties_filter,
            depth_grid=depth,
            source_name='temp_dataframe'
        )

        # Build DataFrame
        data = {'DEPT': depth}
        for name, prop in props_to_export.items():
            # Apply labels to discrete properties if requested
            if discrete_labels and prop.labels:
                data[name] = prop._apply_labels(prop.values)
            else:
                data[name] = prop.values

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
        Return first n rows of well data.

        Convenience method equivalent to `well.data(**kwargs).head(n)`.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return
        include : list[str], optional
            List of property names to include
        exclude : list[str], optional
            List of property names to exclude

        Returns
        -------
        pd.DataFrame
            First n rows of well data

        Examples
        --------
        >>> well.head()
        >>> well.head(10)
        >>> well.head(include=['PHIE', 'SW'])
        >>> well.head(exclude=['QC_Flag'])
        """
        return self.data(include=include, exclude=exclude).head(n)

    def to_las(
        self,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        store_labels: bool = True,
        use_template: Union[bool, LasFile, None] = None
    ) -> LasFile:
        """
        Convert well properties to a LasFile object.

        This creates a LasFile object from the well's properties. If properties have
        different depth grids, they will be automatically aligned using interpolation
        (resample method) to the first property's depth grid. This ensures the exported
        LAS file has a consistent depth grid.

        Note: Unlike the data() method which defaults to 'match' for safety, this method
        always resamples to ensure LAS export succeeds. If you need strict depth alignment
        checking, use well.merge(method='match') before calling to_las().

        Parameters
        ----------
        include : list[str], optional
            List of property names to include. If None, includes all properties.
        exclude : list[str], optional
            List of property names to exclude. If None, no properties are excluded.
        store_labels : bool, default True
            If True, store discrete property label mappings in the ~Parameter section.
        use_template : Union[bool, LasFile, None], optional
            If True, uses the primary (first) LAS file as template to preserve original
            metadata. If a LasFile object is provided, uses that specific file as template.

        Returns
        -------
        LasFile
            LasFile object ready for export or further manipulation

        Raises
        ------
        WellError
            If no properties to export or if use_template=True but no source LAS files exist
        ValueError
            If both include and exclude are specified
        PropertyNotFoundError
            If requested properties are not found

        Examples
        --------
        >>> # Create LasFile and export
        >>> las = well.to_las(include=['PHIE', 'SW', 'PERM'])
        >>> las.export('output.las')

        >>> # Create LasFile with template metadata preserved
        >>> las = well.to_las(use_template=True)
        >>> las.export('updated.las')

        >>> # Create LasFile, modify, then export
        >>> las = well.to_las()
        >>> # ... modify via las.set_data(df) if needed ...
        >>> las.export('output.las')

        >>> # If strict depth alignment is required, merge first
        >>> well.merge(method='match', sources=['source1', 'source2'])
        >>> las = well.to_las()
        """
        if not self._sources:
            raise WellError("No properties to export")

        # Collect all properties across all sources
        all_properties = {}  # {prop_name: property}
        for source_name, source_data in self._sources.items():
            for prop_name, prop in source_data['properties'].items():
                if prop_name not in all_properties:
                    all_properties[prop_name] = prop

        if not all_properties:
            raise WellError("No properties to export")

        # Validate include/exclude
        if include is not None and exclude is not None:
            raise ValueError(
                "Cannot specify both 'include' and 'exclude'. "
                "Use either include to specify properties to include, "
                "or exclude to specify properties to skip."
            )

        # Determine template LAS file if requested
        template_las = None
        if use_template is True:
            if not self.original_las:
                raise WellError(
                    "Cannot use template: no source LAS files have been loaded. "
                    "Either load a LAS file first or set use_template=False."
                )
            template_las = self.original_las
        elif isinstance(use_template, LasFile):
            template_las = use_template

        # Determine which properties to export
        if include is not None:
            properties_to_export = include
            missing = set(include) - set(all_properties.keys())
            if missing:
                available = ', '.join(all_properties.keys())
                raise PropertyNotFoundError(
                    f"Properties not found: {', '.join(missing)}. "
                    f"Available: {available}"
                )
        elif exclude is not None:
            properties_to_export = [name for name in all_properties.keys() if name not in exclude]
        else:
            properties_to_export = list(all_properties.keys())

        # Get properties dict
        props_dict = {name: all_properties[name] for name in properties_to_export}

        # Check if merge needed
        if len(props_dict) > 1:
            ref_depth = next(iter(props_dict.values())).depth
            needs_merge = any(
                not np.array_equal(prop.depth, ref_depth)
                for prop in props_dict.values()
            )

            if needs_merge:
                props_dict = self._merge_properties(
                    method='resample',
                    properties=properties_to_export,
                    depth_grid=ref_depth,
                    source_name='to_las_merged'
                )

        # Build DataFrame directly with original names
        ref_prop = next(iter(props_dict.values()))
        data = {'DEPT': ref_prop.depth}

        for name, prop in props_dict.items():
            data[prop.original_name] = prop.values

        df = pd.DataFrame(data)

        # Build unit mappings and type mappings
        unit_mappings = {'DEPT': 'm'}
        type_mappings = {}
        for prop in props_dict.values():
            unit_mappings[prop.original_name] = prop.unit
            type_mappings[prop.original_name] = prop.type

        # Collect discrete labels if requested
        label_mappings = None
        if store_labels:
            label_mappings = {}
            for prop in props_dict.values():
                if prop.labels:
                    label_mappings[prop.original_name] = prop.labels

        # Create LasFile from DataFrame
        las = LasFile.from_dataframe(
            df=df,
            well_name=self.name,
            source_name='from_well',
            unit_mappings=unit_mappings,
            type_mappings=type_mappings,
            label_mappings=label_mappings
        )

        # If template provided, copy over metadata
        if template_las:
            las.version_info = template_las.version_info.copy()
            # Copy well parameters except the ones we computed
            for key, value in template_las.well_info.items():
                if key not in ['STRT', 'STOP', 'STEP', 'NULL']:
                    las.well_info[key] = value

        return las

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

        This is a convenience method that calls to_las() and then exports.
        For more control, use to_las() to get a LasFile object first.

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
            If properties have different depth grids (call merge() first)
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

        >>> # For more control, use to_las() then export()
        >>> las = well.to_las(include=['PHIE', 'SW'])
        >>> # ... modify las if needed ...
        >>> las.export('output.las')
        """
        # Create LasFile object and export it
        las = self.to_las(
            include=include,
            exclude=exclude,
            store_labels=store_labels,
            use_template=use_template
        )
        las.export(filepath, null_value=null_value)

    def delete_renamed_sources(self, folder_path: Union[str, Path]) -> None:
        """
        Delete old LAS files for sources that were renamed.

        This method is called automatically by WellDataManager.save() after exporting
        sources to delete the old files with the previous names.

        Parameters
        ----------
        folder_path : Union[str, Path]
            Folder path containing the source LAS files

        Examples
        --------
        >>> well.rename_source("log", "wireline")
        >>> well.delete_renamed_sources("/path/to/project/well_36_7_5_B")
        Renamed source: 36_7-5_B_log.las -> 36_7-5_B_wireline.las (old file deleted)
        """
        if not self._renamed_sources:
            return

        folder = Path(folder_path)
        well_name_for_file = sanitize_well_name(self.name, keep_hyphens=True)

        for old_name, new_name in self._renamed_sources.items():
            # Build old filename (new file already exported by export_sources)
            old_filename = f"{well_name_for_file}_{old_name}.las"
            old_filepath = folder / old_filename

            # Delete old file if it exists
            if old_filepath.exists():
                old_filepath.unlink()
                print(f"Renamed source: {old_filename} -> {well_name_for_file}_{new_name}.las (old file deleted)")

        # Clear the rename list after processing
        self._renamed_sources.clear()

    def delete_marked_sources(self, folder_path: Union[str, Path]) -> None:
        """
        Delete LAS files for sources marked for deletion.

        This method is called automatically by WellDataManager.save() to remove
        source files that were deleted using remove_source().

        Parameters
        ----------
        folder_path : Union[str, Path]
            Folder path containing the source LAS files

        Examples
        --------
        >>> well.remove_source("log")
        >>> well.delete_marked_sources("/path/to/project/well_36_7_5_B")
        Deleted source file: /path/to/project/well_36_7_5_B/36_7-5_B_log.las
        """
        if not self._deleted_sources:
            return

        folder = Path(folder_path)
        well_name_for_file = sanitize_well_name(self.name, keep_hyphens=True)

        for source_name in self._deleted_sources:
            # Build expected filename
            filename = f"{well_name_for_file}_{source_name}.las"
            filepath = folder / filename

            # Delete if exists
            if filepath.exists():
                filepath.unlink()
                print(f"Deleted source file: {filepath}")

        # Clear the deletion list after processing
        self._deleted_sources.clear()

    def export_sources(self, folder_path: Union[str, Path]) -> None:
        """
        Export all sources as individual LAS files to a folder.

        Each source is exported as a separate LAS file with the well name prefix.
        Filename format: {well_sanitized_name}_{source_name}.las

        Parameters
        ----------
        folder_path : Union[str, Path]
            Folder path to export LAS files to. Will be created if it doesn't exist.

        Examples
        --------
        >>> well = manager.well_36_7_5_B
        >>> well.export_sources("/path/to/output")
        # Creates (hyphens preserved in filenames):
        # /path/to/output/36_7-5_B_Log.las
        # /path/to/output/36_7-5_B_CorePor.las
        """
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        for source_name, source_data in self._sources.items():
            # Build filename: well_name_source.las
            # Use sanitized name with hyphens preserved for better readability
            well_name_for_file = sanitize_well_name(self.name, keep_hyphens=True)
            filename = f"{well_name_for_file}_{source_name}.las"
            filepath = folder / filename

            # Get properties from this source
            properties = list(source_data['properties'].keys())
            if not properties:
                continue

            # OPTIMIZATION: Skip export if source hasn't been modified and file exists
            is_modified = source_data.get('modified', True)  # Default to True for safety
            if not is_modified and filepath.exists():
                # Source unchanged, skip export
                continue

            # Check if we have the original LAS file
            original_las = source_data.get('las_file')

            if original_las:
                # FAST PATH: Update existing LAS file and export directly
                # This preserves all metadata and is much faster than rebuilding from scratch

                # Update the data in the original LAS file
                ref_prop = next(iter(source_data['properties'].values()))
                depth_data = ref_prop.depth

                # Build data dictionary with original property names
                las_data = {original_las.depth_column: depth_data}
                for prop_name, prop in source_data['properties'].items():
                    las_data[prop.original_name] = prop.values

                # Update the LAS data
                original_las.set_data(pd.DataFrame(las_data))

                # Sync discrete metadata from Property objects to LAS parameter_info
                # This ensures user-modified labels, colors, styles, and thicknesses are persisted
                discrete_props = []
                for prop_name, prop in source_data['properties'].items():
                    if prop.type == 'discrete' and prop.labels:
                        discrete_props.append(prop.original_name)
                        # Add label mappings to parameter_info
                        for value, label in prop.labels.items():
                            param_name = f"{prop.original_name}_{value}"
                            original_las.parameter_info[param_name] = label
                    else:
                        # Remove any old discrete label entries for non-discrete properties
                        # First check if this property was previously discrete
                        keys_to_remove = [
                            key for key in list(original_las.parameter_info.keys())
                            if key.startswith(f"{prop.original_name}_") and key[len(prop.original_name)+1:].isdigit()
                        ]
                        for key in keys_to_remove:
                            del original_las.parameter_info[key]

                    # Sync colors mapping
                    if prop.type == 'discrete' and prop.colors:
                        for value, color in prop.colors.items():
                            param_name = f"{prop.original_name}_{value}_COLOR"
                            original_las.parameter_info[param_name] = color

                    # Sync styles mapping
                    if prop.type == 'discrete' and prop.styles:
                        for value, style in prop.styles.items():
                            param_name = f"{prop.original_name}_{value}_STYLE"
                            original_las.parameter_info[param_name] = style

                    # Sync thicknesses mapping
                    if prop.type == 'discrete' and prop.thicknesses:
                        for value, thickness in prop.thicknesses.items():
                            param_name = f"{prop.original_name}_{value}_THICKNESS"
                            original_las.parameter_info[param_name] = str(thickness)

                # Update DISCRETE_PROPS list
                if discrete_props:
                    original_las.parameter_info['DISCRETE_PROPS'] = ','.join(sorted(discrete_props))
                elif 'DISCRETE_PROPS' in original_las.parameter_info:
                    del original_las.parameter_info['DISCRETE_PROPS']

                # Export directly
                original_las.export(filepath)

            else:
                # SLOW PATH: Create new LAS file from scratch
                # Used for sources created with add_dataframe() or load_tops()
                ref_prop = next(iter(source_data['properties'].values()))
                depth = ref_prop.depth

                data = {'DEPT': depth}
                unit_mappings = {'DEPT': 'm'}
                type_mappings = {}
                label_mappings = {}
                color_mappings = {}
                style_mappings = {}
                thickness_mappings = {}

                for prop_name, prop in source_data['properties'].items():
                    data[prop.original_name] = prop.values
                    unit_mappings[prop.original_name] = prop.unit
                    type_mappings[prop.original_name] = prop.type
                    if prop.labels:
                        label_mappings[prop.original_name] = prop.labels
                    if prop.colors:
                        color_mappings[prop.original_name] = prop.colors
                    if prop.styles:
                        style_mappings[prop.original_name] = prop.styles
                    if prop.thicknesses:
                        thickness_mappings[prop.original_name] = prop.thicknesses

                df = pd.DataFrame(data)

                # Create LasFile from DataFrame with all metadata
                las = LasFile.from_dataframe(
                    df=df,
                    well_name=self.name,
                    source_name=source_name,
                    unit_mappings=unit_mappings,
                    type_mappings=type_mappings,
                    label_mappings=label_mappings if label_mappings else None,
                    color_mappings=color_mappings if color_mappings else None,
                    style_mappings=style_mappings if style_mappings else None,
                    thickness_mappings=thickness_mappings if thickness_mappings else None
                )

                # Export to file
                las.export(filepath)

    def WellView(
        self,
        depth_range: Optional[tuple[float, float]] = None,
        tops: Optional[list[str]] = None,
        template: Optional[Union['Template', dict, str]] = None,
        figsize: Optional[tuple[float, float]] = None,
        dpi: int = 100,
        header_config: Optional[dict] = None
    ) -> 'WellView':
        """
        Create a well log display for this well.

        Convenience method that creates a WellView object for visualization.

        Parameters
        ----------
        depth_range : tuple[float, float], optional
            Depth interval to display [start_depth, end_depth].
            Mutually exclusive with `tops` parameter.
        tops : list[str], optional
            List of formation top names to display. The depth range will be calculated
            automatically from the minimum and maximum depths of these tops, with 5%
            padding added (minimum range of 50m).
            Mutually exclusive with `depth_range` parameter.
            Requires that formation tops have been loaded in the well or added to the template.
        template : Union[Template, dict, str], optional
            Display template (Template object, dict config, or name from manager)
        figsize : tuple[float, float], optional
            Figure size (width, height) in inches
        dpi : int, default 100
            Figure resolution
        header_config : dict, optional
            Header styling configuration. Supported keys:
            - header_box_top (float): Fixed top position of header boxes
            - header_title_spacing (float): Vertical space between log name and scale line
            - header_log_spacing (float): Vertical space allocated per log
            - header_top_padding (float): Padding above content in header box
            - header_bottom_padding (float): Padding below content in header box

        Returns
        -------
        WellView
            Well log display object

        Examples
        --------
        >>> # Simple display with default template
        >>> view = well.WellView(depth_range=[2800, 3000])
        >>> view.show()
        >>>
        >>> # Use stored template from manager
        >>> view = well.WellView(depth_range=[2800, 3000], template="reservoir")
        >>> view.show()
        >>>
        >>> # Use formation tops to auto-calculate range
        >>> view = well.WellView(tops=["Top_Brent", "Top_Statfjord"], template="reservoir")
        >>> view.show()
        >>>
        >>> # Use custom template
        >>> from well_log_toolkit.visualization import Template
        >>> template = Template("custom")
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{"name": "GR", "color": "green"}]
        ... )
        >>> view = well.WellView(depth_range=[2800, 3000], template=template)
        >>> view.save("well_log.png", dpi=300)
        >>>
        >>> # Customize header spacing
        >>> view = well.WellView(
        ...     depth_range=[2800, 3000],
        ...     template=template,
        ...     header_config={"header_log_spacing": 0.04}
        ... )
        """
        from .visualization import WellView as WellViewClass

        return WellViewClass(
            well=self,
            depth_range=depth_range,
            tops=tops,
            template=template,
            figsize=figsize,
            dpi=dpi,
            header_config=header_config
        )

    def Crossplot(
        self,
        x: str,
        y: str,
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
    ) -> 'Crossplot':
        """
        Create a beautiful crossplot for this well.

        Parameters
        ----------
        x : str
            Name of property for x-axis
        y : str
            Name of property for y-axis
        shape : str, optional
            Property name for shape mapping.
            Default: None (single shape)
        color : str, optional
            Property name for color mapping. Use "depth" to color by depth.
            Default: None (single color)
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
            Marker style. Default: "o"
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
            Show legend when using shape mapping. Default: True

        Returns
        -------
        Crossplot
            Crossplot visualization object

        Examples
        --------
        Basic crossplot:

        >>> plot = well.Crossplot(x="RHOB", y="NPHI")
        >>> plot.show()

        With color and size mapping:

        >>> plot = well.Crossplot(
        ...     x="PHIE_2025",
        ...     y="NetSand_2025",
        ...     color="depth",
        ...     size="Sw_2025",
        ...     colortemplate="viridis",
        ...     color_range=[2000, 2500],
        ...     title="Porosity vs Net Sand"
        ... )
        >>> plot.show()

        With regression analysis:

        >>> plot = well.Crossplot(x="RHOB", y="NPHI")
        >>> plot.add_regression("linear")
        >>> plot.add_regression("polynomial", degree=2, line_color="blue")
        >>> plot.show()
        >>> print(plot.regressions["linear"].equation())

        With logarithmic scales:

        >>> plot = well.Crossplot(x="PERM", y="PHIE", x_log=True)
        >>> plot.show()
        """
        from .visualization import Crossplot as CrossplotClass

        return CrossplotClass(
            wells=self,
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
        )

    def __repr__(self) -> str:
        """String representation."""
        # Count total unique properties across all sources
        all_properties = set()
        for source_data in self._sources.values():
            all_properties.update(source_data['properties'].keys())

        return (
            f"Well('{self.name}', "
            f"sources={len(self._sources)}, "
            f"properties={len(all_properties)})"
        )