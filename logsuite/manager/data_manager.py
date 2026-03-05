"""
WellDataManager — global orchestrator for multi-well analysis.
"""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ..core.well import Well
from ..exceptions import LasFileError
from ..io import LasFile
from ..utils import sanitize_property_name, sanitize_well_name, suggest_similar_names
from .proxy import _ManagerMultiPropertyProxy, _ManagerPropertyProxy

if TYPE_CHECKING:
    from ..visualization import Template


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

    def __init__(self, project: str | Path | None = None):
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
        self._project_path: Path | None = None  # Track project path for save()
        self._templates: dict[str, Template] = {}  # {template_name: Template}

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
        if name.startswith("_"):
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
        if name.startswith("well_"):
            if name in self._wells:
                return self._wells[name]
            available = list(self._wells.keys())
            suggestions = suggest_similar_names(name, available)
            msg = f"Well '{name}' not found in manager."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            msg += f" Available wells: {', '.join(available) or 'none'}"
            raise AttributeError(msg)

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
        filepath: str | Path | list[str | Path],
        path: str | Path | None = None,
        sampled: bool = False,
        combine: str | None = None,
        source_name: str | None = None,
        silent: bool = False,
    ) -> "WellDataManager":
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

        See Also
        --------
        save : Save loaded wells to project directory.
        load : Load a previously saved project.
        load_properties : Load properties from a DataFrame.
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
                            name=well_name, sanitized_name=sanitized_name, parent_manager=self
                        )
                        self._name_mapping[well_name] = well_key

                    # Load files into well with combine
                    self._wells[well_key].load_las(
                        files_for_well,
                        path=None,  # Path already prepended
                        sampled=sampled,
                        combine=combine,
                        source_name=source_name,
                    )

                    # Track what was loaded
                    actual_source_name = source_name if source_name else f"combined_{combine}"
                    loaded_sources.append((well_name, actual_source_name, len(files_for_well)))

                # Print debug output
                if not silent:
                    print("Loaded sources:")
                    for well_name, src_name, file_count in loaded_sources:
                        print(
                            f"  - Well {well_name}: {src_name} ({file_count} file{'s' if file_count > 1 else ''} combined)"
                        )

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
                name=well_name, sanitized_name=sanitized_name, parent_manager=self
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
        well_col: str | None = "Well identifier (Well name)",
        well_name: str | None = None,
        discrete_col: str = "Surface",
        depth_col: str = "MD",
        x_col: str | None = "X",
        y_col: str | None = "Y",
        z_col: str | None = "Z",
        include_coordinates: bool = False,
    ) -> "WellDataManager":
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
                    name=well_name, sanitized_name=sanitized_name, parent_manager=self
                )
                self._name_mapping[well_name] = well_key

            well = self._wells[well_key]

            # Build DataFrame for this well
            well_data = {
                "DEPT": well_df[depth_col].values,
                property_name: well_df[discrete_col].map(value_to_code).values,
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
            unit_mappings = {"DEPT": "m", property_name: ""}
            if include_coordinates:
                if x_col and x_col in well_df.columns:
                    unit_mappings[x_col] = "m"
                if y_col and y_col in well_df.columns:
                    unit_mappings[y_col] = "m"
                if z_col and z_col in well_df.columns:
                    unit_mappings[z_col] = "m"

            # Build type mappings (discrete property, coordinates are continuous)
            type_mappings = {property_name: "discrete"}
            if include_coordinates:
                if x_col and x_col in well_df.columns:
                    type_mappings[x_col] = "continuous"
                if y_col and y_col in well_df.columns:
                    type_mappings[y_col] = "continuous"
                if z_col and z_col in well_df.columns:
                    type_mappings[z_col] = "continuous"

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
                label_mappings={property_name: code_to_value},
            )

            # Load it
            well.load_las(las)

        return self

    def load_properties(
        self,
        df: pd.DataFrame,
        source_name: str = "external_df",
        well_col: str | None = "Well",
        well_name: str | None = None,
        depth_col: str = "DEPT",
        unit_mappings: dict[str, str] | None = None,
        type_mappings: dict[str, str] | None = None,
        label_mappings: dict[str, dict[int, str]] | None = None,
        resample_method: str | None = None,
    ) -> "WellDataManager":
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
                    name=well_name, sanitized_name=sanitized_name, parent_manager=self
                )
                self._name_mapping[well_name] = well_key

            well = self._wells[well_key]

            # Build DataFrame for this well (rename depth column to DEPT)
            well_data = {"DEPT": well_df[depth_col].values}
            for prop_col in prop_cols:
                well_data[prop_col] = well_df[prop_col].values

            props_df = pd.DataFrame(well_data)

            # Build unit mappings (include DEPT)
            full_unit_mappings = {"DEPT": unit_mappings.get(depth_col, "m")}
            for prop_col in prop_cols:
                full_unit_mappings[prop_col] = unit_mappings.get(prop_col, "")

            # Build type mappings
            full_type_mappings = {}
            for prop_col in prop_cols:
                full_type_mappings[prop_col] = type_mappings.get(prop_col, "continuous")

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
                label_mappings=label_mappings,
            )

            # Check compatibility if well already has data
            if well._sources:
                # Get an existing LAS file to check compatibility
                existing_source = list(well._sources.values())[0]
                existing_las = existing_source["las_file"]
                compatibility = las.check_depth_compatibility(existing_las)

                if not compatibility["compatible"]:
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
                            UserWarning,
                            stacklevel=2,
                        )

            # Load it (with resampling if specified)
            well.load_las(las, resample_method=resample_method)

            print(
                f"✓ Loaded {len(prop_cols)} properties into well '{well.name}' from source '{base_source_name}'"
            )

        return self

    def save(self, path: str | Path | None = None) -> None:
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
            if hasattr(well, "_saved_filter_intervals") and well._saved_filter_intervals:
                import json

                intervals_file = well_folder / "intervals.json"
                with open(intervals_file, "w") as f:
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

    def load(self, path: str | Path) -> "WellDataManager":
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
            from ..visualization import Template

            template_files = sorted(templates_folder.glob("*.json"))
            for template_file in template_files:
                try:
                    template = Template.load(template_file)
                    # Use filename (without extension) as template name
                    template_name = template_file.stem
                    self._templates[template_name] = template
                except Exception as e:
                    warnings.warn(
                        f"Could not load template {template_file.name}: {e}", stacklevel=2
                    )

        # Find all well folders (well_*) - skip templates folder
        well_folders = sorted(
            [
                folder
                for folder in base_path.glob("well_*")
                if folder.is_dir() and folder.name != "templates"
            ]
        )

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
                    with open(intervals_file) as f:
                        saved_intervals = json.load(f)
                    # Find the well for this folder and set its intervals
                    well_key = well_folder.name  # e.g., "well_35_9_16_A"
                    if well_key in self._wells:
                        self._wells[well_key]._saved_filter_intervals = saved_intervals
                except Exception as e:
                    warnings.warn(
                        f"Could not load intervals from {intervals_file}: {e}", stacklevel=2
                    )

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
                name=well_name, sanitized_name=sanitized_name, parent_manager=self
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
        if not name.startswith("well_"):
            well_key = f"well_{name}"
            if well_key in self._wells:
                return self._wells[well_key]

        # Try as original name
        sanitized = sanitize_well_name(name)
        well_key = f"well_{sanitized}"
        if well_key in self._wells:
            return self._wells[well_key]

        # Not found
        available = ", ".join(self._wells.keys())
        raise KeyError(f"Well '{name}' not found. " f"Available wells: {available or 'none'}")

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

    def add_template(self, template: "Template") -> None:
        """
        Store a template using its built-in name.

        Parameters
        ----------
        template : Template
            Template object (uses template.name as the key)

        Examples
        --------
        >>> from logsuite import Template
        >>>
        >>> template = Template("reservoir")
        >>> template.add_track(track_type="continuous", logs=[...])
        >>> manager.add_template(template)  # Stored as "reservoir"
        >>>
        >>> # Use in WellView
        >>> view = well.WellView(template="reservoir")
        """
        from ..visualization import Template

        if not isinstance(template, Template):
            raise TypeError(f"template must be Template, got {type(template).__name__}")

        self._templates[template.name] = template

    def set_template(self, name: str, template: Union["Template", dict]) -> None:
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
        from ..visualization import Template

        if isinstance(template, dict):
            template = Template.from_dict(template)
        elif not isinstance(template, Template):
            raise TypeError(f"template must be Template or dict, got {type(template).__name__}")

        self._templates[name] = template

    def get_template(self, name: str) -> "Template":
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
            available = ", ".join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. " f"Available templates: {available or 'none'}"
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
            available = ", ".join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. " f"Available templates: {available or 'none'}"
            )

    def Crossplot(
        self,
        x: str | None = None,
        y: str | None = None,
        wells: list[str] | None = None,
        layers: dict[str, list[str]] | None = None,
        shape: str | None = None,
        color: str | None = None,
        size: str | None = None,
        colortemplate: str = "viridis",
        color_range: tuple[float, float] | None = None,
        size_range: tuple[float, float] = (20, 200),
        title: str = "Cross Plot",
        xlabel: str | None = None,
        ylabel: str | None = None,
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
        depth_range: tuple[float, float] | None = None,
        show_colorbar: bool = True,
        show_legend: bool = True,
        show_regression_legend: bool = True,
        show_regression_equation: bool = True,
        show_regression_r2: bool = True,
        regression: str | dict | None = None,
        regression_by_color: str | dict | None = None,
        regression_by_group: str | dict | None = None,
    ) -> "Crossplot":
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
        from ..visualization import Crossplot as CrossplotClass

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

    def validate(self) -> dict[str, list[str]]:
        """
        Check data integrity across all wells.

        Returns a dictionary mapping well names to lists of issue descriptions.
        An empty dict means no issues were found.

        Returns
        -------
        dict[str, list[str]]
            Well names mapped to lists of issue strings. Empty if all OK.

        Examples
        --------
        >>> issues = manager.validate()
        >>> if issues:
        ...     for well, problems in issues.items():
        ...         print(f"{well}: {problems}")
        """
        issues: dict[str, list[str]] = {}

        # Collect all property names across wells for cross-comparison
        all_property_names: set[str] = set()
        for well in self._wells.values():
            all_property_names.update(well.properties)

        for well_name, well in self._wells.items():
            well_issues: list[str] = []
            well_props = set(well.properties)

            # Check for missing properties compared to other wells
            missing = all_property_names - well_props
            if missing:
                well_issues.append(
                    f"Missing properties present in other wells: {', '.join(sorted(missing))}"
                )

            # Check each property for depth issues
            for source_data in well._sources.values():
                for prop_name, prop in source_data["properties"].items():
                    depth = prop.depth
                    values = prop.values

                    # Depth/values length mismatch
                    if len(depth) != len(values):
                        well_issues.append(
                            f"Property '{prop_name}': depth length ({len(depth)}) != "
                            f"values length ({len(values)})"
                        )

                    # Depth monotonicity
                    if len(depth) > 1:
                        diffs = np.diff(depth)
                        non_increasing = np.sum(diffs <= 0)
                        if non_increasing > 0:
                            well_issues.append(
                                f"Property '{prop_name}': depth not monotonically increasing "
                                f"({non_increasing} violation(s))"
                            )

            if well_issues:
                issues[well_name] = well_issues

        return issues

    def __repr__(self) -> str:
        """String representation."""
        return f"WellDataManager(wells={len(self._wells)})"
