"""
Global orchestrator for multi-well analysis.
"""
from pathlib import Path
from typing import Optional, Union
import warnings

import pandas as pd

from .exceptions import LasFileError, PropertyNotFoundError
from .las_file import LasFile
from .well import Well
from .property import Property
from .utils import sanitize_well_name, sanitize_property_name


class _ManagerPropertyProxy:
    """
    Proxy object for manager-level property operations.

    This proxy enables broadcasting property operations across all wells:
        manager.PHIE_scaled = manager.PHIE * 0.01

    The proxy is created when accessing a property name on the manager,
    and operations on the proxy create new proxies that remember the operation.
    When assigned to a manager attribute, the operation is broadcast to all wells.
    """

    def __init__(self, manager: 'WellDataManager', property_name: str, operation=None):
        self._manager = manager
        self._property_name = property_name
        self._operation = operation  # Function to apply to each property

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
        return _ManagerPropertyProxy(self._manager, self._property_name, operation)

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
        as LAS files with well name prefix. Also renames LAS files for any sources
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

    def load(self, path: Union[str, Path]) -> 'WellDataManager':
        """
        Load all wells from a project folder structure.

        Automatically discovers and loads all LAS files from well folders
        (well_* format). Stores the project path for subsequent save() calls.
        Clears any existing wells before loading.

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
        >>> # ... make changes ...
        >>> manager.save()  # Saves back to "my_project"

        >>> # Load clears existing data
        >>> manager.load("other_project")  # Replaces current wells
        """
        base_path = Path(path)

        if not base_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {path}")

        # Clear existing wells before loading new project
        self._wells.clear()
        self._name_mapping.clear()

        # Store project path for save()
        self._project_path = base_path

        # Find all well folders (well_*)
        well_folders = sorted(base_path.glob("well_*"))

        if not well_folders:
            # Try loading all LAS files directly if no well folders
            las_files = list(base_path.glob("*.las"))
            if las_files:
                for las_file in las_files:
                    self.load_las(las_file)
            return self

        # Load from well folders
        for well_folder in well_folders:
            if well_folder.is_dir():
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
    
    def __repr__(self) -> str:
        """String representation."""
        return f"WellDataManager(wells={len(self._wells)})"