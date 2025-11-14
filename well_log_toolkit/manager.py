"""
Global orchestrator for multi-well analysis.
"""
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .exceptions import LasFileError
from .las_file import LasFile
from .well import Well
from .utils import sanitize_well_name, sanitize_property_name


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
    """
    
    def __init__(self):
        self._wells: dict[str, Well] = {}  # {sanitized_name: Well}
        self._name_mapping: dict[str, str] = {}  # {original_name: sanitized_name}
    
    def load_las(self, filepath: Union[str, Path, list[Union[str, Path]]]) -> 'WellDataManager':
        """
        Load LAS file(s), auto-create well if needed.

        Parameters
        ----------
        filepath : Union[str, Path, list[Union[str, Path]]]
            Path to LAS file or list of paths to LAS files

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
        >>> well = manager.well_12_3_2_B
        """
        # Handle list of files
        if isinstance(filepath, list):
            for file in filepath:
                self.load_las(file)
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

        if sanitized_name not in self._wells:
            # Create new well
            self._wells[sanitized_name] = Well(
                name=well_name,
                sanitized_name=sanitized_name,
                parent_manager=self
            )
            self._name_mapping[well_name] = sanitized_name

        # Load into well
        self._wells[sanitized_name].load_las(las)

        return self  # Enable chaining

    def load_tops(
        self,
        df: pd.DataFrame,
        source_name: str = "Well_Tops",
        well_col: str = "Well identifier (Well name)",
        surface_col: str = "Surface",
        depth_col: str = "MD",
        x_col: Optional[str] = "X",
        y_col: Optional[str] = "Y",
        z_col: Optional[str] = "Z",
        include_coordinates: bool = False
    ) -> 'WellDataManager':
        """
        Load formation tops data from a DataFrame into wells.

        Automatically creates wells if they don't exist, converts surface names
        to discrete integers with labels, and adds the data as a source to each well.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing tops data with columns for well name, surface, and depth
        source_name : str, default "Well_Tops"
            Name for this tops source (will be sanitized)
        well_col : str, default "Well identifier (Well name)"
            Column name containing well names
        surface_col : str, default "Surface"
            Column name containing formation/surface names
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
        >>> # Load from Excel
        >>> import pandas as pd
        >>> df = pd.read_excel("formation_tops.xlsx")
        >>> manager = WellDataManager()
        >>> manager.load_tops(df)
        >>>
        >>> # Access tops
        >>> well = manager.well_36_7_5_A
        >>> print(well.sources)  # ['Well_Tops']
        >>> well.Well_Tops.Surface  # Discrete property with formation names
        """
        # Validate required columns exist
        required_cols = [well_col, surface_col, depth_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Required columns missing from DataFrame: {', '.join(missing_cols)}. "
                f"Available columns: {', '.join(df.columns)}"
            )

        # Build global surface label mapping (consistent across all wells)
        unique_surfaces = sorted(df[surface_col].unique())
        surface_to_code = {surface: idx for idx, surface in enumerate(unique_surfaces)}
        code_to_surface = {idx: surface for surface, idx in surface_to_code.items()}

        # Group by well
        grouped = df.groupby(well_col)

        for well_name, well_df in grouped:
            # Get or create well
            sanitized_name = sanitize_well_name(well_name)
            if sanitized_name not in self._wells:
                self._wells[sanitized_name] = Well(
                    name=well_name,
                    sanitized_name=sanitized_name,
                    parent_manager=self
                )
                self._name_mapping[well_name] = sanitized_name

            well = self._wells[sanitized_name]

            # Build DataFrame for this well
            well_data = {
                'DEPT': well_df[depth_col].values,
                surface_col: well_df[surface_col].map(surface_to_code).values
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
            unit_mappings = {'DEPT': 'm', surface_col: ''}
            if include_coordinates:
                if x_col and x_col in well_df.columns:
                    unit_mappings[x_col] = 'm'
                if y_col and y_col in well_df.columns:
                    unit_mappings[y_col] = 'm'
                if z_col and z_col in well_df.columns:
                    unit_mappings[z_col] = 'm'

            # Build type mappings (surface is discrete, coordinates are continuous)
            type_mappings = {surface_col: 'discrete'}
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
                label_mappings={surface_col: code_to_surface}
            )

            # Load it
            well.load_las(las)

        return self

    def export_project(self, path: Union[str, Path]) -> None:
        """
        Export all wells and their sources to a project folder structure.

        Creates a folder for each well (well_xxx format) and exports all sources
        as LAS files with well name prefix.

        Parameters
        ----------
        path : Union[str, Path]
            Root directory path for the project

        Examples
        --------
        >>> manager = WellDataManager()
        >>> manager.load_las(["well1.las", "well2.las"])
        >>> manager.export_project("my_project")
        # Creates:
        # my_project/
        #   well_36_7_5_A/
        #     well_36_7_5_A_Log.las
        #     well_36_7_5_A_CorePor.las
        #   well_36_7_5_B/
        #     well_36_7_5_B_Log.las
        """
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        for well_name, well in self._wells.items():
            # Create well folder
            well_folder = base_path / well_name
            well_folder.mkdir(exist_ok=True)

            # Export each source
            well.export_sources(well_folder)

    def import_project(self, path: Union[str, Path]) -> 'WellDataManager':
        """
        Import all wells from a project folder structure.

        Automatically discovers and loads all LAS files from well folders
        (well_* format).

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
        >>> manager.import_project("my_project")
        >>> print(manager.wells)  # All wells from project
        """
        base_path = Path(path)

        if not base_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {path}")

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
        
        if sanitized_name not in self._wells:
            self._wells[sanitized_name] = Well(
                name=well_name,
                sanitized_name=sanitized_name,
                parent_manager=self
            )
            self._name_mapping[well_name] = sanitized_name
        
        return self._wells[sanitized_name]
    
    def __getattr__(self, name: str) -> Well:
        """
        Enable well access via attributes: manager.well_12_3_2_B
        
        This is called when normal attribute lookup fails.
        """
        # Don't intercept private attributes or methods
        if name.startswith('_') or name in [
            'wells', 'load_las', 'load_tops', 'add_well', 'get_well', 'remove_well'
        ]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        
        # Try to get well
        if name in self._wells:
            return self._wells[name]
        
        # Not found
        available = ', '.join(self._wells.keys())
        raise AttributeError(
            f"Well '{name}' not found in manager. "
            f"Available wells: {available or 'none'}"
        )
    
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
            Either original name ("12/3-2 B") or sanitized ("well_12_3_2_B")
        
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
        >>> well = manager.get_well("12/3-2 B")
        >>> well = manager.get_well("well_12_3_2_B")
        """
        # Try sanitized first
        if name in self._wells:
            return self._wells[name]
        
        # Try as original name
        sanitized = sanitize_well_name(name)
        if sanitized in self._wells:
            return self._wells[sanitized]
        
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
            Well name (original or sanitized)
        
        Examples
        --------
        >>> manager.remove_well("12/3-2 B")
        """
        # Find the well
        well = self.get_well(name)
        sanitized = well.sanitized_name
        
        # Remove from mappings
        del self._wells[sanitized]
        if well.name in self._name_mapping:
            del self._name_mapping[well.name]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"WellDataManager(wells={len(self._wells)})"