"""
Global orchestrator for multi-well analysis.
"""
from pathlib import Path
from typing import Optional

from .exceptions import LasFileError
from .las_file import LasFile
from .well import Well
from .utils import sanitize_well_name


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
    
    def load_las(self, filepath: str | Path) -> 'WellDataManager':
        """
        Load LAS file, auto-create well if needed.
        
        Parameters
        ----------
        filepath : str | Path
            Path to LAS file
        
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
        >>> manager.load_las("well1.las").load_las("well2.las")
        >>> well = manager.well_12_3_2_B
        """
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
            'wells', 'load_las', 'add_well', 'get_well', 'remove_well'
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