"""
Well class for managing log properties from a single well.
"""
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .exceptions import WellError, WellNameMismatchError, PropertyNotFoundError
from .property import Property
from .las_file import LasFile

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
    parent_manager : WellDataManager | None
        Parent manager
    properties : list[str]
        List of property names
    
    Examples
    --------
    >>> well = manager.well_12_3_2_B
    >>> well.load_las("log1.las").load_las("log2.las")
    >>> print(well.properties)
    ['PHIE', 'PERM', 'Zone', 'NTG_Flag']
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
        self._properties: dict[str, Property] = {}
        self._depth_grid: Optional[np.ndarray] = None
    
    def load_las(self, las: LasFile | str | Path) -> 'Well':
        """
        Load LAS file into this well.
        Validates well name matches.
        
        Parameters
        ----------
        las : LasFile | str | Path
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
        
        # Load data
        data = las.data
        depth_col = las.depth_column
        
        if depth_col is None:
            raise WellError(f"No depth column found in LAS file")
        
        depth_values = data[depth_col].values
        
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
            
            # Create property
            prop = Property(
                name=prop_name,
                depth=depth_values,
                values=values.values,
                parent_well=self,
                unit=curve_meta['unit'],
                prop_type=curve_meta['type'],
                description=curve_meta['description'],
                null_value=las.null_value
            )
            
            if prop_name in self._properties:
                # Merge with existing property
                self._merge_property(prop_name, prop)
            else:
                self._properties[prop_name] = prop
        
        return self  # Enable chaining
    
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
        """
        # Don't intercept private attributes, methods, or class attributes
        if name.startswith('_') or name in [
            'name', 'sanitized_name', 'parent_manager', 'properties',
            'load_las', 'get_property', 'resample', 'to_dataframe'
        ]:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        
        # Try to get from properties
        if name in self._properties:
            return self._properties[name]
        
        # Not found
        available = ', '.join(self._properties.keys())
        raise AttributeError(
            f"Well '{self.name}' has no property '{name}'. "
            f"Available properties: {available or 'none'}"
        )
    
    @property
    def properties(self) -> list[str]:
        """List of property names in this well."""
        return list(self._properties.keys())
    
    def get_property(self, name: str) -> Property:
        """
        Explicit property getter.
        
        Parameters
        ----------
        name : str
            Property name
        
        Returns
        -------
        Property
            The requested property
        
        Raises
        ------
        PropertyNotFoundError
            If property not found
        """
        if name not in self._properties:
            available = ', '.join(self._properties.keys())
            raise PropertyNotFoundError(
                f"Property '{name}' not found in well '{self.name}'. "
                f"Available: {available or 'none'}"
            )
        return self._properties[name]
    
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
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all properties as DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with DEPT and all properties
        
        Raises
        ------
        WellError
            If properties have different depth grids
        
        Examples
        --------
        >>> well.resample(depth_step=0.1)
        >>> df = well.to_dataframe()
        """
        if not self._properties:
            return pd.DataFrame()
        
        # Get first property for depth reference
        first_prop = next(iter(self._properties.values()))
        depth = first_prop.depth
        
        # Verify all properties on same grid
        for prop in self._properties.values():
            if not np.array_equal(prop.depth, depth):
                raise WellError(
                    f"Cannot export to DataFrame: properties have different depth grids. "
                    f"Call well.resample() first to align all properties."
                )
        
        # Build DataFrame
        data = {'DEPT': depth}
        for name, prop in self._properties.items():
            data[name] = prop.values
        
        return pd.DataFrame(data)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Well('{self.name}', "
            f"properties={len(self._properties)})"
        )