"""
LAS file reader with lazy data loading.
"""
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .exceptions import LasFileError, UnsupportedVersionError
from .utils import parse_las_line


class LasFile:
    """
    Fast LAS file reader with lazy data loading.
    
    Workflow:
    1. Instantiate and parse headers (fast)
    2. Inspect metadata (well name, curves, units)
    3. Update curve metadata if needed
    4. Load data when ready
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to LAS file
    
    Attributes
    ----------
    filepath : Path
        Path to LAS file
    version_info : dict
        Version section data (VERS, WRAP)
    well_info : dict
        Well section data (WELL, STRT, STOP, NULL, etc.)
    parameter_info : dict
        Parameter section data
    curves : dict
        Curve metadata: {name: {unit, description, type, alias, multiplier}}
    
    Examples
    --------
    >>> las = LasFile("well.las")
    >>> print(las.well_name)
    '36/7-5 B'
    >>> print(las.curves.keys())
    dict_keys(['DEPT', 'PHIE_2025', 'PERM_Lam_2025', ...])
    >>> las.update_curve('PHIE_2025', type='continuous', alias='PHIE')
    >>> df = las.data  # Lazy load
    """
    
    # Supported LAS versions
    SUPPORTED_VERSIONS = {'2.0', '2'}

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise LasFileError(f"File not found: {filepath}")
        
        # Metadata containers
        self.version_info: dict[str, str] = {}
        self.well_info: dict[str, str] = {}
        self.parameter_info: dict[str, str] = {}
        self.curves: dict[str, dict] = {}
        
        # Data management
        self._data: Optional[pd.DataFrame] = None
        self._ascii_start_line: Optional[int] = None
        self._curve_names: list[str] = []  # Preserve original order
        self._file_lines: Optional[list[str]] = None
        
        # Auto-parse headers on init
        self._parse_headers()
        self._validate_version()
    
    @property
    def well_name(self) -> Optional[str]:
        """Extract well name from well info."""
        return self.well_info.get('WELL')
    
    @property
    def depth_column(self) -> Optional[str]:
        """First curve (typically DEPT/DEPTH)."""
        return self._curve_names[0] if self._curve_names else None
    
    @property
    def null_value(self) -> float:
        """NULL value from well section, default -999.25."""
        null_str = self.well_info.get('NULL', '-999.25')
        try:
            return float(null_str)
        except ValueError:
            return -999.25
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Lazy-load and return data.
        
        Returns
        -------
        pd.DataFrame
            Well log data with curves as columns
        """
        if self._data is None:
            self._load_data()
        return self._data
    
    def update_curve(self, name: str, **kwargs) -> None:
        """
        Update curve metadata.
        
        Parameters
        ----------
        name : str
            Curve name to update
        **kwargs
            unit : str - Unit string
            description : str - Description
            type : {'continuous', 'discrete'} - Log type
            alias : str | None - Output column name
            multiplier : float | None - Unit conversion factor
        
        Raises
        ------
        KeyError
            If curve not found
        ValueError
            If invalid attribute or type value
        
        Examples
        --------
        >>> las.update_curve('PHIE_2025', type='continuous', alias='PHIE')
        >>> las.update_curve('ResFlag_2025', type='discrete', alias='ResFlag')
        >>> las.update_curve('PERM_Lam_2025', multiplier=0.001, alias='PERM_D')
        """
        if name not in self.curves:
            available = ', '.join(self.curves.keys())
            raise KeyError(
                f"Curve '{name}' not found in LAS file. "
                f"Available curves: {available}"
            )
        
        valid_attrs = {'unit', 'description', 'type', 'alias', 'multiplier'}
        invalid = set(kwargs.keys()) - valid_attrs
        if invalid:
            raise ValueError(
                f"Invalid curve attributes: {', '.join(invalid)}. "
                f"Valid attributes: {', '.join(valid_attrs)}"
            )
        
        # Validate type if provided
        if 'type' in kwargs:
            if kwargs['type'] not in {'continuous', 'discrete'}:
                raise ValueError(
                    f"type must be 'continuous' or 'discrete', "
                    f"got '{kwargs['type']}'"
                )
        
        # Update the curve metadata
        self.curves[name].update(kwargs)
    
    def bulk_update_curves(self, updates: dict[str, dict]) -> None:
        """
        Update multiple curves at once.
        
        Parameters
        ----------
        updates : dict[str, dict]
            {curve_name: {attr: value, ...}, ...}
        
        Examples
        --------
        >>> las.bulk_update_curves({
        ...     'Cerisa_facies_LF': {'type': 'discrete', 'alias': 'Facies'},
        ...     'PHIE_2025': {'alias': 'PHIE'},
        ...     'PERM_Lam_2025': {'alias': 'PERM', 'multiplier': 0.001}
        ... })
        """
        for curve_name, attrs in updates.items():
            self.update_curve(curve_name, **attrs)
    
    def _parse_headers(self) -> None:
        """Parse LAS file headers (version, well, curve, parameter sections)."""
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        self._file_lines = lines
        
        current_section = None
        i = 0
        n = len(lines)
        
        while i < n:
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check for section headers
            if line.startswith('~'):
                section_name = line[1:].split()[0].lower() if len(line) > 1 else ''
                
                if section_name == 'version':
                    current_section = 'version'
                elif section_name == 'well':
                    current_section = 'well'
                elif section_name.startswith('curv'):  # curve or curves
                    current_section = 'curve'
                elif section_name.startswith('param'):  # parameter or parameters
                    current_section = 'parameter'
                elif section_name.startswith('ascii') or section_name == 'a':
                    self._ascii_start_line = i + 1
                    break  # Stop parsing, data section found
                
                i += 1
                continue
            
            # Parse section content
            if current_section == 'version':
                mnemonic, value, _ = parse_las_line(line)
                if mnemonic:
                    self.version_info[mnemonic] = value
            
            elif current_section == 'well':
                mnemonic, value, _ = parse_las_line(line)
                if mnemonic:
                    self.well_info[mnemonic] = value
            
            elif current_section == 'curve':
                mnemonic, unit, description = parse_las_line(line)
                if mnemonic:
                    self._curve_names.append(mnemonic)
                    self.curves[mnemonic] = {
                        'unit': unit,
                        'description': description,
                        'type': 'continuous',  # Default
                        'alias': None,  # Default (use original name)
                        'multiplier': None  # Default (no conversion)
                    }
            
            elif current_section == 'parameter':
                mnemonic, value, description = parse_las_line(line)
                if mnemonic:
                    self.parameter_info[mnemonic] = value
            
            i += 1
        
        if self._ascii_start_line is None:
            raise LasFileError(
                f"~Ascii section not found in {self.filepath}. "
                "Not a valid LAS file."
            )
        
        if not self._curve_names:
            raise LasFileError(
                f"No curves found in {self.filepath}. "
                "~Curve section may be missing or empty."
            )
    
    def _validate_version(self) -> None:
        """Ensure LAS version is supported."""
        version = self.version_info.get('VERS', '').strip()
        
        if not version:
            raise UnsupportedVersionError(
                f"No version information found in {self.filepath}"
            )
        
        if version not in self.SUPPORTED_VERSIONS:
            raise UnsupportedVersionError(
                f"Unsupported LAS version: {version}. "
                f"Supported versions: {', '.join(self.SUPPORTED_VERSIONS)}"
            )
        
        wrap = self.version_info.get('WRAP', 'NO').strip().upper()
        if wrap != 'NO':
            raise UnsupportedVersionError(
                f"Wrapped LAS files are not supported (WRAP={wrap}). "
                "Only WRAP=NO is supported."
            )
    
    def _load_data(self) -> None:
        """Load ASCII data section into pandas DataFrame."""
        if self._file_lines is None or self._ascii_start_line is None:
            raise LasFileError("Headers not parsed. Cannot load data.")
        
        # Read ASCII section using pandas
        ascii_data = "".join(self._file_lines[self._ascii_start_line:])
        
        try:
            df = pd.read_csv(
                io.StringIO(ascii_data),
                sep=r'\s+',
                names=self._curve_names,
                na_values=[self.null_value],
                engine='c',
                dtype_backend='numpy_nullable'
            )
        except Exception as e:
            raise LasFileError(
                f"Failed to parse ASCII data in {self.filepath}: {e}"
            )
        
        # Apply multipliers and aliases
        for curve_name in self._curve_names:
            curve_meta = self.curves[curve_name]
            
            # Apply multiplier if specified
            if curve_meta['multiplier'] is not None:
                df[curve_name] = df[curve_name] * curve_meta['multiplier']
            
            # Apply alias if specified
            if curve_meta['alias'] is not None:
                df = df.rename(columns={curve_name: curve_meta['alias']})
        
        self._data = df
        
        # Clear file lines to free memory
        self._file_lines = None
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LasFile('{self.filepath.name}', "
            f"well='{self.well_name}', "
            f"curves={len(self.curves)})"
        )