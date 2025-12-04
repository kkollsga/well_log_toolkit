"""
LAS file reader with lazy data loading.
"""
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .exceptions import LasFileError, UnsupportedVersionError
from .utils import parse_las_line, filter_names


class LasFile:
    """
    Fast LAS file reader with lazy data loading.

    Workflow:
    1. Instantiate and parse headers (fast)
    2. Inspect metadata (well name, curves, units)
    3. Update curve metadata if needed
    4. Load data when ready

    When loading LAS files created by this toolkit, discrete properties
    and their label mappings are automatically detected and loaded.

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
        Parameter section data (includes discrete property markers and labels)
    curves : dict
        Curve metadata: {name: {unit, description, type, alias, multiplier}}

    Properties
    ----------
    discrete_properties : list[str]
        List of properties marked as discrete in ~Parameter section

    Methods
    -------
    get_discrete_labels(property_name: str) -> dict[int, str] | None
        Extract label mappings for a discrete property

    Examples
    --------
    >>> las = LasFile("well.las")
    >>> print(las.well_name)
    '12/3-2 B'
    >>> print(las.curves.keys())
    dict_keys(['DEPT', 'PHIE_2025', 'PERM_Lam_2025', ...])
    >>> las.update_curve('PHIE_2025', type='continuous', alias='PHIE')
    >>> df = las.data()  # Lazy load

    >>> # Check for discrete properties
    >>> print(las.discrete_properties)
    ['Zone', 'NTG_Flag']
    >>> labels = las.get_discrete_labels('Zone')
    >>> print(labels)
    {0: 'NonReservoir', 1: 'Reservoir'}
    """
    
    # Supported LAS versions
    SUPPORTED_VERSIONS = {'2.0', '2'}

    def __init__(self, filepath: Union[str, Path], _from_dataframe: bool = False):
        self.filepath = Path(filepath)

        if not _from_dataframe and not self.filepath.exists():
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

        # Auto-parse headers on init (skip if from DataFrame)
        if not _from_dataframe:
            self._parse_headers()
            self._validate_version()
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        well_name: str,
        source_name: str = 'external_df',
        unit_mappings: Optional[dict[str, str]] = None,
        type_mappings: Optional[dict[str, str]] = None,
        label_mappings: Optional[dict[str, dict[int, str]]] = None
    ) -> 'LasFile':
        """
        Create a LasFile object from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with DEPT column and property columns
        well_name : str
            Well name for this data
        source_name : str, default 'external_df'
            Name for this data source (e.g., 'external_df', 'external_df1')
        unit_mappings : dict[str, str], optional
            Mapping of column names to units
        type_mappings : dict[str, str], optional
            Mapping of column names to 'continuous' or 'discrete'
        label_mappings : dict[str, dict[int, str]], optional
            Label mappings for discrete properties

        Returns
        -------
        LasFile
            LasFile object populated with DataFrame data

        Examples
        --------
        >>> df = pd.DataFrame({'DEPT': [2800, 2801], 'PHIE': [0.2, 0.22]})
        >>> las = LasFile.from_dataframe(
        ...     df,
        ...     well_name='12/3-2 B',
        ...     source_name='external_df',
        ...     unit_mappings={'DEPT': 'm', 'PHIE': 'v/v'}
        ... )
        """
        if 'DEPT' not in df.columns:
            raise LasFileError(
                "DataFrame must contain 'DEPT' column. "
                f"Available columns: {', '.join(df.columns)}"
            )

        unit_mappings = unit_mappings or {}
        type_mappings = type_mappings or {}
        label_mappings = label_mappings or {}

        # Create instance without parsing file
        instance = cls(source_name, _from_dataframe=True)

        # Set version info
        instance.version_info = {'VERS': '2.0', 'WRAP': 'NO'}

        # Set well info
        dept = df['DEPT'].dropna()
        instance.well_info = {
            'WELL': well_name,
            'STRT': str(float(dept.min())) if len(dept) > 0 else '0.0',
            'STOP': str(float(dept.max())) if len(dept) > 0 else '0.0',
            'STEP': str(float(dept.diff().median())) if len(dept) > 1 else '0.1',
            'NULL': '-999.25'
        }

        # Set curve metadata
        for col in df.columns:
            instance._curve_names.append(col)
            instance.curves[col] = {
                'unit': unit_mappings.get(col, ''),
                'description': col,
                'type': type_mappings.get(col, 'continuous'),
                'alias': None,
                'multiplier': None
            }

        # Set parameter info (discrete labels)
        if label_mappings:
            discrete_props = ','.join(sorted(label_mappings.keys()))
            instance.parameter_info['DISCRETE_PROPS'] = discrete_props

            for prop_name, labels in label_mappings.items():
                for value, label in labels.items():
                    param_name = f"{prop_name}_{value}"
                    instance.parameter_info[param_name] = label

        # Set data directly
        instance._data = df.copy()

        return instance

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
    def discrete_properties(self) -> list[str]:
        """
        List of properties marked as discrete in ~Parameter section.

        Returns
        -------
        list[str]
            List of discrete property names from DISCRETE_PROPS parameter
        """
        discrete_props_str = self.parameter_info.get('DISCRETE_PROPS', '')
        if not discrete_props_str:
            return []
        # Split by comma and strip whitespace
        return [name.strip() for name in discrete_props_str.split(',') if name.strip()]

    def get_discrete_labels(self, property_name: str) -> Optional[dict[int, str]]:
        """
        Extract label mappings for a discrete property from ~Parameter section.

        Parameters
        ----------
        property_name : str
            Name of the discrete property

        Returns
        -------
        dict[int, str] | None
            Label mapping {0: 'Label0', 1: 'Label1'} or None if no labels found

        Examples
        --------
        >>> las = LasFile("well.las")
        >>> labels = las.get_discrete_labels('Zone')
        >>> # Returns: {0: 'NonReservoir', 1: 'Reservoir'}

        Notes
        -----
        If a label contains a color specification (e.g., "NonNet|red"), only the
        label part is returned. Use get_discrete_colors() to retrieve colors.
        """
        labels = {}
        prefix = f"{property_name}_"

        # Look for parameters like "Zone_0", "Zone_1", etc.
        for param_name, label_value in self.parameter_info.items():
            if param_name.startswith(prefix):
                # Extract the numeric suffix
                suffix = param_name[len(prefix):]
                try:
                    value = int(suffix)
                    # If the label contains a color (e.g., "Label|color"), extract just the label
                    if '|' in label_value:
                        label_only = label_value.split('|')[0].strip()
                        labels[value] = label_only
                    else:
                        labels[value] = label_value.strip()
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        return labels if labels else None

    def get_discrete_colors(self, property_name: str) -> Optional[dict[int, str]]:
        """
        Extract color mappings for a discrete property from ~Parameter section.

        Supports two formats:
        1. Inline: Zone_0 = "NonNet|red" (color after pipe separator)
        2. Separate: Zone_COLOR_0 = "red" (for backward compatibility)

        Inline format takes precedence if both exist.

        Parameters
        ----------
        property_name : str
            Name of the discrete property

        Returns
        -------
        dict[int, str] | None
            Color mapping {0: 'red', 1: 'green'} or None if no colors found

        Examples
        --------
        >>> las = LasFile("well.las")
        >>> colors = las.get_discrete_colors('Zone')
        >>> # Returns: {0: 'red', 1: 'green'}
        """
        colors = {}
        prefix = f"{property_name}_"
        color_prefix = f"{property_name}_COLOR_"

        # First, check for inline colors (e.g., "Label|color")
        for param_name, label_value in self.parameter_info.items():
            if param_name.startswith(prefix) and not param_name.startswith(color_prefix):
                # Extract the numeric suffix
                suffix = param_name[len(prefix):]
                try:
                    value = int(suffix)
                    # If the label contains a color (e.g., "Label|color"), extract the color
                    if '|' in label_value:
                        parts = label_value.split('|')
                        if len(parts) >= 2:
                            color = parts[1].strip()
                            if color:  # Only add if color is not empty
                                colors[value] = color
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        # Second, check for separate color parameters (backward compatibility)
        # Only add if not already defined by inline format
        for param_name, color_value in self.parameter_info.items():
            if param_name.startswith(color_prefix):
                # Extract the numeric suffix
                suffix = param_name[len(color_prefix):]
                try:
                    value = int(suffix)
                    if value not in colors:  # Inline format takes precedence
                        colors[value] = color_value.strip()
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        return colors if colors else None

    def get_discrete_styles(self, property_name: str) -> Optional[dict[int, str]]:
        """
        Extract line style mappings for a discrete property from ~Parameter section.

        Supports two formats:
        1. Inline: Zone_0 = "NonNet|red|dashed" (style after second pipe separator)
        2. Separate: Zone_STYLE_0 = "dashed" (for backward compatibility)

        Inline format takes precedence if both exist.

        Parameters
        ----------
        property_name : str
            Name of the discrete property

        Returns
        -------
        dict[int, str] | None
            Style mapping {0: 'solid', 1: 'dashed'} or None if no styles found

        Examples
        --------
        >>> las = LasFile("well.las")
        >>> styles = las.get_discrete_styles('Zone')
        >>> # Returns: {0: 'solid', 1: 'dashed'}
        """
        styles = {}
        prefix = f"{property_name}_"
        style_prefix = f"{property_name}_STYLE_"

        # First, check for inline styles (e.g., "Label|color|style")
        for param_name, label_value in self.parameter_info.items():
            if param_name.startswith(prefix) and not param_name.startswith(style_prefix):
                # Extract the numeric suffix
                suffix = param_name[len(prefix):]
                try:
                    value = int(suffix)
                    # If the label contains a style (e.g., "Label|color|style"), extract the style
                    if '|' in label_value:
                        parts = label_value.split('|')
                        if len(parts) >= 3:
                            style = parts[2].strip()
                            if style:  # Only add if style is not empty
                                styles[value] = style
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        # Second, check for separate style parameters (backward compatibility)
        # Only add if not already defined by inline format
        for param_name, style_value in self.parameter_info.items():
            if param_name.startswith(style_prefix):
                # Extract the numeric suffix
                suffix = param_name[len(style_prefix):]
                try:
                    value = int(suffix)
                    if value not in styles:  # Inline format takes precedence
                        styles[value] = style_value.strip()
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        return styles if styles else None

    def get_discrete_thicknesses(self, property_name: str) -> Optional[dict[int, float]]:
        """
        Extract line thickness mappings for a discrete property from ~Parameter section.

        Supports two formats:
        1. Inline: Zone_0 = "NonNet|red|dashed|1.5" (thickness after third pipe separator)
        2. Separate: Zone_THICKNESS_0 = "1.5" (for backward compatibility)

        Inline format takes precedence if both exist.

        Parameters
        ----------
        property_name : str
            Name of the discrete property

        Returns
        -------
        dict[int, float] | None
            Thickness mapping {0: 1.5, 1: 2.0} or None if no thicknesses found

        Examples
        --------
        >>> las = LasFile("well.las")
        >>> thicknesses = las.get_discrete_thicknesses('Zone')
        >>> # Returns: {0: 1.5, 1: 2.0}
        """
        thicknesses = {}
        prefix = f"{property_name}_"
        thickness_prefix = f"{property_name}_THICKNESS_"

        # First, check for inline thicknesses (e.g., "Label|color|style|thickness")
        for param_name, label_value in self.parameter_info.items():
            if param_name.startswith(prefix) and not param_name.startswith(thickness_prefix):
                # Extract the numeric suffix
                suffix = param_name[len(prefix):]
                try:
                    value = int(suffix)
                    # If the label contains a thickness (e.g., "Label|color|style|thickness"), extract it
                    if '|' in label_value:
                        parts = label_value.split('|')
                        if len(parts) >= 4:
                            thickness_str = parts[3].strip()
                            if thickness_str:  # Only add if thickness is not empty
                                try:
                                    thickness = float(thickness_str)
                                    thicknesses[value] = thickness
                                except ValueError:
                                    # Invalid float value, skip
                                    pass
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        # Second, check for separate thickness parameters (backward compatibility)
        # Only add if not already defined by inline format
        for param_name, thickness_value in self.parameter_info.items():
            if param_name.startswith(thickness_prefix):
                # Extract the numeric suffix
                suffix = param_name[len(thickness_prefix):]
                try:
                    value = int(suffix)
                    if value not in thicknesses:  # Inline format takes precedence
                        try:
                            thickness = float(thickness_value.strip())
                            thicknesses[value] = thickness
                        except ValueError:
                            # Invalid float value, skip
                            pass
                except ValueError:
                    # Not a valid integer suffix, skip
                    continue

        return thicknesses if thicknesses else None

    def check_depth_compatibility(self, other, well=None) -> dict:
        """
        Check if depth grids are compatible between two LAS files.

        Checks if all depth values in this LAS file exist in the other LAS file
        (within tolerance), or if depths are identical.

        Parameters
        ----------
        other : LasFile or str
            Either a LasFile instance to compare with, or a string source name.
            If string, must provide `well` parameter to look up the source.
        well : Well, optional
            Well object to look up source from if `other` is a string.
            Required when `other` is a string.

        Returns
        -------
        dict
            Dictionary with compatibility information:
            - 'compatible' (bool): True if grids are compatible
            - 'reason' (str): Explanation of compatibility/incompatibility
            - 'existing' (dict): Info about existing depth grid
            - 'new' (dict): Info about new depth grid
            - 'requires_resampling' (bool): Whether resampling is needed

        Raises
        ------
        ValueError
            If `other` is a string but `well` is not provided
        KeyError
            If source name doesn't exist in well

        Examples
        --------
        >>> # Compare two LasFile objects
        >>> las1 = LasFile("well1.las")
        >>> las2 = LasFile("well2.las")
        >>> result = las2.check_depth_compatibility(las1)
        >>> if result['compatible']:
        ...     print("Compatible!")
        >>> else:
        ...     print(f"Incompatible: {result['reason']}")

        >>> # Compare with a source by name
        >>> result = new_las.check_depth_compatibility('Petrophysics', well=well)
        """
        import numpy as np

        # Resolve other to LasFile if it's a string
        if isinstance(other, str):
            if well is None:
                raise ValueError(
                    f"When 'other' is a source name ('{other}'), the 'well' parameter is required"
                )

            # Look up the source in the well
            if other not in well._sources:
                available = ', '.join(well._sources.keys())
                raise KeyError(
                    f"Source '{other}' not found in well '{well.name}'. "
                    f"Available sources: {available or 'none'}"
                )

            other_las = well._sources[other]['las_file']
            if other_las is None:
                raise ValueError(
                    f"Source '{other}' has no LAS file (synthetic source). "
                    f"Cannot check compatibility with synthetic sources."
                )
            other = other_las

        # Get depth arrays
        self_data = self.data()
        other_data = other.data()

        self_depth = self_data[self.depth_column].values
        other_depth = other_data[other.depth_column].values

        # Remove NaN values
        self_depth = self_depth[~np.isnan(self_depth)]
        other_depth = other_depth[~np.isnan(other_depth)]

        if len(self_depth) == 0 or len(other_depth) == 0:
            return {
                'compatible': False,
                'reason': 'Empty depth array',
                'existing': self._depth_info(other_depth),
                'new': self._depth_info(self_depth),
                'requires_resampling': True
            }

        # Check if grids are identical (within tolerance)
        if len(self_depth) == len(other_depth) and np.allclose(self_depth, other_depth, rtol=1e-9, atol=1e-9):
            return {
                'compatible': True,
                'reason': 'Identical depth grids',
                'existing': self._depth_info(other_depth),
                'new': self._depth_info(self_depth),
                'requires_resampling': False
            }

        # Check if all depths in new file exist in existing file (subset check)
        # This allows adding data that's on a subset of existing depths
        tolerance = 1e-6  # 1 micron tolerance for depth matching
        all_depths_found = True
        for depth in self_depth:
            if not np.any(np.abs(other_depth - depth) < tolerance):
                all_depths_found = False
                break

        if all_depths_found:
            return {
                'compatible': True,
                'reason': 'New depths are subset of existing depths',
                'existing': self._depth_info(other_depth),
                'new': self._depth_info(self_depth),
                'requires_resampling': False
            }

        # Check if grids have same spacing but different start/stop
        # (can be handled by adding NaN padding)
        self_spacing = np.median(np.diff(self_depth)) if len(self_depth) > 1 else 0.0
        other_spacing = np.median(np.diff(other_depth)) if len(other_depth) > 1 else 0.0

        if np.abs(self_spacing - other_spacing) < tolerance:
            # Same spacing - could potentially pad with NaN
            return {
                'compatible': True,
                'reason': 'Same spacing, different range (can pad with NaN)',
                'existing': self._depth_info(other_depth),
                'new': self._depth_info(self_depth),
                'requires_resampling': False  # Can use NaN padding
            }

        # Incompatible - different grids requiring resampling
        return {
            'compatible': False,
            'reason': 'Different depth grids requiring resampling',
            'existing': self._depth_info(other_depth),
            'new': self._depth_info(self_depth),
            'requires_resampling': True
        }

    def _depth_info(self, depth: 'np.ndarray') -> dict:
        """Helper to extract depth grid information."""
        import numpy as np
        if len(depth) == 0:
            return {'samples': 0, 'start': np.nan, 'stop': np.nan, 'spacing': np.nan}

        spacing = np.median(np.diff(depth)) if len(depth) > 1 else 0.0
        return {
            'samples': len(depth),
            'start': float(depth[0]),
            'stop': float(depth[-1]),
            'spacing': float(spacing)
        }

    def data(
        self,
        include: Optional[Union[str, list[str]]] = None,
        exclude: Optional[Union[str, list[str]]] = None
    ) -> pd.DataFrame:
        """
        Lazy-load and return data with optional column filtering.

        Parameters
        ----------
        include : str or list[str], optional
            Column name(s) to include. If None, includes all columns.
            Can be a single string or a list of strings.
        exclude : str or list[str], optional
            Column name(s) to exclude. If both include and exclude are specified,
            exclude overrides (removes from include list).
            Can be a single string or a list of strings.

        Returns
        -------
        pd.DataFrame
            Well log data with curves as columns

        Examples
        --------
        >>> df = las.data()
        >>> df = las.data(include='PHIE')  # Single column
        >>> df = las.data(include=['DEPT', 'PHIE', 'SW'])
        >>> df = las.data(exclude='QC_Flag')  # Exclude single column
        >>> df = las.data(include=['DEPT', 'PHIE', 'SW', 'Zone'], exclude='Zone')
        """
        if self._data is None:
            self._load_data()

        # Apply filtering if requested
        if include is not None or exclude is not None:
            all_columns = self._data.columns.tolist()
            columns_filter = filter_names(all_columns, include, exclude)

            if columns_filter is not None:
                return self._data[columns_filter]

        return self._data

    def set_data(self, df: pd.DataFrame) -> None:
        """
        Set data DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Well log data with curves as columns
        """
        self._data = df

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
        # OPTIMIZED: Stream file line by line instead of loading all lines
        current_section = None
        line_number = 0

        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_number, line in enumerate(f):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
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
                        self._ascii_start_line = line_number + 1
                        break  # Stop parsing, data section found

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
        if self._ascii_start_line is None:
            raise LasFileError("Headers not parsed. Cannot load data.")

        # OPTIMIZED: Read directly from file instead of joining lines
        # Skip header rows and let pandas parse the ASCII section directly
        try:
            df = pd.read_csv(
                self.filepath,
                sep=r'\s+',
                names=self._curve_names,
                na_values=[self.null_value],
                skiprows=self._ascii_start_line,  # Skip header rows
                engine='c',  # Fast C parser
                dtype_backend='numpy_nullable',
                encoding='utf-8',
                encoding_errors='ignore',
                on_bad_lines='skip'  # Skip malformed rows
            )
        except Exception as e:
            raise LasFileError(
                f"Failed to parse ASCII data in {self.filepath}: {e}"
            )

        # OPTIMIZED: Batch apply multipliers and aliases instead of looping
        # Build mapping of columns to multiply and rename
        multiply_cols = {}
        rename_cols = {}

        for curve_name in self._curve_names:
            curve_meta = self.curves[curve_name]

            # Collect multipliers
            if curve_meta['multiplier'] is not None:
                multiply_cols[curve_name] = curve_meta['multiplier']

            # Collect aliases
            if curve_meta['alias'] is not None:
                rename_cols[curve_name] = curve_meta['alias']

        # Apply all multipliers at once (vectorized)
        if multiply_cols:
            for col, multiplier in multiply_cols.items():
                if col in df.columns:
                    df[col] = df[col] * multiplier

        # Apply all renames at once (single operation)
        if rename_cols:
            df = df.rename(columns=rename_cols)

        self._data = df
    
    @staticmethod
    def export_las(
        filepath: Union[str, Path],
        well_name: str,
        df: pd.DataFrame,
        unit_mappings: Optional[dict[str, str]] = None,
        null_value: float = -999.25,
        discrete_labels: Optional[dict[str, dict[int, str]]] = None,
        discrete_colors: Optional[dict[str, dict[int, str]]] = None,
        discrete_styles: Optional[dict[str, dict[int, str]]] = None,
        discrete_thicknesses: Optional[dict[str, dict[int, float]]] = None,
        template_las: Optional['LasFile'] = None
    ) -> None:
        """
        Export DataFrame to LAS 2.0 format file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Output LAS file path
        well_name : str
            Well name for the LAS file
        df : pd.DataFrame
            DataFrame to export (must contain DEPT column)
        unit_mappings : dict[str, str], optional
            Mapping of column names to units (e.g., {'PHIE': 'v/v', 'DEPT': 'm'})
            If not provided, uses empty units
        null_value : float, default -999.25
            Value to use for missing data
        discrete_labels : dict[str, dict[int, str]], optional
            Label mappings for discrete properties stored in ~Parameter section.
            Format: {'PropertyName': {0: 'Label0', 1: 'Label1'}}
            Example: {'Zone': {0: 'NonReservoir', 1: 'Reservoir'}}
        discrete_colors : dict[str, dict[int, str]], optional
            Color mappings for discrete properties stored in ~Parameter section.
            Format: {'PropertyName': {0: 'red', 1: 'green'}}
            Example: {'Zone': {0: 'red', 1: 'green'}}
        template_las : LasFile, optional
            Source LAS file to use as template. Preserves original ~Version info,
            ~Well parameters (excluding STRT/STOP/STEP/NULL), and ~Parameter entries
            not related to discrete labels and colors. This prevents data erosion when updating
            existing LAS files.

        Raises
        ------
        ValueError
            If DEPT column not found in DataFrame
        LasFileError
            If file write fails

        Examples
        --------
        >>> df = well.data()
        >>> LasFile.export_las(
        ...     'output.las',
        ...     well_name='12/3-2 B',
        ...     df=df,
        ...     unit_mappings={'DEPT': 'm', 'PHIE': 'v/v', 'SW': 'v/v'}
        ... )

        >>> # Export with discrete labels stored in parameter section
        >>> LasFile.export_las(
        ...     'output.las',
        ...     well_name='12/3-2 B',
        ...     df=df,
        ...     unit_mappings={'DEPT': 'm', 'Zone': ''},
        ...     discrete_labels={'Zone': {0: 'NonReservoir', 1: 'Reservoir'}}
        ... )

        >>> # Export using original LAS as template (preserves metadata)
        >>> LasFile.export_las(
        ...     'updated.las',
        ...     well_name='12/3-2 B',
        ...     df=df,
        ...     unit_mappings={'DEPT': 'm', 'PHIE': 'v/v'},
        ...     template_las=original_las
        ... )
        """
        if 'DEPT' not in df.columns:
            raise ValueError(
                "DataFrame must contain 'DEPT' column. "
                f"Available columns: {', '.join(df.columns)}"
            )

        unit_mappings = unit_mappings or {}
        filepath = Path(filepath)

        # Get depth range
        dept = df['DEPT'].dropna()
        if len(dept) == 0:
            raise ValueError("DEPT column contains no valid data")

        start_depth = float(dept.min())
        stop_depth = float(dept.max())
        step_depth = float(dept.diff().median()) if len(dept) > 1 else 0.1

        # Build LAS file content
        lines = []

        # ~Version section (use template if available)
        lines.append("~Version Information")
        if template_las:
            # Preserve original version info
            version_info = template_las.version_info
            for key, value in version_info.items():
                desc = f"CWLS log ASCII Standard -VERSION {value}" if key == 'VERS' else ""
                lines.append(f" {key:<12}.              {value:>10} : {desc}")
        else:
            # Default version info
            lines.append(" VERS.                          2.0 : CWLS log ASCII Standard -VERSION 2.0")
            lines.append(" WRAP.                          NO  : One line per depth step")
        lines.append("")

        # ~Well section (use template for non-depth parameters)
        lines.append("~Well Information")
        lines.append(f" STRT.m                  {start_depth:10.4f} : START DEPTH")
        lines.append(f" STOP.m                  {stop_depth:10.4f} : STOP DEPTH")
        lines.append(f" STEP.m                  {step_depth:10.4f} : STEP")
        lines.append(f" NULL.                   {null_value:10.4f} : NULL VALUE")

        if template_las:
            # Preserve all original well parameters except STRT/STOP/STEP/NULL
            well_info = template_las.well_info
            skip_params = {'STRT', 'STOP', 'STEP', 'NULL'}
            for key, value in well_info.items():
                if key not in skip_params:
                    # Format parameter line
                    lines.append(f" {key:<12}.              {value:>10} : {key}")
        else:
            # Just add well name if no template
            lines.append(f" WELL.                   {well_name:>10} : WELL")
        lines.append("")

        # ~Curve section (preserve template descriptions if available)
        lines.append("~Curve Information")
        for col in df.columns:
            unit = unit_mappings.get(col, '')
            # Try to get description from template
            description = col
            if template_las and col in template_las.curves:
                description = template_las.curves[col].get('description', col)
            # Format: MNEM.UNIT VALUE : DESCRIPTION
            lines.append(f" {col:<12}.{unit:<8}              : {description}")
        lines.append("")

        # ~Parameter section (preserve non-discrete-label parameters from template, add new discrete labels)
        # Collect parameters to write
        params_to_write = {}

        # First, add non-discrete-label parameters from template if available
        if template_las:
            # Get discrete label parameter names to skip
            skip_params = {'DISCRETE_PROPS'}
            if template_las.discrete_properties:
                for prop_name in template_las.discrete_properties:
                    # Skip any parameter starting with discrete property name
                    for param_key in template_las.parameter_info.keys():
                        if param_key.startswith(f"{prop_name}_"):
                            skip_params.add(param_key)

            # Preserve all other parameters
            for param_key, param_value in template_las.parameter_info.items():
                if param_key not in skip_params:
                    params_to_write[param_key] = (param_value, f"Preserved from original")

        # Add new discrete labels, colors, styles, and thicknesses if provided
        if discrete_labels:
            # Add list of discrete properties
            discrete_prop_names = ','.join(sorted(discrete_labels.keys()))
            params_to_write['DISCRETE_PROPS'] = (discrete_prop_names, "Discrete properties")

            # Add label mappings for each discrete property
            # If colors, styles, or thicknesses are also provided, combine them inline
            # Format: "Label|color|style|thickness"
            for prop_name, label_mapping in sorted(discrete_labels.items()):
                # Sort by value for consistent output
                for value in sorted(label_mapping.keys()):
                    label = label_mapping[value]
                    parts = [label]
                    desc_parts = ["label"]

                    # Check if there's a corresponding color
                    if discrete_colors and prop_name in discrete_colors and value in discrete_colors[prop_name]:
                        color = discrete_colors[prop_name][value]
                        parts.append(color)
                        desc_parts.append("color")

                    # Check if there's a corresponding style
                    if discrete_styles and prop_name in discrete_styles and value in discrete_styles[prop_name]:
                        style = discrete_styles[prop_name][value]
                        # If we have style but no color, add empty color field
                        if len(parts) == 1:
                            parts.append("")
                        parts.append(style)
                        desc_parts.append("style")

                    # Check if there's a corresponding thickness
                    if discrete_thicknesses and prop_name in discrete_thicknesses and value in discrete_thicknesses[prop_name]:
                        thickness = discrete_thicknesses[prop_name][value]
                        # If we have thickness but no color/style, add empty fields
                        while len(parts) < 3:
                            parts.append("")
                        parts.append(str(thickness))
                        desc_parts.append("thickness")

                    # Combine all parts with pipe separator
                    combined = "|".join(parts)
                    param_name = f"{prop_name}_{value}"
                    desc = f"{prop_name} {' and '.join(desc_parts)} for value {value}"
                    params_to_write[param_name] = (combined, desc)

        # For properties with colors/styles/thicknesses but no labels, store separately (legacy format)
        if discrete_colors:
            for prop_name, color_mapping in sorted(discrete_colors.items()):
                # Only write separate color params if no labels were defined for this property
                if not discrete_labels or prop_name not in discrete_labels:
                    for value in sorted(color_mapping.keys()):
                        color = color_mapping[value]
                        param_name = f"{prop_name}_COLOR_{value}"
                        params_to_write[param_name] = (color, f"{prop_name} color for value {value}")

        if discrete_styles:
            for prop_name, style_mapping in sorted(discrete_styles.items()):
                # Only write separate style params if no labels were defined for this property
                if not discrete_labels or prop_name not in discrete_labels:
                    for value in sorted(style_mapping.keys()):
                        style = style_mapping[value]
                        param_name = f"{prop_name}_STYLE_{value}"
                        params_to_write[param_name] = (style, f"{prop_name} style for value {value}")

        if discrete_thicknesses:
            for prop_name, thickness_mapping in sorted(discrete_thicknesses.items()):
                # Only write separate thickness params if no labels were defined for this property
                if not discrete_labels or prop_name not in discrete_labels:
                    for value in sorted(thickness_mapping.keys()):
                        thickness = thickness_mapping[value]
                        param_name = f"{prop_name}_THICKNESS_{value}"
                        params_to_write[param_name] = (str(thickness), f"{prop_name} thickness for value {value}")

        # Write parameter section if we have any parameters
        if params_to_write:
            lines.append("~Parameter Information")
            for param_name, (param_value, param_desc) in params_to_write.items():
                lines.append(f" {param_name:<12}.       {param_value:>10} : {param_desc}")
            lines.append("")

        # ~Ascii section
        lines.append("~Ascii")

        # Write data rows - VECTORIZED for performance
        # Replace NaN with null_value (in-place to avoid copy)
        df_export = df.fillna(null_value)

        # FAST PATH: Use vectorized numpy operations instead of Python loops
        # Convert entire DataFrame to string array at once
        values = df_export.values  # Get numpy array (no copy if possible)

        # Get list of discrete property columns for integer formatting
        discrete_cols = set()
        if discrete_labels:
            discrete_cols = set(discrete_labels.keys())

        # Format all values at once using numpy vectorization
        # This is 10-100x faster than iterrows()
        formatted = np.empty(values.shape, dtype='U12')  # Pre-allocate string array
        for col_idx in range(values.shape[1]):
            col_name = df_export.columns[col_idx]
            col_data = values[:, col_idx]

            # Format column values (all same type within column)
            if np.issubdtype(col_data.dtype, np.number):
                # Discrete properties: format as integers (no decimals)
                if col_name in discrete_cols:
                    formatted[:, col_idx] = np.char.mod('%12.0f', col_data)
                # Continuous/sampled properties: format with decimals
                else:
                    formatted[:, col_idx] = np.char.mod('%12.4f', col_data)
            else:
                formatted[:, col_idx] = np.char.mod('%12s', col_data.astype(str))

        # Join rows efficiently
        data_lines = [''.join(row) for row in formatted]
        lines.extend(data_lines)

        # Write to file using buffered write
        try:
            with open(filepath, 'w', encoding='utf-8', buffering=65536) as f:
                f.write('\n'.join(lines))
        except Exception as e:
            raise LasFileError(f"Failed to write LAS file to {filepath}: {e}")

    def export(
        self,
        filepath: Union[str, Path],
        null_value: float = -999.25
    ) -> None:
        """
        Export this LasFile instance to a LAS 2.0 format file.

        This is an instance method that exports the LasFile's own data,
        including all metadata from the ~Version, ~Well, ~Parameter, and ~Curve sections.

        Parameters
        ----------
        filepath : Union[str, Path]
            Output LAS file path
        null_value : float, default -999.25
            Value to use for missing data

        Examples
        --------
        >>> # Load, modify, and re-export a LAS file
        >>> las = LasFile('input.las')
        >>> # ... modify via las.set_data(df) if needed ...
        >>> las.export('output.las')

        >>> # Create LasFile from Well and export
        >>> las = well.to_las(include=['PHIE', 'SW'])
        >>> las.export('output.las')
        """
        if self._data is None:
            raise LasFileError("No data loaded. Call .data() method first to load data.")

        # Collect curve units and discrete labels
        unit_mappings = {}
        for curve_name, curve_meta in self.curves.items():
            unit_mappings[curve_name] = curve_meta.get('unit', '')

        # Collect discrete metadata from parameter section
        discrete_labels = None
        discrete_colors = None
        discrete_styles = None
        discrete_thicknesses = None
        discrete_props = self.discrete_properties
        if discrete_props:
            discrete_labels = {}
            discrete_colors = {}
            discrete_styles = {}
            discrete_thicknesses = {}
            for prop_name in discrete_props:
                # Collect labels
                labels = self.get_discrete_labels(prop_name)
                if labels:
                    discrete_labels[prop_name] = labels

                # Collect colors
                colors = {}
                for key, value in self.parameter_info.items():
                    if key.startswith(f"{prop_name}_") and key.endswith("_COLOR"):
                        # Extract discrete value from key like "Well_Tops_0_COLOR"
                        value_str = key[len(prop_name)+1:-6]  # Remove prefix and "_COLOR" suffix
                        if value_str.isdigit():
                            colors[int(value_str)] = value
                if colors:
                    discrete_colors[prop_name] = colors

                # Collect styles
                styles = {}
                for key, value in self.parameter_info.items():
                    if key.startswith(f"{prop_name}_") and key.endswith("_STYLE"):
                        # Extract discrete value from key like "Well_Tops_0_STYLE"
                        value_str = key[len(prop_name)+1:-6]  # Remove prefix and "_STYLE" suffix
                        if value_str.isdigit():
                            styles[int(value_str)] = value
                if styles:
                    discrete_styles[prop_name] = styles

                # Collect thicknesses
                thicknesses = {}
                for key, value in self.parameter_info.items():
                    if key.startswith(f"{prop_name}_") and key.endswith("_THICKNESS"):
                        # Extract discrete value from key like "Well_Tops_0_THICKNESS"
                        value_str = key[len(prop_name)+1:-10]  # Remove prefix and "_THICKNESS" suffix
                        if value_str.isdigit():
                            thicknesses[int(value_str)] = float(value)
                if thicknesses:
                    discrete_thicknesses[prop_name] = thicknesses

        # Use static method to do the actual export, passing self as template
        LasFile.export_las(
            filepath=filepath,
            well_name=self.well_name or 'UNKNOWN',
            df=self._data,
            unit_mappings=unit_mappings,
            null_value=null_value,
            discrete_labels=discrete_labels,
            discrete_colors=discrete_colors if discrete_colors else None,
            discrete_styles=discrete_styles if discrete_styles else None,
            discrete_thicknesses=discrete_thicknesses if discrete_thicknesses else None,
            template_las=self
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LasFile('{self.filepath.name}', "
            f"well='{self.well_name}', "
            f"curves={len(self.curves)})"
        )