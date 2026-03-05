"""Template class for well log display configuration."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import json

import pandas as pd

from . import DEFAULT_COLORS


class Template:
    """
    Template for well log display configuration.

    A template defines the layout and styling of tracks in a well log display.
    Each track can contain multiple logs, fills, and tops markers.

    Parameters
    ----------
    name : str, optional
        Template name for identification
    tracks : list[dict], optional
        List of track definitions. If None, creates empty template.

    Attributes
    ----------
    name : str
        Template name
    tracks : list[dict]
        List of track configurations

    Examples
    --------
    >>> # Create empty template
    >>> template = Template("reservoir")
    >>>
    >>> # Add a GR track
    >>> template.add_track(
    ...     track_type="continuous",
    ...     logs=[{"name": "GR", "x_range": [0, 150], "color": "green"}]
    ... )
    >>>
    >>> # Add a porosity/saturation track with fill
    >>> template.add_track(
    ...     track_type="continuous",
    ...     logs=[
    ...         {"name": "PHIE", "x_range": [0.45, 0], "color": "blue"},
    ...         {"name": "SW", "x_range": [0, 1], "color": "red"}
    ...     ],
    ...     fill={
    ...         "left": {"curve": "PHIE"},
    ...         "right": {"value": 0},
    ...         "color": "lightblue",
    ...         "alpha": 0.5
    ...     }
    ... )
    >>>
    >>> # Add depth track
    >>> template.add_track(track_type="depth")
    >>>
    >>> # Save to file
    >>> template.save("reservoir_template.json")
    >>>
    >>> # Load from file
    >>> template2 = Template.load("reservoir_template.json")
    """

    def __init__(self, name: str = "default", tracks: Optional[list[dict]] = None):
        """Initialize template."""
        self.name = name
        self.tracks = tracks if tracks is not None else []
        self.tops = []  # List of well tops configurations

    def add_track(
        self,
        track_type: str = "continuous",
        logs: Optional[list[dict]] = None,
        fill: Optional[Union[dict, list[dict]]] = None,
        tops: Optional[dict] = None,
        width: float = 1.0,
        title: Optional[str] = None,
        log_scale: bool = False
    ) -> 'Template':
        """
        Add a track to the template.

        Parameters
        ----------
        track_type : {"continuous", "discrete", "depth"}, default "continuous"
            Type of track:
            - "continuous": Continuous log curves (GR, RHOB, etc.)
            - "discrete": Discrete/categorical logs (facies, zones)
            - "depth": Depth axis track
        logs : list[dict], optional
            List of log configurations. Each dict can contain:
            - name (str): Property name
            - x_range (list[float, float]): Min and max x-axis values [left, right]
            - scale (str): Optional override for this log's scale ("log" or "linear")
              If not specified, uses the track's log_scale setting
            - color (str): Line color
            - style (str): Line style - supports both matplotlib codes and friendly names:
              Matplotlib: "-" (solid), "--" (dashed), "-." (dashdot), ":" (dotted), "none" (no line)
              Friendly: "solid", "dashed", "dashdot", "dotted", "none"
              Use "none" to show only markers without a connecting line
            - thickness (float): Line width
            - alpha (float): Transparency (0-1)
            - marker (str): Marker style for data points (disabled by default). Supports:
              Matplotlib codes: "o", "s", "D", "^", "v", "<", ">", "+", "x", "*", "p", "h", ".", ",", "|", "_"
              Friendly names: "circle", "square", "diamond", "triangle_up", "triangle_down",
              "triangle_left", "triangle_right", "plus", "cross", "star", "pentagon", "hexagon",
              "point", "pixel", "vline", "hline"
            - marker_size (float): Size of markers (default: 6)
            - marker_outline_color (str): Marker edge color (defaults to 'color')
            - marker_fill (str): Marker fill color (optional). If not specified, markers are unfilled
            - marker_interval (int): Show every nth marker (default: 1, shows all markers)
        fill : Union[dict, list[dict]], optional
            Fill configuration or list of fill configurations. Each fill dict can contain:
            - left: Left boundary (string/number or dict)
              - Simple: "track_edge", "CurveName", or numeric value
              - Dict: {"curve": name}, {"value": float}, or {"track_edge": "left"}
            - right: Right boundary (same format as left)
            - color (str): Fill color name (for solid fills)
            - colormap (str): Matplotlib colormap name (e.g., "viridis", "inferno")
              Creates horizontal bands where depth intervals are colored based on curve values
            - colormap_curve (str): Curve name to use for colormap values (defaults to left boundary curve)
            - color_range (list): [min, max] values for colormap normalization
            - alpha (float): Transparency (0-1)
            Multiple fills are drawn in order (first fill is drawn first, then subsequent fills on top)
        tops : dict, optional
            Formation tops configuration with keys:
            - name (str): Property name containing tops
            - line_style (str): Line style for markers
            - line_width (float): Line thickness
            - dotted (bool): Use dotted lines
            - title_size (int): Font size for labels
            - title_weight (str): Font weight ("normal", "bold")
            - title_orientation (str): Text alignment ("left", "center", "right")
            - line_offset (float): Horizontal offset for line position
        width : float, default 1.0
            Relative width of track (used for layout proportions)
        title : str, optional
            Track title to display at top
        log_scale : bool, default False
            Use logarithmic scale for the entire track. Individual logs can override
            this with the "scale" parameter ("log" or "linear")

        Returns
        -------
        Template
            Self for method chaining

        Examples
        --------
        >>> template = Template("my_template")
        >>>
        >>> # Add GR track
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{
        ...         "name": "GR",
        ...         "x_range": [0, 150],
        ...         "color": "green",
        ...         "style": "solid",  # or "-", both work
        ...         "thickness": 1.0
        ...     }],
        ...     title="Gamma Ray"
        ... )
        >>>
        >>> # Add resistivity track with log scale
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{
        ...         "name": "RES",
        ...         "x_range": [0.2, 2000],
        ...         "color": "red"
        ...     }],
        ...     title="Resistivity",
        ...     log_scale=True  # Apply log scale to entire track
        ... )
        >>>
        >>> # Add track with mixed scales (one log overrides track setting)
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[
        ...         {"name": "ILD", "x_range": [0.2, 2000], "color": "red"},  # Uses track log_scale
        ...         {"name": "GR", "x_range": [0, 150], "scale": "linear", "color": "green"}  # Override to linear
        ...     ],
        ...     log_scale=True  # Default for track is log scale
        ... )
        >>>
        >>> # Add track with different line styles
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[
        ...         {"name": "RHOB", "x_range": [1.95, 2.95], "color": "red", "style": "solid", "thickness": 1.5},
        ...         {"name": "NPHI", "x_range": [0.45, -0.15], "color": "blue", "style": "dashed", "thickness": 2.0}
        ...     ],
        ...     title="Density & Neutron"
        ... )
        >>>
        >>> # Add track with markers (line + markers)
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{
        ...         "name": "PERM",
        ...         "x_range": [0.1, 1000],
        ...         "color": "green",
        ...         "style": "solid",
        ...         "marker": "circle",
        ...         "marker_size": 4,
        ...         "marker_fill": "lightgreen"
        ...     }],
        ...     title="Permeability",
        ...     log_scale=True
        ... )
        >>>
        >>> # Add track with markers only (no line)
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{
        ...         "name": "SAMPLE_POINTS",
        ...         "x_range": [0, 100],
        ...         "color": "red",
        ...         "style": "none",
        ...         "marker": "diamond",
        ...         "marker_size": 8,
        ...         "marker_outline_color": "darkred",
        ...         "marker_fill": "yellow"
        ...     }],
        ...     title="Sample Locations"
        ... )
        >>>
        >>> # Add porosity track with fill (simplified API)
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{
        ...         "name": "PHIE",
        ...         "x_range": [0.45, 0],
        ...         "color": "blue"
        ...     }],
        ...     fill={
        ...         "left": "PHIE",      # Simple: curve name
        ...         "right": 0,          # Simple: numeric value
        ...         "color": "lightblue",
        ...         "alpha": 0.5
        ...     },
        ...     title="Porosity"
        ... )
        >>>
        >>> # Add GR track with colormap fill (horizontal bands colored by GR value)
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{"name": "GR", "x_range": [0, 150], "color": "black"}],
        ...     fill={
        ...         "left": "track_edge",  # Simple: track edge
        ...         "right": "GR",         # Simple: curve name
        ...         "colormap": "viridis",
        ...         "color_range": [20, 150],
        ...         "alpha": 0.7
        ...     },
        ...     title="Gamma Ray"
        ... )
        >>>
        >>> # Add porosity/saturation track with multiple fills
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[
        ...         {"name": "PHIE", "x_range": [0.45, 0], "color": "blue"},
        ...         {"name": "SW", "x_range": [0, 1], "color": "red"}
        ...     ],
        ...     fill=[
        ...         # Fill 1: PHIE to zero with light blue
        ...         {
        ...             "left": "PHIE",
        ...             "right": 0,
        ...             "color": "lightblue",
        ...             "alpha": 0.3
        ...         },
        ...         # Fill 2: SW to one with light red
        ...         {
        ...             "left": "SW",
        ...             "right": 1,
        ...             "color": "lightcoral",
        ...             "alpha": 0.3
        ...         }
        ...     ],
        ...     title="PHIE & SW"
        ... )
        >>>
        >>> # Add discrete facies track
        >>> template.add_track(
        ...     track_type="discrete",
        ...     logs=[{"name": "Facies"}],
        ...     title="Facies"
        ... )
        >>>
        >>> # Add depth track
        >>> template.add_track(track_type="depth", width=0.3)
        """
        # Normalize fill to always be a list internally (for backward compatibility)
        fill_list = None
        if fill is not None:
            if isinstance(fill, dict):
                fill_list = [fill]
            else:
                fill_list = fill

        track = {
            "type": track_type,
            "logs": logs or [],
            "fill": fill_list,
            "tops": tops,
            "width": width,
            "title": title,
            "log_scale": log_scale
        }
        self.tracks.append(track)
        return self

    def remove_track(self, index: int) -> 'Template':
        """
        Remove track at specified index.

        Parameters
        ----------
        index : int
            Track index to remove

        Returns
        -------
        Template
            Self for method chaining

        Examples
        --------
        >>> template.remove_track(0)  # Remove first track
        """
        if 0 <= index < len(self.tracks):
            self.tracks.pop(index)
        else:
            raise IndexError(f"Track index {index} out of range (0-{len(self.tracks)-1})")
        return self

    def add_tops(
        self,
        property_name: Optional[str] = None,
        tops_dict: Optional[dict[float, str]] = None,
        colors: Optional[dict[float, str]] = None,
        styles: Optional[dict[float, str]] = None,
        thicknesses: Optional[dict[float, float]] = None
    ) -> 'Template':
        """
        Add well tops configuration to the template.

        Tops added to the template will be displayed in all WellViews created from
        this template. They span across all tracks (except depth track).

        Parameters
        ----------
        property_name : str, optional
            Name of discrete property in well containing tops data.
            The property name will be resolved when the template is used with a well.
        tops_dict : dict[float, str], optional
            Dictionary mapping depth values to formation names.
            Example: {2850.0: 'Formation A', 2920.5: 'Formation B'}
        colors : dict[float, str], optional
            Optional color mapping for each depth or discrete value.
            - For tops_dict: keys are depths matching tops_dict keys
            - For property_name: keys are discrete values (integers)
            If not provided and using a discrete property with color mapping,
            those colors will be used.
        styles : dict[float, str], optional
            Optional line style mapping for each depth or discrete value.
            Valid styles: 'solid', 'dashed', 'dotted', 'dashdot'
            - For tops_dict: keys are depths matching tops_dict keys
            - For property_name: keys are discrete values (integers)
            If not provided and using a discrete property with style mapping,
            those styles will be used.
        thicknesses : dict[float, float], optional
            Optional line thickness mapping for each depth or discrete value.
            - For tops_dict: keys are depths matching tops_dict keys
            - For property_name: keys are discrete values (integers)
            If not provided and using a discrete property with thickness mapping,
            those thicknesses will be used.

        Returns
        -------
        Template
            Self for method chaining

        Examples
        --------
        >>> # Add tops from discrete property (resolved when used with well)
        >>> template = Template("my_template")
        >>> template.add_track(...)
        >>> template.add_tops(property_name="Formations")
        >>>
        >>> # Add manual tops with colors
        >>> template.add_tops(
        ...     tops_dict={2850.0: 'Reservoir', 2920.5: 'Seal'},
        ...     colors={2850.0: 'yellow', 2920.5: 'gray'}
        ... )
        >>>
        >>> # Add tops from discrete property with color overrides
        >>> template.add_tops(
        ...     property_name='Zone',
        ...     colors={0: 'red', 1: 'green', 2: 'blue'}  # Map discrete values
        ... )

        Notes
        -----
        Tops are drawn as horizontal lines spanning all tracks (except depth track).
        Formation names are displayed at the right end of each line, floating above it.
        """
        if property_name is None and tops_dict is None:
            raise ValueError("Must provide either 'property_name' or 'tops_dict'")

        if property_name is not None and tops_dict is not None:
            raise ValueError("Cannot specify both 'property_name' and 'tops_dict'")

        # Store tops configuration
        tops_config = {
            'property_name': property_name,
            'tops_dict': tops_dict,
            'colors': colors,
            'styles': styles,
            'thicknesses': thicknesses
        }
        self.tops.append(tops_config)
        return self

    def edit_track(self, index: int, **kwargs) -> 'Template':
        """
        Edit track at specified index.

        Parameters
        ----------
        index : int
            Track index to edit
        **kwargs
            Track parameters to update (same as add_track)

        Returns
        -------
        Template
            Self for method chaining

        Examples
        --------
        >>> # Change track title
        >>> template.edit_track(0, title="New Title")
        >>>
        >>> # Update log styling
        >>> template.edit_track(1, logs=[{"name": "PHIE", "color": "red"}])
        """
        if 0 <= index < len(self.tracks):
            for key, value in kwargs.items():
                # Normalize fill to list format (for backward compatibility)
                if key == "fill" and value is not None:
                    if isinstance(value, dict):
                        value = [value]

                if key in self.tracks[index]:
                    self.tracks[index][key] = value
        else:
            raise IndexError(f"Track index {index} out of range (0-{len(self.tracks)-1})")
        return self

    def get_track(self, index: int) -> dict:
        """
        Get track configuration at specified index.

        Parameters
        ----------
        index : int
            Track index

        Returns
        -------
        dict
            Track configuration

        Examples
        --------
        >>> track_config = template.get_track(0)
        >>> print(track_config["type"])
        'continuous'
        """
        if 0 <= index < len(self.tracks):
            return self.tracks[index].copy()
        else:
            raise IndexError(f"Track index {index} out of range (0-{len(self.tracks)-1})")

    def list_tracks(self) -> pd.DataFrame:
        """
        List all tracks with summary information.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Index, Type, Logs, Title, Width

        Examples
        --------
        >>> template.list_tracks()
           Index       Type           Logs      Title  Width
        0      0 continuous          [GR]  Gamma Ray    1.0
        1      1 continuous  [PHIE, SW]   Porosity    1.0
        2      2      depth            []      Depth    0.3
        """
        rows = []
        for i, track in enumerate(self.tracks):
            log_names = [log.get("name", "?") for log in track.get("logs", [])]
            rows.append({
                "Index": i,
                "Type": track.get("type", "?"),
                "Logs": log_names if log_names else [],
                "Title": track.get("title", ""),
                "Width": track.get("width", 1.0)
            })
        return pd.DataFrame(rows)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save template to JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to save JSON file

        Examples
        --------
        >>> template.save("reservoir_template.json")
        """
        filepath = Path(filepath)
        data = {
            "name": self.name,
            "tracks": self.tracks,
            "tops": self.tops
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Template':
        """
        Load template from JSON file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to JSON file

        Returns
        -------
        Template
            Loaded template

        Examples
        --------
        >>> template = Template.load("reservoir_template.json")
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        template = cls(name=data.get("name", "loaded"), tracks=data.get("tracks", []))
        template.tops = data.get("tops", [])
        return template

    def to_dict(self) -> dict:
        """
        Export template as dictionary.

        Returns
        -------
        dict
            Template configuration

        Examples
        --------
        >>> config = template.to_dict()
        >>> print(config.keys())
        dict_keys(['name', 'tracks', 'tops'])
        """
        return {
            "name": self.name,
            "tracks": self.tracks,
            "tops": self.tops
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Template':
        """
        Create template from dictionary.

        Parameters
        ----------
        data : dict
            Template configuration dictionary

        Returns
        -------
        Template
            New template instance

        Examples
        --------
        >>> config = {"name": "test", "tracks": [...], "tops": [...]}
        >>> template = Template.from_dict(config)
        """
        template = cls(name=data.get("name", "unnamed"), tracks=data.get("tracks", []))
        template.tops = data.get("tops", [])
        return template

    def __repr__(self) -> str:
        """String representation."""
        return f"Template('{self.name}', tracks={len(self.tracks)})"
