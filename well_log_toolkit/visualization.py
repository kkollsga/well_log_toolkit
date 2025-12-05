"""
Well log visualization for Jupyter Lab.

Provides Template and WellView classes for creating customizable well log displays.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from .well import Well

# Default color palettes
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


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


class WellView:
    """
    Interactive well log display for Jupyter Lab.

    Creates matplotlib-based well log plots with multiple tracks showing
    continuous logs, discrete properties, fills, and formation tops.

    Parameters
    ----------
    well : Well
        Well object containing log data
    depth_range : tuple[float, float], optional
        Depth interval to display [start_depth, end_depth]. If None, shows full depth range.
    template : Union[Template, dict, str], optional
        Display template. Can be:
        - Template object
        - Dictionary with template configuration
        - String name of template stored in well's parent manager
        If None, creates a simple default view.
    figsize : tuple[float, float], optional
        Figure size (width, height) in inches. If None, calculated from number of tracks.
    dpi : int, default 100
        Figure resolution

    Class Attributes
    ----------------
    HEADER_BOX_TOP : float
        Fixed top position of header boxes for alignment across tracks
    HEADER_TITLE_SPACING : float
        Vertical space between log name and its scale line in continuous tracks
    HEADER_LOG_SPACING : float
        Vertical space allocated per log in continuous tracks
    HEADER_TOP_PADDING : float
        Padding inside header box above content
    HEADER_BOTTOM_PADDING : float
        Padding inside header box below content

    Attributes
    ----------
    well : Well
        Source well object
    depth_range : tuple[float, float]
        Displayed depth range
    template : Template
        Display template configuration
    fig : matplotlib.figure.Figure
        Matplotlib figure object
    axes : list[matplotlib.axes.Axes]
        List of axes for each track

    Examples
    --------
    >>> from well_log_toolkit import WellDataManager
    >>> from well_log_toolkit.visualization import WellView, Template
    >>>
    >>> # Load data
    >>> manager = WellDataManager()
    >>> manager.load_las("well.las")
    >>> well = manager.well_36_7_5_A
    >>>
    >>> # Create template
    >>> template = Template("basic")
    >>> template.add_track(
    ...     track_type="continuous",
    ...     logs=[{"name": "GR", "x_range": [0, 150], "color": "green"}],
    ...     title="Gamma Ray"
    ... )
    >>> template.add_track(track_type="depth")
    >>>
    >>> # Display well log with depth range
    >>> view = WellView(well, depth_range=[2800, 3000], template=template)
    >>> view.show()
    >>>
    >>> # Or auto-calculate from formation tops
    >>> template.add_tops(property_name='Zone')
    >>> view2 = WellView(well, tops=['Top_Brent', 'Top_Statfjord'], template=template)
    >>> view2.show()
    >>>
    >>> # Or use template from manager
    >>> manager.set_template("reservoir", template)
    >>> view3 = WellView(well, depth_range=[3000, 3200], template="reservoir")
    >>> view3.show()
    >>>
    >>> # Save figure
    >>> view.save("well_log.png", dpi=300)
    """

    # Class-level configuration for header styling (easily customizable)
    HEADER_BOX_TOP = 1.1  # Fixed top position of header boxes
    HEADER_TITLE_SPACING = 0.0015  # Space between log name and scale line
    HEADER_LOG_SPACING = 0.025  # Vertical space per log
    HEADER_TOP_PADDING = 0.01  # Padding above content in header box
    HEADER_BOTTOM_PADDING = 0.01  # Padding below content in header box

    def __init__(
        self,
        well: 'Well',
        depth_range: Optional[tuple[float, float]] = None,
        tops: Optional[list[str]] = None,
        template: Optional[Union[Template, dict, str]] = None,
        figsize: Optional[tuple[float, float]] = None,
        dpi: int = 100,
        header_config: Optional[dict] = None
    ):
        """
        Initialize WellView.

        Parameters
        ----------
        well : Well
            Well object containing log data
        depth_range : tuple[float, float], optional
            Depth interval to display [top, bottom].
            Mutually exclusive with `tops` parameter.
        tops : list[str], optional
            List of formation top names to display. The depth range will be calculated
            automatically from the minimum and maximum depths of these tops, with 5%
            padding added (minimum range of 50m).
            Mutually exclusive with `depth_range` parameter.
            Requires that formation tops have been loaded in the well or added to the template.
        template : Union[Template, dict, str], optional
            Display template configuration
        figsize : tuple[float, float], optional
            Figure size in inches
        dpi : int, default 100
            Figure resolution
        header_config : dict, optional
            Header styling configuration. Supported keys:
            - header_box_top (float): Fixed top position of header boxes
            - header_title_spacing (float): Vertical space between log name and scale line
            - header_log_spacing (float): Vertical space allocated per log
            - header_top_padding (float): Padding above content in header box
            - header_bottom_padding (float): Padding below content in header box
            If None or keys omitted, uses class defaults.

        Examples
        --------
        >>> # Use depth range
        >>> view1 = WellView(well, depth_range=[2800, 3000], template=template)
        >>>
        >>> # Use formation tops to auto-calculate range
        >>> view2 = WellView(well, tops=['Top_Brent', 'Top_Statfjord'], template=template)
        >>>
        >>> # Customize header spacing
        >>> view3 = WellView(
        ...     well,
        ...     template=template,
        ...     header_config={"header_log_spacing": 0.04, "header_title_spacing": 0.005}
        ... )
        """
        # Validate mutually exclusive parameters
        if depth_range is not None and tops is not None:
            raise ValueError(
                "Parameters 'depth_range' and 'tops' are mutually exclusive. "
                "Provide one or the other, not both."
            )

        self.well = well
        self.dpi = dpi

        # Header configuration (use provided values or fall back to class defaults)
        if header_config is None:
            header_config = {}

        self.header_box_top = header_config.get('header_box_top', self.HEADER_BOX_TOP)
        self.header_title_spacing = header_config.get('header_title_spacing', self.HEADER_TITLE_SPACING)
        self.header_log_spacing = header_config.get('header_log_spacing', self.HEADER_LOG_SPACING)
        self.header_top_padding = header_config.get('header_top_padding', self.HEADER_TOP_PADDING)
        self.header_bottom_padding = header_config.get('header_bottom_padding', self.HEADER_BOTTOM_PADDING)

        # Handle template parameter
        if isinstance(template, str):
            # Get template from manager
            if well.parent_manager is None:
                raise ValueError(
                    f"Cannot use template name '{template}': well has no parent manager. "
                    "Pass a Template object or dict instead."
                )
            if not hasattr(well.parent_manager, '_templates'):
                raise ValueError(
                    f"Template '{template}' not found in manager. "
                    f"Use manager.set_template('{template}', template_obj) first."
                )
            if template not in well.parent_manager._templates:
                available = list(well.parent_manager._templates.keys())
                raise ValueError(
                    f"Template '{template}' not found. Available templates: {available}"
                )
            self.template = well.parent_manager._templates[template]
        elif isinstance(template, dict):
            self.template = Template.from_dict(template)
        elif isinstance(template, Template):
            self.template = template
        elif template is None:
            # Create default template
            self.template = self._create_default_template()
        else:
            raise TypeError(
                f"template must be Template, dict, or str, got {type(template).__name__}"
            )

        # Initialize tops list (will be populated from template later)
        self.tops = []
        self.temp_tracks = []

        # Load tops from template (needed for tops-based depth range calculation)
        for tops_config in self.template.tops:
            self._add_tops_from_config(tops_config)

        # Determine depth range
        if tops is not None:
            # Calculate depth range from specified tops
            self.depth_range = self._calculate_depth_range_from_tops(tops)
        elif depth_range is None:
            # Use full depth range from first property
            if not well.properties:
                raise ValueError("Well has no properties to display")
            first_prop = well.get_property(well.properties[0].split('.')[0])
            self.depth_range = (float(first_prop.depth.min()), float(first_prop.depth.max()))
        else:
            self.depth_range = depth_range

        # Calculate figure size if not provided
        if figsize is None:
            n_tracks = len(self.template.tracks)
            total_width = sum(track.get("width", 1.0) for track in self.template.tracks)
            figsize = (max(2 * total_width, 8), 10)

        self.figsize = figsize
        self.fig = None
        self.axes = []

    def _create_default_template(self) -> Template:
        """Create a simple default template with all continuous properties."""
        template = Template("default")

        # Add first 3 continuous properties
        continuous_props = []
        for prop_name in self.well.properties:
            try:
                # Handle source.property format
                if '.' in prop_name:
                    prop = self.well.get_property(prop_name.split('.')[1])
                else:
                    prop = self.well.get_property(prop_name)

                if prop.type == 'continuous':
                    continuous_props.append(prop.name)
                    if len(continuous_props) >= 3:
                        break
            except Exception:
                continue

        # Add tracks for found properties
        for prop_name in continuous_props:
            template.add_track(
                track_type="continuous",
                logs=[{"name": prop_name, "color": "blue"}],
                title=prop_name
            )

        # Add depth track
        template.add_track(track_type="depth", width=0.3, title="Depth")

        return template

    def _calculate_depth_range_from_tops(self, tops_list: list[str]) -> tuple[float, float]:
        """
        Calculate depth range from a list of formation top names.

        The depth range is calculated as the min/max depths of the specified tops
        with 5% padding added. Minimum range is 50m.

        Parameters
        ----------
        tops_list : list[str]
            List of formation top names to include in depth range

        Returns
        -------
        tuple[float, float]
            Calculated depth range (top, bottom)

        Raises
        ------
        ValueError
            If no tops have been loaded or if specified tops are not found
        """
        # Check if any tops have been loaded
        if not self.tops:
            raise ValueError(
                "No formation tops have been loaded. Cannot calculate depth range from tops. "
                "Load tops using template.add_tops() or view.add_tops() before using the 'tops' parameter."
            )

        # Collect all tops data from all tops groups
        all_tops_data = {}  # depth -> name mapping
        for tops_group in self.tops:
            tops_data = tops_group['tops']
            all_tops_data.update(tops_data)

        # Find depths for specified tops
        tops_depths = []
        not_found = []
        for top_name in tops_list:
            found = False
            for depth, name in all_tops_data.items():
                if name == top_name:
                    tops_depths.append(depth)
                    found = True
                    break
            if not found:
                not_found.append(top_name)

        # Check if all tops were found
        if not_found:
            available_tops = list(set(all_tops_data.values()))
            raise ValueError(
                f"Formation tops not found: {not_found}. "
                f"Available tops: {available_tops}"
            )

        if not tops_depths:
            raise ValueError(
                f"No depths found for specified tops: {tops_list}"
            )

        # Calculate min and max depths
        min_depth = min(tops_depths)
        max_depth = max(tops_depths)

        # Calculate range and padding
        depth_range = max_depth - min_depth
        padding = depth_range * 0.05

        # Apply padding
        range_top = min_depth - padding
        range_bottom = max_depth + padding

        # Ensure minimum range of 50m
        calculated_range = range_bottom - range_top
        if calculated_range < 50.0:
            # Extend bottom to ensure 50m range
            range_bottom = range_top + 50.0

        return (float(range_top), float(range_bottom))

    def add_track(
        self,
        track_type: str = "continuous",
        logs: Optional[list[dict]] = None,
        fill: Optional[Union[dict, list[dict]]] = None,
        width: float = 1.0,
        title: Optional[str] = None,
        log_scale: bool = False
    ) -> 'WellView':
        """
        Add a temporary track to this view (not saved to template).

        This allows adding tracks to a specific view without modifying the
        underlying template. Temporary tracks are appended after template tracks.

        Parameters
        ----------
        track_type : {"continuous", "discrete", "depth"}, default "continuous"
            Type of track
        logs : list[dict], optional
            List of log configurations (same format as Template.add_track)
        fill : Union[dict, list[dict]], optional
            Fill configuration or list of fills
        width : float, default 1.0
            Relative width of track
        title : str, optional
            Track title
        log_scale : bool, default False
            Use logarithmic scale for the track

        Returns
        -------
        WellView
            Self for method chaining

        Examples
        --------
        >>> # Create view with template, then add temporary track
        >>> view = WellView(well, template=template)
        >>> view.add_track(
        ...     track_type="continuous",
        ...     logs=[{"name": "TEMP_LOG", "x_range": [0, 100], "color": "orange"}],
        ...     title="Temporary"
        ... )
        >>> view.show()

        Notes
        -----
        Temporary tracks are not saved to the template and only exist for this view.
        If you want to reuse tracks across multiple views, add them to the template instead.
        """
        # Normalize fill to list format
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
            "tops": None,
            "width": width,
            "title": title,
            "log_scale": log_scale
        }
        self.temp_tracks.append(track)
        return self

    def add_tops(
        self,
        property_name: Optional[str] = None,
        tops_dict: Optional[dict[float, str]] = None,
        colors: Optional[dict[float, str]] = None,
        styles: Optional[dict[float, str]] = None,
        thicknesses: Optional[dict[float, float]] = None,
        source: Optional[str] = None
    ) -> 'WellView':
        """
        Add temporary well tops to this view (not saved to template).

        Tops can be specified either from a discrete property in the well or
        as a dictionary mapping depths to formation names.

        Parameters
        ----------
        property_name : str, optional
            Name of discrete property in well containing tops data.
            The property should have depth values where formations start.
        tops_dict : dict[float, str], optional
            Dictionary mapping depth values to formation names.
            Example: {2850.0: 'Formation A', 2920.5: 'Formation B'}
        colors : dict[float, str], optional
            Optional color mapping for each depth. If provided, must have same keys
            as tops_dict or match values in the discrete property.
            Colors can be matplotlib color names, hex codes, or RGB tuples.
            If not provided and using a discrete property with color mapping,
            those colors will be used.
        styles : dict[float, str], optional
            Optional line style mapping for each depth or discrete value.
            Valid styles: 'solid', 'dashed', 'dotted', 'dashdot'
            If not provided and using a discrete property with style mapping,
            those styles will be used.
        thicknesses : dict[float, float], optional
            Optional line thickness mapping for each depth or discrete value.
            If not provided and using a discrete property with thickness mapping,
            those thicknesses will be used.
        source : str, optional
            Source name to get property from (if property_name is specified).
            Only needed if property exists in multiple sources.

        Returns
        -------
        WellView
            Self for method chaining

        Examples
        --------
        >>> # Add tops from discrete property
        >>> view = WellView(well, template=template)
        >>> view.add_tops(property_name='Zone')
        >>> view.show()
        >>>
        >>> # Add tops manually with custom colors
        >>> view.add_tops(
        ...     tops_dict={2850.0: 'Reservoir', 2920.5: 'Seal'},
        ...     colors={2850.0: 'yellow', 2920.5: 'gray'}
        ... )
        >>>
        >>> # Add tops from discrete property, overriding colors
        >>> view.add_tops(
        ...     property_name='Formation',
        ...     colors={0: 'red', 1: 'green', 2: 'blue'}  # Map discrete values to colors
        ... )

        Notes
        -----
        Tops are drawn as horizontal lines spanning all tracks (except depth track).
        Formation names are displayed at the right end of each line, floating above it.
        Temporary tops are not saved to the template.
        """
        tops_config = {
            'property_name': property_name,
            'tops_dict': tops_dict,
            'colors': colors,
            'styles': styles,
            'thicknesses': thicknesses,
            'source': source
        }
        self._add_tops_from_config(tops_config)
        return self

    def _add_tops_from_config(self, tops_config: dict) -> None:
        """
        Internal method to add tops from a configuration dict.

        This is used both for loading tops from templates and for adding temporary tops.
        """
        property_name = tops_config.get('property_name')
        tops_dict = tops_config.get('tops_dict')
        colors = tops_config.get('colors')
        styles = tops_config.get('styles')
        thicknesses = tops_config.get('thicknesses')
        source = tops_config.get('source')

        if property_name is None and tops_dict is None:
            raise ValueError("Must provide either 'property_name' or 'tops_dict'")

        if property_name is not None and tops_dict is not None:
            raise ValueError("Cannot specify both 'property_name' and 'tops_dict'")

        # Get tops data
        tops_data = {}  # depth -> formation name
        color_data = {}  # depth -> color
        style_data = {}  # depth -> line style
        thickness_data = {}  # depth -> line thickness

        if property_name is not None:
            # Load from discrete property
            try:
                prop = self.well.get_property(property_name, source=source)
            except KeyError:
                available = ', '.join(self.well.properties)
                raise ValueError(
                    f"Property '{property_name}' not found in well. "
                    f"Available properties: {available}"
                )

            if prop.type != 'discrete':
                raise ValueError(
                    f"Property '{property_name}' must be discrete type, got '{prop.type}'"
                )

            # Extract unique depths and their values
            valid_mask = ~np.isnan(prop.values)
            if not np.any(valid_mask):
                raise ValueError(f"Property '{property_name}' has no valid data")

            # Find where values change (formation boundaries)
            valid_depth = prop.depth[valid_mask]
            valid_values = prop.values[valid_mask]

            # Get boundaries where value changes
            boundaries = [0]  # Start with first point
            for i in range(1, len(valid_values)):
                if valid_values[i] != valid_values[i-1]:
                    boundaries.append(i)

            # Build tops dictionary
            for idx in boundaries:
                depth = float(valid_depth[idx])
                value = int(valid_values[idx])

                # Get label if available
                if prop.labels and value in prop.labels:
                    formation_name = prop.labels[value]
                else:
                    formation_name = f"Zone {value}"

                tops_data[depth] = formation_name

                # Get color if available (colors parameter overrides property colors)
                if colors is not None and value in colors:
                    color_data[depth] = colors[value]
                elif prop.colors and value in prop.colors:
                    color_data[depth] = prop.colors[value]

                # Get style if available (styles parameter overrides property styles)
                if styles is not None and value in styles:
                    style_data[depth] = styles[value]
                elif prop.styles and value in prop.styles:
                    style_data[depth] = prop.styles[value]

                # Get thickness if available (thicknesses parameter overrides property thicknesses)
                if thicknesses is not None and value in thicknesses:
                    thickness_data[depth] = thicknesses[value]
                elif prop.thicknesses and value in prop.thicknesses:
                    thickness_data[depth] = prop.thicknesses[value]

        else:
            # Use provided dictionary
            tops_data = tops_dict
            if colors is not None:
                color_data = colors
            if styles is not None:
                style_data = styles
            if thicknesses is not None:
                thickness_data = thicknesses

        # Store tops for rendering
        self.tops.append({
            'tops': tops_data,
            'colors': color_data if color_data else None,
            'styles': style_data if style_data else None,
            'thicknesses': thickness_data if thickness_data else None
        })

    def _get_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        """Get boolean mask for depth range."""
        return (depth >= self.depth_range[0]) & (depth <= self.depth_range[1])

    def _draw_cross_track_tops(self, all_tracks: list[dict]) -> None:
        """
        Draw well tops that span across all tracks (except depth track).

        Tops are drawn as horizontal lines with formation names displayed
        at the right end, floating above the line.

        Parameters
        ----------
        all_tracks : list[dict]
            Combined list of template tracks and temporary tracks
        """
        # Identify which tracks are depth tracks (skip those)
        non_depth_axes = []
        for ax, track in zip(self.axes, all_tracks):
            if track.get("type", "continuous") != "depth":
                non_depth_axes.append(ax)

        if not non_depth_axes:
            return  # No tracks to draw tops on

        # For each tops group
        for tops_group in self.tops:
            tops_data = tops_group['tops']
            colors_data = tops_group['colors']
            styles_data = tops_group['styles']
            thicknesses_data = tops_group['thicknesses']

            # Draw each top
            for depth, formation_name in tops_data.items():
                # Skip tops outside depth range
                if depth < self.depth_range[0] or depth > self.depth_range[1]:
                    continue

                # Get color for this top
                if colors_data and depth in colors_data:
                    color = colors_data[depth]
                else:
                    color = 'black'  # Default color

                # Get line style for this top
                if styles_data and depth in styles_data:
                    linestyle = styles_data[depth]
                else:
                    linestyle = 'solid'  # Default style

                # Get line thickness for this top
                if thicknesses_data and depth in thicknesses_data:
                    linewidth = thicknesses_data[depth]
                else:
                    linewidth = 1.5  # Default thickness

                # Draw line across all non-depth tracks
                for ax in non_depth_axes:
                    ax.axhline(y=depth, color=color, linestyle=linestyle, linewidth=linewidth, zorder=10)

                # Add label at the right end (on the rightmost non-depth track)
                rightmost_ax = non_depth_axes[-1]
                rightmost_ax.text(
                    1.0, depth,  # x=1.0 is at the right edge of the axes
                    formation_name,
                    transform=rightmost_ax.get_yaxis_transform(),  # x in axes coords, y in data coords
                    ha='right', va='bottom',
                    fontsize=8,
                    color='#272E39',  # Dark grey text color
                    bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', alpha=0.9),
                    zorder=11,
                    clip_on=False  # Allow label to extend beyond axes
                )

    def _plot_continuous_track(
        self,
        ax: plt.Axes,
        track: dict,
        depth: np.ndarray,
        mask: np.ndarray
    ) -> None:
        """
        Plot continuous log track with standard well log format.

        All curves are normalized to 0-1 scale for plotting, with original
        scale ranges shown in the track header.
        """
        logs = track.get("logs", [])
        fill = track.get("fill")
        track_log_scale = track.get("log_scale", False)  # Track-level log scale setting

        # Cache masked depth array (computed once, shared by all logs in this track)
        depth_masked = depth[mask]

        # Plot each log (all normalized to 0-1 scale)
        plotted_curves = {}
        scale_info = []  # Track scale information for header

        for log_config in logs:
            prop_name = log_config.get("name")
            if not prop_name:
                continue

            try:
                prop = self.well.get_property(prop_name)
            except Exception as e:
                warnings.warn(f"Could not get property '{prop_name}': {e}")
                continue

            # Get data (reuse cached depth_masked)
            values = prop.values[mask]

            # Get x_range for normalization
            if "x_range" in log_config:
                x_range = log_config["x_range"]
                x_min, x_max = x_range[0], x_range[1]

                # Determine scale for this log: use track setting unless overridden
                log_scale_override = log_config.get("scale")
                if log_scale_override == "log":
                    log_scale = True
                elif log_scale_override == "linear":
                    log_scale = False
                else:
                    # Use track-level setting
                    log_scale = track_log_scale

                # Normalize values to 0-1 based on x_range
                # x_range[0] always maps to 0 (left), x_range[1] always maps to 1 (right)
                if log_scale:
                    # Log scale normalization
                    # Clip values to avoid log(0) or log(negative)
                    values_clipped = np.clip(values, max(x_min, 1e-10), x_max)
                    normalized_values = (np.log10(values_clipped) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
                else:
                    # Linear scale normalization (default)
                    # This works for both normal [20, 150] and reversed [3.95, 1.95] scales
                    normalized_values = (values - x_range[0]) / (x_range[1] - x_range[0])

                scale_info.append({
                    'name': prop_name,
                    'min': x_min,
                    'max': x_max,
                    'color': log_config.get("color", "blue"),
                    'log_scale': log_scale
                })
            else:
                # No x_range specified, use values as-is
                normalized_values = values
                scale_info.append({
                    'name': prop_name,
                    'min': float(np.nanmin(values)),
                    'max': float(np.nanmax(values)),
                    'color': log_config.get("color", "blue"),
                    'log_scale': False
                })

            # Plot styling
            color = log_config.get("color", "blue")
            style_raw = log_config.get("style", "-")
            # Support both matplotlib codes and friendly names
            style_map = {
                "solid": "-",
                "dashed": "--",
                "dashdot": "-.",
                "dotted": ":",
                "none": ""
            }
            style = style_map.get(style_raw.lower() if isinstance(style_raw, str) else style_raw, style_raw)
            thickness = log_config.get("thickness", 1.0)
            alpha = log_config.get("alpha", 1.0)

            # Marker configuration
            marker = log_config.get("marker", None)
            marker_size = log_config.get("marker_size", 6)
            marker_outline_color = log_config.get("marker_outline_color", color)
            marker_fill = log_config.get("marker_fill", None)
            marker_interval = log_config.get("marker_interval", 1)

            # Convert friendly marker names to matplotlib codes
            marker_map = {
                "circle": "o",
                "square": "s",
                "diamond": "D",
                "triangle_up": "^",
                "triangle_down": "v",
                "triangle_left": "<",
                "triangle_right": ">",
                "plus": "+",
                "cross": "x",
                "star": "*",
                "pentagon": "p",
                "hexagon": "h",
                "point": ".",
                "pixel": ",",
                "vline": "|",
                "hline": "_"
            }
            if marker:
                marker = marker_map.get(marker.lower() if isinstance(marker, str) else marker, marker)

            # Plot normalized line (skip if style is "none" or empty)
            if style and style != "":
                ax.plot(normalized_values, depth_masked, color=color, linestyle=style,
                       linewidth=thickness, alpha=alpha, label=prop_name)

            # Plot markers if specified
            if marker:
                # Apply marker interval (only plot every nth marker)
                marker_mask = np.zeros(len(normalized_values), dtype=bool)
                marker_mask[::marker_interval] = True

                # Determine marker face color
                if marker_fill is not None:
                    markerfacecolor = marker_fill
                else:
                    markerfacecolor = 'none'  # Unfilled markers

                ax.plot(normalized_values[marker_mask], depth_masked[marker_mask],
                       marker=marker, markersize=marker_size,
                       markeredgecolor=marker_outline_color,
                       markerfacecolor=markerfacecolor,
                       linestyle='',  # No connecting line for markers
                       alpha=alpha)

            # Store both original and normalized values
            plotted_curves[prop_name] = (values, depth_masked)

        # Set x-axis to 0-1 for normalized plotting
        ax.set_xlim([0, 1])

        # Remove x-axis tick labels (normalized values are not meaningful to display)
        ax.tick_params(axis='x', labelbottom=False)

        # Handle fills (uses original values for boundary detection)
        # fill is now always a list (or None)
        if fill and plotted_curves:
            # Apply each fill in order
            for fill_config in fill:
                self._add_fill_normalized(ax, fill_config, plotted_curves, depth_masked, logs, track_log_scale)

        # Grid setup
        if track_log_scale and scale_info:
            # For log scale: disable standard x-grid, add custom log grid lines, keep y-grid
            ax.grid(True, alpha=0.3, axis='y')
            self._add_log_scale_grid(ax, scale_info[0]['min'], scale_info[0]['max'])
        else:
            # For linear scale: show standard grid (both x and y)
            ax.grid(True, alpha=0.3)

        # Add visual line indicators with outline box and track title for each curve in the header area
        title_text = track.get("title", "")
        self._add_curve_indicators(ax, scale_info, logs, title_text)

    def _add_log_scale_grid(self, ax: plt.Axes, x_min: float, x_max: float) -> None:
        """
        Add vertical grid lines for log scale.

        For a range like 1-100, shows: 1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100
        """
        # Generate log scale grid positions
        grid_values = []

        # Determine the order of magnitude range
        log_min = np.floor(np.log10(x_min))
        log_max = np.ceil(np.log10(x_max))

        # For each decade, add 1,2,3,4,5,6,7,8,9 * 10^n
        for decade_exp in range(int(log_min), int(log_max) + 1):
            decade = 10 ** decade_exp
            for multiplier in range(1, 10):
                value = multiplier * decade
                if x_min <= value <= x_max:
                    grid_values.append(value)

            # Also add the next power of 10 if it's within range
            next_decade = 10 ** (decade_exp + 1)
            if x_min <= next_decade <= x_max and next_decade not in grid_values:
                grid_values.append(next_decade)

        # Normalize grid values to 0-1 using log scale
        for value in grid_values:
            normalized_x = (np.log10(value) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
            ax.axvline(x=normalized_x, color='gray', linestyle='-', linewidth=0.5, alpha=0.4)

    def _add_curve_indicators(self, ax: plt.Axes, scale_info: list[dict], logs: list[dict], title: str) -> None:
        """
        Add visual line indicators with scale values in an outlined box.

        Format for each curve:
        LogName
        3.95 ---------------- 1.95

        All curves are enclosed in a box with outline, with track title above.
        Curves stack from bottom up; if they don't fit, they clip outside the box.
        """
        if not scale_info:
            return

        # Limit to first 4 logs for simplicity
        scale_info = scale_info[:4]

        # Use instance configuration for spacing
        box_top = self.header_box_top
        title_spacing = self.header_title_spacing
        log_spacing = self.header_log_spacing
        bottom_padding = self.header_bottom_padding

        # Box dimensions (fixed height between top and 1.0)
        box_bottom = 1.0  # Bottom aligns with plot area top
        box_height = box_top - box_bottom

        # Draw outline box around all indicators (full width to match plot area)
        box = Rectangle(
            (0, box_bottom), 1.0, box_height,
            transform=ax.get_xaxis_transform(),
            fill=False,
            edgecolor='black',
            linewidth=1.0,
            clip_on=False,
            zorder=9
        )
        ax.add_patch(box)

        # Add track title above the box
        if title:
            title_y = box_top + 0.005  # Small gap above box
            ax.text(0.5, title_y, title,
                   transform=ax.get_xaxis_transform(),
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   clip_on=False, zorder=11)

        # Draw each curve indicator (stacked from bottom up)
        for idx, info in enumerate(scale_info):
            # Find matching log config to get style
            log_config = next((log for log in logs if log.get("name") == info['name']), None)

            if log_config:
                color = log_config.get("color", "blue")
                style_raw = log_config.get("style", "-")
                # Support both matplotlib codes and friendly names
                style_map = {
                    "solid": "-",
                    "dashed": "--",
                    "dashdot": "-.",
                    "dotted": ":",
                    "none": ""
                }
                style = style_map.get(style_raw.lower() if isinstance(style_raw, str) else style_raw, style_raw)
                thickness = log_config.get("thickness", 1.0)

                # Marker configuration (for legend display)
                marker = log_config.get("marker", None)
                marker_size = log_config.get("marker_size", 6)
                marker_outline_color = log_config.get("marker_outline_color", color)
                marker_fill = log_config.get("marker_fill", None)

                # Convert friendly marker names to matplotlib codes
                marker_map = {
                    "circle": "o",
                    "square": "s",
                    "diamond": "D",
                    "triangle_up": "^",
                    "triangle_down": "v",
                    "triangle_left": "<",
                    "triangle_right": ">",
                    "plus": "+",
                    "cross": "x",
                    "star": "*",
                    "pentagon": "p",
                    "hexagon": "h",
                    "point": ".",
                    "pixel": ",",
                    "vline": "|",
                    "hline": "_"
                }
                if marker:
                    marker = marker_map.get(marker.lower() if isinstance(marker, str) else marker, marker)

                # Calculate y positions for this curve (stack from bottom up)
                # First curve (idx=0) starts from bottom of box + padding
                # Subsequent curves stack above
                base_y = box_bottom + bottom_padding + (idx * log_spacing)

                # Position for scale line (at base)
                scale_y = base_y
                # Position for log name (above the scale line)
                name_y = base_y + title_spacing

                # Add log name text centered
                ax.text(0.5, name_y, info['name'],
                       transform=ax.get_xaxis_transform(),
                       ha='center',
                       va='bottom',
                       fontsize=8,
                       fontweight='bold',
                       clip_on=False,
                       zorder=11)

                # Draw horizontal line between 0.15 and 0.85 (leaving room for scale values)
                # Only draw line if style is not "none" or empty
                if style and style != "":
                    ax.plot([0.15, 0.85], [scale_y, scale_y],
                           color=color,
                           linestyle=style,
                           linewidth=thickness,
                           transform=ax.get_xaxis_transform(),
                           clip_on=False,
                           zorder=10)

                # Draw markers in legend if specified
                # Place two markers: one halfway between edge and center, one halfway between center and other edge
                if marker:
                    # Determine marker face color
                    if marker_fill is not None:
                        markerfacecolor = marker_fill
                    else:
                        markerfacecolor = 'none'  # Unfilled markers

                    # First marker position: halfway between left edge (0.15) and center (0.5)
                    marker_x1 = 0.15 + (0.5 - 0.15) / 2  # = 0.325
                    # Second marker position: halfway between center (0.5) and right edge (0.85)
                    marker_x2 = 0.5 + (0.85 - 0.5) / 2   # = 0.675

                    ax.plot([marker_x1, marker_x2], [scale_y, scale_y],
                           marker=marker,
                           markersize=marker_size,
                           markeredgecolor=marker_outline_color,
                           markerfacecolor=markerfacecolor,
                           linestyle='',  # No connecting line
                           transform=ax.get_xaxis_transform(),
                           clip_on=False,
                           zorder=11)

                # Get scale values
                min_val = info['min']
                max_val = info['max']

                # Add min value text on left side of line (with white background)
                ax.text(0.05, scale_y, f"{min_val:.2f}",
                       transform=ax.get_xaxis_transform(),
                       ha='left',
                       va='center',
                       fontsize=7,
                       color=color,
                       clip_on=False,
                       zorder=11,
                       bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=1.0))

                # Add max value text on right side of line (with white background)
                ax.text(0.95, scale_y, f"{max_val:.2f}",
                       transform=ax.get_xaxis_transform(),
                       ha='right',
                       va='center',
                       fontsize=7,
                       color=color,
                       clip_on=False,
                       zorder=11,
                       bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3', alpha=1.0))

    def _add_discrete_legend(self, ax: plt.Axes, legend_info: list[dict], title: str) -> None:
        """
        Add discrete legend in an outlined box above the plot area.

        Parameters
        ----------
        ax : plt.Axes
            The axes to add the legend to
        legend_info : list[dict]
            List of dicts with 'label' and 'color' keys
        title : str
            Track title
        """
        if not legend_info:
            return

        # Limit to first 4 items for simplicity
        legend_info = legend_info[:4]

        # Use instance configuration (aligned with continuous tracks)
        box_top = self.header_box_top
        item_height = 0.022  # Height for each legend item (compact)
        bottom_padding = self.header_bottom_padding

        # Box dimensions (fixed height between top and 1.0)
        box_bottom = 1.0  # Bottom aligns with plot area top
        box_height = box_top - box_bottom

        # Draw outline box (full width to match plot area)
        box = Rectangle(
            (0, box_bottom), 1.0, box_height,
            transform=ax.get_xaxis_transform(),
            fill=False,
            edgecolor='black',
            linewidth=1.0,
            clip_on=False,
            zorder=9
        )
        ax.add_patch(box)

        # Add track title above the box
        if title:
            title_y = box_top + 0.005  # Small gap above box
            ax.text(0.5, title_y, title,
                   transform=ax.get_xaxis_transform(),
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   clip_on=False, zorder=11)

        # Draw each legend item (stacked from bottom up)
        for idx, item in enumerate(legend_info):
            # Calculate y position (stack from bottom up)
            # First item (idx=0) starts from bottom of box + padding
            item_y = box_bottom + bottom_padding + (idx * item_height) + (item_height / 2)

            # Draw colored rectangle as background (full width)
            color_rect = Rectangle(
                (0.05, item_y - item_height/2), 0.9, item_height * 0.85,
                transform=ax.get_xaxis_transform(),
                facecolor=item['color'],
                edgecolor='none',
                alpha=0.7,
                clip_on=False,
                zorder=10
            )
            ax.add_patch(color_rect)

            # Add label text (centered on colored background, black font)
            ax.text(0.5, item_y, item['label'],
                   transform=ax.get_xaxis_transform(),
                   ha='center',
                   va='center',
                   fontsize=8,
                   fontweight='bold',
                   color='black',
                   clip_on=False,
                   zorder=11)

    def _add_fill_normalized(
        self,
        ax: plt.Axes,
        fill: dict,
        plotted_curves: dict,
        depth: np.ndarray,
        logs: list[dict],
        track_log_scale: bool
    ) -> None:
        """
        Add fill between curves with normalized coordinates.

        This version handles fills when curves are normalized to 0-1 scale.

        Boundary specifications support simple string/number values or dict format:
        - "track_edge": Use track edge on that side (left=0.0, right=1.0)
        - "<curve_name>": Use curve values
        - <number>: Use fixed value
        - {"curve": "<name>"}: Use curve (dict format)
        - {"value": <num>}: Use fixed value (dict format)
        - {"track_edge": "left"|"right"}: Use track edge (dict format)
        """
        # Helper to normalize boundary spec to dict format
        def normalize_boundary_spec(spec, side):
            """Convert simple string/number spec to dict format."""
            if isinstance(spec, dict):
                return spec
            elif isinstance(spec, str):
                if spec == "track_edge":
                    return {"track_edge": side}
                else:
                    # Assume it's a curve name
                    return {"curve": spec}
            elif isinstance(spec, (int, float)):
                return {"value": spec}
            else:
                return {}

        # Normalize boundary specs (support both simple and dict formats)
        left_raw = fill.get("left", {})
        right_raw = fill.get("right", {})
        left_spec = normalize_boundary_spec(left_raw, "left")
        right_spec = normalize_boundary_spec(right_raw, "right")

        # Helper to normalize a value based on x_range
        def normalize_value(value, x_range, log_scale=False):
            if x_range is None:
                return value
            x_min, x_max = x_range[0], x_range[1]
            if log_scale:
                # Log scale normalization
                value_clipped = np.clip(value, max(x_min, 1e-10), x_max)
                return (np.log10(value_clipped) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
            else:
                # Linear scale normalization
                if x_min < x_max:
                    return (value - x_min) / (x_max - x_min)
                else:
                    return (value - x_max) / (x_min - x_max)

        # Helper to get x_range and log_scale for a curve
        def get_curve_info(curve_name):
            for log in logs:
                if log.get("name") == curve_name and "x_range" in log:
                    # Determine scale: check for override, otherwise use track setting
                    scale_override = log.get("scale")
                    if scale_override == "log":
                        log_scale = True
                    elif scale_override == "linear":
                        log_scale = False
                    else:
                        log_scale = track_log_scale
                    return log["x_range"], log_scale
            return None, False

        # Get left boundary (normalized)
        if "curve" in left_spec:
            curve_name = left_spec["curve"]
            if curve_name in plotted_curves:
                values, _ = plotted_curves[curve_name]
                x_range, log_scale = get_curve_info(curve_name)
                if x_range:
                    # Normalize using appropriate scale
                    if log_scale:
                        values_clipped = np.clip(values, max(x_range[0], 1e-10), x_range[1])
                        left_values = (np.log10(values_clipped) - np.log10(x_range[0])) / (np.log10(x_range[1]) - np.log10(x_range[0]))
                    else:
                        # x_range[0] maps to 0, x_range[1] maps to 1 (handles reversed scales)
                        left_values = (values - x_range[0]) / (x_range[1] - x_range[0])
                else:
                    left_values = values
            else:
                warnings.warn(f"Fill left curve '{curve_name}' not found")
                return
        elif "value" in left_spec:
            # For fixed values, need to know which curve's scale to use
            # Use first curve's x_range if available
            fixed_val = left_spec["value"]
            if logs and "x_range" in logs[0]:
                # Get scale from first log (with track default)
                scale_override = logs[0].get("scale")
                if scale_override == "log":
                    log_scale = True
                elif scale_override == "linear":
                    log_scale = False
                else:
                    log_scale = track_log_scale
                left_values = np.full_like(depth, normalize_value(fixed_val, logs[0]["x_range"], log_scale))
            else:
                left_values = np.full_like(depth, fixed_val)
        elif "track_edge" in left_spec:
            if left_spec["track_edge"] == "left":
                left_values = np.full_like(depth, 0.0)
            else:
                left_values = np.full_like(depth, 1.0)
        else:
            warnings.warn("Fill left boundary not properly specified")
            return

        # Get right boundary (normalized)
        if "curve" in right_spec:
            curve_name = right_spec["curve"]
            if curve_name in plotted_curves:
                values, _ = plotted_curves[curve_name]
                x_range, log_scale = get_curve_info(curve_name)
                if x_range:
                    # Normalize using appropriate scale
                    if log_scale:
                        values_clipped = np.clip(values, max(x_range[0], 1e-10), x_range[1])
                        right_values = (np.log10(values_clipped) - np.log10(x_range[0])) / (np.log10(x_range[1]) - np.log10(x_range[0]))
                    else:
                        # x_range[0] maps to 0, x_range[1] maps to 1 (handles reversed scales)
                        right_values = (values - x_range[0]) / (x_range[1] - x_range[0])
                else:
                    right_values = values
            else:
                warnings.warn(f"Fill right curve '{curve_name}' not found")
                return
        elif "value" in right_spec:
            fixed_val = right_spec["value"]
            if logs and "x_range" in logs[0]:
                # Get scale from first log (with track default)
                scale_override = logs[0].get("scale")
                if scale_override == "log":
                    log_scale = True
                elif scale_override == "linear":
                    log_scale = False
                else:
                    log_scale = track_log_scale
                right_values = np.full_like(depth, normalize_value(fixed_val, logs[0]["x_range"], log_scale))
            else:
                right_values = np.full_like(depth, fixed_val)
        elif "track_edge" in right_spec:
            if right_spec["track_edge"] == "left":
                right_values = np.full_like(depth, 0.0)
            else:
                right_values = np.full_like(depth, 1.0)
        else:
            warnings.warn("Fill right boundary not properly specified")
            return

        # Handle crossover - collapse fill where left is to the right of right
        # This prevents fill from appearing when curves cross over
        crossover_mask = left_values > right_values
        left_values = np.where(crossover_mask, right_values, left_values)

        # Apply fill
        fill_color = fill.get("color", "lightblue")
        fill_alpha = fill.get("alpha", 0.3)

        if "colormap" in fill:
            # Use colormap for fill - creates horizontal bands colored by curve value
            cmap_name = fill["colormap"]

            # Determine which curve drives the colormap (use original values, not normalized)
            colormap_curve_name = fill.get("colormap_curve")
            if colormap_curve_name:
                if colormap_curve_name in plotted_curves:
                    colormap_values, _ = plotted_curves[colormap_curve_name]
                else:
                    warnings.warn(f"Colormap curve '{colormap_curve_name}' not found, using boundary curves")
                    # Try left boundary curve first, then right boundary curve
                    if "curve" in left_spec and left_spec["curve"] in plotted_curves:
                        colormap_values, _ = plotted_curves[left_spec["curve"]]
                    elif "curve" in right_spec and right_spec["curve"] in plotted_curves:
                        colormap_values, _ = plotted_curves[right_spec["curve"]]
                    else:
                        warnings.warn("Cannot determine colormap values")
                        return
            else:
                # Default: use left boundary curve's original values if available,
                # otherwise try right boundary curve
                if "curve" in left_spec and left_spec["curve"] in plotted_curves:
                    colormap_values, _ = plotted_curves[left_spec["curve"]]
                elif "curve" in right_spec and right_spec["curve"] in plotted_curves:
                    colormap_values, _ = plotted_curves[right_spec["curve"]]
                else:
                    warnings.warn("Cannot determine colormap values (no curve specified for left or right)")
                    return

            # Get color range for normalization
            # Check if we have valid values
            valid_mask = ~np.isnan(colormap_values)
            if not np.any(valid_mask):
                warnings.warn(f"Colormap curve has no valid (non-NaN) values in the current depth range. Skipping fill.")
                return

            color_range = fill.get("color_range", [np.nanmin(colormap_values), np.nanmax(colormap_values)])
            norm = Normalize(vmin=color_range[0], vmax=color_range[1])
            cmap = plt.get_cmap(cmap_name)

            # Create horizontal bands - each depth interval gets a color based on the curve value
            # Use PolyCollection for performance (1000x faster than loop with fill_betweenx)
            n_intervals = len(depth) - 1

            # Compute color values for each interval (average of adjacent points)
            color_values = (colormap_values[:-1] + colormap_values[1:]) / 2
            colors = cmap(norm(color_values))

            # Create polygon vertices for each depth interval (using normalized coordinates)
            verts = []
            for i in range(n_intervals):
                verts.append([
                    (left_values[i], depth[i]),
                    (right_values[i], depth[i]),
                    (right_values[i+1], depth[i+1]),
                    (left_values[i+1], depth[i+1])
                ])

            # Create PolyCollection with all polygons at once
            poly_collection = PolyCollection(
                verts,
                facecolors=colors,
                alpha=fill_alpha,
                edgecolors='none',
                linewidths=0
            )
            ax.add_collection(poly_collection)
        else:
            # Simple solid color fill
            ax.fill_betweenx(depth, left_values, right_values,
                            color=fill_color, alpha=fill_alpha)

    def _add_fill(
        self,
        ax: plt.Axes,
        fill: dict,
        plotted_curves: dict,
        depth: np.ndarray
    ) -> None:
        """
        Add fill between curves or values.

        For colormap fills, creates horizontal bands where each depth interval
        is colored based on a curve value (e.g., GR from 20-150 maps to colormap).

        Boundary specifications support simple string/number values or dict format:
        - "track_edge": Use track edge on that side
        - "<curve_name>": Use curve values
        - <number>: Use fixed value
        - {"curve": "<name>"}: Use curve (dict format)
        - {"value": <num>}: Use fixed value (dict format)
        - {"track_edge": "left"|"right"}: Use track edge (dict format)
        """
        # Helper to normalize boundary spec to dict format
        def normalize_boundary_spec(spec, side):
            """Convert simple string/number spec to dict format."""
            if isinstance(spec, dict):
                return spec
            elif isinstance(spec, str):
                if spec == "track_edge":
                    return {"track_edge": side}
                else:
                    # Assume it's a curve name
                    return {"curve": spec}
            elif isinstance(spec, (int, float)):
                return {"value": spec}
            else:
                return {}

        # Normalize boundary specs (support both simple and dict formats)
        left_raw = fill.get("left", {})
        right_raw = fill.get("right", {})
        left_spec = normalize_boundary_spec(left_raw, "left")
        right_spec = normalize_boundary_spec(right_raw, "right")

        # Get left boundary
        if "curve" in left_spec:
            curve_name = left_spec["curve"]
            if curve_name in plotted_curves:
                left_values, _ = plotted_curves[curve_name]
            else:
                warnings.warn(f"Fill left curve '{curve_name}' not found")
                return
        elif "value" in left_spec:
            left_values = np.full_like(depth, left_spec["value"])
        elif "track_edge" in left_spec:
            left_values = np.full_like(depth, ax.get_xlim()[0])
        else:
            warnings.warn("Fill left boundary not properly specified")
            return

        # Get right boundary
        if "curve" in right_spec:
            curve_name = right_spec["curve"]
            if curve_name in plotted_curves:
                right_values, _ = plotted_curves[curve_name]
            else:
                warnings.warn(f"Fill right curve '{curve_name}' not found")
                return
        elif "value" in right_spec:
            right_values = np.full_like(depth, right_spec["value"])
        elif "track_edge" in right_spec:
            right_values = np.full_like(depth, ax.get_xlim()[1])
        else:
            warnings.warn("Fill right boundary not properly specified")
            return

        # Handle crossover - collapse fill where left is to the right of right
        # This prevents fill from appearing when curves cross over
        crossover_mask = left_values > right_values
        left_values = np.where(crossover_mask, right_values, left_values)

        # Apply fill
        fill_color = fill.get("color", "lightblue")
        fill_alpha = fill.get("alpha", 0.3)

        if "colormap" in fill:
            # Use colormap for fill - creates horizontal bands colored by curve value
            cmap_name = fill["colormap"]

            # Determine which curve drives the colormap
            # Can be explicitly specified, or defaults to boundary curves
            colormap_curve_name = fill.get("colormap_curve")
            if colormap_curve_name:
                if colormap_curve_name in plotted_curves:
                    colormap_values, _ = plotted_curves[colormap_curve_name]
                else:
                    warnings.warn(f"Colormap curve '{colormap_curve_name}' not found, using boundary curves")
                    # Try left boundary curve first, then right boundary curve
                    if "curve" in left_spec and left_spec["curve"] in plotted_curves:
                        colormap_values, _ = plotted_curves[left_spec["curve"]]
                    elif "curve" in right_spec and right_spec["curve"] in plotted_curves:
                        colormap_values, _ = plotted_curves[right_spec["curve"]]
                    else:
                        warnings.warn("Cannot determine colormap values")
                        return
            else:
                # Default: use left boundary curve if available, otherwise right boundary curve
                if "curve" in left_spec and left_spec["curve"] in plotted_curves:
                    colormap_values, _ = plotted_curves[left_spec["curve"]]
                elif "curve" in right_spec and right_spec["curve"] in plotted_curves:
                    colormap_values, _ = plotted_curves[right_spec["curve"]]
                else:
                    warnings.warn("Cannot determine colormap values (no curve specified for left or right)")
                    return

            # Get color range for normalization
            # Check if we have valid values
            valid_mask = ~np.isnan(colormap_values)
            if not np.any(valid_mask):
                warnings.warn(f"Colormap curve has no valid (non-NaN) values in the current depth range. Skipping fill.")
                return

            color_range = fill.get("color_range", [np.nanmin(colormap_values), np.nanmax(colormap_values)])
            norm = Normalize(vmin=color_range[0], vmax=color_range[1])
            cmap = plt.get_cmap(cmap_name)

            # Create horizontal bands - each depth interval gets a color based on the curve value
            # Use PolyCollection for performance (1000x faster than loop with fill_betweenx)
            n_intervals = len(depth) - 1

            # Compute color values for each interval (average of adjacent points)
            color_values = (colormap_values[:-1] + colormap_values[1:]) / 2
            colors = cmap(norm(color_values))

            # Create polygon vertices for each depth interval
            # Each polygon is a quad: [(left, depth_i), (right, depth_i), (right, depth_i+1), (left, depth_i+1)]
            verts = []
            for i in range(n_intervals):
                verts.append([
                    (left_values[i], depth[i]),
                    (right_values[i], depth[i]),
                    (right_values[i+1], depth[i+1]),
                    (left_values[i+1], depth[i+1])
                ])

            # Create PolyCollection with all polygons at once
            poly_collection = PolyCollection(
                verts,
                facecolors=colors,
                alpha=fill_alpha,
                edgecolors='none',
                linewidths=0
            )
            ax.add_collection(poly_collection)
        else:
            # Simple solid color fill
            ax.fill_betweenx(depth, left_values, right_values,
                            color=fill_color, alpha=fill_alpha)

    def _plot_discrete_track(
        self,
        ax: plt.Axes,
        track: dict,
        depth: np.ndarray,
        mask: np.ndarray
    ) -> None:
        """Plot discrete/categorical track."""
        logs = track.get("logs", [])

        if not logs:
            return

        # Get property (only first one for discrete tracks)
        prop_name = logs[0].get("name")
        if not prop_name:
            return

        try:
            prop = self.well.get_property(prop_name)
        except Exception as e:
            warnings.warn(f"Could not get property '{prop_name}': {e}")
            return

        # Cache masked depth array
        depth_masked = depth[mask]

        # Get data and round to integers (discrete values must be integers)
        # Keep as float to preserve NaN values
        raw_values = prop.values[mask]
        values = np.where(np.isnan(raw_values), np.nan, np.round(raw_values))

        # Get unique values and create color mapping (NaN values filtered out)
        unique_vals = np.unique(values[~np.isnan(values)]).astype(int)

        # Create color mapping - check property colors first, then fall back to defaults
        if prop.colors:
            # Use property's custom colors for values that have them
            color_map = {}
            for val in unique_vals:
                if val in prop.colors:
                    color_map[val] = prop.colors[val]
                else:
                    # Fall back to default color if this value doesn't have a custom color
                    default_idx = list(unique_vals).index(val) % len(DEFAULT_COLORS)
                    color_map[val] = DEFAULT_COLORS[default_idx]
        else:
            # No custom colors defined, use defaults
            colors = DEFAULT_COLORS[:len(unique_vals)]
            color_map = dict(zip(unique_vals, colors))

        # Plot as colored bars
        for val in unique_vals:
            val_mask = (values == val)
            # Create continuous segments
            segments = []
            start_idx = None

            for i, is_val in enumerate(val_mask):
                if is_val and start_idx is None:
                    start_idx = i
                elif not is_val and start_idx is not None:
                    segments.append((start_idx, i-1))
                    start_idx = None

            if start_idx is not None:
                segments.append((start_idx, len(val_mask)-1))

            # Draw segments (no label - will be in header)
            for start, end in segments:
                ax.fill_betweenx(
                    depth_masked[start:end+1],
                    0, 1,
                    color=color_map[val],
                    alpha=0.7
                )

        # Configure axes
        ax.set_xlim([0, 1])
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

        # Add legend header above plot area (similar to continuous tracks)
        title_text = track.get("title", "")
        legend_info = []
        for val in unique_vals:
            label = prop.labels.get(int(val), str(int(val))) if prop.labels else str(int(val))
            legend_info.append({
                'label': label,
                'color': color_map[val]
            })
        self._add_discrete_legend(ax, legend_info, title_text)

    def _plot_depth_track(
        self,
        ax: plt.Axes,
        track: dict,
        depth: np.ndarray,
        mask: np.ndarray
    ) -> None:
        """Plot depth axis track."""
        depth_masked = depth[mask]

        # Plot depth as vertical line
        ax.plot([0.5, 0.5], [depth_masked.min(), depth_masked.max()],
                'k-', linewidth=0.5)

        # Configure
        ax.set_xlim([0, 1])
        ax.set_xticks([])
        ax.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        if track.get("title"):
            ax.set_title(track["title"], fontsize=10, fontweight='bold')

    def _add_tops(
        self,
        ax: plt.Axes,
        tops_config: dict,
        depth: np.ndarray,
        mask: np.ndarray
    ) -> None:
        """Add formation tops markers to track."""
        prop_name = tops_config.get("name")
        if not prop_name:
            return

        try:
            tops_prop = self.well.get_property(prop_name)
        except Exception as e:
            warnings.warn(f"Could not get tops property '{prop_name}': {e}")
            return

        # Cache masked depth array
        tops_depth = depth[mask]

        # Get tops values
        tops_values = tops_prop.values[mask]

        # Find discrete boundaries (changes in value)
        changes = np.where(np.diff(tops_values) != 0)[0]

        # Plot styling
        line_style = tops_config.get("line_style", "--")
        line_width = tops_config.get("line_width", 1.0)
        title_size = tops_config.get("title_size", 8)
        title_weight = tops_config.get("title_weight", "normal")
        title_orientation = tops_config.get("title_orientation", "right")
        line_offset = tops_config.get("line_offset", 0.0)

        # Draw lines and labels at changes
        xlim = ax.get_xlim()
        x_range = xlim[1] - xlim[0]

        for idx in changes:
            depth_val = tops_depth[idx]

            # Draw horizontal line
            ax.axhline(y=depth_val, color='black', linestyle=line_style,
                      linewidth=line_width, alpha=0.7)

            # Add label if tops have labels
            if tops_prop.labels:
                next_val = tops_values[idx + 1] if idx + 1 < len(tops_values) else tops_values[idx]
                label = tops_prop.labels.get(int(next_val), "")

                if label:
                    # Determine text position
                    if title_orientation == "left":
                        x_pos = xlim[0] + 0.05 * x_range + line_offset
                        ha = 'left'
                    elif title_orientation == "center":
                        x_pos = (xlim[0] + xlim[1]) / 2 + line_offset
                        ha = 'center'
                    else:  # right
                        x_pos = xlim[1] - 0.05 * x_range + line_offset
                        ha = 'right'

                    ax.text(x_pos, depth_val, label,
                           fontsize=title_size, fontweight=title_weight,
                           ha=ha, va='bottom')

    def plot(self) -> None:
        """
        Create the well log plot.

        This method generates the matplotlib figure with all configured tracks.
        Call show() or save() after this to display or save the figure.

        Examples
        --------
        >>> view = WellView(well, template=template)
        >>> view.plot()
        >>> view.show()
        """
        # Get reference depth grid
        first_prop_name = self.well.properties[0].split('.')[0]
        first_prop = self.well.get_property(first_prop_name)
        depth = first_prop.depth
        mask = self._get_depth_mask(depth)

        # Combine template tracks with temporary tracks
        all_tracks = self.template.tracks + self.temp_tracks

        # Create figure with subplots
        n_tracks = len(all_tracks)
        widths = [track.get("width", 1.0) for track in all_tracks]

        self.fig, self.axes = plt.subplots(
            1, n_tracks,
            figsize=self.figsize,
            dpi=self.dpi,
            gridspec_kw={'width_ratios': widths, 'wspace': 0},
            sharey=True
        )

        # Handle single track case
        if n_tracks == 1:
            self.axes = [self.axes]

        # Plot each track
        for ax, track in zip(self.axes, all_tracks):
            track_type = track.get("type", "continuous")

            if track_type == "continuous":
                self._plot_continuous_track(ax, track, depth, mask)
            elif track_type == "discrete":
                self._plot_discrete_track(ax, track, depth, mask)
            elif track_type == "depth":
                self._plot_depth_track(ax, track, depth, mask)

            # Add tops if configured
            tops_config = track.get("tops")
            if tops_config:
                self._add_tops(ax, tops_config, depth, mask)

            # Remove y-labels for all but first track
            if ax != self.axes[0]:
                ax.set_ylabel('')

        # Draw cross-track tops (span all tracks except depth track)
        if self.tops:
            self._draw_cross_track_tops(all_tracks)

        # Invert y-axis once for all tracks (depth increases downward)
        # Since sharey=True, this applies to all axes
        self.axes[0].invert_yaxis()

        # Set exact y-axis limits to match depth_range without padding
        # Since sharey=True, this applies to all axes
        self.axes[0].set_ylim(self.depth_range[1], self.depth_range[0])
        self.axes[0].margins(y=0)

        # Set main title
        self.fig.suptitle(f"Well: {self.well.name}", fontsize=12, fontweight='bold', y=0.995)

        # Apply tight layout (suppress warnings from PolyCollection incompatibility)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',
                                      message='.*not compatible with tight_layout.*',
                                      category=UserWarning)
                plt.tight_layout()
        except Exception:
            # If tight_layout fails, continue without it
            pass

    def show(self) -> None:
        """
        Display the well log plot in Jupyter notebook.

        This will render the plot inline in Jupyter Lab/Notebook.

        Examples
        --------
        >>> view = WellView(well, template=template)
        >>> view.show()
        """
        if self.fig is None:
            self.plot()
        plt.show()

    def save(
        self,
        filepath: Union[str, Path],
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save the well log plot to file.

        Parameters
        ----------
        filepath : Union[str, Path]
            Output file path (format determined by extension: .png, .pdf, .svg, etc.)
        dpi : int, optional
            Resolution for raster formats. If None, uses figure's dpi.
        bbox_inches : str, default 'tight'
            Bounding box specification

        Examples
        --------
        >>> view = WellView(well, template=template)
        >>> view.save("well_log.png", dpi=300)
        >>> view.save("well_log.pdf")
        """
        if self.fig is None:
            self.plot()

        if dpi is None:
            dpi = self.dpi

        self.fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)

    def close(self) -> None:
        """
        Close the matplotlib figure and free memory.

        Examples
        --------
        >>> view = WellView(well, template=template)
        >>> view.show()
        >>> view.close()
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = []

    def __repr__(self) -> str:
        """String representation."""
        total_tracks = len(self.template.tracks) + len(self.temp_tracks)
        track_info = f"tracks={total_tracks}"
        if self.temp_tracks:
            track_info += f" ({len(self.template.tracks)} template + {len(self.temp_tracks)} temp)"

        return (
            f"WellView(well='{self.well.name}', "
            f"depth_range={self.depth_range}, "
            f"{track_info})"
        )
