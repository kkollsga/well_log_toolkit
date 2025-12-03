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
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

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

    def add_track(
        self,
        track_type: str = "continuous",
        logs: Optional[list[dict]] = None,
        fill: Optional[dict] = None,
        tops: Optional[dict] = None,
        width: float = 1.0,
        title: Optional[str] = None
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
            - color (str): Line color
            - style (str): Line style ("-", "--", "-.", ":")
            - thickness (float): Line width
            - alpha (float): Transparency (0-1)
        fill : dict, optional
            Fill configuration with keys:
            - left (dict): Left boundary {"curve": name} or {"value": float} or {"track_edge": "left"}
            - right (dict): Right boundary (same format as left)
            - color (str): Fill color name (for solid fills)
            - colormap (str): Matplotlib colormap name (e.g., "viridis", "inferno")
              Creates horizontal bands where depth intervals are colored based on curve values
            - colormap_curve (str): Curve name to use for colormap values (defaults to left boundary curve)
            - color_range (list): [min, max] values for colormap normalization
            - alpha (float): Transparency (0-1)
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
        ...         "style": "-",
        ...         "thickness": 1.0
        ...     }],
        ...     title="Gamma Ray"
        ... )
        >>>
        >>> # Add porosity track with fill
        >>> template.add_track(
        ...     track_type="continuous",
        ...     logs=[{
        ...         "name": "PHIE",
        ...         "x_range": [0.45, 0],
        ...         "color": "blue"
        ...     }],
        ...     fill={
        ...         "left": {"curve": "PHIE"},
        ...         "right": {"value": 0},
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
        ...         "left": {"track_edge": "left"},
        ...         "right": {"curve": "GR"},
        ...         "colormap": "viridis",
        ...         "color_range": [20, 150],
        ...         "alpha": 0.7
        ...     },
        ...     title="Gamma Ray"
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
        track = {
            "type": track_type,
            "logs": logs or [],
            "fill": fill,
            "tops": tops,
            "width": width,
            "title": title
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
            "tracks": self.tracks
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
        return cls(name=data.get("name", "loaded"), tracks=data.get("tracks", []))

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
        dict_keys(['name', 'tracks'])
        """
        return {
            "name": self.name,
            "tracks": self.tracks
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
        >>> config = {"name": "test", "tracks": [...]}
        >>> template = Template.from_dict(config)
        """
        return cls(name=data.get("name", "unnamed"), tracks=data.get("tracks", []))

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
    >>> # Display well log
    >>> view = WellView(well, depth_range=[2800, 3000], template=template)
    >>> view.show()
    >>>
    >>> # Or use template from manager
    >>> manager.set_template("reservoir", template)
    >>> view2 = WellView(well, depth_range=[3000, 3200], template="reservoir")
    >>> view2.show()
    >>>
    >>> # Save figure
    >>> view.save("well_log.png", dpi=300)
    """

    def __init__(
        self,
        well: 'Well',
        depth_range: Optional[tuple[float, float]] = None,
        template: Optional[Union[Template, dict, str]] = None,
        figsize: Optional[tuple[float, float]] = None,
        dpi: int = 100
    ):
        """Initialize WellView."""
        self.well = well
        self.dpi = dpi

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

        # Determine depth range
        if depth_range is None:
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

    def _get_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        """Get boolean mask for depth range."""
        return (depth >= self.depth_range[0]) & (depth <= self.depth_range[1])

    def _plot_continuous_track(
        self,
        ax: plt.Axes,
        track: dict,
        depth: np.ndarray,
        mask: np.ndarray
    ) -> None:
        """Plot continuous log track."""
        logs = track.get("logs", [])
        fill = track.get("fill")

        # Determine x-axis range (use first log's range or auto)
        if logs and "x_range" in logs[0]:
            x_range = logs[0]["x_range"]
            ax.set_xlim(x_range)

        # Cache masked depth array (computed once, shared by all logs in this track)
        depth_masked = depth[mask]

        # Plot each log
        plotted_curves = {}
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

            # Plot styling
            color = log_config.get("color", "blue")
            style = log_config.get("style", "-")
            thickness = log_config.get("thickness", 1.0)
            alpha = log_config.get("alpha", 1.0)

            # Plot line
            line, = ax.plot(values, depth_masked, color=color, linestyle=style,
                           linewidth=thickness, alpha=alpha, label=prop_name)

            plotted_curves[prop_name] = (values, depth_masked)

        # Handle fill
        if fill and plotted_curves:
            self._add_fill(ax, fill, plotted_curves, depth_masked)

        # Invert y-axis (depth increases downward)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # Set title
        if track.get("title"):
            ax.set_title(track["title"], fontsize=10, fontweight='bold')

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
        """
        left_spec = fill.get("left", {})
        right_spec = fill.get("right", {})

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

        # Apply fill
        fill_color = fill.get("color", "lightblue")
        fill_alpha = fill.get("alpha", 0.3)

        if "colormap" in fill:
            # Use colormap for fill - creates horizontal bands colored by curve value
            cmap_name = fill["colormap"]

            # Determine which curve drives the colormap
            # Can be explicitly specified, or defaults to left boundary if it's a curve
            colormap_curve_name = fill.get("colormap_curve")
            if colormap_curve_name:
                if colormap_curve_name in plotted_curves:
                    colormap_values, _ = plotted_curves[colormap_curve_name]
                else:
                    warnings.warn(f"Colormap curve '{colormap_curve_name}' not found, using left boundary")
                    colormap_values = left_values
            else:
                # Default: use left boundary values for colormapping
                colormap_values = left_values

            # Get color range for normalization
            color_range = fill.get("color_range", [colormap_values.min(), colormap_values.max()])
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

        # Get data
        values = prop.values[mask]

        # Get unique values and create color mapping
        unique_vals = np.unique(values[~np.isnan(values)])
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

            # Draw segments
            label = prop.labels.get(int(val), str(int(val))) if prop.labels else str(int(val))
            for start, end in segments:
                ax.fill_betweenx(
                    depth_masked[start:end+1],
                    0, 1,
                    color=color_map[val],
                    alpha=0.7,
                    label=label if start == segments[0][0] else None
                )

        # Configure axes
        ax.set_xlim([0, 1])
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.legend(loc='upper right', fontsize=8)

        if track.get("title"):
            ax.set_title(track["title"], fontsize=10, fontweight='bold')

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
        ax.invert_yaxis()
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

        # Create figure with subplots
        n_tracks = len(self.template.tracks)
        widths = [track.get("width", 1.0) for track in self.template.tracks]

        self.fig, self.axes = plt.subplots(
            1, n_tracks,
            figsize=self.figsize,
            dpi=self.dpi,
            gridspec_kw={'width_ratios': widths, 'wspace': 0.05},
            sharey=True
        )

        # Handle single track case
        if n_tracks == 1:
            self.axes = [self.axes]

        # Plot each track
        for ax, track in zip(self.axes, self.template.tracks):
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

        # Set main title
        self.fig.suptitle(f"Well: {self.well.name}", fontsize=12, fontweight='bold', y=0.995)

        plt.tight_layout()

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
        return (
            f"WellView(well='{self.well.name}', "
            f"depth_range={self.depth_range}, "
            f"tracks={len(self.template.tracks)})"
        )
