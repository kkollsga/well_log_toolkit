"""Crossplot class for well log cross-plotting and analysis."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..exceptions import PropertyNotFoundError
from . import DEFAULT_COLORS, _create_regression

if TYPE_CHECKING:
    from ..core.well import Well


class Crossplot:
    """
    Create beautiful, modern crossplots for well log analysis.

    Supports single and multi-well crossplots with extensive customization options
    including color mapping, size mapping, shape mapping, regression analysis, and
    multi-layer plotting for combining different data types (e.g., Core vs Sidewall).

    Parameters
    ----------
    wells : Well or list of Well
        Single well or list of wells to plot
    x : str, optional
        Name of property for x-axis. Required if layers is not provided.
    y : str, optional
        Name of property for y-axis. Required if layers is not provided.
    layers : dict[str, list[str]], optional
        Dictionary mapping layer labels to [x_property, y_property] lists.
        Use this to combine multiple property pairs in a single plot.
        Example: {"Core": ["CorePor", "CorePerm"], "Sidewall": ["SWPor", "SWPerm"]}
        When using layers, shape defaults to "label" and color defaults to "well" for
        easy visualization of both layer types and wells. Default: None
    shape : str, optional
        Property name for shape mapping. Use "well" to map shapes by well name,
        or "label" (when using layers) to map shapes by layer type.
        Default: "well" for multi-well plots, "label" when layers provided, None otherwise
    color : str, optional
        Property name for color mapping. Use "depth" to color by depth,
        "well" to color by well, or "label" (when using layers) to color by layer type.
        Default: "well" when layers provided, None otherwise
    size : str, optional
        Property name for size mapping, or "label" (when using layers) to
        size by layer type.
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
        Marker style. Default: "o"
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
        Show legend when using shape/well mapping. Default: True
    show_regression_legend : bool, optional
        Show separate legend for regression lines in lower right. Default: True
    show_regression_equation : bool, optional
        Show equations in regression legend. Default: True
    show_regression_r2 : bool, optional
        Show R-squared values in regression legend. Default: True
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
        Regression type to apply separately for each group (well or shape). Creates
        separate regression lines for each well or shape category. Accepts string or dict.
        Default: None
    regression_by_color_and_shape : str or dict, optional
        Regression type to apply separately for each combination of color AND shape groups.
        Creates separate regression lines for each (color, shape) combination. This is useful
        for analyzing how the relationship changes across both dimensions simultaneously
        (e.g., each well in each formation, each layer in each zone). Accepts string or dict.
        Default: None
    regression_by_shape_and_color : str or dict, optional
        Alias for regression_by_color_and_shape. Provided for convenience - both parameters
        do exactly the same thing. Use whichever order feels more natural.
        Default: None

    Examples
    --------
    Basic crossplot from a single well:

    >>> plot = well.Crossplot(x="RHOB", y="NPHI")
    >>> plot.show()

    Multi-well crossplot with color and size mapping:

    >>> plot = manager.Crossplot(
    ...     x="PHIE",
    ...     y="SW",
    ...     color="depth",
    ...     size="PERM",
    ...     shape="well",
    ...     colortemplate="viridis"
    ... )
    >>> plot.show()

    With regression analysis (string format):

    >>> plot = well.Crossplot(x="RHOB", y="NPHI", regression="linear")
    >>> plot.show()

    With regression analysis (dict format for custom styling):

    >>> plot = well.Crossplot(
    ...     x="RHOB", y="NPHI",
    ...     regression={"type": "linear", "line_color": "red", "line_width": 3}
    ... )
    >>> plot.show()

    Multi-well with group-specific regressions:

    >>> plot = manager.Crossplot(
    ...     x="PHIE", y="SW",
    ...     shape="well",
    ...     regression_by_group={"type": "linear", "line_style": "--"}
    ... )
    >>> plot.show()

    Combining multiple data types with layers (Core + Sidewall):

    >>> plot = manager.Crossplot(
    ...     layers={
    ...         "Core": ["CorePor_obds", "CorePerm_obds"],
    ...         "Sidewall": ["SidewallPor_ob", "SidewallPerm_ob"]
    ...     },
    ...     y_log=True
    ...     # shape defaults to "label" - different shapes for Core vs Sidewall
    ...     # color defaults to "well" - different colors for each well
    ... )
    >>> plot.show()

    Using add_layer method with method chaining:

    >>> manager.Crossplot(y_log=True) \\
    ...     .add_layer("CorePor_obds", "CorePerm_obds", label="Core") \\
    ...     .add_layer("SidewallPor_ob", "SidewallPerm_ob", label="Sidewall") \\
    ...     .show()
    ...     # Automatically uses shape="label" and color="well"

    Layers with regression by color (single trend per well):

    >>> plot = manager.Crossplot(
    ...     layers={
    ...         "Core": ["CorePor_obds", "CorePerm_obds"],
    ...         "Sidewall": ["SidewallPor_ob", "SidewallPerm_ob"]
    ...     },
    ...     regression_by_color="linear"  # One trend per well (combining both data types)
    ...     # Defaults: shape="label" (different shapes), color="well" (different colors)
    ... )
    >>> plot.show()

    Access regression objects:

    >>> linear_regs = plot.regression("linear")
    >>> for name, reg in linear_regs.items():
    ...     print(f"{name}: {reg.equation()}, R\u00b2={reg.r_squared:.3f}")
    """

    def __init__(
        self,
        wells: Well | list[Well],
        x: str | None = None,
        y: str | None = None,
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
        grid_alpha: float = 0.7,
        depth_range: tuple[float, float] | None = None,
        show_colorbar: bool = True,
        show_legend: bool = True,
        show_regression_legend: bool = True,
        show_regression_equation: bool = True,
        show_regression_r2: bool = True,
        regression: str | dict | None = None,
        regression_by_color: str | dict | None = None,
        regression_by_group: str | dict | None = None,
        regression_by_color_and_shape: str | dict | None = None,
        regression_by_shape_and_color: str | dict | None = None,
    ):
        # Store wells as list
        if not isinstance(wells, list):
            self.wells = [wells]
        else:
            self.wells = wells

        # Validate input: either (x, y) or layers must be provided
        if layers is None and (x is None or y is None):
            raise ValueError("Either (x, y) or layers must be provided")

        # Initialize layer tracking
        self._layers = []

        # If layers dict provided, convert to internal format
        if layers is not None:
            for label, props in layers.items():
                if len(props) != 2:
                    raise ValueError(f"Layer '{label}' must have exactly 2 properties [x, y]")
                self._layers.append({"label": label, "x": props[0], "y": props[1]})
        # If x and y provided, create a default layer
        elif x is not None and y is not None:
            self._layers.append({"label": None, "x": x, "y": y})

        # Store parameters
        self.x = x
        self.y = y
        # Default shape to "label" when layers are provided
        if shape is None and layers is not None:
            self.shape = "label"
        else:
            self.shape = shape
        # Default color to "well" when layers are provided (for multi-well visualization)
        if color is None and layers is not None and len(self.wells) > 1:
            self.color = "well"
        else:
            self.color = color
        self.size = size
        self.colortemplate = colortemplate
        self.color_range = color_range
        self.size_range = size_range
        self.title = title
        # Set axis labels - use provided labels, or property names, or generic labels for layers
        if xlabel:
            self.xlabel = xlabel
        elif x:
            self.xlabel = x
        else:
            self.xlabel = "X"

        if ylabel:
            self.ylabel = ylabel
        elif y:
            self.ylabel = y
        else:
            self.ylabel = "Y"
        self.figsize = figsize
        self.dpi = dpi
        self.marker = marker
        self.marker_size = marker_size
        self.marker_alpha = marker_alpha
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.x_log = x_log
        self.y_log = y_log
        self.grid = grid
        self.grid_alpha = grid_alpha
        self.depth_range = depth_range
        self.show_colorbar = show_colorbar
        self.show_legend = show_legend
        self.show_regression_legend = show_regression_legend
        self.show_regression_equation = show_regression_equation
        self.show_regression_r2 = show_regression_r2
        self.regression = regression
        self.regression_by_color = regression_by_color
        self.regression_by_group = regression_by_group

        # Handle regression_by_shape_and_color as alias for regression_by_color_and_shape
        if regression_by_shape_and_color is not None and regression_by_color_and_shape is not None:
            warnings.warn(
                "Both regression_by_color_and_shape and regression_by_shape_and_color were specified. "
                "These are aliases for the same feature. Using regression_by_color_and_shape.",
                stacklevel=2,
            )
            self.regression_by_color_and_shape = regression_by_color_and_shape
        elif regression_by_shape_and_color is not None:
            # Use the alias
            self.regression_by_color_and_shape = regression_by_shape_and_color
        else:
            self.regression_by_color_and_shape = regression_by_color_and_shape

        # Plot objects
        self.fig = None
        self.ax = None
        self.scatter = None
        self.colorbar = None

        # Regression storage - nested structure: {type: {identifier: regression_obj}}
        self._regressions = {}
        self.regression_lines = {}
        self.regression_legend = None  # Separate legend for regressions

        # Pending regressions (added before plot() is called)
        self._pending_regressions = []

        # Data cache
        self._data = None

        # Discrete property labels storage
        # Maps property role ('shape', 'color', 'size') to labels dict {0: 'label0', 1: 'label1', ...}
        self._discrete_labels = {}

        # Legend placement tracking
        # Maps segment numbers (1-9) to legend type placed there
        self._occupied_segments = {}

    def add_layer(self, x: str, y: str, label: str) -> Crossplot:
        """
        Add a new data layer to the crossplot.

        This allows combining multiple property pairs in a single plot, useful for
        comparing different data types (e.g., Core vs Sidewall data).

        Parameters
        ----------
        x : str
            Name of property for x-axis for this layer
        y : str
            Name of property for y-axis for this layer
        label : str
            Label for this layer (used in legend and available as "label" property
            for color/shape mapping)

        Returns
        -------
        self
            Returns self to allow method chaining

        Examples
        --------
        >>> plot = manager.Crossplot(y_log=True)
        >>> plot.add_layer('CorePor_obds', 'CorePerm_obds', label='Core')
        >>> plot.add_layer('SidewallPor_ob', 'SidewallPerm_ob', label='Sidewall')
        >>> plot.show()

        With method chaining:
        >>> manager.Crossplot(y_log=True) \\
        ...     .add_layer('CorePor_obds', 'CorePerm_obds', label='Core') \\
        ...     .add_layer('SidewallPor_ob', 'SidewallPerm_ob', label='Sidewall') \\
        ...     .show()
        """
        self._layers.append({"label": label, "x": x, "y": y})
        # Clear data cache since we're adding new data
        self._data = None
        return self

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data from wells for plotting."""
        if self._data is not None:
            return self._data

        all_data = []

        # Helper function to check if alignment is needed
        def needs_alignment(prop_depth, ref_depth):
            """Quick check if depths need alignment."""
            if len(prop_depth) != len(ref_depth):
                return True
            # Fast check: if arrays are identical objects or first/last don't match
            if prop_depth is ref_depth:
                return False
            if prop_depth[0] != ref_depth[0] or prop_depth[-1] != ref_depth[-1]:
                return True
            # Only do expensive allclose if needed
            return not np.allclose(prop_depth, ref_depth)

        # Helper function to align property values to target depth grid
        def align_property(prop, target_depth):
            """
            Align property values to target depth grid.

            Uses appropriate interpolation based on property type:
            - Discrete properties: forward-fill/previous (geological zones extend from
              their top/boundary until the next boundary is encountered)
            - Continuous properties: linear interpolation

            Args:
                prop: Property object to align
                target_depth: Target depth array

            Returns:
                Aligned values array
            """
            if prop.type == "discrete":
                # Use Property's resample method which handles discrete properties correctly
                # (forward-fill to preserve integer codes and geological zone logic)
                resampled = prop.resample(target_depth)
                return resampled.values
            else:
                # For continuous properties, use linear interpolation
                return np.interp(target_depth, prop.depth, prop.values, left=np.nan, right=np.nan)

        # Loop through each layer
        for layer in self._layers:
            layer_x = layer["x"]
            layer_y = layer["y"]
            layer_label = layer["label"]

            for well in self.wells:
                try:
                    # Get x and y properties for this layer
                    x_prop = well.get_property(layer_x)
                    y_prop = well.get_property(layer_y)

                    # Get depths - use x property's depth
                    depths = x_prop.depth
                    x_values = x_prop.values
                    y_values = y_prop.values

                    # Align y values to x depth grid if needed using appropriate method
                    if needs_alignment(y_prop.depth, depths):
                        y_values = align_property(y_prop, depths)

                    # Create dataframe for this well and layer
                    df = pd.DataFrame(
                        {
                            "depth": depths,
                            "x": x_values,
                            "y": y_values,
                            "well": well.name,
                            "label": layer_label,  # Add layer label
                        }
                    )

                    # Add color property if specified
                    if self.color == "label":
                        # Use layer label for color
                        df["color_val"] = layer_label
                    elif self.color == "well":
                        # Use well name for color (categorical)
                        df["color_val"] = well.name
                    elif self.color and self.color != "depth":
                        try:
                            color_prop = well.get_property(self.color)
                            # Store labels if discrete property (only once)
                            if "color" not in self._discrete_labels:
                                self._store_discrete_labels(color_prop, "color")
                            # Align to x depth grid using appropriate method for property type
                            if needs_alignment(color_prop.depth, depths):
                                color_values = align_property(color_prop, depths)
                            else:
                                color_values = color_prop.values
                            df["color_val"] = color_values
                        except (AttributeError, KeyError, PropertyNotFoundError):
                            # Silently use depth as fallback
                            df["color_val"] = depths
                    elif self.color == "depth":
                        df["color_val"] = depths

                    # Add size property if specified
                    if self.size == "label":
                        # Use layer label for size (will need special handling in plot)
                        df["size_val"] = layer_label
                    elif self.size:
                        try:
                            size_prop = well.get_property(self.size)
                            # Store labels if discrete property (only once)
                            if "size" not in self._discrete_labels:
                                self._store_discrete_labels(size_prop, "size")
                            # Align to x depth grid using appropriate method for property type
                            if needs_alignment(size_prop.depth, depths):
                                size_values = align_property(size_prop, depths)
                            else:
                                size_values = size_prop.values
                            df["size_val"] = size_values
                        except (AttributeError, KeyError, PropertyNotFoundError):
                            # Silently skip if size property not found
                            pass

                    # Add shape property if specified
                    if self.shape == "label":
                        # Use layer label for shape
                        df["shape_val"] = layer_label
                    elif self.shape and self.shape != "well":
                        try:
                            shape_prop = well.get_property(self.shape)
                            # Store labels if discrete property (only once)
                            if "shape" not in self._discrete_labels:
                                self._store_discrete_labels(shape_prop, "shape")
                            # Align to x depth grid using appropriate method for property type
                            if needs_alignment(shape_prop.depth, depths):
                                shape_values = align_property(shape_prop, depths)
                            else:
                                shape_values = shape_prop.values
                            df["shape_val"] = shape_values
                        except (AttributeError, KeyError, PropertyNotFoundError):
                            # Silently skip if shape property not found
                            pass

                    all_data.append(df)

                except (AttributeError, KeyError, PropertyNotFoundError):
                    # Silently skip wells that don't have the required properties
                    continue

        if not all_data:
            raise ValueError("No valid data found in any wells")

        # Combine all data
        self._data = pd.concat(all_data, ignore_index=True)

        # Apply depth filter if specified
        if self.depth_range:
            min_depth, max_depth = self.depth_range
            self._data = self._data[
                (self._data["depth"] >= min_depth) & (self._data["depth"] <= max_depth)
            ]

        # Remove rows with NaN in x or y
        self._data = self._data.dropna(subset=["x", "y"])

        return self._data

    def _parse_regression_config(self, config: str | dict) -> dict:
        """Parse regression configuration from string or dict format.

        Args:
            config: Either a string (e.g., "linear") or dict (e.g., {"type": "linear", "line_color": "red"})

        Returns:
            Dictionary with 'type' and optional styling parameters
        """
        if isinstance(config, str):
            return {"type": config}
        elif isinstance(config, dict):
            if "type" not in config:
                raise ValueError("Regression config dict must contain 'type' key")
            return config.copy()
        else:
            raise ValueError(f"Regression config must be string or dict, got {type(config)}")

    def regression(self, regression_type: str | None = None) -> dict:
        """Access regression objects.

        Args:
            regression_type: Optional regression type to filter by (e.g., "linear", "polynomial")

        Returns:
            If regression_type is None: Returns all regressions organized by type:
                {"linear": {"red": RegObj, ...}, "polynomial": {"blue": RegObj, ...}}
            If regression_type specified: Returns regressions of that type:
                {"red": RegObj, ...}

        Examples:
            >>> plot.regression()  # Get all regressions
            >>> plot.regression("linear")  # Get only linear regressions
        """
        if regression_type is None:
            return self._regressions.copy()
        else:
            return self._regressions.get(regression_type, {}).copy()

    def _store_regression(self, reg_type: str, identifier: str, regression_obj) -> None:
        """Store a regression object in the nested structure.

        Args:
            reg_type: Type of regression (e.g., "linear", "polynomial")
            identifier: Unique identifier for this regression (e.g., color, group name)
            regression_obj: The regression object to store
        """
        if reg_type not in self._regressions:
            self._regressions[reg_type] = {}
        self._regressions[reg_type][identifier] = regression_obj

    def _get_group_colors(self, data: pd.DataFrame, group_column: str) -> dict:
        """Get the color assigned to each group in the plot.

        Args:
            data: DataFrame with plotting data
            group_column: Column name used for grouping

        Returns:
            Dictionary mapping group names to their colors
        """
        group_colors = {}

        # Get unique groups in the same order as they'll appear in the plot
        groups = data.groupby(group_column)

        for idx, (group_name, _) in enumerate(groups):
            # Use the same color assignment logic as _plot_by_groups
            group_colors[group_name] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        return group_colors

    def _find_best_legend_locations(self, data: pd.DataFrame) -> tuple[str, str]:
        """Find the two best locations for legends based on data density.

        Divides the plot into a 3x3 grid and finds the two squares with the least data points.

        Args:
            data: DataFrame with 'x' and 'y' columns

        Returns:
            Tuple of (primary_location, secondary_location) as matplotlib location strings
        """
        # Get x and y bounds
        x_vals = data["x"].values
        y_vals = data["y"].values

        # Handle log scales for binning
        if self.x_log:
            x_vals = np.log10(x_vals[x_vals > 0])
        if self.y_log:
            y_vals = np.log10(y_vals[y_vals > 0])

        x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
        y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)

        # Create 3x3 grid and count points in each square
        x_bins = np.linspace(x_min, x_max, 4)
        y_bins = np.linspace(y_min, y_max, 4)

        # Count points in each of 9 squares
        counts = {}
        for i in range(3):
            for j in range(3):
                x_mask = (x_vals >= x_bins[i]) & (x_vals < x_bins[i + 1])
                y_mask = (y_vals >= y_bins[j]) & (y_vals < y_bins[j + 1])
                counts[(i, j)] = np.sum(x_mask & y_mask)

        # Map grid positions to matplotlib location strings
        # Grid: (0,2) (1,2) (2,2)  ->  upper left, upper center, upper right
        #       (0,1) (1,1) (2,1)  ->  center left, center, center right
        #       (0,0) (1,0) (2,0)  ->  lower left, lower center, lower right
        position_map = {
            (0, 2): "upper left",
            (1, 2): "upper center",
            (2, 2): "upper right",
            (0, 1): "center left",
            (1, 1): "center",
            (2, 1): "center right",
            (0, 0): "lower left",
            (1, 0): "lower center",
            (2, 0): "lower right",
        }

        # Sort squares by count (ascending)
        sorted_squares = sorted(counts.items(), key=lambda x: x[1])

        # Get two best locations
        best_pos = position_map[sorted_squares[0][0]]
        second_best_pos = position_map[sorted_squares[1][0]]

        return best_pos, second_best_pos

    def _find_optimal_legend_segment(
        self, data: pd.DataFrame, legend_type: str, is_large: bool = False
    ) -> tuple[int, str]:
        """Find optimal segment for legend placement using priority-based algorithm.

        Segments are numbered 1-9:
            1  2  3     (upper left, upper center, upper right)
            4  5  6     (center left, center, center right)
            7  8  9     (lower left, lower center, lower right)

        Checks segments in priority order: 1,9,4,6,3,7,2,8,5
        A segment is eligible if:
        - It has <10% of datapoints
        - It doesn't have a previous legend, EXCEPT:
          - Shape and color legends can share a segment if neither is very large

        Args:
            data: DataFrame with 'x' and 'y' columns
            legend_type: Type of legend ('shape', 'color', 'size', 'regression', etc.)
            is_large: Whether this legend is considered large (many items)

        Returns:
            Tuple of (segment_number, matplotlib_location_string)
        """
        # Get x and y data values
        x_vals = data["x"].values
        y_vals = data["y"].values

        # Get axes limits to convert data coordinates to axes-normalized coordinates (0-1)
        # This ensures we're dividing the GRAPH AREA, not the data space
        if self.ax is not None:
            x_lim = self.ax.get_xlim()
            y_lim = self.ax.get_ylim()
        else:
            # Fallback if ax not available yet
            x_lim = (np.nanmin(x_vals), np.nanmax(x_vals))
            y_lim = (np.nanmin(y_vals), np.nanmax(y_vals))

        # Handle logarithmic axes - transform to log space for proper visual segment calculation
        # On log axes, equal visual spacing corresponds to equal ratios, not equal differences
        if self.x_log:
            # Filter out non-positive values before log transform
            x_valid = x_vals > 0
            x_vals_transformed = np.where(x_valid, np.log10(x_vals), np.nan)
            x_lim_transformed = (np.log10(max(x_lim[0], 1e-10)), np.log10(max(x_lim[1], 1e-10)))
        else:
            x_vals_transformed = x_vals
            x_lim_transformed = x_lim

        if self.y_log:
            # Filter out non-positive values before log transform
            y_valid = y_vals > 0
            y_vals_transformed = np.where(y_valid, np.log10(y_vals), np.nan)
            y_lim_transformed = (np.log10(max(y_lim[0], 1e-10)), np.log10(max(y_lim[1], 1e-10)))
        else:
            y_vals_transformed = y_vals
            y_lim_transformed = y_lim

        # Normalize transformed coordinates to axes coordinates (0-1)
        # This divides the visible graph area properly, accounting for log scales
        x_norm = (x_vals_transformed - x_lim_transformed[0]) / (
            x_lim_transformed[1] - x_lim_transformed[0]
        )
        y_norm = (y_vals_transformed - y_lim_transformed[0]) / (
            y_lim_transformed[1] - y_lim_transformed[0]
        )

        # Create 3x3 grid in axes-normalized space (0-1)
        x_bins = np.linspace(0, 1, 4)
        y_bins = np.linspace(0, 1, 4)

        # Map segments 1-9 to grid positions (i, j)
        # Segment numbering:
        #   1  2  3
        #   4  5  6
        #   7  8  9
        segment_to_grid = {
            1: (0, 2),  # upper left
            2: (1, 2),  # upper center
            3: (2, 2),  # upper right
            4: (0, 1),  # center left
            5: (1, 1),  # center
            6: (2, 1),  # center right
            7: (0, 0),  # lower left
            8: (1, 0),  # lower center
            9: (2, 0),  # lower right
        }

        # Map segments to matplotlib location strings
        segment_to_location = {
            1: "upper left",
            2: "upper center",
            3: "upper right",
            4: "center left",
            5: "center",
            6: "center right",
            7: "lower left",
            8: "lower center",
            9: "lower right",
        }

        # Count points in each segment using normalized coordinates
        total_points = len(x_norm)
        segment_counts = {}
        for segment, (i, j) in segment_to_grid.items():
            x_mask = (x_norm >= x_bins[i]) & (x_norm < x_bins[i + 1])
            y_mask = (y_norm >= y_bins[j]) & (y_norm < y_bins[j + 1])
            count = np.sum(x_mask & y_mask)
            segment_counts[segment] = count

        # Priority order: 1,9,4,6,3,7,2,8,5
        priority_order = [1, 9, 4, 6, 3, 7, 2, 8, 5]

        # Check each segment in priority order
        for segment in priority_order:
            # Check datapoint percentage
            if total_points > 0:
                percentage = segment_counts[segment] / total_points
                if percentage >= 0.10:  # 10% threshold
                    continue

            # Check if segment is occupied
            if segment in self._occupied_segments:
                existing_type = self._occupied_segments[segment]

                # Allow shape and color to share if neither is very large
                shareable = {"shape", "color"}
                if (
                    legend_type in shareable
                    and existing_type in shareable
                    and not is_large
                    and not self._occupied_segments.get(f"{segment}_large", False)
                ):
                    # Can share this segment
                    pass
                else:
                    # Segment occupied, try next
                    continue

            # Found eligible segment
            return segment, segment_to_location[segment]

        # If no eligible segment found, use fallback (segment 1)
        return 1, segment_to_location[1]

    def _store_discrete_labels(self, prop, role: str) -> None:
        """
        Store labels from a discrete property for later use in legends.

        Args:
            prop: Property object (must have type and labels attributes)
            role: Property role - 'shape', 'color', or 'size'
        """
        if (
            hasattr(prop, "type")
            and prop.type == "discrete"
            and hasattr(prop, "labels")
            and prop.labels
        ):
            self._discrete_labels[role] = prop.labels.copy()

    def _get_display_label(self, value, role: str) -> str:
        """
        Get display label for a value, using stored labels for discrete properties.

        For discrete properties with labels, converts integer codes to readable names.
        For continuous properties or discrete without labels, returns string value.

        Args:
            value: The value to get label for (could be int, float, or string)
            role: Property role - 'shape', 'color', or 'size'

        Returns:
            Display label string

        Examples:
            >>> # For discrete property with labels {0: 'Agat top', 1: 'Cerisa Main top'}
            >>> self._get_display_label(0.0, 'shape')
            'Agat top'

            >>> # For continuous property or no labels
            >>> self._get_display_label(2.5, 'color')
            '2.5'
        """
        if role in self._discrete_labels:
            # Try to convert to integer and look up label
            try:
                int_val = int(np.round(float(value)))
                return self._discrete_labels[role].get(int_val, str(value))
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    def _is_edge_location(self, location: str) -> bool:
        """Check if a legend location is on the left or right edge.

        Args:
            location: Matplotlib location string

        Returns:
            True if on left or right edge (for vertical stacking)
        """
        edge_locations = [
            "upper left",
            "center left",
            "lower left",
            "upper right",
            "center right",
            "lower right",
        ]
        return location in edge_locations

    def _create_grouped_legends(
        self, shape_handles, shape_title: str, color_handles, color_title: str, location: str
    ) -> None:
        """Create grouped legends in the same region, stacked or side-by-side.

        When both shape and color legends are needed, this groups them in the same
        1/9th section without overlap. Stacks vertically on edges, side-by-side elsewhere.

        Args:
            shape_handles: List of handles for shape legend
            shape_title: Title for shape legend
            color_handles: List of handles for color legend
            color_title: Title for color legend
            location: Matplotlib location string for positioning
        """
        is_edge = self._is_edge_location(location)

        # Determine base anchor point from location string
        # Map location to (x, y) coordinates in AXES space (0-1 within the graph area)
        # These match the segment corners:
        # Segment 1=upper left (0,1), 2=upper center (0.5,1), 3=upper right (1,1)
        # Segment 4=center left (0,0.5), 5=center (0.5,0.5), 6=center right (1,0.5)
        # Segment 7=lower left (0,0), 8=lower center (0.5,0), 9=lower right (1,0)
        anchor_map = {
            "upper left": (0, 1),
            "upper center": (0.5, 1),
            "upper right": (1, 1),
            "center left": (0, 0.5),
            "center": (0.5, 0.5),
            "center right": (1, 0.5),
            "lower left": (0, 0),
            "lower center": (0.5, 0),
            "lower right": (1, 0),
        }

        base_x, base_y = anchor_map.get(location, (1, 1))

        if is_edge:
            # Stack vertically on edges
            # Position shape legend at the top
            shape_legend = self.ax.legend(
                handles=shape_handles,
                title=shape_title,
                loc=location,
                frameon=True,
                framealpha=0.9,
                edgecolor="black",
                bbox_to_anchor=(base_x, base_y),
                bbox_transform=self.ax.transAxes,
            )
            shape_legend.get_title().set_fontweight("bold")
            shape_legend.set_clip_on(False)  # Prevent clipping outside axes
            self.ax.add_artist(shape_legend)

            # Calculate offset for color legend below shape legend
            # Estimate shape legend height in axes coordinates
            shape_height = len(shape_handles) * 0.05 + 0.08  # Adjusted for axes space

            # Adjust y position for color legend
            if "upper" in location:
                color_y = base_y - shape_height  # Stack below
            elif "lower" in location:
                color_y = base_y + shape_height  # Stack above
            else:  # center
                color_y = base_y - shape_height / 2  # Stack below

            color_legend = self.ax.legend(
                handles=color_handles,
                title=color_title,
                loc=location,
                frameon=True,
                framealpha=0.9,
                edgecolor="black",
                bbox_to_anchor=(base_x, color_y),
                bbox_transform=self.ax.transAxes,
            )
            color_legend.get_title().set_fontweight("bold")
            color_legend.set_clip_on(False)  # Prevent clipping outside axes
        else:
            # Place side by side for non-edge locations (top, bottom, center)
            # Estimate width of each legend in axes coordinates
            legend_width = 0.20

            if "center" in location and location != "center left" and location != "center right":
                # For center positions, place them side by side
                shape_x = base_x - legend_width / 2
                color_x = base_x + legend_width / 2

                shape_legend = self.ax.legend(
                    handles=shape_handles,
                    title=shape_title,
                    loc="center",
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="black",
                    bbox_to_anchor=(shape_x, base_y),
                    bbox_transform=self.ax.transAxes,
                )
                shape_legend.get_title().set_fontweight("bold")
                shape_legend.set_clip_on(False)  # Prevent clipping outside axes
                self.ax.add_artist(shape_legend)

                color_legend = self.ax.legend(
                    handles=color_handles,
                    title=color_title,
                    loc="center",
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="black",
                    bbox_to_anchor=(color_x, base_y),
                    bbox_transform=self.ax.transAxes,
                )
                color_legend.get_title().set_fontweight("bold")
                color_legend.set_clip_on(False)  # Prevent clipping outside axes
            else:
                # For other positions, fall back to stacking
                shape_legend = self.ax.legend(
                    handles=shape_handles,
                    title=shape_title,
                    loc=location,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="black",
                    bbox_to_anchor=(base_x, base_y),
                    bbox_transform=self.ax.transAxes,
                )
                shape_legend.get_title().set_fontweight("bold")
                shape_legend.set_clip_on(False)  # Prevent clipping outside axes
                self.ax.add_artist(shape_legend)

                # Estimate offset in axes coordinates
                shape_height = len(shape_handles) * 0.05 + 0.08
                if "upper" in location:
                    color_y = base_y - shape_height
                else:
                    color_y = base_y + shape_height

                color_legend = self.ax.legend(
                    handles=color_handles,
                    title=color_title,
                    loc=location,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="black",
                    bbox_to_anchor=(base_x, color_y),
                    bbox_transform=self.ax.transAxes,
                )
                color_legend.get_title().set_fontweight("bold")
                color_legend.set_clip_on(False)  # Prevent clipping outside axes

    def _format_regression_label(
        self, name: str, reg, include_equation: bool = None, include_r2: bool = None
    ) -> str:
        """Format a modern, compact regression label.

        Args:
            name: Name of the regression
            reg: Regression object
            include_equation: Whether to include equation (uses self.show_regression_equation if None)
            include_r2: Whether to include R-squared (uses self.show_regression_r2 if None)

        Returns:
            Formatted label string
        """
        if include_equation is None:
            include_equation = self.show_regression_equation
        if include_r2 is None:
            include_r2 = self.show_regression_r2

        # Format: "Name (equation)" with R-squared on second line
        # Equation and R-squared will be colored grey in the legend update method
        first_line = name
        if include_equation:
            eq = reg.equation()
            eq = eq.replace(" ", "")  # Remove spaces for compactness
            # Add equation in parentheses (will be styled grey later)
            first_line = f"{name} ({eq})"

        # Add R-squared on second line if requested (will be styled grey later)
        if include_r2:
            # Format R-squared (no suffix needed)
            r2_label = f"R\u00b2 = {reg.r_squared:.3f}"
            return f"{first_line}\n{r2_label}"
        else:
            return first_line

    def _update_regression_legend(self) -> None:
        """Create or update the separate regression legend with smart placement."""
        if not self.show_regression_legend or not self.regression_lines:
            return

        if self.ax is None:
            return

        # Remove old regression legend if it exists
        if self.regression_legend is not None:
            self.regression_legend.remove()
            self.regression_legend = None

        # Create new regression legend with only regression lines
        regression_handles = []
        regression_labels = []

        for line in self.regression_lines.values():
            regression_handles.append(line)
            regression_labels.append(line.get_label())

        if regression_handles:
            # Get smart placement based on data density using optimized segment algorithm
            if self._data is not None:
                # Determine if regression legend is large
                regression_is_large = len(regression_handles) > 5
                segment, secondary_loc = self._find_optimal_legend_segment(
                    self._data, legend_type="regression", is_large=regression_is_large
                )
                # Mark segment as occupied
                self._occupied_segments[segment] = "regression"
                self._occupied_segments[f"{segment}_large"] = regression_is_large
            else:
                # Fallback if data not available
                secondary_loc = "lower right"

            # Determine descriptive title based on regression type
            # Extract the regression type and add it to the title
            reg_type_str = None
            if self.regression_by_color_and_shape:
                base_title = "Regressions by color and shape"
                config = self._parse_regression_config(self.regression_by_color_and_shape)
                reg_type_str = config.get("type", None)
            elif self.regression_by_color:
                base_title = "Regressions by color"
                config = self._parse_regression_config(self.regression_by_color)
                reg_type_str = config.get("type", None)
            elif self.regression_by_group:
                base_title = "Regressions by group"
                config = self._parse_regression_config(self.regression_by_group)
                reg_type_str = config.get("type", None)
            else:
                base_title = "Regressions"
                if self.regression:
                    config = self._parse_regression_config(self.regression)
                    reg_type_str = config.get("type", None)

            # Add regression type to title (e.g., "Regressions by color - Power")
            if reg_type_str:
                reg_type_display = reg_type_str.capitalize()
                regression_title = f"{base_title} - {reg_type_display}"
            else:
                regression_title = base_title

            # Import legend from matplotlib
            from matplotlib.legend import Legend

            # Create regression legend at secondary location
            self.regression_legend = Legend(
                self.ax,
                regression_handles,
                regression_labels,
                loc=secondary_loc,
                frameon=True,
                framealpha=0.95,
                edgecolor="#cccccc",
                fancybox=False,
                shadow=False,
                fontsize=9,
                title=regression_title,
                title_fontsize=10,
            )

            # Modern styling with grey text for equation and R-squared
            self.regression_legend.get_frame().set_linewidth(0.8)
            self.regression_legend.get_title().set_fontweight("600")

            # Set text color to grey for all labels
            for text in self.regression_legend.get_texts():
                text.set_color("#555555")

            # Add as artist to avoid replacing the primary legend
            self.ax.add_artist(self.regression_legend)

    def _add_automatic_regressions(self, data: pd.DataFrame) -> None:
        """Add automatic regressions based on initialization parameters."""
        if not any(
            [
                self.regression,
                self.regression_by_color,
                self.regression_by_group,
                self.regression_by_color_and_shape,
            ]
        ):
            return

        total_points = len(data)
        regression_count = 0

        # Define colors for different regression lines
        regression_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        color_idx = 0

        # Add overall regression
        if self.regression:
            config = self._parse_regression_config(self.regression)
            reg_type = config["type"]

            # Set default line color if not specified
            if "line_color" not in config:
                config["line_color"] = regression_colors[color_idx % len(regression_colors)]

            self.add_regression(
                reg_type,
                name=f"Overall {reg_type}",
                line_color=config.get("line_color", "red"),
                line_width=config.get("line_width", 2),
                line_style=config.get("line_style", "-"),
                line_alpha=config.get("line_alpha", 0.8),
            )
            regression_count += 1
            color_idx += 1

        # Add regression by color groups
        if self.regression_by_color:
            config = self._parse_regression_config(self.regression_by_color)
            reg_type = config["type"]

            # Determine grouping column based on what's being used for colors in the plot
            group_column = None

            if self.color and "color_val" in data.columns:
                # User specified explicit color mapping
                group_column = "color_val"
            elif self.shape == "well" and "well" in data.columns:
                # When shape="well", each well gets a different color in the plot
                group_column = "well"
            elif self.shape and self.shape != "well" and "shape_val" in data.columns:
                # When shape is a property, each shape group gets a different color
                group_column = "shape_val"

            if group_column is None:
                warnings.warn(
                    "regression_by_color specified but no color grouping detected in plot. "
                    "Use color=<property>, shape='well', or shape=<property> parameter.",
                    stacklevel=2,
                )
            else:
                # Check if color is categorical (not continuous like depth)
                # Use _is_categorical_color() to properly handle discrete properties
                if group_column == "color_val":
                    is_categorical = self._is_categorical_color(data[group_column].values)
                    if not is_categorical:
                        # For continuous values, we can't create separate regressions
                        warnings.warn(
                            f"regression_by_color requires categorical color mapping, "
                            f"but '{self.color}' is continuous. Use regression_by_group instead.",
                            stacklevel=2,
                        )
                        # Skip this section
                    else:
                        # Categorical values - group and create regressions
                        color_groups = data.groupby(group_column)
                        n_groups = len(color_groups)

                        # Validate regression count
                        if regression_count + n_groups > total_points / 2:
                            raise ValueError(
                                f"Too many regression lines requested: {regression_count + n_groups} lines "
                                f"for {total_points} data points (average < 2 points per line). "
                                f"Reduce the number of groups or use a different regression strategy."
                            )

                        # Get the actual colors used for each group in the plot
                        group_colors_map = self._get_group_colors(data, group_column)

                        for idx, (group_name, group_data) in enumerate(color_groups):
                            x_vals = group_data["x"].values
                            y_vals = group_data["y"].values
                            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                            if np.sum(mask) >= 2:
                                # Copy config and use the same color as the data group
                                group_config = config.copy()
                                if "line_color" not in group_config:
                                    # Use the same color as the data points for this group
                                    group_config["line_color"] = group_colors_map.get(
                                        group_name,
                                        regression_colors[color_idx % len(regression_colors)],
                                    )

                                # Get display label for group name (converts codes to formation names)
                                group_display = self._get_display_label(group_name, "color")

                                # Skip legend update for all but last regression
                                is_last = idx == n_groups - 1
                                self._add_group_regression(
                                    x_vals[mask],
                                    y_vals[mask],
                                    reg_type,
                                    name=group_display,
                                    config=group_config,
                                    update_legend=is_last,
                                )
                                regression_count += 1
                                color_idx += 1
                else:
                    # Categorical values - group and create regressions
                    color_groups = data.groupby(group_column)
                    n_groups = len(color_groups)

                    # Validate regression count
                    if regression_count + n_groups > total_points / 2:
                        raise ValueError(
                            f"Too many regression lines requested: {regression_count + n_groups} lines "
                            f"for {total_points} data points (average < 2 points per line). "
                            f"Reduce the number of groups or use a different regression strategy."
                        )

                    # Get the actual colors used for each group in the plot
                    group_colors_map = self._get_group_colors(data, group_column)

                    for idx, (group_name, group_data) in enumerate(color_groups):
                        x_vals = group_data["x"].values
                        y_vals = group_data["y"].values
                        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                        if np.sum(mask) >= 2:
                            # Copy config and use the same color as the data group
                            group_config = config.copy()
                            if "line_color" not in group_config:
                                # Use the same color as the data points for this group
                                group_config["line_color"] = group_colors_map.get(
                                    group_name,
                                    regression_colors[color_idx % len(regression_colors)],
                                )

                            # Get display label for group name (converts codes to formation names)
                            group_display = self._get_display_label(group_name, "color")

                            # Skip legend update for all but last regression
                            is_last = idx == n_groups - 1
                            self._add_group_regression(
                                x_vals[mask],
                                y_vals[mask],
                                reg_type,
                                name=group_display,
                                config=group_config,
                                update_legend=is_last,
                            )
                            regression_count += 1
                            color_idx += 1

        # Add regression by groups (well or shape)
        if self.regression_by_group:
            config = self._parse_regression_config(self.regression_by_group)
            reg_type = config["type"]

            # Determine grouping
            if self.shape == "well" or (self.shape and "shape_val" in data.columns):
                group_col = "well" if self.shape == "well" else "shape_val"
                groups = data.groupby(group_col)
                n_groups = len(groups)

                # Validate regression count
                if regression_count + n_groups > total_points / 2:
                    raise ValueError(
                        f"Too many regression lines requested: {regression_count + n_groups} lines "
                        f"for {total_points} data points (average < 2 points per line). "
                        f"Reduce the number of groups or use a different regression strategy."
                    )

                # Get the actual colors used for each group in the plot
                group_colors_map = self._get_group_colors(data, group_col)

                for idx, (group_name, group_data) in enumerate(groups):
                    x_vals = group_data["x"].values
                    y_vals = group_data["y"].values
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if np.sum(mask) >= 2:
                        # Copy config and use the same color as the data group
                        group_config = config.copy()
                        if "line_color" not in group_config:
                            # Use the same color as the data points for this group
                            group_config["line_color"] = group_colors_map.get(
                                group_name, regression_colors[color_idx % len(regression_colors)]
                            )

                        # Skip legend update for all but last regression
                        is_last = idx == n_groups - 1
                        self._add_group_regression(
                            x_vals[mask],
                            y_vals[mask],
                            reg_type,
                            name=f"{group_col}={group_name}",
                            config=group_config,
                            update_legend=is_last,
                        )
                        regression_count += 1
                        color_idx += 1
            else:
                warnings.warn(
                    "regression_by_group specified but no shape/well grouping defined. "
                    "Use shape='well' or set shape to a property name.",
                    stacklevel=2,
                )

        # Add regression by color AND shape combinations
        if self.regression_by_color_and_shape:
            config = self._parse_regression_config(self.regression_by_color_and_shape)
            reg_type = config["type"]

            # Determine color and shape columns
            color_col = None
            shape_col = None
            color_label = None
            shape_label = None

            # Identify color column
            if self.color and "color_val" in data.columns:
                # Check if categorical
                if self._is_categorical_color(data["color_val"].values):
                    color_col = "color_val"
                    color_label = self.color
            elif self.shape == "well" and "well" in data.columns:
                # When shape="well", wells provide colors
                color_col = "well"
                color_label = "well"

            # Identify shape column
            if self.shape == "well" and "well" in data.columns:
                shape_col = "well"
                shape_label = "well"
            elif self.shape and self.shape != "well" and "shape_val" in data.columns:
                shape_col = "shape_val"
                shape_label = self.shape

            # Need both color and shape columns for this to work
            if color_col is None or shape_col is None:
                warnings.warn(
                    "regression_by_color_and_shape requires both categorical color mapping AND shape/well grouping. "
                    "Set both color and shape parameters, or use regression_by_color or regression_by_group instead.",
                    stacklevel=2,
                )
            elif color_col == shape_col:
                warnings.warn(
                    "regression_by_color_and_shape requires DIFFERENT color and shape mappings. "
                    "Currently both are mapped to the same property. Use regression_by_color or regression_by_group instead.",
                    stacklevel=2,
                )
            else:
                # Group by both color and shape
                combined_groups = data.groupby([color_col, shape_col])
                n_groups = len(combined_groups)

                # Validate regression count
                if regression_count + n_groups > total_points / 2:
                    raise ValueError(
                        f"Too many regression lines requested: {regression_count + n_groups} lines "
                        f"for {total_points} data points (average < 2 points per line). "
                        f"Reduce the number of groups or use a simpler regression strategy."
                    )

                # Get color maps for both dimensions
                color_colors_map = self._get_group_colors(data, color_col)
                shape_colors_map = self._get_group_colors(data, shape_col)

                for idx, ((color_val, shape_val), group_data) in enumerate(combined_groups):
                    x_vals = group_data["x"].values
                    y_vals = group_data["y"].values
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if np.sum(mask) >= 2:
                        # Copy config and use appropriate color
                        group_config = config.copy()
                        if "line_color" not in group_config:
                            # Prefer color from color dimension, fallback to shape dimension
                            group_config["line_color"] = color_colors_map.get(
                                color_val,
                                shape_colors_map.get(
                                    shape_val, regression_colors[color_idx % len(regression_colors)]
                                ),
                            )

                        # Create descriptive name with both dimensions
                        color_display = self._get_display_label(color_val, "color")
                        shape_display = self._get_display_label(shape_val, "shape")
                        name = f"{color_label}={color_display}, {shape_label}={shape_display}"

                        # Skip legend update for all but last regression
                        is_last = idx == n_groups - 1
                        self._add_group_regression(
                            x_vals[mask],
                            y_vals[mask],
                            reg_type,
                            name=name,
                            config=group_config,
                            update_legend=is_last,
                        )
                        regression_count += 1
                        color_idx += 1

    def _add_group_regression(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        regression_type: str,
        name: str,
        config: dict,
        update_legend: bool = True,
    ) -> None:
        """Add a regression line for a specific group of data.

        Args:
            x_vals: X values for the group
            y_vals: Y values for the group
            regression_type: Type of regression (e.g., "linear", "polynomial")
            name: Name identifier for this regression
            config: Configuration dict with optional keys: line_color, line_width, line_style, line_alpha, x_range
        """
        # Create regression object using factory function
        reg = _create_regression(regression_type, degree=config.get("degree", 2))

        # Fit regression
        try:
            reg.fit(x_vals, y_vals)
        except ValueError as e:
            warnings.warn(
                f"Failed to fit {regression_type} regression for {name}: {e}", stacklevel=2
            )
            return

        # Recalculate R-squared in log space if y-axis is log scale
        if self.y_log:
            y_pred = reg.predict(x_vals)
            reg._calculate_metrics(x_vals, y_vals, y_pred, use_log_space=True)

        # Store regression in nested structure
        self._store_regression(regression_type, name, reg)

        # Get plot data using the regression helper method
        x_range_param = config.get("x_range", None)
        try:
            x_line, y_line = reg.get_plot_data(x_range=x_range_param, num_points=100)
        except ValueError as e:
            warnings.warn(f"Could not generate plot data for {name} regression: {e}", stacklevel=2)
            return

        # Create label using formatter
        label = self._format_regression_label(name, reg)

        # Plot line with config parameters
        line = self.ax.plot(
            x_line,
            y_line,
            color=config.get("line_color", "red"),
            linewidth=config.get("line_width", 1.5),
            linestyle=config.get("line_style", "--"),
            alpha=config.get("line_alpha", 0.7),
            label=label,
        )[0]

        self.regression_lines[name] = line

        # Update regression legend if requested (skipped during batch operations for performance)
        if update_legend and self.ax is not None:
            self._update_regression_legend()

    def _is_categorical_color(self, color_values: np.ndarray) -> bool:
        """
        Determine if color values should be treated as categorical vs continuous.

        Returns True if:
        - Less than 50 unique values

        This helps distinguish between:
        - Categorical: well names, facies, zones, labels
        - Continuous: depth, porosity, saturation
        """
        # Remove NaN values for analysis
        valid_values = color_values[~pd.isna(color_values)]

        if len(valid_values) == 0:
            return False

        unique_values = np.unique(valid_values)
        n_unique = len(unique_values)

        # Check if values are numeric - if not, it's categorical
        try:
            # Try to convert to float - if this fails, it's categorical (strings)
            _ = unique_values.astype(float)
        except (ValueError, TypeError):
            return True

        # Apply the criteria
        return n_unique < 50

    def plot(self) -> Crossplot:
        """Generate the crossplot figure."""
        # Reset legend placement tracking for new plot
        self._occupied_segments = {}

        # Prepare data
        data = self._prepare_data()

        if len(data) == 0:
            raise ValueError("No valid data points to plot")

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Determine plotting approach based on shape mapping
        if self.shape == "well" or (self.shape and "shape_val" in data.columns):
            self._plot_by_groups(data)
        else:
            self._plot_single_group(data)

        # Set scales
        if self.x_log:
            self.ax.set_xscale("log")
        if self.y_log:
            self.ax.set_yscale("log")

        # Disable scientific notation on linear axes only
        # (log axes use matplotlib's default log formatter for proper log scale labels)
        from matplotlib.ticker import ScalarFormatter

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)

        # Only apply to linear axes - log axes need their default formatter
        if not self.y_log:
            self.ax.yaxis.set_major_formatter(formatter)
        if not self.x_log:
            self.ax.xaxis.set_major_formatter(formatter)

        # Labels and title
        self.ax.set_xlabel(self.xlabel, fontsize=12, fontweight="bold")
        self.ax.set_ylabel(self.ylabel, fontsize=12, fontweight="bold")
        self.ax.set_title(self.title, fontsize=14, fontweight="bold", pad=20)

        # Grid
        if self.grid:
            self.ax.grid(
                True,
                which="major",
                alpha=min(self.grid_alpha * 1.2, 1.0),
                linestyle="-",
                linewidth=0.7,
            )
            # Add minor grid lines for log scales
            if self.x_log or self.y_log:
                self.ax.grid(
                    True, which="minor", alpha=self.grid_alpha, linestyle="-", linewidth=0.5
                )

        # Modern styling
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linewidth(1.5)
        self.ax.spines["bottom"].set_linewidth(1.5)

        # Add automatic regressions if specified
        self._add_automatic_regressions(data)

        # Apply pending regressions (added via add_regression() before plot() was called)
        if self._pending_regressions:
            for pending in self._pending_regressions:
                # Get the already-fitted regression object
                reg_type = pending["regression_type"]
                reg_name = pending["name"] if pending["name"] else reg_type

                # Retrieve stored regression
                if reg_type in self._regressions and reg_name in self._regressions[reg_type]:
                    reg = self._regressions[reg_type][reg_name]

                    # Draw the regression line
                    try:
                        x_line, y_line = reg.get_plot_data(
                            x_range=pending["x_range"], num_points=200
                        )
                    except ValueError as e:
                        warnings.warn(
                            f"Could not generate plot data for {reg_type} regression: {e}",
                            stacklevel=2,
                        )
                        continue

                    # Create label using formatter
                    label = self._format_regression_label(
                        reg_name,
                        reg,
                        include_equation=pending["show_equation"],
                        include_r2=pending["show_r2"],
                    )

                    # Plot line
                    line = self.ax.plot(
                        x_line,
                        y_line,
                        color=pending["line_color"],
                        linewidth=pending["line_width"],
                        linestyle=pending["line_style"],
                        alpha=pending["line_alpha"],
                        label=label,
                    )[0]

                    self.regression_lines[reg_name] = line

            # Update regression legend once after all pending regressions
            if self.ax is not None:
                self._update_regression_legend()

            # Clear pending list
            self._pending_regressions = []

        # Tight layout
        self.fig.tight_layout()

        return self

    def _plot_single_group(self, data: pd.DataFrame) -> None:
        """Plot all data as a single group."""
        x_vals = data["x"].values
        y_vals = data["y"].values

        # Determine colors
        is_categorical = False
        if self.color:
            c_vals_raw = data["color_val"].values

            # Check if color data is categorical
            is_categorical = self._is_categorical_color(c_vals_raw)

            if is_categorical:
                # Handle categorical colors with discrete palette
                unique_categories = pd.Series(c_vals_raw).dropna().unique()
                n_categories = len(unique_categories)

                # Create color map for categories
                if n_categories <= len(DEFAULT_COLORS):
                    color_palette = DEFAULT_COLORS
                else:
                    # Use colormap for many categories
                    cmap_obj = cm.get_cmap(self.colortemplate, n_categories)
                    color_palette = [cmap_obj(i) for i in range(n_categories)]

                category_colors = {
                    cat: color_palette[i % len(color_palette)]
                    for i, cat in enumerate(unique_categories)
                }

                # Map each value to its color
                c_vals = [category_colors.get(val, DEFAULT_COLORS[0]) for val in c_vals_raw]
                cmap = None
                vmin = vmax = None
            else:
                # Handle continuous colors
                c_vals = c_vals_raw
                cmap = self.colortemplate
                if self.color_range:
                    vmin, vmax = self.color_range
                else:
                    vmin, vmax = np.nanmin(c_vals), np.nanmax(c_vals)
        else:
            c_vals = DEFAULT_COLORS[0]
            cmap = None
            vmin = vmax = None

        # Determine sizes
        if self.size and "size_val" in data.columns:
            s_vals = data["size_val"].values
            # Normalize sizes to size_range
            s_min, s_max = np.nanmin(s_vals), np.nanmax(s_vals)
            if s_max > s_min:
                s_normalized = (s_vals - s_min) / (s_max - s_min)
                sizes = self.size_range[0] + s_normalized * (
                    self.size_range[1] - self.size_range[0]
                )
            else:
                sizes = self.marker_size
        else:
            sizes = self.marker_size

        # Create scatter plot
        self.scatter = self.ax.scatter(
            x_vals,
            y_vals,
            c=c_vals,
            s=sizes,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=self.marker_alpha,
            edgecolors=self.edge_color,
            linewidths=self.edge_width,
            marker=self.marker,
        )

        # Add colorbar or legend based on color type
        if self.color:
            if is_categorical and self.show_legend:
                # Create legend for categorical colors
                c_vals_raw = data["color_val"].values
                unique_categories = pd.Series(c_vals_raw).dropna().unique()

                # Create custom legend handles
                legend_elements = [
                    Patch(
                        facecolor=category_colors[cat],
                        edgecolor=self.edge_color,
                        label=self._get_display_label(cat, "color"),
                    )
                    for cat in unique_categories
                ]

                # Determine if legend is large
                color_is_large = len(legend_elements) > 5

                # Find optimal segment for color legend
                if self._data is not None:
                    segment, location = self._find_optimal_legend_segment(
                        self._data, legend_type="color", is_large=color_is_large
                    )
                    # Mark segment as occupied
                    self._occupied_segments[segment] = "color"
                    self._occupied_segments[f"{segment}_large"] = color_is_large
                else:
                    location = "best"

                colorbar_label = (
                    self.color if self.color != "depth" and self.color != "label" else "Category"
                )
                legend = self.ax.legend(
                    handles=legend_elements,
                    title=colorbar_label,
                    loc=location,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="black",
                )
                legend.get_title().set_fontweight("bold")
                legend.set_clip_on(False)  # Prevent clipping outside axes
                self.ax.add_artist(legend)
            elif not is_categorical and self.show_colorbar:
                # Add colorbar for continuous colors
                self.colorbar = self.fig.colorbar(self.scatter, ax=self.ax)
                colorbar_label = self.color if self.color != "depth" else "Depth"
                self.colorbar.set_label(colorbar_label, fontsize=11, fontweight="bold")

    def _plot_by_groups(self, data: pd.DataFrame) -> None:
        """Plot data grouped by shape/well."""
        # Determine grouping
        if self.shape == "well":
            groups = data.groupby("well")
            group_label = "Well"
        else:
            groups = data.groupby("shape_val")
            group_label = self.shape

        # Define markers for different groups
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

        # Check if colors are categorical (check once for all data)
        is_categorical = False
        category_colors = {}
        if self.color:
            c_vals_all = data["color_val"].values
            is_categorical = self._is_categorical_color(c_vals_all)

            if is_categorical:
                # Prepare color mapping for categorical values
                unique_categories = pd.Series(c_vals_all).dropna().unique()
                n_categories = len(unique_categories)

                if n_categories <= len(DEFAULT_COLORS):
                    color_palette = DEFAULT_COLORS
                else:
                    cmap_obj = cm.get_cmap(self.colortemplate, n_categories)
                    color_palette = [cmap_obj(i) for i in range(n_categories)]

                category_colors = {
                    cat: color_palette[i % len(color_palette)]
                    for i, cat in enumerate(unique_categories)
                }

        # Track for colorbar (use first scatter)
        first_scatter = None

        for idx, (group_name, group_data) in enumerate(groups):
            x_vals = group_data["x"].values
            y_vals = group_data["y"].values

            # Determine marker
            marker = markers[idx % len(markers)]

            # Determine colors
            if self.color:
                c_vals_raw = group_data["color_val"].values

                if is_categorical:
                    # Map categorical values to colors
                    c_vals = [category_colors.get(val, DEFAULT_COLORS[0]) for val in c_vals_raw]
                    cmap = None
                    vmin = vmax = None
                else:
                    # Use continuous color mapping
                    c_vals = c_vals_raw
                    cmap = self.colortemplate
                    if self.color_range:
                        vmin, vmax = self.color_range
                    else:
                        # Use global range from all data
                        vmin, vmax = np.nanmin(data["color_val"]), np.nanmax(data["color_val"])
            else:
                c_vals = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
                cmap = None
                vmin = vmax = None

            # Determine sizes
            if self.size and "size_val" in group_data.columns:
                s_vals = group_data["size_val"].values
                # Normalize sizes to size_range
                s_min, s_max = np.nanmin(s_vals), np.nanmax(s_vals)
                if s_max > s_min:
                    s_normalized = (s_vals - s_min) / (s_max - s_min)
                    sizes = self.size_range[0] + s_normalized * (
                        self.size_range[1] - self.size_range[0]
                    )
                else:
                    sizes = self.marker_size
            else:
                sizes = self.marker_size

            # Create scatter plot for this group
            scatter = self.ax.scatter(
                x_vals,
                y_vals,
                c=c_vals,
                s=sizes,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=self.marker_alpha,
                edgecolors=self.edge_color,
                linewidths=self.edge_width,
                marker=marker,
                label=self._get_display_label(group_name, "shape"),
            )

            if first_scatter is None and self.color and not is_categorical:
                first_scatter = scatter

        # Check if we need both shape and color legends (grouped layout)
        need_shape_legend = self.show_legend
        need_color_legend = self.color and is_categorical and self.show_legend

        if need_shape_legend and need_color_legend:
            # Create grouped legends in the same region using optimized placement
            # Prepare shape legend handles (from scatter plots)
            shape_handles, _ = self.ax.get_legend_handles_labels()

            # Prepare color legend handles
            c_vals_all = data["color_val"].values
            unique_categories = pd.Series(c_vals_all).dropna().unique()
            color_handles = [
                Patch(
                    facecolor=category_colors[cat],
                    edgecolor=self.edge_color,
                    label=self._get_display_label(cat, "color"),
                )
                for cat in unique_categories
            ]

            # Determine if legends are large (more than 5 items)
            shape_is_large = len(shape_handles) > 5
            color_is_large = len(color_handles) > 5

            # Find optimal segment for the grouped legends
            if self._data is not None:
                segment, location = self._find_optimal_legend_segment(
                    self._data, legend_type="shape", is_large=shape_is_large or color_is_large
                )
                # Mark segment as occupied by both shape and color
                self._occupied_segments[segment] = "shape"
                self._occupied_segments[f"{segment}_large"] = shape_is_large or color_is_large
            else:
                location = "best"

            colorbar_label = (
                self.color if self.color != "depth" and self.color != "label" else "Category"
            )

            # Create grouped legends
            self._create_grouped_legends(
                shape_handles=shape_handles,
                shape_title=group_label,
                color_handles=color_handles,
                color_title=colorbar_label,
                location=location,
            )

        elif need_shape_legend:
            # Only shape legend needed
            shape_handles, _ = self.ax.get_legend_handles_labels()
            shape_is_large = len(shape_handles) > 5

            # Find optimal segment
            if self._data is not None:
                segment, location = self._find_optimal_legend_segment(
                    self._data, legend_type="shape", is_large=shape_is_large
                )
                # Mark segment as occupied
                self._occupied_segments[segment] = "shape"
                self._occupied_segments[f"{segment}_large"] = shape_is_large
            else:
                location = "best"

            legend = self.ax.legend(
                title=group_label, loc=location, frameon=True, framealpha=0.9, edgecolor="black"
            )
            legend.get_title().set_fontweight("bold")
            # Store the primary legend so it persists when regression legend is added
            self.ax.add_artist(legend)

        # Add colorbar for continuous color mapping
        if self.color and not is_categorical and self.show_colorbar and first_scatter:
            # Add continuous colorbar
            self.colorbar = self.fig.colorbar(first_scatter, ax=self.ax)
            colorbar_label = self.color if self.color != "depth" else "Depth"
            self.colorbar.set_label(colorbar_label, fontsize=11, fontweight="bold")

    def add_regression(
        self,
        regression_type: str,
        name: str | None = None,
        line_color: str = "red",
        line_width: float = 2,
        line_style: str = "-",
        line_alpha: float = 0.8,
        show_equation: bool = True,
        show_r2: bool = True,
        x_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> Crossplot:
        """Add a regression line to the crossplot.

        Parameters
        ----------
        regression_type : str
            Type of regression: "linear", "logarithmic", "exponential",
            "polynomial", or "power"
        name : str, optional
            Name for this regression. If None, uses regression_type.
        line_color : str, optional
            Color of regression line. Default: 'red'
        line_width : float, optional
            Width of regression line. Default: 2
        line_style : str, optional
            Style of regression line. Default: '-'
        line_alpha : float, optional
            Transparency of regression line. Default: 0.8
        show_equation : bool, optional
            Show equation in legend. Default: True
        show_r2 : bool, optional
            Show R-squared value in legend. Default: True
        x_range : tuple[float, float], optional
            Custom x-axis range for plotting the regression line.
            If None, uses the data range from fitting.
        **kwargs
            Additional arguments for regression (e.g., degree for polynomial)

        Returns
        -------
        Crossplot
            Self for method chaining

        Examples
        --------
        >>> plot = well.Crossplot(x="RHOB", y="NPHI")
        >>> plot.add_regression("linear")
        >>> plot.add_regression("polynomial", degree=2, line_color="blue")
        >>> plot.add_regression("linear", x_range=(0, 10))  # Custom range
        >>> plot.show()
        """
        # Ensure data is prepared
        data = self._prepare_data()
        x_vals = data["x"].values
        y_vals = data["y"].values

        # Remove any remaining NaN values
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_clean = x_vals[mask]
        y_clean = y_vals[mask]

        if len(x_clean) < 2:
            raise ValueError("Need at least 2 valid data points for regression")

        # Create regression object using factory function
        reg = _create_regression(regression_type, **kwargs)

        # Fit regression
        try:
            reg.fit(x_clean, y_clean)
        except ValueError as e:
            raise ValueError(f"Failed to fit {regression_type} regression: {e}") from e

        # Recalculate R-squared in log space if y-axis is log scale
        if self.y_log:
            y_pred = reg.predict(x_clean)
            reg._calculate_metrics(x_clean, y_clean, y_pred, use_log_space=True)

        # Store regression in nested structure
        reg_name = name if name else regression_type
        self._store_regression(regression_type, reg_name, reg)

        # Plot regression line if figure exists, otherwise store for later
        if self.ax is not None:
            # Get plot data using the regression helper method
            try:
                x_line, y_line = reg.get_plot_data(x_range=x_range, num_points=200)
            except ValueError as e:
                warnings.warn(
                    f"Could not generate plot data for {regression_type} regression: {e}",
                    stacklevel=2,
                )
                return self

            # Create label using formatter
            label = self._format_regression_label(
                reg_name, reg, include_equation=show_equation, include_r2=show_r2
            )

            # Plot line
            line = self.ax.plot(
                x_line,
                y_line,
                color=line_color,
                linewidth=line_width,
                linestyle=line_style,
                alpha=line_alpha,
                label=label,
            )[0]

            self.regression_lines[reg_name] = line

            # Update regression legend
            self._update_regression_legend()
        else:
            # Store for later when plot() is called
            self._pending_regressions.append(
                {
                    "regression_type": regression_type,
                    "name": name,
                    "line_color": line_color,
                    "line_width": line_width,
                    "line_style": line_style,
                    "line_alpha": line_alpha,
                    "show_equation": show_equation,
                    "show_r2": show_r2,
                    "x_range": x_range,
                    "kwargs": kwargs,
                }
            )

        return self

    def remove_regression(self, name: str, regression_type: str | None = None) -> Crossplot:
        """Remove a regression from the plot.

        Parameters
        ----------
        name : str
            Name of regression to remove
        regression_type : str, optional
            Type of regression. If None, searches all types for the name.

        Returns
        -------
        Crossplot
            Self for method chaining
        """
        # Remove from nested structure
        if regression_type:
            # Remove from specific type
            if regression_type in self._regressions and name in self._regressions[regression_type]:
                del self._regressions[regression_type][name]
                # Clean up empty type dict
                if not self._regressions[regression_type]:
                    del self._regressions[regression_type]
        else:
            # Search all types for the name
            for reg_type in list(self._regressions.keys()):
                if name in self._regressions[reg_type]:
                    del self._regressions[reg_type][name]
                    # Clean up empty type dict
                    if not self._regressions[reg_type]:
                        del self._regressions[reg_type]

        # Remove line from plot
        if name in self.regression_lines:
            line = self.regression_lines[name]
            line.remove()
            del self.regression_lines[name]
            # Update legend
            if self.ax is not None:
                self.ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="black")

        return self

    def show(self) -> None:
        """Display the crossplot in Jupyter or interactive environment."""
        if self.fig is None:
            self.plot()

        plt.show()

    def save(self, filepath: str, dpi: int | None = None, bbox_inches: str = "tight") -> None:
        """Save the crossplot to a file.

        Parameters
        ----------
        filepath : str
            Output file path
        dpi : int, optional
            Resolution. If None, uses figure's dpi.
        bbox_inches : str, optional
            Bounding box mode. Default: 'tight'
        """
        if self.fig is None:
            self.plot()

        if dpi is None:
            dpi = self.dpi

        self.fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)

    def close(self) -> None:
        """Close the matplotlib figure and free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.scatter = None
            self.colorbar = None
            self.regression_lines = {}

    def __repr__(self) -> str:
        """String representation."""
        n_wells = len(self.wells)
        well_info = f"wells={n_wells}"
        if n_wells == 1:
            well_info = f"well='{self.wells[0].name}'"

        # Count total regressions across all types
        n_regressions = sum(len(regs) for regs in self._regressions.values())
        reg_info = f", regressions={n_regressions}" if n_regressions > 0 else ""

        return f"Crossplot({well_info}, " f"x='{self.x}', y='{self.y}'{reg_info})"
