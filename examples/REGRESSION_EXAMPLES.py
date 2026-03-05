"""
Quick reference examples for regression parameters in Crossplot.
All three parameters (regression, regression_by_color, regression_by_group)
support both string and dict formats.
"""

# ============================================================================
# Example 1: regression parameter
# ============================================================================

# String format
plot = Crossplot(well, x='RHOB', y='NPHI', regression='linear')

# Dict format with custom styling
plot = Crossplot(
    well,
    x='RHOB',
    y='NPHI',
    regression={
        'type': 'polynomial',
        'degree': 3,
        'line_color': 'darkred',
        'line_width': 2.5,
        'line_style': '-',
        'line_alpha': 0.9,
        'x_range': (1.5, 3.0)  # Custom x-range for plotting
    }
)

# ============================================================================
# Example 2: regression_by_group parameter
# ============================================================================

# String format - auto-generates different colors for each well
plot = Crossplot(
    [well1, well2, well3],
    x='PHIE',
    y='SW',
    shape='well',
    regression_by_group='linear'
)

# Dict format with custom styling - each well gets different auto color
plot = Crossplot(
    [well1, well2, well3],
    x='PHIE',
    y='SW',
    shape='well',
    regression_by_group={
        'type': 'linear',
        'line_width': 2,
        'line_style': '--',
        'line_alpha': 0.8
        # Note: no line_color specified = auto-generate different colors per group
    }
)

# Dict format with SAME color for all groups
plot = Crossplot(
    [well1, well2, well3],
    x='PHIE',
    y='SW',
    shape='well',
    regression_by_group={
        'type': 'linear',
        'line_color': 'blue',  # All wells use blue
        'line_width': 2
    }
)

# ============================================================================
# Example 3: regression_by_color parameter
# ============================================================================

# String format - requires categorical color property
plot = Crossplot(
    well,
    x='PHIE',
    y='PERM',
    color='facies',  # Categorical property (e.g., Sand, Shale, Limestone)
    regression_by_color='polynomial'
)

# Dict format with custom styling - each facies gets different auto color
plot = Crossplot(
    well,
    x='PHIE',
    y='PERM',
    color='facies',
    regression_by_color={
        'type': 'polynomial',
        'degree': 2,
        'line_width': 1.5,
        'line_style': ':',
        'line_alpha': 0.7
        # Note: no line_color = auto-generate different colors per color group
    }
)

# Dict format with SAME color for all facies
plot = Crossplot(
    well,
    x='PHIE',
    y='PERM',
    color='facies',
    regression_by_color={
        'type': 'linear',
        'line_color': 'green',  # All facies use green
        'line_width': 2
    }
)

# ============================================================================
# Example 4: Combining multiple regression types
# ============================================================================

# Overall regression + group-specific regressions
plot = Crossplot(
    [well1, well2, well3],
    x='PHIE',
    y='SW',
    shape='well',
    regression={
        'type': 'linear',
        'line_color': 'black',
        'line_width': 3,
        'line_style': '-'
    },
    regression_by_group={
        'type': 'polynomial',
        'degree': 2,
        'line_style': '--'
        # No color = each well gets different color
    }
)
plot.plot()

# Access the regressions
all_regs = plot.regression()
# Returns: {"linear": {...}, "polynomial": {...}}

linear_regs = plot.regression('linear')
# Returns: {"Overall linear": RegObj}

poly_regs = plot.regression('polynomial')
# Returns: {"well=Well_1": RegObj, "well=Well_2": RegObj, "well=Well_3": RegObj}

# ============================================================================
# Example 5: Accessing regression properties
# ============================================================================

plot = Crossplot(
    [well1, well2],
    x='RHOB',
    y='NPHI',
    shape='well',
    regression_by_group='linear'
)
plot.plot()

# Get all linear regressions
linear_regs = plot.regression('linear')

for name, reg in linear_regs.items():
    print(f"\n{name}:")
    print(f"  Equation: {reg.equation()}")
    print(f"  R²: {reg.r_squared:.4f}")
    print(f"  RMSE: {reg.rmse:.4f}")
    print(f"  X Range: {reg.x_range}")

    # For LinearRegression specifically
    if hasattr(reg, 'slope'):
        print(f"  Slope: {reg.slope:.4f}")
        print(f"  Intercept: {reg.intercept:.4f}")

    # Get plot data (uses stored x_range by default)
    x_plot, y_plot = reg.get_plot_data()

    # Or with custom range for extrapolation
    x_plot_extended, y_plot_extended = reg.get_plot_data(x_range=(0, 5))

# ============================================================================
# Example 6: Manual regression addition with custom range
# ============================================================================

plot = Crossplot(well, x='RHOB', y='NPHI')
plot.plot()

# Add regression that extends beyond data range
plot.add_regression(
    'linear',
    name='Extended Linear',
    x_range=(1.0, 4.0),  # Extends beyond data range
    line_color='purple',
    line_width=2
)

# Add another with default range (data range only)
plot.add_regression(
    'polynomial',
    degree=2,
    name='Poly Fit',
    line_color='orange'
    # No x_range = uses data range automatically
)

plot.show()
