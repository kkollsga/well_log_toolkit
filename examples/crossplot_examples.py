"""
Crossplot Examples - Comprehensive Guide
=========================================

This file demonstrates all the features of the new Crossplot functionality,
including single and multi-well plots, color/size/shape mapping, regression
analysis, and advanced customization.
"""

from logsuite import WellDataManager, LinearRegression, PolynomialRegression

# =============================================================================
# EXAMPLE 1: Basic Crossplot from a Single Well
# =============================================================================

def example_basic_crossplot(manager):
    """Create a simple crossplot of two properties."""
    well = manager.well_36_7_5_A  # Get a well

    # Create basic crossplot
    plot = well.Crossplot(x="RHOB", y="NPHI", title="Density vs Neutron Porosity")
    plot.show()

    # Can also save to file
    plot.save("basic_crossplot.png", dpi=300)


# =============================================================================
# EXAMPLE 2: Crossplot with All Requested Settings
# =============================================================================

def example_full_featured_crossplot(well):
    """Example matching the user's exact requirements."""
    plot = well.Crossplot(
        x="PHIE_2025",
        y="NetSand_2025",
        color="depth",              # Color by depth
        size="Sw_2025",             # Size by water saturation
        colortemplate="viridis",    # Color scheme
        color_range=[2000, 2500],   # Color scale range
        title="Cross Plot",         # Title
        marker_size=50,             # Base marker size
        edge_color="black",         # Marker outline color
        edge_width=0.5,             # Outline width
    )
    plot.show()


# =============================================================================
# EXAMPLE 3: Multi-Well Crossplot from Manager
# =============================================================================

def example_multiwell_crossplot(manager):
    """Create crossplot comparing multiple wells."""

    # Plot all wells with different markers for each
    plot = manager.Crossplot(
        x="PHIE",
        y="SW",
        shape="well",               # Each well gets different marker shape
        color="depth",              # Color by depth
        colortemplate="viridis",
        title="Multi-Well Porosity vs Water Saturation"
    )
    plot.show()

    # Plot specific wells only
    plot = manager.Crossplot(
        x="RHOB",
        y="NPHI",
        wells=["Well_A", "Well_B", "Well_C"],  # Specific wells
        shape="well",
        title="Selected Wells Comparison"
    )
    plot.show()


# =============================================================================
# EXAMPLE 4: Logarithmic Scales
# =============================================================================

def example_log_scales(well):
    """Use logarithmic scales for permeability plots."""
    plot = well.Crossplot(
        x="PERM",
        y="PHIE",
        x_log=True,                 # Logarithmic x-axis
        title="Permeability vs Porosity (Log Scale)"
    )
    plot.show()

    # Both axes logarithmic
    plot = well.Crossplot(
        x="PERM",
        y="Pressure",
        x_log=True,
        y_log=True,
        title="Log-Log Plot"
    )
    plot.show()


# =============================================================================
# EXAMPLE 5: Linear Regression
# =============================================================================

def example_linear_regression(well):
    """Add linear regression to crossplot."""

    # Create plot
    plot = well.Crossplot(
        x="RHOB",
        y="NPHI",
        title="Density vs Neutron with Linear Regression"
    )

    # Add regression
    plot.add_regression("linear", line_color="red", line_width=2)

    plot.show()

    # Access regression object
    linear_reg = plot.regressions["linear"]
    print(f"Equation: {linear_reg.equation()}")
    print(f"R²: {linear_reg.r_squared:.4f}")
    print(f"RMSE: {linear_reg.rmse:.4f}")

    # Use regression for predictions
    predictions = linear_reg([2.3, 2.4, 2.5])
    print(f"Predictions: {predictions}")


# =============================================================================
# EXAMPLE 6: Multiple Regression Lines
# =============================================================================

def example_multiple_regressions(well):
    """Add multiple regression types to compare fits."""

    plot = well.Crossplot(
        x="PHIE",
        y="SW",
        title="Multiple Regression Comparison"
    )

    # Add different regression types
    plot.add_regression("linear", line_color="red", line_width=2)
    plot.add_regression(
        "polynomial",
        degree=2,
        line_color="blue",
        line_width=2,
        name="quadratic"
    )
    plot.add_regression(
        "polynomial",
        degree=3,
        line_color="green",
        line_width=2,
        name="cubic"
    )

    plot.show()

    # Compare R² values
    for name, reg in plot.regressions.items():
        print(f"{name}: R² = {reg.r_squared:.4f}, RMSE = {reg.rmse:.4f}")


# =============================================================================
# EXAMPLE 7: Logarithmic and Exponential Regression
# =============================================================================

def example_nonlinear_regression(well):
    """Use logarithmic and exponential regression."""

    # Logarithmic regression (for data that grows then plateaus)
    plot = well.Crossplot(x="Time", y="Production", title="Production Decline")
    plot.add_regression("logarithmic", line_color="orange")
    plot.show()

    # Exponential regression (for exponential growth/decay)
    plot = well.Crossplot(x="Depth", y="Pressure", title="Pressure Increase")
    plot.add_regression("exponential", line_color="purple")
    plot.show()

    # Power regression (for power law relationships)
    plot = well.Crossplot(x="PERM", y="PHIE", title="Porosity-Permeability")
    plot.add_regression("power", line_color="brown")
    plot.show()


# =============================================================================
# EXAMPLE 8: Using Regression Classes Independently
# =============================================================================

def example_standalone_regression():
    """Use regression classes without crossplot."""
    import numpy as np

    # Generate data
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.9, 16.1, 17.8, 20.2])

    # Fit linear regression
    linear = LinearRegression()
    linear.fit(x, y)

    print(f"Equation: {linear.equation()}")
    print(f"R²: {linear.r_squared:.4f}")

    # Make predictions
    new_x = np.array([11, 12, 13])
    predictions = linear(new_x)  # Can call directly
    print(f"Predictions for {new_x}: {predictions}")

    # Or use predict method
    predictions = linear.predict(new_x)
    print(f"Predictions for {new_x}: {predictions}")

    # Try polynomial
    poly = PolynomialRegression(degree=2)
    poly.fit(x, y)
    print(f"\nPolynomial: {poly.equation()}")
    print(f"R²: {poly.r_squared:.4f}")


# =============================================================================
# EXAMPLE 9: Color and Size Mapping by Property
# =============================================================================

def example_property_mapping(well):
    """Map color and size to different properties."""

    # Color by zone, size by permeability
    plot = well.Crossplot(
        x="PHIE",
        y="SW",
        color="Zone",               # Color by zone
        size="PERM",                # Size by permeability
        colortemplate="plasma",     # Different color scheme
        size_range=(30, 300),       # Size range
        title="Porosity vs SW (colored by Zone, sized by PERM)"
    )
    plot.show()

    # Color by custom property
    plot = well.Crossplot(
        x="RHOB",
        y="NPHI",
        color="VSH",                # Color by shale volume
        colortemplate="RdYlGn_r",   # Red-Yellow-Green reversed
        title="Density vs Neutron (colored by VSH)"
    )
    plot.show()


# =============================================================================
# EXAMPLE 10: Shape Mapping by Property
# =============================================================================

def example_shape_mapping(well):
    """Use different markers for different property values."""

    # Shape by facies
    plot = well.Crossplot(
        x="PHIE",
        y="PERM",
        shape="Facies",             # Different marker for each facies
        color="depth",
        title="Porosity vs Permeability by Facies"
    )
    plot.show()


# =============================================================================
# EXAMPLE 11: Depth Range Filtering
# =============================================================================

def example_depth_filtering(well):
    """Plot only specific depth intervals."""

    # Focus on reservoir zone
    plot = well.Crossplot(
        x="PHIE",
        y="SW",
        depth_range=(2000, 2500),   # Only this depth range
        color="depth",
        title="Reservoir Zone Crossplot (2000-2500m)"
    )
    plot.show()


# =============================================================================
# EXAMPLE 12: Custom Styling
# =============================================================================

def example_custom_styling(well):
    """Customize plot appearance."""

    plot = well.Crossplot(
        x="RHOB",
        y="NPHI",
        title="Beautifully Styled Crossplot",
        figsize=(12, 10),           # Larger figure
        dpi=150,                    # Higher resolution
        marker="D",                 # Diamond markers
        marker_size=80,             # Larger markers
        marker_alpha=0.6,           # More transparent
        edge_color="darkblue",      # Dark blue outline
        edge_width=1.5,             # Thicker outline
        grid=True,                  # Show grid
        grid_alpha=0.2,             # Subtle grid
        xlabel="Bulk Density (g/cc)",     # Custom labels
        ylabel="Neutron Porosity (v/v)"
    )
    plot.show()


# =============================================================================
# EXAMPLE 13: Remove and Re-add Regressions
# =============================================================================

def example_dynamic_regressions(well):
    """Dynamically add and remove regressions."""

    plot = well.Crossplot(x="PHIE", y="PERM", title="Dynamic Regression Demo")

    # Add linear
    plot.add_regression("linear", line_color="red")
    plot.show()

    # Remove linear, add polynomial
    plot.remove_regression("linear")
    plot.add_regression("polynomial", degree=2, line_color="blue")
    plot.show()

    # Add multiple
    plot.add_regression("linear", line_color="red", name="linear_fit")
    plot.add_regression("exponential", line_color="green", name="exp_fit")
    plot.show()


# =============================================================================
# EXAMPLE 14: Regression Without Equation Display
# =============================================================================

def example_clean_regression(well):
    """Show regression line without equation in legend."""

    plot = well.Crossplot(x="RHOB", y="NPHI", title="Clean Regression Display")

    # Add regression without showing equation or R²
    plot.add_regression(
        "linear",
        show_equation=False,
        show_r2=False,
        line_color="red",
        line_style="--",
        line_width=3
    )

    plot.show()


# =============================================================================
# EXAMPLE 15: Multi-Well with Specific Property Colors
# =============================================================================

def example_multiwell_advanced(manager):
    """Advanced multi-well plot with property-based coloring."""

    plot = manager.Crossplot(
        x="PHIE",
        y="SW",
        shape="well",               # Different marker per well
        color="VSH",                # Color by shale volume (across all wells)
        size="PERM",                # Size by permeability
        colortemplate="viridis",
        color_range=[0, 0.5],       # VSH range 0-50%
        size_range=(20, 200),
        title="Multi-Well Analysis: Porosity vs SW"
    )

    plot.add_regression("linear", line_color="red", line_width=2)
    plot.show()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Load your data
    manager = WellDataManager()
    # manager.load_las("path/to/well1.las")
    # manager.load_las("path/to/well2.las")

    # Run examples (uncomment as needed)
    # well = manager.well_36_7_5_A

    # example_basic_crossplot(manager)
    # example_full_featured_crossplot(well)
    # example_multiwell_crossplot(manager)
    # example_log_scales(well)
    # example_linear_regression(well)
    # example_multiple_regressions(well)
    # example_nonlinear_regression(well)
    # example_standalone_regression()
    # example_property_mapping(well)
    # example_shape_mapping(well)
    # example_depth_filtering(well)
    # example_custom_styling(well)
    # example_dynamic_regressions(well)
    # example_clean_regression(well)
    # example_multiwell_advanced(manager)

    print("Examples ready! Uncomment the examples you want to run.")
