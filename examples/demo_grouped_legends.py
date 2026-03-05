"""
Visual demonstration of grouped legend layout for crossplots.

This script creates example plots showing how legends are grouped together
in the same 1/9th section without overlap.
"""

import numpy as np
import matplotlib.pyplot as plt
from logsuite.visualization import Crossplot


def create_demo_wells(num_wells=3):
    """Create mock wells with realistic data for demo."""
    class MockProperty:
        def __init__(self, values, depth):
            self.values = values
            self.depth = depth

    class MockWell:
        def __init__(self, name, seed):
            self.name = name
            np.random.seed(seed)
            depth = np.linspace(2000, 2100, 50)

            # Create realistic porosity and permeability data
            base_por = 0.15 + np.random.rand() * 0.1
            base_perm = 10 + np.random.rand() * 50

            self._properties = {
                'CorePor': MockProperty(
                    base_por + np.random.randn(50) * 0.03,
                    depth
                ),
                'CorePerm': MockProperty(
                    base_perm * np.exp(np.random.randn(50) * 0.3),
                    depth
                ),
                'SWPor': MockProperty(
                    base_por * 0.9 + np.random.randn(50) * 0.025,
                    depth
                ),
                'SWPerm': MockProperty(
                    base_perm * 0.8 * np.exp(np.random.randn(50) * 0.35),
                    depth
                ),
            }

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from logsuite.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return [MockWell(f"Well_{chr(65+i)}", seed=i*10) for i in range(num_wells)]


def demo_grouped_legends():
    """Demonstrate grouped legends with layers (Core + Sidewall data)."""
    print("\n" + "="*70)
    print("DEMO: Grouped Legends with Layers")
    print("="*70)

    wells = create_demo_wells(3)

    print("\nCreating crossplot with:")
    print("  - 3 wells (Well_A, Well_B, Well_C)")
    print("  - 2 layers (Core and Sidewall data)")
    print("  - Shape mapped to layer type (circles vs squares)")
    print("  - Color mapped to well (discrete colors)")
    print("  - Y-axis in log scale")

    plot = Crossplot(
        wells=wells,
        layers={
            "Core": ["CorePor", "CorePerm"],
            "Sidewall": ["SWPor", "SWPerm"]
        },
        y_log=True,
        title="Core vs Sidewall Porosity-Permeability",
        xlabel="Porosity (fraction)",
        ylabel="Permeability (mD)",
        figsize=(10, 8)
    )

    # Generate the plot
    plot.plot()

    print("\n✓ Plot generated successfully!")
    print("\nLegend Layout:")
    print("  - Both 'label' and 'well' legends are grouped in the same region")
    print("  - They are stacked vertically (on edge location)")
    print("  - No overlap between legends")
    print("  - Smart placement based on data density")

    # Save the figure
    output_path = "demo_grouped_legends.png"
    plot.save(output_path, dpi=150)
    print(f"\n✓ Saved to: {output_path}")

    return plot


def demo_comparison():
    """Show comparison of different legend configurations."""
    print("\n" + "="*70)
    print("DEMO: Comparison of Legend Configurations")
    print("="*70)

    wells = create_demo_wells(3)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Legend Layout Comparison", fontsize=14, fontweight='bold')

    # Configuration 1: With layers (grouped legends)
    print("\nConfiguration 1: Layers with shape + color legends (GROUPED)")
    plot1 = Crossplot(
        wells=wells,
        layers={
            "Core": ["CorePor", "CorePerm"],
            "Sidewall": ["SWPor", "SWPerm"]
        },
        y_log=True,
        title="Grouped Legends (Layers)",
        xlabel="Porosity",
        ylabel="Permeability (mD)"
    )

    # Manually set up the plot on first axis
    plot1.fig = fig
    plot1.ax = axes[0]
    plot1.plot()

    # Configuration 2: Simple multi-well (single legend)
    print("Configuration 2: Simple multi-well plot (SINGLE LEGEND)")
    plot2 = Crossplot(
        wells=wells,
        x="CorePor",
        y="CorePerm",
        shape="well",
        y_log=True,
        title="Single Legend (No Layers)",
        xlabel="Core Porosity",
        ylabel="Core Permeability (mD)"
    )

    # Manually set up the plot on second axis
    plot2.fig = fig
    plot2.ax = axes[1]
    plot2.plot()

    plt.tight_layout()

    # Save comparison
    output_path = "demo_legend_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to: {output_path}")

    print("\nKey Differences:")
    print("  Left plot: Grouped legends (shape + color) in same region")
    print("  Right plot: Single legend (shape only)")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GROUPED LEGENDS VISUAL DEMONSTRATION")
    print("="*70)

    # Run demos
    demo_grouped_legends()
    demo_comparison()

    print("\n" + "="*70)
    print("✓ DEMOS COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_grouped_legends.png")
    print("  - demo_legend_comparison.png")
    print("\nYou can now view these files to see the grouped legend layout in action!")
    print()
