"""
Visual demonstration of optimized 9-segment legend placement.

This demo shows how legends are intelligently placed:
- Priority order: 1,9,4,6,3,7,2,8,5
- Avoids segments with >10% of datapoints
- Shape and color can share segments when small
"""

import numpy as np
import matplotlib.pyplot as plt
from pylog.visualization import Crossplot


def create_demo_wells_with_clustered_data():
    """Create wells with data clustered in the center (segment 5)."""
    from pylog.core.property import Property

    class MockWell:
        def __init__(self, name, offset):
            self.name = name
            depth = np.linspace(2000, 2100, 80)

            # Create data clustered in center with small offset per well
            base_por = 0.18 + offset * 0.02
            base_perm = 30 + offset * 10

            self._properties = {
                'CorePor': Property(
                    name='CorePor',
                    depth=depth,
                    values=base_por + np.random.randn(80) * 0.02,
                    prop_type='continuous'
                ),
                'CorePerm': Property(
                    name='CorePerm',
                    depth=depth,
                    values=base_perm * np.exp(np.random.randn(80) * 0.3),
                    prop_type='continuous'
                ),
                'SWPor': Property(
                    name='SWPor',
                    depth=depth,
                    values=base_por * 0.9 + np.random.randn(80) * 0.015,
                    prop_type='continuous'
                ),
                'SWPerm': Property(
                    name='SWPerm',
                    depth=depth,
                    values=base_perm * 0.8 * np.exp(np.random.randn(80) * 0.35),
                    prop_type='continuous'
                ),
            }

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from pylog.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return [MockWell(f"Well_{chr(65+i)}", i) for i in range(3)]


def demo_basic_optimization():
    """Show basic optimized placement with data in center."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Optimized Placement")
    print("="*70)
    print("\nData is clustered in CENTER of plot (segment 5)")
    print("Legend should prefer corners (segments 1, 9, etc.)")

    wells = create_demo_wells_with_clustered_data()

    plot = Crossplot(
        wells=wells,
        x="CorePor",
        y="CorePerm",
        shape="well",
        y_log=True,
        title="Optimized Placement: Data in Center",
        xlabel="Core Porosity (fraction)",
        ylabel="Core Permeability (mD)",
        figsize=(10, 8)
    )

    plot.plot()
    print(f"\n✓ Plot generated")
    print(f"  Legend placement: {plot._occupied_segments}")

    # Annotate segments on the plot
    ax = plot.ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Add segment labels for reference (in data coordinates)
    segment_positions = {
        1: (xlim[0] + 0.05 * (xlim[1] - xlim[0]), ylim[1] - 0.05 * (ylim[1] - ylim[0])),
        2: (xlim[0] + 0.50 * (xlim[1] - xlim[0]), ylim[1] - 0.05 * (ylim[1] - ylim[0])),
        3: (xlim[0] + 0.95 * (xlim[1] - xlim[0]), ylim[1] - 0.05 * (ylim[1] - ylim[0])),
        4: (xlim[0] + 0.05 * (xlim[1] - xlim[0]), ylim[0] + 0.50 * (ylim[1] - ylim[0])),
        5: (xlim[0] + 0.50 * (xlim[1] - xlim[0]), ylim[0] + 0.50 * (ylim[1] - ylim[0])),
        6: (xlim[0] + 0.95 * (xlim[1] - xlim[0]), ylim[0] + 0.50 * (ylim[1] - ylim[0])),
        7: (xlim[0] + 0.05 * (xlim[1] - xlim[0]), ylim[0] + 0.05 * (ylim[1] - ylim[0])),
        8: (xlim[0] + 0.50 * (xlim[1] - xlim[0]), ylim[0] + 0.05 * (ylim[1] - ylim[0])),
        9: (xlim[0] + 0.95 * (xlim[1] - xlim[0]), ylim[0] + 0.05 * (ylim[1] - ylim[0])),
    }

    for seg, (x, y) in segment_positions.items():
        # Highlight occupied segments
        if seg in plot._occupied_segments:
            ax.text(x, y, f'[{seg}]', fontsize=10, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax.text(x, y, f'{seg}', fontsize=8, color='gray', alpha=0.4)

    output_path = "demo_optimized_basic.png"
    plot.save(output_path, dpi=150)
    print(f"✓ Saved to: {output_path}")

    return plot


def demo_multiple_legends():
    """Show multiple legends with optimized placement."""
    print("\n" + "="*70)
    print("DEMO 2: Multiple Legends (Shape + Color + Regression)")
    print("="*70)
    print("\nMultiple legends should occupy different optimal segments")

    wells = create_demo_wells_with_clustered_data()

    plot = Crossplot(
        wells=wells,
        layers={
            "Core": ["CorePor", "CorePerm"],
            "Sidewall": ["SWPor", "SWPerm"]
        },
        regression_by_color="power",
        y_log=True,
        title="Multiple Legends with Optimized Placement",
        xlabel="Porosity (fraction)",
        ylabel="Permeability (mD)",
        figsize=(10, 8)
    )

    plot.plot()
    print(f"\n✓ Plot generated")
    print(f"  Occupied segments: {plot._occupied_segments}")

    # Count unique segments
    segments = {k for k in plot._occupied_segments.keys() if isinstance(k, int)}
    print(f"  Number of segments used: {len(segments)}")
    print(f"  Segments: {sorted(segments)}")

    output_path = "demo_optimized_multiple.png"
    plot.save(output_path, dpi=150)
    print(f"✓ Saved to: {output_path}")

    return plot


def demo_priority_order():
    """Demonstrate the priority order visually."""
    print("\n" + "="*70)
    print("DEMO 3: Priority Order Visualization")
    print("="*70)
    print("\nPriority order: 1 → 9 → 4 → 6 → 3 → 7 → 2 → 8 → 5")

    # Create figure showing the priority order
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw 3x3 grid
    for i in range(4):
        ax.axvline(i, color='black', linewidth=1)
        ax.axhline(i, color='black', linewidth=1)

    # Segment numbering and priority
    segments = {
        1: (0.5, 2.5, 1),    # (x, y, priority)
        2: (1.5, 2.5, 7),
        3: (2.5, 2.5, 5),
        4: (0.5, 1.5, 3),
        5: (1.5, 1.5, 9),
        6: (2.5, 1.5, 4),
        7: (0.5, 0.5, 6),
        8: (1.5, 0.5, 8),
        9: (2.5, 0.5, 2),
    }

    # Priority colors (gradient from green to red)
    priority_colors = {
        1: '#00ff00',  # Green (highest priority)
        2: '#66ff00',
        3: '#99ff00',
        4: '#ccff00',
        5: '#ffff00',  # Yellow
        6: '#ffcc00',
        7: '#ff9900',
        8: '#ff6600',
        9: '#ff0000',  # Red (lowest priority)
    }

    for seg, (x, y, priority) in segments.items():
        color = priority_colors[priority]

        # Draw colored box
        rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8,
                            facecolor=color, alpha=0.6, edgecolor='black')
        ax.add_patch(rect)

        # Add segment number and priority
        ax.text(x, y + 0.15, f'Segment {seg}',
               fontsize=14, fontweight='bold', ha='center')
        ax.text(x, y - 0.15, f'Priority: {priority}',
               fontsize=11, ha='center')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.set_title('Legend Placement Priority Order\n(Lower number = Higher priority)',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add legend
    priority_order_text = "Check order: 1 → 9 → 4 → 6 → 3 → 7 → 2 → 8 → 5"
    ax.text(1.5, -0.3, priority_order_text,
           fontsize=12, ha='center', style='italic')

    plt.tight_layout()
    output_path = "demo_priority_order.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved priority visualization to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("OPTIMIZED LEGEND PLACEMENT DEMONSTRATIONS")
    print("="*70)
    print("\nThese demos show the 9-segment priority-based placement algorithm")
    print("that intelligently avoids data-dense regions and minimizes overlap.")

    # Run demos
    demo_priority_order()
    plot1 = demo_basic_optimization()
    plot2 = demo_multiple_legends()

    print("\n" + "="*70)
    print("✓ ALL DEMOS COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_priority_order.png (shows segment priority)")
    print("  - demo_optimized_basic.png (single legend)")
    print("  - demo_optimized_multiple.png (multiple legends)")
    print("\nKey features demonstrated:")
    print("  ✓ Priority order: 1,9,4,6,3,7,2,8,5")
    print("  ✓ Avoids segments with >10% datapoints")
    print("  ✓ Multiple legends in different optimal segments")
    print("  ✓ Shape and color can share when small")
    print()
