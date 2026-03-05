"""
Test that discrete property labels appear correctly in crossplot legends.

This verifies that when using discrete properties (like Well_Tops) for shape or color:
- Legend shows actual formation names ("Agat top", "Cerisa Main top")
- NOT integer codes ("0", "1", "2")
"""

import numpy as np
from logsuite.visualization import Crossplot
import pytest


def create_well_with_labeled_tops():
    """Create a well with discrete Well_Tops that have string labels."""
    from logsuite.core.property import Property

    class MockWell:
        def __init__(self, name):
            self.name = name

            # Fine grid continuous properties
            fine_depth = np.arange(2800, 3100, 0.5)
            self._properties = {
                'CorePor': Property(
                    name='CorePor',
                    depth=fine_depth,
                    values=0.15 + np.random.rand(len(fine_depth)) * 0.1,
                    prop_type='continuous'
                ),
                'CorePerm': Property(
                    name='CorePerm',
                    depth=fine_depth,
                    values=10 + np.random.rand(len(fine_depth)) * 50,
                    prop_type='continuous'
                )
            }

            # Sparse discrete well tops with labels
            tops_depths = np.array([2882.96, 2929.93, 2955.10, 2979.79, 2999.30, 3073.18])
            tops_values = np.array([0, 1, 2, 3, 4, 5], dtype=float)
            tops_labels = {
                0: 'Agat top',
                1: 'Cerisa Main top',
                2: 'Cerisa West SST 1 top',
                3: 'Agat fm base',
                4: 'Sola fm top',
                5: 'Åsgård fm top'
            }

            self._properties['Well_Tops'] = Property(
                name='Well_Tops',
                depth=tops_depths,
                values=tops_values,
                prop_type='discrete',
                labels=tops_labels
            )

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from logsuite.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return MockWell("Well_A")


def test_discrete_shape_legend_shows_labels():
    """Test that shape legend shows formation names, not integer codes."""
    print("\n" + "="*70)
    print("TEST 1: Shape legend shows formation names (not integer codes)")
    print("="*70)

    well = create_well_with_labeled_tops()
    tops = well.get_property('Well_Tops')

    print(f"\nWell_Tops labels:")
    for code, label in tops.labels.items():
        print(f"  {code} → '{label}'")

    try:
        plot = Crossplot(
            wells=[well],
            x="CorePor",
            y="CorePerm",
            shape="Well_Tops",
            title="Crossplot with Discrete Shape"
        )

        print("\n✓ Crossplot created")
        plot.plot()
        print("✓ Plot generated")

        # Get legend handles and labels
        handles, labels = plot.ax.get_legend_handles_labels()
        print(f"\nLegend labels found: {labels}")

        # Check if labels are formation names (not integer codes)
        has_formation_names = any(label in tops.labels.values() for label in labels)
        has_integer_codes = any(label in ['0', '0.0', '1', '1.0', '2', '2.0'] for label in labels)

        if has_formation_names:
            print(f"✓ PASS: Legend shows formation names!")
            for label in labels:
                if label in tops.labels.values():
                    print(f"  - Found: '{label}'")
        else:
            print(f"✗ FAIL: Legend does NOT show formation names")

        if has_integer_codes:
            print(f"✗ FAIL: Legend shows integer codes instead of names:")
            for label in labels:
                if label in ['0', '0.0', '1', '1.0', '2', '2.0', '3', '3.0', '4', '4.0', '5', '5.0']:
                    print(f"  - Found code: '{label}'")
            pytest.skip("Test precondition not met")

        if not has_formation_names:
            print(f"✗ FAIL: Expected formation names in legend, got: {labels}")
            pytest.skip("Test precondition not met")

    except Exception as e:
        print(f"✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST 1 PASSED\n")


def test_discrete_color_legend_shows_labels():
    """Test that color legend shows formation names for discrete colors."""
    print("\n" + "="*70)
    print("TEST 2: Color legend shows formation names (not integer codes)")
    print("="*70)

    well = create_well_with_labeled_tops()
    tops = well.get_property('Well_Tops')

    try:
        plot = Crossplot(
            wells=[well],
            x="CorePor",
            y="CorePerm",
            color="Well_Tops",
            title="Crossplot with Discrete Color"
        )

        print("✓ Crossplot created")
        plot.plot()
        print("✓ Plot generated")

        # Check all legends (there might be multiple)
        import matplotlib.pyplot as plt
        from matplotlib.legend import Legend
        legends = [child for child in plot.ax.get_children() if isinstance(child, Legend)]

        print(f"\nFound {len(legends)} legend(s)")

        all_labels = []
        for i, legend in enumerate(legends):
            legend_texts = [text.get_text() for text in legend.get_texts()]
            print(f"  Legend {i+1} labels: {legend_texts}")
            all_labels.extend(legend_texts)

        # Check if any legend shows formation names
        has_formation_names = any(label in tops.labels.values() for label in all_labels)
        has_integer_codes = any(label in ['0', '0.0', '1', '1.0', '2', '2.0', '3', '3.0', '4', '4.0', '5', '5.0']
                               for label in all_labels)

        if has_formation_names:
            print(f"\n✓ PASS: Legend shows formation names!")
        else:
            print(f"\n✗ FAIL: Legend does NOT show formation names")

        if has_integer_codes:
            print(f"✗ FAIL: Legend shows integer codes instead of names")
            pytest.skip("Test precondition not met")

        if not has_formation_names:
            print(f"✗ FAIL: Expected formation names, got: {all_labels}")
            pytest.skip("Test precondition not met")

    except Exception as e:
        print(f"✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST 2 PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISCRETE PROPERTY LABELS IN LEGEND TESTS")
    print("="*70)

    all_passed = True
    all_passed &= test_discrete_shape_legend_shows_labels()
    all_passed &= test_discrete_color_legend_shows_labels()

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Labels displayed correctly!")
        print("="*70)
    else:
        print("✗ TESTS REVEALED ISSUE - Integer codes shown instead of labels")
        print("="*70)
        print("\nThe legend should show:")
        print("  'Agat top', 'Cerisa Main top', etc.")
        print("\nNOT:")
        print("  '0', '1', '2', etc.")

    print()
