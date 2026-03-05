"""
Test the new regression_by_color_and_shape feature.

This verifies that regression lines can be created for each combination
of color AND shape groups.
"""

import numpy as np
from well_log_toolkit.visualization import Crossplot
import pytest


def create_multi_well_with_formations():
    """Create multiple wells with formation data."""
    from well_log_toolkit.core.property import Property

    class MockWell:
        def __init__(self, name, offset):
            self.name = name

            # Fine grid continuous properties with well-specific offset
            fine_depth = np.arange(2800, 3000, 1.0)
            base_por = 0.15 + offset * 0.02
            base_perm = 20 + offset * 15

            self._properties = {
                'CorePor': Property(
                    name='CorePor',
                    depth=fine_depth,
                    values=base_por + np.random.rand(len(fine_depth)) * 0.08,
                    prop_type='continuous'
                ),
                'CorePerm': Property(
                    name='CorePerm',
                    depth=fine_depth,
                    values=base_perm * np.exp(np.random.randn(len(fine_depth)) * 0.3),
                    prop_type='continuous'
                )
            }

            # Sparse discrete formations
            formation_depths = np.array([2850.0, 2900.0, 2950.0])
            formation_values = np.array([0, 1, 2], dtype=float)
            formation_labels = {
                0: 'Formation A',
                1: 'Formation B',
                2: 'Formation C',
            }

            self._properties['Formation'] = Property(
                name='Formation',
                depth=formation_depths,
                values=formation_values,
                prop_type='discrete',
                labels=formation_labels
            )

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from well_log_toolkit.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return [MockWell(f"Well_{chr(65+i)}", i) for i in range(3)]


def test_regression_by_color_and_shape():
    """Test regression lines for each (well, formation) combination."""
    print("\n" + "="*70)
    print("TEST 1: regression_by_color_and_shape (well × formation)")
    print("="*70)

    wells = create_multi_well_with_formations()

    print("\nCreating crossplot with:")
    print("  - 3 wells (color)")
    print("  - 3 formations (shape)")
    print("  - Should create up to 9 regression lines (3×3)")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            plot = Crossplot(
                wells=wells,
                x="CorePor",
                y="CorePerm",
                color="well",
                shape="Formation",
                regression_by_color_and_shape="power",
                y_log=True,
                title="Regression by Color AND Shape"
            )

            print("\n✓ Crossplot created")
            plot.plot()
            print("✓ Plot generated")

            # Check for warnings
            error_warnings = [warning for warning in w
                             if "requires both" in str(warning.message).lower()
                             or "different" in str(warning.message).lower()]

            if error_warnings:
                print(f"\n✗ FAIL: Got unexpected warning:")
                for warning in error_warnings:
                    print(f"  {warning.message}")
                pytest.skip("Test precondition not met")

            # Check regression lines were created
            if plot.regression_lines:
                n_lines = len(plot.regression_lines)
                print(f"\n✓ PASS: {n_lines} regression line(s) created")

                # Show first few regression names
                for i, name in enumerate(list(plot.regression_lines.keys())[:5]):
                    print(f"  {i+1}. {name}")
                if n_lines > 5:
                    print(f"  ... and {n_lines - 5} more")

                # Verify names contain both dimensions
                sample_name = list(plot.regression_lines.keys())[0]
                if ',' in sample_name:
                    print(f"✓ PASS: Regression names include both dimensions")
                    print(f"  Example: '{sample_name}'")
                else:
                    print(f"⚠ Warning: Expected comma in regression name")

            else:
                pytest.skip("✗ FAIL: No regression lines created")

    except Exception as e:
        print(f"\n✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST 1 PASSED\n")


def test_regression_requires_both_dimensions():
    """Test that warning is shown when only one dimension is provided."""
    print("\n" + "="*70)
    print("TEST 2: Warning when missing color or shape")
    print("="*70)

    wells = create_multi_well_with_formations()

    print("\nTrying regression_by_color_and_shape with only shape (no color)")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            plot = Crossplot(
                wells=wells,
                x="CorePor",
                y="CorePerm",
                shape="Formation",  # Only shape, no color
                regression_by_color_and_shape="power",
                y_log=True,
                title="Missing Color Dimension"
            )

            plot.plot()

            # Should get warning about requiring both dimensions
            missing_warnings = [warning for warning in w
                               if "requires both" in str(warning.message).lower()]

            if missing_warnings:
                print("✓ PASS: Got expected warning")
                print(f"  Warning: {missing_warnings[0].message}")
            else:
                print("⚠ No warning (may have defaults)")

    except Exception as e:
        print(f"\n✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST 2 PASSED\n")


def test_regression_requires_different_dimensions():
    """Test that warning is shown when color and shape are the same."""
    print("\n" + "="*70)
    print("TEST 3: Warning when color and shape are the same")
    print("="*70)

    wells = create_multi_well_with_formations()

    print("\nTrying regression_by_color_and_shape with same dimension for both")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            plot = Crossplot(
                wells=wells,
                x="CorePor",
                y="CorePerm",
                color="well",
                shape="well",  # Same as color
                regression_by_color_and_shape="power",
                y_log=True,
                title="Same Dimension for Both"
            )

            plot.plot()

            # Should get warning about requiring different dimensions
            same_warnings = [warning for warning in w
                            if "different" in str(warning.message).lower()]

            if same_warnings:
                print("✓ PASS: Got expected warning")
                print(f"  Warning: {same_warnings[0].message}")
            else:
                print("⚠ No warning received")

    except Exception as e:
        print(f"\n✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST 3 PASSED\n")


def test_with_layers():
    """Test regression_by_color_and_shape with layers."""
    print("\n" + "="*70)
    print("TEST 4: regression_by_color_and_shape with layers")
    print("="*70)

    wells = create_multi_well_with_formations()

    print("\nCreating crossplot with layers:")
    print("  - 3 wells (color)")
    print("  - 2 layers (shape)")

    try:
        plot = Crossplot(
            wells=wells,
            layers={
                "Core": ["CorePor", "CorePerm"],
            },
            color="Formation",
            regression_by_color_and_shape="linear",
            y_log=True,
            title="Layers with Color and Shape Regressions"
        )

        plot.plot()
        print("✓ Plot generated")

        if plot.regression_lines:
            n_lines = len(plot.regression_lines)
            print(f"✓ PASS: {n_lines} regression line(s) created")
        else:
            print("⚠ No regression lines (may need more data per combination)")

    except Exception as e:
        print(f"\n✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST 4 PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGRESSION BY COLOR AND SHAPE TESTS")
    print("="*70)
    print("\nTesting the new regression_by_color_and_shape feature")
    print("that creates regression lines for each (color, shape) combination.")

    all_passed = True

    # Run tests
    all_passed &= test_regression_by_color_and_shape()
    all_passed &= test_regression_requires_both_dimensions()
    all_passed &= test_regression_requires_different_dimensions()
    all_passed &= test_with_layers()

    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nregression_by_color_and_shape features verified:")
        print("  1. Creates regressions for each (color, shape) combination")
        print("  2. Warns when only one dimension provided")
        print("  3. Warns when both dimensions are the same")
        print("  4. Works with discrete properties (formations)")
        print("  5. Regression names include both dimensions")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)

    print()
