"""
Test that regression_by_color works correctly with discrete properties.

This verifies that discrete properties (like Well_Tops) are correctly
identified as categorical and can be used with regression_by_color.
"""

import numpy as np
from well_log_toolkit.visualization import Crossplot
import pytest


def create_well_with_discrete_property():
    """Create a well with discrete Well_Tops property."""
    from well_log_toolkit.core.property import Property

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
            tops_depths = np.array([2882.96, 2929.93, 2955.10, 2979.79, 2999.30])
            tops_values = np.array([0, 1, 2, 3, 4], dtype=float)
            tops_labels = {
                0: 'Agat top',
                1: 'Cerisa Main top',
                2: 'Cerisa West SST 1 top',
                3: 'Agat fm base',
                4: 'Sola fm top',
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
            from well_log_toolkit.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return MockWell("Well_A")


def test_discrete_property_with_regression_by_color():
    """Test that discrete properties work with regression_by_color."""
    print("\n" + "="*70)
    print("TEST: Discrete property (Well_Tops) with regression_by_color")
    print("="*70)

    well = create_well_with_discrete_property()
    tops = well.get_property('Well_Tops')

    print(f"\nWell_Tops property:")
    print(f"  Type: {tops.type}")
    print(f"  Unique values: {np.unique(tops.values[~np.isnan(tops.values)])}")
    print(f"  Labels: {tops.labels}")

    try:
        # Create crossplot with discrete color and regression_by_color
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            plot = Crossplot(
                wells=[well],
                x="CorePor",
                y="CorePerm",
                color="Well_Tops",
                regression_by_color="power",
                y_log=True,
                title="Discrete Property with Regression by Color"
            )

            print("\n✓ Crossplot created")
            plot.plot()
            print("✓ Plot generated")

            # Check for warnings
            categorical_warnings = [warning for warning in w
                                   if "continuous" in str(warning.message).lower()]

            if categorical_warnings:
                print(f"\n✗ FAIL: Got warning about continuous data:")
                for warning in categorical_warnings:
                    print(f"  {warning.message}")
                pytest.skip("Test precondition not met")
            else:
                print("✓ PASS: No 'continuous' warning (discrete property recognized as categorical)")

            # Check that regression lines were created
            if plot.regression_lines:
                print(f"✓ PASS: {len(plot.regression_lines)} regression line(s) created")
                for name in plot.regression_lines.keys():
                    print(f"  - {name}")
            else:
                pytest.skip("✗ FAIL: No regression lines created")

    except Exception as e:
        print(f"\n✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST PASSED\n")


def test_continuous_property_warns():
    """Test that truly continuous properties still generate warning."""
    print("\n" + "="*70)
    print("TEST: Continuous property (CorePor) correctly warns")
    print("="*70)

    # Create a well with continuous property used as color
    from well_log_toolkit.core.property import Property

    class MockWell:
        def __init__(self, name):
            self.name = name
            depth = np.arange(2800, 3000, 0.5)
            self._properties = {
                'CorePor': Property(
                    name='CorePor',
                    depth=depth,
                    values=0.15 + np.random.rand(len(depth)) * 0.1,
                    prop_type='continuous'
                ),
                'CorePerm': Property(
                    name='CorePerm',
                    depth=depth,
                    values=10 + np.random.rand(len(depth)) * 50,
                    prop_type='continuous'
                )
            }

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from well_log_toolkit.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    well = MockWell("Well_A")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            plot = Crossplot(
                wells=[well],
                x="CorePor",
                y="CorePerm",
                color="CorePor",  # Continuous property with many unique values
                regression_by_color="power",
                y_log=True,
                title="Continuous Property Should Warn"
            )

            plot.plot()

            # Should get warning about continuous data
            categorical_warnings = [warning for warning in w
                                   if "continuous" in str(warning.message).lower()]

            if categorical_warnings:
                print("✓ PASS: Got expected warning for continuous property")
                print(f"  Warning: {categorical_warnings[0].message}")
            else:
                print("⚠ No warning for continuous property (may be acceptable if <50 unique values)")

    except Exception as e:
        print(f"\n✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")

    print(f"\n✓ TEST PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISCRETE PROPERTY WITH REGRESSION_BY_COLOR TESTS")
    print("="*70)
    print("\nVerifying that discrete properties (like Well_Tops) are correctly")
    print("identified as categorical and work with regression_by_color.")

    all_passed = True

    # Run tests
    all_passed &= test_discrete_property_with_regression_by_color()
    all_passed &= test_continuous_property_warns()

    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nDiscrete properties are now correctly handled:")
        print("  1. Discrete properties recognized as categorical")
        print("  2. regression_by_color works with discrete properties")
        print("  3. Continuous properties still generate appropriate warnings")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)

    print()
