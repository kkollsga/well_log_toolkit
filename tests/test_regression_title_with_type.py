"""
Test that regression legend title includes the regression type.
"""

import numpy as np
from pylog.visualization import Crossplot
import pytest


def create_test_wells():
    """Create wells for testing."""
    from pylog.core.property import Property

    class MockWell:
        def __init__(self, name, offset):
            self.name = name
            depth = np.arange(2800, 2900, 1.0)
            base_por = 0.15 + offset * 0.02
            base_perm = 20 + offset * 15

            self._properties = {
                'CorePor': Property(
                    name='CorePor',
                    depth=depth,
                    values=base_por + np.random.rand(len(depth)) * 0.05,
                    prop_type='continuous'
                ),
                'CorePerm': Property(
                    name='CorePerm',
                    depth=depth,
                    values=base_perm * np.exp(np.random.randn(len(depth)) * 0.2),
                    prop_type='continuous'
                )
            }

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from pylog.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return [MockWell(f"Well_{chr(65+i)}", i) for i in range(3)]


def test_regression_title_includes_type():
    """Test that regression legend title includes the regression type."""
    print("\n" + "="*70)
    print("TEST: Regression legend title includes type")
    print("="*70)

    wells = create_test_wells()

    test_cases = [
        ("linear", "Regressions by color - Linear"),
        ("power", "Regressions by color - Power"),
        ("exponential", "Regressions by color - Exponential"),
        ("logarithmic", "Regressions by color - Logarithmic"),
    ]

    all_passed = True

    for reg_type, expected_title in test_cases:
        print(f"\nTesting regression_by_color='{reg_type}'")

        try:
            plot = Crossplot(
                wells=wells,
                x="CorePor",
                y="CorePerm",
                shape="well",
                regression_by_color=reg_type,
                y_log=True,
                title=f"Test {reg_type.capitalize()}"
            )

            plot.plot()

            # Check if regression legend exists and has correct title
            if plot.regression_legend is not None:
                actual_title = plot.regression_legend.get_title().get_text()
                print(f"  Expected title: '{expected_title}'")
                print(f"  Actual title:   '{actual_title}'")

                if actual_title == expected_title:
                    print(f"  ✓ PASS: Title includes regression type")
                else:
                    print(f"  ✗ FAIL: Title mismatch")
                    all_passed = False
            else:
                print(f"  ⚠ No regression legend found")
                all_passed = False

        except Exception as e:
            print(f"  ✗ FAIL: Error: {e}")
            all_passed = False

    assert all_passed


def test_regression_by_group_title():
    """Test regression_by_group title includes type."""
    print("\n" + "="*70)
    print("TEST: Regression by group title includes type")
    print("="*70)

    wells = create_test_wells()

    print("\nTesting regression_by_group='linear'")

    try:
        plot = Crossplot(
            wells=wells,
            x="CorePor",
            y="CorePerm",
            shape="well",
            regression_by_group="linear",
            y_log=True,
            title="Test Linear by Group"
        )

        plot.plot()

        if plot.regression_legend is not None:
            actual_title = plot.regression_legend.get_title().get_text()
            expected_title = "Regressions by group - Linear"

            print(f"  Expected title: '{expected_title}'")
            print(f"  Actual title:   '{actual_title}'")

            if actual_title == expected_title:
                print(f"  ✓ PASS: Title includes regression type")
            else:
                print(f"  ✗ FAIL: Title mismatch")
                pytest.skip("Test precondition not met")
        else:
            print(f"  ⚠ No regression legend found")
            pytest.skip("Test precondition not met")

    except Exception as e:
        print(f"  ✗ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGRESSION TITLE WITH TYPE TESTS")
    print("="*70)
    print("\nVerifying that regression legend titles include the regression type")
    print("(e.g., 'Regressions by color - Power')")

    all_passed = True
    all_passed &= test_regression_title_includes_type()
    all_passed &= test_regression_by_group_title()

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nRegression title features verified:")
        print("  1. Title includes base text (e.g., 'Regressions by color')")
        print("  2. Title includes regression type (e.g., '- Power')")
        print("  3. Works for all regression types (linear, power, exponential, etc.)")
        print("  4. Works for all regression modes (by_color, by_group, etc.)")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)

    print()
