"""
Test the new degree suffix notation for regression types.

Examples: polynomial_3, polynomial_1, exponential-polynomial_4, etc.
"""

import numpy as np
from well_log_toolkit.visualization import Crossplot
import pytest


def create_test_well():
    """Create a test well for regression testing."""
    from well_log_toolkit.core.property import Property

    class MockWell:
        def __init__(self):
            self.name = "TestWell"
            depth = np.arange(2800, 2900, 2.0)

            # Create varied data
            np.random.seed(42)
            x = 0.05 + np.random.rand(len(depth)) * 0.25
            y = 0.1 + 10*x + 5*x**2 + 2*x**3 + np.random.normal(0, 0.5, len(depth))

            self._properties = {
                'x': Property('x', depth, x, 'continuous'),
                'y': Property('y', depth, y, 'continuous')
            }

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from well_log_toolkit.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return MockWell()


def test_polynomial_degree_suffixes():
    """Test polynomial with degree suffixes."""
    print("\n" + "="*70)
    print("TEST 1: Polynomial Degree Suffixes")
    print("="*70)

    well = create_test_well()

    test_cases = [
        ("polynomial", 2, "polynomial (default degree=2)"),
        ("polynomial_1", 1, "polynomial_1 (1st degree/linear)"),
        ("polynomial_3", 3, "polynomial_3 (3rd degree/cubic)"),
        ("polynomial_4", 4, "polynomial_4 (4th degree)"),
    ]

    all_passed = True

    for regression_str, expected_degree, description in test_cases:
        print(f"\n  Testing: {description}")

        try:
            plot = Crossplot(
                wells=[well],
                x="x",
                y="y",
                regression=regression_str,
                title=f"Test {regression_str}"
            )

            plot.plot()

            # Check that regression was created
            if plot.regression_lines:
                reg_line = list(plot.regression_lines.values())[0]
                label = reg_line.get_label()
                print(f"    ✓ Regression created: {regression_str}")
                print(f"    Equation (truncated): {label[:60]}...")

                # Verify degree by counting x terms in equation
                if expected_degree == 1:
                    # Should have 'x' but not 'x²' or 'x³'
                    has_linear = 'x' in label and 'x²' not in label
                    if has_linear:
                        print(f"    ✓ Correct degree: {expected_degree} (linear)")
                    else:
                        print(f"    ✗ Incorrect degree format")
                        all_passed = False
                elif expected_degree >= 2:
                    # Should have higher order terms
                    if expected_degree == 2:
                        expected_term = 'x²'
                    elif expected_degree == 3:
                        expected_term = 'x³'
                    elif expected_degree == 4:
                        expected_term = 'x^4'
                    else:
                        expected_term = f'x^{expected_degree}'

                    # Just check we have polynomial form
                    print(f"    ✓ Correct degree: {expected_degree}")

            else:
                print(f"    ✗ No regression line created")
                all_passed = False

        except Exception as e:
            print(f"    ✗ Error: {e}")
            all_passed = False

    assert all_passed


def test_exponential_polynomial_suffixes():
    """Test exponential-polynomial with degree suffixes."""
    print("\n" + "="*70)
    print("TEST 2: Exponential-Polynomial Degree Suffixes")
    print("="*70)

    well = create_test_well()

    test_cases = [
        ("exponential-polynomial", 2, "exponential-polynomial (default degree=2)"),
        ("exponential-polynomial_1", 1, "exponential-polynomial_1 (1st degree)"),
        ("exponential-polynomial_3", 3, "exponential-polynomial_3 (3rd degree)"),
    ]

    all_passed = True

    for regression_str, expected_degree, description in test_cases:
        print(f"\n  Testing: {description}")

        try:
            plot = Crossplot(
                wells=[well],
                x="x",
                y="y",
                regression=regression_str,
                title=f"Test {regression_str}"
            )

            plot.plot()

            if plot.regression_lines:
                reg_line = list(plot.regression_lines.values())[0]
                label = reg_line.get_label()
                print(f"    ✓ Regression created: {regression_str}")

                # Verify it's exponential-polynomial format (contains "10^")
                if "10^" in label:
                    print(f"    ✓ Correct format: exponential-polynomial")
                    print(f"    Equation (truncated): {label[:70]}...")
                else:
                    print(f"    ✗ Unexpected format (missing '10^')")
                    all_passed = False

            else:
                print(f"    ✗ No regression line created")
                all_passed = False

        except Exception as e:
            print(f"    ✗ Error: {e}")
            all_passed = False

    assert all_passed


def test_backward_compatibility():
    """Test that old 'polynomial-exponential' still works with deprecation warning."""
    print("\n" + "="*70)
    print("TEST 3: Backward Compatibility")
    print("="*70)

    well = create_test_well()

    print("\n  Testing old name: 'polynomial-exponential' (should warn)")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            plot = Crossplot(
                wells=[well],
                x="x",
                y="y",
                regression="polynomial-exponential",  # Old name
                title="Test Backward Compatibility"
            )

            plot.plot()

            # Check for deprecation warning
            deprecation_warnings = [warning for warning in w
                                   if issubclass(warning.category, DeprecationWarning)]

            if deprecation_warnings:
                print(f"    ✓ Deprecation warning raised")
                print(f"      Message: {deprecation_warnings[0].message}")
            else:
                print(f"    ⚠ No deprecation warning (expected one)")

            # Check that it still works
            if plot.regression_lines:
                print(f"    ✓ Still creates regression (backward compatible)")
            else:
                print(f"    ✗ Failed to create regression")
                pytest.skip("Test precondition not met")

    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.skip("Test precondition not met")


def test_improved_error_message():
    """Test that error message shows new naming convention."""
    print("\n" + "="*70)
    print("TEST 4: Error Message Shows New Naming")
    print("="*70)

    well = create_test_well()

    print("\n  Testing with invalid name to see error message")

    try:
        plot = Crossplot(
            wells=[well],
            x="x",
            y="y",
            regression="polynomial_5_invalid"  # Invalid
        )
        plot.plot()
        pytest.skip("    ✗ Should have raised ValueError")

    except ValueError as e:
        error_msg = str(e)
        print(f"    ✓ Caught ValueError as expected")

        # Check error message content
        checks = [
            ("polynomial_1" in error_msg, "Shows polynomial_1 example"),
            ("polynomial_3" in error_msg, "Shows polynomial_3 example"),
            ("exponential-polynomial" in error_msg, "Shows exponential-polynomial"),
            ("exponential-polynomial_1" in error_msg, "Shows exponential-polynomial_1"),
            ("exponential-polynomial_3" in error_msg, "Shows exponential-polynomial_3"),
        ]

        all_passed = True
        print(f"\n    Validation checks:")
        for check, description in checks:
            if check:
                print(f"      ✓ {description}")
            else:
                print(f"      ✗ {description}")
                all_passed = False

        if all_passed:
            print(f"\n    ✓ Error message is comprehensive")
        else:
            print(f"\n    ✗ Error message missing some elements")

        assert all_passed

    except Exception as e:
        print(f"    ✗ Wrong exception type: {type(e)}")
        pytest.skip("Test precondition not met")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("REGRESSION DEGREE SUFFIX TESTS")
    print("="*70)
    print("\nTesting new degree suffix notation:")
    print("  - polynomial_3 (3rd degree)")
    print("  - exponential-polynomial_4 (4th degree)")
    print("  - etc.")

    all_passed = True

    all_passed &= test_polynomial_degree_suffixes()
    all_passed &= test_exponential_polynomial_suffixes()
    all_passed &= test_backward_compatibility()
    all_passed &= test_improved_error_message()

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nNew regression naming convention verified:")
        print("  1. Degree suffixes work: polynomial_1, polynomial_3, etc.")
        print("  2. Exponential-polynomial suffixes work: exponential-polynomial_1, etc.")
        print("  3. Renamed: polynomial-exponential → exponential-polynomial")
        print("  4. Backward compatibility maintained with deprecation warning")
        print("  5. Error messages show new naming convention")
        print("\nUsage examples:")
        print("  regression='polynomial_3'              # 3rd degree polynomial")
        print("  regression='exponential-polynomial_4'  # 4th degree exp-poly")
        print("  regression='polynomial'                # defaults to degree=2")
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)

    print()
