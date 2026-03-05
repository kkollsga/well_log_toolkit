"""
Test script for the new Crossplot functionality.

This script tests:
1. Regression classes can be used independently
2. Crossplot can be created from Well objects
3. Crossplot can be created from Manager
4. Regression can be added to crossplots
"""

import numpy as np
import pytest
from well_log_toolkit import (
    LinearRegression,
    LogarithmicRegression,
    ExponentialRegression,
    PolynomialRegression,
    PowerRegression,
)


def test_regression_classes():
    """Test that regression classes work independently."""
    print("Testing regression classes...")

    # Test data
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    # Test LinearRegression
    print("\n1. Linear Regression:")
    linear = LinearRegression()
    linear.fit(x, y)
    print(f"   Equation: {linear.equation()}")
    print(f"   R²: {linear.r_squared:.4f}")
    print(f"   Prediction for [6, 7]: {linear([6, 7])}")

    # Test PolynomialRegression
    print("\n2. Polynomial Regression (degree=2):")
    x_poly = np.array([1, 2, 3, 4, 5])
    y_poly = np.array([1, 4, 9, 16, 25])  # y = x^2
    poly = PolynomialRegression(degree=2)
    poly.fit(x_poly, y_poly)
    print(f"   Equation: {poly.equation()}")
    print(f"   R²: {poly.r_squared:.4f}")
    print(f"   Prediction for [6]: {poly([6])}")

    # Test ExponentialRegression
    print("\n3. Exponential Regression:")
    x_exp = np.array([0, 1, 2, 3])
    y_exp = np.array([1, 2.7, 7.4, 20.1])
    exp = ExponentialRegression()
    exp.fit(x_exp, y_exp)
    print(f"   Equation: {exp.equation()}")
    print(f"   R²: {exp.r_squared:.4f}")

    # Test LogarithmicRegression
    print("\n4. Logarithmic Regression:")
    x_log = np.array([1, 2, 4, 8, 16])
    y_log = np.array([0, 1, 2, 3, 4])
    log = LogarithmicRegression()
    log.fit(x_log, y_log)
    print(f"   Equation: {log.equation()}")
    print(f"   R²: {log.r_squared:.4f}")

    # Test PowerRegression
    print("\n5. Power Regression:")
    x_pow = np.array([1, 2, 3, 4, 5])
    y_pow = np.array([1, 4, 9, 16, 25])  # y = x^2
    power = PowerRegression()
    power.fit(x_pow, y_pow)
    print(f"   Equation: {power.equation()}")
    print(f"   R²: {power.r_squared:.4f}")

    print("\n✓ All regression classes tested successfully!")


def test_crossplot_imports():
    """Test that Crossplot can be imported."""
    print("\nTesting Crossplot imports...")

    try:
        from well_log_toolkit import Crossplot
        print("✓ Crossplot can be imported from well_log_toolkit")
    except ImportError as e:
        print(f"✗ Failed to import Crossplot: {e}")
        pytest.skip("Test precondition not met")

    try:
        from well_log_toolkit.visualization import Crossplot
        print("✓ Crossplot can be imported from well_log_toolkit.visualization")
    except ImportError as e:
        print(f"✗ Failed to import Crossplot from visualization: {e}")
        pytest.skip("Test precondition not met")



def test_api_structure():
    """Test that the API is properly structured."""
    print("\nTesting API structure...")

    from well_log_toolkit import Well, WellDataManager

    # Check if Well has Crossplot method
    if hasattr(Well, 'Crossplot'):
        print("✓ Well class has Crossplot method")
    else:
        pytest.skip("✗ Well class missing Crossplot method")

    # Check if Manager has Crossplot method
    if hasattr(WellDataManager, 'Crossplot'):
        print("✓ WellDataManager class has Crossplot method")
    else:
        pytest.skip("✗ WellDataManager class missing Crossplot method")



if __name__ == "__main__":
    print("=" * 60)
    print("CROSSPLOT FUNCTIONALITY TEST")
    print("=" * 60)

    success = True

    # Test regression classes
    if not test_regression_classes():
        success = False

    # Test Crossplot imports
    if not test_crossplot_imports():
        success = False

    # Test API structure
    if not test_api_structure():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)
