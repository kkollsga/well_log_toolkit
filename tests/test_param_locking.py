"""Test script demonstrating parameter locking functionality."""

import numpy as np
from pylog.analysis.regression import (
    LinearRegression,
    LogarithmicRegression,
    ExponentialRegression,
    PolynomialRegression,
    PowerRegression
)

# Generate some test data
np.random.seed(42)
x = np.linspace(1, 10, 50)
y_linear = 2.5 * x + 3.0 + np.random.normal(0, 1, len(x))

print("=" * 60)
print("LINEAR REGRESSION - Parameter Locking Demo")
print("=" * 60)

# Example 1: Lock slope
print("\n1. Lock slope to 2.0, fit intercept only:")
reg1 = LinearRegression(locked_params={'slope': 2.0})
reg1.fit(x, y_linear)
print(f"   {reg1.equation()}")
print(f"   R² = {reg1.r_squared:.4f}")

# Example 2: Lock intercept
print("\n2. Lock intercept to 0.0 (force through origin):")
reg2 = LinearRegression(locked_params={'intercept': 0.0})
reg2.fit(x, y_linear)
print(f"   {reg2.equation()}")
print(f"   R² = {reg2.r_squared:.4f}")

# Example 3: Lock both (fixed line)
print("\n3. Lock both slope and intercept:")
reg3 = LinearRegression(locked_params={'slope': 2.5, 'intercept': 3.0})
reg3.fit(x, y_linear)
print(f"   {reg3.equation()}")
print(f"   R² = {reg3.r_squared:.4f}")

# Example 4: Dynamic locking/unlocking
print("\n4. Dynamic locking/unlocking:")
reg4 = LinearRegression()
reg4.fit(x, y_linear)
print(f"   Normal fit: {reg4.equation()}")

reg4.lock_params(slope=2.0)
reg4.fit(x, y_linear)
print(f"   With slope locked: {reg4.equation()}")

reg4.unlock_params('slope')
reg4.lock_params(intercept=0.0)
reg4.fit(x, y_linear)
print(f"   With intercept locked: {reg4.equation()}")

reg4.unlock_params()  # Unlock all
reg4.fit(x, y_linear)
print(f"   Back to normal: {reg4.equation()}")

# Example 5: Logarithmic with locking
print("\n" + "=" * 60)
print("LOGARITHMIC REGRESSION - Parameter Locking Demo")
print("=" * 60)

y_log = 3.0 * np.log(x) + 5.0 + np.random.normal(0, 0.5, len(x))
reg_log = LogarithmicRegression(locked_params={'a': 3.0})
reg_log.fit(x, y_log)
print(f"\nLock 'a' to 3.0: {reg_log.equation()}")
print(f"R² = {reg_log.r_squared:.4f}")

# Example 6: Exponential with locking
print("\n" + "=" * 60)
print("EXPONENTIAL REGRESSION - Parameter Locking Demo")
print("=" * 60)

y_exp = 2.0 * np.exp(0.2 * x[:20]) + np.random.normal(0, 0.5, 20)
x_exp = x[:20]
reg_exp = ExponentialRegression(locked_params={'a': 2.0})
reg_exp.fit(x_exp, y_exp)
print(f"\nLock 'a' to 2.0: {reg_exp.equation()}")
print(f"R² = {reg_exp.r_squared:.4f}")

# Example 7: Polynomial with locking
print("\n" + "=" * 60)
print("POLYNOMIAL REGRESSION - Parameter Locking Demo")
print("=" * 60)

y_poly = 1.0 * x**2 - 2.0 * x + 3.0 + np.random.normal(0, 2, len(x))
reg_poly = PolynomialRegression(degree=2, locked_params={'c0': 1.0})  # Lock x² coefficient
reg_poly.fit(x, y_poly)
print(f"\nLock x² coefficient to 1.0: {reg_poly.equation()}")
print(f"R² = {reg_poly.r_squared:.4f}")

# Example 8: Power with locking
print("\n" + "=" * 60)
print("POWER REGRESSION - Parameter Locking Demo")
print("=" * 60)

y_power = 2.0 * x**2.0 * (1 + np.random.normal(0, 0.05, len(x)))  # Keep y positive
reg_power = PowerRegression(locked_params={'b': 2.0})  # Lock exponent
reg_power.fit(x, y_power)
print(f"\nLock exponent 'b' to 2.0: {reg_power.equation()}")
print(f"R² = {reg_power.r_squared:.4f}")

# Example 9: Query locked parameters
print("\n" + "=" * 60)
print("QUERYING LOCKED PARAMETERS")
print("=" * 60)

reg = LinearRegression(locked_params={'slope': 1.5, 'intercept': 2.0})
print(f"\nLocked parameters: {reg.get_locked_params()}")
print(f"Is 'slope' locked? {reg.is_param_locked('slope')}")
print(f"Is 'other' locked? {reg.is_param_locked('other')}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
