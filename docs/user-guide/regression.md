# Regression

## Available Models

logsuite provides 6 regression models for crossplot analysis:

| Model | Equation | Use Case |
|-------|----------|----------|
| `LinearRegression` | y = ax + b | General linear trends |
| `PolynomialRegression` | y = a₀ + a₁x + ... + aₙxⁿ | Non-linear trends |
| `ExponentialRegression` | y = a·eᵇˣ | Permeability vs porosity |
| `LogarithmicRegression` | y = a·ln(x) + b | Diminishing returns |
| `PowerRegression` | y = a·xᵇ | Power law relationships |
| `PolynomialExponentialRegression` | y = exp(a₀ + a₁x + ... + aₙxⁿ) | Complex perm-poro |

## Basic Usage

```python
from logsuite import ExponentialRegression
import numpy as np

x = well.PHIE.values
y = well.PERM.values

# Fit regression
reg = ExponentialRegression()
reg.fit(x, y)

# Get equation and R²
print(reg.equation)
print(f"R² = {reg.r_squared:.4f}")

# Predict
y_pred = reg.predict(np.linspace(0.05, 0.35, 100))
```

## With Crossplots

```python
from logsuite import Crossplot, PolynomialRegression

xplot = Crossplot(x=well.PHIE, y=well.PERM)
xplot.add_regression(PolynomialRegression, degree=2)
xplot.plot()
```

## Polynomial Degree

For `PolynomialRegression` and `PolynomialExponentialRegression`:

```python
reg = PolynomialRegression(degree=3)
reg.fit(x, y)
```
