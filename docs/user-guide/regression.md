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
from logsuite import Crossplot

xplot = Crossplot(manager, x="PHIE", y="PERM")
xplot.add_regression("polynomial_2")
xplot.show()
```

### Regression on a subset

`add_regression` accepts a `where=` argument — a dict of public column
names to allowed values, or a callable returning a boolean mask. Subsets
smaller than `min_samples` (default 5) are skipped with a warning:

```python
xplot.add_regression("exponential", name="Sand 2",
                     where={"Facies": ["Reservoir"]})
```

For "one regression per category" use the convenience method:

```python
xplot.add_regression_per("Facies", "exponential")
```

See the [per-group regression how-to](../how-to/fit-regressions-per-group.md).

## Polynomial Degree

For `PolynomialRegression` and `PolynomialExponentialRegression`:

```python
reg = PolynomialRegression(degree=3)
reg.fit(x, y)
```

## Petrel calculator syntax

Each regression artifact exposes its equation in three forms — natural,
log10, and Petrel:

```python
fit = manager.properties(["PHIE", "PERM"]).fit(
    ExponentialRegression(), name="all wells", equation_format="petrel"
)
fit.equation(format="petrel")    # 'pow(10, 19.628*x - 3.676)' — paste into Petrel
```

See the [Petrel-export how-to](../how-to/export-petrel-equations.md).
