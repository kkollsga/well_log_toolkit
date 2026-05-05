# Export regression equations in Petrel calculator syntax

Petrel calculator wants ``pow(10, c1*x + c0)``. Your fit is
``y = a * exp(b * x)``. Same line, different conventions, and the
conversion (``c0 = log10(a)``, ``c1 = b / ln(10)``) is annoying to do
by hand for every fit. logSuite renders the equation in any of three
forms on demand.

## Standalone fit

```python
from logsuite import ExponentialRegression

fit = manager.properties(["PHIE", "PERM"]).fit(
    ExponentialRegression(),
    name="Clean — pooled",
    equation_format="petrel",
    decimals=4,
)

print(fit.equation(format="natural"))   # y = 0.0002*e^(45.1969x)
print(fit.equation(format="log10"))     # y = 10^(19.6288x - 3.6758)
print(fit.equation(format="petrel"))    # pow(10, 19.6288*x - 3.6758)
print(fit.label())                      # legend label
```

## On a Crossplot

Set ``equation_format="petrel"`` at the constructor and every
``add_regression`` call's legend uses Petrel form:

```python
xplot = Crossplot(manager, x="PHIE", y="PERM", color="Facies",
                  equation_format="petrel", decimals=3)
xplot.add_regression_per("Facies", "exponential")
```

The legend now shows ``Clean (pow(10, 8.875*x - 1.356))`` — copy-paste
ready. Per-call ``equation_format=`` and ``decimals=`` on
``add_regression`` override the constructor default.

## What "petrel" form looks like for non-exponential fits

Petrel form is only meaningful for ``ExponentialRegression``. For
``Linear``/``Logarithmic``/``Polynomial``/``Power``, ``format="petrel"``
falls back to the natural equation — the result is the same string
you would get with ``format="natural"``.

## Verifying

``story_tests/story_4_petrel_equations.py`` builds a single fit and
prints the same equation in all three forms with two different decimal
counts.
