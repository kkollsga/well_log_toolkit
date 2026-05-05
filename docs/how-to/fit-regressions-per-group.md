# Fit a regression per group on one crossplot

For DG3 poroperm transforms, you typically want one regression line per
facies (or per zone, or per well) on a single crossplot — each line in
its own colour, with its R² and equation in the legend, all in one
figure. logSuite does this in two lines.

## The minimal pattern

```python
from logsuite import Crossplot

xplot = Crossplot(manager, x="PHIE", y="PERM", color="Facies",
                  equation_format="petrel", decimals=3)
xplot.add_regression_per("Facies", "exponential", legend_loc="upper left")
xplot.save("poroperm.png")
```

Three concrete things this gets right without any extra work:

* **Line colours match the manager palette** — set
  ``manager.Facies.colors = {0: "#999999", 1: "#3b82f6", 2: "#10b981"}``
  once and every regression line picks up the matching colour.
* **Legend equations match what Petrel expects** — because
  ``equation_format="petrel"`` was set at the constructor, every fit's
  legend shows ``pow(10, c1*x + c0)``.
* **No monkey-patching of ``xplot._data``** — ``add_regression_per``
  enumerates the unique facies values internally and calls
  ``add_regression(where={"Facies": [value]})`` for each.

## When you need an arbitrary subset

For a custom mask (e.g. the high-PHIE half of one facies), use
``add_regression(where=callable)`` and get the internal column names
from ``column_for``:

```python
facies_col = xplot.column_for("Facies")
phi_col = xplot.column_for("PHIE")

xplot.add_regression(
    "exponential",
    name="Clean (PHIE>0.20)",
    where=lambda df: (df[facies_col] == 2.0) & (df[phi_col] > 0.20),
    line_color="#fbbf24",
    line_style="--",
)
```

Subsets smaller than ``min_samples`` (default 5) emit a warning and are
skipped — no exception.

## Verifying

Two runnable examples in ``story_tests/``:

* ``story_1_regression_subset.py`` — per-facies fits with palette colours
  and Petrel-form equations.
* ``story_5_per_group_regressions.py`` — same plus an arbitrary callable
  subset.
