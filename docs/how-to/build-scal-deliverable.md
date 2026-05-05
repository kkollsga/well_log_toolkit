# Build a SCAL/DG3 deliverable: crossplot + stats table in one SVG

The standard deliverable for static modelling is a poroperm crossplot
with regression lines and a summary statistics table, exported as one
SVG that pastes into PowerPoint or Word. logSuite builds this in three
calls; no matplotlib glue.

## The minimal pattern

```python
from logsuite import Crossplot

# Per-facies summary table from the same manager + filters used by the
# crossplot — guarantees the numbers match what you're plotting.
stats = manager.PHIE.filter("Facies").stats(
    return_df=True, flat_columns=True,
    methods=["mean", "percentile_10", "percentile_50", "percentile_90"],
)

xplot = Crossplot(manager, x="PHIE", y="PERM", color="Facies", y_log=True,
                  equation_format="petrel", decimals=3,
                  title="Synthetic poroperm — DG3 deliverable shape")

xplot.add_regression_per("Facies", "exponential", legend_loc="upper left")

xplot.add_table_panel(stats, position="bottom",
                      title="Per-facies PHIE summary",
                      formatters={"mean": ".4f", "p10": ".4f",
                                  "p50": ".4f", "p90": ".4f"})

xplot.save("poroperm_deliverable.svg")
```

Output: a single SVG containing the scatter (with three Petrel-form
regression lines, palette colours from
``manager.Facies.colors``, legend in the upper-left) and the stats
table beneath, with formatted numbers and ``N/A`` for any NaNs.

## Layout knobs

* ``position="right"`` puts the table to the right of the chart.
* ``table_fraction=0.30`` (default) sets how much of the figure
  dimension the panel takes; the figure grows along that axis so the
  scatter is not squished.
* ``formatters={"col": ".4f"}`` accepts Python format specs or callables.

## MultiIndex tables

If you build a hierarchical summary (e.g. ``Zone × Facies``):

```python
stats = manager.PHIE.filter("Zone").filter("Facies").stats(
    return_df=True, flat_columns=True,
)
```

``add_table_panel`` flattens MultiIndex column labels with ``" | "`` and
visually merges repeated outer-level row labels (the outer cell is
blanked when the same as the previous row).

## Verifying

``story_tests/story_7_scal_deliverable.py`` produces
``output_story_7.svg`` end-to-end.
