# Visualization

## Template

A `Template` defines the track layout for well log plots:

```python
from logsuite import Template

template = Template()

# Add tracks with logs
template.add_track(track_type="depth", width=0.5)
template.add_track(
    track_type="continuous",
    logs=[{"name": "GR", "color": "green", "x_range": [0, 150]}],
    title="GR",
    width=2,
)
template.add_track(
    track_type="continuous",
    logs=[{"name": "PHIE", "color": "blue", "x_range": [0, 0.4]}],
    title="Porosity",
    width=2,
)
template.add_track(
    track_type="continuous",
    logs=[{"name": "SW", "color": "red", "x_range": [0, 1]}],
    title="Saturation",
    width=2,
)
```

## WellView

`WellView` renders a well log display using a template:

```python
from logsuite import WellView

view = WellView(well, template=template)
view.show()

# With depth range
view = WellView(well, template=template, depth_range=(2800, 3200))
view.show()
```

## Crossplot

Crossplot accepts a `WellDataManager`, a `ManagerView`, a single `Well`, or
a list of wells — pass any of them as the first argument:

```python
from logsuite import Crossplot

xplot = Crossplot(
    manager,                # or manager.filter(wells=[...]) for a subset
    x="PHIE",
    y="PERM",
    color="Facies",
)
xplot.show()
```

### Per-group regression in one call

```python
xplot = Crossplot(manager, x="PHIE", y="PERM", color="Facies",
                  equation_format="petrel", decimals=3)
xplot.add_regression_per("Facies", "exponential", legend_loc="upper left")
xplot.show()
```

Line colours come from `manager.Facies.colors` automatically; the legend
shows Petrel-syntax equations because of the constructor-level
`equation_format`. See the
[per-group regression how-to](../how-to/fit-regressions-per-group.md)
for the full pattern.

### Adding a stats table panel

```python
stats = manager.PHIE.filter("Facies").stats(return_df=True, flat_columns=True)

xplot.add_table_panel(stats, position="bottom",
                      title="Per-facies summary",
                      formatters={"mean": ".4f"})
xplot.save("deliverable.svg")     # crossplot + table in one file
```
