# Visualization

## Template

A `Template` defines the track layout for well log plots:

```python
from logsuite import Template

template = Template()

# Add tracks
template.add_track("Depth", width=0.5)
template.add_track("GR", width=2)
template.add_track("Porosity", width=2)
template.add_track("Saturation", width=2)

# Add curves to tracks
template.add_curve("GR", "GR", color="green", scale=(0, 150))
template.add_curve("Porosity", "PHIE", color="blue", scale=(0, 0.4))
template.add_curve("Saturation", "SW", color="red", scale=(0, 1))
```

## WellView

`WellView` renders a well log display using a template:

```python
from logsuite import WellView

view = WellView(well, template)
view.plot()
view.plot(depth_range=(2800, 3200))
```

## Crossplot

Create crossplots with optional regression:

```python
from logsuite import Crossplot

xplot = Crossplot(
    x=well.PHIE,
    y=well.PERM,
    color_by=well.Zone,
)
xplot.plot()
```

### With Regression

```python
from logsuite import Crossplot, ExponentialRegression

xplot = Crossplot(x=well.PHIE, y=well.PERM)
xplot.add_regression(ExponentialRegression)
xplot.plot()
```
