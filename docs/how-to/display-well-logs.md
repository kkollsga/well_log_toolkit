# Display a single-well log plot

Standard well-log display: depth track, GR, density-neutron overlay,
porosity with reservoir fill, facies colour fill — driven by a Template
so you build the layout once and apply it to any well.

## The minimal pattern

```python
from logsuite import Template, WellView

template = Template("standard")
template.add_track(track_type="depth", width=0.4)
template.add_track(track_type="continuous", title="GR",
                   logs=[{"name": "GR", "x_range": [0, 150], "color": "darkgreen"}])
template.add_track(
    track_type="continuous", title="RHOB / NPHI",
    logs=[
        {"name": "RHOB", "x_range": [1.95, 2.70], "color": "red"},
        {"name": "NPHI", "x_range": [0.45, -0.05], "color": "blue"},  # reversed
    ],
    width=1.2,
)
template.add_track(
    track_type="continuous", title="PHIE",
    logs=[{"name": "PHIE", "x_range": [0.0, 0.35], "color": "navy"}],
    fill={"left": {"curve": "PHIE"}, "right": {"value": 0},
          "color": "lightblue", "alpha": 0.5},
)
template.add_track(track_type="discrete", title="Facies",
                   logs=[{"name": "Facies"}], width=0.5)

view = WellView(manager["Well_A"], depth_range=(2510, 2545),
                template=template, figsize=(10, 8))
view.save("well_a_log.png")
```

## Discrete-fill colours

The Facies track uses ``manager.Facies.colors`` automatically — set the
palette once on the manager, every WellView (and Crossplot) reads from
it. See the
[discrete-colour how-to](set-discrete-colors.md).

## Common track recipes

* **Depth-coloured fill** — pass a colormap name in ``fill``:
  ``fill={"left": "track_edge", "right": "GR", "colormap": "viridis"}``
  fills horizontal bands by GR value.
* **Multiple curves with crossover** — set the second curve's
  ``x_range`` reversed (``[max, min]``) so the standard density-neutron
  signature appears in cleaner zones.
* **Formation tops** — pass ``tops={"name": "Tops_Property", ...}`` on
  any track to draw horizontal markers across it.

## Tops-driven depth range

If you've loaded formation tops, let WellView calculate the depth range
from them:

```python
view = WellView(well, tops=["Top_Brent", "Top_Statfjord"], template=template)
```

## Verifying

``story_tests/story_8_log_plot.py`` produces a five-track display end
to end.
