# Set a discrete-property palette once and have it flow everywhere

You have a facies log and a project palette. You want every visualisation
in logSuite to use those colours — log tracks, crossplot scatter,
crossplot regression lines, legends — without redefining the palette
per chart.

## The minimal pattern

```python
manager.Facies.colors = {
    0: "#999999",  # Tight
    1: "#3b82f6",  # Medium
    2: "#10b981",  # Clean
}
```

That's it. The setter broadcasts to every well's ``Facies`` property.
After that:

* ``Crossplot(manager, color="Facies")`` colours scatter points and the
  legend from your palette.
* ``WellView(well, template=...)`` colours discrete-fill tracks the same
  way.
* ``add_regression(where={"Facies": ["Clean"]})`` and
  ``add_regression_per("Facies", ...)`` use the matching palette colour
  for the regression line automatically.

## Set labels too

```python
manager.Facies.labels = {0: "Tight", 1: "Medium", 2: "Clean"}
```

After this, ``where={"Facies": ["Clean"]}`` works in plain English — the
library translates the label to the integer code internally. ``.data()``
output also returns the label strings.

## Quiet mode

The setter prints ``✓ Set colors for property 'Facies' in 3 well(s)`` by
default. Silence it for scripted use:

```python
import logsuite
logsuite.set_quiet(True)
```

## Verifying

``story_tests/story_2_property_colors.py`` constructs a Crossplot after
setting the palette and asserts that every legend patch matches a
user-supplied colour.
