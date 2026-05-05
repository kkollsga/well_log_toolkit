# Quick start

A working DG3 deliverable in 30 seconds and seven lines.

```python
from logsuite import Crossplot, WellDataManager, set_quiet

set_quiet(True)                                    # silence broadcast prints
manager = WellDataManager()
manager.load_las("12_3-2_B.las").load_las("12_3-2_C.las")
manager.Facies.colors = {0: "#999999", 1: "#3b82f6", 2: "#10b981"}

xplot = Crossplot(manager, x="PHIE", y="PERM", color="Facies", y_log=True,
                  equation_format="petrel", decimals=3)
xplot.add_regression_per("Facies", "exponential", legend_loc="upper left")
xplot.save("poroperm.svg")
```

What this gives you:

* a poroperm crossplot of all wells,
* one regression line per facies in the manager palette,
* legend equations in Petrel calculator syntax (``pow(10, c1*x + c0)``),
* zero monkey-patching of internals.

## The three abstractions

| Layer | Class | What it owns |
|-------|-------|--------------|
| Data | ``WellDataManager``, ``ManagerView`` | Wells, properties, filtering, broadcasting |
| Result | ``RegressionFit`` (and other ``Artifact`` subclasses) | Fitted state + equation + render methods |
| Display | ``Crossplot``, ``WellView``, ``Template`` | Reads from a Manager-substrate, accepts artifacts via ``.add()`` |

The architecture is layered. Data layers know nothing about display.
Display consumers read from the manager substrate or from a filtered
view; they don't construct themselves from below.

## Common patterns

* **One filter per group**: ``manager.PHIE.filter("Zone").filter("Facies").stats()``
* **Pooled raw data**: ``manager.PHIE.filter("Facies").data(weighted=True)``
* **Subset view**: ``manager.filter(wells=["A", "B"], where={"Facies": "Clean"})``
* **Per-category regression**: ``xplot.add_regression_per("Facies", "exponential")``
* **Combined deliverable**: ``xplot.add_table_panel(stats); xplot.save(...)``
* **Single-well log**: ``WellView(manager["Well_A"], template=template).save(...)``

Each has a dedicated [how-to guide](how-to/index.md).

## Where next

* [How-to guides](how-to/index.md) — task-focused recipes (regression per group, Petrel equations, deliverable composition, log plots).
* [User guide](user-guide/loading-data.md) — topical reference (loading data, statistics, visualisation, regression).
* [API reference](api/index.md) — every public class and method.
