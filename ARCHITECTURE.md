# logSuite Architecture

The durable architectural reference for logSuite. Defines the layered dependency rule that governs every module, and the structure that follows from it. Pull requests that violate the constitution should be rejected.

## Constitution

**Article I — Layered dependency.** Every module belongs to exactly one layer. Layers are linearly ordered. A module imports only from layers strictly below its own. Never sideways, never up.

**Article II — Layers, lowest to highest.**

| # | Layer | Purpose |
|---|---|---|
| 1 | Foundation (`utils.py`, `exceptions.py`) | Pure helpers; no internal imports |
| 2 | `io/` | LAS file readers and writers |
| 3 | `core/` | `Well`, `Property`, `PropertyOperationsMixin`, `Artifact` protocol |
| 4 | `analysis/` | Statistics, regression models, `sums_avg`, concrete artifact classes |
| 5 | `manager/` | `WellDataManager`, `ManagerView`, property proxies |
| 6 | `visualization/` | `WellView`, `Crossplot`, `Template`, `Composite`, `Table`, `style` |

**Article III — No upward calls.** A lower layer never references a higher one. Convenience methods that bridge upward (e.g., `Well.Crossplot()`, `WellDataManager.Crossplot()`) are violations.

**Article IV — One entry point per concept.** Each consumer has exactly one constructor (`Crossplot(view, ...)`, `WellView(well_or_view, ...)`). No alternative entry points on lower layers.

**Article V — Artifacts compose into consumers via `.add()`.** Consumers (Crossplot, Table, Composite, WellView) do not grow methods per artifact type. Artifacts (RegressionFit, Trend, ...) implement `_render_in_<consumer>(...)` for the consumers they support and raise `TypeError` for those they do not.

**Article VI — Public surface is minimal.** Each layer's `__init__.py` re-exports only what users need. Internals stay internal.

**Article VII — Compartmentalization.** Code is organized into focused modules and classes. One module, one coherent topic; one class, one well-defined responsibility. God files and god classes are violations.

Soft refactor triggers (warning signs, not hard rules):

- A module beyond ~600 lines.
- A class beyond ~300 lines.
- A method beyond ~50 lines.
- A class with sections that read like "## Filtering / ## Statistics / ## Export" — that is multiple responsibilities asking to become multiple classes.
- Multiple classes in one file that do not reference each other — they want to be separate modules.

When a trigger fires, prefer splitting along **responsibility seams**: the regression base class lives in `base.py`; each concrete regression in its own file; an artifact wrapper in `fit.py`. Files stay readable; tests stay focused; the dependency arrow remains scannable.

This article is forward-looking — several existing files are over the soft limits (`property.py`, `well.py`, `wellview.py`, `crossplot.py`, `proxy.py`). Splitting them is M3+ work. For new code, the limits apply now.

## Target structure

```
logsuite/
├── __init__.py
├── _version.py
├── exceptions.py
├── utils.py
├── py.typed
│
├── io/                       # Layer 2
│   ├── __init__.py
│   └── las_file.py
│
├── core/                     # Layer 3
│   ├── __init__.py
│   ├── well.py               # Well — no .Crossplot/.WellView methods
│   ├── property.py
│   ├── operations.py
│   └── artifact.py           # Artifact protocol (matplotlib via TYPE_CHECKING only)
│
├── analysis/                 # Layer 4
│   ├── __init__.py
│   ├── statistics.py
│   ├── sums_avg.py
│   └── regression/
│       ├── __init__.py
│       ├── base.py
│       ├── linear.py
│       ├── exponential.py
│       ├── logarithmic.py
│       ├── polynomial.py
│       ├── polynomial_exponential.py
│       ├── power.py
│       └── fit.py            # RegressionFit artifact
│
├── manager/                  # Layer 5
│   ├── __init__.py
│   ├── data_manager.py       # WellDataManager — no .Crossplot method
│   ├── view.py               # ManagerView (filtered manager)
│   └── proxy.py              # proxies + .data() + .fit() + .stats(return_df=)
│
└── visualization/            # Layer 6
    ├── __init__.py
    ├── style.py              # single color/style resolver, reads Property.colors
    ├── template.py
    ├── wellview.py
    ├── crossplot.py          # has .add(artifact); no regression* kwargs
    ├── composite.py
    └── table.py
```

## Adding new things

- **A new visualization?** Add a class to `visualization/`. Its constructor accepts a `ManagerView` (or `Well` for single-well views). It does NOT add a method to any lower layer.
- **A new artifact type?** Add a class where its compute logic lives (regression fits under `analysis/regression/`, trend smoothers under `analysis/`, depth-based markers under `core/`). Implement `_render_in_<consumer>(...)` for each consumer that should accept it.
- **A new statistic?** Add a function to `analysis/statistics.py` and surface it via `Property.stats(method=...)`. Never reach up into manager or visualization.
- **A new I/O format?** Add a module to `io/`. Surface it through a method on `Well` or `WellDataManager` that loads from the new format.
- **A new convenience accessor on Well or Manager?** Stop. If it constructs anything from a higher layer, it violates Article III. The user calls the higher-layer constructor directly.
- **A new file is bigger than ~600 lines on first commit?** Stop. Find the responsibility seams and split before merging. Bigger files only get bigger.

## Verifying compliance

A reviewer should be able to scan a module's imports and confirm they all reference lower layers.

- Any `from logsuite.visualization import ...` inside `core/`, `analysis/`, or `manager/` is a violation.
- Any `from logsuite.manager import ...` inside `core/`, `analysis/`, or `io/` is a violation.
- Any method on `Well` or `WellDataManager` that constructs a visualization consumer is a violation.
- Any consumer that takes `wells: list[Well]` (rather than a Manager/view) is a violation of Article IV.
