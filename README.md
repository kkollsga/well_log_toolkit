# logSuite

Petrophysical well-log analysis in Python. Load LAS files, set a project palette once, fit per-facies regressions, and export deliverable-ready SVGs — without writing matplotlib glue.

[![PyPI version](https://img.shields.io/pypi/v/logsuite.svg)](https://pypi.org/project/logsuite/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/kkollsga/logsuite/actions/workflows/build-and-publish.yml/badge.svg)](https://github.com/kkollsga/logsuite/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/logsuite/badge/?version=latest)](https://logsuite.readthedocs.io)

## Why logSuite

The library aims to stop you from writing the boilerplate that usually surrounds a reservoir-engineering notebook:

- **No per-well loops.** `manager.PHIE.filter("Facies").data(weighted=True)` returns one DataFrame across every well, with depth-interval weights for pooled percentiles.
- **No monkey-patching.** `xplot.add_regression_per("Facies", "exponential")` fits one line per facies; the line colours come from `manager.Facies.colors` automatically.
- **No matplotlib glue.** `xplot.add_table_panel(stats); xplot.save("deliverable.svg")` produces the SCAL/DG3 figure as one file.
- **No equation rewriting.** `equation_format="petrel"` renders fitted exponentials as `pow(10, c1*x + c0)` — paste straight into Petrel.

## Installation

```bash
pip install logsuite
```

Requires Python 3.10+, numpy, pandas, scipy, matplotlib.

## Quick start

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

Seven lines. One regression line per facies in the manager palette, legend equations in Petrel syntax, ready for Word.

## What's in the box

| Class | Role |
|-------|------|
| `WellDataManager` | Load and orchestrate wells; the data substrate every consumer reads from. |
| `ManagerView` | Read-only filtered subset of a manager (``manager.filter(wells=, where=)``). |
| `Property`, `Well` | Domain objects with depth-aligned arithmetic and chained filtering. |
| `Crossplot` | 2D scatter with `.add_regression(...)`, `.add_regression_per(...)`, `.add(...)`, `.add_table_panel(...)`. |
| `WellView`, `Template` | Single-well log display configured via a track template. |
| `RegressionFit`, `Artifact` | Renderable result objects added to consumers via `.add()`. Equations available in natural / log10 / Petrel form. |

## Common workflows

Each is a runnable script in [`story_tests/`](story_tests/) and a
how-to guide in the docs:

| Task | How-to | Script |
|------|--------|--------|
| One regression per facies / zone / well | [fit-regressions-per-group](https://logsuite.readthedocs.io/en/latest/how-to/fit-regressions-per-group.html) | `story_1_regression_subset.py`, `story_5_per_group_regressions.py` |
| Project palette flowing into every plot | [set-discrete-colors](https://logsuite.readthedocs.io/en/latest/how-to/set-discrete-colors.html) | `story_2_property_colors.py` |
| Pooled raw-data DataFrame across wells | [pool-data-across-wells](https://logsuite.readthedocs.io/en/latest/how-to/pool-data-across-wells.html) | `story_3_pooled_data.py`, `story_6_pooled_extraction.py` |
| Petrel-syntax regression equations | [export-petrel-equations](https://logsuite.readthedocs.io/en/latest/how-to/export-petrel-equations.html) | `story_4_petrel_equations.py` |
| Crossplot + stats table in one SVG | [build-scal-deliverable](https://logsuite.readthedocs.io/en/latest/how-to/build-scal-deliverable.html) | `story_7_scal_deliverable.py` |
| Single-well log display | [display-well-logs](https://logsuite.readthedocs.io/en/latest/how-to/display-well-logs.html) | `story_8_log_plot.py` |

## Documentation

[**logsuite.readthedocs.io**](https://logsuite.readthedocs.io)

- [Quick start](https://logsuite.readthedocs.io/en/latest/quickstart.html)
- [How-to guides](https://logsuite.readthedocs.io/en/latest/how-to/index.html) — task-focused recipes
- [User guide](https://logsuite.readthedocs.io/en/latest/user-guide/loading-data.html) — topical reference
- [API reference](https://logsuite.readthedocs.io/en/latest/api/index.html)

## Architecture

The library follows a strict layered-dependency rule documented in
[`ARCHITECTURE.md`](ARCHITECTURE.md). Lower layers do not know consumers
exist; data flows up — never down. The design exists to keep the API
small and predictable as features are added.

## License

MIT — see [LICENSE](LICENSE).
