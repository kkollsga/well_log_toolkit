# logSuite

Fast, Pythonic petrophysical well log analysis with depth-weighted statistics,
hierarchical filtering, and template-driven visualization.

## Key Features

- **LAS 2.0/3.0 file loading** with lazy data loading and automatic metadata parsing
- **Depth-weighted statistics** using midpoint interval method
- **Chained filtering** on discrete properties (zones, facies, net/gross)
- **Mathematical operations** between properties with automatic depth alignment
- **Multi-well analysis** via `WellDataManager` broadcasting
- **Template-driven visualization** for well logs and crossplots
- **6 regression models** for crossplot analysis

## Quick Start

```python
from logsuite import WellDataManager

manager = WellDataManager()
manager.load_las("well_a.las").load_las("well_b.las")

well = manager.well_12_3_2_B
stats = well.PHIE.filter('Zone').sums_avg()
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/loading-data
user-guide/wells-properties
user-guide/statistics
user-guide/visualization
user-guide/regression
user-guide/multi-well
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
api/core
api/io
api/statistics
api/regression
api/visualization
api/manager
api/exceptions
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
```
