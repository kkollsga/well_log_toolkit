# Loading Data

## LAS Files

The primary way to load well log data is from LAS files:

```python
from logsuite import WellDataManager

manager = WellDataManager()
manager.load_las("path/to/well.las")
```

### Lazy Loading

`LasFile` uses lazy loading — headers are parsed on initialization, but data
is only loaded when you call `.data()`:

```python
from logsuite import LasFile

las = LasFile("well.las")
print(las.well_name)       # Immediate — from headers
print(las.curves.keys())   # Immediate — from headers
df = las.data()            # Data loaded here
```

### Curve Metadata

Before loading data, you can inspect and modify curve metadata:

```python
# Check available curves
for name, meta in las.curves.items():
    print(f"{name}: unit={meta['unit']}, desc={meta['description']}")

# Set aliases and type before loading
las.update_curve('PHIE_2025', alias='PHIE', type='continuous')
las.update_curve('Zone_Log', type='discrete')
```

### Supported Versions

- **LAS 2.0**: Full support (space-delimited data)
- **LAS 3.0**: Basic support (tab-delimited data, single data section)

## DataFrames

Load data directly from pandas DataFrames via the manager:

```python
import pandas as pd

df = pd.DataFrame({
    'DEPT': [2800, 2801, 2802],
    'PHIE': [0.20, 0.22, 0.18],
    'SW': [0.45, 0.40, 0.50],
})

manager.load_properties(
    df,
    well_col=None,
    well_name="Test Well",
    source_name="petrophysics",
)
```

### With Metadata

```python
manager.load_properties(
    df,
    well_col=None,
    well_name="Test Well",
    source_name="petrophysics",
    unit_mappings={'DEPT': 'm', 'PHIE': 'v/v'},
    type_mappings={'Zone': 'discrete'},
    label_mappings={'Zone': {0: 'NonRes', 1: 'Reservoir'}},
)
```

Set ``logsuite.set_quiet(True)`` once at startup to silence the ``✓ Loaded …`` confirmation messages in scripted use.

## Multiple Sources

Wells can have multiple data sources (LAS files or DataFrames):

```python
manager.load_las("core_data.las")
manager.load_las("log_data.las")

# Properties from all sources are merged
well = manager.well_Test
print(well.properties)  # Shows properties from both files
```
