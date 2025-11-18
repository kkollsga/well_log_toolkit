# Well Log Toolkit

Fast, intuitive Python library for petrophysical well log analysis. Load LAS files, filter by zones, and compute depth-weighted statistics in just a few lines.

## Key Features

- **Lazy Loading** - Parse headers instantly, load data on demand
- **Numpy-Style Operations** - `well.HC_Volume = well.PHIE * (1 - well.SW)`
- **Hierarchical Filtering** - Chain filters: `well.PHIE.filter('Zone').filter('Facies').sums_avg()`
- **Depth-Weighted Statistics** - Proper averaging for irregular sampling
- **Multi-Well Management** - Broadcast operations: `manager.PHIE_percent = manager.PHIE * 100`
- **Project Persistence** - Save/load entire projects with metadata

## Installation

```bash
pip install well-log-toolkit
```

## Table of Contents

- [1-Minute Tutorial](#1-minute-tutorial) - Get started immediately
- [Quick Start](#quick-start) - Core workflow in 5 minutes
- [Core Concepts](#core-concepts) - Essential patterns
- [Full Documentation](#documentation) - Complete guides

---

## 1-Minute Tutorial

Load LAS files, filter by zones, and compute statistics:

```python
from well_log_toolkit import WellDataManager

# Load and analyze
manager = WellDataManager()
manager.load_las('well.las')

well = manager.well_12_3_4_A
stats = well.PHIE.filter('Zone').sums_avg()

print(stats['Top_Brent']['mean'])  # → 0.182 (depth-weighted)
```

**That's it!** Three lines to go from LAS file to zonal statistics.

> **New to this?** Continue to [Quick Start](#quick-start) for a complete 5-minute walkthrough.

---

## Quick Start

### 1. Load Data

```python
from well_log_toolkit import WellDataManager
import pandas as pd

# Load LAS files
manager = WellDataManager()
manager.load_las('well_A.las')
manager.load_las('well_B.las')

# Load formation tops from DataFrame
tops_df = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 B'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Brent'],
    'MD': [2850.0, 3100.0, 2900.0]
})

manager.load_tops(tops_df, well_col='Well', discrete_col='Surface', depth_col='MD')
```

### 2. Access Wells and Properties

```python
# Access well (special characters auto-sanitized)
well = manager.well_12_3_4_A

# Access properties directly
phie = well.PHIE
sw = well.SW

# List everything
print(well.properties)  # ['PHIE', 'SW', 'PERM', 'Zone', ...]
print(well.sources)     # ['Petrophysics', 'Imported_Tops']
```

### 3. Compute Statistics with Filtering

```python
# Single filter - group by Zone
stats = well.PHIE.filter('Zone').sums_avg()
# → {'Top_Brent': {'mean': 0.182, 'thickness': 250.0, ...}, 'Top_Statfjord': {...}}

# Chain filters - hierarchical grouping
stats = well.PHIE.filter('Zone').filter('Facies').sums_avg()
# → {'Top_Brent': {'Sandstone': {...}, 'Shale': {...}}, 'Top_Statfjord': {...}}
```

> **💡 Key:** Statistics are **depth-weighted** by default, accounting for irregular sampling. Add `arithmetic=True` to compare methods.

### 4. Create New Properties

```python
# Mathematical expressions (numpy-style)
well.HC_Volume = well.PHIE * (1 - well.SW)
well.PHIE_percent = well.PHIE * 100

# Comparison operations (auto-creates discrete flags)
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)

# Apply to all wells at once
manager.PHIE_percent = manager.PHIE * 100
# → Converts PHIE to percent in every well
```

> **⚠️ Important:** Operations require matching depth grids (like numpy). Use `.resample()` to align different grids.

### 5. Export Results

```python
# Export to DataFrame
df = well.data(include=['PHIE', 'SW', 'HC_Volume'])

# Export to LAS file
well.export_to_las('output.las')

# Save entire project
manager.save('my_project/')

# Load project later
manager = WellDataManager('my_project/')
```

**Done!** You've learned the core workflow in 5 minutes.

---

## Core Concepts

### Hierarchical Filtering

Filter properties by discrete logs (zones, facies, flags) to compute grouped statistics:

```python
# Add labels for readable output
ntg_flag = well.get_property('NTG_Flag')
ntg_flag.type = 'discrete'
ntg_flag.labels = {0: 'NonNet', 1: 'Net'}

# Chain filters for hierarchical grouping
stats = well.PHIE.filter('Zone').filter('NTG_Flag').sums_avg()
# {
#   'Top_Brent': {
#     'Net': {'mean': 0.21, 'thickness': 150.0, 'samples': 150},
#     'NonNet': {'mean': 0.08, 'thickness': 100.0, 'samples': 100}
#   },
#   'Top_Statfjord': {...}
# }
```

**Each statistics group includes:**
- `mean`, `sum`, `std_dev` - Depth-weighted by default
- `percentile` - p10, p50, p90 values
- `thickness` - Depth interval for this group
- `samples` - Number of valid measurements
- `range`, `depth_range` - Min/max values and depths

### Property Operations

Create computed properties using natural mathematical syntax:

```python
# Arithmetic (strict depth matching like numpy)
well.HC_Volume = well.PHIE * (1 - well.SW)
well.Porosity_Avg = (well.PHIE + well.PHIT) / 2

# Comparisons (auto-creates discrete properties)
well.High_Poro = well.PHIE > 0.15
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)

# Use computed properties in filtering
stats = well.PHIE.filter('Reservoir').sums_avg()
# → {'False': {...}, 'True': {...}}
```

> **💡 Pro Tip:** Computed properties are stored in a special `'computed'` source and can be exported to LAS files.

### Depth Alignment

Operations fail if depth grids don't match (prevents silent interpolation errors):

```python
# This fails if depths don't match
result = well.PHIE + well.CorePHIE  # DepthAlignmentError

# Explicit resampling required
core_resampled = well.CorePHIE.resample(well.PHIE)
result = well.PHIE + core_resampled  # ✓ Works
```

### Manager Broadcasting

Apply operations to all wells at once:

```python
# Broadcast to all wells with PHIE
manager.PHIE_percent = manager.PHIE * 100

# Broadcast complex operations
manager.HC_Volume = manager.PHIE * (1 - manager.SW)
# ✓ Created property 'HC_Volume' in 12 well(s)
# ⚠ Skipped 3 well(s) without property 'PHIE' or 'SW'
```

Wells without required properties are automatically skipped with a warning.

### Depth-Weighted Statistics

Standard arithmetic mean fails with irregular sampling:

```python
# Example: NTG flag
# Depths: 1500m, 1501m, 1505m
# Values:    0,    1,    0

# Arithmetic mean: (0+1+0)/3 = 0.33 ❌ (treats all samples equally)
# Weighted mean: accounts for 2.5m interval at depth 1501m = 0.50 ✓

# Compare both methods
stats = well.NTG.filter('Zone').sums_avg(arithmetic=True)
# Returns: {'mean': {'weighted': 0.50, 'arithmetic': 0.33}, ...}
```

> **✨ Key Insight:** Weighted statistics properly handle irregular sample spacing by accounting for depth intervals.

### Sampled Data (Core Plugs)

Core plug measurements are point samples requiring different treatment:

```python
# Load core data as sampled
manager.load_las('core_plugs.las', sampled=True)

# Or mark properties as sampled
well.CorePHIE.type = 'sampled'

# Sampled data uses arithmetic mean (each plug counts equally)
stats = well.CorePHIE.filter('Zone').sums_avg()
# → {'Top_Brent': {'mean': 0.205, 'samples': 12, 'calculation': 'arithmetic'}}
```

### Project Persistence

Save and restore entire projects:

```python
# Save project structure
manager.save('my_project/')
# Creates: my_project/well_12_3_4_A/Petrophysics.las, Imported_Tops.las, ...

# Load project (restores everything)
manager = WellDataManager('my_project/')

# All wells, properties, labels, and metadata are restored
```

---

## Documentation

### Quick References

Jump directly to specific topics:

- **[Managing Wells](#managing-wells)** - Add, remove, access wells
- **[Formation Tops](#formation-tops-setup)** - Load and configure formation tops
- **[Discrete Properties](#discrete-properties--labels)** - Set up labels for readable output
- **[Statistics Explained](#understanding-statistics-output)** - What each statistic means
- **[Export Options](#export-options)** - DataFrame and LAS export
- **[Managing Sources](#managing-sources)** - Organize and rename sources
- **[Adding Data](#adding-external-data)** - Import from DataFrames
- **[Property Printing](#property-printing)** - Inspect data visually
- **[Troubleshooting](#troubleshooting)** - Common issues and solutions

### API Reference

```python
# Main classes
from well_log_toolkit import WellDataManager, Well, Property, LasFile

# Statistics functions
from well_log_toolkit import compute_intervals, mean, sum, std, percentile

# Exceptions
from well_log_toolkit import (
    DepthAlignmentError,
    PropertyNotFoundError,
    PropertyTypeError
)
```

### Common Patterns

**Load and analyze quickly:**
```python
manager = WellDataManager()
manager.load_las('well.las')
stats = manager.well_12_3_4_A.PHIE.filter('Zone').sums_avg()
```

**Chain multiple filters:**
```python
stats = well.PHIE.filter('Zone').filter('Facies').filter('NTG_Flag').sums_avg()
```

**Create computed properties:**
```python
well.HC_Volume = well.PHIE * (1 - well.SW)
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)
```

**Broadcast across wells:**
```python
manager.PHIE_percent = manager.PHIE * 100
manager.Reservoir = (manager.PHIE > 0.15) & (manager.SW < 0.35)
```

**Save and restore projects:**
```python
manager.save('project/')
manager = WellDataManager('project/')
```

---

## Detailed Guide

### Managing Wells

```python
# List all wells
print(manager.wells)  # ['well_12_3_4_A', 'well_12_3_4_B']

# Access by sanitized name (attribute access)
well = manager.well_12_3_4_A

# Access by original name
well = manager.get_well('12/3-4 A')  # Works with original name
well = manager.get_well('12_3_4_A')  # Works with sanitized name
well = manager.get_well('well_12_3_4_A')  # Works with well_ prefix

# Add well manually
well = manager.add_well('12/3-4 C')

# Remove well
manager.remove_well('12_3_4_A')
```

### Formation Tops Setup

Formation tops require a specific DataFrame structure:

```python
import pandas as pd

# Required columns: Well, Surface (formation name), MD (depth)
tops_df = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 A'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Cook'],
    'MD': [2850.0, 3100.0, 3400.0]
})

manager.load_tops(
    tops_df,
    property_name='Zone',      # Name for discrete property (default: 'Well_Tops')
    source_name='Tops',        # Source name (default: 'Imported_Tops')
    well_col='Well',           # Column with well names
    discrete_col='Surface',    # Column with formation names
    depth_col='MD'             # Column with depths
)
```

**How tops work:**
- Each top marks the **start** of that formation
- Uses **forward-fill**: Top_Brent applies from 2850m down to 3100m
- At 3100m, Top_Statfjord takes over and applies down to 3400m
- Labels are automatically created: `{0: 'Top_Brent', 1: 'Top_Statfjord', 2: 'Top_Cook'}`

### Discrete Properties & Labels

Labels make discrete properties human-readable:

```python
# Get discrete property
ntg = well.get_property('NTG_Flag')

# Mark as discrete
ntg.type = 'discrete'

# Add labels (maps numeric values to names)
ntg.labels = {
    0: 'NonNet',
    1: 'Net'
}

# Now statistics use labels instead of numbers
stats = well.PHIE.filter('NTG_Flag').sums_avg()
# Returns: {'NonNet': {...}, 'Net': {...}}
# Instead of: {0: {...}, 1: {...}}
```

**When to use discrete type:**
- Zones/formations
- Facies classifications
- Flags (net/non-net, pay/non-pay)
- Rock types
- Any categorical data

### Understanding Statistics Output

Each statistics group contains:

```python
stats = well.PHIE.filter('Zone').sums_avg()

# Example output for one zone:
{
  'Top_Brent': {
    # Core statistics (depth-weighted by default)
    'mean': 0.182,              # Average porosity
    'sum': 45.5,                # Sum (useful for flags: sum of NTG = net thickness)
    'std_dev': 0.044,           # Standard deviation

    # Percentiles
    'percentile': {
      'p10': 0.09,              # 10th percentile (pessimistic)
      'p50': 0.18,              # Median
      'p90': 0.24               # 90th percentile (optimistic)
    },

    # Value range
    'range': {
      'min': 0.05,              # Minimum value
      'max': 0.28               # Maximum value
    },

    # Depth information
    'depth_range': {
      'min': 2850.0,            # Top of zone
      'max': 3100.0             # Base of zone
    },

    # Sample information
    'samples': 250,             # Number of non-NaN measurements
    'thickness': 250.0,         # Interval thickness (sum of depth intervals)
    'gross_thickness': 555.0,   # Total thickness across all zones
    'thickness_fraction': 0.45, # Fraction of total (thickness/gross_thickness)

    # Metadata
    'calculation': 'weighted'   # Method: 'weighted', 'arithmetic', or 'both'
  }
}
```

**Compare weighted vs arithmetic:**
```python
stats = well.PHIE.filter('Zone').sums_avg(arithmetic=True)

# Values become dicts with both methods:
{
  'Top_Brent': {
    'mean': {'weighted': 0.182, 'arithmetic': 0.179},
    'sum': {'weighted': 45.5, 'arithmetic': 44.8},
    # ... other fields also have both methods
    'calculation': 'both'
  }
}
```

### Export Options

**To DataFrame:**
```python
# All properties
df = well.data()

# Specific properties only
df = well.data(include=['PHIE', 'SW', 'PERM'])

# Exclude properties
df = well.data(exclude=['DEPT'])

# Auto-resample to common depth grid (when properties have different depths)
df = well.data(auto_resample=True)

# Use labels for discrete properties
df = well.data(discrete_labels=True)
# Zone column shows: 'Top_Brent', 'Top_Statfjord'
# Instead of: 0, 1
```

**To LAS file:**
```python
# Export all properties
well.export_to_las('output.las')

# Specific properties
well.export_to_las('output.las', include=['PHIE', 'SW'])

# Use original LAS as template (preserves header info)
well.export_to_las('output.las', use_template=True)

# Export each source separately
well.export_sources('output_folder/')
# Creates: Petrophysics.las, Imported_Tops.las, computed.las
```

### Managing Sources

Sources organize properties by origin (e.g., different LAS files, imported data):

```python
# List all sources
print(well.sources)  # ['Petrophysics', 'CoreData', 'Imported_Tops']

# Access properties through source
phie = well.Petrophysics.PHIE
core_phie = well.CoreData.CorePHIE

# List properties in a source
print(well.Petrophysics.properties)  # ['DEPT', 'PHIE', 'SW', 'PERM']

# Rename source
well.rename_source('CoreData', 'Core_Porosity')
print(well.sources)  # ['Petrophysics', 'Core_Porosity', 'Imported_Tops']

# Remove source (deletes all its properties)
well.remove_source('Core_Porosity')
print(well.sources)  # ['Petrophysics', 'Imported_Tops']
```

**Changes are saved to disk:**
```python
manager.save()  # Renamed files updated, removed files deleted
```

### Adding External Data

Load data from pandas DataFrames:

```python
import pandas as pd

# Create DataFrame with depth column
external_df = pd.DataFrame({
    'DEPT': [2800, 2801, 2802, 2803],
    'CorePHIE': [0.20, 0.22, 0.19, 0.21],
    'CorePERM': [150, 200, 120, 180]
})

# Add to well
well.add_dataframe(
    external_df,
    source_name='CoreData',              # Optional, defaults to 'external_df'
    unit_mappings={                      # Optional, specify units
        'CorePHIE': 'v/v',
        'CorePERM': 'mD'
    },
    type_mappings={                      # Optional, specify types
        'CorePHIE': 'continuous',
        'CorePERM': 'continuous'
    },
    label_mappings={}                    # Optional, for discrete properties
)

# Access new properties
print(well.CoreData.CorePHIE.values)
```

### Property Printing

Inspect properties directly:

```python
# Print property (numpy-style, auto-clips large arrays)
print(well.PHIE)
# [PHIE] (1001 samples)
# depth: [2800.00, 2801.00, 2802.00, ..., 3798.00, 3799.00, 3800.00]
# values (v/v): [0.180, 0.185, 0.192, ..., 0.215, 0.212, 0.210]

# Print filtered property (shows filter values)
filtered = well.PHIE.filter('Zone').filter('NTG_Flag')
print(filtered)
# [PHIE] (1001 samples)
# depth: [2800.00, 2801.00, ..., 3800.00]
# values (v/v): [0.180, 0.185, ..., 0.210]
#
# Filters:
#   Zone: [Top_Brent, Top_Brent, ..., Top_Statfjord]
#   NTG_Flag: [NonNet, Net, ..., Net]

# Print manager-level property (shows all wells)
print(manager.PHIE)
# [PHIE] across 3 well(s):
#
# Well: well_12_3_4_A
# [PHIE] (1001 samples)
# ...
#
# Well: well_12_3_4_B
# [PHIE] (856 samples)
# ...
```

### Troubleshooting

**DepthAlignmentError: Cannot combine properties with different depth grids**

```python
# Problem: Properties have different depths
result = well.PHIE + well.CorePHIE  # Error!

# Solution: Explicitly resample
core_resampled = well.CorePHIE.resample(well.PHIE)
result = well.PHIE + core_resampled  # Works!
```

**PropertyNotFoundError: Property not found**

```python
# Problem: Property doesn't exist or wrong name
phie = well.PHIE_TOTAL  # Error if property doesn't exist

# Solution: Check available properties
print(well.properties)  # List all properties
print(well.sources)     # List all sources

# Or check if property exists
try:
    phie = well.get_property('PHIE_TOTAL')
except PropertyNotFoundError:
    print("Property not found, using default")
    phie = well.PHIE
```

**PropertyTypeError: Property must be discrete type**

```python
# Problem: Trying to filter by non-discrete property
stats = well.PHIE.filter('PERM').sums_avg()  # Error!

# Solution: Mark property as discrete
perm = well.get_property('PERM')
perm.type = 'discrete'
perm.labels = {0: 'Low', 1: 'Medium', 2: 'High'}
stats = well.PHIE.filter('PERM').sums_avg()  # Works!
```

**Missing statistics for some zones**

```python
# Problem: No valid data in some zones
stats = well.PHIE.filter('Zone').sums_avg()
# Some zones might be missing if all PHIE values are NaN in that zone

# Solution: Check raw data
print(well.PHIE.values)  # Look for NaN values
print(well.Zone.values)  # Check zone distribution

# Or filter to remove NaN
valid_mask = ~np.isnan(well.PHIE.values)
valid_depths = well.PHIE.depth[valid_mask]
```

**Computed properties not showing up**

```python
# After creating computed properties
well.HC_Volume = well.PHIE * (1 - well.SW)

# Check they exist
print(well.sources)  # Should include 'computed'
print(well.computed.properties)  # List computed properties

# Access directly
hc = well.HC_Volume  # Works

# Or through source
hc = well.computed.HC_Volume  # Also works
```

---

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## Performance

All operations use **vectorized numpy** for maximum speed:
- 100M+ samples/second throughput
- Typical well logs (1k-10k samples) process in < 1ms
- Bottleneck is I/O, not computation (optimized with lazy loading)

See [test_performance.py](test_performance.py) for benchmarks.

## Contributing

Contributions welcome! Please submit a Pull Request.

## License

MIT License
