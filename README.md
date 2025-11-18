# Well Log Toolkit

A Python library for petrophysical well log analysis with lazy loading, multi-well management, and hierarchical filtering. Designed for efficient processing of LAS files with automatic statistics computation by zone and facies.

## Key Features

- **Lazy Loading** - Headers parsed immediately, data loaded only when accessed
- **Multi-Well Management** - Intuitive attribute-based access to wells and properties
- **Property Computation** - Create new properties using natural mathematical expressions with automatic depth alignment
- **Hierarchical Filtering** - Chain discrete filters and compute grouped statistics
- **Depth-Weighted Statistics** - Proper averaging that accounts for depth intervals
- **Accurate Zone Boundaries** - Synthetic samples inserted at zone tops for precise interval partitioning
- **Source Tracking** - Know which LAS file each property came from
- **Project Persistence** - Save and load entire analysis projects
- **Metadata Preservation** - Round-trip LAS files without data loss

## Installation

```bash
pip install well-log-toolkit
```

## Quick Start

Load LAS files, add formation tops, and compute statistics in just a few lines:

```python
from well_log_toolkit import WellDataManager
import pandas as pd

# Initialize manager and load LAS files
manager = WellDataManager()
manager.load_las('path/to/12_3-4_A.las')
manager.load_las('path/to/12_3-4_B.las')

# Load formation tops from DataFrame
tops_df = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 B', '12/3-4 B'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Brent', 'Top_Statfjord'],
    'MD': [2850.0, 3100.0, 2900.0, 3150.0]
})

manager.load_tops(
    tops_df,
    property_name='Zone',         # Custom name for the discrete property (default: 'Well_Tops')
    source_name='Formation_Tops', # Custom name for the source (default: 'Imported_Tops')
    well_col='Well',
    discrete_col='Surface',
    depth_col='MD'
)

# Access well and compute hierarchical statistics
well = manager.well_12_3_4_A
stats = well.PHIE.filter('Zone').sums_avg()

print(stats)
# {
#   'Top_Brent': {
#     'mean': 0.182,               # Depth-weighted average (default)
#     'sum': 45.5,                 # Useful for net thickness calculations
#     'std_dev': 0.044,            # Depth-weighted standard deviation
#     'percentile': {
#       'p10': 0.09,               # 10th percentile
#       'p50': 0.18,               # Median (50th percentile)
#       'p90': 0.24                # 90th percentile
#     },
#     'range': {
#       'min': 0.05,               # Minimum value
#       'max': 0.28                # Maximum value
#     },
#     'depth_range': {
#       'min': 2850.0,             # Minimum depth in zone
#       'max': 3100.0              # Maximum depth in zone
#     },
#     'samples': 250,              # Number of valid samples
#     'thickness': 250.0,          # Interval thickness for this zone (m)
#     'gross_thickness': 555.0,    # Total gross thickness (m)
#     'thickness_fraction': 0.45,  # Fraction of gross thickness
#     'calculation': 'weighted'    # Method used
#   },
#   'Top_Statfjord': {...}
# }

# Include both weighted and arithmetic statistics
stats = well.PHIE.filter('Zone').sums_avg(arithmetic=True)
# 'mean': {'weighted': 0.182, 'arithmetic': 0.179}
# 'calculation': 'both'

# Create new properties with mathematical expressions
well.HC_Volume = well.PHIE * (1 - well.SW)  # Hydrocarbon pore volume
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)  # Reservoir flag

# Apply operations across all wells
manager.PHIE_percent = manager.PHIE * 100  # Converts PHIE to percent in all wells
```

---

## Step-by-Step Guide

### 1. Loading LAS Files

The `WellDataManager` is your entry point for all operations. It automatically creates wells based on the WELL field in LAS files.

```python
from well_log_toolkit import WellDataManager

# Create manager
manager = WellDataManager()

# Load single file
manager.load_las('path/to/well.las')

# Load multiple files (method chaining)
manager.load_las('file1.las').load_las('file2.las').load_las('file3.las')

# Load list of files
manager.load_las(['file1.las', 'file2.las', 'file3.las'])

# Load all LAS files from a directory
from pathlib import Path
las_files = list(Path('data/').glob('*.las'))
manager.load_las(las_files)
```

Each LAS file becomes a **source** within its well. Wells are automatically created or matched based on the WELL name in the LAS header.

### 2. Managing Wells

Wells are accessible as attributes on the manager. Special characters in well names are automatically sanitized:

```python
# Well names are sanitized for attribute access
# "12/3-4 A" becomes "well_12_3_4_A"
well = manager.well_12_3_4_A

# List all wells
print(manager.wells)  # ['12_3_4_A', '12_3_4_B']

# Get well by original name
well = manager.get_well('12/3-4 A')

# Remove a well
manager.remove_well('12_3_4_A')

# Add a well manually
from well_log_toolkit import Well
new_well = Well(name='12/3-4 C')
manager.add_well(new_well)
```

### 3. Accessing Properties

Properties are accessible as attributes on wells. Each property knows its source:

```python
well = manager.well_12_3_4_A

# Direct property access
phie = well.PHIE
sw = well.SW
perm = well.PERM

# Access through source
phie_from_source = well.Petrophysics.PHIE

# List all properties
print(well.properties)  # ['DEPT', 'PHIE', 'SW', 'PERM', 'Zone', ...]

# List all sources
print(well.sources)  # ['Petrophysics', 'CoreData', 'Imported_Tops']

# Get property with special characters (sanitized)
# Original: "SW (v/v)" -> Accessible as: well.SW__v_v_
```

### 4. Loading Formation Tops

Formation tops create discrete properties that can be used for filtering. The DataFrame must have specific columns:

#### Required DataFrame Layout

| Column | Description | Example |
|--------|-------------|---------|
| Well column | Well identifier (matches LAS WELL field) | `'12/3-4 A'` |
| Depth column | Measured depth of the top | `2850.0` |
| Discrete column | Name/label of the formation | `'Top_Brent'` |

```python
import pandas as pd

# Create tops DataFrame
tops_df = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 A', '12/3-4 B', '12/3-4 B'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Cook', 'Top_Brent', 'Top_Statfjord'],
    'MD': [2850.0, 3100.0, 3400.0, 2900.0, 3150.0]
})

# Load tops
manager.load_tops(
    tops_df,
    property_name='Zone',          # Name for the discrete property
    well_col='Well',               # Column containing well names
    discrete_col='Surface',        # Column containing formation names
    depth_col='MD'                 # Column containing depths
)

# Tops are added as a new source called 'Imported_Tops'
print(manager.well_12_3_4_A.sources)  # [..., 'Imported_Tops']

# Access the discrete property
zone = manager.well_12_3_4_A.Imported_Tops.Zone
print(zone.type)    # 'discrete'
print(zone.labels)  # {0: 'Top_Brent', 1: 'Top_Statfjord', 2: 'Top_Cook'}
```

**Important**: Tops represent the **start** of each interval. The toolkit uses forward-fill interpolation, meaning each top's label applies from its depth downward until the next top.

### 5. Adding Labels for Discrete Logs

Discrete logs (facies, flags, zones) need labels to make statistics readable:

```python
well = manager.well_12_3_4_A

# Get the property
ntg_flag = well.get_property('NTG_Flag')

# Set as discrete
ntg_flag.type = 'discrete'

# Add labels
ntg_flag.labels = {
    0: 'NonNet',
    1: 'Net'
}

# Now filtering will use these labels
stats = well.PHIE.filter('NTG_Flag').sums_avg()
# Returns: {'NonNet': {...}, 'Net': {...}}
# Instead of: {'NTG_Flag_0': {...}, 'NTG_Flag_1': {...}}
```

Labels are automatically preserved when exporting to LAS format and recovered when loading.

### 6. Hierarchical Filtering and Statistics

The real power comes from chaining multiple filters:

```python
well = manager.well_12_3_4_A

# Single filter - group by Zone
stats = well.PHIE.filter('Zone').sums_avg()
# {'Top_Brent': {...}, 'Top_Statfjord': {...}}

# Chain filters - group by Zone, then by NTG_Flag
stats = well.PHIE.filter('Zone').filter('NTG_Flag').sums_avg()
# {
#   'Top_Brent': {
#     'Net': {'mean': 0.21, 'samples': 150, 'thickness': 150.0, ...},
#     'NonNet': {'mean': 0.08, 'samples': 100, 'thickness': 100.0, ...}
#   },
#   'Top_Statfjord': {
#     'Net': {...},
#     'NonNet': {...}
#   }
# }

# Add more filters as needed
stats = well.PHIE.filter('Zone').filter('Facies').filter('NTG_Flag').sums_avg()

# Include arithmetic statistics for comparison
stats = well.PHIE.filter('Zone').sums_avg(arithmetic=True)
# Returns both: {'mean': {'weighted': 0.21, 'arithmetic': 0.19}, ...}
```

Each statistics dictionary contains:

**Core Statistics** (depth-weighted by default):
- `mean` - Average value (weighted or arithmetic based on parameters)
- `sum` - Sum of values (weighted or arithmetic)
- `std_dev` - Standard deviation
- `percentile` - Dictionary with `p10`, `p50`, `p90` values
- `range` - Dictionary with `min`, `max` values

**Metadata**:
- `samples` - Number of non-NaN samples
- `thickness` - Depth interval for this group (sum of intervals)
- `gross_thickness` - Total depth interval across all groups
- `thickness_fraction` - Fraction of gross thickness (thickness / gross_thickness)
- `depth_range` - Dictionary with `min`, `max` depth within the zone
- `calculation` - Method used: 'weighted', 'arithmetic', or 'both'

**Multi-method output** (when `arithmetic=True`):
- Values become dictionaries: `{'weighted': x, 'arithmetic': y}`

#### Why Weighted Statistics Matter

Standard arithmetic averaging treats all samples equally, which is incorrect when sample spacing is irregular:

```python
# Example: NTG flag with irregular sampling
# NTG=0 @ 1500m, NTG=1 @ 1501m, NTG=0 @ 1505m
#
# Arithmetic mean: (0 + 1 + 0) / 3 = 0.33 ❌
# This incorrectly suggests only 33% net
#
# Weighted mean: considers that the middle sample represents 2.5m
# (0×0.5 + 1×2.5 + 0×2.0) / 5.0 = 0.50 ✓
# This correctly reflects that 50% of the interval is net
```

#### Accurate Zone Boundary Handling

When filtering by zones, the toolkit automatically inserts synthetic samples at zone boundaries to properly partition intervals:

```python
# Log samples: NTG=0 @ 1500m, NTG=1 @ 1501m, NTG=0 @ 1505m
# Zone boundary at 1503m

# Without boundary insertion:
# Zone 1 would incorrectly get the full NTG=1 sample

# With boundary insertion:
# A synthetic sample is inserted at 1503m
# Zone 1 (1500-1503): 1.5m net out of 2.0m total
# Zone 2 (1503-1505): 2.0m net out of 3.0m total
```

This ensures statistics accurately reflect the true distribution of properties within each zone.

### 7. Exporting Data

#### To DataFrame

```python
# Export all properties
df = well.data()

# Export specific properties
df = well.data(include=['PHIE', 'SW', 'PERM'])

# Auto-resample to common depth grid (when properties have different depths)
df = well.data(auto_resample=True)

# Apply discrete labels in output
df = well.data(discrete_labels=True)
# Zone column will contain 'Top_Brent', 'Top_Statfjord' instead of 0, 1
```

#### To LAS File

```python
# Export all properties
well.export_to_las('output.las')

# Export specific properties
well.export_to_las('output.las', include=['PHIE', 'SW', 'PERM'])

# Preserve original metadata (recommended)
well.export_to_las('output.las', use_template=True)

# Export each source as separate LAS file
well.export_sources('output_folder/')
```

### 8. Removing and Renaming Sources

Manage sources within a well:

```python
well = manager.well_12_3_4_A

# List sources
print(well.sources)  # ['Petrophysics', 'CoreData', 'Imported_Tops']

# Rename a source
well.rename_source('CoreData', 'Core_Porosity')
print(well.sources)  # ['Petrophysics', 'Core_Porosity', 'Imported_Tops']

# Remove a source (and all its properties)
well.remove_source('Core_Porosity')
print(well.sources)  # ['Petrophysics', 'Imported_Tops']
```

When you save the project, renamed files will be updated and removed sources will be deleted from disk.

### 9. Project Persistence

Save and load entire analysis projects:

```python
# Save project to folder
manager.save('my_project/')

# This creates:
# my_project/
#   well_12_3_4_A/
#     Petrophysics.las
#     Imported_Tops.las
#   well_12_3_4_B/
#     Petrophysics.las
#     Imported_Tops.las

# Load project
manager = WellDataManager('my_project/')
# All wells and sources are restored automatically

# Or load into existing manager
manager = WellDataManager()
manager.load('my_project/')

# Save back to same location
manager.save()

# Save to new location
manager.save('backup_project/')
```

The project structure preserves:
- All wells and their properties
- Source organization
- Discrete labels and types
- Property metadata (units, descriptions)

### 10. Adding External Data

Add data from DataFrames as new sources:

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
    source_name='CoreData',  # Optional, defaults to 'external_df'
    unit_mappings={'CorePHIE': 'v/v', 'CorePERM': 'mD'},
    type_mappings={'CorePHIE': 'continuous'},
    label_mappings={}  # For discrete properties
)

# Access new properties
print(well.CoreData.CorePHIE.values)
```

### 11. Sampled Data (Core Plugs)

Core plug measurements are point samples, not continuous logs. They require different statistical treatment:

```python
# Load core plug data as sampled
manager.load_las('core_plugs.las', sampled=True)

# Or mark individual properties as sampled
well.CorePHIE.type = 'sampled'

# Sampled data behavior:
# - Uses arithmetic mean by default (each plug counts equally)
# - No boundary insertion when filtering (preserves original measurements)
# - Weighted statistics not appropriate for point measurements
```

**Filtering sampled data**:

```python
# Filtering core plugs by zone
core_phie = well.CorePHIE
core_phie.type = 'sampled'

# No synthetic samples inserted at zone boundaries (preserves plug locations)
stats = core_phie.filter('Zone').sums_avg()
# {
#   'Top_Brent': {
#     'mean': 0.205,           # Arithmetic mean (each plug equally weighted)
#     'samples': 12,           # Number of core plugs in zone
#     'calculation': 'arithmetic'
#   },
#   'Top_Statfjord': {...}
# }

# Can override boundary insertion if needed
filtered = core_phie.filter('Zone', insert_boundaries=True)

# Can force weighted statistics if desired
stats = core_phie.filter('Zone').sums_avg(weighted=True)
```

**Why sampled matters**:

```python
# Continuous log: zone boundary at 2850m
# Log samples at: 2848, 2849, 2850, 2851, 2852
# → Synthetic sample inserted at 2850m for accurate zone attribution

# Core plugs at: 2848, 2851, 2855
# → No synthetic samples - these are discrete physical measurements
# → Each plug is assigned to zone based on its actual depth
```

### 12. Property Computation Operations

Create new properties using intuitive mathematical expressions. The toolkit enforces **strict depth matching** (like numpy) - operations fail if depths don't match exactly. Use explicit `.resample()` to align properties with different sampling rates.

#### Arithmetic Operations

```python
well = manager.well_12_3_4_A

# Scalar operations - multiply, divide, add, subtract
well.PHIE_fraction = well.PHIE * 0.01
well.PHIE_percent = well.PHIE * 100
well.PHIE_adjusted = well.PHIE - 0.02
well.PHIE_doubled = well.PHIE * 2

# Property-to-property operations (same depth grid required)
well.HC_Volume = well.PHIE * (1 - well.SW)  # Works if depths match
well.Porosity_Diff = well.PHIE - well.PHIT

# Complex expressions
well.Bulk_Volume = well.PHIE / (well.NTG + 0.001)  # Avoid division by zero
well.Combined = (well.PHIE + well.PERM * 0.001) / 2

# Unary operations
well.Neg_PHIE = -well.PHIE
well.Abs_Delta = abs(well.PHIE - well.PHIT)

# Assignment behavior
well.new_property = well.PHIE * 2  # Creates NEW property
well.PHIE = well.PHIE * 2          # OVERWRITES existing PHIE data
```

**Strict Depth Matching**: Operations fail if properties have different depth grids (like numpy's broadcasting rules). This prevents silent interpolation errors and makes data handling explicit.

#### Property Inspection

Properties can be printed to inspect their data in a numpy-style format:

```python
# Print property (numpy-style display)
print(well.PHIE)
# depth: 2800.00, 2801.00, 2802.00, 2803.00, 2804.00, ..., 3796.00, 3797.00, 3798.00, 3799.00, 3800.00
# PHIE (v/v): 0.180, 0.185, 0.192, 0.188, 0.175, ..., 0.220, 0.218, 0.215, 0.212, 0.210

# Detailed repr
repr(well.PHIE)
# Property(name='PHIE' (v/v), samples=1001)
#   depth: [2800.00, ..., 3800.00]
#   PHIE: [0.180, 0.185, 0.192, ..., 0.212, 0.210]
```

#### Explicit Resampling

When properties have different depth grids, use `.resample()` to explicitly align them:

```python
# Different sampling rates - operation fails
try:
    result = well.PHIE + well.CorePHIE  # Error: depths don't match
except DepthAlignmentError as e:
    print(e)
    # DepthAlignmentError: Cannot combine properties with different depth grids.
    #   PHIE: 1000 samples (2800.00-3800.00m, 1.00m spacing)
    #   CorePHIE: 50 samples (2800.00-3800.00m, 20.00m spacing)

# Explicit resampling - now it works
core_resampled = well.CorePHIE.resample(well.PHIE)
result = well.PHIE + core_resampled

# Or pass Property directly
core_resampled = well.CorePHIE.resample(well.PHIE)

# Resample to custom grid
target_depth = np.arange(2800, 3800, 0.5)  # 0.5m regular grid
phie_fine = well.PHIE.resample(target_depth)
```

**Resampling Notes**:
- Uses linear interpolation for continuous properties
- Uses nearest-neighbor for discrete properties
- NaN values are excluded from interpolation
- Values outside original depth range become NaN
- Always creates a new Property (non-destructive)

#### Comparison Operations

Create discrete flag properties using comparison operators. These are automatically marked as discrete type:

```python
# Simple comparisons
well.High_Poro = well.PHIE > 0.15          # Flag: porosity > 15%
well.Low_Water = well.SW < 0.35            # Flag: water saturation < 35%
well.Good_Perm = well.PERM >= 100          # Flag: permeability >= 100mD

# Comparison operations return discrete properties with auto-generated labels
print(well.High_Poro.type)    # 'discrete'
print(well.High_Poro.labels)  # {0: 'False', 1: 'True'}
print(well.High_Poro)
# depth: 2800.00, 2801.00, ..., 3799.00, 3800.00
# High_Poro (flag): 0.000, 0.000, ..., 1.000, 1.000

# Use in filtering
stats = well.PHIE.filter('High_Poro').sums_avg()
# {'False': {...}, 'True': {...}}
```

#### Logical Operations

Combine conditions using logical operators to create complex selection criteria:

```python
# AND operation - reservoir flag
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)

# OR operation - either condition
well.Problem_Zone = (well.PHIE < 0.05) | (well.SW > 0.90)

# NOT operation - invert flag
well.Non_Reservoir = ~well.Reservoir

# Complex logic
well.Good_Reservoir = (well.PHIE > 0.15) & (well.SW < 0.4) & (well.PERM > 50)

# Use in filtering
stats = well.PHIE.filter('Reservoir').sums_avg()
# {'False': {...}, 'True': {...}}
```

#### Manager-Level Broadcasting

Apply operations to all wells at once. The operation is automatically applied to every well that has the source property:

```python
# Scale property across all wells
manager.PHIE_scaled = manager.PHIE * 100

# Create flags across all wells
manager.High_Poro = manager.PHIE > 0.15

# Complex operations
manager.HC_Volume = manager.PHIE * (1 - manager.SW)

# Output:
# ✓ Created property 'HC_Volume' in 12 well(s)
# ⚠ Skipped 3 well(s) without property 'PHIE' or 'SW'
```

Wells without the required property are automatically skipped with a warning. The created properties are added to each well's `computed` source.

#### The 'computed' Source

All computed properties are stored in a special source called `computed`:

```python
# View all computed properties
print(well.sources)  # ['Petrophysics', 'Imported_Tops', 'computed']
print(well.computed.properties)  # ['PHIE_scaled', 'High_Poro', 'HC_Volume', ...]

# Access computed properties
hc_volume = well.computed.HC_Volume

# Or directly via well
hc_volume = well.HC_Volume

# Export computed properties
well.export_to_las('output_with_computed.las')

# Remove computed source
well.remove_source('computed')
```

#### Property Operations in Workflows

Combine property operations with filtering and statistics:

```python
# Create reservoir flag
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)

# Compute hydrocarbon volume
well.HC_Volume = well.PHIE * (1 - well.SW)

# Filter by zone and reservoir flag
stats = well.HC_Volume.filter('Zone').filter('Reservoir').sums_avg()
# {
#   'Top_Brent': {
#     'True': {'mean': 0.14, 'sum': 35.0, 'thickness': 250.0, ...},
#     'False': {...}
#   },
#   'Top_Statfjord': {...}
# }

# Export with computed properties
well.export_to_las('analysis_output.las')
```

#### Advanced Examples

```python
# Normalize property to 0-1 range
min_val = np.nanmin(well.PHIE.values)
max_val = np.nanmax(well.PHIE.values)
well.PHIE_normalized = (well.PHIE - min_val) / (max_val - min_val)

# Create quality score
well.Quality_Score = (well.PHIE * 0.4 +
                      (1 - well.SW) * 0.3 +
                      well.PERM / 1000 * 0.3)

# Multi-tier reservoir classification
well.Tier_1 = (well.PHIE > 0.20) & (well.SW < 0.30) & (well.PERM > 200)
well.Tier_2 = (well.PHIE > 0.15) & (well.SW < 0.40) & (well.PERM > 100)
well.Tier_3 = (well.PHIE > 0.10) & (well.SW < 0.50) & (well.PERM > 50)

# Compute tier statistics
for tier in ['Tier_1', 'Tier_2', 'Tier_3']:
    stats = well.PHIE.filter(tier).sums_avg()
    print(f"{tier}: {stats['True']['thickness']} m thickness")
```

#### Notes

- **Strict Depth Matching**: Operations fail if depths don't match (like numpy). Use `.resample()` to explicitly align
- **Assignment Behavior**:
  - `well.new_prop = expr` creates NEW property in `computed` source
  - `well.existing_prop = expr` OVERWRITES existing property data
- **NaN Handling**: NaN values propagate through operations (numpy default behavior)
- **Unit Tracking**: Units are tracked but not validated. Combined units shown as `unit1*unit2`, etc.
- **Auto-Labels**: Comparison operations automatically get `{0: 'False', 1: 'True'}` labels
- **Computed Properties**: Automatically marked with `source_name='computed'`
- **Type Conversion**: Comparison operations automatically create discrete properties
- **Error Handling**: Operations raise clear errors for mismatched depths

---

## Advanced Usage

### Standalone Statistics Functions

Use the statistics functions directly for custom analysis:

```python
from well_log_toolkit import (
    compute_intervals,
    mean,
    sum,
    std,
    percentile,
    compute_all_statistics
)
import numpy as np

# Compute depth intervals
depth = np.array([1500.0, 1501.0, 1505.0])
intervals = compute_intervals(depth)
# [0.5, 2.5, 2.0] - midpoint method

# Unified statistics functions with method parameter
values = np.array([0.15, 0.22, 0.18])

# Returns dict with both methods when no method specified
result = mean(values, intervals)
# {'weighted': 0.196, 'arithmetic': 0.183}

# Returns single value when method specified
w_mean = mean(values, intervals, method='weighted')
a_mean = mean(values, intervals, method='arithmetic')

# Same pattern for sum, std, percentile
w_std = std(values, intervals, method='weighted')
p50 = percentile(values, 50, intervals, method='weighted')

# For NTG flags, weighted sum gives net thickness
ntg = np.array([0, 1, 0])
net_thickness = sum(ntg, intervals, method='weighted')  # 2.5m

# Get all statistics at once
all_stats = compute_all_statistics(values, depth)
# Returns dict with weighted_mean, weighted_sum, weighted_std,
# weighted_p10/p50/p90, arithmetic_mean, arithmetic_std,
# count, depth_thickness, min, max
```

### Merging Properties

Combine properties from different sources onto a common depth grid:

```python
# Resample all properties to regular depth grid
merged_df = well.merge(method='resample', depth_step=0.1)

# Concatenate unique depths
merged_df = well.merge(method='concat')
```

### Direct LAS Export from DataFrame

Export any DataFrame to LAS format without using Well objects:

```python
from well_log_toolkit import LasFile

df = pd.DataFrame({
    'DEPT': [2800, 2801, 2802],
    'PHIE': [0.20, 0.22, 0.19],
    'Zone': [0, 1, 1]
})

LasFile.export_las(
    'output.las',
    well_name='12/3-4 A',
    df=df,
    unit_mappings={'DEPT': 'm', 'PHIE': 'v/v'},
    discrete_labels={'Zone': {0: 'Brent', 1: 'Statfjord'}}
)
```

### Updating Curve Metadata

Modify curve information without reloading data:

```python
# Access the LAS file object
las = well.original_las

# Update single curve
las.update_curve('PERM', unit='mD', description='Permeability')

# Apply multiplier (useful for unit conversion)
las.update_curve('PERM', multiplier=0.001)  # Convert to Darcy

# Bulk update
las.bulk_update_curves({
    'PHIE': {'unit': 'v/v', 'description': 'Effective Porosity'},
    'SW': {'unit': 'v/v', 'description': 'Water Saturation'}
})
```

---

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
