# Well Log Toolkit

A Python library for petrophysical well log analysis with lazy loading, multi-well management, and hierarchical filtering. Designed for efficient processing of LAS files with automatic statistics computation by zone and facies.

## Key Features

- **Lazy Loading** - Headers parsed immediately, data loaded only when accessed
- **Multi-Well Management** - Intuitive attribute-based access to wells and properties
- **Hierarchical Filtering** - Chain discrete filters and compute grouped statistics
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
    property_name='Zone',
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
#     'mean': 0.182,
#     'sum': 45.5,
#     'count': 250,
#     'depth_thickness': 250.0,
#     'min': 0.05,
#     'max': 0.28,
#     'std': 0.045
#   },
#   'Top_Statfjord': {...}
# }
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
#     'Net': {'mean': 0.21, 'count': 150, 'depth_thickness': 150.0, ...},
#     'NonNet': {'mean': 0.08, 'count': 100, 'depth_thickness': 100.0, ...}
#   },
#   'Top_Statfjord': {
#     'Net': {...},
#     'NonNet': {...}
#   }
# }

# Add more filters as needed
stats = well.PHIE.filter('Zone').filter('Facies').filter('NTG_Flag').sums_avg()
```

Each statistics dictionary contains:
- `mean` - Average value
- `sum` - Sum of values
- `count` - Number of samples
- `depth_thickness` - Total depth interval (count × depth step)
- `min` - Minimum value
- `max` - Maximum value
- `std` - Standard deviation

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

---

## Advanced Usage

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
