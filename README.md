# Well Log Toolkit

Fast LAS file processing with lazy loading and filtering for well log analysis.

## Features

- **Lazy Loading**: Efficient reading of large LAS files
- **Multi-well Management**: Orchestrate analysis across multiple wells
- **Property Filtering**: Chain filters on well log properties
- **Type Support**: Handle both continuous and discrete logs
- **Statistics**: Compute statistics on filtered data
- **LAS Export**: Export processed data back to LAS 2.0 format
- **Template-Based Export**: Preserve original metadata when updating LAS files

## Installation

```bash
pip install well-log-toolkit
```

## Quick Start

```python
from well_log_toolkit import WellDataManager

# Load LAS files
manager = WellDataManager()
manager.load_las("well1.las").load_las("well2.las")

# Access well and properties
well = manager.well_12_3_2_B

# Mark discrete logs and add labels
zone_prop = well.get_property('Zone')
zone_prop.type = 'discrete'
zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir'}

ntg_prop = well.get_property('NTG_Flag')
ntg_prop.type = 'discrete'
ntg_prop.labels = {0: 'NonNet', 1: 'Net'}

# Filter and compute statistics (labels appear in output)
stats = well.phie.filter('Zone').filter('NTG_Flag').sums_avg()
# Returns: {'Reservoir': {'Net': {...}, 'NonNet': {...}}, ...}

# Export to LAS format (labels stored in ~Parameter section)
well.export_to_las('output.las')  # All properties with labels
well.export_to_las('subset.las', include=['PHIE', 'SW'])  # Specific properties

# Template-based export: Preserve original metadata (prevents data erosion)
well.export_to_las('updated.las', use_template=True)
# Preserves: ~Version info, ~Well parameters, ~Curve descriptions, ~Parameters

# Access source LAS files
print(well.sources)  # ['well1.las', 'well2.las']
original = well.original_las  # First LAS file object loaded

# Properties know their source
print(well.phie.source)  # 'path/to/well1.las'

# Add DataFrame as source
df = pd.DataFrame({
    'DEPT': [2800, 2801],
    'PHIE': [0.2, 0.22],
    'Zone': [0, 1]
})
well.add_dataframe(
    df,
    unit_mappings={'PHIE': 'v/v', 'Zone': ''},
    type_mappings={'Zone': 'discrete'},
    label_mappings={'Zone': {0: 'NonReservoir', 1: 'Reservoir'}}
)
print(well.sources)  # ['well1.las', 'well2.las', 'external_df']

# Round-trip: Discrete properties and labels are automatically preserved
manager2 = WellDataManager()
manager2.load_las('output.las')  # Discrete types and labels auto-loaded!
well2 = manager2.well_12_3_2_B
print(well2.Zone.type)  # 'discrete'
print(well2.Zone.labels)  # {0: 'NonReservoir', 1: 'Reservoir'}

# Standalone export from any DataFrame
from well_log_toolkit import LasFile
LasFile.export_las(
    'output.las',
    well_name='12/3-2 B',
    df=df,
    unit_mappings={'DEPT': 'm', 'PHIE': 'v/v'},
    discrete_labels={'Zone': {0: 'NonReservoir', 1: 'Reservoir'}}
)

# Properties with special characters are automatically sanitized
# "Zoneloglinkedto'CerisaTops'" -> well.Zoneloglinkedto_CerisaTops_
zone_prop = well.Zoneloglinkedto_CerisaTops_
```

## Main Classes

### WellDataManager
Global orchestrator for multi-well analysis. Manages multiple wells with attribute-based access.

### Well
Single well containing multiple properties with convenient property access.

### Property
Single log property with depth-value pairs and filtering operations.

### LasFile
LAS file reader with lazy loading capabilities.

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
