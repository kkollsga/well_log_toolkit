# Well Log Toolkit

Fast LAS file processing with lazy loading and filtering for well log analysis.

## Features

- **Lazy Loading**: Efficient reading of large LAS files
- **Multi-well Management**: Orchestrate analysis across multiple wells
- **Property Filtering**: Chain filters on well log properties
- **Type Support**: Handle both continuous and discrete logs
- **Statistics**: Compute statistics on filtered data

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
