# Quick Start

A 5-minute introduction to well-log-toolkit.

## Loading LAS Files

```python
from well_log_toolkit import WellDataManager

manager = WellDataManager()
manager.load_las("12_3-2_B.las")

# Access wells by sanitized name
well = manager.well_12_3_2_B
print(well.property_names)
```

## Working with Properties

```python
# Access properties as attributes
phie = well.PHIE
print(f"Mean porosity: {phie.mean():.3f}")
print(f"Range: {phie.min():.3f} - {phie.max():.3f}")

# Create computed properties
well.HC_Volume = well.PHIE * (1 - well.SW)
```

## Filtering and Statistics

```python
# Mark discrete properties
zone = well.get_property('Zone')
zone.type = 'discrete'
zone.labels = {0: 'NonReservoir', 1: 'Reservoir'}

# Chain filters and compute statistics
stats = well.PHIE.filter('Zone').filter('NTG_Flag').sums_avg()
# Returns nested dict: {'Reservoir': {'Net': {...}, 'NonNet': {...}}, ...}
```

## Multi-Well Analysis

```python
# Load multiple wells
manager.load_las("well_a.las").load_las("well_b.las")

# Broadcast operations across all wells
all_means = manager.PHIE.mean()
zone_stats = manager.PHIE.filter('Zone').sums_avg()
```

## Visualization

```python
from well_log_toolkit import Template, WellView

# Define track layout
template = Template()
template.add_track("Porosity", width=2)
template.add_curve("Porosity", "PHIE", color="blue")

# Create well view
view = WellView(well, template)
view.plot()
```

## Next Steps

- {doc}`user-guide/loading-data` for LAS file loading details
- {doc}`user-guide/wells-properties` for property operations
- {doc}`user-guide/statistics` for depth-weighted calculations
- {doc}`user-guide/visualization` for plotting options
