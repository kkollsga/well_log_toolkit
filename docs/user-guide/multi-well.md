# Multi-Well Analysis

## WellDataManager

`WellDataManager` is the central orchestrator for multi-well workflows:

```python
from pylog import WellDataManager

manager = WellDataManager()
manager.load_las("well_a.las")
manager.load_las("well_b.las")
manager.load_las("well_c.las")
```

## Accessing Wells

```python
# By sanitized attribute name
well = manager.well_12_3_2_B

# List all wells
print(manager.well_names)
```

## Broadcasting

Access properties across all wells simultaneously:

```python
# Mean porosity per well
means = manager.PHIE.mean()
# {'well_A': 0.185, 'well_B': 0.192, ...}

# Filtered statistics across all wells
stats = manager.PHIE.filter('Zone').sums_avg()
# {'well_A': {'Reservoir': {...}, 'NonReservoir': {...}}, ...}
```

## Data Validation

Check data integrity across all wells:

```python
issues = manager.validate()
# {'well_B': ['Missing property: SW', ...]}
```

## Skipped Well Warnings

When a property doesn't exist in some wells, the manager warns:

```python
import warnings
warnings.simplefilter("always")

# If well_C lacks PHIE, you'll see:
# UserWarning: Skipped 1 well(s) without property 'PHIE': well_C
means = manager.PHIE.mean()
```
