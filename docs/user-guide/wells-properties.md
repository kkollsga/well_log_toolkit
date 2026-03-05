# Wells and Properties

## Well Object

A `Well` contains multiple `Property` objects on potentially different depth grids.

```python
well = manager.well_12_3_2_B

# List properties
print(well.property_names)

# Access by attribute
phie = well.PHIE

# Access by name (useful for names with special characters)
prop = well.get_property('PHIE_2025')
```

### Computed Properties

Create new properties using mathematical expressions:

```python
# Arithmetic
well.HC_Volume = well.PHIE * (1 - well.SW)
well.PHIE_pct = well.PHIE * 100

# Boolean (creates 0/1 discrete property)
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)
```

## Property Object

### Basic Statistics

```python
prop = well.PHIE
print(f"Min: {prop.min()}")
print(f"Max: {prop.max()}")
print(f"Mean: {prop.mean()}")
```

### Filtering

Chain filters on discrete properties for hierarchical analysis:

```python
# Single filter
stats = well.PHIE.filter('Zone').sums_avg()

# Chained filters
stats = well.PHIE.filter('Zone').filter('Facies').sums_avg()
# Returns: {'Zone_A': {'Facies_1': {...}, 'Facies_2': {...}}, ...}
```

### Transform with apply()

Apply arbitrary functions to property values:

```python
log_perm = well.PERM.apply(np.log10, name='LOG_PERM')
normalized = well.PHIE.apply(lambda v: (v - v.min()) / (v.max() - v.min()))
```

### Histogram

```python
counts, edges = well.PHIE.histogram(bins=30)
counts_weighted, edges = well.PHIE.histogram(bins=30, weighted=True)
```

### Resampling

Align properties to the same depth grid:

```python
# Resample core data to log depth grid
core_resampled = well.CorePHIE.resample(well.PHIE)
difference = well.PHIE - core_resampled
```

## Depth Alignment

Operations between properties require matching depth grids. If grids differ,
you must resample explicitly:

```python
# This will raise DepthAlignmentError if grids differ:
result = well.PHIE + well.CorePHIE  # Error!

# Resample first:
core_resampled = well.CorePHIE.resample(well.PHIE)
result = well.PHIE + core_resampled  # OK
```
