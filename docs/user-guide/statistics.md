# Statistics

## Depth-Weighted vs Arithmetic

Well log data is sampled at depth intervals that may not be uniform.
Depth-weighted statistics account for the thickness each sample represents,
giving physically meaningful results.

```python
from logsuite import mean, compute_intervals

# Compute depth intervals (midpoint method)
intervals = compute_intervals(depth)

# Both methods
result = mean(values, intervals)
# {'weighted': 0.185, 'arithmetic': 0.190}

# Specific method
weighted_mean = mean(values, intervals, method='weighted')
```

## Available Functions

### Arithmetic and Weighted

All functions support `method='weighted'`, `method='arithmetic'`, or `method=None`
(returns dict with both):

- `mean()` — Arithmetic mean
- `sum()` — Sum (weighted sum = cumulative thickness)
- `std()` — Standard deviation
- `percentile(values, p, weights)` — Percentile (0–100)
- `mode()` — Mode (binned for continuous, exact for discrete)

### Specialized Means

For permeability and other log-normally distributed properties:

- `geometric_mean()` — exp(mean(ln(v))): appropriate for permeability averaging
- `harmonic_mean()` — n / sum(1/v): appropriate for parallel flow properties

```python
from logsuite import geometric_mean, harmonic_mean

# Permeability averaging
geo = geometric_mean(perm_values, intervals, method='weighted')
har = harmonic_mean(perm_values, intervals, method='weighted')

# Always: harmonic <= geometric <= arithmetic
```

### Comprehensive Statistics

```python
from logsuite import compute_all_statistics

stats = compute_all_statistics(values, depth)
# Returns dict with weighted_mean, arithmetic_mean, weighted_std,
# percentiles (P10, P50, P90), count, thickness, min, max
```

## sums_avg()

The `sums_avg()` method on Property computes full statistics with optional
hierarchical filtering:

```python
# Unfiltered
result = well.PHIE.sums_avg()
# SumsAvgResult with weighted/arithmetic mean, sum, std, percentiles

# With filters (nested dict output)
result = well.PHIE.filter('Zone').filter('NTG_Flag').sums_avg()
# {'Reservoir': {'Net': SumsAvgResult, 'NonNet': SumsAvgResult}, ...}
```

## Interval Computation

```python
from logsuite import compute_intervals
from logsuite.analysis.statistics import compute_zone_intervals

# Full depth grid intervals (midpoint method)
intervals = compute_intervals(depth)

# Zone-truncated intervals
zone_intervals = compute_zone_intervals(depth, top=2900.0, base=3100.0)
```
