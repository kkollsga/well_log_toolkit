# Pool raw plug values across wells in one call

You need a single long-form DataFrame of plug values + grouping columns,
pooled across every well, so you can compute pooled percentiles, build
diagnostic plots, or merge with external metadata. No per-well loop.

## The minimal pattern

```python
df = manager.PHIE.filter("Facies").data(weighted=True)
```

Result columns:

| well | DEPT | PHIE | Facies | Weight |
|------|------|------|--------|--------|
| Well_A | 2500.0 | 0.193 | Clean | 0.25 |
| Well_A | 2500.5 | 0.062 | Medium | 0.50 |
| ... | ... | ... | ... | ... |

* ``well`` — original well name (not the sanitized key).
* ``DEPT`` — depth.
* ``PHIE`` — the property values; uses your label strings if you set
  ``Property.labels``.
* ``Facies`` — one column per active filter, named after the filter
  property (no ``Group1/Group2``).
* ``Weight`` — depth-interval weight per row, present when
  ``weighted=True``. Use it for depth-weighted percentiles externally:
  ``np.average(df["PHIE"], weights=df["Weight"])``.

Wells lacking the property are silently skipped (with a warning). Pass
``warn_missing=False`` to silence the warning in scripted use.

## Multiple properties at once

```python
df = manager.properties(["PHIE", "PERM"]).filter("Facies").data()
```

Yields one column per property plus the filter columns; depths are
joined per well so the result is one row per (well, depth).

## Built-in summary

The same long-form data is available aggregated:

```python
manager.PHIE.filter("Facies").stats(
    return_df=True,
    flat_columns=True,           # Facies column instead of "Group"
    methods=["mean", "percentile_50", "percentile_90"],
)
```

## Verifying

* ``story_tests/story_3_pooled_data.py`` shows the basic shape.
* ``story_tests/story_6_pooled_extraction.py`` walks three real use
  cases — pooled percentile table, suspect-plug diagnostic subset, and
  cross-check against the library aggregation.
