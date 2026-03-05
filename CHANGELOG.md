# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Future entries are managed by [towncrier](https://towncrier.readthedocs.io/).
To add a changelog entry, create a file in the `changes/` directory.

<!-- towncrier release notes start -->

## [0.1.158] - 2026-03-05

### Documentation

- **Sphinx documentation**: Full docs site with furo theme, MyST markdown,
  and autodoc API reference. Includes installation, quickstart, user guide
  (loading data, wells/properties, statistics, visualization, regression,
  multi-well), and API reference for all public classes and functions.
- **ReadTheDocs configuration**: `.readthedocs.yaml` for automated builds.
- **Documentation dependencies**: Added `[docs]` optional dependency group.
- **Repo rename**: Updated all project URLs from `well_log_toolkit` to
  `well-log-toolkit` to match PyPI package name.

## [0.1.157] - 2026-03-05

### Features

- **`geometric_mean()` and `harmonic_mean()`**: New statistical functions for
  permeability averaging (geometric) and parallel flow averaging (harmonic).
  Both support depth-weighted and arithmetic modes.
- **`Property.apply(func)`**: Apply arbitrary functions to property values,
  returning a new Property on the same depth grid.
- **`Property.histogram(bins, weighted)`**: Compute histogram of property
  values, optionally weighted by depth intervals.
- **`PolynomialExponentialRegression`**: Now exported at top-level
  (`from well_log_toolkit import PolynomialExponentialRegression`).
- **Basic LAS 3.0 support**: Read LAS 3.0 files with `~Log_Definition` and
  `~Log_Data` sections (single data section, tab-delimited).

## [0.1.156] - 2026-03-05

### Performance

- **Vectorized `compute_zone_intervals()`**: Replaced Python for-loop with numpy
  vectorized operations for zone interval calculation, matching the existing
  `compute_intervals()` pattern.
- Added performance benchmark test for `compute_zone_intervals` with 10K-point arrays.

## [0.1.155] - 2026-03-05

### Features

- **Fuzzy name matching**: Typos in property/well names now show "Did you mean: ..."
  suggestions using `difflib.get_close_matches()`.
- **`WellDataManager.validate()`**: New method to check data integrity across all wells
  (missing properties, depth monotonicity, length mismatches).
- **Proxy stats warnings**: `manager.PHIE.mean()` now warns when wells are silently
  skipped because they lack the requested property.
- **Input validation**: `Property()` now rejects mismatched depth/values lengths and
  non-monotonic depth arrays. `LasFile()` rejects non-`.las` file extensions.

### Testing

- Added `tests/conftest.py` with 10 centralized pytest fixtures.
- Added 4 new test files: `test_error_handling.py` (20 tests), `test_resample_edge_cases.py`
  (6 tests), `test_las_export_roundtrip.py` (5 tests), `test_sums_avg_report.py` (8 tests).
- Fixed `return True/False` → `assert`/`pytest.skip()` across 20 test files,
  eliminating all 62 `PytestReturnNotNoneWarning` warnings.
- Moved 6 demo/example scripts from `tests/` to `examples/`.

### Documentation

- Added `See Also` cross-references to key methods (`filter`, `sums_avg`, `resample`,
  `get_property`, `load_las`, `mean`).

## [0.1.154] - 2026-03-05

### Breaking Changes

- Restructured package into domain-driven subpackages. Direct imports from flat
  modules (e.g., `from well_log_toolkit.statistics import mean`) must now use
  subpackage paths (e.g., `from well_log_toolkit.analysis.statistics import mean`).
  Top-level imports (`from well_log_toolkit import Well, Property`) are unchanged.

### Internal Changes

- Created `io/` subpackage with `las_file.py` for LAS file I/O.
- Created `core/` subpackage with `well.py`, `property.py`, `operations.py`.
- Created `analysis/` subpackage with `statistics.py`, `regression.py`, `sums_avg.py`.
- Created `manager/` subpackage, splitting 3500-line `manager.py` into `data_manager.py` and `proxy.py`.
- Created `visualization/` subpackage, splitting 5050-line `visualization.py` into `template.py`, `wellview.py`, `crossplot.py`.
- Extracted `_version.py` for dynamic version detection.
- Updated all test imports to match new package structure.

## [0.1.153] - 2026-03-04

### Internal Changes

- Moved test directory from `pytest/` to `tests/` to match standard Python conventions.
- CI now runs the full pytest suite across Python 3.10-3.13 instead of a smoke import.
- Dropped Python 3.9 support; minimum version is now 3.10.
- Added towncrier-based changelog system for fragment-based release notes.
- Updated project classifier from Alpha to Beta.
- Updated project description to reflect full petrophysical analysis capabilities.
- Fixed issues URL in README (was pointing to `yourusername` placeholder).
- Added PyPI and CI status badges to README.

## [0.1.152] - 2024-12-29

### Summary

This is a retrospective entry covering the cumulative state of the library at v0.1.152.

### Features

- Lazy LAS 2.0 file reader with header-only parsing and on-demand data loading.
- `Property` class with depth-weighted statistics (mean, sum, std, percentile, mode).
- Chained hierarchical filtering: `well.PHIE.filter('Zone').filter('Facies').sums_avg()`.
- Numpy-style operator overloading on Property objects (24 operators).
- `WellDataManager` for multi-well orchestration with property broadcasting.
- `_ManagerPropertyProxy` for cross-well operations: `manager.PHIE.filter('Zone').mean()`.
- `SumsAvgResult.report()` with arithmetic, geometric, pooled, and sum aggregation.
- Template-driven well log visualization (`Template`, `WellView`).
- Interactive crossplots with multi-dimensional mapping (`Crossplot`).
- 6 regression types: linear, polynomial, exponential, logarithmic, power, polynomial-exponential.
- Parameter locking for regression coefficients.
- Discrete property support with label/color/style/thickness mappings.
- Source-aware property storage for round-trip LAS export.
- Computed property creation via `well.HC = well.PHIE * (1 - well.SW)`.
- Project save/load to JSON with full metadata preservation.
- Depth interval computation using midpoint method with zone boundary truncation.
- Property resampling with configurable methods (linear, nearest, forward-fill).
- Strict depth alignment enforcement with detailed error guidance.
