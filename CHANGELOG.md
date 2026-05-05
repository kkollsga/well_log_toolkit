# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Future entries are managed by [towncrier](https://towncrier.readthedocs.io/).
To add a changelog entry, create a file in the `changes/` directory.

<!-- towncrier release notes start -->

## [0.2.4] - 2026-05-05

### Features

- Added ``_ManagerMultiPropertyProxy.fit_per(group_property, model, ...)`` — fits one regression per unique value of a discrete grouping property and returns a ``{label: RegressionFit}`` dict. Mirrors ``Crossplot.add_regression_per`` at the proxy level so users can produce per-group fits for tables and reports without going through a plot. Each group gets a fresh deep-copy of ``model`` so locked params and degree settings persist across groups; subsets smaller than ``min_samples`` (default 5) are skipped with a warning. Also available through ``manager.filter(where=...)`` views via ``_ValueFilteringProxy.fit_per``. ([#proxy-fit-per](https://github.com/kkollsga/logsuite/issues/proxy-fit-per))
- Added ``pool=True`` to ``_ManagerPropertyProxy.stats()``. When set, returns a DataFrame with one row per filter-group combination instead of one row per (well, group) — ``mean``, ``std``, ``min``, ``max``, and percentiles are computed across the pooled long-form data, so dispersion statistics are correct (not the mean of per-well point estimates). The new ``samples`` method counts data points per group. Percentile methods accept both long form (``percentile_10``) and short form (``p10``). ([#stats-pool](https://github.com/kkollsga/logsuite/issues/stats-pool))

### Bug Fixes

- Fixed: ``manager.filter(where={"Zone": "Reservoir"}).PHIE.data()`` now actually filters by Zone. Previously the post-filter on ``_ValueFilteringProxy.data()`` silently no-op'd when the filter column was missing from the underlying proxy's chain — the column wasn't in the DataFrame and the ``col in df.columns`` guard skipped. The wrapper now auto-adds ``proxy.filter(col)`` for any ``where=`` key whose column the proxy doesn't already emit, so the where clause is applied consistently across ``view.PROP.data()``, ``view.properties([...]).data()``, and ``view.PROP_PAIR.fit()`` paths. ([#where-actually-filters](https://github.com/kkollsga/logsuite/issues/where-actually-filters))


## [0.2.3] - 2026-05-05

### Features

- ``Crossplot.add_regression`` gained ``equation_format=`` and ``decimals=`` arguments that propagate to the legend label. ``equation_format="petrel"`` renders an exponential fit as ``pow(10, c1*x + c0)`` — copy-paste ready for Petrel calculator workflows. ``decimals`` overrides the default 4-decimal precision (e.g. ``decimals=2`` for shorter legend text). Default behavior is unchanged for back-compat. Closes Story 4 on the Crossplot side. ([#add-regression-equation-format](https://github.com/kkollsga/logsuite/issues/add-regression-equation-format))
- ``Crossplot.add_regression`` gained ``legend_loc=`` (matplotlib loc string or coords) and ``legend_decimals=`` (alias for ``decimals=``) arguments. ``legend_loc=`` overrides the auto-placement algorithm for the regression legend — pass ``"upper left"`` once and stop hand-rolling ``mlegend.Legend`` reconstructions. The latest ``legend_loc=`` value across multiple add_regression calls is used since the regression legend is rebuilt as a single block. ([#add-regression-legend-loc](https://github.com/kkollsga/logsuite/issues/add-regression-legend-loc))
- ``Crossplot.add_regression(where=...)`` now reuses ``Property.colors`` for the regression line when ``where`` is a single-key, single-value filter on a discrete property whose palette is set. Set the palette once on the manager, fit per-facies regressions, and the lines pick up the matching color automatically. Explicit ``line_color=`` still overrides; ``where`` filters spanning multiple values fall back to the existing ``"red"`` default. ([#add-regression-palette-reuse](https://github.com/kkollsga/logsuite/issues/add-regression-palette-reuse))
- Added ``Crossplot.add_regression_per(group_property, regression_type, ...)`` — fits one regression per unique value of a discrete grouping property and adds them all in a single call. Names default to the property labels; line colors come from ``Property.colors`` when set; all other ``add_regression`` kwargs (``equation_format``, ``legend_loc``, ``min_samples``, ...) are forwarded. Replaces the typical ``for`` loop that enumerates facies/zones by hand. ([#add-regression-per](https://github.com/kkollsga/logsuite/issues/add-regression-per))
- ``Crossplot.add_regression`` gained ``where=`` and ``min_samples=`` arguments. ``where`` accepts a dict mapping public column names (the same as ``x``, ``y``, ``color``, ``shape``, ``size`` on the Crossplot) to allowed values, or a callable returning a boolean mask over the prepared data. The fit uses only the matching rows; ``R²`` and equation reflect that subset. If the filtered subset has fewer than ``min_samples`` (default 5) points, a warning is emitted and the regression is skipped. Users no longer need to monkey-patch ``_data`` to fit per-zone or per-facies trends. ([#add-regression-where](https://github.com/kkollsga/logsuite/issues/add-regression-where))
- Added the ``Artifact`` protocol (``logsuite.Artifact``) — a base class for self-contained result objects that consumers (Crossplot, Table, ...) accept via ``.add(artifact)``. Subclasses implement ``_render_in_<consumer>(...)`` for the consumers they support; unsupported consumers raise ``TypeError``. ([#artifact-protocol](https://github.com/kkollsga/logsuite/issues/artifact-protocol))
- ``Crossplot`` now accepts a ``ManagerView`` or ``WellDataManager`` as its first argument, matching the architectural direction of consumers reading from the manager substrate. The recommended call shape becomes ``Crossplot(manager.filter(wells=[...]), x="PHIE", y="PERM")`` — single source of truth for which wells are included. Existing ``Crossplot(well_or_list, ...)`` calls continue to work unchanged. ([#crossplot-accepts-view](https://github.com/kkollsga/logsuite/issues/crossplot-accepts-view))
- Added ``Crossplot.add(artifact)`` — single composition entry point for any object implementing ``_render_in_crossplot(ax)``. The plot is rendered first if needed, then the artifact draws onto the axis and the legend is refreshed. Coexists with the existing kwargs/methods; the artifact protocol is the path forward for new visualization extensions. ([#crossplot-add](https://github.com/kkollsga/logsuite/issues/crossplot-add))
- Added ``Crossplot.column_for(public_name)`` — returns the prepared-data column name for a property bound to ``x``, ``y``, ``color``, ``shape``, or ``size`` on the crossplot, or ``None`` if no binding matches. Useful when writing a callable mask for ``add_regression(where=...)``: instead of hardcoding ``df["color_val"]``, ask the crossplot which column the property maps to. The internal logic (previously ``_resolve_where_column``) is now exposed as a stable public method. ([#crossplot-column-for](https://github.com/kkollsga/logsuite/issues/crossplot-column-for))
- ``Crossplot`` constructor now accepts ``equation_format=`` and ``decimals=`` as defaults that propagate to every ``add_regression`` call. Set ``Crossplot(equation_format="petrel", decimals=3)`` once and all regression legends render in Petrel calculator syntax with three-decimal precision. Per-call kwargs on ``add_regression`` still override. ([#crossplot-equation-format-default](https://github.com/kkollsga/logsuite/issues/crossplot-equation-format-default))
- Added ``Crossplot.add_table_panel(df, position="bottom"|"right", title=None, formatters=None, table_fraction=0.30)``. Attaches a DataFrame as a matplotlib-rendered table inside the same figure; the figure grows along the panel axis so the scatter is not squished. ``xplot.save("file.svg")`` then produces the combined deliverable in one file — the standard SCAL/DG3 shape.

  Defaults handle the common deliverable cases out of the box: NaN values render as ``"N/A"``, MultiIndex column labels flatten via ``" | "``, and MultiIndex row labels visually merge repeated outer levels (the outer cell is blanked when it matches the previous row, mimicking the merged-cell look without needing matplotlib cell merging). ``formatters={"PHIE": ".4f"}`` accepts callables or Python format specs per column.

  Implementation lives in a new module ``logsuite.visualization.table_panel`` whose functions are reusable by future composite/Template consumers. Closes Story 7 and friction #11. ([#crossplot-table-panel](https://github.com/kkollsga/logsuite/issues/crossplot-table-panel))
- Added ``flat_columns=`` argument to ``_ManagerPropertyProxy.stats(return_df=True)``. When ``True``, grouping columns use the actual filter property names (``Zone``, ``NTG``, ...) instead of the generic ``Group``/``Group1``/``Group2``. Default remains ``False`` for backward compatibility; this is expected to flip in a later major version. ([#flat-columns](https://github.com/kkollsga/logsuite/issues/flat-columns))
- ``WellDataManager`` and ``ManagerView`` now support dict-like access:
  ``manager["Well_A"]`` returns the Well by original name, ``"Well_A" in manager`` checks membership, ``len(manager)`` reports the well count, and iterating the manager yields Well objects in insertion order. Original names containing characters that aren't valid Python attributes (slashes, hyphens, spaces) work as keys without sanitisation: ``manager["12/3-2 B"]``. ([#manager-dict-access](https://github.com/kkollsga/logsuite/issues/manager-dict-access))
- Added ``WellDataManager.filter(wells=...)`` returning a ``ManagerView``. The view is a read-only subset of the manager: property proxies, statistics, and ``.data()`` calls operate only on the selected wells. Views are immutable; ``view.filter(wells=...)`` further narrows. ``ManagerView`` is now exported from ``logsuite``. ([#manager-view](https://github.com/kkollsga/logsuite/issues/manager-view))
- Extended ``WellDataManager.filter()`` and ``ManagerView.filter()`` with a ``where=`` argument for value-subset filtering. ``where={"Zone": "Reservoir"}`` post-filters ``.data()`` outputs to matching rows; the special key ``where={"well": [...]}`` selects wells. Stat methods (``mean``, ``stats``, ``sums_avg``, ...) raise ``NotImplementedError`` while value filters are active — use ``.data()`` and compute externally, or ``.filter("Zone")`` for grouped stats. ([#manager-view-where](https://github.com/kkollsga/logsuite/issues/manager-view-where))
- Added ``.data()`` method to ``_ManagerPropertyProxy`` and ``_ManagerMultiPropertyProxy``. Returns a long-format DataFrame across all wells with a leading ``well`` column and one column per active filter (named after the filter property). Replaces the manual loop-and-concat pattern users previously needed for pooled raw data, and gives Crossplot/regression artifacts a stable substrate going forward. ([#proxy-data](https://github.com/kkollsga/logsuite/issues/proxy-data))
- ``_ManagerPropertyProxy.data()`` gained ``warn_missing=True`` (default). Set ``warn_missing=False`` for batch / scripted workflows where wells lacking the property are expected and the per-call ``UserWarning`` is noise. Closes Story 6 — pooled data extraction is now fully covered. ([#proxy-data-warn-missing](https://github.com/kkollsga/logsuite/issues/proxy-data-warn-missing))
- Added ``weighted=`` argument to ``_ManagerPropertyProxy.data()`` and ``_ManagerMultiPropertyProxy.data()``. When ``True``, appends a ``Weight`` column with the per-row depth interval (computed via ``compute_intervals``, edge-corrected per well). Users can replicate ``stats()``-style depth-weighted percentiles externally with e.g. ``np.average(df["PHIE"], weights=df["Weight"])``. ([#proxy-data-weighted](https://github.com/kkollsga/logsuite/issues/proxy-data-weighted))
- Added ``.fit(model, ...)`` on multi-property proxies. Pools data across the (sub-)view, fits the regression with the first property as ``x`` and the second as ``y``, and returns a ``RegressionFit``. Works through ``manager.filter(where=...)`` so users can fit on subsets in one expression: ``manager.filter(where={"Zone": "Reservoir"}).properties(["PHIE", "PERM"]).fit(ExponentialRegression(), name="Reservoir")``. ([#proxy-fit](https://github.com/kkollsga/logsuite/issues/proxy-fit))
- Added ``RegressionFit`` (``logsuite.RegressionFit``) — a renderable artifact wrapping a fitted regression model. Provides ``equation(format="natural"|"log10"|"petrel", decimals=4)`` for Petrel-calculator-syntax exports of exponential fits, configurable decimal precision in legend labels via ``label(decimals=)``, and ``_render_in_crossplot`` / ``_render_in_table`` dispatch. ``ExponentialRegression`` natively supports all three formats; other models fall back to natural form for ``log10``/``petrel``. ([#regression-fit](https://github.com/kkollsga/logsuite/issues/regression-fit))
- Added ``logsuite.set_quiet(True)`` to silence library-emitted informational status messages (``✓ Set colors for …``, ``✓ Loaded N properties …``). Defaults stay verbose so existing notebooks see the same output, but scripted / batch users can call ``set_quiet(True)`` once at startup for clean stdout. Warnings and errors are unaffected. Implemented via a tiny ``emit_status`` helper in ``logsuite.utils``; the broadcast prints in ``WellDataManager.load_properties`` and the property-proxy setters route through it. ([#set-quiet](https://github.com/kkollsga/logsuite/issues/set-quiet))
- Added ``visualization/style.py`` with ``resolve_discrete_palette`` — a single resolver for categorical colors that honors ``Property.colors`` first and falls back to a default palette only for codes the user has not assigned. Crossplot now consumes this resolver in both single-group and multi-group plotting paths, matching WellView's existing behavior. Setting ``manager.<discrete_property>.colors = {...}`` now propagates to every visualization consistently. ([#style-resolver](https://github.com/kkollsga/logsuite/issues/style-resolver))

### Bug Fixes

- Fixed ``TypeError: 'NoneType' object is not callable`` when calling ``Crossplot.regression()``. The constructor parameter ``regression=`` was being assigned to ``self.regression``, shadowing the method of the same name. The constructor now stores it on ``self._initial_regression`` internally; the public ``regression=`` kwarg name is unchanged. ([#regression-method-shadow-fix](https://github.com/kkollsga/logsuite/issues/regression-method-shadow-fix))

### Deprecations

- Passing ``regression``, ``regression_by_color``, ``regression_by_group``, ``regression_by_color_and_shape``, or ``regression_by_shape_and_color`` to the ``Crossplot`` constructor is now deprecated. They will be removed in a future minor release. Use ``crossplot.add_regression(kind, where=...)`` for subsetted fits, or build a ``RegressionFit`` and call ``crossplot.add(fit)`` for full control over the artifact. ([#deprecate-regression-kwargs](https://github.com/kkollsga/logsuite/issues/deprecate-regression-kwargs))
- ``Well.Crossplot()``, ``Well.WellView()``, and ``WellDataManager.Crossplot()`` now emit ``DeprecationWarning`` when called. They violate the layered-dependency rule (lower layer importing from higher) and will be removed in a future minor release. Construct visualizations directly instead: ``Crossplot(manager.filter(wells=[...]), x=..., y=...)`` or ``WellView(well, ...)``. Existing call sites continue to function during the deprecation window. ([#deprecate-upward-constructors](https://github.com/kkollsga/logsuite/issues/deprecate-upward-constructors))

### Documentation

- Added ``ARCHITECTURE.md`` documenting the layered-dependency constitution that governs every module. Pull requests that violate the layer rule (lower layer importing from higher) should be rejected. ([#architecture-doc](https://github.com/kkollsga/logsuite/issues/architecture-doc))
- Added Article VII (Compartmentalization) to the architectural constitution: code is organized into focused modules and classes, with soft size triggers (~600-line modules, ~300-line classes, ~50-line methods) flagging refactor candidates. Several existing files exceed these limits and are scheduled for splitting in M3+. ([#article-vii](https://github.com/kkollsga/logsuite/issues/article-vii))
- Documentation overhaul: README and Sphinx docs rewritten around use cases.

  * New ``docs/how-to/`` section with six task-focused guides — fitting per-group regressions, setting a discrete-property palette, pooling raw data across wells, exporting Petrel-syntax equations, building a SCAL/DG3 deliverable, and displaying single-well log plots.
  * ``docs/quickstart.md`` rewritten to a 7-line working deliverable using the post-M3 patterns (Manager substrate, ``add_regression_per``, ``add_table_panel``).
  * ``docs/index.md`` adds a "How-to guides" toctree section above the topical user guide.
  * ``README.md`` rewritten to lead with what the library lets you stop doing (no per-well loops, no monkey-patching, no matplotlib glue, no equation rewriting), with cross-references to the how-to pages and ``story_tests/`` scripts.
  * User-guide pages ``loading-data``, ``multi-well``, ``visualization``, ``regression`` updated to use canonical post-M3 patterns and corrected two real bugs: ``manager.add_dataframe`` (which doesn't exist; it's ``manager.load_properties``) and ``manager.well_names`` (which is ``manager.wells``).

  ([#docs-overhaul](https://github.com/kkollsga/logsuite/issues/docs-overhaul))


## [0.2.0] - 2026-03-05

### Breaking Changes

- **Package renamed to logSuite**: PyPI package is now `logsuite` (was `pylog`).
  Import changes: `from logsuite import WellDataManager, Well, Property`.
  Install: `pip install logsuite`.
  All internal references, docs, CI/CD, and tests updated.

## [0.1.159] - 2026-03-05

### Breaking Changes

- **Package renamed to pyLog**: PyPI package name changed from `well-log-toolkit` to `pylog`.

## [0.1.158] - 2026-03-05

### Documentation

- **Sphinx documentation**: Full docs site with furo theme, MyST markdown,
  and autodoc API reference. Includes installation, quickstart, user guide
  (loading data, wells/properties, statistics, visualization, regression,
  multi-well), and API reference for all public classes and functions.
- **ReadTheDocs configuration**: `.readthedocs.yaml` for automated builds.
- **Documentation dependencies**: Added `[docs]` optional dependency group.
- **Repo rename**: Updated all project URLs from `logsuite` to
  `logsuite` to match PyPI package name.

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
  (`from logsuite import PolynomialExponentialRegression`).
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
  modules (e.g., `from logsuite.statistics import mean`) must now use
  subpackage paths (e.g., `from logsuite.analysis.statistics import mean`).
  Top-level imports (`from logsuite import Well, Property`) are unchanged.

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
