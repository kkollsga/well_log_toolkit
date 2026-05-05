# logsuite — 10-Step Improvement Plan

## Current State

| Metric | Value |
|--------|-------|
| Source lines | ~18,380 across 21 modules (restructured from 11) |
| Classes | 28 (7 public, 21 internal/regression) |
| Public methods | 320 |
| Tests | 172 passing in `tests/` (CI runs them as of v0.1.153) |
| PyPI version | 0.1.155 |
| Largest files | property.py (2,718), well.py (2,697), data_manager.py (1,553) |

---

## Step 1: Fix CI Pipeline

**Goal:** CI actually runs the test suite instead of just a smoke import.

**Current problem:**
- `build-and-publish.yml` test job only does `python -c "from logsuite import ..."`
- `pyproject.toml` has `testpaths = ["tests"]` but the directory is `pytest/`
- `addopts` includes `--cov=logsuite` but `pytest-cov` isn't installed in CI

**Tasks:**
- [x] Rename `pytest/` → `tests/`
- [x] Update `pyproject.toml` addopts to remove `--cov` requirement
- [x] Update CI test job to: `pip install -e ".[dev]" && python -m pytest`
- [x] Drop Python 3.9 support; minimum is now 3.10
- [ ] Add a separate lint job (ruff + mypy) — deferred to future PR

**Status: DONE (v0.1.153)**

**Files touched:** `.github/workflows/build-and-publish.yml`, `pyproject.toml`

---

## Step 2: Restructure into Proper Package Layout

**Goal:** Transform the flat module layout into a compartmentalized package structure with clear domain boundaries.

**Current structure (flat):**
```
logsuite/
├── __init__.py          (139 lines)
├── exceptions.py        (48 lines)
├── las_file.py          (1,343 lines)
├── manager.py           (4,145 lines)  ← 4 classes crammed together
├── operations.py        (494 lines)
├── property.py          (2,718 lines)
├── regression.py        (938 lines)
├── statistics.py        (590 lines)
├── utils.py             (225 lines)
├── visualization.py     (5,050 lines)  ← 3 classes crammed together
└── well.py              (2,697 lines)
```

**Target structure (domain-driven):**
```
logsuite/
├── __init__.py                     # Public API re-exports (keep stable)
├── _version.py                     # Version detection logic
├── exceptions.py                   # Exception hierarchy (unchanged)
│
├── io/                             # I/O layer
│   ├── __init__.py                 # Re-exports LasFile
│   ├── las2.py                     # LAS 2.0 reader (current las_file.py)
│   ├── las3.py                     # LAS 3.0 reader (future, Step 6)
│   ├── las_common.py               # Shared LAS parsing utilities
│   └── export.py                   # LAS export logic (extracted from las_file.py)
│
├── core/                           # Core domain objects
│   ├── __init__.py                 # Re-exports Well, Property
│   ├── well.py                     # Well class + SourceView
│   ├── property.py                 # Property class
│   └── operations.py               # PropertyOperationsMixin
│
├── analysis/                       # Statistical & analytical layer
│   ├── __init__.py                 # Re-exports statistics, SumsAvgResult
│   ├── statistics.py               # Depth-weighted stat functions
│   ├── sums_avg.py                 # SumsAvgResult class (from manager.py:51-627)
│   └── regression/                 # Regression subpackage
│       ├── __init__.py             # Re-exports all regression types
│       ├── base.py                 # RegressionBase ABC
│       ├── linear.py               # LinearRegression
│       ├── polynomial.py           # PolynomialRegression
│       ├── exponential.py          # ExponentialRegression, PowerRegression
│       ├── logarithmic.py          # LogarithmicRegression
│       └── poly_exponential.py     # PolynomialExponentialRegression
│
├── manager/                        # Multi-well orchestration
│   ├── __init__.py                 # Re-exports WellDataManager
│   ├── core.py                     # WellDataManager (from manager.py:2592-4145)
│   └── proxy.py                    # _ManagerPropertyProxy, _ManagerMultiPropertyProxy
│
├── visualization/                  # Presentation layer
│   ├── __init__.py                 # Re-exports Template, WellView, Crossplot
│   ├── template.py                 # Template class (viz.py:217-826)
│   ├── wellview.py                 # WellView class (viz.py:827-2868)
│   ├── crossplot.py                # Crossplot class (viz.py:2869-5050)
│   └── _helpers.py                 # _create_regression(), DEFAULT_COLORS, shared utils
│
└── utils.py                        # String sanitization, general helpers
```

**Migration rules:**
1. Every existing `from logsuite import X` must keep working — the top-level `__init__.py` re-exports everything
2. Each subpackage `__init__.py` re-exports its public API
3. Internal cross-references use relative imports within subpackages
4. No circular imports — the dependency graph flows: `io → core → analysis → manager → visualization`

**Tasks:**
- [x] Extract version logic from `__init__.py` → `_version.py`
- [x] Create `io/` subpackage, move `las_file.py` → `io/las_file.py`
- [x] Create `visualization/` subpackage, split 5050-line `visualization.py` into `template.py`, `wellview.py`, `crossplot.py`
- [x] Create `analysis/` subpackage with `statistics.py`, `regression.py`, `sums_avg.py` (extracted from manager.py)
- [x] Create `core/` subpackage, move `well.py`, `property.py`, `operations.py`
- [x] Create `manager/` subpackage, split 3500-line `manager.py` into `data_manager.py` and `proxy.py`
- [x] Write all subpackage `__init__.py` files with proper re-exports
- [x] Update top-level `__init__.py` to import from new subpackages
- [x] Update all internal imports across all files
- [x] Update all test imports to match new package structure
- [x] Run full test suite — 152 tests pass, zero breakage
- [ ] Further split `regression.py` into one file per class — deferred (low priority)
- [ ] Extract LAS export logic → `io/export.py` — deferred (low priority)

**Status: DONE (v0.1.154)**

**Files touched:** Everything. This was the largest single step.

---

## Step 3: Harden the Test Suite

**Goal:** Proper test infrastructure with fixtures, edge-case coverage, and CI integration.

**Current problems:**
- Tests in `pytest/` (wrong directory, CI doesn't find them)
- Several tests use `return True` instead of `assert` (pytest warns)
- No `conftest.py` with shared fixtures
- No coverage of `manager.py` proxy operations or `well.py` computed properties
- Tests create real LAS files with hardcoded paths

**Tasks:**
- [ ] Move `pytest/` → `tests/` (done in Step 1)
- [ ] Fix all `return` → `assert` warnings (test_source_aware.py and others)
- [ ] Create `tests/conftest.py` with:
  - `tmp_path`-based LAS file fixtures (no hardcoded paths)
  - Pre-built `Well` and `WellDataManager` instances
  - Sample Property objects with known values
- [ ] Add unit tests for untested areas:
  - `Well.__setattr__` computed property creation
  - `_ManagerPropertyProxy` broadcast arithmetic
  - `Property.filter()` chaining (2+ levels)
  - `Property.resample()` correctness
  - `LasFile.export_las()` round-trip fidelity
  - `SumsAvgResult.report()` aggregation modes
- [ ] Add edge-case tests for statistics:
  - Empty arrays, single point, all-NaN values
  - Negative depths, unsorted depths (should error or handle)
  - Zero-weight intervals
  - Percentile at 0 and 100
- [ ] Add integration test: load LAS → filter → compute stats → export �� reload → compare
- [ ] Target: 80%+ line coverage on `statistics.py`, `property.py`, `well.py`, `operations.py`
- [ ] Separate tests into unit/ and integration/ subdirectories

**Status: DONE (v0.1.155) — 172 tests passing, 0 PytestReturnNotNoneWarning**

---

## Step 4: Add Changelog System

**Goal:** Automated, standards-compliant changelog generation tied to releases.

**Approach:** [Keep a Changelog](https://keepachangelog.com/) format + towncrier for fragment-based automation.

**Tasks:**
- [x] Install and configure `towncrier` in pyproject.toml
- [x] Create `changes/` directory with `README.md` contributor instructions
- [x] Create initial `CHANGELOG.md` with retrospective v0.1.152 summary and v0.1.153 entry
- [x] Add `towncrier` to dev dependencies
- [ ] Add towncrier fragment creation to PR workflow — deferred
- [ ] Add `towncrier build --yes` to release CI workflow — deferred
- [ ] Consolidate useful information from `dev_docs/` into the retrospective CHANGELOG
- [ ] Remove or archive `dev_docs/` (session notes, not user-facing)

**Status: DONE (core system in place, v0.1.153)**

**Files touched:** `pyproject.toml`, `CHANGELOG.md`, `changes/`, `.github/workflows/`

---

## Step 5: Rewrite All Docstrings for Sphinx Autodoc

**Goal:** Every public class, method, and function has NumPy-style docstrings that render correctly in Sphinx with the napoleon extension.

**Docstring standard:** NumPy style (matches scientific Python ecosystem: numpy, pandas, scipy, xarray).

**Template for classes:**
```python
class Property:
    """Single well log property with filtering and statistical operations.

    A Property represents one log curve (e.g., PHIE, GR, SW) with its
    depth array and values. Properties support chained hierarchical
    filtering, depth-weighted statistics, and numpy-style arithmetic.

    Parameters
    ----------
    name : str
        Property mnemonic (e.g., ``'PHIE'``, ``'GR'``).
    depth : np.ndarray
        Monotonically increasing depth values.
    values : np.ndarray
        Log values corresponding to each depth point.
    well : Well
        Parent well that owns this property.

    Attributes
    ----------
    source : str
        Name of the LAS file this property was loaded from.
    is_filtered : bool
        Whether any filters are currently applied.

    See Also
    --------
    Well : Container for multiple properties from one wellbore.
    PropertyOperationsMixin : Arithmetic and comparison operators.

    Examples
    --------
    Access a property from a well and compute statistics:

    >>> phie = well.get_property('PHIE')
    >>> phie.mean(weighted=True)
    0.182

    Chain filters and get summaries:

    >>> stats = phie.filter('Zone').filter('NTG_Flag').sums_avg()

    Create computed properties with operators:

    >>> well.HC_Volume = well.PHIE * (1 - well.SW)
    """
```

**Template for methods:**
```python
def filter(self, property_name: str) -> 'Property':
    """Apply a hierarchical filter using a discrete property.

    Creates a filtered view grouped by the unique values of the
    filter property. Multiple filters can be chained to create
    nested groupings.

    Parameters
    ----------
    property_name : str
        Name of a discrete property on the same well.

    Returns
    -------
    Property
        Filtered property (new object; original is unchanged).

    Raises
    ------
    PropertyNotFoundError
        If ``property_name`` does not exist on the parent well.
    PropertyTypeError
        If the filter property is not discrete.

    See Also
    --------
    filter_intervals : Filter by depth range instead of property.
    sums_avg : Compute statistics on filtered results.

    Examples
    --------
    Single filter:

    >>> filtered = well.PHIE.filter('Zone')
    >>> filtered.sums_avg()
    {'Reservoir': {'mean': 0.182, ...}, 'NonReservoir': {'mean': 0.05, ...}}

    Chained filters:

    >>> deep = well.PHIE.filter('Zone').filter('Facies')
    """
```

**Tasks:**
- [ ] Define the docstring standard in a `CONTRIBUTING.md` or `docs/docstring_guide.md`
- [ ] Rewrite docstrings for all public classes (7 main classes):
  - `WellDataManager`, `Well`, `Property`, `LasFile`
  - `Template`, `WellView`, `Crossplot`
- [ ] Rewrite docstrings for all public methods (~150 public methods across main classes)
- [ ] Rewrite docstrings for all statistical functions (10 functions in statistics.py)
- [ ] Rewrite docstrings for all regression classes (7 classes)
- [ ] Add module-level docstrings to every `__init__.py` in every subpackage
- [ ] Add cross-references using `See Also` sections
- [ ] Add `Examples` sections with doctestable code where practical
- [ ] Verify all docstrings render correctly with `sphinx-build` (Step 6)
- [ ] Add `Raises` sections documenting which exceptions each method can throw
- [ ] Add `Notes` sections for mathematical explanations (statistics, regression)

**Status: DONE (v0.1.155) — All public methods already had NumPy-style docstrings. Added See Also cross-references to key methods.**

---

## Step 6: Set Up Sphinx + ReadTheDocs

**Goal:** Auto-generated API documentation hosted on ReadTheDocs, built from docstrings.

**Tasks:**
- [ ] Create `docs/` directory structure:
  ```
  docs/
  ├── conf.py                  # Sphinx configuration
  ├── index.rst                # Landing page
  ├── requirements.txt         # Docs build dependencies
  ├── getting_started/
  │   ├── index.rst
  │   ├── installation.rst
  │   ├── quickstart.rst
  │   └── concepts.rst         # Core concepts (Well, Property, filtering)
  ├── user_guide/
  │   ├── index.rst
  │   ├── loading_data.rst     # LAS loading, DataFrame loading
  │   ├── filtering.rst        # Hierarchical filtering, depth intervals
  │   ├── statistics.rst       # Depth-weighted stats, sums_avg
  │   ├── visualization.rst    # Templates, WellView, Crossplot
  │   ├── regression.rst       # Regression types & usage
  │   ├── multi_well.rst       # WellDataManager, broadcasting
  │   └── export.rst           # LAS export, project save/load
  ├── api/
  │   ├── index.rst
  │   ├── core.rst             # Well, Property, operations
  │   ├── io.rst               # LasFile
  │   ├── analysis.rst         # Statistics, SumsAvgResult
  │   ├── regression.rst       # Regression classes
  │   ├── manager.rst          # WellDataManager
  │   ├── visualization.rst    # Template, WellView, Crossplot
  │   └── exceptions.rst       # Exception hierarchy
  ├── cookbook/
  │   ├── index.rst
  │   └── recipes.rst          # Common patterns (from README)
  └── changelog.rst            # Includes CHANGELOG.md
  ```

- [ ] Configure `docs/conf.py`:
  ```python
  project = 'logsuite'
  extensions = [
      'sphinx.ext.autodoc',
      'sphinx.ext.napoleon',       # NumPy-style docstrings
      'sphinx.ext.intersphinx',    # Link to numpy/pandas docs
      'sphinx.ext.viewcode',       # Source code links
      'sphinx.ext.autosummary',    # Auto-generate summary tables
      'sphinx_copybutton',         # Copy button on code blocks
  ]
  napoleon_google_docstring = False
  napoleon_numpy_docstring = True
  napoleon_use_rtype = False
  autodoc_member_order = 'bysource'
  autodoc_typehints = 'description'
  intersphinx_mapping = {
      'python': ('https://docs.python.org/3', None),
      'numpy': ('https://numpy.org/doc/stable/', None),
      'pandas': ('https://pandas.pydata.org/docs/', None),
      'matplotlib': ('https://matplotlib.org/stable/', None),
  }
  html_theme = 'furo'
  ```

- [ ] Add docs dependencies to `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  docs = [
      "sphinx>=7.0",
      "furo",
      "sphinx-copybutton",
      "sphinx-autodoc-typehints",
  ]
  ```

- [ ] Create `.readthedocs.yaml`:
  ```yaml
  version: 2
  build:
    os: ubuntu-22.04
    tools:
      python: "3.11"
  sphinx:
    configuration: docs/conf.py
  python:
    install:
      - method: pip
        path: .
        extra_requirements:
          - docs
  ```

- [ ] Write narrative documentation pages (getting_started, user_guide)
- [ ] Write API reference pages using `.. automodule::` and `.. autoclass::`
- [ ] Migrate cookbook/recipes content from README
- [ ] Add `docs` build to CI: `sphinx-build docs docs/_build -W` (fail on warnings)
- [ ] Register project on readthedocs.org
- [ ] Add ReadTheDocs badge to README

**Files touched:** `docs/` (new), `pyproject.toml`, `.readthedocs.yaml`, `README.md`

---

## Step 7: Fix README and Project Metadata

**Goal:** Professional first impression. Accurate metadata. Concise README that points to full docs.

**Current problems:**
- Issues URL: `github.com/yourusername/logsuite/issues`
- Classifier: `Development Status :: 3 - Alpha` at 152 releases
- Description undersells: "Fast LAS file processing" (it's a full analytics library)
- README is 2,367 lines (should be ~300 with links to docs site)

**Tasks:**
- [x] Fix the `yourusername` → `kkollsga` in README.md
- [x] Update pyproject.toml classifier to `Development Status :: 4 - Beta`
- [x] Rewrite description to reflect full petrophysical analysis capabilities
- [x] Add badges to README: PyPI version, Python 3.10+, CI status, license
- [x] Added `CLAUDE.md` with project conventions for Claude Code
- [ ] Trim README to: overview, install, 1-minute example, feature highlights, link to docs — deferred to Step 6
- [ ] Move all detailed content to docs site (Step 6)
- [ ] Archive `dev_docs/` into a `dev_docs/archive/` or remove entirely

**Status: DONE (core fixes in place, v0.1.153). README trim deferred until docs site exists.**

**Files touched:** `README.md`, `pyproject.toml`, `CLAUDE.md`

---

## Step 8: Improve Error Messages and Input Validation

**Goal:** Every error a user can hit has a clear message with actionable guidance.

**Tasks:**
- [ ] Audit every `__getattr__` override (Well, WellDataManager, proxies):
  - Suggest similar property names on `PropertyNotFoundError` (fuzzy matching)
  - Distinguish "property doesn't exist" from "typo in well name" on manager
- [ ] Add validation to `Property.filter()`:
  - If filter property doesn't exist, list available discrete properties
  - If filter property exists but isn't discrete, suggest setting `.type = 'discrete'`
- [ ] Add `WellDataManager.validate()`:
  - Report which wells are missing which properties
  - Report depth grid inconsistencies across wells
- [ ] Improve `DepthAlignmentError` to show actual depth grids (start, stop, step, count)
- [ ] Add `warnings.warn()` (not errors) when manager operations silently skip wells
- [ ] Add input validation at system boundaries:
  - `load_las()`: check file extension, file size, encoding
  - `Property` constructor: validate depth is monotonic, values length matches depth
  - `Template.add_track()`: validate scale parameters

**Status: DONE (v0.1.155) — Fuzzy matching, validate(), proxy warnings, input validation all implemented and tested.**

---

## Step 9: Performance Optimization

**Goal:** Handle field-scale datasets (100+ wells, 10K+ samples per well).

**Tasks:**
- [ ] Vectorize `compute_zone_intervals()` — replace Python for-loop with numpy broadcasting
- [ ] Profile `LasFile._parse_data()` — benchmark `pd.read_csv(skiprows=...)` vs current line-by-line parser
- [ ] Add `__slots__` to `Property` class to reduce per-instance memory
- [ ] Cache `compute_intervals()` results on Property (depth grid is immutable after creation)
- [ ] Lazy evaluation for `_ManagerPropertyProxy`: defer computation until terminal operation
- [ ] Add benchmarks in `tests/benchmarks/` using pytest-benchmark:
  - LAS loading (1MB, 10MB, 100MB files)
  - Statistics computation across well count (10, 50, 100 wells)
  - Filter chain depth (1, 3, 5 levels)

**Files touched:** `analysis/statistics.py`, `io/las2.py`, `core/property.py`, `manager/proxy.py`

---

## Step 10: LAS 3.0 Support + Extension Points

**Goal:** Support modern LAS format and allow community extensions.

**LAS 3.0 tasks:**
- [ ] Create `io/las3.py` with section-tagged, tab-delimited parser
- [ ] Handle multiple data sections (log, core, drilling)
- [ ] Map LAS 3.0 metadata to existing internal structures
- [ ] Auto-detect version in a factory: `LasFile.open(path)` returns `Las2File` or `Las3File`
- [ ] Add `version` property to all LasFile implementations

**Extension system tasks:**
- [ ] Add `geometric_mean()` and `harmonic_mean()` to statistics module
- [ ] Add `Property.apply(func, weighted=True)` for user-defined statistics
- [ ] Add `Property.histogram()` returning bin edges and weighted counts
- [ ] Add a registration pattern for custom statistics in `sums_avg()`:
  ```python
  from logsuite import register_statistic

  @register_statistic
  def dykstra_parsons(values, weights):
      """Dykstra-Parsons coefficient for permeability heterogeneity."""
      ...
  ```
- [ ] Add built-in registered statistics: Dykstra-Parsons, Lorenz coefficient

**Files touched:** `io/las3.py` (new), `io/__init__.py`, `analysis/statistics.py`, `core/property.py`

---

## Execution Order & Dependencies

```
Step 1 (Fix CI) ─────────────────┐
                                  ├──→ Step 3 (Harden Tests) ──→ Step 9 (Performance)
Step 7 (Fix README) ─────────────┤
                                  ├──→ Step 2 (Restructure) ──→ Step 5 (Docstrings) ──→ Step 6 (ReadTheDocs)
Step 4 (Changelog) ──────────────┘
                                       Step 8 (Error Messages) ← can start after Step 2
                                       Step 10 (LAS 3.0 + Extensions) ← can start after Step 3
```

**Phase 1 (foundation, parallel):** Steps 1, 4, 7
**Phase 2 (structure):** Step 2
**Phase 3 (quality, parallel):** Steps 3, 5, 8
**Phase 4 (publication):** Step 6
**Phase 5 (capability):** Steps 9, 10

---

## Estimated Scope per Step

| Step | New/Modified Files | Risk | Can Break API? |
|------|-------------------|------|----------------|
| 1. Fix CI | 2 | Low | No |
| 2. Restructure | 30+ | High | No (re-exports preserve API) |
| 3. Harden Tests | 15+ | Low | No |
| 4. Changelog | 5 | Low | No |
| 5. Docstrings | 20+ | Low | No |
| 6. ReadTheDocs | 20+ | Low | No |
| 7. Fix README | 2 | Low | No |
| 8. Error Messages | 6 | Medium | No (additive) |
| 9. Performance | 5 | Medium | No (same behavior, faster) |
| 10. LAS 3.0 + Extensions | 8+ | High | No (additive) |

---

## Future: Analysis Gaps

- [ ] Cutoff/flag operations — create net pay flags from cutoffs (PHIE > 0.08 AND SW < 0.5) as a first-class workflow
- [ ] Volumetrics — STOIIP/GIIP calculations (building blocks exist: thickness, porosity, saturation)
- [ ] Log normalization — histogram equalization or shifting between wells
- [ ] Missing data interpolation — gap-filling methods beyond resample()
- [ ] Variance/covariance — raw variance, covariance matrix, correlation coefficients
- [ ] Confidence intervals on regressions — prediction bands, standard errors
- [ ] Residual access — inspect regression residuals for diagnostics

## Future: Ecosystem

- [ ] DLIS/LIS support — common in modern workflows, currently LAS-only
- [ ] CSV/Excel direct import — reduce friction for non-LAS data
- [ ] Wrapped LAS support — currently WRAP=NO required
- [ ] Well header editing — read-only today, no API to modify/export well metadata
- [ ] Pandas integration beyond .data() — .to_dataframe() on Manager, DataFrame indexing on wells
- [ ] Cloud/database backends — S3, database support (currently file-based only)
- [ ] Plugin/extension system — custom property types, statistics, track renderers
- [ ] CLI tool — `logsuite info`, `logsuite convert`, etc.
- [ ] Interactive visualization — ipywidgets, Plotly, or Bokeh for exploration

---

# Active Roadmap (post-constitution)

The roadmap below follows the layered-dependency constitution in `ARCHITECTURE.md`. Milestones are ordered to land additive changes first (M1, M2) and breaking changes last (M3) so users can migrate over a deprecation window.

## M1 — Manager substrate (additive)

- [x] **M1.1** `_ManagerPropertyProxy.data()` and `_ManagerMultiPropertyProxy.data()` — long-format DataFrame across all wells with property-named filter columns. (Closes friction #4, #5, #8.)
- [x] **M1.2** `WellDataManager.filter(wells=...) → ManagerView` — read-only subset view; property proxies and statistics scoped to the subset.
- [x] **M1.3** `weighted=` / `Weight` column on `proxy.data()` so users can replicate depth-weighted percentiles externally. (Story 3.)
- [x] **M1.4** `flat_columns=True` on `stats(return_df=True)` to use property names instead of `Group1/Group2`. Default flips in v0.2. (Story 3.)
- [x] **M1.5** `WellDataManager.filter(where={...})` — value-subset filtering at manager level via `_ValueFilteringProxy` post-filter on `.data()`. Stat methods (mean/stats/...) raise `NotImplementedError` while value filters are active.

## M2 — Artifact protocol (additive)

- [x] `core/artifact.py` — `Artifact` protocol/base. `_render_in_<consumer>(...)` convention. (M3 will split into a regression/ subpackage when crossplot.py is touched.)
- [x] `analysis/regression_fit.py` — `RegressionFit` artifact wrapping a fitted `RegressionBase`.
  - [x] `equation(format="natural"|"log10"|"petrel", decimals=4)` (Story 4)
  - [x] `label(decimals=4)` (friction #7)
  - [x] `_render_in_crossplot(ax)`, `_render_in_table(tbl)`
- [x] `proxy.fit(regression_model) → RegressionFit` on `_ManagerMultiPropertyProxy` and on `_ValueFilteringProxy` (works through `where=` filters).
- [x] `Crossplot.add(artifact)` — coexists with existing kwargs.

## M3 — Constitutional cleanup (breaking, with deprecation shims)

### Additive pieces (landed)

- [x] **M3.1** `visualization/style.py` — `resolve_discrete_palette` resolver. Honors user palette first, falls back to default in encounter order. (Story 2)
- [x] **M3.2** Crossplot honors `Property.colors` via `_store_discrete_colors` + `_resolve_categorical_palette`. Both single-group and multi-group plot paths route through the resolver.
- [x] **M3.3** `Crossplot.add_regression(where=, min_samples=)` — `where=` accepts a dict (with public column-name aliasing) or a callable mask; `min_samples` warns and skips when the subset is too small. (Story 1, partial — `decimals=`/`equation_format=` propagation deferred to a follow-up.)
- [x] **M3.5** `regression=` constructor parameter no longer shadows the `.regression()` method (renamed to `self._initial_regression` internally; the public kwarg name is unchanged for now).

### Deprecation/removal passes (deferred)

- [x] **M3.4** Deprecation warning fires when any of the five `regression*` constructor kwargs are passed. Users pointed to `add_regression(where=...)` and `add(fit)`.
- [x] **M3.6** `Crossplot` constructor accepts `ManagerView` and `WellDataManager` directly (alongside `Well`/`list[Well]`). The `wells=` parameter name is unchanged — deferred as cosmetic; revisit if friction emerges.
- [~] **M3.6b** Skipped — `wells=` parameter still accepts manager/view/well/list and the name is clear enough that renaming would create user-facing churn for marginal benefit.
- [x] **M3.7a** Deprecation shims on `Well.Crossplot()`, `Well.WellView()`, `WellDataManager.Crossplot()`. Each emits a `DeprecationWarning` and forwards to the proper constructor.
- [ ] **M3.7b** Delete the three deprecated methods in a future minor release (after one deprecation window).
- [x] **M3.8** `add_regression(decimals=, equation_format=)` propagation — when set, `_format_regression_label` routes through `RegressionFit` so the legend matches `equation_format="petrel"` etc. Default unchanged for back-compat.
- [ ] One-minor-version deprecation window before any deletions.

## M4 — Polish & second-tier artifacts

- [x] **M4.1** `add_regression(where=...)` reuses `Property.colors` for the line color when `where` matches a single discrete value whose palette is set. The Story 1 + Story 2 bleed-through is closed.
- [x] **M4.2** `Crossplot.add_regression_per(group_property, kind, ...)` — convenience wrapper that calls `add_regression(where={group_property: [v]})` for each unique value of `group_property`. Forwards all other regression kwargs.
- [x] **M4.3** `logsuite.set_quiet(True)` silences informational broadcast prints (`✓ Set colors for …`, `✓ Loaded N properties …`). Implemented via a small `emit_status` helper in `utils.py`; default verbose to preserve current behaviour. Story scripts call `set_quiet(True)` from `synthetic_data.build_manager` for clean output.
- [x] **M4.8** `Crossplot.column_for(public_name)` — public introspection helper that returns the prepared-data column name for a given property name. Lets users write `where=callable` masks without hardcoding internal names like `color_val`.
- [x] **M4.4** `manager.PROP.data(warn_missing=False)` controls the skipped-wells warning explicitly. (Story 6 closed.)
- [x] **M4.5** `Crossplot(equation_format="petrel", decimals=N)` constructor-level defaults that propagate to every `add_regression` call. Explicit kwargs on `add_regression` still override. (Story 7.)
- [x] **M4.6** `add_regression(legend_loc=..., legend_decimals=...)` — `legend_loc=` overrides the auto-placement algorithm for the regression legend; `legend_decimals=` is an alias for `decimals=` and wins when both are passed. Last `legend_loc=` across multiple add_regression calls is the one used (the regression legend is rebuilt as one block). (Story 7.)
- [x] **M4.7** `Crossplot.add_table_panel(df, position="bottom"|"right", title=None, formatters=None)` — attaches a matplotlib-rendered DataFrame as a sibling axes inside the same figure; figure grows along the panel axis so the scatter is not squished. NaN→`"N/A"`, MultiIndex columns flatten via `" | "`, MultiIndex rows visually merge repeated outer levels. New module `visualization/table_panel.py` houses the rendering helpers. Closes Story 7 and friction #11.
- [ ] `Property.filter(boundary_buffer=N)` — drop samples within N units of facies transitions. (Friction #9.)
- [ ] Stats methods documentation page. (Friction #10.)
- [ ] Second-tier artifacts as use cases drive them: `Trend`, `BoundaryMarker`, `StatsBundle`.

---

# User Stories (durable backlog)

## Story 1 — Fit a regression on a subset of points without monkey-patching

As a reservoir geologist building poroperm transforms for DG3, I want to add regression lines that fit only on a subset of the crossplot data (e.g. one zone group, one facies, one well), so that I can produce per-unit transforms in a single figure without manipulating private attributes.

**Acceptance:**
- `Crossplot.add_regression(...)` accepts `where=` (dict of `{column: allowed_values}` or callable returning a boolean mask).
- The fit uses only matching rows; R² and equation reflect that subset.
- The line is clipped to `x_range` (existing behaviour) and labelled with the user-supplied name.
- If the filtered subset has fewer than `min_samples` (default 5), warns clearly and skips the fit instead of erroring.
- Public method, public column names — no need to know about `shape_val`/`color_val` internals; column aliases like `zone`, `facies`, `well` resolve to the underlying property name.

**Maps to:** M2 (artifact + RegressionFit), M3 (`add_regression` sugar layer + alias resolution).

## Story 2 — Crossplot honours `Property.colors` and `Property.labels` consistently with WellView

As a user who has set discrete-property colors and labels once on the manager (e.g. `manager.Facies_2025_NonNet.colors = {...}`), I want every visualization in logsuite to use them, so my facies palette is consistent across log tracks and crossplots without redefining per chart.

**Acceptance:**
- Crossplot reads `prop.colors` first, falls back to `DEFAULT_COLORS` only for codes not present in `prop.colors`, matching WellView precedence.
- Colorbar/legend in Crossplot shows labels from `prop.labels` and colors from `prop.colors` together.
- Existing crossplots without `prop.colors` are unchanged.
- Same applies via `_ManagerPropertyProxy.colors = ...` (broadcast path).
- Regression test covers a fixture where the same property is rendered in WellView and Crossplot, asserting both use the user palette.

**Maps to:** M3 (`visualization/style.py` resolver).

## Story 3 — Pooled raw-data extraction from manager filters

As a reservoir geologist computing pooled statistics across wells, I want a single long-form DataFrame of raw plug values along with grouping columns directly from a manager-level filter, so I can compute pooled percentiles, build diagnostics, and merge with external metadata without per-well loops.

**Acceptance:**
- `manager.<property>.filter(group_a).filter(group_b).data()` returns a `DataFrame` with `well`, `DEPT`, `<property>`, `<group_a>`, `<group_b>` (and any further chained filters).
- Discrete grouping columns are returned as labels (strings) when `prop.labels` is set, integers otherwise — consistent with `Property.data()`.
- A `weighted` argument or separate `Weight` column for replicating `stats()`-style depth-weighted percentiles.
- Group-column naming uses actual property names rather than `Group1/Group2`. `flat_columns=True` opt-in on `stats(return_df=True)` initially; default in a later major version.
- Documented as the recommended path for "give me the underlying numbers" workflows.

**Maps to:** M1.1 (landed: `proxy.data()`), M1.3 (weight column), M1.4 (`flat_columns`).

## Story 4 — Petrel-syntax regression equations for poroperm transforms

As a static modeller transferring poroperm transforms from Python into Petrel, I want regression objects to expose their equations in Petrel calculator syntax (`pow(10, c1*phi + c0)`), so I can copy-paste fits straight into the property model without re-deriving coefficients or running base-conversion math each time.

**Acceptance:**
- Regression artifacts gain `equation(format="natural"|"log10"|"petrel", decimals=4)`.
- For `ExponentialRegression`: petrel form returns `pow(10, c1*PHIE + c0)` with `c0 = log10(a)`, `c1 = b/ln(10)`.
- `Crossplot.add_regression(..., equation_format="petrel")` propagates to the legend label so on-plot text matches what gets pasted into Petrel.
- Default decimal precision configurable via `decimals` (default 4 to match current behaviour).
- Documentation includes a worked example showing the same fit in natural-log and log10/Petrel form.

**Maps to:** M2 (`RegressionFit.equation(format=)`).

## Story 5 — Per-group regression fits without touching private state

As a reservoir geologist producing poroperm transforms for a DG3 deliverable, I want to add multiple regression lines to a single crossplot, each fitted to a different subset of the data (zone group, facies, well, or any combination), so that I can present zone-specific perm transforms on one figure and have the line equations, R² values, and legend entries appear alongside the existing color/shape legends.

**Why this story exists:** today we had to snapshot `xplot._data`, overwrite it with a filtered DataFrame inside a `try/finally`, call `add_regression` once per group, and restore the original. The pattern works but relies on a private attribute, requires knowing that the internal column for shape values is called `shape_val`, and would silently break in a future release. The use case (one fit per zone group, plotted together) is the standard reservoir poroperm workflow, not an edge case — it should be a single argument on `add_regression`.

**Acceptance:**
- `Crossplot.add_regression(...)` accepts a `where` argument: dict mapping public column names (zone, facies, well) to allowed values, or a callable taking the data DataFrame and returning a boolean mask.
- The R², equation, and legend entry reflect the subset the line was fitted on.
- Subsets smaller than `min_samples` (default 5) emit a warning and skip the fit instead of raising.
- Public column names in `where` resolve to the right internal columns regardless of which property is bound to color/shape.

**Status:** ✅ shipped in M3.3 + M3.5 + the post-script-test fix that translates label strings to discrete codes inside `_apply_where_filter`. The existing `story_tests/story_1_regression_subset.py` exercises this story end-to-end.

## Story 6 — One-call pooled extraction of raw plug values across wells

As a reservoir geologist computing pooled percentiles, mean/std summaries, or QC histograms of core plug data across a multi-well field, I want a single call on a manager-level property to return a long-form DataFrame of raw values together with their grouping columns, so that I can compute statistics or build diagnostic plots without writing a per-well loop.

**Why this story exists:** today we wrote the same loop three times — once for the pooled percentile table, once for the histogram diagnostic of the suspect Channel Sand plugs, and once when rebuilding pooled after switching to the corrected facies log. None of this is logic specific to our project — it should be one method.

**Acceptance:**
- `manager.<prop>.filter(group_a).filter(group_b).data()` returns a single DataFrame with columns Well, Depth, `<prop>`, `<group_a>`, `<group_b>` (and any further filters), discrete groups already resolved to labels when `prop.labels` is set.
- Wells lacking the property or any of the filter properties are silently skipped (or surfaced via a `warn_missing=True` flag).
- The grouping columns use their real property names rather than `Group1/Group2`, both here and as an opt-in `flat_columns=True` mode on `stats(return_df=True)`.
- Documented as the recommended path for "give me the underlying numbers across all wells" workflows.

**Status:** ✅ mostly shipped in M1.1 + M1.4. The `warn_missing=` flag is the one piece still missing — today the warning is unconditional and users have to suppress with `warnings.filterwarnings`. **M4.4** would replace it with an explicit kwarg.

## Story 7 — Plot-deliverable composition: titles, table panels, reusable equation formats

As a geologist preparing a DG3 figure for inclusion in a Word/PowerPoint document, I want logsuite to handle the standard deliverable patterns — moving a regression legend to a chosen corner, attaching a summary statistics table beneath the crossplot, exporting the combined result to a single SVG, and rendering regression equations in the formats my downstream tools expect — so that I can produce a final figure in a few cells instead of writing matplotlib boilerplate every time.

**Why this story exists:** three separate blocks of matplotlib glue code kept appearing in our notebooks because the library stopped at "draw the scatter":

1. To put the regression legend in the upper left, we had to detect the existing legend, save its handles and title, remove it, and rebuild a fresh `mlegend.Legend` with custom styling — because there's no `legend_loc` kwarg on `add_regression`.
2. To combine the crossplot and a porosity statistics table into a single SVG (HTML output ruled out by the MS Office requirement), we had to resize `xplot.fig`, reposition `xplot.ax` with explicit figure-coordinate math, add a new axes, format the DataFrame values into strings, render via `ax.table()`, and add a `fig.text()` title manually.
3. To get equations in the form Petrel actually expects, we wrote conversion code (`c0 = log10(a)`, `c1 = b / ln(10)`) and printed it alongside every fit, then explained to ourselves repeatedly that the plot legend's `y = a*e^(b*phi)` describes the same line as `pow(10, c1*phi + c0)`.

**Acceptance:**
- `add_regression` accepts a `legend_loc` argument (matplotlib loc string or coords) and a `legend_decimals` argument controlling equation precision.
- `Crossplot.add_table_panel(df, position="bottom"|"right", title=None, formatters=None)` attaches a DataFrame as a matplotlib-rendered table inside the same figure, so `xplot.save("file.svg")` produces the combined output. Sensible defaults handle MultiIndex DataFrames (visual merge of repeated outer levels) and NaN→"N/A".
- Each regression class exposes its equation in multiple formats — at minimum natural-log (current default) and Petrel/log10 — selectable via a `format=` argument on `equation()`. A top-level `Crossplot(equation_format="petrel")` propagates this choice to the on-plot legend so what the user sees is what they paste into Petrel.

**Status:** partially landed.
- `RegressionFit.equation(format="natural"|"log10"|"petrel")` ✅ M2.
- `add_regression(equation_format=, decimals=)` ✅ M3.8 (accepts `decimals=`; current `legend_decimals` rename is cosmetic but consistent with the story's wording).
- `Crossplot(equation_format=...)` constructor-level default that propagates to every `add_regression` ❌ — **M4.5**.
- `legend_loc=` on `add_regression` ❌ — **M4.6**.
- `Crossplot.add_table_panel(...)` ✅ — M4.7 landed via `visualization/table_panel.py`. Demonstrated end-to-end in `story_tests/story_7_scal_deliverable.py`.
