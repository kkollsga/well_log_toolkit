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
