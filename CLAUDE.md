# well-log-toolkit — Claude Code Conventions

## Build & Test

```bash
python -m pytest                              # run full test suite
python -m pytest --cov=well_log_toolkit       # with coverage
ruff check well_log_toolkit/                  # lint (always run before pushing)
black well_log_toolkit/                       # format
mypy well_log_toolkit/                        # type check
```

## Architecture

```text
well_log_toolkit/
├── io/              LAS 2.0 reader with lazy loading (headers on init, data on .data())
│   └── las_file.py
├── core/            Well → Property → PropertyOperationsMixin
│   ├── well.py
│   ├── property.py
│   └── operations.py
├── analysis/        Depth-weighted statistics, regression models, SumsAvgResult
│   ├── statistics.py
│   ├── regression.py
│   └── sums_avg.py
├── manager/         WellDataManager + proxy classes for broadcasting across wells
│   ├── data_manager.py
│   └── proxy.py
├── visualization/   Template, WellView, Crossplot for Jupyter Lab displays
│   ├── template.py
│   ├── wellview.py
│   └── crossplot.py
├── exceptions.py    WellLogError base with typed subclasses
├── utils.py         Name sanitization and filtering helpers
└── _version.py      Dynamic version from package metadata
```

## Key Patterns

- Chained filtering: `well.PHIE.filter('Zone').filter('Facies').sums_avg()` → nested dict.
- Computed properties via `__setattr__`: `well.HC = well.PHIE * (1 - well.SW)`.
- Manager broadcasting: `manager.PHIE.filter('Zone').mean()` operates across all wells.
- Strict depth alignment: operations fail on mismatched grids — users must `.resample()` explicitly.
- Source tracking: properties remember their origin LAS file for round-trip export.
- Private helpers use single underscore prefix (`_parse_headers`, `_ManagerPropertyProxy`).

## Conventions

- **Python:** 3.10+
- **Type hints:** Modern syntax (`dict[str, str]` not `Dict`), use `Optional[T]`, `Union[T1, T2]`
- **Docstrings:** NumPy style (Parameters, Returns, Raises, Examples, See Also)
- **Line length:** 100 (black + ruff)
- **Imports:** Relative within package (`from .property import Property`, `from ..exceptions import ...`)
- **Errors:** Custom exceptions from `exceptions.py`, never bare `except:`

## Changelog

Uses [towncrier](https://towncrier.readthedocs.io/) with fragments in `changes/`.
Fragment naming: `<id>.<type>` where type is: feature, bugfix, breaking, deprecation, doc, misc.

Update `CHANGELOG.md` for user-visible changes. Skip for internal refactors, CI, test-only, formatting.

## Commits & Releases

**NEVER push without explicit user approval.** Before pushing:

1. Confirm version number with user (bump patch +0.0.1)
2. Update `pyproject.toml` version + promote changelog
3. Commit, then push after user approves

Version source of truth: `pyproject.toml` line 7.

## CI/CD

Two GitHub Actions workflows:

- **ci.yml**: Tests (Python 3.10–3.13 matrix) + lint (black, ruff). Runs on every push/PR to main.
- **build-and-publish.yml**: Version check → ci-gate (waits for CI to pass) → build → publish to PyPI → GitHub release. Only runs when pyproject.toml or source changes. If tests fail, publish is blocked.
