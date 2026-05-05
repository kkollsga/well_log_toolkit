"""
Story 5 — Per-group regression fits without touching private state.

A reservoir geologist wants multiple regression lines on a single crossplot,
each fitted to a different subset (per-facies, per-well, custom mask),
without snapshotting ``xplot._data`` or knowing about internal column
names like ``shape_val``. Two acceptance paths are exercised:

* The ergonomic one: ``add_regression_per(group_property, kind)`` enumerates
  the unique values internally and picks line colors from
  ``Property.colors``.
* The arbitrary one: ``add_regression(where=callable)`` fits on any
  user-defined boolean mask over the prepared data.

Run with: ``python story_tests/story_5_per_group_regressions.py``
"""

from __future__ import annotations

from synthetic_data import build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import matplotlib

matplotlib.use("Agg")  # headless

from logsuite import Crossplot


def main() -> None:
    manager = build_manager()

    xplot = Crossplot(
        manager,
        x="PHIE",
        y="PERM",
        color="Facies",
        y_log=True,
        title="Story 5 — per-facies fits + custom subset",
        figsize=(9, 7),
        equation_format="petrel",
        decimals=3,
    )
    xplot.plot()

    # Per-facies, palette-colored, in one call.
    xplot.add_regression_per(
        "Facies",
        "exponential",
        min_samples=10,
        legend_loc="upper left",
    )

    # An arbitrary subset via callable — the high-PHIE half of Clean.
    # column_for() avoids hardcoding the internal column names: ask the
    # crossplot which prepared-data column corresponds to each public
    # property name.
    facies_col = xplot.column_for("Facies")
    phi_col = xplot.column_for("PHIE")
    xplot.add_regression(
        "exponential",
        name="Clean (PHIE>0.20)",
        where=lambda df: (df[facies_col] == 2.0) & (df[phi_col] > 0.20),
        line_color="#fbbf24",
        line_style="--",
    )

    # No image saved — Story 5 demonstrates the API (subset filters via
    # add_regression_per and a callable where=). The canonical scatter
    # visuals live in Story 1 (per-facies fits) and Story 7 (deliverable).
    print("Per-fit summary — each fit is on its own subset:")
    print(f"  {'name':<24}{'a':>10}{'b':>10}{'R²':>10}")
    for name, reg in xplot.regression()["exponential"].items():
        print(f"  {name:<24}{reg.a:>10.4f}{reg.b:>10.4f}{reg.r_squared:>10.3f}")


if __name__ == "__main__":
    main()
