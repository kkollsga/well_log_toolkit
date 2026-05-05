"""
Story 1 — fit a regression on a subset of points without monkey-patching.

A reservoir geologist building DG3 poroperm transforms wants three regression
lines on one crossplot — one per facies — without touching internals like
``_data`` or ``shape_val``. The acceptance criteria call for a public
``where=`` argument that takes a column→value(s) dict (or a callable mask)
and a ``min_samples`` guard.

Run with: ``python story_tests/story_1_regression_subset.py``
"""

from __future__ import annotations

from pathlib import Path

from synthetic_data import FACIES_PALETTE, build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import matplotlib

matplotlib.use("Agg")  # headless

from logsuite import Crossplot

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    manager = build_manager()

    # Crossplot accepts the manager substrate directly (M3.6 contract).
    # equation_format and decimals at construction time propagate to every
    # add_regression call (M4.5).
    xplot = Crossplot(
        manager,
        x="PHIE",
        y="PERM",
        color="Facies",
        y_log=True,
        title="Per-facies poroperm transforms",
        figsize=(9, 7),
        equation_format="petrel",
        decimals=3,
    )
    xplot.plot()

    # One regression per facies in a single call — line colors are read from
    # manager.Facies.colors automatically (M4.1 + M4.2).
    xplot.add_regression_per(
        "Facies",
        "exponential",
        min_samples=10,
        legend_loc="upper left",  # M4.6: explicit legend placement
    )

    out = OUT_DIR / "output_story_1.png"
    xplot.save(str(out))
    print(f"Saved {out}")
    print(f"Stored regressions: {list(xplot.regression()['exponential'].keys())}")
    for name, line in xplot.regression_lines.items():
        print(f"  {name}: line color = {line.get_color()}, label = {line.get_label()!r}")


if __name__ == "__main__":
    main()
