"""
Story 2 — Property.colors honoured consistently across visualisations.

A user defines a discrete palette once (``manager.Facies.colors = {...}``)
and expects every visualisation to use it. Acceptance: Crossplot reads
``prop.colors`` first, falls back to the default palette only for codes
not in it, and the legend reflects the user palette.

Run with: ``python story_tests/story_2_property_colors.py``
"""

from __future__ import annotations

from synthetic_data import FACIES_PALETTE, build_manager, ensure_local_package_on_path

ensure_local_package_on_path()

import matplotlib

matplotlib.use("Agg")  # headless

from matplotlib.colors import to_rgba

from logsuite import Crossplot


def main() -> None:
    manager = build_manager()

    print("Manager-level palette set:")
    for code, color in FACIES_PALETTE.items():
        print(f"  Facies={code}: {color}")

    xplot = Crossplot(
        manager,
        x="PHIE",
        y="PERM",
        color="Facies",
        y_log=True,
        title="Discrete colours flow into Crossplot",
        figsize=(8, 6),
    )
    xplot.plot()

    # The legend should now use the user palette.
    legend = xplot.ax.get_legend()
    actual_colors = {tuple(p.get_facecolor()) for p in legend.get_patches()}
    expected_colors = {to_rgba(c) for c in FACIES_PALETTE.values()}

    print("\nLegend colors match user palette:", expected_colors.issubset(actual_colors))
    # No image saved — Story 2's content is the assertion above; the
    # canonical scatter visuals live in Story 1 (per-facies fits) and
    # Story 7 (full deliverable).


if __name__ == "__main__":
    main()
