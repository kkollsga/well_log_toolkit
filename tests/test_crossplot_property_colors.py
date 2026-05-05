"""Tests for Crossplot honoring Property.colors (Story 2)."""

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import Crossplot, WellDataManager


@pytest.fixture
def manager_with_facies():
    """Manager with PHIE/PERM and a discrete Facies property carrying user colors."""
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1100.0, 1.0)
    rng = np.random.default_rng(0)

    for wname, base in [("Well_A", 0.18), ("Well_B", 0.22)]:
        df = pd.DataFrame(
            {
                "DEPT": depth,
                "PHIE": np.clip(rng.normal(base, 0.03, len(depth)), 0.05, 0.35),
                "PERM": rng.uniform(50, 200, len(depth)),
                "Facies": rng.integers(0, 3, len(depth)).astype(float),
            }
        )
        mgr.load_properties(
            df,
            well_col=None,
            well_name=wname,
            source_name="petrophysics",
            type_mappings={"Facies": "discrete"},
            label_mappings={"Facies": {0: "Sand", 1: "Shale", 2: "Coal"}},
        )

    # Set user colors on Facies in both wells
    user_palette = {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"}
    for well in mgr._wells.values():
        prop = well.get_property("Facies")
        prop.colors = user_palette

    return mgr, user_palette


class TestCrossplotPropertyColors:
    def test_property_colors_used_in_categorical_legend(self, manager_with_facies):
        mgr, user_palette = manager_with_facies
        wells = list(mgr._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot.plot()

        # The crossplot should have stored the user palette on _discrete_colors
        assert "color" in xplot._discrete_colors
        assert xplot._discrete_colors["color"] == user_palette

    def test_legend_entries_use_user_colors(self, manager_with_facies):
        mgr, user_palette = manager_with_facies
        wells = list(mgr._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot.plot()

        # Inspect legend handles for matching face colors
        legend = xplot.ax.get_legend()
        assert legend is not None
        # Convert hex colors to RGBA for comparison
        from matplotlib.colors import to_rgba

        legend_facecolors = [tuple(p.get_facecolor()) for p in legend.get_patches()]
        expected_colors = {to_rgba(c) for c in user_palette.values()}
        # All three user colors should appear among legend entries
        legend_color_set = {tuple(c) for c in legend_facecolors}
        assert expected_colors.issubset(legend_color_set)

    def test_no_property_colors_falls_back_to_defaults(self):
        # A property without user colors should still get a palette
        mgr = WellDataManager()
        depth = np.arange(1000.0, 1010.0)
        for wname in ["A", "B"]:
            df = pd.DataFrame(
                {
                    "DEPT": depth,
                    "PHIE": np.linspace(0.1, 0.3, len(depth)),
                    "PERM": np.linspace(50, 200, len(depth)),
                    "Facies": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
                }
            )
            mgr.load_properties(
                df,
                well_col=None,
                well_name=wname,
                source_name="x",
                type_mappings={"Facies": "discrete"},
            )
        wells = list(mgr._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        # Should not crash; legend exists
        assert xplot.ax.get_legend() is not None
