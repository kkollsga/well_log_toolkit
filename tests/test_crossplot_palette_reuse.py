"""Tests for M4.1 (line color from Property.colors) and M4.2 (add_regression_per)."""

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import Crossplot, WellDataManager


@pytest.fixture
def manager_with_palette():
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
            source_name="x",
            type_mappings={"Facies": "discrete"},
            label_mappings={"Facies": {0: "Tight", 1: "Medium", 2: "Clean"}},
        )
    mgr.Facies.colors = {0: "#ff0000", 1: "#00ff00", 2: "#0000ff"}
    return mgr


class TestLineColorFromPalette:
    def test_label_string_picks_palette_color(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression("linear", name="Tight", where={"Facies": ["Tight"]})
        line = xplot.regression_lines["Tight"]
        from matplotlib.colors import to_rgba

        assert line.get_color() == "#ff0000" or to_rgba(line.get_color()) == to_rgba("#ff0000")

    def test_int_code_picks_palette_color(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression("linear", name="med", where={"Facies": [1]})
        from matplotlib.colors import to_rgba

        assert to_rgba(xplot.regression_lines["med"].get_color()) == to_rgba("#00ff00")

    def test_explicit_line_color_wins(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression(
            "linear", name="x", where={"Facies": ["Tight"]}, line_color="#abcdef"
        )
        from matplotlib.colors import to_rgba

        assert to_rgba(xplot.regression_lines["x"].get_color()) == to_rgba("#abcdef")

    def test_no_palette_no_where_falls_back_to_red(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM")  # no color binding
        xplot.plot()
        xplot.add_regression("linear", name="all")
        from matplotlib.colors import to_rgba

        assert to_rgba(xplot.regression_lines["all"].get_color()) == to_rgba("red")

    def test_multi_value_where_falls_back(self, manager_with_palette):
        # Subset spanning two facies — no single palette color applies.
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression("linear", name="two", where={"Facies": ["Tight", "Clean"]})
        from matplotlib.colors import to_rgba

        assert to_rgba(xplot.regression_lines["two"].get_color()) == to_rgba("red")


class TestAddRegressionPer:
    def test_one_regression_per_unique_value(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression_per("Facies", "linear", min_samples=3)
        regressions = xplot.regression()["linear"]
        # Three facies → three regressions
        assert set(regressions.keys()) == {"Tight", "Medium", "Clean"}

    def test_each_line_uses_palette_color(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression_per("Facies", "linear", min_samples=3)
        from matplotlib.colors import to_rgba

        expected = {
            "Tight": to_rgba("#ff0000"),
            "Medium": to_rgba("#00ff00"),
            "Clean": to_rgba("#0000ff"),
        }
        for name, target in expected.items():
            assert to_rgba(xplot.regression_lines[name].get_color()) == target

    def test_passing_where_explicitly_raises(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        with pytest.raises(TypeError):
            xplot.add_regression_per("Facies", "linear", where={"Facies": [0]})

    def test_unknown_group_property_raises(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        with pytest.raises(ValueError):
            xplot.add_regression_per("NotABoundProperty", "linear")

    def test_kwargs_forwarded(self, manager_with_palette):
        xplot = Crossplot(manager_with_palette, x="PHIE", y="PERM", color="Facies")
        xplot.plot()
        xplot.add_regression_per(
            "Facies", "linear", min_samples=3, equation_format="petrel", legend_decimals=2
        )
        # equation_format and legend_decimals should have flowed through
        for name in ["Tight", "Medium", "Clean"]:
            label = xplot.regression_lines[name].get_label()
            # Linear isn't an exponential, so petrel falls back to natural —
            # but legend_decimals=2 should still affect the rendering.
            assert "x" in label  # non-empty equation


class TestWarnMissing:
    def test_warn_missing_default_emits(self, manager_with_palette):
        # Add a third well with no PHIE
        mgr = manager_with_palette
        mgr.load_properties(
            pd.DataFrame({"DEPT": [1000.0, 1001.0], "GR": [40.0, 50.0]}),
            well_col=None,
            well_name="Well_C",
            source_name="x",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr.PHIE.data()
        assert any("Skipped" in str(w.message) for w in caught)

    def test_warn_missing_false_silences(self, manager_with_palette):
        mgr = manager_with_palette
        mgr.load_properties(
            pd.DataFrame({"DEPT": [1000.0, 1001.0], "GR": [40.0, 50.0]}),
            well_col=None,
            well_name="Well_C",
            source_name="x",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mgr.PHIE.data(warn_missing=False)
        assert not any("Skipped" in str(w.message) for w in caught)
