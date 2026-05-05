"""Tests for M4.5 (Crossplot(equation_format=, decimals=)) and M4.6 (add_regression(legend_loc=, legend_decimals=))."""

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import Crossplot, WellDataManager


@pytest.fixture
def manager():
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1100.0, 1.0)
    rng = np.random.default_rng(42)
    for wname, base in [("Well_A", 0.18), ("Well_B", 0.22)]:
        phi = np.clip(rng.normal(base, 0.03, len(depth)), 0.05, 0.35)
        perm = 0.01 * np.exp(20 * phi)
        df = pd.DataFrame({"DEPT": depth, "PHIE": phi, "PERM": perm})
        mgr.load_properties(
            df, well_col=None, well_name=wname, source_name="petrophysics"
        )
    return mgr


class TestCrossplotEquationFormatDefault:
    def test_constructor_default_propagates_to_add_regression(self, manager):
        # Crossplot-level equation_format should drive every add_regression
        # call's legend rendering.
        xplot = Crossplot(manager, x="PHIE", y="PERM", equation_format="petrel", decimals=2)
        xplot.plot()
        xplot.add_regression("exponential", name="all")
        label = xplot.regression_lines["all"].get_label()
        assert "pow(10," in label

    def test_explicit_call_overrides_constructor_default(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM", equation_format="petrel")
        xplot.plot()
        # Explicit "natural" on the call should override the constructor's "petrel".
        xplot.add_regression("exponential", name="natural", equation_format="natural")
        label = xplot.regression_lines["natural"].get_label()
        assert "pow(10," not in label
        assert "10^" not in label

    def test_constructor_decimals_propagates(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM", decimals=2)
        xplot.plot()
        xplot.add_regression("linear", name="d2")
        label_2 = xplot.regression_lines["d2"].get_label()

        xplot6 = Crossplot(manager, x="PHIE", y="PERM", decimals=6)
        xplot6.plot()
        xplot6.add_regression("linear", name="d6")
        label_6 = xplot6.regression_lines["d6"].get_label()

        # Different constructor decimals → different labels (more digits in d6).
        assert label_2 != label_6


class TestLegendDecimalsAlias:
    @staticmethod
    def _equation(label: str) -> str:
        """Strip the leading 'name (' and trailing ')\nR² = ...' for comparison."""
        # label is like "name (y=15.36x-2.37)\nR² = 0.794"
        first_line = label.split("\n", 1)[0]
        return first_line[first_line.index("(") + 1 : first_line.rindex(")")]

    def test_legend_decimals_acts_like_decimals(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="ld", legend_decimals=2)
        eq_ld = self._equation(xplot.regression_lines["ld"].get_label())

        xplot_eq = Crossplot(manager, x="PHIE", y="PERM")
        xplot_eq.plot()
        xplot_eq.add_regression("linear", name="d", decimals=2)
        eq_d = self._equation(xplot_eq.regression_lines["d"].get_label())

        assert eq_ld == eq_d

    def test_legend_decimals_wins_over_decimals(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="ld_wins", decimals=8, legend_decimals=2)
        eq_ld = self._equation(xplot.regression_lines["ld_wins"].get_label())

        xplot2 = Crossplot(manager, x="PHIE", y="PERM")
        xplot2.plot()
        xplot2.add_regression("linear", name="ref", decimals=2)
        eq_ref = self._equation(xplot2.regression_lines["ref"].get_label())

        # legend_decimals=2 wins over decimals=8
        assert eq_ld == eq_ref


class TestLegendLoc:
    def test_legend_loc_sets_legend_position(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="A", legend_loc="upper left")
        # The regression legend should now be at the upper-left location
        assert xplot.regression_legend is not None
        assert xplot.regression_legend._loc in (
            2,
            (0, 1),
            "upper left",
        ) or xplot.regression_legend._get_loc() == 2  # matplotlib stores upper left as 2

    def test_legend_loc_persists_across_subsequent_adds(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="first", legend_loc="upper left")
        xplot.add_regression("linear", name="second")  # no legend_loc
        # The stored override should persist for the second call too.
        assert xplot._regression_legend_loc == "upper left"

    def test_later_legend_loc_overrides_earlier(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="first", legend_loc="upper left")
        xplot.add_regression("linear", name="second", legend_loc="lower right")
        # Last call wins.
        assert xplot._regression_legend_loc == "lower right"

    def test_no_legend_loc_uses_auto_placement(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="auto")
        assert xplot._regression_legend_loc is None
        # Legend was created via the auto-placement path
        assert xplot.regression_legend is not None
