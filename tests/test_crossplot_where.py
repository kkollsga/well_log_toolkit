"""Tests for Crossplot.add_regression(where=, min_samples=) — Story 1."""

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import Crossplot, WellDataManager


@pytest.fixture
def manager_with_facies():
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1100.0, 1.0)
    rng = np.random.default_rng(0)
    for wname, base in [("Well_A", 0.18), ("Well_B", 0.22)]:
        phi = np.clip(rng.normal(base, 0.03, len(depth)), 0.05, 0.35)
        perm = 0.01 * np.exp(20 * phi)
        df = pd.DataFrame(
            {
                "DEPT": depth,
                "PHIE": phi,
                "PERM": perm,
                "Facies": rng.integers(0, 3, len(depth)).astype(float),
            }
        )
        mgr.load_properties(
            df,
            well_col=None,
            well_name=wname,
            source_name="x",
            type_mappings={"Facies": "discrete"},
            label_mappings={"Facies": {0: "Sand", 1: "Shale", 2: "Coal"}},
        )
    return mgr


class TestRegressionShadowFix:
    def test_regression_method_callable(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        # Method should be callable, returning empty dict before any regressions added
        result = xplot.regression()
        assert result == {}

    def test_add_then_regression_roundtrip(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        xplot.add_regression("linear")
        regs = xplot.regression()
        assert "linear" in regs


class TestWhereDict:
    def test_where_filters_by_color_property(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot.add_regression("linear", name="sand", where={"Facies": [0]})
        # Regression was stored
        assert "sand" in xplot.regression().get("linear", {})

    def test_where_with_internal_color_val_works(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        # Internal column name should also work (back-compat / power users)
        xplot.add_regression("linear", name="t", where={"color_val": [0]})
        assert "t" in xplot.regression().get("linear", {})

    def test_where_unknown_key_raises(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        with pytest.raises(ValueError, match="does not match"):
            xplot.add_regression("linear", where={"NotAProperty": [0]})


class TestWhereCallable:
    def test_callable_mask(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot.add_regression(
            "linear",
            name="cb",
            where=lambda df: df["color_val"].isin([0, 1]),
        )
        assert "cb" in xplot.regression().get("linear", {})

    def test_invalid_where_type_raises(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        with pytest.raises(TypeError, match="must be a dict or callable"):
            xplot.add_regression("linear", where="bogus")


class TestMinSamples:
    def test_subset_below_min_samples_warns_and_skips(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xplot.add_regression(
                "linear",
                name="tiny",
                where=lambda df: df.index < 2,  # at most 2 rows
                min_samples=10,
            )
        # Warning fired; regression NOT stored
        assert any("min_samples" in str(w.message) for w in caught)
        assert "tiny" not in xplot.regression().get("linear", {})

    def test_subset_meets_min_samples_proceeds(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot.add_regression(
            "linear",
            name="ok",
            where=lambda df: df["color_val"].isin([0, 1, 2]),  # all rows
            min_samples=5,
        )
        assert "ok" in xplot.regression().get("linear", {})


class TestCrossplotAcceptsManagerView:
    def test_crossplot_accepts_manager_view(self, manager_with_facies):
        view = manager_with_facies.filter(wells=["Well_A"])
        xplot = Crossplot(view, x="PHIE", y="PERM")
        assert len(xplot.wells) == 1
        assert xplot.wells[0].name == "Well_A"

    def test_crossplot_accepts_full_manager(self, manager_with_facies):
        xplot = Crossplot(manager_with_facies, x="PHIE", y="PERM")
        assert len(xplot.wells) == 2

    def test_view_filters_propagate_to_crossplot(self, manager_with_facies):
        view = manager_with_facies.filter(wells=["Well_B"])
        xplot = Crossplot(view, x="PHIE", y="PERM")
        xplot.add_regression("linear")
        # Fit ran on Well_B only
        assert "linear" in xplot.regression()


class TestSubsetActuallyRestrictsFit:
    def test_fit_uses_only_subset(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        # All-data fit
        xplot_all = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot_all.add_regression("linear", name="all")
        slope_all = xplot_all.regression()["linear"]["all"].slope

        # Subset-fit
        xplot_sub = Crossplot(wells, x="PHIE", y="PERM", color="Facies")
        xplot_sub.add_regression("linear", name="f0", where={"Facies": [0]})
        slope_sub = xplot_sub.regression()["linear"]["f0"].slope

        # Slopes should generally differ (different data points)
        # Loose check — they MAY coincide by chance, but with rng seed they shouldn't
        assert slope_all != slope_sub


class TestRegressionKwargDeprecation:
    def test_passing_regression_emits_deprecation(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Crossplot(wells, x="PHIE", y="PERM", regression="linear")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "regression" in str(w.message)
            and "add_regression" in str(w.message)
            for w in caught
        )

    def test_passing_regression_by_color_emits_deprecation(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Crossplot(wells, x="PHIE", y="PERM", color="Facies", regression_by_color="linear")
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_no_deprecation_when_no_regression_kwargs(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Crossplot(wells, x="PHIE", y="PERM")
        # No DeprecationWarning about regression* kwargs
        assert not any(
            issubclass(w.category, DeprecationWarning) and "add_regression" in str(w.message)
            for w in caught
        )


class TestEquationFormatAndDecimalsInLegend:
    def test_petrel_format_appears_in_legend_label(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression(
            "exponential",
            name="petrel-style",
            equation_format="petrel",
            decimals=2,
        )
        # Inspect the line's label
        line = xplot.regression_lines["petrel-style"]
        label = line.get_label()
        assert "pow(10," in label

    def test_decimals_kwarg_changes_label(self, manager_with_facies):
        wells = list(manager_with_facies._wells.values())
        xplot_2 = Crossplot(wells, x="PHIE", y="PERM")
        xplot_2.plot()
        xplot_2.add_regression("linear", name="d2", decimals=2)
        label_2 = xplot_2.regression_lines["d2"].get_label()

        xplot_6 = Crossplot(wells, x="PHIE", y="PERM")
        xplot_6.plot()
        xplot_6.add_regression("linear", name="d6", decimals=6)
        label_6 = xplot_6.regression_lines["d6"].get_label()

        # Different decimals → different formatted equations
        assert label_2 != label_6

    def test_default_format_unchanged(self, manager_with_facies):
        # When neither decimals nor equation_format is set, legend keeps prior
        # behavior (model's hardcoded equation, 4 decimals).
        wells = list(manager_with_facies._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        xplot.plot()
        xplot.add_regression("linear", name="default")
        label = xplot.regression_lines["default"].get_label()
        # Natural form has no "pow" or "10^"
        assert "pow(10" not in label
        assert "10^" not in label
