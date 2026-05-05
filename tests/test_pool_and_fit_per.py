"""Tests for stats(pool=True) and proxy.fit_per()."""

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import ExponentialRegression, LinearRegression, RegressionFit, WellDataManager


@pytest.fixture
def manager():
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1100.0, 1.0)
    rng = np.random.default_rng(42)
    for wname, base in [("Well_A", 0.18), ("Well_B", 0.22)]:
        phi = np.clip(rng.normal(base, 0.03, len(depth)), 0.05, 0.35)
        perm = 0.01 * np.exp(20 * phi) * rng.lognormal(0, 0.1, len(depth))
        zone = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10, dtype=float)
        df = pd.DataFrame({"DEPT": depth, "PHIE": phi, "PERM": perm, "Zone": zone})
        mgr.load_properties(
            df,
            well_col=None,
            well_name=wname,
            source_name="petrophysics",
            type_mappings={"Zone": "discrete"},
            label_mappings={"Zone": {0: "NonRes", 1: "Res"}},
        )
    return mgr


# ---------------------------- stats(pool=True) -----------------------------


class TestStatsPool:
    def test_pool_with_filter_returns_one_row_per_group(self, manager):
        df = manager.PHIE.filter("Zone").stats(
            return_df=True, pool=True, flat_columns=True,
            methods=["samples", "mean", "std", "min", "max"],
        )
        assert "Zone" in df.columns
        assert "samples" in df.columns
        assert "mean" in df.columns
        assert set(df["Zone"]) == {"NonRes", "Res"}
        assert len(df) == 2

    def test_pool_no_filter_returns_single_row(self, manager):
        df = manager.PHIE.stats(
            return_df=True, pool=True,
            methods=["samples", "mean", "std", "min", "max"],
        )
        assert len(df) == 1
        assert df["samples"].iloc[0] == 200  # 100 rows × 2 wells

    def test_pool_omits_well_column(self, manager):
        df = manager.PHIE.filter("Zone").stats(return_df=True, pool=True, flat_columns=True)
        assert "Well" not in df.columns
        assert "well" not in df.columns

    def test_pool_default_methods_include_samples_and_dispersion(self, manager):
        df = manager.PHIE.filter("Zone").stats(return_df=True, pool=True, flat_columns=True)
        for col in ["samples", "mean", "std", "min", "max", "p10", "p50", "p90"]:
            assert col in df.columns

    def test_pool_percentiles_match_data(self, manager):
        # Pooled p50 should equal the median of the long-form raw data.
        raw = manager.PHIE.filter("Zone").data(warn_missing=False)
        df = manager.PHIE.filter("Zone").stats(
            return_df=True, pool=True, flat_columns=True, methods=["percentile_50"],
        )
        for label in df["Zone"]:
            expected = raw[raw["Zone"] == label]["PHIE"].quantile(0.50)
            actual = df[df["Zone"] == label]["p50"].iloc[0]
            np.testing.assert_almost_equal(actual, expected, decimal=6)

    def test_pool_unknown_method_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown statistic"):
            manager.PHIE.filter("Zone").stats(
                return_df=True, pool=True, methods=["bogus"],
            )

    def test_pool_flat_columns_false_uses_group(self, manager):
        df = manager.PHIE.filter("Zone").stats(return_df=True, pool=True, flat_columns=False)
        assert "Group" in df.columns
        assert "Zone" not in df.columns

    def test_samples_method(self, manager):
        df = manager.PHIE.filter("Zone").stats(
            return_df=True, pool=True, flat_columns=True, methods=["samples"],
        )
        # 100 rows × 2 wells = 200 total, 50/50 split between zones
        assert df["samples"].sum() == 200


# ----------------------------- fit_per -------------------------------------


class TestFitPer:
    def test_returns_dict_of_regression_fits(self, manager):
        fits = manager.properties(["PHIE", "PERM"]).fit_per("Zone", LinearRegression())
        assert isinstance(fits, dict)
        assert set(fits.keys()) == {"NonRes", "Res"}
        for fit in fits.values():
            assert isinstance(fit, RegressionFit)

    def test_each_fit_uses_subset_data(self, manager):
        fits = manager.properties(["PHIE", "PERM"]).fit_per(
            "Zone", ExponentialRegression()
        )
        # The two fits should differ — they were fit on different subsets.
        nonres_eq = fits["NonRes"].equation()
        res_eq = fits["Res"].equation()
        assert nonres_eq != res_eq

    def test_min_samples_skips_small_subset(self, manager):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fits = manager.properties(["PHIE", "PERM"]).fit_per(
                "Zone", LinearRegression(), min_samples=500
            )
        assert fits == {}  # both subsets are below min_samples
        assert any("min_samples" in str(w.message) for w in caught)

    def test_kwargs_forwarded(self, manager):
        fits = manager.properties(["PHIE", "PERM"]).fit_per(
            "Zone", ExponentialRegression(), equation_format="petrel", decimals=2
        )
        assert "pow(10," in fits["NonRes"].equation()

    def test_requires_two_properties(self, manager):
        with pytest.raises(ValueError, match="exactly 2 properties"):
            manager.properties(["PHIE"]).fit_per("Zone", LinearRegression())

    def test_works_through_where_filter(self, manager):
        # Filter to just one well, then fit_per Zone — should still yield two fits
        view = manager.filter(wells=["Well_A"])
        fits = view.properties(["PHIE", "PERM"]).fit_per("Zone", LinearRegression())
        assert set(fits.keys()) == {"NonRes", "Res"}

    def test_works_through_value_filter(self, manager):
        # where filter to one zone, then fit_per by Well — only one well's worth
        # of data flows through. fit_per still works (with one entry per
        # remaining group key — here, the single Zone label).
        view = manager.filter(where={"Zone": "Res"})
        fits = view.properties(["PHIE", "PERM"]).fit_per("Zone", LinearRegression())
        # After value filter, only "Res" data remains
        assert set(fits.keys()) == {"Res"}

    def test_each_fit_has_distinct_model_instance(self, manager):
        # The deepcopy guarantees the original passed model is unchanged
        # and each fit has its own state.
        original = LinearRegression()
        fits = manager.properties(["PHIE", "PERM"]).fit_per("Zone", original)
        # original should still be unfitted
        assert not original.fitted
        # each fit's underlying model is its own instance
        models = [f.model for f in fits.values()]
        assert all(m is not original for m in models)
        assert models[0] is not models[1]
