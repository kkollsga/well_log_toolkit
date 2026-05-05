"""Tests for _ManagerPropertyProxy.data() and _ManagerMultiPropertyProxy.data()."""

import warnings

import numpy as np
import pandas as pd
import pytest

from logsuite import WellDataManager


@pytest.fixture
def manager():
    """Manager with 2 wells, PHIE/SW continuous, Zone discrete."""
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1010.0, 1.0)

    for wname, phie_base, sw_base in [("Well_A", 0.18, 0.30), ("Well_B", 0.22, 0.35)]:
        df = pd.DataFrame(
            {
                "DEPT": depth,
                "PHIE": np.linspace(phie_base - 0.03, phie_base + 0.03, len(depth)),
                "SW": np.linspace(sw_base - 0.05, sw_base + 0.05, len(depth)),
                "Zone": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                "NTG": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            }
        )
        mgr.load_properties(
            df,
            well_col=None,
            well_name=wname,
            source_name="petrophysics",
            unit_mappings={"PHIE": "v/v", "SW": "v/v"},
            type_mappings={"Zone": "discrete", "NTG": "discrete"},
            label_mappings={
                "Zone": {0: "NonReservoir", 1: "Reservoir"},
                "NTG": {0: "NonNet", 1: "Net"},
            },
        )

    return mgr


class TestSinglePropertyData:
    def test_no_filters_returns_well_dept_value(self, manager):
        df = manager.PHIE.data()
        assert list(df.columns) == ["well", "DEPT", "PHIE"]
        assert set(df["well"].unique()) == {"Well_A", "Well_B"}
        assert len(df) == 20

    def test_per_well_row_counts(self, manager):
        df = manager.PHIE.data()
        counts = df.groupby("well").size().to_dict()
        assert counts == {"Well_A": 10, "Well_B": 10}

    def test_with_single_filter_adds_filter_column(self, manager):
        df = manager.PHIE.filter("Zone").data()
        assert list(df.columns) == ["well", "DEPT", "PHIE", "Zone"]
        # Filter values are the label strings, not raw codes
        assert set(df["Zone"].dropna().unique()) <= {"NonReservoir", "Reservoir"}

    def test_with_two_filters_adds_both_columns(self, manager):
        df = manager.PHIE.filter("Zone").filter("NTG").data()
        assert "Zone" in df.columns
        assert "NTG" in df.columns
        assert set(df["NTG"].dropna().unique()) <= {"Net", "NonNet"}

    def test_discrete_labels_false_returns_codes(self, manager):
        df = manager.PHIE.filter("Zone").data(discrete_labels=False)
        codes = set(df["Zone"].dropna().unique())
        assert codes <= {0.0, 1.0}

    def test_skips_missing_property_with_warning(self, manager):
        # Add a third well without PHIE
        df = pd.DataFrame({"DEPT": np.arange(1000.0, 1005.0), "GR": [40.0, 50.0, 60.0, 70.0, 80.0]})
        manager.load_properties(
            df, well_col=None, well_name="Well_C", source_name="petrophysics"
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = manager.PHIE.data()

        assert "Well_C" not in result["well"].unique()
        assert any("Well_C" in str(w.message) for w in caught)

    def test_empty_when_no_well_has_property(self, manager):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = manager.NONEXISTENT.data() if hasattr(manager, "NONEXISTENT") else pd.DataFrame()
        # Accessing nonexistent property raises; this test just verifies the empty-result path
        # by using a manager with no matching wells via filter that nothing satisfies
        empty_mgr = WellDataManager()
        proxy = empty_mgr.__getattr__("PHIE") if hasattr(empty_mgr, "__getattr__") else None
        # Direct attribute access on empty manager should produce a proxy but data() returns empty
        if proxy is not None:
            assert proxy.data().empty

    def test_dept_monotonic_within_well(self, manager):
        df = manager.PHIE.filter("Zone").data()
        for _, group in df.groupby("well"):
            depths = group["DEPT"].to_numpy()
            assert np.all(np.diff(depths) >= 0)

    def test_phie_values_match_input(self, manager):
        df = manager.PHIE.data()
        well_a = df[df["well"] == "Well_A"].sort_values("DEPT")
        expected = np.linspace(0.18 - 0.03, 0.18 + 0.03, 10)
        np.testing.assert_array_almost_equal(well_a["PHIE"].to_numpy(), expected)


class TestMultiPropertyData:
    def test_two_properties_one_column_each(self, manager):
        df = manager.properties(["PHIE", "SW"]).data()
        assert "PHIE" in df.columns
        assert "SW" in df.columns
        assert "well" in df.columns
        assert "DEPT" in df.columns

    def test_two_properties_with_filter(self, manager):
        df = manager.properties(["PHIE", "SW"]).filter("Zone").data()
        assert list(df.columns)[:4] == ["well", "DEPT", "PHIE", "SW"]
        assert "Zone" in df.columns

    def test_filter_emitted_once_for_multi(self, manager):
        df = manager.properties(["PHIE", "SW"]).filter("Zone").data()
        # Zone should appear exactly once
        assert sum(c == "Zone" for c in df.columns) == 1

    def test_multi_empty_when_no_matching_wells(self):
        empty_mgr = WellDataManager()
        df = empty_mgr.properties(["PHIE"]).data()
        assert df.empty

    def test_multi_dept_monotonic_within_well(self, manager):
        df = manager.properties(["PHIE", "SW"]).filter("Zone").data()
        for _, group in df.groupby("well"):
            depths = group["DEPT"].to_numpy()
            assert np.all(np.diff(depths) >= 0)


class TestWeightedData:
    def test_no_weight_column_by_default(self, manager):
        df = manager.PHIE.data()
        assert "Weight" not in df.columns

    def test_weighted_adds_column(self, manager):
        df = manager.PHIE.data(weighted=True)
        assert "Weight" in df.columns

    def test_weights_sum_to_well_depth_range(self, manager):
        df = manager.PHIE.data(weighted=True)
        # For each well, weights should sum to (DEPT_max - DEPT_min)
        # since compute_intervals uses half-intervals with edge correction
        for well_name, group in df.groupby("well"):
            depth_range = group["DEPT"].max() - group["DEPT"].min()
            weight_sum = group["Weight"].sum()
            np.testing.assert_almost_equal(weight_sum, depth_range, decimal=5)

    def test_weights_replicate_weighted_mean(self, manager):
        # Pooled depth-weighted mean using df["Weight"] should match
        # the well-level weighted mean returned by .mean()
        df = manager.PHIE.data(weighted=True)
        for well_name, group in df.groupby("well"):
            np_weighted_mean = np.average(group["PHIE"], weights=group["Weight"])
            mgr_mean = manager.PHIE.mean()[f"well_{well_name}"]
            np.testing.assert_almost_equal(np_weighted_mean, mgr_mean, decimal=4)

    def test_weights_with_filter(self, manager):
        df = manager.PHIE.filter("Zone").data(weighted=True)
        assert "Weight" in df.columns
        assert "Zone" in df.columns
        # Weight column should be positive everywhere
        assert (df["Weight"] > 0).all()

    def test_weights_multi_property(self, manager):
        df = manager.properties(["PHIE", "SW"]).data(weighted=True)
        assert "Weight" in df.columns
        assert "PHIE" in df.columns
        assert "SW" in df.columns
