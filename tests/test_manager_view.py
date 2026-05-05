"""Tests for WellDataManager.filter() and ManagerView."""

import numpy as np
import pandas as pd
import pytest

from logsuite import ManagerView, Well, WellDataManager


@pytest.fixture
def manager():
    """Manager with 3 wells, Zone and NTG discrete properties."""
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1010.0, 1.0)

    for wname, phie_base in [("Well_A", 0.18), ("Well_B", 0.22), ("Well_C", 0.15)]:
        df = pd.DataFrame(
            {
                "DEPT": depth,
                "PHIE": np.linspace(phie_base - 0.03, phie_base + 0.03, len(depth)),
                "SW": np.linspace(0.30, 0.40, len(depth)),
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


class TestFilterShape:
    def test_filter_returns_manager_view(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        assert isinstance(view, ManagerView)

    def test_filter_no_args_returns_full_view(self, manager):
        view = manager.filter()
        assert len(view) == 3
        assert set(view.wells) == {"Well_A", "Well_B", "Well_C"}

    def test_filter_keyword_only(self, manager):
        # wells= is keyword-only — positional argument must fail
        with pytest.raises(TypeError):
            manager.filter("Well_A")

    def test_string_wells_accepted(self, manager):
        view = manager.filter(wells="Well_A")
        assert view.wells == ["Well_A"]

    def test_filter_subset(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        assert set(view.wells) == {"Well_A", "Well_B"}
        assert len(view) == 2

    def test_filter_preserves_well_order(self, manager):
        view = manager.filter(wells=["Well_B", "Well_A"])
        # Order follows input list
        assert view.wells == ["Well_B", "Well_A"]

    def test_filter_unknown_well_silently_ignored(self, manager):
        view = manager.filter(wells=["Well_A", "Nonexistent"])
        assert view.wells == ["Well_A"]

    def test_filter_empty_input_yields_empty_view(self, manager):
        view = manager.filter(wells=[])
        assert len(view) == 0


class TestViewProxyOperations:
    def test_data_restricted_to_subset(self, manager):
        view = manager.filter(wells=["Well_A"])
        df = view.PHIE.data()
        assert set(df["well"].unique()) == {"Well_A"}
        assert len(df) == 10

    def test_data_with_filter_restricted_to_subset(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        df = view.PHIE.filter("Zone").data()
        assert set(df["well"].unique()) == {"Well_A", "Well_B"}
        assert "Zone" in df.columns

    def test_mean_restricted_to_subset(self, manager):
        view = manager.filter(wells=["Well_A"])
        result = view.PHIE.mean()
        # Existing stat methods use sanitized manager keys (well_<name>);
        # only .data() uses the original names.
        assert "well_Well_A" in result
        assert "well_Well_B" not in result
        assert "well_Well_C" not in result

    def test_stats_restricted_to_subset(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        df = view.PHIE.filter("Zone").stats(return_df=True)
        well_names = set(df["Well"].unique())
        assert well_names == {"well_Well_A", "well_Well_B"}

    def test_multi_property_data_restricted_to_subset(self, manager):
        view = manager.filter(wells=["Well_A"])
        df = view.properties(["PHIE", "SW"]).data()
        assert set(df["well"].unique()) == {"Well_A"}
        assert "PHIE" in df.columns
        assert "SW" in df.columns


class TestViewWellAccess:
    def test_well_attribute_access(self, manager):
        view = manager.filter(wells=["Well_A"])
        well = view.well_Well_A
        assert isinstance(well, Well)
        assert well.name == "Well_A"

    def test_excluded_well_raises_attribute_error(self, manager):
        view = manager.filter(wells=["Well_A"])
        with pytest.raises(AttributeError):
            _ = view.well_Well_B

    def test_iteration_yields_wells(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        names = {w.name for w in view}
        assert names == {"Well_A", "Well_B"}

    def test_contains_by_name(self, manager):
        view = manager.filter(wells=["Well_A"])
        assert "Well_A" in view
        assert "Well_B" not in view


class TestViewChain:
    def test_filter_again_narrows_further(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B", "Well_C"])
        narrower = view.filter(wells=["Well_A"])
        assert narrower.wells == ["Well_A"]

    def test_filter_again_outside_subset_drops(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        narrower = view.filter(wells=["Well_A", "Well_C"])
        # Well_C is not in the parent view, should be ignored
        assert narrower.wells == ["Well_A"]

    def test_view_filter_no_args_returns_self(self, manager):
        view = manager.filter(wells=["Well_A"])
        same = view.filter()
        assert same is view


class TestRepr:
    def test_repr_contains_well_count(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        r = repr(view)
        assert "2 wells" in r
        assert "Well_A" in r
        assert "Well_B" in r

    def test_repr_singular_well(self, manager):
        view = manager.filter(wells=["Well_A"])
        r = repr(view)
        assert "1 well" in r and "1 wells" not in r


class TestWhereFilter:
    def test_where_value_filter_on_data(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        df = sub.PHIE.filter("Zone").data()
        # All remaining rows are Reservoir
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_with_list_values(self, manager):
        # A view with both Zone categories allowed should keep both
        sub = manager.filter(where={"Zone": ["Reservoir", "NonReservoir"]})
        df = sub.PHIE.filter("Zone").data()
        assert set(df["Zone"].dropna().unique()) <= {"Reservoir", "NonReservoir"}

    def test_where_well_key_selects_wells(self, manager):
        sub = manager.filter(where={"well": ["Well_A"]})
        assert sub.wells == ["Well_A"]

    def test_where_well_key_combined_with_property_filter(self, manager):
        sub = manager.filter(where={"well": ["Well_A", "Well_B"], "Zone": "Reservoir"})
        df = sub.PHIE.filter("Zone").data()
        assert set(df["well"].unique()) == {"Well_A", "Well_B"}
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_blocks_stat_methods(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        with pytest.raises(NotImplementedError):
            sub.PHIE.mean()

    def test_where_blocks_stats(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        with pytest.raises(NotImplementedError):
            sub.PHIE.stats()

    def test_where_filter_then_chain(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        df = sub.PHIE.filter("Zone").filter("NTG").data()
        # Both filter columns present and where applied
        assert "Zone" in df.columns
        assert "NTG" in df.columns
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_multi_property_data(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        df = sub.properties(["PHIE"]).filter("Zone").data()
        assert "PHIE" in df.columns
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_compose_via_view_filter(self, manager):
        view = manager.filter(wells=["Well_A", "Well_B"])
        narrower = view.filter(where={"Zone": "Reservoir"})
        df = narrower.PHIE.filter("Zone").data()
        assert set(df["well"].unique()) <= {"Well_A", "Well_B"}
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_repr_includes_filters(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        r = repr(sub)
        assert "where=" in r
        assert "Zone" in r

    def test_where_actually_filters_when_chain_lacks_filter(self, manager):
        # Regression: previously a where filter on a column not in the
        # proxy's filter chain was silently skipped (the post-filter
        # checked `col in df.columns` and dropped through). Result was
        # that view.PHIE.data() returned the FULL dataset instead of
        # the where-filtered subset.
        sub = manager.filter(where={"Zone": "Reservoir"})
        df = sub.PHIE.data(warn_missing=False)
        assert "Zone" in df.columns
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_filters_multi_property_data(self, manager):
        sub = manager.filter(where={"Zone": "Reservoir"})
        df = sub.properties(["PHIE"]).data()
        assert "Zone" in df.columns
        assert set(df["Zone"].dropna().unique()) == {"Reservoir"}

    def test_where_fit_uses_actual_subset(self, manager):
        # The fit on a where-filtered view must use the subset, not all data.
        from logsuite import LinearRegression

        full = manager.properties(["PHIE", "SW"]).fit(LinearRegression())
        sub = (
            manager.filter(where={"Zone": "Reservoir"})
            .properties(["PHIE", "SW"])
            .fit(LinearRegression())
        )
        # Fits on different subsets generally produce different slopes.
        assert full.model.slope != sub.model.slope
