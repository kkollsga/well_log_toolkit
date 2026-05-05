"""Tests for flat_columns= on _ManagerPropertyProxy.stats()."""

import numpy as np
import pandas as pd
import pytest

from logsuite import WellDataManager


@pytest.fixture
def manager():
    """Manager with 2 wells and discrete Zone, NTG properties."""
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1010.0, 1.0)

    for wname, phie_base in [("Well_A", 0.18), ("Well_B", 0.22)]:
        df = pd.DataFrame(
            {
                "DEPT": depth,
                "PHIE": np.linspace(phie_base - 0.03, phie_base + 0.03, len(depth)),
                "Zone": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                "NTG": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            }
        )
        mgr.load_properties(
            df,
            well_col=None,
            well_name=wname,
            source_name="petrophysics",
            unit_mappings={"PHIE": "v/v"},
            type_mappings={"Zone": "discrete", "NTG": "discrete"},
            label_mappings={
                "Zone": {0: "NonReservoir", 1: "Reservoir"},
                "NTG": {0: "NonNet", 1: "Net"},
            },
        )

    return mgr


class TestFlatColumns:
    def test_default_uses_group_label(self, manager):
        df = manager.PHIE.filter("Zone").stats(return_df=True)
        # Default: single grouping level uses "Group" label
        assert "Group" in df.columns
        assert "Zone" not in df.columns

    def test_flat_columns_uses_property_name(self, manager):
        df = manager.PHIE.filter("Zone").stats(return_df=True, flat_columns=True)
        assert "Zone" in df.columns
        assert "Group" not in df.columns

    def test_flat_columns_two_filters(self, manager):
        df = manager.PHIE.filter("Zone").filter("NTG").stats(return_df=True, flat_columns=True)
        assert "Zone" in df.columns
        assert "NTG" in df.columns
        assert "Group1" not in df.columns
        assert "Group2" not in df.columns

    def test_default_two_filters_uses_generic_names(self, manager):
        df = manager.PHIE.filter("Zone").filter("NTG").stats(return_df=True)
        assert "Group1" in df.columns
        assert "Group2" in df.columns

    def test_no_filters_no_group_columns(self, manager):
        df = manager.PHIE.stats(return_df=True)
        # Without filters, only Well + stat columns
        assert "Group" not in df.columns
        assert "Group1" not in df.columns

    def test_flat_columns_no_filters_is_no_op(self, manager):
        df_a = manager.PHIE.stats(return_df=True)
        df_b = manager.PHIE.stats(return_df=True, flat_columns=True)
        # No filter levels means no group columns to rename
        assert list(df_a.columns) == list(df_b.columns)

    def test_flat_columns_values_match_default(self, manager):
        df_a = manager.PHIE.filter("Zone").stats(return_df=True)
        df_b = manager.PHIE.filter("Zone").stats(return_df=True, flat_columns=True)
        # Same data, just different column names
        assert df_a["Group"].tolist() == df_b["Zone"].tolist()
        # Numeric columns should match
        for col in ["mean", "std", "p50"]:
            if col in df_a.columns and col in df_b.columns:
                np.testing.assert_array_equal(df_a[col].to_numpy(), df_b[col].to_numpy())
