"""Tests for M4.7: Crossplot.add_table_panel and the table_panel renderer."""

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import Crossplot, WellDataManager
from logsuite.visualization.table_panel import (
    _column_labels,
    _format_cells,
    _row_labels_with_visual_merge,
    _value_to_str,
    render_table_panel,
)


# ----------------------------- helpers --------------------------------------


@pytest.fixture
def manager():
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1100.0, 1.0)
    rng = np.random.default_rng(0)
    for wname, base in [("A", 0.18), ("B", 0.22)]:
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
    return mgr


# ----------------------------- value formatting -----------------------------


class TestValueToStr:
    def test_nan_becomes_na(self):
        assert _value_to_str(float("nan"), None) == "N/A"

    def test_no_formatter_uses_str(self):
        assert _value_to_str(0.123456, None) == "0.123456"

    def test_callable_formatter(self):
        assert _value_to_str(0.5, lambda v: f">>{v}<<") == ">>0.5<<"

    def test_format_spec(self):
        assert _value_to_str(0.123456, ".2f") == "0.12"

    def test_format_spec_invalid_falls_back(self):
        # ".2f" on a string would error; should fall back to str().
        assert _value_to_str("abc", ".2f") == "abc"


# ----------------------------- label flatten --------------------------------


class TestColumnLabels:
    def test_simple_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert _column_labels(df) == ["a", "b"]

    def test_multiindex_columns_flatten_with_pipe(self):
        df = pd.DataFrame(
            [[1, 2]],
            columns=pd.MultiIndex.from_tuples([("Sand", "mean"), ("Sand", "p50")]),
        )
        assert _column_labels(df) == ["Sand | mean", "Sand | p50"]


class TestRowLabelsVisualMerge:
    def test_range_index_returns_none(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _row_labels_with_visual_merge(df) is None

    def test_string_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=["foo", "bar"])
        assert _row_labels_with_visual_merge(df) == ["foo", "bar"]

    def test_multiindex_blanks_repeated_outer_levels(self):
        idx = pd.MultiIndex.from_tuples(
            [
                ("Sand 2", "Reservoir"),
                ("Sand 2", "NonReservoir"),
                ("Sand 3", "Reservoir"),
                ("Sand 3", "NonReservoir"),
            ]
        )
        df = pd.DataFrame({"a": [1, 2, 3, 4]}, index=idx)
        labels = _row_labels_with_visual_merge(df)
        # First Sand 2 row has the outer level; second has it blank.
        assert labels[0] == "Sand 2 / Reservoir"
        assert labels[1] == "NonReservoir"  # outer blanked
        assert labels[2] == "Sand 3 / Reservoir"
        assert labels[3] == "NonReservoir"


# ----------------------------- format cells ---------------------------------


class TestFormatCells:
    def test_per_column_formatter_dict(self):
        df = pd.DataFrame({"a": [0.1234, 0.5678], "b": [10.0, 20.0]})
        cells = _format_cells(df, {"a": ".2f", "b": ".0f"})
        assert cells == [["0.12", "10"], ["0.57", "20"]]

    def test_nan_in_data_yields_na(self):
        df = pd.DataFrame({"a": [1.0, float("nan")]})
        cells = _format_cells(df, None)
        assert cells == [["1.0"], ["N/A"]]


# ----------------------------- end-to-end -----------------------------------


class TestAddTablePanel:
    @pytest.fixture
    def stats_df(self, manager):
        return manager.PHIE.filter("Facies").stats(
            return_df=True, flat_columns=True, methods=["mean", "percentile_50"]
        )

    def test_returns_self(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        assert xplot.add_table_panel(stats_df) is xplot

    def test_creates_new_axes(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        n_before = len(xplot.fig.axes) if xplot.fig is not None else 0
        xplot.add_table_panel(stats_df)
        n_after = len(xplot.fig.axes)
        # New axes for the table panel (at least one more)
        assert n_after >= n_before + 1

    def test_grows_figure_height_for_bottom(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM", figsize=(8, 6))
        xplot.plot()
        h_before = xplot.fig.get_size_inches()[1]
        xplot.add_table_panel(stats_df, position="bottom")
        h_after = xplot.fig.get_size_inches()[1]
        assert h_after > h_before

    def test_grows_figure_width_for_right(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM", figsize=(8, 6))
        xplot.plot()
        w_before = xplot.fig.get_size_inches()[0]
        xplot.add_table_panel(stats_df, position="right")
        w_after = xplot.fig.get_size_inches()[0]
        assert w_after > w_before

    def test_invalid_position_raises(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        with pytest.raises(ValueError, match="position must be"):
            xplot.add_table_panel(stats_df, position="diagonal")

    def test_invalid_table_fraction_raises(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        with pytest.raises(ValueError, match="table_fraction"):
            xplot.add_table_panel(stats_df, table_fraction=0.0)
        with pytest.raises(ValueError, match="table_fraction"):
            xplot.add_table_panel(stats_df, table_fraction=1.0)

    def test_title_renders(self, manager, stats_df):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.add_table_panel(stats_df, title="Per-facies stats")
        # Find an axes with that title
        titles = [ax.get_title() for ax in xplot.fig.axes]
        assert any("Per-facies stats" in t for t in titles)

    def test_save_after_add_table_panel(self, manager, stats_df, tmp_path):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.add_table_panel(stats_df, title="Stats")
        out = tmp_path / "deliverable.svg"
        xplot.save(str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_dataframe_does_not_crash(self, manager):
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        xplot.add_table_panel(pd.DataFrame())  # no rows, no columns

    def test_formatters_dict(self, manager):
        df = pd.DataFrame({"PHIE": [0.1234, 0.5678], "PERM": [10.0, 20.0]})
        xplot = Crossplot(manager, x="PHIE", y="PERM")
        # Should not raise; formatter applies during render
        xplot.add_table_panel(df, formatters={"PHIE": ".4f", "PERM": ".0f"})
