"""Tests for set_quiet (M4.3) and Crossplot.column_for (introspection helper)."""

import io
import sys

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import Crossplot, WellDataManager, set_quiet
from logsuite.utils import emit_status, is_quiet


# ----------------------------- set_quiet --------------------------------------


@pytest.fixture(autouse=True)
def _restore_quiet():
    """Always reset the global flag after each test."""
    initial = is_quiet()
    yield
    set_quiet(initial)


class TestSetQuiet:
    def test_default_emits_status(self, capsys):
        set_quiet(False)
        emit_status("hello")
        out = capsys.readouterr().out
        assert "hello" in out

    def test_set_quiet_true_silences(self, capsys):
        set_quiet(True)
        emit_status("should not appear")
        out = capsys.readouterr().out
        assert "should not appear" not in out

    def test_set_quiet_false_re_enables(self, capsys):
        set_quiet(True)
        emit_status("hidden")
        capsys.readouterr()  # drain
        set_quiet(False)
        emit_status("visible")
        out = capsys.readouterr().out
        assert "visible" in out

    def test_no_arg_defaults_to_quiet_true(self):
        set_quiet(False)
        assert is_quiet() is False
        set_quiet()  # no arg → True
        assert is_quiet() is True


class TestSetQuietSilencesProxyBroadcasts:
    @pytest.fixture
    def manager(self):
        mgr = WellDataManager()
        df = pd.DataFrame(
            {"DEPT": np.arange(1000.0, 1010.0), "PHIE": np.linspace(0.1, 0.3, 10)}
        )
        mgr.load_properties(df, well_col=None, well_name="A", source_name="x")
        return mgr

    def test_colors_setter_silenced(self, manager, capsys):
        capsys.readouterr()  # drain any earlier prints
        set_quiet(True)
        manager.PHIE.colors = {0: "#000000"}
        out = capsys.readouterr().out
        assert "✓ Set colors" not in out

    def test_colors_setter_loud_by_default(self, manager, capsys):
        set_quiet(False)
        capsys.readouterr()
        manager.PHIE.colors = {0: "#000000"}
        out = capsys.readouterr().out
        assert "✓ Set colors" in out


class TestSetQuietSilencesLoadProperties:
    def test_load_properties_silenced(self, capsys):
        set_quiet(True)
        capsys.readouterr()
        mgr = WellDataManager()
        df = pd.DataFrame(
            {"DEPT": np.arange(1000.0, 1010.0), "PHIE": np.linspace(0.1, 0.3, 10)}
        )
        mgr.load_properties(df, well_col=None, well_name="A", source_name="x")
        out = capsys.readouterr().out
        assert "✓ Loaded" not in out


# ----------------------------- column_for ------------------------------------


class TestColumnFor:
    @pytest.fixture
    def xplot(self):
        mgr = WellDataManager()
        df = pd.DataFrame(
            {
                "DEPT": np.arange(1000.0, 1010.0),
                "PHIE": np.linspace(0.1, 0.3, 10),
                "PERM": np.linspace(50, 200, 10),
                "Facies": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0],
            }
        )
        mgr.load_properties(
            df,
            well_col=None,
            well_name="A",
            source_name="x",
            type_mappings={"Facies": "discrete"},
            label_mappings={"Facies": {0: "T", 1: "M", 2: "C"}},
        )
        return Crossplot(mgr, x="PHIE", y="PERM", color="Facies")

    def test_x_property_resolves(self, xplot):
        assert xplot.column_for("PHIE") == "x"

    def test_y_property_resolves(self, xplot):
        assert xplot.column_for("PERM") == "y"

    def test_color_property_resolves(self, xplot):
        assert xplot.column_for("Facies") == "color_val"

    def test_internal_name_passes_through(self, xplot):
        assert xplot.column_for("color_val") == "color_val"
        assert xplot.column_for("x") == "x"

    def test_unknown_returns_none(self, xplot):
        assert xplot.column_for("NotABoundProperty") is None
