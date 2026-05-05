"""Tests for Crossplot.add(artifact)."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # noqa: E402

from logsuite import (
    Artifact,
    Crossplot,
    ExponentialRegression,
    LinearRegression,
    RegressionFit,
    WellDataManager,
)


@pytest.fixture
def manager_with_two_wells():
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1100.0, 1.0)
    rng = np.random.default_rng(42)
    for wname, base in [("Well_A", 0.18), ("Well_B", 0.22)]:
        phi = np.clip(rng.normal(base, 0.03, size=len(depth)), 0.05, 0.35)
        perm = 0.01 * np.exp(20 * phi) + rng.normal(0, 0.5, size=len(depth))
        perm = np.clip(perm, 0.001, None)
        df = pd.DataFrame({"DEPT": depth, "PHIE": phi, "PERM": perm})
        mgr.load_properties(
            df,
            well_col=None,
            well_name=wname,
            source_name="petrophysics",
            unit_mappings={"PHIE": "v/v", "PERM": "mD"},
        )
    return mgr


class TestCrossplotAdd:
    def test_add_returns_self(self, manager_with_two_wells):
        wells = list(manager_with_two_wells._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        df = manager_with_two_wells.properties(["PHIE", "PERM"]).data()
        reg = LinearRegression().fit(df["PHIE"], df["PERM"])
        fit = RegressionFit(reg, name="test")
        result = xplot.add(fit)
        assert result is xplot

    def test_add_renders_artifact_on_axis(self, manager_with_two_wells):
        wells = list(manager_with_two_wells._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        df = manager_with_two_wells.properties(["PHIE", "PERM"]).data()
        reg = LinearRegression().fit(df["PHIE"], df["PERM"])
        fit = RegressionFit(reg, line_color="red")
        # Number of lines on axis before / after add
        xplot.plot()
        n_before = len(xplot.ax.lines)
        xplot.add(fit)
        assert len(xplot.ax.lines) == n_before + 1

    def test_add_auto_plots_if_not_yet_rendered(self, manager_with_two_wells):
        wells = list(manager_with_two_wells._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        assert xplot.fig is None
        df = manager_with_two_wells.properties(["PHIE", "PERM"]).data()
        reg = LinearRegression().fit(df["PHIE"], df["PERM"])
        fit = RegressionFit(reg)
        xplot.add(fit)
        # add() triggers .plot() so figure should exist
        assert xplot.fig is not None

    def test_add_unsupported_artifact_raises(self, manager_with_two_wells):
        class WellViewOnly(Artifact):
            def _render_in_wellview(self, view, **kwargs):
                pass

        wells = list(manager_with_two_wells._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        with pytest.raises(TypeError, match="cannot be rendered in a Crossplot"):
            xplot.add(WellViewOnly())

    def test_chained_adds(self, manager_with_two_wells):
        wells = list(manager_with_two_wells._wells.values())
        xplot = Crossplot(wells, x="PHIE", y="PERM")
        df = manager_with_two_wells.properties(["PHIE", "PERM"]).data()

        reg1 = LinearRegression().fit(df["PHIE"], df["PERM"])
        reg2 = ExponentialRegression().fit(df["PHIE"], df["PERM"])

        xplot.add(RegressionFit(reg1, name="linear", line_color="red"))
        xplot.add(RegressionFit(reg2, name="exp", line_color="blue"))

        # Both lines were added
        labels = [line.get_label() for line in xplot.ax.lines]
        assert any("linear" in lbl for lbl in labels)
        assert any("exp" in lbl for lbl in labels)
