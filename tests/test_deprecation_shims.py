"""Tests for M3.7 deprecation shims on Article III violators."""

import warnings

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from logsuite import WellDataManager


@pytest.fixture
def manager_with_well():
    mgr = WellDataManager()
    df = pd.DataFrame(
        {
            "DEPT": np.arange(1000.0, 1010.0),
            "PHIE": np.linspace(0.1, 0.3, 10),
            "PERM": np.linspace(50, 200, 10),
        }
    )
    mgr.load_properties(df, well_col=None, well_name="A", source_name="x")
    return mgr


class TestWellWellViewDeprecation:
    def test_emits_deprecation_warning(self, manager_with_well):
        well = manager_with_well.well_A
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            well.WellView(depth_range=(1000, 1010))
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "Well.WellView()" in str(w.message)
            for w in caught
        )

    def test_still_returns_wellview(self, manager_with_well):
        from logsuite import WellView

        well = manager_with_well.well_A
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            view = well.WellView(depth_range=(1000, 1010))
        assert isinstance(view, WellView)


class TestWellCrossplotDeprecation:
    def test_emits_deprecation_warning(self, manager_with_well):
        well = manager_with_well.well_A
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            well.Crossplot(x="PHIE", y="PERM")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "Well.Crossplot()" in str(w.message)
            for w in caught
        )

    def test_still_returns_crossplot(self, manager_with_well):
        from logsuite import Crossplot

        well = manager_with_well.well_A
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            xplot = well.Crossplot(x="PHIE", y="PERM")
        assert isinstance(xplot, Crossplot)


class TestManagerCrossplotDeprecation:
    def test_emits_deprecation_warning(self, manager_with_well):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager_with_well.Crossplot(x="PHIE", y="PERM")
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "WellDataManager.Crossplot()" in str(w.message)
            for w in caught
        )

    def test_still_returns_crossplot(self, manager_with_well):
        from logsuite import Crossplot

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            xplot = manager_with_well.Crossplot(x="PHIE", y="PERM")
        assert isinstance(xplot, Crossplot)


class TestRecommendedReplacement:
    def test_direct_constructor_works(self, manager_with_well):
        from logsuite import Crossplot, WellView

        # The replacement patterns
        view = WellView(manager_with_well.well_A, depth_range=(1000, 1010))
        xplot = Crossplot(manager_with_well.filter(), x="PHIE", y="PERM")
        assert view is not None
        assert xplot is not None
