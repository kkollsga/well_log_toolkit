"""Tests for dict-like access on WellDataManager (M5 polish)."""

import numpy as np
import pandas as pd
import pytest

from logsuite import Well, WellDataManager


@pytest.fixture
def manager():
    mgr = WellDataManager()
    depth = np.arange(1000.0, 1010.0)
    for name in ["Well_A", "12/3-2 B"]:
        df = pd.DataFrame({"DEPT": depth, "PHIE": np.linspace(0.1, 0.3, 10)})
        mgr.load_properties(df, well_col=None, well_name=name, source_name="x")
    return mgr


class TestGetItem:
    def test_original_name(self, manager):
        well = manager["Well_A"]
        assert isinstance(well, Well)
        assert well.name == "Well_A"

    def test_special_character_name(self, manager):
        # Forward slashes, spaces — invalid as Python attribute names but
        # fine as dict keys.
        well = manager["12/3-2 B"]
        assert well.name == "12/3-2 B"

    def test_sanitized_dict_key(self, manager):
        well = manager["well_12_3_2_B"]
        assert well.name == "12/3-2 B"

    def test_unknown_well_raises(self, manager):
        with pytest.raises(KeyError, match="Nonexistent"):
            manager["Nonexistent"]

    def test_non_string_raises(self, manager):
        with pytest.raises(TypeError):
            manager[42]


class TestContains:
    def test_original_name(self, manager):
        assert "Well_A" in manager
        assert "12/3-2 B" in manager

    def test_sanitized_form(self, manager):
        assert "well_Well_A" in manager

    def test_unknown(self, manager):
        assert "Nonexistent" not in manager

    def test_non_string_returns_false(self, manager):
        assert (42 in manager) is False


class TestIter:
    def test_iterates_well_objects(self, manager):
        wells = list(manager)
        assert all(isinstance(w, Well) for w in wells)
        assert {w.name for w in wells} == {"Well_A", "12/3-2 B"}

    def test_len(self, manager):
        assert len(manager) == 2


class TestManagerViewGetItem:
    def test_view_getitem_original_name(self, manager):
        view = manager.filter()
        assert view["Well_A"].name == "Well_A"

    def test_view_getitem_special_chars(self, manager):
        view = manager.filter()
        assert view["12/3-2 B"].name == "12/3-2 B"

    def test_view_getitem_unknown_raises(self, manager):
        view = manager.filter(wells=["Well_A"])
        with pytest.raises(KeyError):
            view["12/3-2 B"]  # excluded by filter
