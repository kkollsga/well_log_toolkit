"""Tests for basic LAS 3.0 file support."""
import numpy as np
import pytest
from pathlib import Path

from logsuite.io import LasFile


@pytest.fixture
def las3_file():
    """Load the LAS 3.0 test fixture."""
    fixture_path = Path(__file__).parent / "fixtures" / "test_las3.las"
    return LasFile(str(fixture_path))


class TestLas3Loading:
    """Tests for loading LAS 3.0 files."""

    def test_version_detected(self, las3_file):
        assert las3_file._las_version == '3.0'
        assert las3_file.version_info['VERS'] == '3.0'

    def test_well_name(self, las3_file):
        assert las3_file.well_info['WELL'] == 'Test-3'

    def test_curves_parsed(self, las3_file):
        assert 'DEPT' in las3_file.curves
        assert 'PHIE' in las3_file.curves
        assert 'SW' in las3_file.curves
        assert len(las3_file.curves) == 3

    def test_curve_units(self, las3_file):
        assert las3_file.curves['DEPT']['unit'] == 'm'
        assert las3_file.curves['PHIE']['unit'] == 'v/v'

    def test_data_loads(self, las3_file):
        df = las3_file.data()
        assert len(df) == 5
        assert list(df.columns) == ['DEPT', 'PHIE', 'SW']

    def test_data_values(self, las3_file):
        df = las3_file.data()
        assert np.isclose(df['DEPT'].iloc[0], 1000.0)
        assert np.isclose(df['PHIE'].iloc[0], 0.20)
        assert np.isclose(df['SW'].iloc[3], 0.35)

    def test_depth_range(self, las3_file):
        df = las3_file.data()
        assert np.isclose(df['DEPT'].min(), 1000.0)
        assert np.isclose(df['DEPT'].max(), 1004.0)
