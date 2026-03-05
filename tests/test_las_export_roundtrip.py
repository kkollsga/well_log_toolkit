"""Tests for LAS export and re-import round-trip fidelity."""
import numpy as np
import pandas as pd
import pytest

from pylog import WellDataManager, Well, Property
from pylog.io import LasFile


class TestExportRoundTrip:
    """Tests for exporting a well to LAS and re-importing it."""

    def test_basic_roundtrip(self, well_with_properties, tmp_path):
        """Export to LAS and re-import, verify continuous values match."""
        las_path = tmp_path / "roundtrip.las"
        well_with_properties.export_to_las(str(las_path))

        # Re-import
        manager = WellDataManager()
        manager.load_las(str(las_path))
        reloaded = list(manager._wells.values())[0]

        # Check PHIE values match
        original = well_with_properties.PHIE.values
        reimported = reloaded.get_property('PHIE').values
        np.testing.assert_array_almost_equal(reimported, original, decimal=4)

    def test_depth_preservation(self, well_with_properties, tmp_path):
        """Exported depths should match original depths."""
        las_path = tmp_path / "depths.las"
        well_with_properties.export_to_las(str(las_path))

        manager = WellDataManager()
        manager.load_las(str(las_path))
        reloaded = list(manager._wells.values())[0]

        original_depth = well_with_properties.PHIE.depth
        reimported_depth = reloaded.get_property('PHIE').depth
        np.testing.assert_array_almost_equal(reimported_depth, original_depth, decimal=4)

    def test_nan_values_roundtrip(self, tmp_path):
        """NaN values should survive export/import as null values."""
        well = Well(name='NaN Test', sanitized_name='nan_test')
        depth = np.arange(1000.0, 1005.0, 1.0)
        values = np.array([0.15, np.nan, 0.22, np.nan, 0.25])
        df = pd.DataFrame({'DEPT': depth, 'PHIE': values})
        well.add_dataframe(df, unit_mappings={'PHIE': 'v/v'})

        las_path = tmp_path / "nan_test.las"
        well.export_to_las(str(las_path))

        manager = WellDataManager()
        manager.load_las(str(las_path))
        reloaded = list(manager._wells.values())[0]

        reimported = reloaded.get_property('PHIE').values
        assert np.isnan(reimported[1]), "NaN at index 1 should survive round-trip"
        assert np.isnan(reimported[3]), "NaN at index 3 should survive round-trip"
        assert reimported[0] == pytest.approx(0.15, abs=0.01)

    def test_multiple_properties_roundtrip(self, well_with_properties, tmp_path):
        """Multiple properties should all survive round-trip."""
        las_path = tmp_path / "multi.las"
        well_with_properties.export_to_las(str(las_path))

        manager = WellDataManager()
        manager.load_las(str(las_path))
        reloaded = list(manager._wells.values())[0]

        props = reloaded.properties
        assert 'PHIE' in props
        assert 'SW' in props


class TestExportFromLas:
    """Tests for loading from LAS file and re-exporting."""

    def test_las_load_and_reexport(self, tmp_las_file, tmp_path):
        """Load a LAS file and re-export to verify format preservation."""
        manager = WellDataManager()
        manager.load_las(str(tmp_las_file))
        well = list(manager._wells.values())[0]

        reexport_path = tmp_path / "reexported.las"
        well.export_to_las(str(reexport_path))

        # Verify the reexported file is valid LAS
        manager2 = WellDataManager()
        manager2.load_las(str(reexport_path))
        well2 = list(manager2._wells.values())[0]

        assert 'PHIE' in well2.properties
        np.testing.assert_array_almost_equal(
            well2.get_property('PHIE').values,
            well.get_property('PHIE').values,
            decimal=3
        )
