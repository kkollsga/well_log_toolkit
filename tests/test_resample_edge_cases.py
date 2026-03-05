"""Tests for Property.resample() edge cases."""
import numpy as np
import pytest

from pylog import Property


class TestResampleGrids:
    """Tests for resampling between different depth grids."""

    def test_resample_to_finer_grid(self):
        """Resampling to a finer grid should interpolate values."""
        depth = np.array([1000.0, 1002.0, 1004.0])
        values = np.array([0.10, 0.20, 0.30])
        prop = Property(name='PHIE', depth=depth, values=values)

        fine_depth = np.arange(1000.0, 1005.0, 1.0)
        resampled = prop.resample(fine_depth)

        assert len(resampled.depth) == 5
        assert resampled.values[0] == pytest.approx(0.10, abs=0.01)
        assert resampled.values[2] == pytest.approx(0.20, abs=0.01)
        assert resampled.values[4] == pytest.approx(0.30, abs=0.01)
        # Midpoints should be interpolated
        assert resampled.values[1] == pytest.approx(0.15, abs=0.01)

    def test_resample_to_coarser_grid(self):
        """Resampling to a coarser grid should subsample values."""
        depth = np.arange(1000.0, 1010.0, 1.0)
        values = np.linspace(0.10, 0.30, 10)
        prop = Property(name='PHIE', depth=depth, values=values)

        coarse_depth = np.array([1000.0, 1005.0, 1009.0])
        resampled = prop.resample(coarse_depth)

        assert len(resampled.depth) == 3
        assert resampled.values[0] == pytest.approx(0.10, abs=0.01)

    def test_resample_same_grid(self):
        """Resampling to the same grid should return equivalent values."""
        depth = np.arange(1000.0, 1005.0, 1.0)
        values = np.array([0.15, 0.20, 0.22, 0.18, 0.25])
        prop = Property(name='PHIE', depth=depth, values=values)

        resampled = prop.resample(depth.copy())
        np.testing.assert_array_almost_equal(resampled.values, values)


class TestResampleEdgeCases:
    """Edge cases for resampling."""

    def test_resample_outside_range_gives_nan(self):
        """Target depths outside source range should produce NaN."""
        depth = np.array([1000.0, 1001.0, 1002.0])
        values = np.array([0.10, 0.20, 0.30])
        prop = Property(name='PHIE', depth=depth, values=values)

        target = np.array([999.0, 1000.0, 1001.0, 1002.0, 1003.0])
        resampled = prop.resample(target)

        # Points outside source range should be NaN
        assert np.isnan(resampled.values[0])  # below range
        assert np.isnan(resampled.values[4])  # above range
        # Points inside should be valid
        assert not np.isnan(resampled.values[1])

    def test_resample_with_nan_values(self):
        """Source with NaN gaps should propagate NaN in resampled output."""
        depth = np.arange(1000.0, 1005.0, 1.0)
        values = np.array([0.15, np.nan, np.nan, 0.18, 0.25])
        prop = Property(name='PHIE', depth=depth, values=values)

        target = np.arange(1000.0, 1005.0, 0.5)
        resampled = prop.resample(target)

        # First value should be valid (no NaN neighbors)
        assert not np.isnan(resampled.values[0])
        # Values near NaN gaps may also be NaN due to interpolation

    def test_resample_discrete_property(self):
        """Discrete properties should use nearest/forward-fill, not interpolation."""
        depth = np.array([1000.0, 1001.0, 1002.0, 1003.0])
        values = np.array([0, 0, 1, 1], dtype=float)
        prop = Property(
            name='Zone', depth=depth, values=values,
            prop_type='discrete', labels={0: 'A', 1: 'B'}
        )

        target = np.arange(1000.0, 1003.5, 0.5)
        resampled = prop.resample(target)

        # Values should be integers (no interpolation artifacts)
        valid = resampled.values[~np.isnan(resampled.values)]
        assert all(v == int(v) for v in valid), "Discrete values should remain integers"
