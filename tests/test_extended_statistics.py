"""Tests for geometric_mean, harmonic_mean, Property.apply, and Property.histogram."""
import numpy as np
import pytest

from pylog import Property, geometric_mean, harmonic_mean
from pylog.analysis.statistics import compute_intervals


class TestGeometricMean:
    """Tests for geometric_mean()."""

    def test_arithmetic_simple(self):
        values = np.array([1.0, 10.0, 100.0])
        result = geometric_mean(values, method='arithmetic')
        assert np.isclose(result, 10.0)

    def test_weighted(self):
        values = np.array([1.0, 100.0])
        weights = np.array([1.0, 1.0])
        result = geometric_mean(values, weights, method='weighted')
        assert np.isclose(result, 10.0)

    def test_returns_dict_when_no_method(self):
        values = np.array([1.0, 10.0, 100.0])
        weights = np.array([1.0, 1.0, 1.0])
        result = geometric_mean(values, weights)
        assert isinstance(result, dict)
        assert 'weighted' in result
        assert 'arithmetic' in result

    def test_nan_handling(self):
        values = np.array([1.0, np.nan, 100.0])
        result = geometric_mean(values, method='arithmetic')
        assert np.isclose(result, 10.0)

    def test_negative_values_return_nan(self):
        values = np.array([1.0, -5.0, 100.0])
        result = geometric_mean(values, method='arithmetic')
        assert np.isnan(result)

    def test_zero_values_return_nan(self):
        values = np.array([0.0, 10.0, 100.0])
        result = geometric_mean(values, method='arithmetic')
        assert np.isnan(result)

    def test_empty_returns_nan(self):
        values = np.array([np.nan, np.nan])
        result = geometric_mean(values, method='arithmetic')
        assert np.isnan(result)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            geometric_mean(np.array([1.0]), method='invalid')


class TestHarmonicMean:
    """Tests for harmonic_mean()."""

    def test_arithmetic_simple(self):
        values = np.array([1.0, 2.0, 4.0])
        result = harmonic_mean(values, method='arithmetic')
        expected = 3.0 / (1.0 + 0.5 + 0.25)  # 1.714...
        assert np.isclose(result, expected)

    def test_weighted(self):
        values = np.array([1.0, 4.0])
        weights = np.array([1.0, 1.0])
        result = harmonic_mean(values, weights, method='weighted')
        expected = 2.0 / (1.0 + 0.25)  # 1.6
        assert np.isclose(result, expected)

    def test_returns_dict_when_no_method(self):
        values = np.array([1.0, 2.0])
        weights = np.array([1.0, 1.0])
        result = harmonic_mean(values, weights)
        assert isinstance(result, dict)
        assert 'weighted' in result
        assert 'arithmetic' in result

    def test_nan_handling(self):
        values = np.array([1.0, np.nan, 4.0])
        result = harmonic_mean(values, method='arithmetic')
        expected = 2.0 / (1.0 + 0.25)
        assert np.isclose(result, expected)

    def test_negative_values_return_nan(self):
        values = np.array([1.0, -5.0, 4.0])
        result = harmonic_mean(values, method='arithmetic')
        assert np.isnan(result)

    def test_zero_values_return_nan(self):
        values = np.array([0.0, 2.0, 4.0])
        result = harmonic_mean(values, method='arithmetic')
        assert np.isnan(result)

    def test_harmonic_le_geometric_le_arithmetic(self):
        """Harmonic <= Geometric <= Arithmetic for positive values."""
        values = np.array([1.0, 2.0, 4.0, 8.0])
        h = harmonic_mean(values, method='arithmetic')
        g = geometric_mean(values, method='arithmetic')
        a = float(np.mean(values))
        assert h <= g <= a


class TestPropertyApply:
    """Tests for Property.apply()."""

    def test_basic_apply(self):
        depth = np.arange(1000.0, 1005.0, 1.0)
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        prop = Property(name='PHIE', depth=depth, values=values)

        result = prop.apply(np.log10, name='LOG_PHIE')
        assert result.name == 'LOG_PHIE'
        assert np.allclose(result.values, np.log10(values))
        assert np.array_equal(result.depth, depth)

    def test_default_name(self):
        depth = np.arange(1000.0, 1003.0, 1.0)
        values = np.array([1.0, 2.0, 3.0])
        prop = Property(name='PERM', depth=depth, values=values)

        result = prop.apply(lambda v: v * 2)
        assert result.name == 'PERM_applied'

    def test_does_not_mutate_original(self):
        depth = np.arange(1000.0, 1003.0, 1.0)
        values = np.array([1.0, 2.0, 3.0])
        prop = Property(name='X', depth=depth, values=values)
        original_values = values.copy()

        prop.apply(lambda v: v * 100)
        assert np.array_equal(prop.values, original_values)


class TestPropertyHistogram:
    """Tests for Property.histogram()."""

    def test_unweighted_histogram(self):
        depth = np.arange(1000.0, 1010.0, 1.0)
        values = np.random.rand(10)
        prop = Property(name='PHIE', depth=depth, values=values)

        counts, edges = prop.histogram(bins=5, weighted=False)
        assert len(counts) == 5
        assert len(edges) == 6
        assert np.sum(counts) == 10

    def test_weighted_histogram(self):
        depth = np.arange(1000.0, 1010.0, 1.0)
        values = np.random.rand(10)
        prop = Property(name='PHIE', depth=depth, values=values)

        counts, edges = prop.histogram(bins=5, weighted=True)
        assert len(counts) == 5
        assert len(edges) == 6
        # Weighted sum should equal total thickness
        intervals = compute_intervals(depth)
        assert np.isclose(np.sum(counts), np.sum(intervals))

    def test_nan_values_excluded(self):
        depth = np.arange(1000.0, 1005.0, 1.0)
        values = np.array([0.1, np.nan, 0.3, np.nan, 0.5])
        prop = Property(name='PHIE', depth=depth, values=values)

        counts, edges = prop.histogram(bins=3, weighted=False)
        assert np.sum(counts) == 3

    def test_empty_returns_empty(self):
        depth = np.arange(1000.0, 1005.0, 1.0)
        values = np.full(5, np.nan)
        prop = Property(name='PHIE', depth=depth, values=values)

        counts, edges = prop.histogram()
        assert len(counts) == 0
        assert len(edges) == 0
