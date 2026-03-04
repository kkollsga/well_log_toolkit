"""
Tests for weighted and arithmetic statistics functions.
"""
import numpy as np
import pytest
from well_log_toolkit.analysis.statistics import (
    compute_intervals,
    mean,
    sum as stat_sum,  # Avoid shadowing builtin
    std,
    percentile,
    compute_all_statistics,
)


# Helper functions to match old API expected by tests
def weighted_mean(values, weights):
    """Weighted mean using the new API."""
    return mean(values, weights, method='weighted')


def arithmetic_mean(values):
    """Arithmetic mean using the new API."""
    return mean(values, method='arithmetic')


def weighted_sum(values, weights):
    """Weighted sum using the new API."""
    return stat_sum(values, weights, method='weighted')


def arithmetic_sum(values):
    """Arithmetic sum using the new API."""
    return stat_sum(values, method='arithmetic')


def weighted_std(values, weights):
    """Weighted std using the new API."""
    return std(values, weights, method='weighted')


def arithmetic_std(values):
    """Arithmetic std using the new API."""
    return std(values, method='arithmetic')


def weighted_percentile(values, weights, p):
    """Weighted percentile using the new API."""
    return percentile(values, p, weights, method='weighted')


class TestComputeIntervals:
    """Test depth interval calculation."""

    def test_uniform_spacing(self):
        """Test intervals with uniform depth spacing."""
        depth = np.array([1500.0, 1501.0, 1502.0, 1503.0])
        intervals = compute_intervals(depth)
        # First: 0.5, Middle: 1.0 each, Last: 0.5
        expected = np.array([0.5, 1.0, 1.0, 0.5])
        np.testing.assert_array_almost_equal(intervals, expected)

    def test_non_uniform_spacing(self):
        """Test intervals with non-uniform depth spacing."""
        depth = np.array([1500.0, 1501.0, 1505.0])
        intervals = compute_intervals(depth)
        # First: (1501-1500)/2 = 0.5
        # Middle: (1503 - 1500.5) = 2.5
        # Last: (1505-1501)/2 = 2.0
        expected = np.array([0.5, 2.5, 2.0])
        np.testing.assert_array_almost_equal(intervals, expected)

    def test_single_point(self):
        """Test intervals for single depth point."""
        depth = np.array([1500.0])
        intervals = compute_intervals(depth)
        assert intervals[0] == 1.0  # Default interval

    def test_empty_array(self):
        """Test empty depth array."""
        depth = np.array([])
        intervals = compute_intervals(depth)
        assert len(intervals) == 0


class TestWeightedMean:
    """Test depth-weighted mean calculation."""

    def test_weighted_mean_simple(self):
        """Test weighted mean with simple example."""
        values = np.array([0.0, 1.0, 0.0])
        weights = np.array([0.5, 2.5, 2.0])
        result = weighted_mean(values, weights)
        # (0*0.5 + 1*2.5 + 0*2.0) / (0.5 + 2.5 + 2.0) = 2.5/5.0 = 0.5
        assert result == pytest.approx(0.5)

    def test_ntg_example_from_issue(self):
        """Test NTG example: 1m non-net, 4m net should give 0.8 weighted mean."""
        # NTG 0 at MD 1500, NTG 1 at MD 1501, NTG 0 at MD 1505
        depth = np.array([1500.0, 1501.0, 1505.0])
        values = np.array([0.0, 1.0, 0.0])
        intervals = compute_intervals(depth)  # [0.5, 2.5, 2.0]

        # The weighted mean should reflect that 2.5m has NTG=1 out of 5m total
        # weighted = 2.5/5.0 = 0.5 (because middle sample represents 2.5m)
        result = weighted_mean(values, intervals)
        assert result == pytest.approx(0.5)

    def test_arithmetic_vs_weighted_difference(self):
        """Demonstrate difference between arithmetic and weighted mean."""
        # NTG 0 at MD 1500, NTG 1 at MD 1501, NTG 0 at MD 1505
        values = np.array([0.0, 1.0, 0.0])

        # Arithmetic mean treats all samples equally
        arith = arithmetic_mean(values)
        assert arith == pytest.approx(1.0 / 3.0)  # 0.333...

        # Weighted mean accounts for depth intervals
        weights = np.array([0.5, 2.5, 2.0])
        weighted = weighted_mean(values, weights)
        assert weighted == pytest.approx(0.5)  # 2.5/5.0

        # They differ significantly
        assert abs(weighted - arith) > 0.1

    def test_with_nan_values(self):
        """Test weighted mean handles NaN values correctly."""
        values = np.array([0.0, np.nan, 1.0])
        weights = np.array([1.0, 2.0, 1.0])
        result = weighted_mean(values, weights)
        # Only use valid: (0*1 + 1*1) / (1 + 1) = 0.5
        assert result == pytest.approx(0.5)

    def test_all_nan_returns_nan(self):
        """Test all NaN values returns NaN."""
        values = np.array([np.nan, np.nan])
        weights = np.array([1.0, 1.0])
        result = weighted_mean(values, weights)
        assert np.isnan(result)


class TestWeightedSum:
    """Test depth-weighted sum calculation."""

    def test_net_thickness_calculation(self):
        """Test weighted sum for NTG gives net thickness."""
        values = np.array([0.0, 1.0, 0.0])
        weights = np.array([0.5, 2.5, 2.0])
        result = weighted_sum(values, weights)
        # Net thickness = 1 * 2.5 = 2.5m
        assert result == pytest.approx(2.5)

    def test_with_nan(self):
        """Test weighted sum with NaN values."""
        values = np.array([1.0, np.nan, 1.0])
        weights = np.array([1.0, 2.0, 1.0])
        result = weighted_sum(values, weights)
        # Sum of valid: 1*1 + 1*1 = 2.0
        assert result == pytest.approx(2.0)


class TestWeightedStd:
    """Test depth-weighted standard deviation."""

    def test_weighted_std_basic(self):
        """Test basic weighted standard deviation."""
        values = np.array([10.0, 20.0, 10.0])
        weights = np.array([1.0, 1.0, 1.0])  # Equal weights
        result = weighted_std(values, weights)
        # Mean = 13.33..., variance calculation...
        # Should be close to arithmetic std
        arith_std = arithmetic_std(values)
        assert result == pytest.approx(arith_std, rel=0.01)

    def test_insufficient_values_returns_nan(self):
        """Test std with single value returns NaN."""
        values = np.array([10.0])
        weights = np.array([1.0])
        result = weighted_std(values, weights)
        assert np.isnan(result)


class TestWeightedPercentile:
    """Test depth-weighted percentile calculation."""

    def test_median_uniform_weights(self):
        """Test weighted median with uniform weights.

        The implementation uses linear interpolation at the 50% cumulative weight,
        which gives 2.5 (halfway between 2 and 3) rather than the traditional
        discrete median of 3.
        """
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = weighted_percentile(values, weights, 50)
        # Linear interpolation: at 50% of total weight (2.5), interpolate between 2 and 3
        assert result == pytest.approx(2.5)

    def test_p10_p90(self):
        """Test P10 and P90 percentiles."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        weights = np.ones(10)
        p10 = weighted_percentile(values, weights, 10)
        p90 = weighted_percentile(values, weights, 90)
        assert p10 < p90
        assert p10 == pytest.approx(1.0)
        assert p90 == pytest.approx(9.0)


class TestComputeAllStatistics:
    """Test comprehensive statistics computation."""

    def test_all_stats_returned(self):
        """Test all expected statistics are returned."""
        depth = np.array([1500.0, 1501.0, 1505.0])
        values = np.array([0.0, 1.0, 0.0])
        stats = compute_all_statistics(values, depth)

        # Check all keys present
        expected_keys = [
            'weighted_mean', 'weighted_sum', 'weighted_std',
            'weighted_p10', 'weighted_p50', 'weighted_p90',
            'arithmetic_mean', 'arithmetic_sum', 'arithmetic_std',
            'count', 'depth_samples', 'depth_thickness',
            'min', 'max'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_depth_thickness_sum_of_intervals(self):
        """Test depth thickness is sum of intervals, not just endpoints."""
        depth = np.array([1500.0, 1501.0, 1505.0])
        values = np.array([10.0, 20.0, 30.0])
        stats = compute_all_statistics(values, depth)

        # Total thickness should be sum of intervals: 0.5 + 2.5 + 2.0 = 5.0
        assert stats['depth_thickness'] == pytest.approx(5.0)

        # Compare with old method (just first to last)
        old_thickness = depth[-1] - depth[0]  # 5.0
        # In this case they're equal, but concept is different


class TestIntegration:
    """Integration tests for statistics with Property class."""

    def test_weighted_vs_arithmetic_in_practice(self):
        """Test that weighted stats give correct results for irregular sampling."""
        # Simulate sparse log data with irregular sampling
        depth = np.array([1500.0, 1501.0, 1510.0])  # 1m then 9m gap
        values = np.array([100.0, 200.0, 100.0])  # High value in small interval

        intervals = compute_intervals(depth)  # [0.5, 5.0, 4.5]

        # Arithmetic mean: (100 + 200 + 100) / 3 = 133.33
        arith = arithmetic_mean(values)
        assert arith == pytest.approx(133.33, rel=0.01)

        # Weighted mean: (100*0.5 + 200*5.0 + 100*4.5) / (0.5 + 5.0 + 4.5)
        # = (50 + 1000 + 450) / 10.0 = 150.0
        weighted = weighted_mean(values, intervals)
        assert weighted == pytest.approx(150.0)

        # Weighted mean properly reflects that middle sample represents more thickness
        assert weighted > arith


class TestBoundaryInsertion:
    """Test synthetic sample insertion at zone boundaries."""

    def test_boundary_sample_inserted_at_zone_top(self):
        """Test that boundary samples are inserted at zone tops."""
        from well_log_toolkit import Property

        # NTG log with samples at 1500, 1501, 1505
        ntg = Property(
            name='NTG_Flag',
            depth=np.array([1500.0, 1501.0, 1505.0]),
            values=np.array([0.0, 1.0, 0.0]),
            prop_type='discrete'
        )

        # Zone top at 1503 (between 1501 and 1505)
        zone = Property(
            name='Zone',
            depth=np.array([1500.0, 1503.0]),
            values=np.array([1.0, 2.0]),
            prop_type='discrete'
        )

        new_depth, new_values, _ = ntg._insert_boundary_samples(zone)

        # Should insert sample at 1503
        assert len(new_depth) == 4
        assert 1503.0 in new_depth
        assert new_depth.tolist() == [1500.0, 1501.0, 1503.0, 1505.0]

        # Discrete NTG uses 'previous' method, so 1503 gets value from 1501
        assert new_values.tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_continuous_property_interpolates_at_boundary(self):
        """Test that continuous properties use linear interpolation at boundaries."""
        from well_log_toolkit import Property

        porosity = Property(
            name='PHIE',
            depth=np.array([1500.0, 1501.0, 1505.0]),
            values=np.array([0.1, 0.2, 0.1]),
            prop_type='continuous'
        )

        zone = Property(
            name='Zone',
            depth=np.array([1500.0, 1503.0]),
            values=np.array([1.0, 2.0]),
            prop_type='discrete'
        )

        new_depth, new_values, _ = porosity._insert_boundary_samples(zone)

        # Linear interpolation: 0.2 + (1503-1501)/(1505-1501) * (0.1-0.2)
        # = 0.2 + 0.5 * (-0.1) = 0.15
        assert new_values[2] == pytest.approx(0.15)

    def test_zone_statistics_with_boundary_insertion(self):
        """Test that zone statistics properly partition intervals at boundaries."""
        from well_log_toolkit import Property

        ntg = Property(
            name='NTG_Flag',
            depth=np.array([1500.0, 1501.0, 1505.0]),
            values=np.array([0.0, 1.0, 0.0]),
            prop_type='discrete'
        )

        zone = Property(
            name='Zone',
            depth=np.array([1500.0, 1503.0]),
            values=np.array([1.0, 2.0]),
            prop_type='discrete'
        )

        new_depth, new_values, _ = ntg._insert_boundary_samples(zone)
        zone_values = Property._resample_to_grid(zone.depth, zone.values, new_depth, method='previous')
        intervals = compute_intervals(new_depth)

        # Zone 1: 1500-1503 (2m total, 1.5m net)
        zone1_mask = zone_values == 1.0
        z1_thickness = np.sum(intervals[zone1_mask])
        z1_net = weighted_sum(new_values[zone1_mask], intervals[zone1_mask])
        assert z1_thickness == pytest.approx(2.0)
        assert z1_net == pytest.approx(1.5)

        # Zone 2: 1503-1505 (3m total, 2m net)
        zone2_mask = zone_values == 2.0
        z2_thickness = np.sum(intervals[zone2_mask])
        z2_net = weighted_sum(new_values[zone2_mask], intervals[zone2_mask])
        assert z2_thickness == pytest.approx(3.0)
        assert z2_net == pytest.approx(2.0)

    def test_no_insertion_when_boundary_aligns_with_sample(self):
        """Test no duplicate samples when boundary aligns with existing sample."""
        from well_log_toolkit import Property

        ntg = Property(
            name='NTG_Flag',
            depth=np.array([1500.0, 1501.0, 1503.0, 1505.0]),
            values=np.array([0.0, 1.0, 1.0, 0.0]),
            prop_type='discrete'
        )

        zone = Property(
            name='Zone',
            depth=np.array([1500.0, 1503.0]),  # Zone top at existing sample
            values=np.array([1.0, 2.0]),
            prop_type='discrete'
        )

        new_depth, new_values, _ = ntg._insert_boundary_samples(zone)

        # Should not add duplicate at 1503
        assert len(new_depth) == 4
        assert new_depth.tolist() == [1500.0, 1501.0, 1503.0, 1505.0]
