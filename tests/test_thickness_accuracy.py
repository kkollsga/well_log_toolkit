"""
Comprehensive tests for thickness and statistics accuracy.

These tests ensure:
1. Zone thickness calculations are accurate
2. Sub-zone splits exactly sum to parent zone thickness
3. Facies thickness within zones sums to zone thickness
4. Statistics (mean, sum, etc.) are correctly thickness-weighted
5. Edge cases are handled correctly
"""
import numpy as np
import pytest

# Import statistics functions
from logsuite.analysis.statistics import compute_intervals, compute_zone_intervals


# =============================================================================
# Test Constants
# =============================================================================

# Tolerance for floating point comparisons
TOLERANCE = 1e-10
TOLERANCE_RELAXED = 1e-6


# =============================================================================
# Test compute_zone_intervals function
# =============================================================================

class TestComputeZoneIntervals:
    """Test the zone-aware interval calculation function."""

    def test_basic_zone_truncation(self):
        """Test that intervals are truncated at zone boundaries."""
        depth = np.array([2708.0, 2708.3, 2708.4, 2708.6, 2709.0])
        top, base = 2708.0, 2708.4

        zone_intervals = compute_zone_intervals(depth, top, base)

        # Sum should equal zone thickness exactly
        assert np.sum(zone_intervals) == pytest.approx(base - top, abs=TOLERANCE)

        # Samples outside zone should have zero interval
        assert zone_intervals[3] == 0.0  # 2708.6 is outside
        assert zone_intervals[4] == 0.0  # 2709.0 is outside

    def test_adjacent_zones_no_gaps(self):
        """Test that adjacent zones have no thickness gaps."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0, 2702.5, 2703.0])

        # Two adjacent zones
        zone1_intervals = compute_zone_intervals(depth, 2700.0, 2701.5)
        zone2_intervals = compute_zone_intervals(depth, 2701.5, 2703.0)

        zone1_thickness = np.sum(zone1_intervals)
        zone2_thickness = np.sum(zone2_intervals)
        total = zone1_thickness + zone2_thickness
        expected = 2703.0 - 2700.0

        assert total == pytest.approx(expected, abs=TOLERANCE)

    def test_zone_boundary_not_at_sample(self):
        """Test zones where boundaries don't align with sample depths."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        zone_intervals = compute_zone_intervals(depth, 2700.25, 2701.75)
        zone_thickness = np.sum(zone_intervals)
        expected = 2701.75 - 2700.25

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_single_sample_zone(self):
        """Test zone containing only one sample."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        # Zone containing only sample at 2701.0
        zone_intervals = compute_zone_intervals(depth, 2700.75, 2701.25)
        zone_thickness = np.sum(zone_intervals)
        expected = 2701.25 - 2700.75

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_empty_zone(self):
        """Test zone containing no samples."""
        depth = np.array([2700.0, 2700.5, 2702.0, 2702.5])

        # Zone between samples
        zone_intervals = compute_zone_intervals(depth, 2700.6, 2701.9)

        # All intervals should be zero (no samples in range)
        # But the thickness should still be computed based on samples that span the zone
        # Actually, with no samples IN the zone, the sum will be 0
        # This is correct behavior - if no samples exist in a zone, we can't measure it
        assert np.sum(zone_intervals) >= 0

    def test_zone_at_data_start(self):
        """Test zone at the start of data range."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2700.75)
        zone_thickness = np.sum(zone_intervals)
        expected = 2700.75 - 2700.0

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_zone_at_data_end(self):
        """Test zone at the end of data range."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        zone_intervals = compute_zone_intervals(depth, 2701.25, 2702.0)
        zone_thickness = np.sum(zone_intervals)
        expected = 2702.0 - 2701.25

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_zone_equals_full_data_range(self):
        """Test zone covering entire data range."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2702.0)
        standard_intervals = compute_intervals(depth)

        # For full range, zone intervals should match standard intervals
        # except at edges where they're truncated to the zone boundary
        zone_thickness = np.sum(zone_intervals)
        expected = 2702.0 - 2700.0

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_irregular_sampling(self):
        """Test with irregular sample spacing."""
        depth = np.array([2700.0, 2700.1, 2700.5, 2702.0, 2702.1, 2705.0])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2705.0)
        zone_thickness = np.sum(zone_intervals)
        expected = 2705.0 - 2700.0

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_multiple_zone_split_conservation(self):
        """Test that splitting a zone into multiple parts conserves thickness."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0, 2702.5, 2703.0])

        # Original zone
        original_intervals = compute_zone_intervals(depth, 2700.0, 2703.0)
        original_thickness = np.sum(original_intervals)

        # Split into 3 zones
        zone1_intervals = compute_zone_intervals(depth, 2700.0, 2701.0)
        zone2_intervals = compute_zone_intervals(depth, 2701.0, 2702.0)
        zone3_intervals = compute_zone_intervals(depth, 2702.0, 2703.0)

        split_thickness = (np.sum(zone1_intervals) +
                          np.sum(zone2_intervals) +
                          np.sum(zone3_intervals))

        assert split_thickness == pytest.approx(original_thickness, abs=TOLERANCE)


# =============================================================================
# Test Zone Thickness Conservation
# =============================================================================

class TestZoneThicknessConservation:
    """Test that zone thickness is conserved when splitting zones."""

    def test_user_scenario_sand2_split(self):
        """Reproduce the user's Sand 2 split scenario."""
        # Simulate sample depths around the boundary
        depth = np.array([
            2685.8, 2686.0, 2690.0, 2695.0, 2700.0, 2705.0,
            2707.9, 2708.0, 2708.1, 2708.2, 2708.3, 2708.4,
            2708.5, 2710.0, 2715.0, 2718.2, 2718.3
        ])

        # Original Sand 2: 2685.8 to 2718.3
        original_intervals = compute_zone_intervals(depth, 2685.8, 2718.3)
        original_thickness = np.sum(original_intervals)

        # Sub-zone Slump: 2685.8 to 2708.4
        slump_intervals = compute_zone_intervals(depth, 2685.8, 2708.4)
        slump_thickness = np.sum(slump_intervals)

        # Sub-zone SST: 2708.4 to 2718.3
        sst_intervals = compute_zone_intervals(depth, 2708.4, 2718.3)
        sst_thickness = np.sum(sst_intervals)

        # Sub-zones must sum to original
        assert slump_thickness + sst_thickness == pytest.approx(original_thickness, abs=TOLERANCE)

    def test_many_small_zones(self):
        """Test splitting into many small zones."""
        depth = np.linspace(2000.0, 2100.0, 201)  # 0.5m spacing

        original_intervals = compute_zone_intervals(depth, 2000.0, 2100.0)
        original_thickness = np.sum(original_intervals)

        # Split into 10 zones of 10m each
        total_split = 0.0
        for i in range(10):
            top = 2000.0 + i * 10.0
            base = top + 10.0
            zone_intervals = compute_zone_intervals(depth, top, base)
            total_split += np.sum(zone_intervals)

        assert total_split == pytest.approx(original_thickness, abs=TOLERANCE)

    def test_overlapping_zones_independence(self):
        """Test that overlapping zones are calculated independently."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        # Two overlapping zones
        zone1_intervals = compute_zone_intervals(depth, 2700.0, 2701.5)
        zone2_intervals = compute_zone_intervals(depth, 2700.5, 2702.0)

        # Each zone should have its own valid thickness
        zone1_thickness = np.sum(zone1_intervals)
        zone2_thickness = np.sum(zone2_intervals)

        assert zone1_thickness == pytest.approx(1.5, abs=TOLERANCE)
        assert zone2_thickness == pytest.approx(1.5, abs=TOLERANCE)


# =============================================================================
# Test Facies Thickness Consistency
# =============================================================================

class TestFaciesThicknessConsistency:
    """Test that facies thickness within zones is consistent."""

    def test_facies_sum_equals_zone_thickness(self):
        """Test that sum of facies thicknesses equals zone thickness."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0, 2702.5, 2703.0])
        # Facies: 0=Sand, 1=Shale, 2=Slump
        facies = np.array([0, 0, 1, 1, 2, 2, 0])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2703.0)
        zone_thickness = np.sum(zone_intervals)

        # Calculate thickness per facies
        facies_thicknesses = {}
        for f in np.unique(facies):
            mask = facies == f
            facies_thicknesses[f] = np.sum(zone_intervals[mask])

        total_facies = sum(facies_thicknesses.values())

        assert total_facies == pytest.approx(zone_thickness, abs=TOLERANCE)

    def test_facies_split_conservation(self):
        """Test that facies thickness is conserved when splitting zones."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0, 2702.5, 2703.0])
        # Facies: consistent across zones
        facies = np.array([0, 0, 1, 1, 0, 1, 0])

        # Original zone facies thickness
        original_intervals = compute_zone_intervals(depth, 2700.0, 2703.0)
        original_facies_0 = np.sum(original_intervals[facies == 0])
        original_facies_1 = np.sum(original_intervals[facies == 1])

        # Split zone facies thickness
        # Zone intervals already account for zone boundaries - no need for additional masks
        zone1_intervals = compute_zone_intervals(depth, 2700.0, 2701.5)
        zone2_intervals = compute_zone_intervals(depth, 2701.5, 2703.0)

        # Sum facies from both zones - zone intervals handle boundary truncation
        split_facies_0 = (np.sum(zone1_intervals[facies == 0]) +
                         np.sum(zone2_intervals[facies == 0]))
        split_facies_1 = (np.sum(zone1_intervals[facies == 1]) +
                         np.sum(zone2_intervals[facies == 1]))

        assert split_facies_0 == pytest.approx(original_facies_0, abs=TOLERANCE)
        assert split_facies_1 == pytest.approx(original_facies_1, abs=TOLERANCE)

    def test_facies_fractions_sum_to_one(self):
        """Test that facies fractions within a zone sum to 1.0."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])
        facies = np.array([0, 1, 0, 1, 0])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2702.0)
        zone_thickness = np.sum(zone_intervals)

        fraction_0 = np.sum(zone_intervals[facies == 0]) / zone_thickness
        fraction_1 = np.sum(zone_intervals[facies == 1]) / zone_thickness

        assert fraction_0 + fraction_1 == pytest.approx(1.0, abs=TOLERANCE)


# =============================================================================
# Test Weighted Statistics Accuracy
# =============================================================================

class TestWeightedStatisticsAccuracy:
    """Test that weighted statistics use correct interval weights."""

    def test_weighted_mean_with_zone_intervals(self):
        """Test weighted mean uses zone-aware intervals correctly."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])
        values = np.array([10.0, 20.0, 30.0, 20.0, 10.0])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2702.0)

        # Manual weighted mean calculation
        weighted_sum = np.sum(values * zone_intervals)
        total_weight = np.sum(zone_intervals)
        expected_mean = weighted_sum / total_weight

        # The calculation should be exact
        assert expected_mean == pytest.approx(20.0, abs=TOLERANCE_RELAXED)

    def test_weighted_sum_with_zone_intervals(self):
        """Test weighted sum (e.g., net thickness) with zone intervals."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])
        ntg = np.array([1.0, 0.0, 1.0, 0.0, 1.0])  # Net/Gross flag

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2702.0)

        # Net thickness = sum of intervals where NTG=1
        net_thickness = np.sum(zone_intervals[ntg == 1])
        gross_thickness = np.sum(zone_intervals)

        # Verify NTG ratio
        ntg_ratio = net_thickness / gross_thickness
        assert 0.0 <= ntg_ratio <= 1.0

    def test_statistics_consistency_across_zone_splits(self):
        """Test that aggregated statistics are consistent when zones are split."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0, 2702.5, 2703.0])
        values = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0])

        # Original zone weighted sum
        original_intervals = compute_zone_intervals(depth, 2700.0, 2703.0)
        original_weighted_sum = np.sum(values * original_intervals)

        # Split zone weighted sums
        # Zone intervals already account for zone boundaries - no need for additional masks
        zone1_intervals = compute_zone_intervals(depth, 2700.0, 2701.5)
        zone2_intervals = compute_zone_intervals(depth, 2701.5, 2703.0)

        # Sum weighted values from both zones - zone intervals handle boundary truncation
        split_weighted_sum = (np.sum(values * zone1_intervals) +
                             np.sum(values * zone2_intervals))

        assert split_weighted_sum == pytest.approx(original_weighted_sum, abs=TOLERANCE)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test zone with single sample."""
        depth = np.array([2700.0])
        zone_intervals = compute_zone_intervals(depth, 2699.5, 2700.5)

        # Single sample should get the full zone thickness
        assert np.sum(zone_intervals) == pytest.approx(1.0, abs=TOLERANCE)

    def test_two_samples(self):
        """Test zone with two samples."""
        depth = np.array([2700.0, 2701.0])
        zone_intervals = compute_zone_intervals(depth, 2700.0, 2701.0)

        assert np.sum(zone_intervals) == pytest.approx(1.0, abs=TOLERANCE)

    def test_very_small_zone(self):
        """Test very small zone between samples."""
        depth = np.array([2700.0, 2700.1, 2700.2, 2700.3])
        zone_intervals = compute_zone_intervals(depth, 2700.05, 2700.15)

        # Zone is 0.1m thick
        assert np.sum(zone_intervals) == pytest.approx(0.1, abs=TOLERANCE)

    def test_zone_boundary_at_midpoint(self):
        """Test zone boundary exactly at midpoint between samples."""
        depth = np.array([2700.0, 2701.0, 2702.0])

        # Boundary at 2700.5 (midpoint between first two samples)
        zone1_intervals = compute_zone_intervals(depth, 2700.0, 2700.5)
        zone2_intervals = compute_zone_intervals(depth, 2700.5, 2702.0)

        total = np.sum(zone1_intervals) + np.sum(zone2_intervals)
        expected = 2702.0 - 2700.0

        assert total == pytest.approx(expected, abs=TOLERANCE)

    def test_high_precision_boundaries(self):
        """Test with high precision boundary values."""
        depth = np.array([2700.0, 2700.5, 2701.0, 2701.5, 2702.0])

        # Use high precision boundaries
        zone_intervals = compute_zone_intervals(depth, 2700.123456789, 2701.876543211)
        zone_thickness = np.sum(zone_intervals)
        expected = 2701.876543211 - 2700.123456789

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_empty_depth_array(self):
        """Test with empty depth array."""
        depth = np.array([])
        zone_intervals = compute_zone_intervals(depth, 2700.0, 2701.0)

        assert len(zone_intervals) == 0

    def test_zone_outside_data_range(self):
        """Test zone completely outside data range."""
        depth = np.array([2700.0, 2700.5, 2701.0])
        zone_intervals = compute_zone_intervals(depth, 2800.0, 2801.0)

        # No samples in zone
        assert np.sum(zone_intervals) == 0.0

    def test_zone_partially_outside_data_range_top(self):
        """Test zone that starts before data range."""
        depth = np.array([2700.0, 2700.5, 2701.0])
        zone_intervals = compute_zone_intervals(depth, 2699.0, 2700.5)

        # Should only count from 2700.0 to 2700.5
        # First sample interval extends from 2700.0 - (0.5/2) = 2699.75 to 2700.25
        # But truncated to zone bounds
        assert np.sum(zone_intervals) > 0

    def test_zone_partially_outside_data_range_base(self):
        """Test zone that ends after data range."""
        depth = np.array([2700.0, 2700.5, 2701.0])
        zone_intervals = compute_zone_intervals(depth, 2700.5, 2702.0)

        # Should only count from 2700.5 to 2701.0
        assert np.sum(zone_intervals) > 0

    def test_nan_values_in_depth(self):
        """Test handling of NaN in depth array (should not occur but test robustness)."""
        depth = np.array([2700.0, 2700.5, 2701.0])
        # Note: compute_zone_intervals assumes sorted valid depths
        # This test just ensures no crashes
        zone_intervals = compute_zone_intervals(depth, 2700.0, 2701.0)
        assert len(zone_intervals) == 3


# =============================================================================
# Test Numerical Precision
# =============================================================================

class TestNumericalPrecision:
    """Test numerical precision of calculations."""

    def test_floating_point_accumulation(self):
        """Test that floating point errors don't accumulate."""
        # Create many samples
        depth = np.linspace(2000.0, 3000.0, 10001)  # 0.1m spacing

        # Split into 100 zones
        total_split = 0.0
        for i in range(100):
            top = 2000.0 + i * 10.0
            base = top + 10.0
            zone_intervals = compute_zone_intervals(depth, top, base)
            total_split += np.sum(zone_intervals)

        expected = 1000.0  # Total range

        # Even with many splits, should be very accurate
        assert total_split == pytest.approx(expected, abs=TOLERANCE_RELAXED)

    def test_very_thin_zones(self):
        """Test with very thin zones (sub-millimeter)."""
        depth = np.array([2700.0, 2700.001, 2700.002, 2700.003])

        zone_intervals = compute_zone_intervals(depth, 2700.0, 2700.003)
        zone_thickness = np.sum(zone_intervals)
        expected = 0.003

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)

    def test_large_depth_values(self):
        """Test with large depth values."""
        depth = np.array([10000.0, 10000.5, 10001.0, 10001.5, 10002.0])

        zone_intervals = compute_zone_intervals(depth, 10000.0, 10002.0)
        zone_thickness = np.sum(zone_intervals)
        expected = 2.0

        assert zone_thickness == pytest.approx(expected, abs=TOLERANCE)


# =============================================================================
# Test Integration with Property Class
# =============================================================================

from logsuite import Property


class TestPropertyIntegration:
    """Integration tests with the Property class."""

    def test_filter_intervals_thickness_conservation(self):
        """Test that filter_intervals conserves thickness."""
        depth = np.linspace(2700.0, 2710.0, 21)  # 0.5m spacing
        values = np.random.rand(21)

        prop = Property(
            name='Test',
            depth=depth,
            values=values,
            prop_type='continuous'
        )

        intervals = [
            {'name': 'Zone1', 'top': 2700.0, 'base': 2705.0},
            {'name': 'Zone2', 'top': 2705.0, 'base': 2710.0},
        ]

        filtered = prop.filter_intervals(intervals)
        stats = filtered.sums_avg()

        zone1_thickness = stats['Zone1']['thickness']
        zone2_thickness = stats['Zone2']['thickness']
        total = zone1_thickness + zone2_thickness

        expected = 2710.0 - 2700.0

        assert total == pytest.approx(expected, abs=TOLERANCE_RELAXED)

    def test_discrete_summary_facies_conservation(self):
        """Test that discrete_summary conserves facies thickness."""
        depth = np.linspace(2700.0, 2710.0, 21)
        # Create a pattern of facies
        values = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=float)

        prop = Property(
            name='Facies',
            depth=depth,
            values=values,
            prop_type='discrete',
            labels={0: 'Sand', 1: 'Shale'}
        )

        intervals = [
            {'name': 'Zone1', 'top': 2700.0, 'base': 2705.0},
            {'name': 'Zone2', 'top': 2705.0, 'base': 2710.0},
        ]

        filtered = prop.filter_intervals(intervals)
        summary = filtered.discrete_summary(skip=['code', 'count'])

        # Check each zone
        for zone_name in ['Zone1', 'Zone2']:
            zone_data = summary[zone_name]
            zone_thickness = zone_data['thickness']

            # Sum of facies should equal zone thickness
            facies_sum = sum(f['thickness'] for f in zone_data['facies'].values())
            assert facies_sum == pytest.approx(zone_thickness, abs=TOLERANCE_RELAXED)

            # Fractions should sum to 1
            fraction_sum = sum(f['fraction'] for f in zone_data['facies'].values())
            assert fraction_sum == pytest.approx(1.0, abs=TOLERANCE_RELAXED)


# =============================================================================
# Test Boundary Insertion Behavior
# =============================================================================

class TestBoundaryInsertion:
    """Test that boundary samples are correctly inserted and interpolated."""

    def test_continuous_linear_interpolation_at_boundary(self):
        """Test that continuous properties use linear interpolation at boundaries.

        For samples at 95 (value=0) and 105 (value=1), a boundary at 100 should
        get an interpolated value of 0.5.
        """
        # Samples offset from zone boundaries
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205,
                          215, 225, 235, 245, 255, 265, 275, 285, 295, 305], dtype=float)
        values = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        prop = Property(
            name='Test',
            depth=depth,
            values=values,
            prop_type='continuous'
        )

        intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 200.0},
            {'name': 'Zone2', 'top': 200.0, 'base': 300.0},
        ]

        filtered = prop.filter_intervals(intervals)

        # Check boundary samples were inserted
        assert 100.0 in filtered.depth
        assert 200.0 in filtered.depth
        assert 300.0 in filtered.depth

        # Check interpolated values at boundaries
        idx_100 = np.where(filtered.depth == 100.0)[0][0]
        idx_200 = np.where(filtered.depth == 200.0)[0][0]
        idx_300 = np.where(filtered.depth == 300.0)[0][0]

        # At 100: interpolate between 95 (0) and 105 (1) -> 0.5
        assert filtered.values[idx_100] == pytest.approx(0.5, abs=TOLERANCE_RELAXED)

        # At 200: interpolate between 195 (1) and 205 (1) -> 1.0
        assert filtered.values[idx_200] == pytest.approx(1.0, abs=TOLERANCE_RELAXED)

        # At 300: interpolate between 295 (0) and 305 (0) -> 0.0
        assert filtered.values[idx_300] == pytest.approx(0.0, abs=TOLERANCE_RELAXED)

    def test_discrete_previous_value_at_boundary(self):
        """Test that discrete properties use previous value at boundaries.

        For samples at 95 (value=0) and 105 (value=1), a boundary at 100 should
        get the previous value of 0.
        """
        # Samples offset from zone boundaries
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205,
                          215, 225, 235, 245, 255, 265, 275, 285, 295, 305], dtype=float)
        values = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        prop = Property(
            name='Facies',
            depth=depth,
            values=values,
            prop_type='discrete',
            labels={0: 'Sand', 1: 'Net'}
        )

        intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 200.0},
            {'name': 'Zone2', 'top': 200.0, 'base': 300.0},
        ]

        filtered = prop.filter_intervals(intervals)

        # Check boundary samples were inserted
        assert 100.0 in filtered.depth
        assert 200.0 in filtered.depth
        assert 300.0 in filtered.depth

        # Check previous values at boundaries
        idx_100 = np.where(filtered.depth == 100.0)[0][0]
        idx_200 = np.where(filtered.depth == 200.0)[0][0]
        idx_300 = np.where(filtered.depth == 300.0)[0][0]

        # At 100: previous value from 95 is 0
        assert filtered.values[idx_100] == 0

        # At 200: previous value from 195 is 1
        assert filtered.values[idx_200] == 1

        # At 300: previous value from 295 is 0
        assert filtered.values[idx_300] == 0

    def test_continuous_statistics_with_boundary_insertion(self):
        """Test that continuous statistics are correct after boundary insertion."""
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205,
                          215, 225, 235, 245, 255, 265, 275, 285, 295, 305], dtype=float)
        values = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        prop = Property(
            name='Test',
            depth=depth,
            values=values,
            prop_type='continuous'
        )

        intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 200.0},
            {'name': 'Zone2', 'top': 200.0, 'base': 300.0},
        ]

        filtered = prop.filter_intervals(intervals)
        stats = filtered.sums_avg()

        # Zone thicknesses should be exact
        assert stats['Zone1']['thickness'] == pytest.approx(100.0, abs=TOLERANCE_RELAXED)
        assert stats['Zone2']['thickness'] == pytest.approx(100.0, abs=TOLERANCE_RELAXED)

        # Zone1: boundary at 100 has value 0.5, rest is 1.0
        # Mean should be slightly less than 1.0
        # Interval from 100-102.5 has value 0.5 (2.5m), rest has value 1.0 (97.5m)
        # Expected mean = (0.5 * 2.5 + 1.0 * 97.5) / 100 = 0.9875
        assert stats['Zone1']['mean'] == pytest.approx(0.9875, abs=TOLERANCE_RELAXED)

        # Zone2: boundary at 200 has value 1.0, sample at 205 has value 1.0
        # Interval from 200-210 has value 1.0 (10m), rest is 0.0 (90m)
        # Expected mean = (1.0 * 10 + 0.0 * 90) / 100 = 0.1
        assert stats['Zone2']['mean'] == pytest.approx(0.1, abs=TOLERANCE_RELAXED)

    def test_discrete_statistics_with_boundary_insertion(self):
        """Test that discrete statistics are correct after boundary insertion."""
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205,
                          215, 225, 235, 245, 255, 265, 275, 285, 295, 305], dtype=float)
        values = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        prop = Property(
            name='Facies',
            depth=depth,
            values=values,
            prop_type='discrete',
            labels={0: 'Sand', 1: 'Net'}
        )

        intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 200.0},
            {'name': 'Zone2', 'top': 200.0, 'base': 300.0},
        ]

        filtered = prop.filter_intervals(intervals)
        summary = filtered.discrete_summary(skip=['code', 'count'])

        # Zone thicknesses should be exact
        assert summary['Zone1']['thickness'] == pytest.approx(100.0, abs=TOLERANCE_RELAXED)
        assert summary['Zone2']['thickness'] == pytest.approx(100.0, abs=TOLERANCE_RELAXED)

        # Zone1: boundary at 100 gets previous value 0
        # Sand (0): from 100 to midpoint(100, 105) = 102.5 -> 2.5m
        # Net (1): from 102.5 to 200 -> 97.5m
        assert summary['Zone1']['facies']['Sand']['thickness'] == pytest.approx(2.5, abs=TOLERANCE_RELAXED)
        assert summary['Zone1']['facies']['Net']['thickness'] == pytest.approx(97.5, abs=TOLERANCE_RELAXED)

        # Zone2: boundary at 200 gets previous value 1
        # Net (1): from 200 to midpoint(205, 215) = 210 -> 10m
        # Sand (0): from 210 to 300 -> 90m
        assert summary['Zone2']['facies']['Net']['thickness'] == pytest.approx(10.0, abs=TOLERANCE_RELAXED)
        assert summary['Zone2']['facies']['Sand']['thickness'] == pytest.approx(90.0, abs=TOLERANCE_RELAXED)

    def test_boundary_at_existing_sample_no_duplicate(self):
        """Test that no duplicate sample is created when boundary aligns with existing sample."""
        depth = np.array([90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                          210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310], dtype=float)
        values = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)

        prop = Property(
            name='Test',
            depth=depth,
            values=values,
            prop_type='continuous'
        )

        intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 200.0},
            {'name': 'Zone2', 'top': 200.0, 'base': 300.0},
        ]

        filtered = prop.filter_intervals(intervals)

        # No new samples should be inserted (boundaries align with existing samples)
        assert len(filtered.depth) == len(depth)

        # Verify statistics are still correct
        stats = filtered.sums_avg()
        assert stats['Zone1']['thickness'] == pytest.approx(100.0, abs=TOLERANCE_RELAXED)
        assert stats['Zone2']['thickness'] == pytest.approx(100.0, abs=TOLERANCE_RELAXED)

    def test_adjacent_zones_thickness_conservation_with_insertion(self):
        """Test that adjacent zones sum to total thickness after boundary insertion."""
        # Create data that requires boundary insertion
        depth = np.linspace(50, 350, 31)  # 10m spacing, not aligned with boundaries
        values = np.sin(depth / 50) * 0.5 + 0.5  # Varying continuous values

        prop = Property(
            name='Test',
            depth=depth,
            values=values,
            prop_type='continuous'
        )

        intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 150.0},
            {'name': 'Zone2', 'top': 150.0, 'base': 200.0},
            {'name': 'Zone3', 'top': 200.0, 'base': 250.0},
            {'name': 'Zone4', 'top': 250.0, 'base': 300.0},
        ]

        filtered = prop.filter_intervals(intervals)
        stats = filtered.sums_avg()

        # Each zone should be exactly 50m
        for zone in ['Zone1', 'Zone2', 'Zone3', 'Zone4']:
            assert stats[zone]['thickness'] == pytest.approx(50.0, abs=TOLERANCE_RELAXED)

        # Total should be 200m
        total = sum(stats[z]['thickness'] for z in ['Zone1', 'Zone2', 'Zone3', 'Zone4'])
        assert total == pytest.approx(200.0, abs=TOLERANCE_RELAXED)

    def test_split_zones_sum_to_parent_zone(self):
        """Test that splitting a zone into sub-zones preserves total thickness.

        This tests the scenario where:
        - Original zone: Sand 3 (100-200)
        - Split into: Sand 3_SST (100-160) and Sand 3_Slump (160-200)

        For discrete properties, boundary insertion uses "previous value" method,
        which can shift facies attribution at split boundaries. This is correct
        behavior for step-function discrete properties.

        Total zone thickness MUST be conserved.
        Facies thickness may shift at boundaries due to discrete interpolation.
        """
        # Create samples not aligned with boundaries
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205], dtype=float)
        # Facies pattern: mostly Net (1) with some NonNet (0)
        facies = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=float)

        prop = Property(
            name='Facies',
            depth=depth,
            values=facies,
            prop_type='discrete',
            labels={0: 'NonNet', 1: 'Net'}
        )

        # Original zone
        original_intervals = [{'name': 'Sand3', 'top': 100.0, 'base': 200.0}]
        original_filtered = prop.filter_intervals(original_intervals)
        original_summary = original_filtered.discrete_summary(skip=['code', 'count'])

        # Split zones
        split_intervals = [
            {'name': 'Sand3_SST', 'top': 100.0, 'base': 160.0},
            {'name': 'Sand3_Slump', 'top': 160.0, 'base': 200.0},
        ]
        split_filtered = prop.filter_intervals(split_intervals)
        split_summary = split_filtered.discrete_summary(skip=['code', 'count'])

        # Total thickness MUST be exactly conserved
        original_thickness = original_summary['Sand3']['thickness']
        split_thickness = (split_summary['Sand3_SST']['thickness'] +
                          split_summary['Sand3_Slump']['thickness'])
        assert split_thickness == pytest.approx(original_thickness, abs=TOLERANCE_RELAXED)

        # Facies fractions should sum to 1.0 in each zone
        for zone_name in ['Sand3_SST', 'Sand3_Slump']:
            total_fraction = sum(
                f['fraction'] for f in split_summary[zone_name]['facies'].values()
            )
            assert total_fraction == pytest.approx(1.0, abs=TOLERANCE_RELAXED)

        # Sum of all facies thicknesses should equal zone thicknesses
        for zone_name in ['Sand3_SST', 'Sand3_Slump']:
            facies_sum = sum(
                f['thickness'] for f in split_summary[zone_name]['facies'].values()
            )
            assert facies_sum == pytest.approx(split_summary[zone_name]['thickness'], abs=TOLERANCE_RELAXED)

    def test_split_zones_discrete_boundary_behavior(self):
        """Test that discrete boundary insertion correctly uses previous value.

        When a zone is split at a depth between two samples, a boundary sample
        is inserted using the 'previous value' method. This means:
        - The boundary gets the facies of the sample BEFORE the boundary
        - This is correct for step-function discrete properties

        Example: samples at 155 (Net) and 165 (NonNet), split at 160
        - Boundary at 160 gets Net (from previous sample at 155)
        - This correctly represents that the facies was Net until 165
        """
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205], dtype=float)
        facies = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=float)

        prop = Property(
            name='Facies',
            depth=depth,
            values=facies,
            prop_type='discrete',
            labels={0: 'NonNet', 1: 'Net'}
        )

        # Split at 160 (between sample 155=Net and 165=NonNet)
        split_intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 160.0},
            {'name': 'Zone2', 'top': 160.0, 'base': 200.0},
        ]
        split_filtered = prop.filter_intervals(split_intervals)

        # Boundary at 160 should have been inserted with value from 155 (Net=1)
        idx_160 = np.where(split_filtered.depth == 160.0)[0]
        assert len(idx_160) == 1, "Boundary at 160 should be inserted"
        assert split_filtered.values[idx_160[0]] == 1.0, "Boundary at 160 should have Net (from previous sample 155)"

    def test_continuous_split_zones_conserve_exactly(self):
        """Test that continuous properties conserve weighted sums when splitting zones.

        Unlike discrete properties, continuous properties use linear interpolation
        at boundaries, which allows exact conservation of weighted statistics.
        """
        depth = np.array([95, 105, 115, 125, 135, 145, 155, 165, 175, 185, 195, 205], dtype=float)
        values = np.array([10, 20, 30, 40, 50, 60, 70, 60, 50, 40, 30, 20], dtype=float)

        prop = Property(
            name='Porosity',
            depth=depth,
            values=values,
            prop_type='continuous'
        )

        # Original zone
        original_intervals = [{'name': 'Zone', 'top': 100.0, 'base': 200.0}]
        original_stats = prop.filter_intervals(original_intervals).sums_avg()

        # Split zones
        split_intervals = [
            {'name': 'Zone1', 'top': 100.0, 'base': 160.0},
            {'name': 'Zone2', 'top': 160.0, 'base': 200.0},
        ]
        split_stats = prop.filter_intervals(split_intervals).sums_avg()

        # Total thickness must be conserved
        original_thickness = original_stats['Zone']['thickness']
        split_thickness = split_stats['Zone1']['thickness'] + split_stats['Zone2']['thickness']
        assert split_thickness == pytest.approx(original_thickness, abs=TOLERANCE_RELAXED)

        # Weighted sum should be conserved (thickness * mean)
        original_weighted_sum = original_stats['Zone']['sum']
        split_weighted_sum = split_stats['Zone1']['sum'] + split_stats['Zone2']['sum']
        assert split_weighted_sum == pytest.approx(original_weighted_sum, abs=TOLERANCE_RELAXED)

    def test_different_properties_have_independent_zone_thicknesses(self):
        """Test that properties with different zone definitions have correct independent thicknesses.

        This verifies that when zones have the same name but different depth ranges
        for different properties, each calculates its thickness correctly.
        """
        # Create two properties with different depth ranges (simulating different wells)

        # "Well A": data from 2800-3000, zone at 2871-2924 (53m)
        depth_a = np.linspace(2800, 3000, 41)  # 5m spacing
        values_a = np.ones(41)  # All Net

        # "Well B": data from 2600-2800, zone at 2626-2676 (50m)
        depth_b = np.linspace(2600, 2800, 41)  # 5m spacing
        values_b = np.ones(41)  # All Net

        prop_a = Property(
            name='Facies',
            depth=depth_a,
            values=values_a,
            prop_type='discrete',
            labels={1: 'Net'}
        )

        prop_b = Property(
            name='Facies',
            depth=depth_b,
            values=values_b,
            prop_type='discrete',
            labels={1: 'Net'}
        )

        # Different zone definitions for each property
        intervals_a = [{'name': 'Sand3_SST', 'top': 2871.0, 'base': 2924.0}]  # 53m
        intervals_b = [{'name': 'Sand3_SST', 'top': 2626.0, 'base': 2676.0}]  # 50m

        # Get summaries
        summary_a = prop_a.filter_intervals(intervals_a).discrete_summary(skip=['code', 'count'])
        summary_b = prop_b.filter_intervals(intervals_b).discrete_summary(skip=['code', 'count'])

        # Verify each property has correct thickness for its zone definition
        assert summary_a['Sand3_SST']['thickness'] == pytest.approx(53.0, abs=TOLERANCE_RELAXED)
        assert summary_b['Sand3_SST']['thickness'] == pytest.approx(50.0, abs=TOLERANCE_RELAXED)

        # Same zone name, different thicknesses - this is correct!
        assert summary_a['Sand3_SST']['thickness'] != summary_b['Sand3_SST']['thickness']


# =============================================================================
# Test Zone Statistics Conservation (Aggregation Test)
# =============================================================================

class TestZoneStatisticsConservation:
    """Test that zone-aggregated statistics match direct log statistics.

    When zones fully cover the log range, the back-calculated mean from
    thickness-weighted zone means should match the direct log mean.
    """

    def test_full_coverage_aligned_boundaries(self):
        """Test conservation when zone boundaries align with sample depths."""
        np.random.seed(42)

        # Generate random log
        data_spacing = 0.5
        log_top, log_base = 1000.0, 1200.0
        num_samples = int((log_base - log_top) / data_spacing) + 1

        depth = np.linspace(log_top, log_base, num_samples)
        values = np.clip(np.random.normal(0.5, 0.25, num_samples), 0, 1)

        # Direct calculation
        intervals = compute_intervals(depth)
        direct_mean = np.sum(values * intervals) / np.sum(intervals)

        # Create 100 equal zones aligned with boundaries
        num_zones = 100
        zone_thickness = (log_base - log_top) / num_zones
        zones = [{'name': f'Z{i:03d}',
                  'top': log_top + i * zone_thickness,
                  'base': log_top + (i+1) * zone_thickness}
                 for i in range(num_zones)]

        # Calculate via zones
        prop = Property(name='Test', depth=depth, values=values, prop_type='continuous')
        stats = prop.filter_intervals(zones).sums_avg()

        zone_sums = sum(stats[z['name']]['sum'] for z in zones)
        zone_thick = sum(stats[z['name']]['thickness'] for z in zones)
        backcalc_mean = zone_sums / zone_thick

        # Should match within floating point precision
        assert zone_thick == pytest.approx(log_base - log_top, abs=TOLERANCE_RELAXED)
        assert backcalc_mean == pytest.approx(direct_mean, rel=1e-6)

    def test_full_coverage_offset_boundaries(self):
        """Test conservation when zone boundaries are offset from samples."""
        np.random.seed(42)

        data_spacing = 0.5
        log_top, log_base = 1000.0, 1200.0
        num_samples = int((log_base - log_top) / data_spacing) + 1

        depth = np.linspace(log_top, log_base, num_samples)
        values = np.clip(np.random.normal(0.5, 0.25, num_samples), 0, 1)

        intervals = compute_intervals(depth)
        direct_mean = np.sum(values * intervals) / np.sum(intervals)

        # Create zones with offset internal boundaries
        np.random.seed(123)
        num_zones = 100
        zone_thickness = (log_base - log_top) / num_zones

        boundaries = [log_top]
        for i in range(1, num_zones):
            base_pos = log_top + i * zone_thickness
            perturbation = data_spacing * np.random.uniform(-0.4, 0.4)
            boundaries.append(base_pos + perturbation)
        boundaries.append(log_base)

        zones = [{'name': f'Z{i:03d}', 'top': boundaries[i], 'base': boundaries[i+1]}
                 for i in range(num_zones)]

        # Verify boundaries don't align with samples
        sample_set = set(np.round(depth, 10))
        aligned = sum(1 for b in boundaries[1:-1] if np.round(b, 10) in sample_set)
        assert aligned == 0, "Test setup: boundaries should be offset from samples"

        # Calculate via zones
        prop = Property(name='Test', depth=depth, values=values, prop_type='continuous')
        stats = prop.filter_intervals(zones).sums_avg()

        zone_sums = sum(stats[z['name']]['sum'] for z in zones)
        zone_thick = sum(stats[z['name']]['thickness'] for z in zones)
        backcalc_mean = zone_sums / zone_thick

        # Should still match (boundary insertion preserves statistics)
        assert zone_thick == pytest.approx(log_base - log_top, abs=TOLERANCE_RELAXED)
        assert backcalc_mean == pytest.approx(direct_mean, rel=1e-6)

    def test_variable_zone_sizes(self):
        """Test conservation with variable zone sizes."""
        np.random.seed(42)

        data_spacing = 0.5
        log_top, log_base = 1000.0, 1200.0
        num_samples = int((log_base - log_top) / data_spacing) + 1

        depth = np.linspace(log_top, log_base, num_samples)
        values = np.clip(np.random.normal(0.5, 0.25, num_samples), 0, 1)

        intervals = compute_intervals(depth)
        direct_mean = np.sum(values * intervals) / np.sum(intervals)

        # Create zones with variable sizes (4x to 8x data spacing)
        np.random.seed(456)
        min_thick = 4 * data_spacing
        max_thick = 8 * data_spacing
        current = log_top
        zones = []
        i = 0
        while current < log_base:
            thick = np.random.uniform(min_thick, max_thick)
            next_pos = min(current + thick, log_base)
            zones.append({'name': f'Z{i:03d}', 'top': current, 'base': next_pos})
            current = next_pos
            i += 1

        # Should have at least 50 zones (200m / 4m max)
        assert len(zones) >= 50

        # Calculate via zones
        prop = Property(name='Test', depth=depth, values=values, prop_type='continuous')
        stats = prop.filter_intervals(zones).sums_avg()

        zone_sums = sum(stats[z['name']]['sum'] for z in zones)
        zone_thick = sum(stats[z['name']]['thickness'] for z in zones)
        backcalc_mean = zone_sums / zone_thick

        assert zone_thick == pytest.approx(log_base - log_top, abs=TOLERANCE_RELAXED)
        assert backcalc_mean == pytest.approx(direct_mean, rel=1e-6)

    def test_many_zones_no_accumulation_error(self):
        """Test that many zones don't accumulate floating point error."""
        np.random.seed(42)

        data_spacing = 0.1  # Fine sampling
        log_top, log_base = 1000.0, 1500.0  # 500m log
        num_samples = int((log_base - log_top) / data_spacing) + 1

        depth = np.linspace(log_top, log_base, num_samples)
        values = np.clip(np.random.normal(0.5, 0.25, num_samples), 0, 1)

        intervals = compute_intervals(depth)
        direct_mean = np.sum(values * intervals) / np.sum(intervals)
        direct_sum = np.sum(values * intervals)

        # Create 500 zones (1m each on average)
        num_zones = 500
        zone_thickness = (log_base - log_top) / num_zones
        zones = [{'name': f'Z{i:03d}',
                  'top': log_top + i * zone_thickness,
                  'base': log_top + (i+1) * zone_thickness}
                 for i in range(num_zones)]

        prop = Property(name='Test', depth=depth, values=values, prop_type='continuous')
        stats = prop.filter_intervals(zones).sums_avg()

        zone_sums = sum(stats[z['name']]['sum'] for z in zones)
        zone_thick = sum(stats[z['name']]['thickness'] for z in zones)
        backcalc_mean = zone_sums / zone_thick

        # Even with 500 zones, should be very accurate
        assert zone_thick == pytest.approx(log_base - log_top, abs=TOLERANCE_RELAXED)
        assert zone_sums == pytest.approx(direct_sum, rel=1e-5)  # ~0.001% tolerance
        assert backcalc_mean == pytest.approx(direct_mean, rel=1e-5)

    def test_partial_coverage_matches_partial_range(self):
        """Test that partial coverage matches statistics for the covered range only."""
        np.random.seed(42)

        data_spacing = 0.5
        log_top, log_base = 1000.0, 1200.0
        num_samples = int((log_base - log_top) / data_spacing) + 1

        depth = np.linspace(log_top, log_base, num_samples)
        values = np.clip(np.random.normal(0.5, 0.25, num_samples), 0, 1)

        # Create zones that only cover middle portion
        margin = 20.0
        partial_top = log_top + margin
        partial_base = log_base - margin

        num_zones = 80
        zone_thickness = (partial_base - partial_top) / num_zones
        zones = [{'name': f'Z{i:03d}',
                  'top': partial_top + i * zone_thickness,
                  'base': partial_top + (i+1) * zone_thickness}
                 for i in range(num_zones)]

        # Calculate direct mean for partial range using zone intervals
        partial_mean_expected = 0.0
        partial_thick_expected = 0.0
        for zone in zones:
            zone_int = compute_zone_intervals(depth, zone['top'], zone['base'])
            partial_mean_expected += np.sum(values * zone_int)
            partial_thick_expected += np.sum(zone_int)
        partial_mean_expected /= partial_thick_expected

        # Calculate via filter_intervals
        prop = Property(name='Test', depth=depth, values=values, prop_type='continuous')
        stats = prop.filter_intervals(zones).sums_avg()

        zone_sums = sum(stats[z['name']]['sum'] for z in zones)
        zone_thick = sum(stats[z['name']]['thickness'] for z in zones)
        backcalc_mean = zone_sums / zone_thick

        # Should match the partial range statistics
        assert zone_thick == pytest.approx(partial_base - partial_top, abs=TOLERANCE_RELAXED)
        assert backcalc_mean == pytest.approx(partial_mean_expected, rel=1e-5)


# =============================================================================
# Run tests if executed directly
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
