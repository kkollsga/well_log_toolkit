"""
Test script to verify that discrete properties (like Well_Tops) are properly
handled when used for shape or color mapping in crossplots.

This test verifies that:
1. Discrete properties use nearest-neighbor interpolation (not linear)
2. Shape labels are actual well top names, not fractional values
3. Color codes are preserved integers, not interpolated decimals
"""

import numpy as np
from well_log_toolkit.visualization import Crossplot


def create_well_with_tops():
    """Create a mock well with continuous logs and sparse discrete tops."""
    from well_log_toolkit.core.property import Property

    class MockWell:
        def __init__(self, name):
            self.name = name

            # Fine grid continuous properties (0.5m sampling)
            fine_depth = np.arange(2800, 3100, 0.5)  # 600 samples
            self._properties = {
                'CorePor': Property(
                    name='CorePor',
                    depth=fine_depth,
                    values=0.15 + np.random.rand(len(fine_depth)) * 0.1,
                    prop_type='continuous'
                ),
                'CorePerm': Property(
                    name='CorePerm',
                    depth=fine_depth,
                    values=10 + np.random.rand(len(fine_depth)) * 50,
                    prop_type='continuous'
                )
            }

            # Sparse discrete well tops (like real data - only at formation boundaries)
            tops_depths = np.array([
                2882.96,
                2929.93,
                2955.10,
                2979.79,
                2999.30,
                3073.18
            ])

            # Integer codes for each top
            tops_values = np.array([0, 1, 2, 3, 4, 5], dtype=float)

            # Labels mapping
            tops_labels = {
                0: 'Agat top',
                1: 'Cerisa Main top',
                2: 'Cerisa West SST 1 top',
                3: 'Agat fm base',
                4: 'Sola fm top',
                5: 'Åsgård fm top'
            }

            self._properties['Well_Tops'] = Property(
                name='Well_Tops',
                depth=tops_depths,
                values=tops_values,
                prop_type='discrete',
                labels=tops_labels
            )

        def get_property(self, name):
            if name in self._properties:
                return self._properties[name]
            from well_log_toolkit.exceptions import PropertyNotFoundError
            raise PropertyNotFoundError(f"Property {name} not found")

    return MockWell("Well_A")


def test_discrete_shape_alignment():
    """Test that discrete properties used for shape preserve integer codes."""
    print("\n" + "="*70)
    print("TEST 1: Discrete property shape alignment")
    print("="*70)

    well = create_well_with_tops()

    print("\nWell_Tops property info:")
    tops = well.get_property('Well_Tops')
    print(f"  Type: {tops.type}")
    print(f"  Depth samples: {len(tops.depth)}")
    print(f"  Unique values: {np.unique(tops.values[~np.isnan(tops.values)])}")
    print(f"  Labels: {tops.labels}")

    print("\nContinuous property info:")
    por = well.get_property('CorePor')
    print(f"  CorePor depth samples: {len(por.depth)}")

    try:
        # Create crossplot with discrete shape
        plot = Crossplot(
            wells=[well],
            x="CorePor",
            y="CorePerm",
            shape="Well_Tops",
            title="Crossplot with Discrete Shape (Well Tops)"
        )

        print("\n✓ Crossplot created successfully")

        # Prepare data
        data = plot._prepare_data()
        print(f"✓ Data prepared: {len(data)} points")

        # Check shape values
        if 'shape_val' in data.columns:
            shape_values = data['shape_val'].values
            unique_shapes = np.unique(shape_values[~np.isnan(shape_values)])

            print(f"\nShape values analysis:")
            print(f"  Total data points: {len(shape_values)}")
            print(f"  Unique shape values: {unique_shapes}")
            print(f"  Data type: {shape_values.dtype}")

            # Check if values are integers (or very close to integers)
            non_nan_shapes = shape_values[~np.isnan(shape_values)]
            if len(non_nan_shapes) > 0:
                # Check if all values are close to integers
                is_integer = np.allclose(non_nan_shapes, np.round(non_nan_shapes), atol=1e-6)

                if is_integer:
                    print(f"  ✓ PASS: Shape values are integers (discrete)")

                    # Verify they're in the expected range [0, 5]
                    min_val = np.min(non_nan_shapes)
                    max_val = np.max(non_nan_shapes)
                    print(f"  ✓ Shape value range: [{min_val:.0f}, {max_val:.0f}]")

                    if min_val >= 0 and max_val <= 5:
                        print(f"  ✓ PASS: Shape values in expected range")
                    else:
                        print(f"  ✗ FAIL: Shape values outside expected range [0, 5]")
                        return False
                else:
                    print(f"  ✗ FAIL: Shape values are NOT integers (fractional):")
                    print(f"      Sample values: {non_nan_shapes[:10]}")
                    return False
            else:
                print(f"  ✗ FAIL: No valid shape values found")
                return False
        else:
            print(f"✗ FAIL: shape_val column not found")
            return False

    except Exception as e:
        print(f"✗ FAIL: Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✓ TEST 1 PASSED\n")
    return True


def test_discrete_vs_continuous_alignment():
    """Compare discrete vs continuous property alignment behavior."""
    print("\n" + "="*70)
    print("TEST 2: Discrete forward-fill (previous) vs continuous interpolation")
    print("="*70)

    from well_log_toolkit.core.property import Property

    # Create a discrete property with sparse samples (like well tops)
    discrete_depth = np.array([2800.0, 2850.0, 2900.0, 2950.0, 3000.0])
    discrete_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    discrete_prop = Property(
        name='Discrete',
        depth=discrete_depth,
        values=discrete_values,
        prop_type='discrete'
    )

    # Create a continuous property with same sparse samples
    continuous_prop = Property(
        name='Continuous',
        depth=discrete_depth.copy(),
        values=discrete_values.copy(),
        prop_type='continuous'
    )

    # Target fine grid
    target_depth = np.arange(2800, 3001, 10)  # Every 10m

    print(f"Source property depths: {discrete_depth}")
    print(f"Source property values: {discrete_values}")
    print(f"  (e.g., 2850.0 marks start of zone '1', which extends until 2900.0)")
    print(f"\nTarget depth grid: {len(target_depth)} samples from {target_depth[0]} to {target_depth[-1]}")

    # Resample discrete property (should use forward-fill/previous)
    discrete_resampled = discrete_prop.resample(target_depth)

    # Resample continuous property (should use linear interpolation)
    continuous_resampled = continuous_prop.resample(target_depth)

    print(f"\nDiscrete resampled values (forward-fill):")
    print(f"  {discrete_resampled.values}")
    print(f"  All integer? {np.allclose(discrete_resampled.values[~np.isnan(discrete_resampled.values)], np.round(discrete_resampled.values[~np.isnan(discrete_resampled.values)]))}")

    print(f"\nContinuous resampled values (linear interpolation):")
    print(f"  {continuous_resampled.values}")

    # Check that discrete values are integers
    discrete_non_nan = discrete_resampled.values[~np.isnan(discrete_resampled.values)]
    if len(discrete_non_nan) > 0:
        is_integer = np.allclose(discrete_non_nan, np.round(discrete_non_nan), atol=1e-6)
        if is_integer:
            print(f"\n  ✓ PASS: Discrete property preserved integer values")
        else:
            print(f"  ✗ FAIL: Discrete property has fractional values")
            return False

    # CRITICAL TEST: Check forward-fill behavior
    # At depth 2840 (between 2800 and 2850), should have value 0 (previous zone)
    # NOT value 1 (which would be nearest neighbor to 2850)
    idx_2840 = np.argmin(np.abs(target_depth - 2840))
    val_2840 = discrete_resampled.values[idx_2840]
    print(f"\n  Depth 2840m (between 2800 and 2850):")
    print(f"    Discrete value: {val_2840:.0f}")
    if val_2840 == 0:
        print(f"    ✓ PASS: Uses PREVIOUS zone (0), not nearest (would be 1)")
    else:
        print(f"    ✗ FAIL: Should be 0 (previous), got {val_2840:.0f}")
        return False

    # At depth 2890 (between 2850 and 2900), should have value 1 (previous zone)
    idx_2890 = np.argmin(np.abs(target_depth - 2890))
    val_2890 = discrete_resampled.values[idx_2890]
    print(f"\n  Depth 2890m (between 2850 and 2900):")
    print(f"    Discrete value: {val_2890:.0f}")
    if val_2890 == 1:
        print(f"    ✓ PASS: Uses PREVIOUS zone (1), not nearest (would be 2)")
    else:
        print(f"    ✗ FAIL: Should be 1 (previous), got {val_2890:.0f}")
        return False

    # Check that continuous values have interpolated values
    continuous_non_nan = continuous_resampled.values[~np.isnan(continuous_resampled.values)]
    if len(continuous_non_nan) > 0:
        # At depth 2825, should have interpolated value between 0 and 1
        idx_2825 = np.argmin(np.abs(target_depth - 2825))
        val_2825 = continuous_resampled.values[idx_2825]
        print(f"\n  Continuous value at 2825m (between 2800 and 2850): {val_2825:.2f}")
        if 0 < val_2825 < 1:
            print(f"    ✓ PASS: Continuous property interpolated between samples")
        else:
            print(f"    ✗ FAIL: Continuous property should interpolate")
            return False

    print(f"\n✓ TEST 2 PASSED\n")
    return True


def test_well_tops_with_labels():
    """Test that well top labels are preserved and accessible."""
    print("\n" + "="*70)
    print("TEST 3: Well tops labels preservation")
    print("="*70)

    well = create_well_with_tops()
    tops = well.get_property('Well_Tops')

    print(f"Original Well_Tops labels:")
    for code, label in tops.labels.items():
        print(f"  {code}: {label}")

    # Resample to a fine grid
    fine_grid = np.arange(2880, 3080, 5)  # Every 5m
    resampled = tops.resample(fine_grid)

    print(f"\nResampled to {len(fine_grid)} depths")
    print(f"Unique values in resampled: {np.unique(resampled.values[~np.isnan(resampled.values)])}")

    # Check that labels are preserved
    if resampled.labels == tops.labels:
        print(f"✓ PASS: Labels preserved after resampling")
    else:
        print(f"✗ FAIL: Labels changed after resampling")
        return False

    # Check that we can map values to labels
    sample_value = resampled.values[~np.isnan(resampled.values)][0]
    if not np.isnan(sample_value):
        sample_int = int(np.round(sample_value))
        if sample_int in resampled.labels:
            label = resampled.labels[sample_int]
            print(f"✓ PASS: Can map value {sample_int} to label '{label}'")
        else:
            print(f"✗ FAIL: Value {sample_int} not in labels")
            return False

    print(f"\n✓ TEST 3 PASSED\n")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISCRETE PROPERTY ALIGNMENT TESTS")
    print("="*70)

    all_passed = True

    # Run tests
    all_passed &= test_discrete_shape_alignment()
    all_passed &= test_discrete_vs_continuous_alignment()
    all_passed &= test_well_tops_with_labels()

    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Discrete properties handled correctly!")
        print("="*70)
        print("\nThe fix successfully:")
        print("  1. Uses forward-fill (previous) interpolation for discrete properties")
        print("  2. Preserves integer codes (no fractional values)")
        print("  3. Maintains label mappings for well tops")
        print("  4. Geological zones extend from their top until next boundary")
        print("  5. Distinguishes between discrete and continuous properties")
    else:
        print("✗ SOME TESTS FAILED - Review implementation")
        print("="*70)

    print()
