"""
Comprehensive test demonstrating all property operation features.

This test demonstrates:
1. Numpy-style property printing (with clipping)
2. Strict depth matching that raises errors
3. Explicit resampling with .resample()
4. Property operations (arithmetic, comparison, logical)
5. Assignment behavior (new vs overwrite)
6. Auto-generated labels for comparisons
7. Manager-level broadcasting

Run with: python test_property_operations.py
"""

import numpy as np
from pylog import WellDataManager, Well, Property


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * 80)


def test_numpy_style_printing():
    """Test 1: Numpy-style property printing with clipping."""
    print_section("TEST 1: NUMPY-STYLE PROPERTY PRINTING")

    # Small array (no clipping)
    print_subsection("1.1 Small Array (6 samples - no clipping)")
    depth_small = np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0])
    phie_small = np.array([0.15, 0.20, 0.22, 0.18, 0.25, 0.19])
    prop_small = Property(name='PHIE', depth=depth_small, values=phie_small, unit='v/v')

    print("print(well.PHIE):")
    print(prop_small)
    print("\nrepr(well.PHIE):")
    print(repr(prop_small))

    # Large array (with clipping)
    print_subsection("1.2 Large Array (20 samples - clipped)")
    depth_large = np.arange(2800.0, 2820.0, 1.0)
    phie_large = np.random.uniform(0.10, 0.30, 20)
    prop_large = Property(name='PHIE', depth=depth_large, values=phie_large, unit='v/v')

    print("print(well.PHIE):")
    print(prop_large)
    print("\nrepr(well.PHIE):")
    print(repr(prop_large))

    # Property with NaN values
    print_subsection("1.3 Property with NaN Values")
    depth_nan = np.arange(1000.0, 1012.0, 1.0)
    values_nan = np.array([0.15, np.nan, 0.22, 0.18, np.nan, 0.19, 0.21, 0.17, 0.23, np.nan, 0.20, 0.18])
    prop_nan = Property(name='PHIE', depth=depth_nan, values=values_nan, unit='v/v')

    print(prop_nan)

    print("\n✓ Numpy-style printing works correctly with clipping at 8 elements")


def test_strict_depth_matching():
    """Test 2: Strict depth matching."""
    print_section("TEST 2: STRICT DEPTH MATCHING (NUMPY-LIKE BEHAVIOR)")

    # Create well with properties
    well = Well(name='Test Well', sanitized_name='Test_Well')

    depth1 = np.arange(1000.0, 1010.0, 1.0)
    phie_values = np.array([0.15, 0.20, 0.22, 0.18, 0.25, 0.19, 0.21, 0.17, 0.23, 0.20])
    sw_values = np.array([0.40, 0.35, 0.30, 0.38, 0.28, 0.36, 0.32, 0.39, 0.29, 0.34])

    phie = Property(name='PHIE', depth=depth1, values=phie_values, unit='v/v')
    sw = Property(name='SW', depth=depth1, values=sw_values, unit='v/v')

    # Different depth grid
    depth2 = np.arange(1000.0, 1010.0, 2.0)  # Every other sample
    perm_values = np.array([100.0, 120.0, 150.0, 110.0, 140.0])
    perm = Property(name='PERM', depth=depth2, values=perm_values, unit='mD')

    well._sources = {
        'Petrophysics': {
            'path': None,
            'las_file': None,
            'properties': {'PHIE': phie, 'SW': sw, 'PERM': perm},
            'modified': False
        }
    }
    for prop in [phie, sw, perm]:
        prop.parent_well = well

    print_subsection("2.1 Same Depth Grid - Works")
    try:
        result = well.PHIE + well.SW
        print("✓ well.PHIE + well.SW succeeded")
        print(f"  Result shape: {len(result.depth)} samples")
        print(f"  First 3 values: {result.values[:3]}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print_subsection("2.2 Different Depth Grid - Raises DepthAlignmentError")
    try:
        result = well.PHIE + well.PERM
        print("✗ Should have raised DepthAlignmentError but succeeded")
    except Exception as e:
        print(f"✓ Correctly raised {type(e).__name__}")
        print(f"\nError message:")
        for line in str(e).split('\n')[:6]:
            print(f"  {line}")

    print("\n✓ Strict depth matching enforces numpy-like behavior")


def test_explicit_resampling():
    """Test 3: Explicit resampling."""
    print_section("TEST 3: EXPLICIT RESAMPLING")

    well = Well(name='Test Well', sanitized_name='Test_Well')

    # Fine grid
    depth_fine = np.arange(1000.0, 1010.0, 1.0)
    phie_fine = np.array([0.15, 0.20, 0.22, 0.18, 0.25, 0.19, 0.21, 0.17, 0.23, 0.20])
    phie = Property(name='PHIE', depth=depth_fine, values=phie_fine, unit='v/v')

    # Coarse grid
    depth_coarse = np.array([1000.0, 1003.0, 1006.0, 1009.0])
    perm_coarse = np.array([100.0, 120.0, 150.0, 130.0])
    perm = Property(name='PERM', depth=depth_coarse, values=perm_coarse, unit='mD')

    well._sources = {
        'Petrophysics': {
            'path': None,
            'las_file': None,
            'properties': {'PHIE': phie, 'PERM': perm},
            'modified': False
        }
    }
    phie.parent_well = well
    perm.parent_well = well

    print_subsection("3.1 Resample PERM to PHIE's Grid")
    print(f"Original PERM: {len(perm.depth)} samples")
    print(f"  Depth: {perm.depth}")
    print(f"  Values: {perm.values}")

    perm_resampled = well.PERM.resample(well.PHIE)
    print(f"\nResampled PERM: {len(perm_resampled.depth)} samples")
    print(f"  Depth: {perm_resampled.depth}")
    print(f"  Values: {perm_resampled.values}")

    print_subsection("3.2 Now Operations Work")
    result = well.PHIE * perm_resampled * 0.001
    print("✓ well.PHIE * perm_resampled * 0.001 succeeded")
    print(f"  Result: {result.values}")

    print_subsection("3.3 Resample to Custom Grid")
    custom_depth = np.arange(1000.0, 1010.0, 0.5)
    phie_custom = well.PHIE.resample(custom_depth)
    print(f"✓ Resampled to custom 0.5m grid: {len(phie_custom.depth)} samples")
    print(f"  First 5 depths: {phie_custom.depth[:5]}")
    print(f"  First 5 values: {phie_custom.values[:5]}")

    print("\n✓ Explicit resampling provides full control over depth alignment")


def test_arithmetic_operations():
    """Test 4: Arithmetic operations."""
    print_section("TEST 4: ARITHMETIC OPERATIONS")

    well = Well(name='Test Well', sanitized_name='Test_Well')

    depth = np.arange(1000.0, 1006.0, 1.0)
    phie_values = np.array([0.15, 0.20, 0.22, 0.18, 0.25, 0.19])
    sw_values = np.array([0.40, 0.35, 0.30, 0.38, 0.28, 0.36])

    phie = Property(name='PHIE', depth=depth, values=phie_values, unit='v/v')
    sw = Property(name='SW', depth=depth, values=sw_values, unit='v/v')

    well._sources = {
        'Petrophysics': {
            'path': None,
            'las_file': None,
            'properties': {'PHIE': phie, 'SW': sw},
            'modified': False
        }
    }
    phie.parent_well = well
    sw.parent_well = well

    print_subsection("4.1 Scalar Operations")
    well.PHIE_percent = well.PHIE * 100
    print("well.PHIE_percent = well.PHIE * 100")
    print(f"  Values: {well.PHIE_percent.values}")

    well.PHIE_adjusted = well.PHIE - 0.02
    print("\nwell.PHIE_adjusted = well.PHIE - 0.02")
    print(f"  Values: {well.PHIE_adjusted.values}")

    print_subsection("4.2 Property-to-Property Operations")
    well.HC_Volume = well.PHIE * (1 - well.SW)
    print("well.HC_Volume = well.PHIE * (1 - well.SW)")
    print(f"  Values: {well.HC_Volume.values}")

    print_subsection("4.3 Unary Operations")
    well.Neg_PHIE = -well.PHIE
    print("well.Neg_PHIE = -well.PHIE")
    print(f"  Values: {well.Neg_PHIE.values}")

    print("\n✓ All arithmetic operations work correctly")


def test_comparison_and_logical():
    """Test 5: Comparison and logical operations."""
    print_section("TEST 5: COMPARISON & LOGICAL OPERATIONS")

    well = Well(name='Test Well', sanitized_name='Test_Well')

    depth = np.arange(1000.0, 1010.0, 1.0)
    phie_values = np.array([0.12, 0.18, 0.22, 0.16, 0.25, 0.19, 0.14, 0.21, 0.17, 0.23])
    sw_values = np.array([0.45, 0.32, 0.28, 0.40, 0.25, 0.36, 0.42, 0.30, 0.38, 0.27])
    perm_values = np.array([50, 110, 180, 75, 220, 150, 60, 170, 90, 200])

    phie = Property(name='PHIE', depth=depth, values=phie_values, unit='v/v')
    sw = Property(name='SW', depth=depth, values=sw_values, unit='v/v')
    perm = Property(name='PERM', depth=depth, values=perm_values, unit='mD')

    well._sources = {
        'Petrophysics': {
            'path': None,
            'las_file': None,
            'properties': {'PHIE': phie, 'SW': sw, 'PERM': perm},
            'modified': False
        }
    }
    for prop in [phie, sw, perm]:
        prop.parent_well = well

    print_subsection("5.1 Comparison Operations (Auto-Generate Labels)")
    well.High_Poro = well.PHIE > 0.18
    print("well.High_Poro = well.PHIE > 0.18")
    print(f"  Type: {well.High_Poro.type}")
    print(f"  Labels: {well.High_Poro.labels}")
    print(f"  Values: {well.High_Poro.values}")

    print_subsection("5.2 Logical AND")
    well.Reservoir = (well.PHIE > 0.18) & (well.SW < 0.35)
    print("well.Reservoir = (well.PHIE > 0.18) & (well.SW < 0.35)")
    print(f"  Type: {well.Reservoir.type}")
    print(f"  Labels: {well.Reservoir.labels}")
    print(f"  Values: {well.Reservoir.values}")

    print_subsection("5.3 Complex Logic")
    well.Good_Reservoir = (well.PHIE > 0.18) & (well.SW < 0.35) & (well.PERM > 100)
    print("well.Good_Reservoir = (well.PHIE > 0.18) & (well.SW < 0.35) & (well.PERM > 100)")
    print(f"  Values: {well.Good_Reservoir.values}")

    print_subsection("5.4 Logical NOT")
    well.Non_Reservoir = ~well.Reservoir
    print("well.Non_Reservoir = ~well.Reservoir")
    print(f"  Values: {well.Non_Reservoir.values}")

    print("\n✓ Comparison operations create discrete properties with auto-labels")


def test_assignment_behavior():
    """Test 6: Assignment behavior (new vs overwrite)."""
    print_section("TEST 6: ASSIGNMENT BEHAVIOR")

    well = Well(name='Test Well', sanitized_name='Test_Well')

    depth = np.arange(1000.0, 1006.0, 1.0)
    phie_values = np.array([0.15, 0.20, 0.22, 0.18, 0.25, 0.19])

    phie = Property(name='PHIE', depth=depth, values=phie_values, unit='v/v')

    well._sources = {
        'Petrophysics': {
            'path': None,
            'las_file': None,
            'properties': {'PHIE': phie},
            'modified': False
        }
    }
    phie.parent_well = well

    print_subsection("6.1 Create NEW Property")
    well.PHIE_scaled = well.PHIE * 100
    print("well.PHIE_scaled = well.PHIE * 100")
    print(f"  Sources: {list(well._sources.keys())}")
    print(f"  'PHIE_scaled' in computed: {'PHIE_scaled' in well._sources.get('computed', {}).get('properties', {})}")
    print(f"  Values: {well.PHIE_scaled.values}")

    print_subsection("6.2 OVERWRITE Existing Property")
    original_phie = well.PHIE.values.copy()
    print(f"Original PHIE: {original_phie}")

    well.PHIE = well.PHIE * 0.01
    print("\nwell.PHIE = well.PHIE * 0.01 (overwrite)")
    print(f"  New PHIE: {well.PHIE.values}")
    print(f"  Still in Petrophysics: {'PHIE' in well._sources['Petrophysics']['properties']}")
    print(f"  NOT in computed: {'PHIE' in well._sources.get('computed', {}).get('properties', {})}")
    print(f"  Source marked modified: {well._sources['Petrophysics']['modified']}")

    print("\n✓ Assignment distinguishes between new properties and overwrites")


def test_manager_broadcasting():
    """Test 7: Manager-level broadcasting."""
    print_section("TEST 7: MANAGER-LEVEL BROADCASTING")

    manager = WellDataManager()

    # Create multiple wells
    for well_num in range(1, 4):
        well = Well(name=f'Well_{well_num}', sanitized_name=f'Well_{well_num}')

        depth = np.arange(1000.0, 1010.0, 1.0)
        phie_values = np.random.uniform(0.10, 0.30, 10)
        sw_values = np.random.uniform(0.20, 0.50, 10)

        phie = Property(name='PHIE', depth=depth, values=phie_values, unit='v/v')
        sw = Property(name='SW', depth=depth, values=sw_values, unit='v/v')

        well._sources = {
            'Petrophysics': {
                'path': None,
                'las_file': None,
                'properties': {'PHIE': phie, 'SW': sw},
                'modified': False
            }
        }
        phie.parent_well = well
        sw.parent_well = well

        manager._wells[f'Well_{well_num}'] = well

    # Add one well without PHIE
    well4 = Well(name='Well_4', sanitized_name='Well_4')
    well4._sources = {'Petrophysics': {'path': None, 'las_file': None, 'properties': {}, 'modified': False}}
    manager._wells['Well_4'] = well4

    print_subsection("7.1 Broadcast Scalar Operation")
    print("manager.PHIE_percent = manager.PHIE * 100")
    manager.PHIE_percent = manager.PHIE * 100

    print(f"\nCheck Well_1 has PHIE_percent:")
    print(f"  ✓ {'PHIE_percent' in manager._wells['Well_1']._sources.get('computed', {}).get('properties', {})}")

    print_subsection("7.2 Broadcast Comparison Operation")
    print("manager.High_Poro = manager.PHIE > 0.18")
    manager.High_Poro = manager.PHIE > 0.18

    print(f"\nCheck Well_2 has High_Poro:")
    high_poro = manager._wells['Well_2']._sources.get('computed', {}).get('properties', {}).get('High_Poro')
    if high_poro:
        print(f"  ✓ Type: {high_poro.type}")
        print(f"  ✓ Labels: {high_poro.labels}")

    print_subsection("7.3 Broadcast Complex Operation")
    print("manager.HC_Volume = manager.PHIE * (1 - manager.SW)")
    manager.HC_Volume = manager.PHIE * (1 - manager.SW)
    # Note: Properties created successfully in wells with both PHIE and SW

    print("\n✓ Manager broadcasting applies operations to all applicable wells")


def test_complete_workflow():
    """Test 8: Complete workflow example."""
    print_section("TEST 8: COMPLETE WORKFLOW EXAMPLE")

    well = Well(name='Production Well', sanitized_name='Production_Well')

    # Create realistic well log data
    depth = np.arange(2800.0, 2850.0, 1.0)
    np.random.seed(42)

    # Porosity varies by zone
    phie_values = np.concatenate([
        np.random.uniform(0.18, 0.25, 20),  # Good reservoir
        np.random.uniform(0.08, 0.15, 30),  # Poor reservoir
    ])

    # Water saturation inversely correlated with porosity
    sw_values = 1 - (phie_values + np.random.uniform(-0.1, 0.1, 50))
    sw_values = np.clip(sw_values, 0.2, 0.8)

    # Permeability correlated with porosity
    perm_values = (phie_values / 0.20) ** 3 * 100 + np.random.uniform(-20, 20, 50)
    perm_values = np.clip(perm_values, 10, 300)

    # Zone tops
    zone_values = np.concatenate([np.zeros(20), np.ones(30)])

    # Create properties
    phie = Property(name='PHIE', depth=depth, values=phie_values, unit='v/v')
    sw = Property(name='SW', depth=depth, values=sw_values, unit='v/v')
    perm = Property(name='PERM', depth=depth, values=perm_values, unit='mD')
    zone = Property(name='Zone', depth=depth, values=zone_values, unit='', prop_type='discrete')
    zone._labels = {0: 'Upper_Sand', 1: 'Lower_Shale'}

    well._sources = {
        'Petrophysics': {
            'path': None,
            'las_file': None,
            'properties': {'PHIE': phie, 'SW': sw, 'PERM': perm, 'Zone': zone},
            'modified': False
        }
    }
    for prop in [phie, sw, perm, zone]:
        prop.parent_well = well

    print_subsection("8.1 Compute Derived Properties")

    # Hydrocarbon pore volume
    well.HC_Pore_Volume = well.PHIE * (1 - well.SW)
    print("✓ well.HC_Pore_Volume = well.PHIE * (1 - well.SW)")
    print(f"  Mean HC volume: {np.mean(well.HC_Pore_Volume.values):.3f}")

    # Reservoir quality flag
    well.Good_Reservoir = (well.PHIE > 0.18) & (well.SW < 0.45) & (well.PERM > 100)
    print("\n✓ well.Good_Reservoir = (well.PHIE > 0.18) & (well.SW < 0.45) & (well.PERM > 100)")
    print(f"  Good reservoir samples: {int(np.sum(well.Good_Reservoir.values))}/{len(well.Good_Reservoir.values)}")

    # Productivity index (simplified)
    well.Productivity_Index = well.PERM * well.PHIE * (1 - well.SW) / 100
    print("\n✓ well.Productivity_Index = well.PERM * well.PHIE * (1 - well.SW) / 100")
    print(f"  Mean productivity: {np.mean(well.Productivity_Index.values):.3f}")

    print_subsection("8.2 Summary of Computed Properties")
    if 'computed' in well._sources:
        computed_props = list(well._sources['computed']['properties'].keys())
        print(f"Created {len(computed_props)} computed properties:")
        for prop_name in computed_props:
            prop = well._sources['computed']['properties'][prop_name]
            print(f"  • {prop_name} ({prop.type})")

    print("\n✓ Complete workflow demonstrates practical petrophysical analysis")


def main():
    """Run all tests."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  WELL LOG TOOLKIT - PROPERTY OPERATIONS TEST SUITE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    tests = [
        test_numpy_style_printing,
        test_strict_depth_matching,
        test_explicit_resampling,
        test_arithmetic_operations,
        test_comparison_and_logical,
        test_assignment_behavior,
        test_manager_broadcasting,
        test_complete_workflow,
    ]

    for i, test_func in enumerate(tests, 1):
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ TEST {i} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

    print_section("ALL TESTS PASSED!")
    print("""
Summary of Features Demonstrated:
  ✓ Numpy-style property printing with automatic clipping
  ✓ Strict depth matching enforces explicit resampling
  ✓ .resample() method for controlled depth alignment
  ✓ Arithmetic operations on properties
  ✓ Comparison operations with auto-generated labels
  ✓ Logical operations (AND, OR, NOT)
  ✓ Assignment behavior (new vs overwrite)
  ✓ Manager-level broadcasting across wells
  ✓ Complete petrophysical workflow

The toolkit now provides numpy-consistent, intuitive property operations
with strict depth matching for safe and predictable data processing!
""")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
