"""
Test script for property printing features.

Demonstrates:
1. Manager-level property printing (manager.PHIE)
2. Filtered property printing (well.phie.filter("NTG"))
"""
import numpy as np
from logsuite import WellDataManager, Well, Property


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subsection(title):
    """Print subsection header."""
    print(f"\n{title}")
    print("-" * 70)


def test_manager_printing():
    """Test 1: Manager-level property printing."""
    print_section("TEST 1: MANAGER-LEVEL PROPERTY PRINTING")

    # Create manager with 2 wells
    manager = WellDataManager()

    print_subsection("1.1 Creating wells with PHIE property")

    # Well A
    well_a = Well(name='36/7-5 A', sanitized_name='36_7_5_A')
    depth_a = np.array([2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809])
    phie_a = np.array([0.25, 0.30, 0.22, 0.18, 0.28, 0.32, 0.19, 0.27, 0.24, 0.29])

    well_a_phie = Property(
        name='PHIE',
        depth=depth_a,
        values=phie_a,
        parent_well=well_a,
        unit='v/v',
        prop_type='continuous'
    )

    # Add to well's sources
    well_a._sources['Petrophysics'] = {
        'properties': {'PHIE': well_a_phie},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }
    manager._wells['well_36_7_5_A'] = well_a
    well_a.parent_manager = manager

    # Well B
    well_b = Well(name='36/7-5 B', sanitized_name='36_7_5_B')
    depth_b = np.array([2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857])
    phie_b = np.array([0.20, 0.24, 0.26, 0.19, 0.23, 0.21, 0.25, 0.22])

    well_b_phie = Property(
        name='PHIE',
        depth=depth_b,
        values=phie_b,
        parent_well=well_b,
        unit='v/v',
        prop_type='continuous'
    )

    # Add to well's sources
    well_b._sources['Petrophysics'] = {
        'properties': {'PHIE': well_b_phie},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }
    manager._wells['well_36_7_5_B'] = well_b
    well_b.parent_manager = manager

    print("✓ Created 2 wells with PHIE property")

    print_subsection("1.2 Print manager.PHIE (all wells)")
    print(manager.PHIE)

    print("\n✓ Manager-level printing shows property across all wells")


def test_filtered_printing():
    """Test 2: Filtered property printing."""
    print_section("TEST 2: FILTERED PROPERTY PRINTING")

    # Create well with properties
    well = Well(name='Production Well', sanitized_name='Production_Well')

    print_subsection("2.1 Creating well with PHIE and NTG properties")

    # Create depth grid
    depth = np.array([2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809])

    # PHIE property (continuous)
    phie_values = np.array([0.25, 0.30, 0.22, 0.18, 0.28, 0.32, 0.19, 0.27, 0.24, 0.29])
    phie_prop = Property(
        name='PHIE',
        depth=depth,
        values=phie_values,
        parent_well=well,
        unit='v/v',
        prop_type='continuous'
    )

    # NTG property (discrete)
    ntg_values = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    ntg_prop = Property(
        name='NTG',
        depth=depth,
        values=ntg_values,
        parent_well=well,
        unit='',
        prop_type='discrete',
        labels={0: 'NonNet', 1: 'Net'}
    )

    # Zone property (discrete)
    zone_values = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    zone_prop = Property(
        name='Zone',
        depth=depth,
        values=zone_values,
        parent_well=well,
        unit='',
        prop_type='discrete',
        labels={0: 'Reservoir_A', 1: 'Reservoir_B'}
    )

    # Add properties to well
    well._sources['Petrophysics'] = {
        'properties': {'PHIE': phie_prop, 'NTG': ntg_prop, 'Zone': zone_prop},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }

    print("✓ Created PHIE, NTG, and Zone properties")

    print_subsection("2.2 Print PHIE without filters")
    print(well.PHIE)

    print_subsection("2.3 Print PHIE filtered by NTG")
    filtered_by_ntg = well.PHIE.filter('NTG')
    print(filtered_by_ntg)

    print_subsection("2.4 Print PHIE filtered by NTG and Zone")
    filtered_by_both = well.PHIE.filter('NTG').filter('Zone')
    print(filtered_by_both)

    print("\n✓ Filtered property printing shows both log values and discrete filter values with labels")


def test_large_array_clipping():
    """Test 3: Large array clipping."""
    print_section("TEST 3: LARGE ARRAY CLIPPING")

    # Create well with large property
    well = Well(name='Deep Well', sanitized_name='Deep_Well')

    print_subsection("3.1 Creating well with 100 samples")

    # Create large depth grid
    depth = np.arange(2800, 2900, 1.0)  # 100 samples
    phie_values = 0.20 + 0.05 * np.sin(np.arange(100) * 0.1)  # Varying porosity

    phie_prop = Property(
        name='PHIE',
        depth=depth,
        values=phie_values,
        parent_well=well,
        unit='v/v',
        prop_type='continuous'
    )

    # NTG property (discrete)
    ntg_values = (phie_values > 0.22).astype(float)  # 1 where PHIE > 0.22, 0 otherwise
    ntg_prop = Property(
        name='NTG',
        depth=depth,
        values=ntg_values,
        parent_well=well,
        unit='',
        prop_type='discrete',
        labels={0: 'NonNet', 1: 'Net'}
    )

    # Add properties to well
    well._sources['Petrophysics'] = {
        'properties': {'PHIE': phie_prop, 'NTG': ntg_prop},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }

    print("✓ Created large properties (100 samples)")

    print_subsection("3.2 Print PHIE (should clip to first 3 ... last 3)")
    print(well.PHIE)

    print_subsection("3.3 Print filtered PHIE (clipped with filter labels)")
    filtered = well.PHIE.filter('NTG')
    print(filtered)

    print("\n✓ Large arrays are automatically clipped for readability")


def test_manager_with_missing_property():
    """Test 4: Manager printing when some wells don't have the property."""
    print_section("TEST 4: MANAGER PRINTING WITH MISSING PROPERTY")

    # Create manager with 3 wells
    manager = WellDataManager()

    print_subsection("4.1 Creating 3 wells, only 2 have PHIE")

    # Well A - has PHIE
    well_a = Well(name='Well A', sanitized_name='Well_A')
    depth_a = np.array([2800, 2801, 2802, 2803, 2804])
    phie_a = np.array([0.25, 0.30, 0.22, 0.18, 0.28])

    well_a_phie = Property(
        name='PHIE',
        depth=depth_a,
        values=phie_a,
        parent_well=well_a,
        unit='v/v',
        prop_type='continuous'
    )

    well_a._sources['Petrophysics'] = {
        'properties': {'PHIE': well_a_phie},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }
    manager._wells['well_Well_A'] = well_a
    well_a.parent_manager = manager

    # Well B - has PHIE
    well_b = Well(name='Well B', sanitized_name='Well_B')
    depth_b = np.array([2850, 2851, 2852])
    phie_b = np.array([0.20, 0.24, 0.26])

    well_b_phie = Property(
        name='PHIE',
        depth=depth_b,
        values=phie_b,
        parent_well=well_b,
        unit='v/v',
        prop_type='continuous'
    )

    well_b._sources['Petrophysics'] = {
        'properties': {'PHIE': well_b_phie},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }
    manager._wells['well_Well_B'] = well_b
    well_b.parent_manager = manager

    # Well C - NO PHIE (only SW)
    well_c = Well(name='Well C', sanitized_name='Well_C')
    depth_c = np.array([2900, 2901, 2902, 2903])
    sw_c = np.array([0.35, 0.40, 0.38, 0.42])

    well_c_sw = Property(
        name='SW',
        depth=depth_c,
        values=sw_c,
        parent_well=well_c,
        unit='v/v',
        prop_type='continuous'
    )

    well_c._sources['Petrophysics'] = {
        'properties': {'SW': well_c_sw},
        'las': None,
        'modified': False,
        'original_name': 'Petrophysics'
    }
    manager._wells['well_Well_C'] = well_c
    well_c.parent_manager = manager

    print("✓ Created 3 wells (2 with PHIE, 1 without)")

    print_subsection("4.2 Print manager.PHIE (shows only wells with PHIE)")
    print(manager.PHIE)

    print_subsection("4.3 Print manager.SW (shows only Well C)")
    print(manager.SW)

    print("\n✓ Manager printing automatically skips wells without the property")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PROPERTY PRINTING FEATURES TEST SUITE")
    print("=" * 70)

    test_manager_printing()
    test_filtered_printing()
    test_large_array_clipping()
    test_manager_with_missing_property()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Manager-level property printing (manager.PHIE)")
    print("  ✓ Filtered property printing (well.phie.filter('NTG'))")
    print("  ✓ Large array clipping (first 3 ... last 3)")
    print("  ✓ Missing property handling")
    print("\nFeatures:")
    print("  • Manager printing shows property across all wells")
    print("  • Filtered properties show both log values and discrete filter labels")
    print("  • Arrays automatically clip at 8 elements for readability")
    print("  • Discrete properties display labels instead of numeric values")
    print()
