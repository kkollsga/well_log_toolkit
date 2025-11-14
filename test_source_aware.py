#!/usr/bin/env python3
"""
Test script for source-aware architecture refactoring.
"""
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from well_log_toolkit import WellDataManager

def test_basic_loading():
    """Test basic LAS file loading with source-aware architecture."""
    print("=" * 60)
    print("Test 1: Basic Loading")
    print("=" * 60)

    manager = WellDataManager()

    # Load multiple LAS files from the WellData directory
    import glob
    files = glob.glob('WellData/*')

    if not files:
        print("No LAS files found in WellData/ directory")
        return False

    print(f"Found {len(files)} files")

    # Load files
    manager.load_las(files)

    print(f"\nWells loaded: {manager.wells}")

    for well_name in manager.wells:
        well = getattr(manager, well_name)
        print(f"\n{well}")
        print(f"  Sources: {well.sources}")
        print(f"  Properties: {well.properties[:5]}...")  # Show first 5

    return True


def test_source_access():
    """Test source-based property access."""
    print("\n" + "=" * 60)
    print("Test 2: Source Access")
    print("=" * 60)

    manager = WellDataManager()

    import glob
    files = glob.glob('WellData/*')[:1]  # Just use first file

    if not files:
        print("No LAS files found")
        return False

    manager.load_las(files[0])

    well_name = manager.wells[0]
    well = getattr(manager, well_name)

    print(f"\nWell: {well}")
    print(f"Sources: {well.sources}")

    if well.sources:
        source_name = well.sources[0]
        print(f"\nAccessing source '{source_name}':")

        # Access source
        source = getattr(well, source_name)
        print(f"  Source view: {source}")

        # Access property from source
        if well.properties:
            prop_name = well.properties[0]
            print(f"\nAccessing property '{prop_name}' from source '{source_name}':")
            prop = getattr(source, prop_name)
            print(f"  Property: {prop}")
            print(f"  Depth samples: {len(prop.depth)}")
            print(f"  Value range: [{prop.values.min():.2f}, {prop.values.max():.2f}]")

    return True


def test_unique_property_access():
    """Test accessing unique properties directly."""
    print("\n" + "=" * 60)
    print("Test 3: Unique Property Access")
    print("=" * 60)

    manager = WellDataManager()

    import glob
    files = glob.glob('WellData/*')[:1]

    if not files:
        print("No LAS files found")
        return False

    manager.load_las(files[0])

    well_name = manager.wells[0]
    well = getattr(manager, well_name)

    print(f"\nWell: {well}")

    # Try accessing a property directly (should work if unique)
    if well.properties:
        prop_name = well.properties[0]
        print(f"\nAccessing unique property '{prop_name}' directly:")
        try:
            prop = getattr(well, prop_name)
            print(f"  Success! Property: {prop}")
            print(f"  Source: {prop.source}")
        except AttributeError as e:
            print(f"  Error (expected if ambiguous): {e}")

    return True


def test_property_methods():
    """Test to_dataframe and other property methods."""
    print("\n" + "=" * 60)
    print("Test 4: Property Methods")
    print("=" * 60)

    manager = WellDataManager()

    import glob
    files = glob.glob('WellData/*')[:1]

    if not files:
        print("No LAS files found")
        return False

    manager.load_las(files[0])

    well_name = manager.wells[0]
    well = getattr(manager, well_name)

    print(f"\nWell: {well}")

    # Test to_dataframe
    print("\nTesting to_dataframe():")
    try:
        df = well.to_dataframe()
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns[:5])}...")
    except Exception as e:
        print(f"  Error: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("Testing Source-Aware Architecture Refactoring")
    print("=" * 60)

    tests = [
        test_basic_loading,
        test_source_access,
        test_unique_property_access,
        test_property_methods
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"\n✓ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Tests: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
