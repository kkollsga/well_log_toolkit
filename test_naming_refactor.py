#!/usr/bin/env python3
"""
Test script to verify the sanitized naming refactor works correctly.
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from well_log_toolkit import WellDataManager


def test_sanitized_name_format():
    """Test that sanitized_name doesn't include well_ prefix."""
    print("=" * 60)
    print("Test: Sanitized Name Format")
    print("=" * 60)

    manager = WellDataManager()

    import glob
    files = glob.glob('WellData/*')[:1]

    if not files:
        print("No LAS files found in WellData/ directory")
        return False

    print(f"\nLoading file: {files[0]}")
    manager.load_las(files[0])

    if not manager.wells:
        print("No wells loaded")
        return False

    # Check wells list has well_ prefix
    print(f"\nWells in manager: {manager.wells}")
    for well_key in manager.wells:
        if not well_key.startswith('well_'):
            print(f"✗ Well key '{well_key}' doesn't start with 'well_'")
            return False
        print(f"  ✓ Well key: {well_key}")

    # Check actual well object's sanitized_name does NOT have well_ prefix
    well_key = manager.wells[0]
    well = getattr(manager, well_key)

    print(f"\nWell object:")
    print(f"  name: {well.name}")
    print(f"  sanitized_name: {well.sanitized_name}")

    if well.sanitized_name.startswith('well_'):
        print(f"  ✗ sanitized_name '{well.sanitized_name}' should NOT start with 'well_'")
        return False

    print(f"  ✓ sanitized_name does not have 'well_' prefix")

    # Verify folder structure when using well_ prefix
    if well_key != f"well_{well.sanitized_name}":
        print(f"  ✗ Well key '{well_key}' != 'well_{well.sanitized_name}'")
        return False

    print(f"  ✓ Well key is correctly formatted as 'well_{{sanitized_name}}'")

    return True


def test_filename_format():
    """Test that filenames use sanitized_name directly."""
    print("\n" + "=" * 60)
    print("Test: Filename Format")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        manager = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:1]

        if not files:
            print("No LAS files found in WellData/ directory")
            return False

        manager.load_las(files[0])
        well_key = manager.wells[0]
        well = getattr(manager, well_key)

        print(f"\nWell: {well.name}")
        print(f"Sanitized name: {well.sanitized_name}")
        print(f"Manager key: {well_key}")

        # Save project
        project_path = Path(temp_dir) / "test_project"
        manager.save(project_path)

        # Check folder name
        well_folder = project_path / well_key
        if not well_folder.exists():
            print(f"✗ Well folder '{well_key}' doesn't exist")
            return False

        print(f"\n✓ Well folder created: {well_key}")

        # Check filenames
        print("\nChecking filenames:")
        las_files = list(well_folder.glob("*.las"))

        if not las_files:
            print("✗ No LAS files found!")
            return False

        all_correct = True
        for las_file in las_files:
            print(f"  {las_file.name}")

            # Filename should start with sanitized_name (without well_ prefix)
            if not las_file.name.startswith(well.sanitized_name):
                print(f"    ✗ Filename doesn't start with '{well.sanitized_name}'")
                all_correct = False
            else:
                print(f"    ✓ Starts with sanitized_name")

            # Filename should NOT start with well_
            if las_file.name.startswith('well_'):
                print(f"    ✗ Filename has 'well_' prefix (should not)")
                all_correct = False

        if not all_correct:
            return False

        print("\n✓ All filenames correctly formatted!")
        return True

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_attribute_access():
    """Test that attribute access works with well_ prefix."""
    print("\n" + "=" * 60)
    print("Test: Attribute Access")
    print("=" * 60)

    manager = WellDataManager()

    import glob
    files = glob.glob('WellData/*')[:1]

    if not files:
        print("No LAS files found in WellData/ directory")
        return False

    manager.load_las(files[0])
    well_key = manager.wells[0]

    print(f"\nWell key: {well_key}")

    # Test attribute access
    print(f"\nTesting attribute access: manager.{well_key}")
    try:
        well = getattr(manager, well_key)
        print(f"✓ Attribute access works: {well}")
    except AttributeError as e:
        print(f"✗ Attribute access failed: {e}")
        return False

    # Test that we can't access without well_ prefix
    well_obj = manager._wells[well_key]
    sanitized_without_prefix = well_obj.sanitized_name

    print(f"\nTesting access without 'well_' prefix: manager.{sanitized_without_prefix}")
    try:
        well_wrong = getattr(manager, sanitized_without_prefix)
        print(f"✗ Access without 'well_' prefix should fail but succeeded")
        return False
    except AttributeError:
        print(f"✓ Correctly raises AttributeError for access without 'well_' prefix")

    return True


def main():
    """Run all tests."""
    print("Testing Sanitized Naming Refactor")
    print("=" * 60)

    tests = [
        test_sanitized_name_format,
        test_filename_format,
        test_attribute_access
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
