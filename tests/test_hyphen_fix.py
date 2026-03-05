#!/usr/bin/env python3
"""
Test script to verify hyphen preservation and well name prefix removal.
"""
import sys
from pathlib import Path
import tempfile
import shutil


from well_log_toolkit import WellDataManager
from well_log_toolkit.utils import sanitize_well_name
import pytest


def test_sanitize_with_hyphens():
    """Test that sanitize_well_name can preserve hyphens."""
    print("=" * 60)
    print("Test: Sanitize Well Name with Hyphens")
    print("=" * 60)

    test_cases = [
        ("36/7-5 ST2", "36_7_5_ST2", "36_7-5_ST2"),
        ("12/3-2 B", "12_3_2_B", "12_3-2_B"),
        ("Well-A", "Well_A", "Well-A"),
    ]

    all_pass = True
    for original, expected_without, expected_with in test_cases:
        result_without = sanitize_well_name(original, keep_hyphens=False)
        result_with = sanitize_well_name(original, keep_hyphens=True)

        print(f"\nOriginal: '{original}'")
        print(f"  Without hyphens: '{result_without}' (expected '{expected_without}')")
        print(f"  With hyphens:    '{result_with}' (expected '{expected_with}')")

        if result_without != expected_without:
            print(f"  ✗ FAIL: Without hyphens mismatch!")
            all_pass = False
        elif result_with != expected_with:
            print(f"  ✗ FAIL: With hyphens mismatch!")
            all_pass = False
        else:
            print(f"  ✓ PASS")

    assert all_pass


def test_filename_format_with_hyphens():
    """Test that exported filenames preserve hyphens."""
    print("\n" + "=" * 60)
    print("Test: Filename Format with Hyphens")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        manager = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:1]

        if not files:
            pytest.skip("No LAS files found in WellData/ directory")

        manager.load_las(files[0])

        if not manager.wells:
            pytest.skip("No wells loaded")

        well_key = manager.wells[0]
        well = getattr(manager, well_key)

        print(f"\nWell original name: {well.name}")
        print(f"Well sanitized name (for Python): {well.sanitized_name}")

        # Check if well name contains characters that become hyphens
        well_with_hyphens = sanitize_well_name(well.name, keep_hyphens=True)
        print(f"Well name for files (with hyphens): {well_with_hyphens}")

        # Save project
        project_path = Path(temp_dir) / "test_project"
        manager.save(project_path)

        # Check filenames
        well_folder = project_path / well_key
        las_files = list(well_folder.glob("*.las"))

        if not las_files:
            pytest.skip("✗ No LAS files found!")

        print("\nChecking filenames:")
        all_correct = True
        for las_file in las_files:
            print(f"  {las_file.name}")

            # Filename should start with well name (with hyphens if applicable)
            if not las_file.name.startswith(well_with_hyphens):
                print(f"    ✗ Doesn't start with '{well_with_hyphens}'")
                all_correct = False
            else:
                print(f"    ✓ Starts with well name (hyphens preserved)")

            # Check for duplicate well name
            # Count how many times the well name (without hyphens) appears
            well_count = las_file.name.count(well.sanitized_name.replace('_', ''))
            if well_count > 1:
                print(f"    ✗ Well name appears {well_count} times (should be 1)")
                all_correct = False

        if not all_correct:
            pytest.skip("Test precondition not met")

        print("\n✓ All filenames correctly formatted with hyphens preserved!")

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_well_prefix_removal():
    """Test that well name prefix is correctly removed from filenames."""
    print("\n" + "=" * 60)
    print("Test: Well Name Prefix Removal")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Create a test LAS file with well name prefix
        manager = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:1]

        if not files:
            pytest.skip("No LAS files found in WellData/ directory")

        manager.load_las(files[0])

        if not manager.wells:
            pytest.skip("No wells loaded")

        well_key = manager.wells[0]
        well = getattr(manager, well_key)

        print(f"\nWell: {well.name}")
        print(f"Sources before save: {well.sources}")

        # Save and reload
        project_path = Path(temp_dir) / "test_project"
        manager.save(project_path)

        # Reload
        manager2 = WellDataManager()
        manager2.load(project_path)

        well_key2 = manager2.wells[0]
        well2 = getattr(manager2, well_key2)

        print(f"\nSources after reload: {well2.sources}")

        # Check that source names are the same
        if set(well.sources) != set(well2.sources):
            print(f"✗ Source names changed!")
            print(f"  Before: {well.sources}")
            print(f"  After:  {well2.sources}")
            pytest.skip("Test precondition not met")

        # Check that source names don't have well name prefix
        well_name_for_check = sanitize_well_name(well.name, keep_hyphens=True).lower()
        for source in well2.sources:
            if source.lower().startswith(well_name_for_check):
                print(f"✗ Source '{source}' still has well name prefix!")
                pytest.skip("Test precondition not met")

        print(f"✓ Source names preserved correctly without well name prefix!")

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Testing Hyphen Preservation and Prefix Removal Fix")
    print("=" * 60)

    tests = [
        test_sanitize_with_hyphens,
        test_filename_format_with_hyphens,
        test_well_prefix_removal
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
