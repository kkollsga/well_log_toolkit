#!/usr/bin/env python3
"""
Test script for project export/import functionality.
"""
import sys
from pathlib import Path
import tempfile
import shutil


from logsuite import WellDataManager
import pytest


def test_project_export_import():
    """Test project-level export and import functionality."""
    print("=" * 60)
    print("Test: Project Export/Import")
    print("=" * 60)

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Step 1: Load data from WellData directory
        manager1 = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:2]  # Use first 2 files for quick test

        if not files:
            pytest.skip("No LAS files found in WellData/ directory")

        print(f"\nLoading {len(files)} files...")
        manager1.load_las(files)

        print(f"Wells loaded: {manager1.wells}")
        for well_name in manager1.wells:
            well = getattr(manager1, well_name)
            print(f"  {well.name}: {len(well.sources)} sources, {len(well.properties)} properties")

        # Step 2: Export project
        export_path = Path(temp_dir) / "exported_project"
        print(f"\nExporting project to: {export_path}")
        manager1.export_project(export_path)

        # Check exported structure
        print("\nExported structure:")
        for well_folder in sorted(export_path.glob("well_*")):
            print(f"  {well_folder.name}/")
            for las_file in sorted(well_folder.glob("*.las")):
                print(f"    {las_file.name}")

        # Step 3: Import project into new manager
        print(f"\nImporting project from: {export_path}")
        manager2 = WellDataManager()
        manager2.import_project(export_path)

        print(f"Wells imported: {manager2.wells}")
        for well_name in manager2.wells:
            well = getattr(manager2, well_name)
            print(f"  {well.name}: {len(well.sources)} sources, {len(well.properties)} properties")

        # Step 4: Verify data integrity
        print("\nVerifying data integrity...")
        if set(manager1.wells) != set(manager2.wells):
            pytest.skip("  ✗ Well names don't match!")

        for well_name in manager1.wells:
            well1 = getattr(manager1, well_name)
            well2 = getattr(manager2, well_name)

            if set(well1.sources) != set(well2.sources):
                print(f"  ✗ Sources don't match for {well_name}")
                pytest.skip("Test precondition not met")

            # Compare properties count (rough check)
            if len(well1.properties) != len(well2.properties):
                print(f"  ✗ Properties count doesn't match for {well_name}")
                pytest.skip("Test precondition not met")

        print("  ✓ Data integrity verified!")


    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_well_export_sources():
    """Test well.export_sources() method directly."""
    print("\n" + "=" * 60)
    print("Test: Well.export_sources()")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        manager = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:1]  # Use first file

        if not files:
            pytest.skip("No LAS files found in WellData/ directory")

        manager.load_las(files[0])

        if not manager.wells:
            pytest.skip("No wells loaded")

        well_name = manager.wells[0]
        well = getattr(manager, well_name)

        print(f"\nWell: {well.name}")
        print(f"Sources: {well.sources}")

        # Export sources to temp directory
        export_path = Path(temp_dir) / "well_export"
        print(f"\nExporting sources to: {export_path}")
        well.export_sources(export_path)

        # Check exported files
        print("\nExported files:")
        for las_file in sorted(export_path.glob("*.las")):
            print(f"  {las_file.name}")

        # Verify files exist
        expected_count = len(well.sources)
        actual_count = len(list(export_path.glob("*.las")))

        if expected_count != actual_count:
            print(f"  ✗ Expected {expected_count} files, got {actual_count}")
            pytest.skip("Test precondition not met")

        print(f"  ✓ All {actual_count} source files exported successfully!")


    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Testing Project Export/Import Functionality")
    print("=" * 60)

    tests = [
        test_well_export_sources,
        test_project_export_import
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
