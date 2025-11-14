#!/usr/bin/env python3
"""
Test script to verify filename format is correct during save/load cycles.
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from well_log_toolkit import WellDataManager


def test_filename_format():
    """Test that filenames don't have duplicate well_ prefix."""
    print("=" * 60)
    print("Test: Filename Format (no duplicate well_ prefix)")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Load data
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

        well_name = manager.wells[0]
        well = getattr(manager, well_name)

        print(f"\nWell: {well.name}")
        print(f"Sanitized name (for folder): {well.sanitized_name}")
        print(f"Sources: {well.sources}")

        # Save project
        project_path = Path(temp_dir) / "test_project"
        print(f"\nSaving project to: {project_path}")
        manager.save(project_path)

        # Check filename format
        print("\nChecking saved file structure:")
        well_folder = project_path / well.sanitized_name
        print(f"Well folder: {well_folder.name}")

        las_files = list(well_folder.glob("*.las"))
        if not las_files:
            print("✗ No LAS files found!")
            return False

        print("\nFiles in well folder:")
        all_correct = True
        for las_file in las_files:
            print(f"  {las_file.name}")

            # Check that filename does NOT start with "well_"
            if las_file.name.startswith("well_"):
                print(f"    ✗ ERROR: Filename has 'well_' prefix!")
                all_correct = False
            else:
                print(f"    ✓ Correct: No 'well_' prefix in filename")

        if not all_correct:
            return False

        print("\n✓ All filenames have correct format!")
        return True

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_multiple_save_load_cycles():
    """Test that multiple save/load cycles don't accumulate well_ prefixes."""
    print("\n" + "=" * 60)
    print("Test: Multiple Save/Load Cycles")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Initial load
        manager1 = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:1]

        if not files:
            print("No LAS files found in WellData/ directory")
            return False

        print("\n1. Initial load from LAS file")
        manager1.load_las(files[0])

        well_name = manager1.wells[0]
        well1 = getattr(manager1, well_name)
        initial_sources = set(well1.sources)
        print(f"   Sources: {initial_sources}")

        project_path = Path(temp_dir) / "test_project"

        # First save
        print("\n2. First save")
        manager1.save(project_path)

        # Check filenames after first save
        well_folder = project_path / well1.sanitized_name
        files_after_first_save = {f.name for f in well_folder.glob("*.las")}
        print(f"   Files: {files_after_first_save}")

        # First load
        print("\n3. First load (from saved project)")
        manager2 = WellDataManager()
        manager2.load(project_path)

        well2 = getattr(manager2, well_name)
        print(f"   Sources: {set(well2.sources)}")

        # Second save
        print("\n4. Second save (overwriting)")
        manager2.save()

        # Check filenames after second save
        files_after_second_save = {f.name for f in well_folder.glob("*.las")}
        print(f"   Files: {files_after_second_save}")

        # Second load
        print("\n5. Second load")
        manager3 = WellDataManager()
        manager3.load(project_path)

        well3 = getattr(manager3, well_name)
        final_sources = set(well3.sources)
        print(f"   Sources: {final_sources}")

        # Third save
        print("\n6. Third save (overwriting)")
        manager3.save()

        # Check filenames after third save
        files_after_third_save = {f.name for f in well_folder.glob("*.las")}
        print(f"   Files: {files_after_third_save}")

        # Verify sources are consistent
        print("\n7. Verifying consistency...")
        if initial_sources != final_sources:
            print(f"   ✗ Source names changed!")
            print(f"      Initial:  {initial_sources}")
            print(f"      Final:    {final_sources}")
            return False

        # Verify filenames haven't changed
        if files_after_first_save != files_after_second_save:
            print(f"   ✗ Filenames changed after second save!")
            return False

        if files_after_second_save != files_after_third_save:
            print(f"   ✗ Filenames changed after third save!")
            return False

        # Verify no files have "well_well_" or similar duplication
        for filename in files_after_third_save:
            if "well_well_" in filename or filename.count("well_") > 0:
                print(f"   ✗ Filename has duplicate or incorrect well_ prefix: {filename}")
                return False

        print("   ✓ Sources consistent across all cycles!")
        print("   ✓ Filenames remain stable!")
        print("   ✓ No duplicate well_ prefixes!")

        return True

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Testing Filename Format Fix")
    print("=" * 60)

    tests = [
        test_filename_format,
        test_multiple_save_load_cycles
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
