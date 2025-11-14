#!/usr/bin/env python3
"""
Test script for project save/load functionality.
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from well_log_toolkit import WellDataManager


def test_save_with_path():
    """Test save() with explicit path."""
    print("=" * 60)
    print("Test: Save with explicit path")
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

        print(f"\nLoading {len(files)} files...")
        manager.load_las(files)

        print(f"Wells loaded: {manager.wells}")

        # Save with explicit path
        project_path = Path(temp_dir) / "test_project"
        print(f"\nSaving project to: {project_path}")
        manager.save(project_path)

        # Check saved structure
        print("\nSaved structure:")
        for well_folder in sorted(project_path.glob("well_*")):
            print(f"  {well_folder.name}/")
            for las_file in sorted(well_folder.glob("*.las")):
                print(f"    {las_file.name}")

        if not list(project_path.glob("well_*")):
            print("✗ No well folders created!")
            return False

        print("✓ Project saved successfully!")
        return True

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_load_then_save():
    """Test load() then save() without path."""
    print("\n" + "=" * 60)
    print("Test: Load then save without path")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Step 1: Create a project
        manager1 = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:1]

        if not files:
            print("No LAS files found in WellData/ directory")
            return False

        manager1.load_las(files)
        project_path = Path(temp_dir) / "test_project"

        print(f"\n1. Creating initial project at: {project_path}")
        manager1.save(project_path)

        # Step 2: Load the project
        manager2 = WellDataManager()
        print(f"\n2. Loading project from: {project_path}")
        manager2.load(project_path)

        print(f"Wells loaded: {manager2.wells}")

        # Step 3: Save without providing path (should use stored path)
        print("\n3. Saving without providing path (should save to same location)")
        try:
            manager2.save()
            print("✓ Save without path succeeded!")
        except ValueError as e:
            print(f"✗ Save without path failed: {e}")
            return False

        # Verify files still exist
        if not list(project_path.glob("well_*")):
            print("✗ No well folders found after save!")
            return False

        print("✓ Project saved successfully to stored path!")
        return True

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def test_save_without_load_fails():
    """Test that save() without path and without prior load() raises error."""
    print("\n" + "=" * 60)
    print("Test: Save without path and without prior load should fail")
    print("=" * 60)

    manager = WellDataManager()

    import glob
    files = glob.glob('WellData/*')[:1]

    if not files:
        print("No LAS files found in WellData/ directory")
        return False

    manager.load_las(files)

    print("\nTrying to save() without path and without prior load()...")
    try:
        manager.save()
        print("✗ Save without path succeeded when it should have failed!")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        return True


def test_full_roundtrip():
    """Test full save/load roundtrip with data integrity check."""
    print("\n" + "=" * 60)
    print("Test: Full Save/Load Roundtrip")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Step 1: Load data
        manager1 = WellDataManager()

        import glob
        files = glob.glob('WellData/*')[:2]

        if not files:
            print("No LAS files found in WellData/ directory")
            return False

        print(f"\n1. Loading {len(files)} files...")
        manager1.load_las(files)

        print(f"Wells loaded: {manager1.wells}")
        for well_name in manager1.wells:
            well = getattr(manager1, well_name)
            print(f"  {well.name}: {len(well.sources)} sources")

        # Step 2: Save project
        project_path = Path(temp_dir) / "test_project"
        print(f"\n2. Saving project to: {project_path}")
        manager1.save(project_path)

        # Step 3: Load project
        print(f"\n3. Loading project from: {project_path}")
        manager2 = WellDataManager()
        manager2.load(project_path)

        print(f"Wells loaded: {manager2.wells}")
        for well_name in manager2.wells:
            well = getattr(manager2, well_name)
            print(f"  {well.name}: {len(well.sources)} sources")

        # Step 4: Verify data integrity
        print("\n4. Verifying data integrity...")
        if set(manager1.wells) != set(manager2.wells):
            print("  ✗ Well names don't match!")
            return False

        for well_name in manager1.wells:
            well1 = getattr(manager1, well_name)
            well2 = getattr(manager2, well_name)

            if set(well1.sources) != set(well2.sources):
                print(f"  ✗ Sources don't match for {well_name}")
                return False

            if len(well1.properties) != len(well2.properties):
                print(f"  ✗ Properties count doesn't match for {well_name}")
                return False

        print("  ✓ Data integrity verified!")
        return True

    finally:
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Testing Project Save/Load Functionality")
    print("=" * 60)

    tests = [
        test_save_with_path,
        test_save_without_load_fails,
        test_load_then_save,
        test_full_roundtrip
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
