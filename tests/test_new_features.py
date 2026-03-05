"""Test new features: auto-discrete type, add_template, and duplicate depth tops."""
import numpy as np
import tempfile
import os
from pathlib import Path

# Add parent to path for local testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from logsuite import WellDataManager, Template
from logsuite.core.property import Property


def test_labels_auto_sets_discrete_type_on_property():
    """Setting labels on a property should auto-set type to 'discrete'."""
    # Create a property with continuous type (default)
    depth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([0.0, 1.0, 1.0, 2.0, 2.0])

    prop = Property(
        name="Facies",
        depth=depth,
        values=values,
        unit="",
        prop_type="continuous"  # Start as continuous
    )

    assert prop.type == "continuous", "Should start as continuous"

    # Set labels - should auto-set type to discrete
    prop.labels = {0: "Sand", 1: "Shale", 2: "Limestone"}

    assert prop.type == "discrete", "Type should auto-change to discrete when labels are set"
    assert prop.labels == {0: "Sand", 1: "Shale", 2: "Limestone"}
    print("✓ Property.labels auto-sets type to 'discrete'")


def test_labels_auto_sets_discrete_type_via_manager():
    """Setting labels via manager should auto-set type to 'discrete'."""
    import pandas as pd

    manager = WellDataManager()
    well = manager.add_well("TestWell")

    # Add a property as continuous (default)
    depth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    values = np.array([0.0, 1.0, 1.0, 2.0, 2.0])

    df = pd.DataFrame({
        'DEPT': depth,
        'Facies': values
    })
    well.add_dataframe(df)  # Default type is continuous

    # Check initial type
    prop = well.get_property('Facies')
    assert prop.type == "continuous", f"Should start as continuous, got '{prop.type}'"

    # Set labels via manager - should auto-set type to discrete
    manager.Facies.labels = {0: "Sand", 1: "Shale", 2: "Limestone"}

    # Check type changed to discrete
    prop = well.get_property('Facies')
    assert prop.type == "discrete", f"Type should be 'discrete', got '{prop.type}'"
    assert prop.labels == {0: "Sand", 1: "Shale", 2: "Limestone"}
    print("✓ Manager.property.labels auto-sets type to 'discrete'")


def test_add_template_uses_template_name():
    """add_template() should use the template's built-in name."""
    manager = WellDataManager()

    # Create template with name "reservoir"
    template = Template("reservoir")
    template.add_track(
        track_type="continuous",
        logs=[{"name": "GR", "x_range": [0, 150]}],
        title="Gamma Ray"
    )

    # Add using add_template (should use template.name)
    manager.add_template(template)

    # Verify it's stored under "reservoir"
    assert "reservoir" in manager.list_templates(), "Template should be stored as 'reservoir'"
    retrieved = manager.get_template("reservoir")
    assert retrieved.name == "reservoir"
    print("✓ add_template() uses template's built-in name")


def test_set_template_allows_custom_name():
    """set_template() should allow storing with a different name."""
    manager = WellDataManager()

    # Create template with name "reservoir"
    template = Template("reservoir")

    # Store with custom name
    manager.set_template("custom_name", template)

    # Verify it's stored under custom name
    assert "custom_name" in manager.list_templates()
    assert "reservoir" not in manager.list_templates()
    print("✓ set_template() allows custom name override")


def test_tops_with_duplicate_depths():
    """Tops at the same depth should all be preserved."""
    import pandas as pd

    manager = WellDataManager()
    well = manager.add_well("TestWell")

    # Create a discrete property with multiple tops at same depth
    # Simulating: Top Agat @ 2626.1, Agat Sand 3 @ 2626.1, etc.
    depth = np.array([2600.0, 2626.1, 2650.0, 2685.8, 2718.3, 2721.5])
    # Values: different zone codes
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    df = pd.DataFrame({
        'DEPT': depth,
        'Well_Tops': values
    })
    well.add_dataframe(df, type_mappings={'Well_Tops': 'discrete'})

    # Set labels
    prop = well.get_property('Well_Tops')
    prop.labels = {
        1: "Top Formation",
        2: "Top Agat",
        3: "Mid Zone",
        4: "Agat Sand 2",
        5: "Top Sola",
        6: "OWC"
    }

    # Create template and add tops
    template = Template("test")
    template.add_tops(property_name='Well_Tops')
    template.add_track(track_type="continuous", logs=[{"name": "Well_Tops"}], title="Test")

    # Create WellView - this should work now
    from logsuite.visualization import WellView

    # Test that we can look up tops by name
    view = WellView(
        well=well,
        depth_range=[2600, 2750],
        template=template
    )

    # Collect all tops from the view
    all_tops = []
    for tops_group in view.tops:
        entries = tops_group.get('entries', [])
        for entry in entries:
            all_tops.append((entry['depth'], entry['name']))

    # Verify all tops are preserved
    top_names = [name for _, name in all_tops]
    assert "Top Agat" in top_names, "Top Agat should be in tops"
    assert "Top Sola" in top_names, "Top Sola should be in tops"
    print(f"✓ All tops preserved: {top_names}")


def test_tops_depth_range_from_names():
    """Should be able to calculate depth range from top names even with duplicate depths."""
    import pandas as pd

    manager = WellDataManager()
    well = manager.add_well("TestWell")

    # Create tops with some at same depth
    depth = np.array([2600.0, 2626.1, 2626.1, 2700.0, 2718.3, 2718.3])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    df = pd.DataFrame({
        'DEPT': depth,
        'Tops': values
    })
    well.add_dataframe(df, type_mappings={'Tops': 'discrete'})

    prop = well.get_property('Tops')
    prop.labels = {
        1: "Formation A",
        2: "Top Agat",      # @ 2626.1
        3: "Agat Sand 3",   # @ 2626.1 (same depth!)
        4: "Mid Zone",
        5: "Top Sola",      # @ 2718.3
        6: "Base Sands"     # @ 2718.3 (same depth!)
    }

    template = Template("test")
    template.add_tops(property_name='Tops')
    template.add_track(track_type="continuous", logs=[{"name": "Tops"}], title="Test")

    from logsuite.visualization import WellView

    # This should now work - looking up "Top Agat" and "Top Sola" by name
    try:
        view = WellView(
            well=well,
            tops=["Top Agat", "Top Sola"],
            template=template
        )
        # Check depth range was calculated correctly
        # Top Agat @ 2626.1, Top Sola @ 2718.3
        # With padding, range should be approximately [2621, 2723]
        assert view.depth_range[0] < 2626.1, f"Min depth should be < 2626.1, got {view.depth_range[0]}"
        assert view.depth_range[1] > 2718.3, f"Max depth should be > 2718.3, got {view.depth_range[1]}"
        print(f"✓ Depth range calculated correctly: {view.depth_range}")
    except ValueError as e:
        print(f"✗ Failed to find tops: {e}")
        raise


def test_template_save_load_with_new_format():
    """Templates should save and load correctly."""
    manager = WellDataManager()

    template = Template("test_template")
    template.add_track(
        track_type="continuous",
        logs=[{"name": "GR", "x_range": [0, 150]}],
        title="Gamma Ray"
    )
    template.add_tops(property_name='Zone')

    manager.add_template(template)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        manager.save(tmpdir)

        # Check template file exists
        template_file = Path(tmpdir) / "templates" / "test_template.json"
        assert template_file.exists(), "Template file should be saved"

        # Load into new manager
        manager2 = WellDataManager()
        manager2.load(tmpdir)

        # Verify template was loaded
        assert "test_template" in manager2.list_templates()
        loaded = manager2.get_template("test_template")
        assert loaded.name == "test_template"
        assert len(loaded.tracks) == 1
        print("✓ Template save/load works correctly")


if __name__ == "__main__":
    print("\n=== Testing New Features ===\n")

    test_labels_auto_sets_discrete_type_on_property()
    test_labels_auto_sets_discrete_type_via_manager()
    test_add_template_uses_template_name()
    test_set_template_allows_custom_name()
    test_tops_with_duplicate_depths()
    test_tops_depth_range_from_names()
    test_template_save_load_with_new_format()

    print("\n=== All Tests Passed ===\n")
