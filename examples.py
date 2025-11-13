"""
Examples demonstrating the usage of well_log_toolkit.

This module provides practical examples of how to use the well_log_toolkit
for loading, analyzing, and filtering well log data from LAS files.
"""

from well_log_toolkit import WellDataManager, LasFile


# =============================================================================
# Example 1: Basic Usage - Loading and Accessing Data
# =============================================================================

def example_basic_usage():
    """Load a single LAS file and access well data."""
    print("Example 1: Basic Usage")
    print("-" * 50)

    # Initialize the manager
    manager = WellDataManager()

    # Load a LAS file (replace with your actual file path)
    manager.load_las("path/to/well1.las")

    # List available wells
    print(f"Available wells: {manager.wells}")

    # Access well by attribute (sanitized name)
    well = manager.well_12_3_2_B

    # List available properties
    print(f"Available properties: {well.properties}")

    # Access a property
    phie = well.phie
    print(f"PHIE property: {phie}")
    print()


# =============================================================================
# Example 2: Method Chaining - Loading Multiple LAS Files
# =============================================================================

def example_method_chaining():
    """Demonstrate method chaining for loading multiple files."""
    print("Example 2: Method Chaining")
    print("-" * 50)

    manager = WellDataManager()

    # Load multiple LAS files using method chaining
    (manager
     .load_las("path/to/well1.las")
     .load_las("path/to/well2.las")
     .load_las("path/to/well3.las"))

    print(f"Loaded {len(manager.wells)} wells")
    for well_name in manager.wells:
        print(f"  - {well_name}")
    print()


# =============================================================================
# Example 3: Property Types - Discrete vs Continuous
# =============================================================================

def example_property_types():
    """Set property types for proper filtering."""
    print("Example 3: Property Types")
    print("-" * 50)

    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    well = manager.well_12_3_2_B

    # Mark discrete properties (facies, flags, zones)
    well.get_property('Zone').type = 'discrete'
    well.get_property('NTG_Flag').type = 'discrete'
    well.get_property('Facies').type = 'discrete'

    # Continuous properties are the default type
    print(f"PHIE type: {well.phie.type}")
    print(f"Zone type: {well.Zone.type}")
    print()


# =============================================================================
# Example 4: Filtering Data with Labels
# =============================================================================

def example_filtering():
    """Filter well log data using discrete properties with label mapping."""
    print("Example 4: Filtering Data with Labels")
    print("-" * 50)

    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    well = manager.well_12_3_2_B

    # Mark discrete properties and set labels
    zone_prop = well.get_property('Zone')
    zone_prop.type = 'discrete'
    zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir', 2: 'Transition'}

    ntg_prop = well.get_property('NTG_Flag')
    ntg_prop.type = 'discrete'
    ntg_prop.labels = {0: 'NonNet', 1: 'Net'}

    # Filter by a single discrete property
    filtered_phie = well.phie.filter('Zone')
    print(f"PHIE filtered by Zone: {filtered_phie}")

    # Chain multiple filters
    multi_filtered = well.phie.filter('Zone').filter('NTG_Flag')
    print(f"PHIE filtered by Zone and NTG_Flag: {multi_filtered}")

    # Compute statistics - labels will appear in results
    stats = multi_filtered.sums_avg()
    print("\nStatistics with labels:")
    for zone_label, ntg_dict in stats.items():
        print(f"  {zone_label}:")
        for ntg_label, stat_dict in ntg_dict.items():
            print(f"    {ntg_label}: mean={stat_dict['mean']:.4f}, count={stat_dict['count']}")
    print()


# =============================================================================
# Example 5: Computing Statistics
# =============================================================================

def example_statistics():
    """Compute statistics on filtered data."""
    print("Example 5: Computing Statistics")
    print("-" * 50)

    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    well = manager.well_12_3_2_B
    well.get_property('Zone').type = 'discrete'

    # Compute sum-averaged statistics
    stats = well.phie.filter('Zone').sums_avg()
    print("Sum-averaged PHIE by Zone:")
    for zone, avg_value in stats.items():
        print(f"  Zone {zone}: {avg_value:.4f}")

    # Compute statistics for multiple filters
    well.get_property('NTG_Flag').type = 'discrete'
    nested_stats = well.phie.filter('Zone').filter('NTG_Flag').sums_avg()
    print("\nNested statistics (Zone -> NTG_Flag):")
    for zone, ntg_dict in nested_stats.items():
        print(f"  Zone {zone}:")
        for ntg_flag, avg_value in ntg_dict.items():
            print(f"    NTG_Flag {ntg_flag}: {avg_value:.4f}")
    print()


# =============================================================================
# Example 6: Advanced LAS File Handling
# =============================================================================

def example_advanced_las_handling():
    """Work with LAS files directly for metadata inspection."""
    print("Example 6: Advanced LAS File Handling")
    print("-" * 50)

    # Load LAS file without adding to manager
    las = LasFile("path/to/well1.las")

    # Inspect metadata before loading data
    print(f"Well name: {las.well_name}")
    print(f"Depth column: {las.depth_column}")
    print(f"NULL value: {las.null_value}")
    print(f"Available curves: {list(las.curves.keys())}")

    # Update curve metadata (aliases, types, unit conversions)
    las.update_curve('PHIE_2025', type='continuous', alias='PHIE')
    las.update_curve('ResFlag_2025', type='discrete', alias='ResFlag')
    las.update_curve('PERM_Lam_2025', multiplier=0.001, alias='PERM_D')

    # Bulk update multiple curves
    las.bulk_update_curves({
        'Cerisa_facies_LF': {'type': 'discrete', 'alias': 'Facies'},
        'NTG_2025': {'alias': 'NTG'},
        'SW_2025': {'alias': 'SW'}
    })

    # Now load the data (lazy loading)
    df = las.data
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()


# =============================================================================
# Example 7: Multi-Well Analysis
# =============================================================================

def example_multi_well_analysis():
    """Analyze data across multiple wells."""
    print("Example 7: Multi-Well Analysis")
    print("-" * 50)

    manager = WellDataManager()

    # Load multiple wells
    manager.load_las("path/to/well1.las")
    manager.load_las("path/to/well2.las")
    manager.load_las("path/to/well3.las")

    # Analyze each well
    results = {}
    for well_name in manager.wells:
        well = getattr(manager, well_name)

        # Set property types
        well.get_property('Zone').type = 'discrete'

        # Compute statistics
        stats = well.phie.filter('Zone').sums_avg()
        results[well_name] = stats

    # Display results
    print("PHIE statistics by Zone for all wells:")
    for well_name, stats in results.items():
        print(f"\n{well_name}:")
        for zone, avg_value in stats.items():
            print(f"  Zone {zone}: {avg_value:.4f}")
    print()


# =============================================================================
# Example 8: Exporting Data to DataFrame
# =============================================================================

def example_dataframe_export():
    """Export well data to pandas DataFrame with various options."""
    print("Example 8: Exporting to DataFrame")
    print("-" * 50)

    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    well = manager.well_12_3_2_B

    # 1. Export all properties (automatically resamples to first property's grid)
    # Note: By default, uses the first property added as reference
    # (typically the first property from the first LAS file loaded)
    df_all = well.to_dataframe()
    print("All properties (auto-resampled to first property's grid):")
    print(df_all.head())
    print(f"Columns: {list(df_all.columns)}\n")

    # 2. Export with specific reference property (auto-resamples to PHIE's grid)
    df_ref = well.to_dataframe(reference_property='PHIE')
    print("Using PHIE as reference (auto-resampled):")
    print(f"Shape: {df_ref.shape}\n")

    # 3. Include only specific properties
    df_subset = well.to_dataframe(include=['PHIE', 'SW', 'PERM'])
    print("Only PHIE, SW, and PERM:")
    print(df_subset.head())
    print()

    # 4. Exclude specific properties (useful when you want most properties)
    df_exclude = well.to_dataframe(exclude=['QC_Flag', 'Temp_Data'])
    print("All properties except QC_Flag and Temp_Data:")
    print(f"Columns: {list(df_exclude.columns)}\n")

    # 5. Combine reference property with filtering
    df_combined = well.to_dataframe(
        reference_property='PHIE',
        exclude=['Zone', 'Facies']
    )
    print("Using PHIE reference, excluding Zone and Facies:")
    print(f"Columns: {list(df_combined.columns)}\n")

    # 6. Export without auto-resampling (only if data is already aligned)
    # Note: Only use auto_resample=False if you've pre-aligned the data
    # well.resample(depth_step=0.1)
    # df_no_resample = well.to_dataframe(auto_resample=False)
    print()

    # 7. Typical workflow: just call to_dataframe() and it handles everything
    df = well.to_dataframe(exclude=['QC_Flag'])
    print("Typical usage - simple and automatic:")
    print(f"Shape: {df.shape}, Columns: {len(df.columns)}")
    print()

    # 8. Export with discrete property labels
    # Set up labels for discrete properties
    zone_prop = well.get_property('Zone')
    zone_prop.type = 'discrete'
    zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir', 2: 'Transition'}

    # By default, labels are applied
    df_with_labels = well.to_dataframe()
    print("DataFrame with discrete labels applied:")
    print(df_with_labels[['DEPT', 'Zone']].head())
    print()

    # Can disable labels if needed
    df_without_labels = well.to_dataframe(discrete_labels=False)
    print("DataFrame without labels (numeric values):")
    print(df_without_labels[['DEPT', 'Zone']].head())
    print()


# =============================================================================
# Example 9: Working with Raw Data
# =============================================================================

def example_raw_data_access():
    """Access raw data directly for custom analysis."""
    print("Example 9: Raw Data Access")
    print("-" * 50)

    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    well = manager.well_12_3_2_B

    # Access property as numpy array
    phie_values = well.phie.values
    print(f"PHIE values (numpy array): {phie_values[:5]}")

    # Access depth values
    depth = well.phie.depth
    print(f"\nDepth values: {depth[:5]}")

    # Get property metadata
    phie_prop = well.get_property('PHIE')
    print(f"\nProperty name: {phie_prop.name}")
    print(f"Property type: {phie_prop.type}")
    print(f"Parent well: {phie_prop.parent_well.name}")

    # Export single property to DataFrame
    df_prop = phie_prop.to_dataframe()
    print(f"\nProperty as DataFrame:\n{df_prop.head()}")
    print()


# =============================================================================
# Example 10: Error Handling
# =============================================================================

def example_error_handling():
    """Demonstrate proper error handling."""
    print("Example 10: Error Handling")
    print("-" * 50)

    manager = WellDataManager()

    try:
        # Attempt to load non-existent file
        manager.load_las("path/to/nonexistent.las")
    except Exception as e:
        print(f"Caught error: {type(e).__name__}: {e}")

    manager.load_las("path/to/well1.las")
    well = manager.well_12_3_2_B

    try:
        # Attempt to access non-existent property
        prop = well.get_property('NonExistentProperty')
    except Exception as e:
        print(f"Caught error: {type(e).__name__}: {e}")

    try:
        # Attempt to filter by continuous property (should be discrete)
        well.phie.filter('PHIE')
    except Exception as e:
        print(f"Caught error: {type(e).__name__}: {e}")
    print()


# =============================================================================
# Example 11: Complete Workflow
# =============================================================================

def example_complete_workflow():
    """Complete workflow from loading to analysis."""
    print("Example 11: Complete Workflow")
    print("-" * 50)

    # Step 1: Initialize and load data
    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    # Step 2: Access well
    well = manager.well_12_3_2_B
    print(f"Loaded well: {well.name}")
    print(f"Properties: {well.properties}")

    # Step 3: Configure property types
    discrete_props = ['Zone', 'NTG_Flag', 'Facies']
    for prop_name in discrete_props:
        if prop_name in well.properties:
            well.get_property(prop_name).type = 'discrete'

    # Step 4: Perform analysis
    print("\nAnalysis Results:")
    print("=" * 50)

    # Porosity analysis by zone
    phie_by_zone = well.phie.filter('Zone').sums_avg()
    print("\nPorosity (PHIE) by Zone:")
    for zone, value in phie_by_zone.items():
        print(f"  Zone {zone}: {value:.4f}")

    # Permeability analysis by zone and NTG flag
    if 'PERM' in well.properties and 'NTG_Flag' in well.properties:
        perm_stats = well.PERM.filter('Zone').filter('NTG_Flag').sums_avg()
        print("\nPermeability by Zone and NTG Flag:")
        for zone, ntg_dict in perm_stats.items():
            print(f"  Zone {zone}:")
            for ntg, value in ntg_dict.items():
                print(f"    NTG_Flag {ntg}: {value:.2f} mD")

    print("\nWorkflow complete!")
    print()


# =============================================================================
# Example 12: Exporting to LAS Format
# =============================================================================

def example_las_export():
    """Export well data back to LAS format."""
    print("Example 12: Exporting to LAS Format")
    print("-" * 50)

    manager = WellDataManager()
    manager.load_las("path/to/well1.las")

    well = manager.well_12_3_2_B

    # 1. Export all properties to LAS
    well.export_to_las("output_all.las")
    print("Exported all properties to output_all.las")

    # 2. Export specific properties only
    well.export_to_las("output_subset.las", include=['PHIE', 'SW', 'PERM'])
    print("Exported PHIE, SW, PERM to output_subset.las")

    # 3. Export all except certain properties
    well.export_to_las("output_no_qc.las", exclude=['QC_Flag', 'Temp_Data'])
    print("Exported all except QC flags to output_no_qc.las")

    # 4. Export single property to LAS
    phie_prop = well.get_property('PHIE')
    phie_prop.export_to_las("phie_only.las")
    print("Exported PHIE property to phie_only.las")

    # 5. Export filtered property with filters included
    zone_prop = well.get_property('Zone')
    zone_prop.type = 'discrete'
    zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir'}

    ntg_prop = well.get_property('NTG_Flag')
    ntg_prop.type = 'discrete'
    ntg_prop.labels = {0: 'NonNet', 1: 'Net'}

    # Filtered property includes secondary properties in export
    filtered = well.phie.filter('Zone').filter('NTG_Flag')
    filtered.export_to_las("filtered_phie.las")
    print("Exported filtered PHIE with Zone and NTG_Flag to filtered_phie.las")

    # 6. Export with discrete labels stored in parameter section
    well.export_to_las("output_with_labels.las", store_labels=True)
    print("Exported with discrete labels stored in ~Parameter section (data remains numeric)")

    # 7. Export with custom null value
    well.export_to_las("output_custom_null.las", null_value=-9999.0)
    print("Exported with custom null value (-9999.0)")

    # 8. Standalone export using LasFile.export_las()
    from well_log_toolkit import LasFile

    # Create any DataFrame with DEPT column
    df = well.to_dataframe(include=['DEPT', 'PHIE', 'SW'])

    # Export with custom unit mappings and labels
    LasFile.export_las(
        filepath="standalone_export.las",
        well_name="12/3-2 B",
        df=df,
        unit_mappings={'DEPT': 'm', 'PHIE': 'v/v', 'SW': 'v/v'},
        discrete_labels={'Zone': {0: 'NonReservoir', 1: 'Reservoir'}}
    )
    print("Exported using standalone LasFile.export_las() method")

    # 9. Round-trip: Export and reload with automatic discrete detection
    # Export with discrete properties and labels
    zone_prop.type = 'discrete'
    zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir'}
    ntg_prop.type = 'discrete'
    ntg_prop.labels = {0: 'NonNet', 1: 'Net'}

    well.export_to_las("roundtrip.las", store_labels=True)
    print("\nExported with discrete properties marked")

    # Reload the file - discrete properties and labels are automatically loaded
    manager2 = WellDataManager()
    manager2.load_las("roundtrip.las")
    well2 = manager2.well_12_3_2_B

    # Check that discrete types and labels were preserved
    zone_reloaded = well2.get_property('Zone')
    print(f"Zone type after reload: {zone_reloaded.type}")  # 'discrete'
    print(f"Zone labels after reload: {zone_reloaded.labels}")  # {0: 'NonReservoir', 1: 'Reservoir'}

    # 10. Template-based export: Preserve original metadata
    # This prevents data erosion when updating existing LAS files
    print("\n10. Template-based export (preserves metadata):")

    # Load a LAS file
    manager3 = WellDataManager()
    manager3.load_las("path/to/original.las")
    well3 = manager3.well_12_3_2_B

    # Access the original LAS object
    original_las = well3.original_las
    print(f"Original LAS file loaded: {original_las is not None}")
    print(f"Source LAS files: {well3.sources}")

    # Properties know their source
    zone_prop = well3.get_property('Zone')
    print(f"Zone property source: {zone_prop.source}")

    # Add new discrete mappings or modify properties
    zone_prop = well3.get_property('Zone')
    zone_prop.type = 'discrete'
    zone_prop.labels = {0: 'NonReservoir', 1: 'Reservoir', 2: 'Transition'}

    # Export using original LAS as template (preserves all metadata)
    well3.export_to_las("updated.las", use_template=True)
    print("Exported with original metadata preserved using use_template=True")

    # Alternative: specify a specific template
    well3.export_to_las("updated2.las", use_template=original_las)
    print("Exported with specific LAS file as template")

    # Without template (creates new minimal LAS file)
    well3.export_to_las("minimal.las", use_template=False)
    print("Exported without template (minimal metadata)")

    print("\nTemplate mode preserves:")
    print("  - Original ~Version information")
    print("  - Original ~Well parameters (FLD, COMP, LOC, etc.)")
    print("  - Original ~Curve descriptions")
    print("  - Original ~Parameter entries (except discrete labels)")
    print("This ensures zero data erosion when updating LAS files!")

    print("\nNote: All exported LAS files are in LAS 2.0 format")
    print("Discrete properties are marked with DISCRETE_PROPS parameter")
    print("Labels are stored in the ~Parameter section and automatically loaded")
    print()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("WELL LOG TOOLKIT - EXAMPLES")
    print("=" * 50)
    print()

    # Note: These examples use placeholder file paths.
    # Replace "path/to/well1.las" with actual LAS file paths to run the examples.

    print("To run these examples, replace the placeholder file paths")
    print("with actual LAS file paths from your dataset.")
    print()

    # Uncomment the examples you want to run:

    # example_basic_usage()
    # example_method_chaining()
    # example_property_types()
    # example_filtering()
    # example_statistics()
    # example_advanced_las_handling()
    # example_multi_well_analysis()
    # example_dataframe_export()
    # example_raw_data_access()
    # example_error_handling()
    # example_complete_workflow()
    # example_las_export()
