"""
Test flexible loading patterns for load_properties() and load_tops()
Supports: multi-well, single-well with name, single-well with default
"""
import pandas as pd
import numpy as np
from logsuite import WellDataManager

print("=" * 80)
print("Testing Flexible Loading Patterns (v0.1.57)")
print("=" * 80)

# =============================================================================
# TEST 1: load_properties() - Multi-well pattern
# =============================================================================
print("\n1. load_properties() - Multi-well pattern (well_col specified)")
print("-" * 80)

manager = WellDataManager()

# Create multi-well DataFrame
df_multi = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 A', '12/3-4 B', '12/3-4 B'],
    'DEPT': [2850.0, 2851.0, 2852.0, 2850.0, 2851.0],
    'PHIE': [0.20, 0.22, 0.19, 0.18, 0.21],
    'SW': [0.30, 0.32, 0.28, 0.35, 0.33]
})

manager.load_properties(
    df_multi,
    source_name='Petrophysics',
    well_col='Well',  # Explicitly specify column
    unit_mappings={'PHIE': 'v/v', 'SW': 'v/v'}
)

print(f"Wells created: {manager.wells}")
print(f"Well A sources: {manager.well_12_3_4_A.sources}")
print(f"Well B sources: {manager.well_12_3_4_B.sources}")
print(f"Well A PHIE values: {manager.well_12_3_4_A.PHIE.values}")
print(f"Well B PHIE values: {manager.well_12_3_4_B.PHIE.values}")

# =============================================================================
# TEST 2: load_properties() - Single-well with explicit name
# =============================================================================
print("\n2. load_properties() - Single-well with explicit name")
print("-" * 80)

manager2 = WellDataManager()

# Create single-well DataFrame (no well column)
df_single = pd.DataFrame({
    'DEPT': [2850.0, 2851.0, 2852.0],
    'PERM': [150, 200, 120]
})

manager2.load_properties(
    df_single,
    source_name='CoreData',
    well_col=None,  # No well column
    well_name='36/7-5 A',  # Explicit well name
    unit_mappings={'PERM': 'mD'}
)

print(f"Wells created: {manager2.wells}")
print(f"Well sources: {manager2.well_36_7_5_A.sources}")
print(f"Well PERM values: {manager2.well_36_7_5_A.PERM.values}")

# =============================================================================
# TEST 3: load_properties() - Single-well with default name "Well"
# =============================================================================
print("\n3. load_properties() - Single-well with default name 'Well'")
print("-" * 80)

manager3 = WellDataManager()

df_default = pd.DataFrame({
    'DEPT': [2850.0, 2851.0, 2852.0],
    'RHOB': [2.35, 2.40, 2.38]
})

manager3.load_properties(
    df_default,
    source_name='Logs',
    well_col=None,  # No well column, no well name
    unit_mappings={'RHOB': 'g/cm3'}
)

print(f"Wells created: {manager3.wells}")
print(f"Well sources: {manager3.well_Well.sources}")
print(f"Well RHOB values: {manager3.well_Well.RHOB.values}")

# =============================================================================
# TEST 4: load_tops() - Multi-well pattern
# =============================================================================
print("\n4. load_tops() - Multi-well pattern")
print("-" * 80)

manager4 = WellDataManager()

df_tops_multi = pd.DataFrame({
    'Well identifier (Well name)': ['12/3-4 A', '12/3-4 A', '12/3-4 B'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Brent'],
    'MD': [2850.0, 3100.0, 2860.0]
})

manager4.load_tops(df_tops_multi)  # Uses default well_col

print(f"Wells created: {manager4.wells}")
print(f"Well A sources: {manager4.well_12_3_4_A.sources}")
print(f"Well B sources: {manager4.well_12_3_4_B.sources}")
well_tops_a = manager4.well_12_3_4_A.get_property('Well_Tops')
print(f"Well A tops labels: {well_tops_a.labels}")

# =============================================================================
# TEST 5: load_tops() - Single-well with explicit name
# =============================================================================
print("\n5. load_tops() - Single-well with explicit name")
print("-" * 80)

manager5 = WellDataManager()

df_tops_single = pd.DataFrame({
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Cook'],
    'MD': [2850.0, 3100.0, 3400.0]
})

manager5.load_tops(
    df_tops_single,
    well_col=None,
    well_name='36/7-5 B'
)

print(f"Wells created: {manager5.wells}")
print(f"Well sources: {manager5.well_36_7_5_B.sources}")
well_tops_b = manager5.well_36_7_5_B.get_property('Well_Tops')
print(f"Well B tops labels: {well_tops_b.labels}")

# =============================================================================
# TEST 6: load_tops() - Single-well with default name
# =============================================================================
print("\n6. load_tops() - Single-well with default name 'Well'")
print("-" * 80)

manager6 = WellDataManager()

manager6.load_tops(
    df_tops_single,
    well_col=None  # No well_col, no well_name -> defaults to "Well"
)

print(f"Wells created: {manager6.wells}")
print(f"Well sources: {manager6.well_Well.sources}")
well_tops_default = manager6.well_Well.get_property('Well_Tops')
print(f"Default well tops labels: {well_tops_default.labels}")

# =============================================================================
# TEST 7: Mixed usage - Add data to existing well using different patterns
# =============================================================================
print("\n7. Mixed usage - Add data to existing well")
print("-" * 80)

manager7 = WellDataManager()

# First, load with explicit well name
df1 = pd.DataFrame({
    'DEPT': [2850.0, 2851.0, 2852.0],
    'PHIE': [0.20, 0.22, 0.19]
})

manager7.load_properties(
    df1,
    source_name='Logs',
    well_col=None,
    well_name='12/3-4 A'
)

# Then, add more properties to the same well
df2 = pd.DataFrame({
    'DEPT': [2850.0, 2851.0, 2852.0],
    'SW': [0.30, 0.32, 0.28]
})

manager7.load_properties(
    df2,
    source_name='Saturation',
    well_col=None,
    well_name='12/3-4 A'  # Same well
)

print(f"Wells created: {manager7.wells}")
print(f"Well sources: {manager7.well_12_3_4_A.sources}")
print(f"Available properties: {manager7.well_12_3_4_A.properties}")
print(f"PHIE values: {manager7.well_12_3_4_A.PHIE.values}")
print(f"SW values: {manager7.well_12_3_4_A.SW.values}")

print("\n" + "=" * 80)
print("All tests completed successfully! ✓")
print("=" * 80)
