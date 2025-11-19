# Flexible Loading Enhancement (v0.1.57)

## Summary

Enhanced `load_properties()` and `load_tops()` methods to support three flexible loading patterns:

1. **Multi-well loading**: Groups DataFrame by well column (existing behavior)
2. **Single-well with explicit name**: Loads all data to a named well without requiring a well column
3. **Single-well with default name**: Simplest pattern, loads all data to generic "Well"

## Changes Made

### 1. `load_properties()` Enhancement

**New Parameters:**
- `well_col: Optional[str] = "Well"` - Now optional, set to None for single-well mode
- `well_name: Optional[str] = None` - Specify well name when well_col=None

**Usage Patterns:**

```python
# Pattern 1: Multi-well (groups by well column)
df_multi = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 B'],
    'DEPT': [2850.0, 2851.0, 2850.0],
    'PHIE': [0.20, 0.22, 0.18]
})
manager.load_properties(df_multi, well_col='Well')
# Creates: well_12_3_4_A and well_12_3_4_B

# Pattern 2: Single-well with explicit name
df_single = pd.DataFrame({
    'DEPT': [2850.0, 2851.0, 2852.0],
    'PERM': [150, 200, 120]
})
manager.load_properties(df_single, well_col=None, well_name='36/7-5 A')
# Creates: well_36_7_5_A

# Pattern 3: Single-well with default name
manager.load_properties(df_single, well_col=None)
# Creates: well_Well (generic well)
```

### 2. `load_tops()` Enhancement

**New Parameters:**
- `well_col: Optional[str] = "Well identifier (Well name)"` - Now optional
- `well_name: Optional[str] = None` - Specify well name when well_col=None

**Usage Patterns:**

```python
# Pattern 1: Multi-well (groups by well column)
df_tops_multi = pd.DataFrame({
    'Well identifier (Well name)': ['12/3-4 A', '12/3-4 B'],
    'Surface': ['Top_Brent', 'Top_Brent'],
    'MD': [2850.0, 2860.0]
})
manager.load_tops(df_tops_multi)  # Uses default well_col

# Pattern 2: Single-well with explicit name
df_tops_single = pd.DataFrame({
    'Surface': ['Top_Brent', 'Top_Statfjord'],
    'MD': [2850.0, 3100.0]
})
manager.load_tops(df_tops_single, well_col=None, well_name='36/7-5 B')

# Pattern 3: Single-well with default name
manager.load_tops(df_tops_single, well_col=None)
# Loads to well_Well
```

## Benefits

1. **Flexibility**: Supports both multi-well and single-well workflows
2. **Simplicity**: Single-well users don't need to add a well column to their DataFrames
3. **Backward compatibility**: Existing code continues to work (defaults unchanged)
4. **Robustness**: Clear error messages for missing columns based on loading pattern
5. **Convenience**: Default "Well" name for quickest possible loading

## Implementation Details

### Internal Logic

Both methods now check `well_col` parameter:

```python
if well_col is None:
    # SINGLE-WELL MODE
    target_well_name = well_name if well_name is not None else "Well"
    # Validate only depth/discrete columns (no well column needed)
    # Create fake grouped structure: [(well_name, df)]
else:
    # MULTI-WELL MODE (existing behavior)
    # Validate well column exists
    # Group by well column: df.groupby(well_col)
```

### Modified Files

1. `/well_log_toolkit/manager.py`:
   - Updated `load_properties()` signature and implementation
   - Updated `load_tops()` signature and implementation
   - Enhanced docstrings with examples for all three patterns

2. `/well_log_toolkit/__init__.py`:
   - Version bumped to 0.1.57

### Testing

All patterns tested successfully in `test_flexible_loading.py`:
- ✓ Multi-well loading with well column
- ✓ Single-well loading with explicit well name
- ✓ Single-well loading with default "Well" name
- ✓ Mixed usage (adding data to existing wells)
- ✓ Both load_properties() and load_tops()

## Use Cases

### Single-Well Analysis

Perfect for users working with a single well who don't want to manage well objects:

```python
manager = WellDataManager()

# Load logs
logs_df = pd.read_csv("logs.csv")  # Just DEPT and properties
manager.load_properties(logs_df, well_col=None, well_name="MyWell")

# Load tops
tops_df = pd.read_csv("tops.csv")  # Just Surface and MD
manager.load_tops(tops_df, well_col=None, well_name="MyWell")

# Access data
well = manager.well_MyWell
stats = well.PHIE.filter('Well_Tops').sums_avg()
```

### Quick Prototyping

Fastest possible loading for testing:

```python
manager = WellDataManager()

df = pd.DataFrame({'DEPT': [2850, 2851], 'PHIE': [0.2, 0.22]})
manager.load_properties(df, well_col=None)  # Instant loading

# Access immediately
print(manager.well_Well.PHIE.values)
```

### Multi-Well Projects

Original behavior maintained for multi-well workflows:

```python
# Load multiple wells from single DataFrame
df = pd.read_excel("all_wells.xlsx")
manager.load_properties(df, well_col='Well Name')  # Groups by well

# Each well created automatically
for well_name in manager.wells:
    print(well_name)
```

## Backward Compatibility

All existing code continues to work:
- Default `well_col` values unchanged
- Existing multi-well workflows unaffected
- API additions only (no breaking changes)
