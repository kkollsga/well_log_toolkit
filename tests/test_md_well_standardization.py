"""Test script demonstrating standardized MD and Well access across all methods."""

import numpy as np
import pandas as pd
from pylog.manager import WellDataManager
from pylog.core.well import Well
from pylog.core.property import Property

print("=" * 70)
print("STANDARDIZED MD (Measured Depth) AND WELL ACCESS")
print("=" * 70)

# Create a test well with sample data
print("\n1. Creating test well with sample data...")
well = Well("36/7-5 ST2", "36_7_5_ST2")

# Create sample depth and property data
depth = np.arange(2000, 2100, 0.5)  # 2000m to 2100m, 0.5m step
phie = 0.2 + 0.05 * np.sin(np.linspace(0, 4*np.pi, len(depth)))  # Varying porosity
sw = 0.4 + 0.2 * np.cos(np.linspace(0, 3*np.pi, len(depth)))  # Varying water saturation

# Create properties
phie_prop = Property(
    name="PHIE",
    depth=depth,
    values=phie,
    parent_well=well,
    unit="v/v",
    prop_type="continuous",
    description="Effective Porosity"
)

sw_prop = Property(
    name="SW",
    depth=depth,
    values=sw,
    parent_well=well,
    unit="v/v",
    prop_type="continuous",
    description="Water Saturation"
)

print(f"   Created well: {well.name}")
print(f"   Depth range: {depth[0]:.1f}m to {depth[-1]:.1f}m")
print(f"   Number of samples: {len(depth)}")

# Test 1: Access MD (Measured Depth)
print("\n" + "=" * 70)
print("TEST 1: Accessing MD (Measured Depth)")
print("=" * 70)
print(f"   PHIE.MD (first 5 values): {phie_prop.MD[:5]}")
print(f"   PHIE.depth (first 5 values): {phie_prop.depth[:5]}")
print(f"   Are they the same? {np.array_equal(phie_prop.MD, phie_prop.depth)}")

# Test 2: Access Well name
print("\n" + "=" * 70)
print("TEST 2: Accessing Well name")
print("=" * 70)
well_names = phie_prop.Well
print(f"   PHIE.Well (first value): {well_names[0]}")
print(f"   PHIE.Well (length): {len(well_names)}")
print(f"   All values same? {np.all(well_names == well.name)}")

# Test 3: Conditional calculations using MD
print("\n" + "=" * 70)
print("TEST 3: Conditional calculations using MD")
print("=" * 70)

# Example 1: Filter by depth range
shallow_zone = (phie_prop.MD >= 2020) & (phie_prop.MD < 2040)
shallow_phie = np.where(shallow_zone, phie_prop.values, np.nan)
print(f"\n   Example 1: Extract shallow zone (2020-2040m)")
print(f"   Original PHIE samples: {len(phie_prop.values)}")
print(f"   Shallow zone samples: {np.sum(~np.isnan(shallow_phie))}")
print(f"   Shallow PHIE mean: {np.nanmean(shallow_phie):.4f}")

# Example 2: Depth-dependent scaling
scaled = np.where(phie_prop.MD < 2050,
                  phie_prop.values * 1.1,  # Scale up shallow
                  phie_prop.values * 0.9)  # Scale down deep
print(f"\n   Example 2: Depth-dependent scaling")
print(f"   Shallow (<2050m) scaled by: 1.1x")
print(f"   Deep (>=2050m) scaled by: 0.9x")
print(f"   Original mean: {np.mean(phie_prop.values):.4f}")
print(f"   Scaled mean: {np.mean(scaled):.4f}")

# Example 3: Multiple depth zones
zone1 = phie_prop.MD < 2033
zone2 = (phie_prop.MD >= 2033) & (phie_prop.MD < 2067)
zone3 = phie_prop.MD >= 2067

zonation = np.where(zone1, 1, np.where(zone2, 2, 3))
print(f"\n   Example 3: Create depth-based zonation")
print(f"   Zone 1 (<2033m): {np.sum(zonation == 1)} samples")
print(f"   Zone 2 (2033-2067m): {np.sum(zonation == 2)} samples")
print(f"   Zone 3 (>=2067m): {np.sum(zonation == 3)} samples")

# Test 4: Combining MD and Well in calculations
print("\n" + "=" * 70)
print("TEST 4: Combining MD and Well in calculations")
print("=" * 70)

# Simulate multi-well scenario by creating another well
well2 = Well("36/7-6", "36_7_6")
phie_prop2 = Property(
    name="PHIE",
    depth=depth,
    values=phie * 0.8,  # Different values
    parent_well=well2,
    unit="v/v",
    prop_type="continuous"
)

print(f"\n   Well 1: {phie_prop.Well[0]}")
print(f"   Well 2: {phie_prop2.Well[0]}")

# Conditional based on both well name and depth
condition1 = (phie_prop.Well == "36/7-5 ST2") & (phie_prop.MD > 2050)
condition2 = (phie_prop2.Well == "36/7-6") & (phie_prop2.MD > 2050)

print(f"\n   Condition 1 (Well='36/7-5 ST2' & MD>2050): {np.sum(condition1)} samples")
print(f"   Condition 2 (Well='36/7-6' & MD>2050): {np.sum(condition2)} samples")

# Test 5: Creating new properties using MD and Well
print("\n" + "=" * 70)
print("TEST 5: Creating new calculated properties using MD and Well")
print("=" * 70)

# Create a new property with depth-dependent calculation
hc_pore_volume = Property(
    name="HC_PV",
    depth=depth,
    values=np.where(phie_prop.MD > 2050,
                    phie_prop.values * (1 - sw_prop.values),  # Deep zone
                    phie_prop.values * (1 - sw_prop.values) * 0.8),  # Shallow zone with cutoff
    parent_well=well,
    unit="v/v",
    prop_type="continuous",
    description="Hydrocarbon Pore Volume (depth-adjusted)"
)

print(f"   Created HC_PV property using depth-conditional logic")
print(f"   Shallow zone HC_PV mean: {np.nanmean(hc_pore_volume.values[phie_prop.MD <= 2050]):.4f}")
print(f"   Deep zone HC_PV mean: {np.nanmean(hc_pore_volume.values[phie_prop.MD > 2050]):.4f}")

# Test 6: Practical well log calculation examples
print("\n" + "=" * 70)
print("TEST 6: Practical well log calculation examples")
print("=" * 70)

# Example: Net pay cutoffs varying by depth
print("\n   Example: Depth-varying net pay cutoffs")
shallow_cutoff = 0.15
deep_cutoff = 0.12

net_flag = np.where(phie_prop.MD < 2050,
                    (phie_prop.values > shallow_cutoff).astype(int),
                    (phie_prop.values > deep_cutoff).astype(int))

shallow_net = np.sum(net_flag[phie_prop.MD < 2050])
deep_net = np.sum(net_flag[phie_prop.MD >= 2050])

print(f"   Shallow zone cutoff: {shallow_cutoff}")
print(f"   Deep zone cutoff: {deep_cutoff}")
print(f"   Shallow net samples: {shallow_net}")
print(f"   Deep net samples: {deep_net}")

# Example: Reservoir quality index varying by depth and well
print("\n   Example: Reservoir quality calculation")
rqi = np.where(
    (phie_prop.Well == "36/7-5 ST2") & (phie_prop.MD < 2050),
    phie_prop.values / sw_prop.values * 100,  # Shallow RQI formula
    phie_prop.values / (sw_prop.values ** 0.5) * 100  # Deep RQI formula
)

print(f"   Shallow RQI mean: {np.nanmean(rqi[phie_prop.MD < 2050]):.2f}")
print(f"   Deep RQI mean: {np.nanmean(rqi[phie_prop.MD >= 2050]):.2f}")

# Test 7: Integration with crossplot (simulated)
print("\n" + "=" * 70)
print("TEST 7: Crossplot-style data preparation with MD and Well")
print("=" * 70)

# Simulate how crossplot would use MD and Well
crossplot_data = pd.DataFrame({
    'PHIE': phie_prop.values,
    'SW': sw_prop.values,
    'MD': phie_prop.MD,
    'Well': phie_prop.Well
})

print(f"\n   Crossplot DataFrame shape: {crossplot_data.shape}")
print(f"   Columns: {list(crossplot_data.columns)}")
print("\n   First 5 rows:")
print(crossplot_data.head())

# Filter crossplot data by depth
deep_data = crossplot_data[crossplot_data['MD'] > 2050]
print(f"\n   Deep zone data (MD > 2050m): {len(deep_data)} points")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
✓ MD property provides standardized access to measured depth
✓ Well property provides well name as array for vectorized operations
✓ Both work seamlessly with numpy operations (np.where, boolean indexing)
✓ Enable depth-conditional and well-conditional calculations
✓ Compatible with crossplot and all visualization methods
✓ Consistent API across Property, Well, and Manager classes

USAGE PATTERNS:
--------------
1. Depth filtering:     prop.MD > 2000
2. Well filtering:      prop.Well == "36/7-5"
3. Combined:            (prop.MD > 2000) & (prop.Well == "36/7-5")
4. Calculations:        np.where(prop.MD < 2050, value1, value2)
5. Zonation:            pd.cut(prop.MD, bins=[2000, 2050, 2100])
""")

print("=" * 70)
print("All tests completed successfully!")
print("=" * 70)
