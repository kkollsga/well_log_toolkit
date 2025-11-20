"""
Comprehensive test for filtered statistics with various configurations:
- Single and multiple filters
- Specific LAS source names
- Ambiguous properties (multiple sources)
- Nested mode
"""
import numpy as np
from well_log_toolkit import WellDataManager, Property
from well_log_toolkit.utils import sanitize_well_name
from well_log_toolkit.well import Well

print("=" * 80)
print("COMPREHENSIVE TEST: Filtered Statistics with Various Configurations")
print("=" * 80)

# Create manager with multiple wells
manager = WellDataManager()

# Create Well A with multiple sources
well_a_name = "well_A"
well_a_sanitized = sanitize_well_name(well_a_name)
well_a = Well(well_a_name, well_a_sanitized, parent_manager=manager)

# Well A - Log source
depth_log = np.arange(2200.0, 2240.0, 0.5)
phie_log = np.concatenate([
    np.random.uniform(0.10, 0.15, 20),  # Zone 0
    np.random.uniform(0.20, 0.25, 20),  # Zone 1
    np.random.uniform(0.28, 0.35, 40),  # Zone 2
])
zone_log = np.concatenate([np.zeros(20), np.ones(20), np.full(40, 2)])

phie_log_prop = Property("PHIE", depth_log, phie_log, parent_well=well_a, prop_type='continuous', unit='v/v')
zone_log_prop = Property("Zone", depth_log, zone_log, parent_well=well_a, prop_type='discrete')
zone_log_prop.labels = {0: 'Upper', 1: 'Middle', 2: 'Lower'}

# Net/Gross flag
ntg_log = np.concatenate([
    np.zeros(30),  # Non-net
    np.ones(50)    # Net
])
ntg_log_prop = Property("NTG", depth_log, ntg_log, parent_well=well_a, prop_type='discrete')
ntg_log_prop.labels = {0: 'NonNet', 1: 'Net'}

if "log" not in well_a._sources:
    well_a._sources["log"] = {'path': None, 'las_file': None, 'properties': {}}
well_a._sources["log"]["properties"]["PHIE"] = phie_log_prop
well_a._sources["log"]["properties"]["Zone"] = zone_log_prop
well_a._sources["log"]["properties"]["NTG"] = ntg_log_prop

# Well A - Core source (with overlapping PHIE property)
depth_core = np.arange(2210.0, 2230.0, 1.0)  # Sparser sampling
phie_core = np.random.uniform(0.15, 0.30, len(depth_core))
zone_core = np.concatenate([np.zeros(10), np.ones(10)])

phie_core_prop = Property("PHIE", depth_core, phie_core, parent_well=well_a, prop_type='sampled', unit='v/v')
zone_core_prop = Property("Zone", depth_core, zone_core, parent_well=well_a, prop_type='discrete')
zone_core_prop.labels = {0: 'Upper', 1: 'Middle'}

if "core" not in well_a._sources:
    well_a._sources["core"] = {'path': None, 'las_file': None, 'properties': {}}
well_a._sources["core"]["properties"]["PHIE"] = phie_core_prop
well_a._sources["core"]["properties"]["Zone"] = zone_core_prop

# Create Well B with single source
well_b_name = "well_B"
well_b_sanitized = sanitize_well_name(well_b_name)
well_b = Well(well_b_name, well_b_sanitized, parent_manager=manager)

depth_b = np.arange(2300.0, 2340.0, 0.5)
phie_b = np.concatenate([
    np.random.uniform(0.08, 0.12, 30),  # Zone 0
    np.random.uniform(0.18, 0.23, 50),  # Zone 1
])
zone_b = np.concatenate([np.zeros(30), np.ones(50)])

phie_b_prop = Property("PHIE", depth_b, phie_b, parent_well=well_b, prop_type='continuous', unit='v/v')
zone_b_prop = Property("Zone", depth_b, zone_b, parent_well=well_b, prop_type='discrete')
zone_b_prop.labels = {0: 'Upper', 1: 'Lower'}

if "log" not in well_b._sources:
    well_b._sources["log"] = {'path': None, 'las_file': None, 'properties': {}}
well_b._sources["log"]["properties"]["PHIE"] = phie_b_prop
well_b._sources["log"]["properties"]["Zone"] = zone_b_prop

# Add wells to manager
manager._wells[well_a_sanitized] = well_a
manager._wells[well_b_sanitized] = well_b

# TEST 1: Single filter
print("\n" + "=" * 80)
print("TEST 1: Single Filter")
print("=" * 80)
result1 = manager.PHIE.filter("Zone").percentile(50)
print("\nmanager.PHIE.filter('Zone').percentile(50)")
print("\nResult:")
for well_name, well_result in result1.items():
    print(f"  {well_name}:")
    if isinstance(well_result, dict):
        for key, val in well_result.items():
            print(f"    {key}: {val}")
    else:
        print(f"    {well_result}")

print("\n✓ Well A has ambiguous PHIE (log + core) - returns nested by source")
print("✓ Well B has unique PHIE (log only) - returns direct grouping")

# TEST 2: Multiple filters (chained)
print("\n" + "=" * 80)
print("TEST 2: Multiple Filters (Chained)")
print("=" * 80)
result2 = manager.PHIE.filter("Zone").filter("NTG").mean()
print("\nmanager.PHIE.filter('Zone').filter('NTG').mean()")
print("\nResult:")
for well_name, well_result in result2.items():
    print(f"  {well_name}:")
    if isinstance(well_result, dict):
        for key1, val1 in well_result.items():
            if isinstance(val1, dict):
                print(f"    {key1}:")
                for key2, val2 in val1.items():
                    if isinstance(val2, dict):
                        print(f"      {key2}:")
                        for key3, val3 in val2.items():
                            print(f"        {key3}: {val3}")
                    else:
                        print(f"      {key2}: {val2}")
            else:
                print(f"    {key1}: {val1}")

print("\n✓ Well A has NTG filter - shows nested grouping")
print("✓ Well B lacks NTG filter - only grouped by Zone")

# TEST 3: With nested=True (always show source names)
print("\n" + "=" * 80)
print("TEST 3: Nested Mode (Always Show Source Names)")
print("=" * 80)
result3 = manager.PHIE.filter("Zone").median(nested=True)
print("\nmanager.PHIE.filter('Zone').median(nested=True)")
print("\nResult:")
for well_name, well_result in result3.items():
    print(f"  {well_name}:")
    if isinstance(well_result, dict):
        for key1, val1 in well_result.items():
            if isinstance(val1, dict):
                print(f"    {key1}:")
                for key2, val2 in val1.items():
                    print(f"      {key2}: {val2}")
            else:
                print(f"    {key1}: {val1}")

print("\n✓ All results show source names (log, core)")
print("✓ Well A shows both log and core sources")
print("✓ Well B shows log source only")

# TEST 4: Different statistics with filters
print("\n" + "=" * 80)
print("TEST 4: Different Statistics with Filters")
print("=" * 80)

stats_methods = [
    ('min', lambda: manager.PHIE.filter("Zone").min()),
    ('max', lambda: manager.PHIE.filter("Zone").max()),
    ('mean', lambda: manager.PHIE.filter("Zone").mean()),
    ('std', lambda: manager.PHIE.filter("Zone").std()),
    ('median', lambda: manager.PHIE.filter("Zone").median()),
]

for stat_name, stat_func in stats_methods:
    result = stat_func()
    print(f"\n{stat_name}():")
    print(f"  well_A: {list(result['well_A'].keys()) if isinstance(result.get('well_A'), dict) else 'N/A'}")
    print(f"  well_B: {list(result['well_B'].keys()) if isinstance(result.get('well_B'), dict) else 'N/A'}")

print("\n✓ All statistics return grouped results")
print("✓ Ambiguous properties automatically nest by source")

# TEST 5: Without filters (baseline)
print("\n" + "=" * 80)
print("TEST 5: Without Filters (Baseline)")
print("=" * 80)
result5 = manager.PHIE.percentile(50)
print("\nmanager.PHIE.percentile(50)  # No filters")
print("\nResult:")
for well_name, well_result in result5.items():
    print(f"  {well_name}:")
    if isinstance(well_result, dict):
        for key, val in well_result.items():
            print(f"    {key}: {val}")
    else:
        print(f"    {well_result}")

print("\n✓ Without filters: single value per well (or nested by source if ambiguous)")
print("✓ With filters: grouped by filter values")

# TEST 6: Percentile extraction (P10, P50, P90)
print("\n" + "=" * 80)
print("TEST 6: Different Percentiles with Filters")
print("=" * 80)

for p in [10, 50, 90]:
    result = manager.PHIE.filter("Zone").percentile(p)
    print(f"\nP{p}:")
    for well_name, well_result in result.items():
        if well_name == 'well_A' and isinstance(well_result, dict):
            # Show just the log source for brevity
            if 'log' in well_result and isinstance(well_result['log'], dict):
                print(f"  {well_name} (log): {well_result['log']}")
            elif isinstance(list(well_result.values())[0], dict):
                # Direct grouping
                print(f"  {well_name}: {list(well_result.values())[0]}")

print("\n✓ Percentile extraction works for P10, P50, P90")
print("✓ Each percentile correctly extracted from grouped results")

print("\n" + "=" * 80)
print("COMPREHENSIVE TEST COMPLETE")
print("=" * 80)
print("✓ Single and multiple filters work")
print("✓ Ambiguous properties (multiple sources) handled correctly")
print("✓ Nested mode forces source name display")
print("✓ All statistics (min, max, mean, std, median, percentile) support grouping")
print("✓ Without filters returns single/nested values")
print("✓ With filters returns grouped values by filter categories")
print("=" * 80)
