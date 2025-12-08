# Well Log Toolkit

Fast, intuitive Python library for petrophysical well log analysis. Load LAS files, filter by zones, compute depth-weighted statistics, and create publication-quality log displays—all in just a few lines.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Key Features

- **🚀 Lazy Loading** - Parse headers instantly, load data on demand
- **🧮 Numpy-Style Operations** - `well.HC_Volume = well.PHIE * (1 - well.SW)`
- **🔍 Hierarchical Filtering** - Chain filters: `well.PHIE.filter('Zone').filter('Facies').sums_avg()`
- **⚖️ Depth-Weighted Statistics** - Proper averaging for irregular sampling
- **📊 Multi-Well Analytics** - Cross-well statistics: `manager.PHIE.filter('Zone').percentile(50)`
- **🎨 Professional Visualization** - Create customizable well log displays with templates
- **📊 Interactive Crossplots** - Beautiful scatter plots with color/size/shape mapping by property
- **📈 Regression Analysis** - 5 regression types (linear, polynomial, exponential, log, power)
- **💾 Project Persistence** - Save/load entire projects with metadata and templates

---

## Table of Contents

### Getting Started
- [Installation](#installation)
- [1-Minute Tutorial](#1-minute-tutorial)
- [5-Minute Quick Start](#5-minute-quick-start)

### Learning Path
- [Core Concepts](#core-concepts) - Essential patterns and workflows
- [Visualization Guide](#visualization-guide) - Creating well log displays
- [Crossplot & Regression Guide](#crossplot--regression-guide) - Data analysis and trend visualization
- [Advanced Topics](#advanced-topics) - Deep dives into specific features

### Quick Reference
- [Style & Marker Reference](#style--marker-reference) - Line styles, markers, colors
- [Colormap Reference](#colormap-reference) - Available colormaps
- [API Reference](#api-reference) - Classes, methods, exceptions
- [Common Patterns](#common-patterns) - Copy-paste examples
- [Troubleshooting](#troubleshooting) - Solutions to common issues

---

## Installation

```bash
pip install well-log-toolkit
```

**Requirements:** Python 3.9+, numpy, pandas, scipy, matplotlib

---

## 1-Minute Tutorial

Load LAS files, filter by zones, and compute statistics:

```python
from well_log_toolkit import WellDataManager

# Load and analyze
manager = WellDataManager()
manager.load_las('well.las')

well = manager.well_12_3_4_A
stats = well.PHIE.filter('Zone').sums_avg()

print(stats['Top_Brent']['mean'])  # → 0.182 (depth-weighted)
```

**That's it!** Three lines to go from LAS file to zonal statistics.

> **New to this?** Continue to [5-Minute Quick Start](#5-minute-quick-start) for a complete walkthrough.

---

## 5-Minute Quick Start

### Step 1: Load Your Data

```python
from well_log_toolkit import WellDataManager
import pandas as pd

# Create manager and load LAS files
manager = WellDataManager()
manager.load_las('well_A.las')
manager.load_las('well_B.las')

# Load formation tops from DataFrame
tops_df = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 B'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Brent'],
    'MD': [2850.0, 3100.0, 2900.0]
})

manager.load_tops(tops_df, well_col='Well', discrete_col='Surface', depth_col='MD')
```

### Step 2: Access Wells and Properties

```python
# Access well (special characters auto-sanitized)
well = manager.well_12_3_4_A

# Access properties directly
phie = well.PHIE
sw = well.SW

# List everything
print(well.properties)  # ['PHIE', 'SW', 'PERM', 'Zone', ...]
print(well.sources)     # ['Petrophysics', 'Imported_Tops']
```

### Step 3: Compute Statistics

```python
# Single filter - group by Zone
stats = well.PHIE.filter('Zone').sums_avg()
# → {'Top_Brent': {'mean': 0.182, 'thickness': 250.0, ...}, ...}

# Chain filters - hierarchical grouping
stats = well.PHIE.filter('Zone').filter('Facies').sums_avg()
# → {'Top_Brent': {'Sandstone': {...}, 'Shale': {...}}, ...}
```

> **💡 Key Insight:** Statistics are **depth-weighted** by default, accounting for irregular sampling.

### Step 4: Create Computed Properties

```python
# Mathematical expressions (numpy-style)
well.HC_Volume = well.PHIE * (1 - well.SW)
well.PHIE_percent = well.PHIE * 100

# Comparison operations (creates discrete flags)
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)

# Apply to all wells at once
manager.PHIE_percent = manager.PHIE * 100
```

### Step 5: Visualize Well Logs

```python
from well_log_toolkit import Template

# Create template
template = Template("basic")

# Add GR track
template.add_track(
    track_type="continuous",
    logs=[{"name": "GR", "x_range": [0, 150], "color": "green"}],
    title="Gamma Ray"
)

# Add depth track
template.add_track(track_type="depth", width=0.3)

# Display
view = well.WellView(depth_range=[2800, 3000], template=template)
view.show()
view.save("well_log.png", dpi=300)
```

### Step 6: Save Your Work

```python
# Save entire project
manager.save('my_project/')

# Load later
manager = WellDataManager('my_project/')
```

**Done!** You've learned the core workflow in 5 minutes.

> **Next Steps:** Explore [Core Concepts](#core-concepts) to understand the library's design patterns, or jump to [Visualization Guide](#visualization-guide) for creating professional log displays.

---

## Core Concepts

### Understanding Well Log Data

Well log data consists of measurements taken at various depths. This library organizes data into three key components:

1. **Wells** - Individual wellbores (e.g., "12/3-4 A")
2. **Properties** - Measurements or computed values (e.g., PHIE, SW, GR)
3. **Sources** - Origin of data (e.g., "Petrophysics", "CoreData", "computed")

```python
# Access structure
well = manager.well_12_3_4_A
print(well.sources)     # ['Petrophysics', 'CoreData']
print(well.properties)  # ['PHIE', 'SW', 'GR', ...]

# Get property
phie = well.PHIE  # Shorthand
phie = well.get_property('PHIE')  # Explicit
phie = well.Petrophysics.PHIE  # From specific source
```

### Property Types

Properties can be **continuous** (numeric measurements), **discrete** (categories), or **sampled** (point measurements like core plugs):

```python
# Continuous (default) - log curves
well.PHIE.type  # → 'continuous'

# Discrete - zones, facies, flags
zone = well.get_property('Zone')
zone.type = 'discrete'
zone.labels = {0: 'Top_Brent', 1: 'Top_Statfjord', 2: 'Top_Cook'}

# Sampled - core plugs (arithmetic mean instead of depth-weighted)
core_phie = well.get_property('CorePHIE')
core_phie.type = 'sampled'
```

### Hierarchical Filtering

Filter properties by discrete logs to compute grouped statistics:

```python
# Single filter
stats = well.PHIE.filter('Zone').sums_avg()
# {
#   'Top_Brent': {'mean': 0.21, 'thickness': 150.0, ...},
#   'Top_Statfjord': {'mean': 0.17, 'thickness': 180.0, ...}
# }

# Chain multiple filters for hierarchical grouping
stats = well.PHIE.filter('Zone').filter('Facies').sums_avg()
# {
#   'Top_Brent': {
#     'Sandstone': {'mean': 0.23, 'thickness': 120.0, ...},
#     'Shale': {'mean': 0.08, 'thickness': 30.0, ...}
#   },
#   'Top_Statfjord': {...}
# }
```

**Statistics include:**
- `mean`, `sum`, `std_dev` - Depth-weighted by default
- `percentile` - p10, p50, p90 values
- `thickness` - Depth interval thickness
- `samples` - Number of valid measurements
- `range`, `depth_range` - Min/max values and depths

### Property Operations

Create computed properties using natural mathematical syntax:

```python
# Arithmetic operations (requires matching depth grids)
well.HC_Volume = well.PHIE * (1 - well.SW)
well.Porosity_Avg = (well.PHIE + well.PHIT) / 2

# Comparison operations (auto-creates discrete properties)
well.High_Poro = well.PHIE > 0.15
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)

# Use computed properties in filtering
stats = well.PHIE.filter('Reservoir').sums_avg()
# → {False: {...}, True: {...}}
```

> **💡 Pro Tip:** Computed properties are stored in the `'computed'` source and can be exported to LAS files.

### Depth Alignment

Operations require matching depth grids (like numpy arrays) to prevent silent interpolation errors:

```python
# This fails if depths don't match
result = well.PHIE + well.CorePHIE  # DepthAlignmentError

# Explicit resampling required
core_resampled = well.CorePHIE.resample(well.PHIE)
result = well.PHIE + core_resampled  # ✓ Works
```

### Multi-Well Analytics

Compute statistics across all wells in a single call:

```python
# Single statistic across all wells
p50 = manager.PHIE.percentile(50)
# → {'well_A': 0.182, 'well_B': 0.195, 'well_C': 0.173}

# With filtering - grouped by filter values per well
stats = manager.PHIE.filter('Zone').percentile(50)
# → {
#   'well_A': {'Top_Brent': 0.21, 'Top_Statfjord': 0.15},
#   'well_B': {'Top_Brent': 0.19, 'Top_Statfjord': 0.17}
# }

# Chain filters for hierarchical grouping
stats = manager.PHIE.filter('Zone').filter('Facies').mean()

# All statistics: min, max, mean, median, std, percentile
```

**Ambiguous properties** (existing in multiple sources) automatically nest by source:

```python
# If well_A has PHIE in both 'log' and 'core' sources:
p50 = manager.PHIE.percentile(50)
# → {'well_A': {'log': 0.182, 'core': 0.205}, 'well_B': 0.195}
```

### Manager Broadcasting

Apply operations to all wells at once:

```python
# Broadcast to all wells with PHIE
manager.PHIE_percent = manager.PHIE * 100

# Broadcast complex operations
manager.HC_Volume = manager.PHIE * (1 - manager.SW)
# ✓ Created property 'HC_Volume' in 12 well(s)
# ⚠ Skipped 3 well(s) without property 'PHIE' or 'SW'
```

### Depth-Weighted vs Arithmetic Statistics

Standard arithmetic mean fails with irregular sampling:

```python
# Example: NTG flag at depths 1500m, 1501m, 1505m with values 0, 1, 0
# Arithmetic mean: (0+1+0)/3 = 0.33 ❌ (treats all samples equally)
# Weighted mean: accounts for 2.5m interval at 1501m = 0.50 ✓

# Compare both methods
stats = well.NTG.filter('Zone').sums_avg(arithmetic=True)
# Returns: {'mean': {'weighted': 0.50, 'arithmetic': 0.33}, ...}
```

> **✨ Key Insight:** Depth-weighted statistics properly handle irregular sample spacing by accounting for depth intervals.

### Project Persistence

Save and restore entire projects:

```python
# Save project structure
manager.save('my_project/')
# Creates: my_project/well_12_3_4_A/Petrophysics.las, templates/*.json, ...

# Load project (restores everything)
manager = WellDataManager('my_project/')
```

---

## Visualization Guide

Create publication-quality well log displays optimized for Jupyter Lab. Build customizable templates with multiple tracks showing continuous logs, discrete properties, fills, formation tops, and markers.

### Quick Start

```python
from well_log_toolkit import WellDataManager

# Load data
manager = WellDataManager()
manager.load_las("well.las")
well = manager.well_36_7_5_A

# Simple display with default template
view = well.WellView(depth_range=[2800, 3000])
view.show()  # Displays inline in Jupyter

# Save to file
view.save("well_log.png", dpi=300)
```

#### Auto-Calculate Depth Range from Tops

Instead of manually specifying depth ranges, let WellView automatically calculate the range from formation tops:

```python
# Load formation tops
manager.load_tops(tops_df, well_col='Well', discrete_col='Surface', depth_col='MD')

# Add tops to template
template = Template("reservoir")
template.add_tops(property_name='Zone')

# Auto-calculate depth range from specific tops
view = well.WellView(
    tops=['Top_Brent', 'Top_Statfjord'],  # Specify which tops to show
    template=template
)
view.show()
# Automatically shows Top_Brent to Top_Statfjord with 5% padding (min 50m range)
```

**How it works:**
- Finds the minimum and maximum depths of specified tops
- Adds 5% padding above and below
- Ensures minimum range of 50 meters
- Perfect for focusing on specific intervals without manual depth calculations

### Building Templates

Templates define the layout and styling of well log displays. Think of a template as a blueprint that can be reused across multiple wells.

#### Basic Template Structure

```python
from well_log_toolkit import Template

# Create template
template = Template("reservoir")

# Add tracks (order matters - left to right)
template.add_track(track_type="continuous", logs=[...], title="GR")
template.add_track(track_type="continuous", logs=[...], title="Resistivity")
template.add_track(track_type="discrete", logs=[...], title="Facies")
template.add_track(track_type="depth", width=0.3, title="Depth")

# Save for reuse
template.save("reservoir_template.json")
```

#### Track Types Explained

**1. Continuous Tracks** - For numeric log curves

Shows one or more curves with configurable scales, styles, fills, and markers.

```python
template.add_track(
    track_type="continuous",
    logs=[
        {
            "name": "GR",              # Property name
            "x_range": [0, 150],       # Scale limits [left, right]
            "color": "green",          # Line color
            "style": "solid",          # Line style (solid/dashed/dotted/none)
            "thickness": 1.5,          # Line width
            "alpha": 0.8               # Transparency (0-1)
        }
    ],
    title="Gamma Ray (API)",
    log_scale=False                    # Use logarithmic scale?
)
```

**2. Discrete Tracks** - For categorical data

Displays colored bands for facies, zones, or other categorical properties.

```python
template.add_track(
    track_type="discrete",
    logs=[{"name": "Facies"}],
    title="Lithofacies"
)
```

Colors come from the property's color mapping:
```python
facies = well.get_property('Facies')
facies.colors = {
    0: 'yellow',      # Sand
    1: 'gray',        # Shale
    2: 'lightblue'    # Limestone
}
```

**3. Depth Tracks** - Show depth axis

```python
template.add_track(
    track_type="depth",
    width=0.3,                         # Narrow width
    title="MD (m)"
)
```

### Styling Log Curves

#### Line Styles

```python
logs=[
    {"name": "GR", "style": "solid"},      # ─────
    {"name": "CALI", "style": "dashed"},   # ─ ─ ─
    {"name": "SP", "style": "dotted"},     # ·····
    {"name": "TEMP", "style": "dashdot"},  # ─·─·─
    {"name": "POINTS", "style": "none"}    # (markers only)
]
```

**Supported styles:** `"solid"` (`"-"`), `"dashed"` (`"--"`), `"dotted"` (`":"`), `"dashdot"` (`"-."`), `"none"` (`""`)

#### Colors

```python
logs=[
    {"name": "RHOB", "color": "red"},           # Color names
    {"name": "NPHI", "color": "#1f77b4"},       # Hex codes
    {"name": "GR", "color": (0.2, 0.5, 0.8)}    # RGB tuples
]
```

#### Thickness and Transparency

```python
logs=[
    {"name": "ILD", "thickness": 2.0, "alpha": 1.0},    # Thick, opaque
    {"name": "ILM", "thickness": 1.0, "alpha": 0.6}     # Thin, transparent
]
```

### Markers for Data Points

Display markers at each data point to show actual measurement locations. Useful for sparse data like core plugs, pressure tests, or sample points.

#### Basic Markers

```python
# Markers with line
logs=[{
    "name": "PERM",
    "x_range": [0.1, 1000],
    "color": "green",
    "style": "solid",           # Show connecting line
    "marker": "circle",         # Add circular markers
    "marker_size": 4,           # Marker size
    "marker_fill": "lightgreen" # Fill color (optional)
}]

# Markers only (no line)
logs=[{
    "name": "CORE_PHIE",
    "x_range": [0, 0.4],
    "color": "blue",
    "style": "none",            # No connecting line
    "marker": "diamond",        # Diamond markers
    "marker_size": 8,
    "marker_outline_color": "darkblue",
    "marker_fill": "yellow"
}]
```

#### Marker Types

**Common markers:**
- `"circle"` (○), `"square"` (□), `"diamond"` (◇)
- `"triangle_up"` (△), `"triangle_down"` (▽)
- `"star"` (★), `"plus"` (+), `"cross"` (×)

**All supported markers:** See [Style & Marker Reference](#style--marker-reference)

#### Marker Configuration

```python
logs=[{
    "name": "SAMPLE_POINTS",
    "marker": "circle",                    # Marker shape
    "marker_size": 6,                      # Size (default: 6)
    "marker_outline_color": "red",         # Edge color (defaults to line color)
    "marker_fill": "yellow",               # Fill color (optional, default: unfilled)
    "marker_interval": 5,                  # Show every 5th marker (default: 1)
}]
```

**Marker interval** is useful for dense data - showing every nth marker reduces clutter:
```python
# Show every 10th marker on a high-resolution log
{"name": "GR", "marker": "point", "marker_interval": 10}
```

### Fill Patterns

Fills highlight areas between curves or track edges. Useful for showing porosity, crossover, or lithology.

#### Solid Color Fill

Fill between a curve and a fixed value:

```python
template.add_track(
    track_type="continuous",
    logs=[{"name": "PHIE", "x_range": [0.45, 0], "color": "blue"}],
    fill={
        "left": "PHIE",         # Curve name
        "right": 0,             # Fixed value
        "color": "lightblue",
        "alpha": 0.5
    }
)
```

Fill between track edge and curve:

```python
fill={
    "left": "track_edge",       # Left edge of track
    "right": "GR",              # GR curve
    "color": "lightgreen",
    "alpha": 0.3
}
```

#### Colormap Fill

Create horizontal color bands where each depth interval is colored based on curve values:

```python
template.add_track(
    track_type="continuous",
    logs=[{"name": "GR", "x_range": [0, 150], "color": "black"}],
    fill={
        "left": "track_edge",
        "right": "GR",
        "colormap": "viridis",          # Colormap name
        "color_range": [20, 150],       # GR values map to colors
        "alpha": 0.7
    },
    title="Gamma Ray"
)
# Low GR (20) → dark purple, High GR (150) → bright yellow
```

**Popular colormaps:**
- `"viridis"` - Perceptually uniform (recommended)
- `"inferno"`, `"plasma"` - Dark to bright
- `"RdYlGn"` - Red-Yellow-Green (diverging)
- `"jet"` - Rainbow (not recommended for scientific use)

See [Colormap Reference](#colormap-reference) for all options.

#### Fill Between Two Curves

```python
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "RHOB", "x_range": [1.95, 2.95], "color": "red"},
        {"name": "NPHI", "x_range": [0.45, -0.15], "color": "blue"}
    ],
    fill={
        "left": "RHOB",
        "right": "NPHI",
        "colormap": "RdYlGn",
        "colormap_curve": "NPHI",       # Use NPHI values for colors
        "color_range": [-0.15, 0.45],
        "alpha": 0.6
    },
    title="Density-Neutron Crossover"
)
```

#### Multiple Fills

Apply multiple fills to a single track (drawn in order):

```python
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "PHIE", "x_range": [0.45, 0], "color": "blue"},
        {"name": "SW", "x_range": [0, 1], "color": "red"}
    ],
    fill=[
        # Fill 1: PHIE to zero
        {
            "left": "PHIE",
            "right": 0,
            "color": "lightblue",
            "alpha": 0.3
        },
        # Fill 2: SW to one
        {
            "left": "SW",
            "right": 1,
            "color": "lightcoral",
            "alpha": 0.3
        }
    ]
)
```

### Formation Tops

Add horizontal lines marking formation boundaries across all tracks:

```python
# Add tops to template (applies to all wells using this template)
template.add_tops(property_name='Zone')

# Or add tops to specific view (only this display)
view = well.WellView(template=template)
view.add_tops(property_name='Zone')
view.show()

# Or provide tops manually
view.add_tops(
    tops_dict={
        2850.0: 'Top Brent',
        3100.0: 'Top Statfjord',
        3400.0: 'Base Statfjord'
    },
    colors={
        2850.0: 'yellow',
        3100.0: 'orange',
        3400.0: 'brown'
    }
)
```

Tops can also be added to individual tracks:

```python
template.add_track(
    track_type="discrete",
    logs=[{"name": "Facies"}],
    tops={
        "name": "Zone",                    # Property containing tops
        "line_style": "--",                # Dashed lines
        "line_width": 2.0,                 # Line thickness
        "title_size": 9,                   # Label font size
        "title_weight": "bold",            # Font weight
        "title_orientation": "right",      # Label position (left/center/right)
        "line_offset": 0.0                 # Horizontal offset
    }
)
```

### Logarithmic Scales

Use logarithmic scales for resistivity, permeability, or other exponential data:

```python
# Track-level log scale (applies to all logs in track)
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "ILD", "x_range": [0.2, 2000], "color": "red"},
        {"name": "ILM", "x_range": [0.2, 2000], "color": "green"}
    ],
    title="Resistivity",
    log_scale=True                         # Logarithmic x-axis
)

# Per-log scale override
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "ILD", "x_range": [0.2, 2000], "color": "red"},      # Uses track log_scale
        {"name": "GR", "x_range": [0, 150], "scale": "linear", "color": "green"}  # Override
    ],
    log_scale=True                         # Default for track
)
```

### Using Templates

**Option 1: Pass template directly**
```python
view = well.WellView(depth_range=[2800, 3000], template=template)
view.show()
```

**Option 2: Store in manager (recommended for multi-well projects)**
```python
# Store template in manager
manager.set_template("reservoir", template)

# Use by name in any well
view = well.WellView(depth_range=[2800, 3000], template="reservoir")
view.show()

# List all templates
print(manager.list_templates())  # ['reservoir', 'qc', 'basic']

# Templates save with projects
manager.save("my_project/")
# Creates: my_project/templates/reservoir.json
```

**Option 3: Load from file**
```python
template = Template.load("reservoir_template.json")
view = well.WellView(depth_range=[2800, 3000], template=template)
```

### Template Management

```python
# Retrieve template
template = manager.get_template("reservoir")

# List all templates
templates = manager.list_templates()

# View tracks in template
df = template.list_tracks()
print(df)
#    Index       Type           Logs         Title  Width
# 0      0 continuous          [GR]    Gamma Ray    1.0
# 1      1 continuous  [PHIE, SW]    Porosity    1.0
# 2      2      depth            []        Depth    0.3

# Edit track
template.edit_track(0, title="New Title")

# Remove track
template.remove_track(2)

# Add new track
template.add_track(track_type="continuous", logs=[{"name": "RT"}])

# Save changes
manager.set_template("reservoir", template)  # Update in manager
template.save("updated_template.json")        # Save to file
```

### Customization

#### Figure Settings

```python
view = well.WellView(
    depth_range=[2800, 3000],
    template="reservoir",
    figsize=(12, 10),              # Width x height in inches
    dpi=100                        # Resolution (default: 100)
)
```

#### Track Widths

Control relative track widths:

```python
template.add_track(track_type="continuous", logs=[...], width=1.0)   # Normal
template.add_track(track_type="discrete", logs=[...], width=1.5)     # 50% wider
template.add_track(track_type="depth", width=0.3)                    # Narrow
```

#### Export Options

```python
# PNG for presentations (raster)
view.save("well_log.png", dpi=300)

# PDF for publications (vector)
view.save("well_log.pdf")

# SVG for editing in Illustrator/Inkscape (vector)
view.save("well_log.svg")
```

### Complete Example

A comprehensive template showcasing all features:

```python
from well_log_toolkit import WellDataManager, Template

# Setup
manager = WellDataManager()
manager.load_las("well.las")
well = manager.well_36_7_5_A

# Create template
template = Template("comprehensive")

# Track 1: GR with colormap and markers
template.add_track(
    track_type="continuous",
    logs=[{
        "name": "GR",
        "x_range": [0, 150],
        "color": "black",
        "marker": "point",
        "marker_interval": 20  # Show every 20th sample
    }],
    fill={
        "left": "track_edge",
        "right": "GR",
        "colormap": "viridis",
        "color_range": [20, 150],
        "alpha": 0.7
    },
    title="Gamma Ray (API)"
)

# Track 2: Resistivity (log scale)
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "ILD", "x_range": [0.2, 2000], "color": "red", "thickness": 1.5},
        {"name": "ILM", "x_range": [0.2, 2000], "color": "green"}
    ],
    title="Resistivity (ohmm)",
    log_scale=True
)

# Track 3: Density-Neutron with crossover
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "RHOB", "x_range": [1.95, 2.95], "color": "red"},
        {"name": "NPHI", "x_range": [0.45, -0.15], "color": "blue"}
    ],
    fill={
        "left": "RHOB",
        "right": "NPHI",
        "colormap": "RdYlGn",
        "alpha": 0.5
    },
    title="Density-Neutron"
)

# Track 4: Porosity & Saturation
template.add_track(
    track_type="continuous",
    logs=[
        {"name": "PHIE", "x_range": [0.45, 0], "color": "blue"},
        {"name": "SW", "x_range": [0, 1], "color": "red"}
    ],
    fill={
        "left": "PHIE",
        "right": 0,
        "color": "lightblue",
        "alpha": 0.5
    },
    title="PHIE & SW"
)

# Track 5: Core data (markers only, no lines)
template.add_track(
    track_type="continuous",
    logs=[{
        "name": "CorePHIE",
        "x_range": [0, 0.4],
        "color": "darkblue",
        "style": "none",           # No connecting line
        "marker": "diamond",
        "marker_size": 8,
        "marker_outline_color": "darkblue",
        "marker_fill": "yellow"
    }],
    title="Core Porosity"
)

# Track 6: Facies with tops
template.add_track(
    track_type="discrete",
    logs=[{"name": "Facies"}],
    title="Lithofacies"
)

# Track 7: Depth
template.add_track(track_type="depth", width=0.3, title="MD (m)")

# Add formation tops spanning all tracks
template.add_tops(property_name='Zone')

# Save and display
manager.set_template("comprehensive", template)
view = well.WellView(depth_range=[2800, 3200], template="comprehensive")
view.save("comprehensive_log.png", dpi=300)
```

---

## Crossplot & Regression Guide

Create beautiful, publication-quality crossplots for petrophysical analysis with sophisticated color/size/shape mapping and built-in regression analysis.

### Quick Start

```python
from well_log_toolkit import WellDataManager

manager = WellDataManager()
manager.load_las("well.las")
well = manager.well_36_7_5_A

# Simple crossplot
plot = well.Crossplot(x="RHOB", y="NPHI")
plot.show()
```

That's it! One line to create a scatter plot from any two properties.

### Basic Crossplots

#### Single Well Analysis

```python
# Density vs Neutron Porosity
plot = well.Crossplot(
    x="RHOB",
    y="NPHI",
    title="Density-Neutron Crossplot"
)
plot.show()

# Save high-resolution image
plot.save("density_neutron.png", dpi=300)
```

#### Multi-Well Comparison

Compare multiple wells on the same plot:

```python
# All wells with different markers
plot = manager.Crossplot(
    x="PHIE",
    y="SW",
    shape="well",  # Different marker shape per well
    title="Multi-Well Porosity vs Saturation"
)
plot.show()

# Specific wells only
plot = manager.Crossplot(
    x="RHOB",
    y="NPHI",
    wells=["Well_A", "Well_B", "Well_C"],
    shape="well"
)
plot.show()
```

### Advanced Mapping

#### Color by Property or Depth

Visualize a third dimension using color:

```python
# Color by depth
plot = well.Crossplot(
    x="PHIE",
    y="SW",
    color="depth",
    colortemplate="viridis",
    color_range=[2000, 2500],  # Depth range in meters
    title="Porosity vs SW (colored by depth)"
)
plot.show()

# Color by shale volume
plot = well.Crossplot(
    x="PHIE",
    y="PERM",
    color="VSH",
    colortemplate="RdYlGn_r",  # Red=high shale, Green=low shale
    title="Porosity-Permeability (colored by VSH)"
)
plot.show()
```

**Available colormaps:** `"viridis"`, `"plasma"`, `"coolwarm"`, `"RdYlGn"`, `"jet"`, and 100+ more matplotlib colormaps.

#### Size by Property

Make marker size represent a fourth dimension:

```python
plot = well.Crossplot(
    x="PHIE",
    y="SW",
    size="PERM",              # Bigger markers = higher permeability
    size_range=(20, 200),     # Min/max marker sizes
    color="depth",
    colortemplate="viridis",
    title="Porosity vs SW (sized by PERM)"
)
plot.show()
```

#### Shape by Category

Use different marker shapes for different groups:

```python
# Different shapes for different facies
plot = well.Crossplot(
    x="PHIE",
    y="PERM",
    shape="Facies",           # Different marker per facies type
    color="depth",
    title="Porosity-Permeability by Facies"
)
plot.show()

# Multi-well: different shapes per well
plot = manager.Crossplot(
    x="PHIE",
    y="SW",
    shape="well",             # Circle, square, triangle, etc.
    color="VSH",
    size="PERM"
)
plot.show()
```

#### All Features Combined

Combine color, size, and shape for comprehensive visualization:

```python
plot = manager.Crossplot(
    x="PHIE",
    y="SW",
    wells=["Well_A", "Well_B"],    # Specific wells
    shape="well",                   # Different marker per well
    color="depth",                  # Color by depth
    size="PERM",                    # Size by permeability
    colortemplate="viridis",
    color_range=[2000, 2500],
    size_range=(30, 200),
    title="Multi-Dimensional Analysis",
    figsize=(12, 10),
    dpi=150
)
plot.show()
```

### Logarithmic Scales

Perfect for permeability and resistivity data:

```python
# Log scale on x-axis (permeability)
plot = well.Crossplot(
    x="PERM",
    y="PHIE",
    x_log=True,
    title="Porosity-Permeability (log scale)"
)
plot.show()

# Log-log plot
plot = well.Crossplot(
    x="PERM",
    y="Pressure",
    x_log=True,
    y_log=True,
    title="Log-Log Analysis"
)
plot.show()
```

### Depth Filtering

Focus on specific intervals:

```python
# Reservoir zone only
plot = well.Crossplot(
    x="PHIE",
    y="SW",
    depth_range=(2000, 2500),
    color="VSH",
    title="Reservoir Zone Analysis (2000-2500m)"
)
plot.show()
```

### Regression Analysis

Add trend lines to identify relationships between properties.

#### Linear Regression

```python
plot = well.Crossplot(x="RHOB", y="NPHI", title="Density-Neutron")

# Add linear regression
plot.add_regression("linear", line_color="red", line_width=2)
plot.show()

# Access regression results
reg = plot.regressions["linear"]
print(reg.equation())      # y = -0.2956x + 0.9305
print(f"R² = {reg.r_squared:.4f}")  # R² = 0.8147
print(f"RMSE = {reg.rmse:.4f}")     # RMSE = 0.0208
```

#### Multiple Regression Types

Compare different regression models:

```python
plot = well.Crossplot(x="PHIE", y="SW", title="Porosity vs Saturation")

# Add multiple regressions
plot.add_regression("linear", line_color="red")
plot.add_regression("polynomial", degree=2, line_color="blue")
plot.add_regression("exponential", line_color="green")

plot.show()

# Compare R² values
for name, reg in plot.regressions.items():
    print(f"{name}: R² = {reg.r_squared:.4f}")
# linear: R² = 0.0144
# polynomial: R² = 0.0155
# exponential: R² = 0.0201  ← Best fit
```

#### Available Regression Types

| Type | Equation | Use Case | Example |
|------|----------|----------|---------|
| `"linear"` | y = ax + b | Straight trends | Density-Porosity |
| `"polynomial"` | y = aₙxⁿ + ... + a₁x + a₀ | Curved relationships | Sonic-Porosity |
| `"exponential"` | y = ae^(bx) | Exponential growth | Production decline |
| `"logarithmic"` | y = a·ln(x) + b | Diminishing returns | Time-dependent |
| `"power"` | y = ax^b | Power law | Porosity-Permeability |

#### Polynomial Regression

Fit higher-order polynomials for curved relationships:

```python
plot = well.Crossplot(x="DT", y="PHIE")

# Quadratic (degree 2)
plot.add_regression("polynomial", degree=2, line_color="blue")

# Cubic (degree 3)
plot.add_regression("polynomial", degree=3, line_color="green", name="cubic")

plot.show()
```

#### Regression Customization

Control regression line appearance:

```python
plot.add_regression(
    "linear",
    name="best_fit",           # Custom name
    line_color="red",           # Line color
    line_width=2,               # Line thickness
    line_style="--",            # Dashed: "--", dotted: ":", solid: "-"
    line_alpha=0.8,             # Transparency (0-1)
    show_equation=True,         # Show equation in legend
    show_r2=True                # Show R² value
)
```

#### Using Regression for Predictions

Extract regression objects and use them for calculations:

```python
plot = well.Crossplot(x="RHOB", y="NPHI")
plot.add_regression("linear")

# Get regression object
reg = plot.regressions["linear"]

# Predict values
density_values = [2.3, 2.4, 2.5, 2.6]
predicted_nphi = reg(density_values)
print(predicted_nphi)  # [0.249, 0.220, 0.191, 0.161]

# Or use predict method
predicted_nphi = reg.predict(density_values)

# Get statistics
print(f"Equation: {reg.equation()}")
print(f"R²: {reg.r_squared:.4f}")
print(f"RMSE: {reg.rmse:.4f}")
```

### Standalone Regression Classes

Use regression classes independently for data analysis:

```python
from well_log_toolkit import LinearRegression, PolynomialRegression
import numpy as np

# Prepare data
x_data = np.array([2.2, 2.3, 2.4, 2.5, 2.6])
y_data = np.array([0.28, 0.25, 0.22, 0.19, 0.16])

# Fit linear regression
reg = LinearRegression()
reg.fit(x_data, y_data)

# Get results
print(reg.equation())           # y = -0.3000x + 0.9400
print(f"R² = {reg.r_squared}")  # R² = 1.0000
print(f"RMSE = {reg.rmse}")     # RMSE = 0.0000

# Make predictions
new_densities = [2.35, 2.45, 2.55]
predicted = reg(new_densities)
print(predicted)  # [0.235, 0.205, 0.175]

# Try polynomial
poly = PolynomialRegression(degree=2)
poly.fit(x_data, y_data)
print(poly.equation())
```

#### All Regression Classes

```python
from well_log_toolkit import (
    LinearRegression,          # y = ax + b
    PolynomialRegression,      # y = aₙxⁿ + ... + a₀
    ExponentialRegression,     # y = ae^(bx)
    LogarithmicRegression,     # y = a·ln(x) + b
    PowerRegression            # y = ax^b
)

# Each has the same interface
reg = LinearRegression()
reg.fit(x, y)
y_pred = reg.predict(x_new)
print(reg.equation())
print(reg.r_squared)
print(reg.rmse)
```

### Customization Options

Fine-tune your crossplot appearance:

```python
plot = well.Crossplot(
    x="RHOB",
    y="NPHI",
    # Plot settings
    title="Custom Crossplot",
    xlabel="Bulk Density (g/cc)",
    ylabel="Neutron Porosity (v/v)",
    figsize=(12, 10),           # Figure size (width, height)
    dpi=150,                    # Resolution

    # Marker settings
    marker="D",                 # Diamond markers
    marker_size=80,             # Larger markers
    marker_alpha=0.7,           # 70% opaque
    edge_color="darkblue",      # Marker outline color
    edge_width=1.5,             # Outline thickness

    # Grid settings
    grid=True,
    grid_alpha=0.3,             # Subtle grid

    # Display options
    show_colorbar=True,         # Show colorbar
    show_legend=True            # Show legend
)
plot.show()
```

**Marker styles:** `"o"` (circle), `"s"` (square), `"^"` (triangle), `"D"` (diamond), `"v"` (inverted triangle), `"*"` (star), `"+"` (plus), `"x"` (cross)

### Practical Examples

#### Porosity-Permeability Analysis

```python
# Classic log-scale relationship
plot = well.Crossplot(
    x="PHIE",
    y="PERM",
    y_log=True,                 # Log scale for permeability
    color="depth",
    colortemplate="viridis",
    title="Porosity-Permeability Transform"
)

# Add power law regression (typical for poro-perm)
plot.add_regression("power", line_color="red", line_width=2)
plot.show()

# Use regression for permeability prediction
power_reg = plot.regressions["power"]
print(power_reg.equation())    # y = 2.5*x^3.2

# Predict permeability from porosity
porosities = [0.10, 0.15, 0.20, 0.25, 0.30]
perms = power_reg(porosities)
print(perms)  # [0.003, 0.025, 0.100, 0.275, 0.562] mD
```

#### Reservoir Quality Classification

```python
# Multi-well reservoir quality
plot = manager.Crossplot(
    x="PHIE",
    y="SW",
    shape="well",              # Different marker per well
    color="VSH",               # Color by shale volume
    size="PERM",               # Size by permeability
    colortemplate="RdYlGn_r",  # Red=shaly, Green=clean
    title="Reservoir Quality Classification"
)

# Add cutoff lines
plot.add_regression("linear", line_color="red", show_equation=False)
plot.show()

# Identify sweet spots: PHIE > 0.15 and SW < 0.4
```

#### Lithology Identification

```python
# Density-Neutron crossplot for lithology
plot = well.Crossplot(
    x="RHOB",
    y="NPHI",
    color="GR",                # Color by gamma ray
    colortemplate="viridis",
    color_range=[0, 150],
    title="Density-Neutron Lithology Plot"
)

# Add lithology lines
plot.add_regression("linear", line_color="yellow", name="Sandstone")
plot.add_regression("polynomial", degree=2, line_color="gray", name="Shale")
plot.show()
```

### Best Practices

1. **Choose appropriate scales:** Use log scales for permeability, resistivity
2. **Color consistency:** Specify `color_range` to keep colors consistent across plots
3. **Multiple regressions:** Try different types and compare R² values
4. **Depth filtering:** Focus on specific intervals with `depth_range`
5. **Save high-res:** Use `dpi=300` for publication-quality images

### Quick Reference

```python
# Basic crossplot
plot = well.Crossplot(x="RHOB", y="NPHI")
plot.show()

# With color and size
plot = well.Crossplot(x="PHIE", y="SW", color="depth", size="PERM")
plot.show()

# Multi-well
plot = manager.Crossplot(x="PHIE", y="SW", shape="well")
plot.show()

# With regression
plot = well.Crossplot(x="RHOB", y="NPHI")
plot.add_regression("linear", line_color="red")
plot.show()

# Standalone regression
from well_log_toolkit import LinearRegression
reg = LinearRegression()
reg.fit(x, y)
predictions = reg([10, 20, 30])
```

For comprehensive examples and API details, see:
- **[CROSSPLOT_README.md](CROSSPLOT_README.md)** - Complete documentation
- **[CROSSPLOT_QUICK_REFERENCE.md](CROSSPLOT_QUICK_REFERENCE.md)** - Quick reference card
- **[examples/crossplot_examples.py](examples/crossplot_examples.py)** - 15+ examples

---

## Style & Marker Reference

### Line Styles

| Style Name | Code | Example | Usage |
|------------|------|---------|-------|
| `"solid"` | `"-"` | ───── | Default, primary curves |
| `"dashed"` | `"--"` | ─ ─ ─ | Secondary curves |
| `"dotted"` | `":"` | ····· | Tertiary curves |
| `"dashdot"` | `"-."` | ─·─·─ | Alternate curves |
| `"none"` | `""` | (none) | Markers only |

### Markers

#### Basic Shapes

| Name | Code | Symbol | Usage |
|------|------|--------|-------|
| `"circle"` | `"o"` | ○ | General purpose, most common |
| `"square"` | `"s"` | □ | Grid data, regular samples |
| `"diamond"` | `"D"` | ◇ | Special points, core data |
| `"star"` | `"*"` | ★ | Important points |
| `"plus"` | `"+"` | + | Crosshairs, reference points |
| `"cross"` | `"x"` | × | Outliers, rejected points |

#### Triangles

| Name | Code | Symbol | Usage |
|------|------|--------|-------|
| `"triangle_up"` | `"^"` | △ | Increasing trend |
| `"triangle_down"` | `"v"` | ▽ | Decreasing trend |
| `"triangle_left"` | `"<"` | ◁ | Directional indicators |
| `"triangle_right"` | `">"` | ▷ | Directional indicators |

#### Special

| Name | Code | Symbol | Usage |
|------|------|--------|-------|
| `"pentagon"` | `"p"` | ⬟ | Alternative shape |
| `"hexagon"` | `"h"` | ⬢ | Honeycomb patterns |
| `"point"` | `"."` | · | Dense data, minimal marker |
| `"pixel"` | `","` | , | Very dense data |
| `"vline"` | `"|"` | │ | Vertical emphasis |
| `"hline"` | `"_"` | ─ | Horizontal emphasis |

### Color Names

**Basic colors:** `"red"`, `"blue"`, `"green"`, `"yellow"`, `"orange"`, `"purple"`, `"pink"`, `"brown"`, `"gray"`, `"black"`, `"white"`

**Light colors:** `"lightblue"`, `"lightgreen"`, `"lightcoral"`, `"lightgray"`, `"lightyellow"`

**Dark colors:** `"darkblue"`, `"darkgreen"`, `"darkred"`, `"darkgray"`

**Advanced:** Use hex codes (`"#1f77b4"`) or RGB tuples (`(0.2, 0.5, 0.8)`) for precise colors.

---

## Colormap Reference

### Sequential (Light to Dark)

Perfect for showing magnitude or intensity:

| Colormap | Description | Use Case |
|----------|-------------|----------|
| `"viridis"` | Yellow-green-blue (perceptually uniform) | **Recommended default** |
| `"plasma"` | Purple-pink-yellow | High contrast |
| `"inferno"` | Black-purple-yellow | Dark backgrounds |
| `"magma"` | Black-purple-white | Maximum contrast |
| `"cividis"` | Blue-yellow (colorblind-safe) | Accessibility |

### Diverging (Low-Mid-High)

Perfect for data with a meaningful center (e.g., 0, neutral point):

| Colormap | Description | Use Case |
|----------|-------------|----------|
| `"RdYlGn"` | Red-Yellow-Green | Good/bad (e.g., quality) |
| `"RdBu"` | Red-Blue | Hot/cold, positive/negative |
| `"PiYG"` | Pink-Yellow-Green | Alternative diverging |
| `"BrBG"` | Brown-Blue-Green | Earth tones |

### Qualitative

For categorical data (use discrete tracks instead):

| Colormap | Description |
|----------|-------------|
| `"tab10"` | 10 distinct colors |
| `"tab20"` | 20 distinct colors |
| `"Paired"` | Paired colors |

### Classic (Not Recommended)

| Colormap | Issue |
|----------|-------|
| `"jet"` | Not perceptually uniform, creates false boundaries |
| `"rainbow"` | Similar issues to jet |

> **💡 Recommendation:** Use `"viridis"` for general purposes. Use `"RdYlGn"` for diverging data. Avoid `"jet"`.

---

## Advanced Topics

### Formation Tops Setup

Formation tops create discrete zones that start at each top and extend to the next:

```python
import pandas as pd

# Create tops DataFrame
tops_df = pd.DataFrame({
    'Well': ['12/3-4 A', '12/3-4 A', '12/3-4 A'],
    'Surface': ['Top_Brent', 'Top_Statfjord', 'Top_Cook'],
    'MD': [2850.0, 3100.0, 3400.0]
})

# Load tops
manager.load_tops(
    tops_df,
    property_name='Zone',      # Name for discrete property
    source_name='Tops',        # Source name
    well_col='Well',           # Column with well names
    discrete_col='Surface',    # Column with formation names
    depth_col='MD'             # Column with depths
)

# How it works:
# - Top_Brent applies from 2850m to 3100m
# - Top_Statfjord applies from 3100m to 3400m
# - Top_Cook applies from 3400m to bottom of log
```

### Discrete Properties & Labels

```python
# Create or modify discrete property
ntg = well.get_property('NTG_Flag')
ntg.type = 'discrete'
ntg.labels = {0: 'NonNet', 1: 'Net'}

# Use in filtering
stats = well.PHIE.filter('NTG_Flag').sums_avg()
# Returns: {'NonNet': {...}, 'Net': {...}}

# Add colors for visualization
ntg.colors = {0: 'gray', 1: 'yellow'}
```

### Understanding Statistics

Each statistics group provides comprehensive information:

```python
stats = well.PHIE.filter('Zone').sums_avg()

# Example output structure:
{
  'Top_Brent': {
    'mean': 0.182,              # Depth-weighted average
    'sum': 45.5,                # Sum (for flags: net thickness)
    'std_dev': 0.044,           # Standard deviation
    'percentile': {
      'p10': 0.09,              # 10th percentile (pessimistic)
      'p50': 0.18,              # Median
      'p90': 0.24               # 90th percentile (optimistic)
    },
    'range': {'min': 0.05, 'max': 0.28},
    'depth_range': {'min': 2850.0, 'max': 3100.0},
    'samples': 250,             # Number of valid measurements
    'thickness': 250.0,         # Interval thickness
    'gross_thickness': 555.0,   # Total across all zones
    'thickness_fraction': 0.45, # Fraction of total
    'calculation': 'weighted'   # Method used
  }
}
```

### Export Options

**To DataFrame:**
```python
# All properties (default: errors if depths don't match exactly)
df = well.data()

# Specific properties
df = well.data(include=['PHIE', 'SW', 'PERM'])

# Interpolate to common depth grid if depths don't align
df = well.data(merge_method='resample')

# Use labels for discrete properties
df = well.data(discrete_labels=True)
```

**To LAS:**
```python
# Export all properties
well.export_to_las('output.las')

# Specific properties
well.export_to_las('output.las', include=['PHIE', 'SW'])

# Use original LAS as template (preserves headers)
well.export_to_las('output.las', use_template=True)

# Export each source separately
well.export_sources('output_folder/')
# Creates: Petrophysics.las, CoreData.las, computed.las
```

### Managing Sources

```python
# List sources
print(well.sources)  # ['Petrophysics', 'CoreData']

# Access through source
phie_log = well.Petrophysics.PHIE
phie_core = well.CoreData.CorePHIE

# Rename source
well.rename_source('CoreData', 'Core_Analysis')

# Remove source (deletes all properties)
well.remove_source('Core_Analysis')
```

### Adding External Data

```python
import pandas as pd

# Create DataFrame
external_df = pd.DataFrame({
    'DEPT': [2800, 2801, 2802],
    'CorePHIE': [0.20, 0.22, 0.19],
    'CorePERM': [150, 200, 120]
})

# Add to well
well.add_dataframe(
    external_df,
    source_name='CoreData',
    unit_mappings={'CorePHIE': 'v/v', 'CorePERM': 'mD'},
    type_mappings={'CorePHIE': 'continuous', 'CorePERM': 'continuous'}
)
```

### Sampled Data (Core Plugs)

Core plugs are point samples requiring arithmetic (not depth-weighted) statistics:

```python
# Load as sampled
manager.load_las('core_plugs.las', sampled=True)

# Or mark properties as sampled
well.CorePHIE.type = 'sampled'

# Statistics use arithmetic mean
stats = well.CorePHIE.filter('Zone').sums_avg()
# → {'calculation': 'arithmetic'} (each plug counts equally)
```

### Managing Wells

```python
# List wells
print(manager.wells)  # ['well_12_3_4_A', 'well_12_3_4_B']

# Access by name
well = manager.well_12_3_4_A              # Sanitized name (attribute)
well = manager.get_well('12/3-4 A')       # Original name
well = manager.get_well('12_3_4_A')       # Sanitized name
well = manager.get_well('well_12_3_4_A')  # With prefix

# Add well
well = manager.add_well('12/3-4 C')

# Remove well
manager.remove_well('12_3_4_A')
```

### Property Inspection

```python
# Print property (auto-clips large arrays)
print(well.PHIE)
# [PHIE] (1001 samples)
# depth: [2800.00, 2801.00, ..., 3800.00]
# values (v/v): [0.180, 0.185, ..., 0.210]

# Print filtered property
filtered = well.PHIE.filter('Zone')
print(filtered)
# [PHIE] (1001 samples)
# Filters: Zone: [Top_Brent, Top_Brent, ...]

# Print manager-level property
print(manager.PHIE)
# [PHIE] across 3 well(s):
# Well: well_12_3_4_A
# [PHIE] (1001 samples)
# ...
```

---

## API Reference

### Main Classes

```python
from well_log_toolkit import WellDataManager, Well, Property, LasFile
```

**WellDataManager** - Manages multiple wells
- `load_las(filepath, sampled=False)` - Load LAS file
- `load_tops(df, well_col, discrete_col, depth_col)` - Load formation tops
- `add_well(name)` - Add empty well
- `get_well(name)` - Get well by name
- `remove_well(name)` - Remove well
- `save(directory)` - Save project
- `load(directory)` - Load project
- `set_template(name, template)` - Store template
- `get_template(name)` - Retrieve template
- `list_templates()` - List template names
- `Crossplot(x, y, wells=None, shape="well", ...)` - Create multi-well crossplot

**Well** - Individual wellbore
- `get_property(name, source=None)` - Get property
- `add_dataframe(df, source_name, ...)` - Add external data
- `data(include=None, exclude=None)` - Export to DataFrame
- `export_to_las(filepath, ...)` - Export to LAS
- `export_sources(directory)` - Export each source
- `rename_source(old, new)` - Rename source
- `remove_source(name)` - Remove source
- `WellView(depth_range=None, tops=None, template, ...)` - Create log visualization
- `Crossplot(x, y, color=None, size=None, shape=None, ...)` - Create crossplot

**Property** - Single measurement or computed value
- `filter(discrete_property)` - Filter by discrete property
- `sums_avg(arithmetic=False)` - Compute statistics
- `resample(reference_property)` - Resample to new depth grid
- Attributes: `name`, `depth`, `values`, `unit`, `type`, `labels`, `colors`

### Visualization Classes

```python
from well_log_toolkit import Template, WellView, Crossplot
```

**Template** - Display layout configuration
- `add_track(track_type, logs, fill, tops, ...)` - Add track
- `add_tops(property_name, tops_dict, ...)` - Add formation tops
- `edit_track(index, **kwargs)` - Edit track
- `remove_track(index)` - Remove track
- `get_track(index)` - Get track config
- `list_tracks()` - List all tracks
- `save(filepath)` - Save to JSON
- `load(filepath)` - Load from JSON (classmethod)
- `to_dict()`, `from_dict(data)` - Dict conversion

**WellView** - Well log display
- `plot()` - Create matplotlib figure
- `show()` - Display in Jupyter
- `save(filepath, dpi)` - Save to file
- `close()` - Close figure
- `add_track(...)` - Add temporary track
- `add_tops(...)` - Add temporary tops

**Crossplot** - Scatter plot with regression analysis
- `plot()` - Create matplotlib figure
- `show()` - Display plot
- `save(filepath, dpi)` - Save to file
- `close()` - Close figure
- `add_regression(type, **kwargs)` - Add regression line
- `remove_regression(name)` - Remove regression
- Attributes: `regressions`, `fig`, `ax`

### Regression Classes

```python
from well_log_toolkit import (
    LinearRegression,
    PolynomialRegression,
    ExponentialRegression,
    LogarithmicRegression,
    PowerRegression
)
```

All regression classes share the same interface:

- `fit(x, y)` - Fit regression model to data
- `predict(x)` - Predict y values for given x
- `__call__(x)` - Alternative prediction syntax: `reg([1, 2, 3])`
- `equation()` - Get equation string (e.g., "y = 2.5x + 1.3")
- Attributes: `r_squared`, `rmse`, `x_data`, `y_data`

**PolynomialRegression** - Additional parameter:
- `__init__(degree=2)` - Specify polynomial degree

### Statistics Functions

```python
from well_log_toolkit import compute_intervals, mean, sum, std, percentile
```

These are low-level functions used internally. Most users should use the high-level filtering API (`property.filter().sums_avg()`).

### Exceptions

```python
from well_log_toolkit import (
    DepthAlignmentError,
    PropertyNotFoundError,
    PropertyTypeError
)
```

- `DepthAlignmentError` - Raised when combining properties with different depth grids
- `PropertyNotFoundError` - Raised when accessing non-existent property
- `PropertyTypeError` - Raised when property has wrong type (e.g., filtering by continuous property)

---

## Common Patterns

Copy-paste examples for common tasks:

### Load and Analyze

```python
manager = WellDataManager()
manager.load_las('well.las')
stats = manager.well_12_3_4_A.PHIE.filter('Zone').sums_avg()
```

### Chain Multiple Filters

```python
stats = well.PHIE.filter('Zone').filter('Facies').filter('NTG_Flag').sums_avg()
```

### Multi-Well Statistics

```python
# P50 by zone across all wells
p50 = manager.PHIE.filter('Zone').percentile(50)

# All statistics
means = manager.PHIE.filter('Zone').mean()
stds = manager.PHIE.filter('Zone').std()
```

### Create Computed Properties

```python
well.HC_Volume = well.PHIE * (1 - well.SW)
well.Reservoir = (well.PHIE > 0.15) & (well.SW < 0.35)
```

### Broadcast Across Wells

```python
manager.PHIE_percent = manager.PHIE * 100
manager.Reservoir = (manager.PHIE > 0.15) & (manager.SW < 0.35)
```

### Quick Visualization

```python
# With depth range
view = well.WellView(depth_range=[2800, 3000])
view.show()

# Auto-calculate from tops
view = well.WellView(tops=['Top_Brent', 'Top_Statfjord'])
view.show()
```

### Build Custom Template

```python
template = Template("custom")
template.add_track(
    track_type="continuous",
    logs=[{"name": "GR", "x_range": [0, 150], "color": "green"}],
    title="Gamma Ray"
)
manager.set_template("custom", template)
view = well.WellView(template="custom")
view.save("log.png", dpi=300)
```

### Crossplots

```python
# Simple crossplot
plot = well.Crossplot(x="RHOB", y="NPHI")
plot.show()

# With color and regression
plot = well.Crossplot(x="PHIE", y="SW", color="depth")
plot.add_regression("linear", line_color="red")
plot.show()

# Multi-well
plot = manager.Crossplot(x="PHIE", y="SW", shape="well")
plot.show()
```

### Regression Analysis

```python
# With crossplot
plot = well.Crossplot(x="RHOB", y="NPHI")
plot.add_regression("linear")
reg = plot.regressions["linear"]
predictions = reg([2.3, 2.4, 2.5])

# Standalone
from well_log_toolkit import LinearRegression
reg = LinearRegression()
reg.fit(x_data, y_data)
print(reg.equation())
y_pred = reg(new_x_values)
```

### Save and Load Projects

```python
manager.save('project/')
manager = WellDataManager('project/')
```

---

## Troubleshooting

### DepthAlignmentError

**Problem:** Properties have different depth grids

```python
result = well.PHIE + well.CorePHIE  # Error!
```

**Solution:** Explicitly resample

```python
core_resampled = well.CorePHIE.resample(well.PHIE)
result = well.PHIE + core_resampled  # Works!
```

### PropertyNotFoundError

**Problem:** Property doesn't exist

```python
phie = well.PHIE_TOTAL  # Error if property doesn't exist
```

**Solution:** Check available properties

```python
print(well.properties)  # List all
print(well.sources)     # Check sources

# Or handle gracefully
try:
    phie = well.get_property('PHIE_TOTAL')
except PropertyNotFoundError:
    phie = well.PHIE  # Use fallback
```

### PropertyTypeError

**Problem:** Filtering by non-discrete property

```python
stats = well.PHIE.filter('PERM').sums_avg()  # Error!
```

**Solution:** Mark as discrete

```python
perm = well.get_property('PERM')
perm.type = 'discrete'
perm.labels = {0: 'Low', 1: 'Medium', 2: 'High'}
stats = well.PHIE.filter('PERM').sums_avg()  # Works!
```

### Missing Statistics for Some Zones

**Problem:** No valid data in some zones

```python
stats = well.PHIE.filter('Zone').sums_avg()
# Some zones missing if all PHIE values are NaN
```

**Solution:** Check raw data

```python
print(well.PHIE.values)  # Look for NaN
print(well.Zone.values)  # Check distribution

# Filter NaN values
import numpy as np
valid_mask = ~np.isnan(well.PHIE.values)
```

### Template Not Found

**Problem:** Template doesn't exist

```python
view = well.WellView(template="missing")  # Error!
```

**Solution:** Check available templates

```python
print(manager.list_templates())  # ['reservoir', 'qc']

# Or pass template directly
template = Template("custom")
view = well.WellView(template=template)
```

### Visualization Not Showing

**Problem:** Display doesn't appear in Jupyter

```python
view = well.WellView(template="reservoir")
# Nothing shows
```

**Solution:** Call show() explicitly

```python
view = well.WellView(template="reservoir")
view.show()  # Required in Jupyter
```

### Markers Not Appearing

**Problem:** Markers not visible in log display

```python
logs=[{"name": "GR", "marker": "circle"}]
# No markers show
```

**Solution:** Check marker configuration

```python
# Ensure marker size is visible
logs=[{"name": "GR", "marker": "circle", "marker_size": 6}]

# If line is very thick, markers might be hidden
logs=[{
    "name": "GR",
    "marker": "circle",
    "marker_size": 8,           # Larger markers
    "marker_outline_color": "red",  # Distinct color
    "marker_fill": "yellow"     # Filled markers stand out
}]

# For markers only, use style="none"
logs=[{
    "name": "CORE_PHIE",
    "style": "none",            # Remove line
    "marker": "diamond",
    "marker_size": 10
}]
```

### Tops Parameter Error

**Problem:** No formation tops loaded

```python
view = well.WellView(tops=['Top_Brent', 'Top_Statfjord'])
# ValueError: No formation tops have been loaded
```

**Solution:** Add tops to template or view first

```python
# Option 1: Add tops to template
template = Template("reservoir")
template.add_tops(property_name='Zone')
view = well.WellView(tops=['Top_Brent', 'Top_Statfjord'], template=template)

# Option 2: Add tops to view
view = well.WellView(template=template)
view.add_tops(property_name='Zone')
# Note: Can't use tops parameter if tops aren't in template

# Option 3: Use depth_range instead
view = well.WellView(depth_range=[2800, 3000], template=template)
```

**Problem:** Specified tops not found

```python
view = well.WellView(tops=['Top_Missing'])
# ValueError: Formation tops not found: ['Top_Missing']
```

**Solution:** Check available tops

```python
# Load and check tops
manager.load_tops(tops_df, well_col='Well', discrete_col='Surface', depth_col='MD')

# Check what tops are available
zone = well.get_property('Zone')
print(zone.labels)  # {0: 'Top_Brent', 1: 'Top_Statfjord', ...}

# Use correct names
view = well.WellView(tops=['Top_Brent', 'Top_Statfjord'], template=template)
```

---

## Performance

All operations use **vectorized numpy** for maximum speed:

- **100M+ samples/second** throughput
- Typical well logs (1k-10k samples) process in **< 1ms**
- Filtered statistics (2 filters, 10 wells): **~9ms**
- Manager-level operations optimized with property caching
- I/O bottleneck eliminated with lazy loading

---

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0

---

## Contributing

Contributions welcome! Please submit a Pull Request.

---

## License

MIT License

---

## Need Help?

- **Issues:** [GitHub Issues](https://github.com/yourusername/well-log-toolkit/issues)
- **Documentation:** See sections above
- **Examples:** Check `/examples` directory (if available)
