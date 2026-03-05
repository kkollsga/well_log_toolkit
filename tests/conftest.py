"""
Centralized pytest fixtures for logsuite test suite.

Provides reusable depth arrays, properties, wells, and managers
for tests without requiring external LAS files.
"""
import textwrap

import numpy as np
import pandas as pd
import pytest

from logsuite import WellDataManager, Well, Property


# ---------------------------------------------------------------------------
# Depth arrays
# ---------------------------------------------------------------------------

@pytest.fixture
def depth_short():
    """Short 10-sample depth grid at 1m spacing (1000-1009m)."""
    return np.arange(1000.0, 1010.0, 1.0)


@pytest.fixture
def depth_uniform():
    """Standard 100-sample depth grid at 0.15m spacing (2800-2815m)."""
    return np.arange(2800.0, 2815.0, 0.15)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@pytest.fixture
def continuous_property(depth_short):
    """Continuous PHIE property with known values on short depth grid."""
    values = np.array([0.15, 0.20, 0.22, 0.18, 0.25, 0.19, 0.21, 0.17, 0.23, 0.16])
    return Property(name='PHIE', depth=depth_short, values=values, unit='v/v')


@pytest.fixture
def sw_property(depth_short):
    """Continuous SW property with known values on short depth grid."""
    values = np.array([0.30, 0.35, 0.32, 0.28, 0.40, 0.38, 0.33, 0.29, 0.42, 0.31])
    return Property(name='SW', depth=depth_short, values=values, unit='v/v')


@pytest.fixture
def discrete_property(depth_short):
    """Discrete Zone property with 2 zones on short depth grid."""
    values = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0], dtype=float)
    return Property(
        name='Zone', depth=depth_short, values=values, unit='',
        prop_type='discrete', labels={0: 'NonReservoir', 1: 'Reservoir'}
    )


@pytest.fixture
def ntg_property(depth_short):
    """Discrete NTG property for secondary filtering."""
    values = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=float)
    return Property(
        name='NTG', depth=depth_short, values=values, unit='',
        prop_type='discrete', labels={0: 'NonNet', 1: 'Net'}
    )


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

@pytest.fixture
def well_with_properties(depth_short):
    """Well with PHIE, SW, Zone, and NTG properties from a DataFrame source."""
    well = Well(name='Test Well A', sanitized_name='test_well_a')
    df = pd.DataFrame({
        'DEPT': depth_short,
        'PHIE': [0.15, 0.20, 0.22, 0.18, 0.25, 0.19, 0.21, 0.17, 0.23, 0.16],
        'SW':   [0.30, 0.35, 0.32, 0.28, 0.40, 0.38, 0.33, 0.29, 0.42, 0.31],
        'Zone': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        'NTG':  [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    })
    well.add_dataframe(
        df,
        unit_mappings={'PHIE': 'v/v', 'SW': 'v/v'},
        type_mappings={'Zone': 'discrete', 'NTG': 'discrete'},
        label_mappings={
            'Zone': {0: 'NonReservoir', 1: 'Reservoir'},
            'NTG': {0: 'NonNet', 1: 'Net'},
        }
    )
    return well


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

@pytest.fixture
def manager_with_wells():
    """Manager with 2 synthetic wells for broadcasting tests."""
    manager = WellDataManager()
    depth = np.arange(1000.0, 1010.0, 1.0)

    for i, (wname, phie_base, sw_base) in enumerate([
        ('Well_A', 0.18, 0.30),
        ('Well_B', 0.22, 0.35),
    ]):
        df = pd.DataFrame({
            'DEPT': depth,
            'PHIE': np.linspace(phie_base - 0.03, phie_base + 0.03, len(depth)),
            'SW': np.linspace(sw_base - 0.05, sw_base + 0.05, len(depth)),
            'Zone': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        })
        manager.load_properties(
            df, well_col=None, well_name=wname, source_name='petrophysics',
            unit_mappings={'PHIE': 'v/v', 'SW': 'v/v'},
            type_mappings={'Zone': 'discrete'},
            label_mappings={'Zone': {0: 'NonReservoir', 1: 'Reservoir'}},
        )

    return manager


# ---------------------------------------------------------------------------
# LAS file content
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_las_content():
    """Valid LAS 2.0 file content as a string."""
    return textwrap.dedent("""\
        ~VERSION INFORMATION
         VERS.                          2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
         WRAP.                          NO  : ONE LINE PER DEPTH STEP
        ~WELL INFORMATION
         WELL.                    Test Well : WELL
         STRT.m                     1000.0  : START DEPTH
         STOP.m                     1004.0  : STOP DEPTH
         STEP.m                        1.0  : STEP
         NULL.                     -999.25  : NULL VALUE
         COMP.                   TestCo     : COMPANY
        ~CURVE INFORMATION
         DEPT.m                             : DEPTH
         PHIE.v/v                           : POROSITY
         GR  .gAPI                          : GAMMA RAY
        ~A
         1000.0  0.150   45.2
         1001.0  0.200   62.1
         1002.0  0.220   55.8
         1003.0  0.180   71.3
         1004.0  0.250   48.9
    """)


@pytest.fixture
def tmp_las_file(tmp_path, sample_las_content):
    """Write sample LAS content to a temp file and return its path."""
    las_path = tmp_path / "test_well.las"
    las_path.write_text(sample_las_content)
    return las_path
