"""Tests for error handling and exception raising across the toolkit."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pylog import Well, Property, WellDataManager
from pylog.exceptions import (
    LasFileError,
    PropertyNotFoundError,
    PropertyTypeError,
    DepthAlignmentError,
)


class TestPropertyNotFoundError:
    """Tests for missing property access on Well."""

    def test_well_getattr_missing_property(self, well_with_properties):
        with pytest.raises(AttributeError, match="no source or property named"):
            _ = well_with_properties.NONEXISTENT

    def test_well_getattr_lists_available(self, well_with_properties):
        with pytest.raises(AttributeError, match="PHIE"):
            _ = well_with_properties.NONEXISTENT

    def test_well_get_property_missing(self, well_with_properties):
        with pytest.raises((AttributeError, PropertyNotFoundError)):
            well_with_properties.get_property('NONEXISTENT')


class TestPropertyTypeError:
    """Tests for filtering on non-discrete properties."""

    def test_filter_on_continuous_property(self, well_with_properties):
        """Filtering on a continuous property should raise PropertyTypeError."""
        with pytest.raises(PropertyTypeError, match="must be discrete"):
            well_with_properties.PHIE.filter('SW')


class TestDepthAlignmentError:
    """Tests for mismatched depth grids in operations."""

    def test_mismatched_depth_grids(self):
        depth_a = np.arange(1000.0, 1010.0, 1.0)
        depth_b = np.arange(2000.0, 2010.0, 1.0)
        prop_a = Property(name='A', depth=depth_a, values=np.ones(10))
        prop_b = Property(name='B', depth=depth_b, values=np.ones(10))
        with pytest.raises(DepthAlignmentError, match="different depth grids"):
            _ = prop_a + prop_b

    def test_error_shows_depth_details(self):
        depth_a = np.arange(1000.0, 1010.0, 1.0)
        depth_b = np.arange(1000.0, 1005.0, 0.5)
        prop_a = Property(name='A', depth=depth_a, values=np.ones(10))
        prop_b = Property(name='B', depth=depth_b, values=np.ones(10))
        with pytest.raises(DepthAlignmentError, match="samples"):
            _ = prop_a + prop_b


class TestLasFileError:
    """Tests for LAS file loading errors."""

    def test_missing_file(self):
        with pytest.raises(LasFileError, match="File not found"):
            from pylog.io import LasFile
            LasFile("/nonexistent/path/to/file.las")


class TestWellGetattr:
    """Tests for Well.__getattr__ helpful error messages."""

    def test_lists_available_properties(self, well_with_properties):
        try:
            _ = well_with_properties.NONEXISTENT
        except AttributeError as e:
            msg = str(e)
            assert "PHIE" in msg
            assert "SW" in msg
            assert "Zone" in msg

    def test_lists_available_sources(self, well_with_properties):
        try:
            _ = well_with_properties.NONEXISTENT
        except AttributeError as e:
            msg = str(e)
            assert "source" in msg.lower() or "external_df" in msg


class TestFilterValidation:
    """Tests for Property.filter() input validation."""

    def test_filter_missing_property(self, well_with_properties):
        """Filtering by a non-existent property should raise PropertyNotFoundError."""
        with pytest.raises(PropertyNotFoundError):
            well_with_properties.PHIE.filter('NONEXISTENT')

    def test_filter_suggests_discrete_type(self, well_with_properties):
        """Filtering on continuous property should suggest setting type to 'discrete'."""
        with pytest.raises(PropertyTypeError, match="discrete"):
            well_with_properties.PHIE.filter('SW')


class TestFuzzyMatching:
    """Tests for 'Did you mean' suggestions in error messages."""

    def test_well_getattr_suggests_similar(self, well_with_properties):
        """Typo in property name should suggest the correct name."""
        with pytest.raises(AttributeError, match="Did you mean.*PHIE"):
            _ = well_with_properties.PHI  # close to PHIE

    def test_filter_suggests_similar_property(self, well_with_properties):
        """Typo in filter property name should suggest similar names."""
        with pytest.raises(PropertyNotFoundError, match="Did you mean.*Zone"):
            well_with_properties.PHIE.filter('Zon')

    def test_manager_well_suggests_similar(self, manager_with_wells):
        """Typo in well name should suggest similar well names."""
        with pytest.raises(AttributeError, match="Did you mean"):
            _ = manager_with_wells.well_Wll_A  # close to well_Well_A or similar


class TestInputValidation:
    """Tests for input validation at construction time."""

    def test_property_rejects_mismatched_lengths(self):
        """Property should reject depth/values with different lengths."""
        with pytest.raises(ValueError, match="depth length.*values length"):
            Property(
                name='BAD',
                depth=np.array([1.0, 2.0, 3.0]),
                values=np.array([0.1, 0.2]),
            )

    def test_property_rejects_non_monotonic_depth(self):
        """Property should reject non-monotonically increasing depth."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            Property(
                name='BAD',
                depth=np.array([3.0, 1.0, 2.0]),
                values=np.array([0.1, 0.2, 0.3]),
            )

    def test_las_rejects_non_las_extension(self):
        """LasFile should reject files without .las extension."""
        from pylog.io import LasFile
        with pytest.raises(LasFileError, match="Expected .las"):
            LasFile("/some/path/data.csv")


class TestValidateMethod:
    """Tests for WellDataManager.validate()."""

    def test_validate_clean_data(self, manager_with_wells):
        """Clean data should return empty issues dict."""
        issues = manager_with_wells.validate()
        assert isinstance(issues, dict)

    def test_validate_detects_missing_properties(self):
        """validate() should detect missing properties across wells."""
        manager = WellDataManager()
        depth = np.arange(1000.0, 1005.0, 1.0)

        # Well A has PHIE + SW
        well_a = Well(name='A', sanitized_name='well_A')
        well_a.add_dataframe(
            __import__('pandas').DataFrame({
                'DEPT': depth, 'PHIE': np.ones(5) * 0.2, 'SW': np.ones(5) * 0.5,
            })
        )
        manager._wells['well_A'] = well_a

        # Well B has only PHIE (missing SW)
        well_b = Well(name='B', sanitized_name='well_B')
        well_b.add_dataframe(
            __import__('pandas').DataFrame({
                'DEPT': depth, 'PHIE': np.ones(5) * 0.18,
            })
        )
        manager._wells['well_B'] = well_b

        issues = manager.validate()
        assert 'well_B' in issues
        assert any('SW' in issue for issue in issues['well_B'])


class TestProxyWarnings:
    """Tests for proxy stat warnings when wells are skipped."""

    def test_proxy_warns_skipped_wells(self):
        """Stats should warn when wells lack the requested property."""
        manager = WellDataManager()
        depth = np.arange(1000.0, 1005.0, 1.0)

        # Well A has PHIE
        well_a = Well(name='A', sanitized_name='well_A')
        well_a.add_dataframe(
            __import__('pandas').DataFrame({
                'DEPT': depth, 'PHIE': np.ones(5) * 0.2,
            })
        )
        manager._wells['well_A'] = well_a

        # Well B has no PHIE
        well_b = Well(name='B', sanitized_name='well_B')
        well_b.add_dataframe(
            __import__('pandas').DataFrame({
                'DEPT': depth, 'SW': np.ones(5) * 0.5,
            })
        )
        manager._wells['well_B'] = well_b

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.PHIE.mean()
            assert any("Skipped" in str(warning.message) and "well_B" in str(warning.message)
                        for warning in w)
