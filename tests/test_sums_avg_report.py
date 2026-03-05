"""Tests for SumsAvgResult.report() aggregation and reporting."""
import numpy as np
import pandas as pd
import pytest

from well_log_toolkit import WellDataManager
from well_log_toolkit.analysis.sums_avg import SumsAvgResult


@pytest.fixture
def manager_for_report():
    """Manager with 2 wells, each having PHIE, Zone and Facies for report testing."""
    manager = WellDataManager()
    depth = np.arange(1000.0, 1020.0, 1.0)  # 20 samples

    for wname, phie_base in [('Well_A', 0.18), ('Well_B', 0.22)]:
        zone_vals = np.array([0]*5 + [1]*10 + [0]*5, dtype=float)
        facies_vals = np.array([0]*5 + [1]*5 + [2]*5 + [0]*5, dtype=float)
        phie_vals = np.full(20, phie_base) + np.random.RandomState(42).uniform(-0.02, 0.02, 20)
        df = pd.DataFrame({
            'DEPT': depth,
            'PHIE': phie_vals,
            'Zone': zone_vals,
            'Facies': facies_vals,
        })
        manager.load_properties(
            df, well_col=None, well_name=wname, source_name='petro',
            type_mappings={'Zone': 'discrete', 'Facies': 'discrete'},
            label_mappings={
                'Zone': {0: 'NonRes', 1: 'Reservoir'},
                'Facies': {0: 'Shale', 1: 'Sand', 2: 'Silt'},
            },
        )

    return manager


class TestSumsAvgResult:
    """Tests for SumsAvgResult container."""

    def test_sums_avg_returns_sumsavgresult(self, manager_for_report):
        """sums_avg() on filtered property should return SumsAvgResult."""
        result = manager_for_report.PHIE.filter('Zone').sums_avg()
        assert isinstance(result, SumsAvgResult)

    def test_sums_avg_has_well_keys(self, manager_for_report):
        """Result should have well names as top-level keys."""
        result = manager_for_report.PHIE.filter('Zone').sums_avg()
        assert 'well_a' in result or 'Well_A' in result or any('well' in k.lower() for k in result)

    def test_sums_avg_has_zone_keys(self, manager_for_report):
        """Result should contain zone labels as keys."""
        result = manager_for_report.PHIE.filter('Zone').sums_avg()
        # Check that at least one well has zone data
        for well_name, well_data in result.items():
            if isinstance(well_data, dict):
                assert any(k in ['Reservoir', 'NonRes'] for k in well_data.keys())
                break

    def test_sums_avg_nested_filter(self, manager_for_report):
        """Double-filtered sums_avg should have nested dict structure."""
        result = manager_for_report.PHIE.filter('Zone').filter('Facies').sums_avg()
        assert isinstance(result, SumsAvgResult)
        # Should have nested structure: well -> zone -> facies -> stats
        for well_name, well_data in result.items():
            if isinstance(well_data, dict):
                for zone_name, zone_data in well_data.items():
                    if isinstance(zone_data, dict):
                        # Should find facies within zones
                        assert any(isinstance(v, dict) for v in zone_data.values())
                        break
                break


class TestSumsAvgReportMethod:
    """Tests for SumsAvgResult.report() method."""

    def test_report_returns_none_when_printing(self, manager_for_report, capsys):
        """report(print_report=True) should print and return None."""
        result = manager_for_report.PHIE.filter('Zone').filter('Facies').sums_avg()
        ret = result.report(
            zones=['Reservoir', 'NonRes'],
            groups={'Net': ['Sand', 'Silt'], 'NonNet': ['Shale']},
            columns=[{'property': 'PHIE', 'stat': 'mean', 'label': 'por', 'format': '.4f'}],
            print_report=True,
        )
        assert ret is None
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Something was printed

    def test_report_returns_dict_when_not_printing(self, manager_for_report):
        """report(print_report=False) should return structured data dict."""
        result = manager_for_report.PHIE.filter('Zone').filter('Facies').sums_avg()
        report_data = result.report(
            zones=['Reservoir', 'NonRes'],
            groups={'Net': ['Sand', 'Silt'], 'NonNet': ['Shale']},
            columns=[{'property': 'PHIE', 'stat': 'mean', 'label': 'por', 'format': '.4f'}],
            print_report=False,
        )
        assert isinstance(report_data, dict)

    def test_report_validates_columns(self):
        """report() should raise ValueError for missing required column fields."""
        result = SumsAvgResult({'test': {}})
        with pytest.raises(ValueError, match="missing required"):
            result.report(
                zones=['Zone1'],
                groups={'G': ['F']},
                columns=[{'label': 'test'}],  # Missing 'property' and 'stat'
            )

    def test_report_validates_pooled_without_mean(self):
        """report() should raise ValueError for pooled std without corresponding mean column."""
        result = SumsAvgResult({'test': {}})
        with pytest.raises(ValueError, match="pooled"):
            result.report(
                zones=['Zone1'],
                groups={'G': ['F']},
                columns=[
                    {'property': 'PHIE', 'stat': 'std_dev', 'agg': 'pooled'},
                    # No mean column for PHIE — should fail
                ],
            )
