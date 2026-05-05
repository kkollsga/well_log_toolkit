"""Tests for the Artifact protocol and RegressionFit."""

import numpy as np
import pytest

from logsuite import (
    Artifact,
    ExponentialRegression,
    LinearRegression,
    LogarithmicRegression,
    PowerRegression,
    RegressionFit,
)


class TestArtifactBase:
    def test_unimplemented_raises_type_error(self):
        class Empty(Artifact):
            pass

        art = Empty()
        with pytest.raises(TypeError, match="cannot be rendered in a Crossplot"):
            art._render_in_crossplot(None)
        with pytest.raises(TypeError, match="cannot be rendered in a Table"):
            art._render_in_table(None)
        with pytest.raises(TypeError, match="cannot be rendered in a WellView"):
            art._render_in_wellview(None)

    def test_subclass_can_implement_one_consumer(self):
        class CrossplotOnly(Artifact):
            def _render_in_crossplot(self, ax, **kwargs):
                ax.called = True

        class Ax:
            called = False

        ax = Ax()
        CrossplotOnly()._render_in_crossplot(ax)
        assert ax.called is True
        # Other consumers still fail
        with pytest.raises(TypeError):
            CrossplotOnly()._render_in_table(None)


class TestRegressionFitConstruction:
    def test_unfitted_model_raises(self):
        reg = LinearRegression()  # not fitted
        with pytest.raises(ValueError, match="fitted"):
            RegressionFit(reg)

    def test_fitted_model_accepts(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg, name="line")
        assert fit.name == "line"
        assert fit.r_squared is not None

    def test_default_name_uses_model_class(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg)
        assert fit.name == "LinearRegression"


class TestEquationFormatting:
    @pytest.fixture
    def exp_fit(self):
        x = np.linspace(0, 1, 50)
        y = 0.01 * np.exp(7.0 * x)  # a=0.01, b=7
        reg = ExponentialRegression().fit(x, y)
        return RegressionFit(reg)

    def test_natural_form_for_exponential(self, exp_fit):
        eq = exp_fit.equation(format="natural")
        assert "e^" in eq
        assert "y =" in eq

    def test_log10_form_for_exponential(self, exp_fit):
        eq = exp_fit.equation(format="log10")
        assert "10^" in eq

    def test_petrel_form_for_exponential(self, exp_fit):
        eq = exp_fit.equation(format="petrel")
        assert "pow(10," in eq
        assert "*x" in eq
        # Sign chosen by c0 = log10(a); always one of "+" or "-" between *x and the constant
        assert " + " in eq or " - " in eq

    def test_log10_coefficients_correct(self, exp_fit):
        # y = a*e^(b*x) → 10^(c1*x + c0); c0 = log10(a), c1 = b/ln(10)
        a = exp_fit.model.a
        b = exp_fit.model.b
        c0_expected = np.log10(a)
        c1_expected = b / np.log(10)
        eq = exp_fit.equation(format="log10", decimals=6)
        # Loose check: both numbers appear in the string
        assert f"{c1_expected:.6f}" in eq
        assert f"{abs(c0_expected):.6f}" in eq

    def test_decimals_kwarg_overrides_default(self, exp_fit):
        fit = RegressionFit(exp_fit.model, decimals=4)
        eq2 = fit.equation(decimals=2)
        eq6 = fit.equation(decimals=6)
        # eq2 should have 2-decimal numbers, eq6 should have 6-decimal numbers
        assert eq2 != eq6

    def test_unknown_format_raises(self, exp_fit):
        with pytest.raises(ValueError, match="Unknown equation format"):
            exp_fit.equation(format="bogus")

    def test_linear_natural_format(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg, decimals=2)
        eq = fit.equation()
        assert "y =" in eq
        assert "x" in eq

    def test_logarithmic_natural_format(self):
        reg = LogarithmicRegression().fit([1, 2, 4, 8], [1, 2, 3, 4])
        fit = RegressionFit(reg, decimals=2)
        assert "ln(x)" in fit.equation()

    def test_power_natural_format(self):
        reg = PowerRegression().fit([1, 2, 4, 8], [2, 4, 8, 16])
        fit = RegressionFit(reg, decimals=2)
        assert "x^" in fit.equation()

    def test_petrel_falls_back_for_non_exponential(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg)
        eq = fit.equation(format="petrel")
        # Non-exponential petrel falls back to natural form
        assert "pow(10," not in eq


class TestRegressionFitLabel:
    def test_label_includes_name_and_equation(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg, name="myfit")
        label = fit.label()
        assert "myfit" in label
        assert "y =" in label

    def test_label_show_r2_true(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg)
        assert "R²=" in fit.label(show_r2=True)
        assert "R²=" not in fit.label(show_r2=False)


class TestRegressionFitRender:
    def test_render_in_crossplot_calls_ax_plot(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg, line_color="red")

        plot_calls = []

        class MockAx:
            def plot(self, x, y, **kwargs):
                plot_calls.append((x, y, kwargs))

        fit._render_in_crossplot(MockAx())
        assert len(plot_calls) == 1
        x_arr, y_arr, kwargs = plot_calls[0]
        assert len(x_arr) == 200
        assert len(y_arr) == 200
        assert kwargs["color"] == "red"
        assert "label" in kwargs

    def test_render_with_explicit_x_range(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg)

        plot_calls = []

        class MockAx:
            def plot(self, x, y, **kwargs):
                plot_calls.append(x)

        fit._render_in_crossplot(MockAx(), x_range=(0, 10))
        x_arr = plot_calls[0]
        assert x_arr[0] == 0
        assert x_arr[-1] == 10


class TestRegressionFitTable:
    def test_table_row_has_expected_keys(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg, name="test")
        row = fit._render_in_table(None)
        assert row["name"] == "test"
        assert "equation" in row
        assert "r_squared" in row
        assert "x_range" in row


class TestArtifactDispatch:
    def test_regression_fit_does_not_render_in_wellview(self):
        reg = LinearRegression().fit([1, 2, 3, 4], [2, 4, 6, 8])
        fit = RegressionFit(reg)
        with pytest.raises(TypeError, match="cannot be rendered in a WellView"):
            fit._render_in_wellview(None)


class TestProxyFit:
    @pytest.fixture
    def manager(self):
        import pandas as pd

        from logsuite import WellDataManager

        mgr = WellDataManager()
        depth = np.arange(1000.0, 1100.0, 1.0)
        rng = np.random.default_rng(0)
        for wname, base in [("Well_A", 0.18), ("Well_B", 0.22)]:
            phi = np.clip(rng.normal(base, 0.03, size=len(depth)), 0.05, 0.35)
            perm = 0.01 * np.exp(20 * phi)
            df = pd.DataFrame({"DEPT": depth, "PHIE": phi, "PERM": perm})
            mgr.load_properties(
                df,
                well_col=None,
                well_name=wname,
                source_name="petrophysics",
            )
        return mgr

    def test_fit_returns_regression_fit(self, manager):
        fit = manager.properties(["PHIE", "PERM"]).fit(ExponentialRegression())
        assert isinstance(fit, RegressionFit)

    def test_fit_default_name_uses_properties(self, manager):
        fit = manager.properties(["PHIE", "PERM"]).fit(ExponentialRegression())
        assert "PHIE" in fit.name and "PERM" in fit.name

    def test_fit_custom_name_kwargs(self, manager):
        fit = manager.properties(["PHIE", "PERM"]).fit(
            ExponentialRegression(), name="all wells", equation_format="petrel"
        )
        assert fit.name == "all wells"
        assert "pow(10," in fit.equation()

    def test_fit_requires_two_properties(self, manager):
        with pytest.raises(ValueError, match="exactly 2 properties"):
            manager.properties(["PHIE"]).fit(ExponentialRegression())
        with pytest.raises(ValueError, match="exactly 2 properties"):
            manager.properties(["PHIE", "PERM", "DEPT"]).fit(ExponentialRegression())

    def test_fit_with_where_uses_filtered_data(self, manager):
        # All wells fit
        fit_all = manager.properties(["PHIE", "PERM"]).fit(ExponentialRegression())
        # Only Well_A
        fit_a = (
            manager.filter(where={"well": ["Well_A"]})
            .properties(["PHIE", "PERM"])
            .fit(ExponentialRegression())
        )
        # Different fits, generally different parameters
        assert fit_all.model.a != fit_a.model.a or fit_all.model.b != fit_a.model.b
