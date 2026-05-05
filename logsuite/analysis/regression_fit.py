"""
RegressionFit — a fitted regression as a renderable artifact.

Wraps a fitted :class:`RegressionBase` and adds presentation concerns:
multiple equation formats (natural log / log10 / Petrel calculator syntax),
configurable decimal precision, and dispatch methods for visualization
consumers (``Crossplot``, ``Table``).

This module belongs to the ``analysis`` layer. It depends on ``core``
(``Artifact``) and on its sibling ``regression`` module for the model
classes it wraps; it imports matplotlib only for runtime drawing in
``_render_in_crossplot``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ..core.artifact import Artifact
from .regression import (
    ExponentialRegression,
    LinearRegression,
    LogarithmicRegression,
    PolynomialRegression,
    PowerRegression,
    RegressionBase,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


EquationFormat = Literal["natural", "log10", "petrel"]


class RegressionFit(Artifact):
    """A fitted regression as a renderable artifact.

    Wraps a fitted :class:`RegressionBase`. Provides:

    - :meth:`equation` with ``natural`` / ``log10`` / ``petrel`` formatting
      (Petrel form is currently meaningful for :class:`ExponentialRegression`;
      other models fall back to natural).
    - :meth:`label` for legend / table row text.
    - :meth:`_render_in_crossplot` to draw the fitted curve on a matplotlib
      axis.
    - :meth:`_render_in_table` to produce a row dict for tabular consumers.

    Parameters
    ----------
    model : RegressionBase
        A fitted regression model. Must have ``fitted=True``.
    name : str, optional
        Display name used for legend / table rows. Defaults to the model
        class name.
    decimals : int, default 4
        Default decimal precision for equation/label formatting.
    equation_format : {"natural", "log10", "petrel"}, default "natural"
        Default equation format. Overridable per-call on :meth:`equation`.
    line_color, line_width, line_style, line_alpha :
        Styling for ``_render_in_crossplot``.

    Raises
    ------
    ValueError
        If ``model.fitted`` is not ``True``.

    Examples
    --------
    >>> reg = ExponentialRegression().fit(phi, perm)
    >>> fit = RegressionFit(reg, name="Sand 2", equation_format="petrel")
    >>> fit.equation()
    'pow(10, 3.4567x + -1.2345)'
    >>> crossplot.add(fit)
    """

    def __init__(
        self,
        model: RegressionBase,
        name: str | None = None,
        decimals: int = 4,
        equation_format: EquationFormat = "natural",
        line_color: str | None = None,
        line_width: float = 2.0,
        line_style: str = "-",
        line_alpha: float = 1.0,
    ):
        if not getattr(model, "fitted", False):
            raise ValueError(
                "RegressionFit requires a fitted model. " "Call .fit(x, y) on the model first."
            )
        self.model = model
        self.name = name or type(model).__name__
        self.decimals = decimals
        self.equation_format: EquationFormat = equation_format
        self.line_color = line_color
        self.line_width = line_width
        self.line_style = line_style
        self.line_alpha = line_alpha

    @property
    def r_squared(self) -> float | None:
        return self.model.r_squared

    @property
    def x_range(self) -> tuple[float, float] | None:
        return self.model.x_range

    def equation(
        self,
        format: EquationFormat | None = None,
        decimals: int | None = None,
    ) -> str:
        """Return the equation as a string in the requested format.

        Parameters
        ----------
        format : {"natural", "log10", "petrel"}, optional
            Override the artifact's stored format. ``natural`` matches the
            model's own form. ``log10`` and ``petrel`` are currently only
            meaningful for :class:`ExponentialRegression`; other models
            fall back to natural.
        decimals : int, optional
            Override the artifact's stored decimal precision.
        """
        fmt = format if format is not None else self.equation_format
        d = decimals if decimals is not None else self.decimals

        if fmt == "natural":
            return self._format_natural(d)
        if fmt == "log10":
            return self._format_log10(d)
        if fmt == "petrel":
            return self._format_petrel(d)
        raise ValueError(f"Unknown equation format '{fmt}'. Use 'natural', 'log10', or 'petrel'.")

    def _format_natural(self, decimals: int) -> str:
        m = self.model
        if isinstance(m, LinearRegression):
            sign = "+" if m.intercept >= 0 else "-"
            return f"y = {m.slope:.{decimals}f}x {sign} {abs(m.intercept):.{decimals}f}"
        if isinstance(m, ExponentialRegression):
            return f"y = {m.a:.{decimals}f}*e^({m.b:.{decimals}f}x)"
        if isinstance(m, LogarithmicRegression):
            sign = "+" if m.b >= 0 else "-"
            return f"y = {m.a:.{decimals}f}*ln(x) {sign} {abs(m.b):.{decimals}f}"
        if isinstance(m, PowerRegression):
            return f"y = {m.a:.{decimals}f}*x^{m.b:.{decimals}f}"
        if isinstance(m, PolynomialRegression):
            terms = []
            coefs = list(m.coefficients)
            n = len(coefs)
            for i, c in enumerate(coefs):
                power = n - 1 - i
                if power == 0:
                    terms.append(f"{c:.{decimals}f}")
                elif power == 1:
                    terms.append(f"{c:.{decimals}f}x")
                else:
                    terms.append(f"{c:.{decimals}f}x^{power}")
            return "y = " + " + ".join(terms)
        # Fallback: model's own equation (decimal control lost)
        return m.equation()

    def _format_log10(self, decimals: int) -> str:
        m = self.model
        if isinstance(m, ExponentialRegression):
            # y = a*e^(b*x) = 10^(c1*x + c0) where c0 = log10(a), c1 = b/ln(10)
            c0 = float(np.log10(m.a))
            c1 = float(m.b / np.log(10))
            sign = "+" if c0 >= 0 else "-"
            return f"y = 10^({c1:.{decimals}f}x {sign} {abs(c0):.{decimals}f})"
        return self._format_natural(decimals)

    def _format_petrel(self, decimals: int) -> str:
        m = self.model
        if isinstance(m, ExponentialRegression):
            c0 = float(np.log10(m.a))
            c1 = float(m.b / np.log(10))
            sign = "+" if c0 >= 0 else "-"
            return f"pow(10, {c1:.{decimals}f}*x {sign} {abs(c0):.{decimals}f})"
        return self._format_natural(decimals)

    def label(
        self,
        decimals: int | None = None,
        show_r2: bool = True,
        show_equation: bool = True,
    ) -> str:
        """Compose a legend / table label."""
        d = decimals if decimals is not None else self.decimals
        parts = [self.name]
        if show_equation:
            parts.append(self.equation(decimals=d))
        if show_r2 and self.r_squared is not None:
            parts.append(f"R²={self.r_squared:.3f}")
        return ": ".join([parts[0], " — ".join(parts[1:])]) if len(parts) > 1 else parts[0]

    def _render_in_crossplot(
        self,
        ax: Axes,
        x_range: tuple[float, float] | None = None,
        **_: object,
    ) -> None:
        """Draw the fitted curve on a matplotlib axis."""
        rng = x_range if x_range is not None else self.x_range
        if rng is None:
            raise ValueError(f"{type(self).__name__} has no x_range; cannot render in Crossplot.")
        x_plot = np.linspace(rng[0], rng[1], 200)
        y_plot = self.model.predict(x_plot)
        ax.plot(
            x_plot,
            y_plot,
            color=self.line_color,
            linewidth=self.line_width,
            linestyle=self.line_style,
            alpha=self.line_alpha,
            label=self.label(),
        )

    def _render_in_table(self, table: object, **_: object) -> dict:
        """Return a dict row for tabular consumers."""
        return {
            "name": self.name,
            "equation": self.equation(),
            "r_squared": self.r_squared,
            "x_range": self.x_range,
        }

    def __repr__(self) -> str:
        r2_str = f", R²={self.r_squared:.3f}" if self.r_squared is not None else ""
        return f"RegressionFit({self.name}: {self.equation(decimals=2)}{r2_str})"
