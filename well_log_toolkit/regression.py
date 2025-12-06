"""Regression classes for crossplot analysis.

This module provides various regression classes that can fit data and be used
for prediction. Each regression class can be used independently or as part of
crossplot visualizations.
"""

from typing import Optional, Union, Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod


class RegressionBase(ABC):
    """Base class for all regression types."""

    def __init__(self):
        self.fitted = False
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.r_squared: Optional[float] = None
        self.rmse: Optional[float] = None

    @abstractmethod
    def fit(self, x: ArrayLike, y: ArrayLike) -> 'RegressionBase':
        """Fit the regression model to data.

        Args:
            x: Independent variable values
            y: Dependent variable values

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict y values for given x values.

        Args:
            x: Independent variable values

        Returns:
            Predicted y values
        """
        pass

    @abstractmethod
    def equation(self) -> str:
        """Return the regression equation as a string."""
        pass

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Allow calling the regression object directly for prediction.

        Args:
            x: Independent variable values

        Returns:
            Predicted y values
        """
        return self.predict(x)

    def _calculate_metrics(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        """Calculate R² and RMSE metrics.

        Args:
            x: Independent variable values
            y: Actual dependent variable values
            y_pred: Predicted dependent variable values
        """
        # Store original data
        self.x_data = x
        self.y_data = y

        # R² calculation
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # RMSE calculation
        self.rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    def _prepare_data(self, x: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and clean data for regression.

        Args:
            x: Independent variable values
            y: Dependent variable values

        Returns:
            Tuple of cleaned x and y arrays with NaN/inf removed
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Remove NaN and inf values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) == 0:
            raise ValueError("No valid data points after removing NaN/inf values")

        return x_clean, y_clean


class LinearRegression(RegressionBase):
    """Linear regression: y = a*x + b

    Example:
        >>> reg = LinearRegression()
        >>> reg.fit([1, 2, 3, 4], [2, 4, 6, 8])
        >>> reg.predict([5, 6])
        array([10., 12.])
        >>> print(reg.equation())
        y = 2.00x + 0.00
        >>> print(f"R² = {reg.r_squared:.3f}")
        R² = 1.000
    """

    def __init__(self):
        super().__init__()
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> 'LinearRegression':
        """Fit linear regression model.

        Args:
            x: Independent variable values
            y: Dependent variable values

        Returns:
            Self for method chaining
        """
        x_clean, y_clean = self._prepare_data(x, y)

        # Calculate slope and intercept using least squares
        self.slope, self.intercept = np.polyfit(x_clean, y_clean, 1)

        # Calculate metrics
        y_pred = self.slope * x_clean + self.intercept
        self._calculate_metrics(x_clean, y_clean, y_pred)

        self.fitted = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict y values using linear model.

        Args:
            x: Independent variable values

        Returns:
            Predicted y values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        x = np.asarray(x, dtype=float)
        return self.slope * x + self.intercept

    def equation(self) -> str:
        """Return the linear equation as a string."""
        if not self.fitted:
            return "Model not fitted"

        sign = "+" if self.intercept >= 0 else "-"
        return f"y = {self.slope:.4f}x {sign} {abs(self.intercept):.4f}"


class LogarithmicRegression(RegressionBase):
    """Logarithmic regression: y = a*ln(x) + b

    Note: Only valid for positive x values.

    Example:
        >>> reg = LogarithmicRegression()
        >>> reg.fit([1, 2, 4, 8], [1, 2, 3, 4])
        >>> reg.predict([16])
        array([5.])
    """

    def __init__(self):
        super().__init__()
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> 'LogarithmicRegression':
        """Fit logarithmic regression model.

        Args:
            x: Independent variable values (must be positive)
            y: Dependent variable values

        Returns:
            Self for method chaining
        """
        x_clean, y_clean = self._prepare_data(x, y)

        # Check for positive x values
        if np.any(x_clean <= 0):
            raise ValueError("Logarithmic regression requires all x values to be positive")

        # Transform to linear: y = a*ln(x) + b
        ln_x = np.log(x_clean)
        self.a, self.b = np.polyfit(ln_x, y_clean, 1)

        # Calculate metrics
        y_pred = self.a * ln_x + self.b
        self._calculate_metrics(x_clean, y_clean, y_pred)

        self.fitted = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict y values using logarithmic model.

        Args:
            x: Independent variable values (must be positive)

        Returns:
            Predicted y values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        x = np.asarray(x, dtype=float)

        if np.any(x <= 0):
            raise ValueError("Logarithmic regression requires all x values to be positive")

        return self.a * np.log(x) + self.b

    def equation(self) -> str:
        """Return the logarithmic equation as a string."""
        if not self.fitted:
            return "Model not fitted"

        sign = "+" if self.b >= 0 else "-"
        return f"y = {self.a:.4f}*ln(x) {sign} {abs(self.b):.4f}"


class ExponentialRegression(RegressionBase):
    """Exponential regression: y = a*e^(b*x)

    Note: Only valid for positive y values.

    Example:
        >>> reg = ExponentialRegression()
        >>> reg.fit([0, 1, 2, 3], [1, 2.7, 7.4, 20.1])
        >>> reg.predict([4])
        array([54.6])
    """

    def __init__(self):
        super().__init__()
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> 'ExponentialRegression':
        """Fit exponential regression model.

        Args:
            x: Independent variable values
            y: Dependent variable values (must be positive)

        Returns:
            Self for method chaining
        """
        x_clean, y_clean = self._prepare_data(x, y)

        # Check for positive y values
        if np.any(y_clean <= 0):
            raise ValueError("Exponential regression requires all y values to be positive")

        # Transform to linear: ln(y) = ln(a) + b*x
        ln_y = np.log(y_clean)
        b, ln_a = np.polyfit(x_clean, ln_y, 1)
        self.b = b
        self.a = np.exp(ln_a)

        # Calculate metrics
        y_pred = self.a * np.exp(self.b * x_clean)
        self._calculate_metrics(x_clean, y_clean, y_pred)

        self.fitted = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict y values using exponential model.

        Args:
            x: Independent variable values

        Returns:
            Predicted y values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        x = np.asarray(x, dtype=float)
        return self.a * np.exp(self.b * x)

    def equation(self) -> str:
        """Return the exponential equation as a string."""
        if not self.fitted:
            return "Model not fitted"

        return f"y = {self.a:.4f}*e^({self.b:.4f}x)"


class PolynomialRegression(RegressionBase):
    """Polynomial regression: y = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0

    Example:
        >>> reg = PolynomialRegression(degree=2)
        >>> reg.fit([1, 2, 3, 4], [1, 4, 9, 16])
        >>> reg.predict([5])
        array([25.])
        >>> print(reg.equation())
        y = 1.00x² + 0.00x + 0.00
    """

    def __init__(self, degree: int = 2):
        """Initialize polynomial regression.

        Args:
            degree: Polynomial degree (default: 2 for quadratic)
        """
        super().__init__()
        if degree < 1:
            raise ValueError("Polynomial degree must be at least 1")
        self.degree = degree
        self.coefficients: Optional[np.ndarray] = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> 'PolynomialRegression':
        """Fit polynomial regression model.

        Args:
            x: Independent variable values
            y: Dependent variable values

        Returns:
            Self for method chaining
        """
        x_clean, y_clean = self._prepare_data(x, y)

        # Fit polynomial
        self.coefficients = np.polyfit(x_clean, y_clean, self.degree)

        # Calculate metrics
        y_pred = np.polyval(self.coefficients, x_clean)
        self._calculate_metrics(x_clean, y_clean, y_pred)

        self.fitted = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict y values using polynomial model.

        Args:
            x: Independent variable values

        Returns:
            Predicted y values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        x = np.asarray(x, dtype=float)
        return np.polyval(self.coefficients, x)

    def equation(self) -> str:
        """Return the polynomial equation as a string."""
        if not self.fitted:
            return "Model not fitted"

        terms = []
        for i, coef in enumerate(self.coefficients):
            power = self.degree - i

            if abs(coef) < 1e-10:  # Skip near-zero coefficients
                continue

            # Format coefficient
            if i == 0:
                coef_str = f"{coef:.4f}"
            else:
                sign = "+" if coef >= 0 else "-"
                coef_str = f"{sign} {abs(coef):.4f}"

            # Format power
            if power == 0:
                term = coef_str
            elif power == 1:
                term = f"{coef_str}x"
            elif power == 2:
                term = f"{coef_str}x²"
            elif power == 3:
                term = f"{coef_str}x³"
            else:
                term = f"{coef_str}x^{power}"

            terms.append(term)

        if not terms:
            return "y = 0"

        equation = "y = " + "".join(terms).strip()
        # Clean up leading plus sign
        equation = equation.replace("= +", "= ")
        return equation


class PowerRegression(RegressionBase):
    """Power regression: y = a*x^b

    Note: Only valid for positive x and y values.

    Example:
        >>> reg = PowerRegression()
        >>> reg.fit([1, 2, 3, 4], [1, 4, 9, 16])
        >>> reg.predict([5])
        array([25.])
    """

    def __init__(self):
        super().__init__()
        self.a: Optional[float] = None
        self.b: Optional[float] = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> 'PowerRegression':
        """Fit power regression model.

        Args:
            x: Independent variable values (must be positive)
            y: Dependent variable values (must be positive)

        Returns:
            Self for method chaining
        """
        x_clean, y_clean = self._prepare_data(x, y)

        # Check for positive values
        if np.any(x_clean <= 0):
            raise ValueError("Power regression requires all x values to be positive")
        if np.any(y_clean <= 0):
            raise ValueError("Power regression requires all y values to be positive")

        # Transform to linear: ln(y) = ln(a) + b*ln(x)
        ln_x = np.log(x_clean)
        ln_y = np.log(y_clean)
        self.b, ln_a = np.polyfit(ln_x, ln_y, 1)
        self.a = np.exp(ln_a)

        # Calculate metrics
        y_pred = self.a * np.power(x_clean, self.b)
        self._calculate_metrics(x_clean, y_clean, y_pred)

        self.fitted = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Predict y values using power model.

        Args:
            x: Independent variable values (must be positive)

        Returns:
            Predicted y values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        x = np.asarray(x, dtype=float)

        if np.any(x <= 0):
            raise ValueError("Power regression requires all x values to be positive")

        return self.a * np.power(x, self.b)

    def equation(self) -> str:
        """Return the power equation as a string."""
        if not self.fitted:
            return "Model not fitted"

        return f"y = {self.a:.4f}*x^{self.b:.4f}"


__all__ = [
    'RegressionBase',
    'LinearRegression',
    'LogarithmicRegression',
    'ExponentialRegression',
    'PolynomialRegression',
    'PowerRegression'
]
