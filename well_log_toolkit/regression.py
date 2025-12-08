"""Regression classes for crossplot analysis.

This module provides various regression classes that can fit data and be used
for prediction. Each regression class can be used independently or as part of
crossplot visualizations.
"""

from typing import Optional, Union, Tuple, Callable, Dict
import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod


class RegressionBase(ABC):
    """Base class for all regression types."""

    def __init__(self, locked_params: Optional[Dict[str, float]] = None):
        """Initialize regression base class.

        Args:
            locked_params: Dictionary of parameter names and their locked values.
                          These parameters will not be fitted and will remain constant.
        """
        self.fitted = False
        self.x_data: Optional[np.ndarray] = None
        self.y_data: Optional[np.ndarray] = None
        self.r_squared: Optional[float] = None
        self.rmse: Optional[float] = None
        self.x_range: Optional[Tuple[float, float]] = None
        self._locked_params: Dict[str, float] = locked_params if locked_params is not None else {}

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

    def _calculate_metrics(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, use_log_space: bool = False) -> None:
        """Calculate R² and RMSE metrics.

        Args:
            x: Independent variable values
            y: Actual dependent variable values
            y_pred: Predicted dependent variable values
            use_log_space: If True, calculate R² in log space (useful when y spans orders of magnitude)
        """
        # Store original data
        self.x_data = x
        self.y_data = y

        # Store x-axis range
        self.x_range = (float(np.min(x)), float(np.max(x)))

        # R² calculation - in log space if requested
        if use_log_space and np.all(y > 0) and np.all(y_pred > 0):
            # Calculate R² in log space for data spanning orders of magnitude
            log_y = np.log10(y)
            log_y_pred = np.log10(y_pred)
            ss_res = np.sum((log_y - log_y_pred) ** 2)
            ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        else:
            # Standard R² in linear space
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # RMSE calculation (always in linear space)
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

    def lock_params(self, **params: float) -> 'RegressionBase':
        """Lock one or more parameters to fixed values.

        Locked parameters will not be optimized during fitting.

        Args:
            **params: Parameter names and their locked values

        Returns:
            Self for method chaining

        Example:
            >>> reg = LinearRegression()
            >>> reg.lock_params(slope=2.0)
            >>> reg.fit(x, y)  # Will fit intercept only, slope fixed at 2.0
        """
        self._locked_params.update(params)
        return self

    def unlock_params(self, *param_names: str) -> 'RegressionBase':
        """Unlock one or more parameters.

        Args:
            *param_names: Names of parameters to unlock. If none provided, unlocks all.

        Returns:
            Self for method chaining

        Example:
            >>> reg.unlock_params('slope')  # Unlock specific parameter
            >>> reg.unlock_params()  # Unlock all parameters
        """
        if not param_names:
            self._locked_params.clear()
        else:
            for name in param_names:
                self._locked_params.pop(name, None)
        return self

    def get_locked_params(self) -> Dict[str, float]:
        """Get currently locked parameters.

        Returns:
            Dictionary of locked parameter names and values
        """
        return self._locked_params.copy()

    def is_param_locked(self, param_name: str) -> bool:
        """Check if a parameter is locked.

        Args:
            param_name: Name of the parameter to check

        Returns:
            True if parameter is locked, False otherwise
        """
        return param_name in self._locked_params

    def get_plot_data(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get x and y data for plotting the regression line.

        Args:
            x_range: Optional tuple of (x_min, x_max) for the plot range.
                    If None, uses the stored x_range from fitting.
                    If stored x_range is also None, raises an error.
            num_points: Number of points to generate for the line. Default: 100

        Returns:
            Tuple of (x_values, y_values) for plotting

        Raises:
            ValueError: If model is not fitted or x_range cannot be determined
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating plot data")

        # Determine x range
        if x_range is not None:
            x_min, x_max = x_range
        elif self.x_range is not None:
            x_min, x_max = self.x_range
        else:
            raise ValueError("x_range not available. Provide x_range parameter.")

        # Generate x values
        x_line = np.linspace(x_min, x_max, num_points)

        # Predict y values
        y_line = self.predict(x_line)

        return x_line, y_line


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

        # Lock slope to force through origin with specific slope
        >>> reg = LinearRegression(locked_params={'slope': 2.0})
        >>> reg.fit(x, y)  # Only fits intercept
    """

    def __init__(self, locked_params: Optional[Dict[str, float]] = None):
        """Initialize linear regression.

        Args:
            locked_params: Dictionary to lock parameters. Valid keys: 'slope', 'intercept'
        """
        super().__init__(locked_params)
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

        # Check if parameters are locked
        slope_locked = self.is_param_locked('slope')
        intercept_locked = self.is_param_locked('intercept')

        if slope_locked and intercept_locked:
            # Both locked - just use the locked values
            self.slope = self._locked_params['slope']
            self.intercept = self._locked_params['intercept']
        elif slope_locked:
            # Fit intercept only: y - slope*x = intercept
            self.slope = self._locked_params['slope']
            self.intercept = np.mean(y_clean - self.slope * x_clean)
        elif intercept_locked:
            # Fit slope only: (y - intercept) / x = slope
            self.intercept = self._locked_params['intercept']
            self.slope = np.sum((y_clean - self.intercept) * x_clean) / np.sum(x_clean ** 2)
        else:
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

        # Lock the coefficient
        >>> reg = LogarithmicRegression(locked_params={'a': 1.5})
        >>> reg.fit(x, y)  # Only fits b
    """

    def __init__(self, locked_params: Optional[Dict[str, float]] = None):
        """Initialize logarithmic regression.

        Args:
            locked_params: Dictionary to lock parameters. Valid keys: 'a', 'b'
        """
        super().__init__(locked_params)
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

        # Check if parameters are locked
        a_locked = self.is_param_locked('a')
        b_locked = self.is_param_locked('b')

        if a_locked and b_locked:
            # Both locked - just use the locked values
            self.a = self._locked_params['a']
            self.b = self._locked_params['b']
        elif a_locked:
            # Fit b only: y - a*ln(x) = b
            self.a = self._locked_params['a']
            self.b = np.mean(y_clean - self.a * ln_x)
        elif b_locked:
            # Fit a only: (y - b) / ln(x) = a
            self.b = self._locked_params['b']
            self.a = np.sum((y_clean - self.b) * ln_x) / np.sum(ln_x ** 2)
        else:
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

        # Lock the base value
        >>> reg = ExponentialRegression(locked_params={'a': 1.0})
        >>> reg.fit(x, y)  # Only fits b
    """

    def __init__(self, locked_params: Optional[Dict[str, float]] = None):
        """Initialize exponential regression.

        Args:
            locked_params: Dictionary to lock parameters. Valid keys: 'a', 'b'
        """
        super().__init__(locked_params)
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

        # Check if parameters are locked
        a_locked = self.is_param_locked('a')
        b_locked = self.is_param_locked('b')

        if a_locked and b_locked:
            # Both locked - just use the locked values
            self.a = self._locked_params['a']
            self.b = self._locked_params['b']
        elif a_locked:
            # Fit b only: ln(y/a) = b*x
            self.a = self._locked_params['a']
            ln_y_ratio = np.log(y_clean / self.a)
            self.b = np.sum(ln_y_ratio * x_clean) / np.sum(x_clean ** 2)
        elif b_locked:
            # Fit a only: ln(y) = ln(a) + b*x => a = exp(mean(ln(y) - b*x))
            self.b = self._locked_params['b']
            self.a = np.exp(np.mean(np.log(y_clean) - self.b * x_clean))
        else:
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

        # Lock specific coefficients (indexed from highest to lowest degree)
        >>> reg = PolynomialRegression(degree=2, locked_params={'c0': 1.0})  # Lock x² coefficient
        >>> reg.fit(x, y)  # Only fits c1 and c2
    """

    def __init__(self, degree: int = 2, locked_params: Optional[Dict[str, float]] = None):
        """Initialize polynomial regression.

        Args:
            degree: Polynomial degree (default: 2 for quadratic)
            locked_params: Dictionary to lock coefficients. Keys: 'c0', 'c1', ..., 'c{degree}'
                          where c0 is the coefficient for x^degree (highest power)
        """
        super().__init__(locked_params)
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

        # Check for locked coefficients
        locked_indices = {int(k[1:]): v for k, v in self._locked_params.items() if k.startswith('c')}

        if not locked_indices:
            # No locked coefficients - use standard polyfit
            self.coefficients = np.polyfit(x_clean, y_clean, self.degree)
        else:
            # Initialize coefficients array
            self.coefficients = np.zeros(self.degree + 1)

            # Set locked coefficients
            for idx, val in locked_indices.items():
                if idx < 0 or idx > self.degree:
                    raise ValueError(f"Coefficient index c{idx} out of range for degree {self.degree}")
                self.coefficients[idx] = val

            # Create design matrix for unlocked coefficients
            # Subtract contribution of locked coefficients from y
            y_adjusted = y_clean.copy()
            for idx, val in locked_indices.items():
                power = self.degree - idx
                y_adjusted -= val * (x_clean ** power)

            # Fit only unlocked coefficients
            unlocked_indices = [i for i in range(self.degree + 1) if i not in locked_indices]

            if unlocked_indices:
                # Build design matrix for unlocked terms
                X = np.column_stack([x_clean ** (self.degree - i) for i in unlocked_indices])

                # Solve least squares
                coefs_unlocked = np.linalg.lstsq(X, y_adjusted, rcond=None)[0]

                # Assign unlocked coefficients
                for i, idx in enumerate(unlocked_indices):
                    self.coefficients[idx] = coefs_unlocked[i]

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

        # Lock the exponent to fit a scaled relationship
        >>> reg = PowerRegression(locked_params={'b': 2.0})
        >>> reg.fit(x, y)  # Only fits a
    """

    def __init__(self, locked_params: Optional[Dict[str, float]] = None):
        """Initialize power regression.

        Args:
            locked_params: Dictionary to lock parameters. Valid keys: 'a', 'b'
        """
        super().__init__(locked_params)
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

        # Check if parameters are locked
        a_locked = self.is_param_locked('a')
        b_locked = self.is_param_locked('b')

        if a_locked and b_locked:
            # Both locked - just use the locked values
            self.a = self._locked_params['a']
            self.b = self._locked_params['b']
        elif a_locked:
            # Fit b only: ln(y/a) = b*ln(x)
            self.a = self._locked_params['a']
            ln_x = np.log(x_clean)
            ln_y_ratio = np.log(y_clean / self.a)
            self.b = np.sum(ln_y_ratio * ln_x) / np.sum(ln_x ** 2)
        elif b_locked:
            # Fit a only: ln(y) = ln(a) + b*ln(x) => a = exp(mean(ln(y) - b*ln(x)))
            self.b = self._locked_params['b']
            ln_x = np.log(x_clean)
            ln_y = np.log(y_clean)
            self.a = np.exp(np.mean(ln_y - self.b * ln_x))
        else:
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
