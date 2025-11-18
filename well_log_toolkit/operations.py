"""
Property computation operations with strict depth matching.

This module provides operator overloading and computational operations for
Property objects, enabling intuitive mathematical expressions like:

    well.New_Prop = well.PHIE * 0.01
    well.Reservoir = well.PHIE > 0.15
    well.HC_Volume = well.PHIE * (1 - well.SW)

IMPORTANT: Operations enforce strict depth matching (like numpy). Properties
with different depth grids must be explicitly resampled first:

    core_resampled = well.CorePHIE.resample(well.PHIE)
    result = well.PHIE + core_resampled
"""

import numpy as np
import warnings
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .property import Property


def align_properties(prop1: 'Property', prop2: 'Property') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check that two properties have identical depth grids and return aligned arrays.

    This function enforces strict depth matching - if the depths don't match exactly,
    a DepthAlignmentError is raised. Users must explicitly resample properties using
    the .resample() method before combining them.

    Parameters
    ----------
    prop1 : Property
        First property
    prop2 : Property
        Second property

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (depth, values1, values2) - all sharing the same depth grid

    Raises
    ------
    DepthAlignmentError
        If the depth grids don't match exactly

    Notes
    -----
    This function follows numpy's philosophy: operations fail on mismatched shapes.
    To combine properties with different depth grids, use:
        prop1_resampled = prop1.resample(prop2.depth)
        result = prop1_resampled + prop2

    Examples
    --------
    >>> # Properties with same depth - OK
    >>> result = well.PHIE + well.SW

    >>> # Properties with different depth - ERROR
    >>> result = well.PHIE + well.CorePHIE  # Raises DepthAlignmentError

    >>> # Fix by explicit resampling
    >>> core_resampled = well.CorePHIE.resample(well.PHIE.depth)
    >>> result = well.PHIE + core_resampled
    """
    from .exceptions import DepthAlignmentError

    # Check if grids are identical (within tolerance)
    if len(prop1.depth) == len(prop2.depth) and np.allclose(prop1.depth, prop2.depth, rtol=1e-9, atol=1e-9):
        return prop1.depth.copy(), prop1.values.copy(), prop2.values.copy()

    # Grids don't match - raise error
    spacing1 = np.median(np.diff(prop1.depth)) if len(prop1.depth) > 1 else 0.0
    spacing2 = np.median(np.diff(prop2.depth)) if len(prop2.depth) > 1 else 0.0

    raise DepthAlignmentError(
        f"Cannot combine properties with different depth grids.\n"
        f"  {prop1.name}: {len(prop1.depth)} samples "
        f"({prop1.depth[0]:.2f}-{prop1.depth[-1]:.2f}m, {spacing1:.2f}m spacing)\n"
        f"  {prop2.name}: {len(prop2.depth)} samples "
        f"({prop2.depth[0]:.2f}-{prop2.depth[-1]:.2f}m, {spacing2:.2f}m spacing)\n"
        f"Use .resample() to explicitly align depths:\n"
        f"  {prop1.name}_resampled = {prop1.name}.resample({prop2.name}.depth)\n"
        f"  result = {prop1.name}_resampled + {prop2.name}"
    )


class PropertyOperationsMixin:
    """
    Mixin class providing arithmetic and comparison operations for Property objects.

    This mixin enables natural mathematical expressions on properties:
    - Arithmetic: +, -, *, /, **, //, %
    - Comparison: <, <=, >, >=, ==, !=
    - Unary: -, +, abs()
    """

    def _create_result_property(
        self,
        name: str,
        depth: np.ndarray,
        values: np.ndarray,
        unit: str = '',
        prop_type: str = 'continuous',
        labels: dict = None
    ) -> 'Property':
        """Create a new Property object from operation result."""
        from .property import Property

        # For discrete properties without explicit labels, use default True/False
        if prop_type == 'discrete' and labels is None:
            labels = {0: 'False', 1: 'True'}

        return Property(
            name=name,
            depth=depth,
            values=values,
            parent_well=getattr(self, 'parent_well', None),
            unit=unit,
            prop_type=prop_type,
            description=f'Computed from {self.name}',
            labels=labels,
            source_name='computed'
        )

    # Arithmetic operations
    def __add__(self, other: Union['Property', float, int]) -> 'Property':
        """Add scalar or property: self + other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_plus_{other}',
                depth=self.depth.copy(),
                values=self.values + other,
                unit=self.unit
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            return self._create_result_property(
                name=f'{self.name}_plus_{other.name}',
                depth=depth,
                values=vals1 + vals2,
                unit=self.unit
            )
        return NotImplemented

    def __radd__(self, other: Union[float, int]) -> 'Property':
        """Reverse add: other + self"""
        return self.__add__(other)

    def __sub__(self, other: Union['Property', float, int]) -> 'Property':
        """Subtract scalar or property: self - other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_minus_{other}',
                depth=self.depth.copy(),
                values=self.values - other,
                unit=self.unit
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            return self._create_result_property(
                name=f'{self.name}_minus_{other.name}',
                depth=depth,
                values=vals1 - vals2,
                unit=self.unit
            )
        return NotImplemented

    def __rsub__(self, other: Union[float, int]) -> 'Property':
        """Reverse subtract: other - self"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{other}_minus_{self.name}',
                depth=self.depth.copy(),
                values=other - self.values,
                unit=self.unit
            )
        return NotImplemented

    def __mul__(self, other: Union['Property', float, int]) -> 'Property':
        """Multiply by scalar or property: self * other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_times_{other}',
                depth=self.depth.copy(),
                values=self.values * other,
                unit=self.unit
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            return self._create_result_property(
                name=f'{self.name}_times_{other.name}',
                depth=depth,
                values=vals1 * vals2,
                unit=f'{self.unit}*{other.unit}' if self.unit and other.unit else ''
            )
        return NotImplemented

    def __rmul__(self, other: Union[float, int]) -> 'Property':
        """Reverse multiply: other * self"""
        return self.__mul__(other)

    def __truediv__(self, other: Union['Property', float, int]) -> 'Property':
        """Divide by scalar or property: self / other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_div_{other}',
                depth=self.depth.copy(),
                values=self.values / other,
                unit=self.unit
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            return self._create_result_property(
                name=f'{self.name}_div_{other.name}',
                depth=depth,
                values=vals1 / vals2,
                unit=f'{self.unit}/{other.unit}' if self.unit and other.unit else ''
            )
        return NotImplemented

    def __rtruediv__(self, other: Union[float, int]) -> 'Property':
        """Reverse divide: other / self"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{other}_div_{self.name}',
                depth=self.depth.copy(),
                values=other / self.values,
                unit=f'1/{self.unit}' if self.unit else ''
            )
        return NotImplemented

    def __pow__(self, other: Union[float, int]) -> 'Property':
        """Power: self ** other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_pow_{other}',
                depth=self.depth.copy(),
                values=self.values ** other,
                unit=f'{self.unit}^{other}' if self.unit else ''
            )
        return NotImplemented

    def __floordiv__(self, other: Union['Property', float, int]) -> 'Property':
        """Floor divide: self // other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_floordiv_{other}',
                depth=self.depth.copy(),
                values=self.values // other,
                unit=self.unit
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            return self._create_result_property(
                name=f'{self.name}_floordiv_{other.name}',
                depth=depth,
                values=vals1 // vals2,
                unit=f'{self.unit}/{other.unit}' if self.unit and other.unit else ''
            )
        return NotImplemented

    def __mod__(self, other: Union['Property', float, int]) -> 'Property':
        """Modulo: self % other"""
        if np.isscalar(other):
            return self._create_result_property(
                name=f'{self.name}_mod_{other}',
                depth=self.depth.copy(),
                values=self.values % other,
                unit=self.unit
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            return self._create_result_property(
                name=f'{self.name}_mod_{other.name}',
                depth=depth,
                values=vals1 % vals2,
                unit=self.unit
            )
        return NotImplemented

    # Unary operations
    def __neg__(self) -> 'Property':
        """Negate: -self"""
        return self._create_result_property(
            name=f'neg_{self.name}',
            depth=self.depth.copy(),
            values=-self.values,
            unit=self.unit
        )

    def __pos__(self) -> 'Property':
        """Positive: +self"""
        return self._create_result_property(
            name=f'pos_{self.name}',
            depth=self.depth.copy(),
            values=+self.values,
            unit=self.unit
        )

    def __abs__(self) -> 'Property':
        """Absolute value: abs(self)"""
        return self._create_result_property(
            name=f'abs_{self.name}',
            depth=self.depth.copy(),
            values=np.abs(self.values),
            unit=self.unit
        )

    # Comparison operations (return discrete binary properties)
    def __lt__(self, other: Union['Property', float, int]) -> 'Property':
        """Less than: self < other"""
        if np.isscalar(other):
            values = (self.values < other).astype(float)
            return self._create_result_property(
                name=f'{self.name}_lt_{other}',
                depth=self.depth.copy(),
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            values = (vals1 < vals2).astype(float)
            return self._create_result_property(
                name=f'{self.name}_lt_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __le__(self, other: Union['Property', float, int]) -> 'Property':
        """Less than or equal: self <= other"""
        if np.isscalar(other):
            values = (self.values <= other).astype(float)
            return self._create_result_property(
                name=f'{self.name}_le_{other}',
                depth=self.depth.copy(),
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            values = (vals1 <= vals2).astype(float)
            return self._create_result_property(
                name=f'{self.name}_le_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __gt__(self, other: Union['Property', float, int]) -> 'Property':
        """Greater than: self > other"""
        if np.isscalar(other):
            values = (self.values > other).astype(float)
            return self._create_result_property(
                name=f'{self.name}_gt_{other}',
                depth=self.depth.copy(),
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            values = (vals1 > vals2).astype(float)
            return self._create_result_property(
                name=f'{self.name}_gt_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __ge__(self, other: Union['Property', float, int]) -> 'Property':
        """Greater than or equal: self >= other"""
        if np.isscalar(other):
            values = (self.values >= other).astype(float)
            return self._create_result_property(
                name=f'{self.name}_ge_{other}',
                depth=self.depth.copy(),
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            values = (vals1 >= vals2).astype(float)
            return self._create_result_property(
                name=f'{self.name}_ge_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __eq__(self, other: Union['Property', float, int]) -> 'Property':
        """Equal: self == other"""
        if np.isscalar(other):
            values = (self.values == other).astype(float)
            return self._create_result_property(
                name=f'{self.name}_eq_{other}',
                depth=self.depth.copy(),
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            values = (vals1 == vals2).astype(float)
            return self._create_result_property(
                name=f'{self.name}_eq_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __ne__(self, other: Union['Property', float, int]) -> 'Property':
        """Not equal: self != other"""
        if np.isscalar(other):
            values = (self.values != other).astype(float)
            return self._create_result_property(
                name=f'{self.name}_ne_{other}',
                depth=self.depth.copy(),
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        elif isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            values = (vals1 != vals2).astype(float)
            return self._create_result_property(
                name=f'{self.name}_ne_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    # Logical operations for combining boolean properties
    def __and__(self, other: 'Property') -> 'Property':
        """Logical AND: self & other"""
        if isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            # Treat non-zero as True
            values = ((vals1 != 0) & (vals2 != 0)).astype(float)
            return self._create_result_property(
                name=f'{self.name}_and_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __or__(self, other: 'Property') -> 'Property':
        """Logical OR: self | other"""
        if isinstance(other, PropertyOperationsMixin):
            depth, vals1, vals2 = align_properties(self, other)
            # Treat non-zero as True
            values = ((vals1 != 0) | (vals2 != 0)).astype(float)
            return self._create_result_property(
                name=f'{self.name}_or_{other.name}',
                depth=depth,
                values=values,
                unit='flag',
                prop_type='discrete'
            )
        return NotImplemented

    def __invert__(self) -> 'Property':
        """Logical NOT: ~self"""
        # Treat non-zero as True, flip it
        values = (self.values == 0).astype(float)
        return self._create_result_property(
            name=f'not_{self.name}',
            depth=self.depth.copy(),
            values=values,
            unit='flag',
            prop_type='discrete'
        )
