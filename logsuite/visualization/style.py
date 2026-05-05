"""
Color and style resolution for visualization consumers.

Single source of truth for how categorical colors are picked across
``WellView``, ``Crossplot``, and other consumers. Resolution rule:

1. ``Property.colors`` is honored first (user-defined palette).
2. Codes not present in ``Property.colors`` fall back to a default
   palette in the order encountered.

This module belongs to the ``visualization`` layer. It depends on no
other higher-level visualization modules; consumers import its functions
to ensure consistent palettes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_DISCRETE_PALETTE: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def _coerce_int_code(value: Any) -> int | None:
    """Try to coerce a value to its integer property-code form.

    Property values are stored as floats even for discrete properties,
    while ``Property.colors`` keys are conventionally integer codes.
    Returns the integer if coercible, else ``None``.
    """
    try:
        return int(round(float(value)))
    except (ValueError, TypeError):
        return None


def resolve_discrete_palette(
    user_colors: dict[Any, Any] | None,
    categories: Sequence[Any],
    default_palette: Sequence[str] | None = None,
) -> dict[Any, Any]:
    """Return a category -> color mapping that honors user colors first.

    Parameters
    ----------
    user_colors : dict, optional
        ``Property.colors`` mapping (typically ``{int_code: color}``).
        ``None`` or empty falls back entirely to the default palette.
    categories : sequence
        Unique category values present in the data. Iteration order is
        preserved in the output.
    default_palette : sequence of str, optional
        Fallback palette for categories not in ``user_colors``. Defaults
        to :data:`DEFAULT_DISCRETE_PALETTE`.

    Returns
    -------
    dict
        Mapping from each input category to a color. Categories present
        in ``user_colors`` (directly or via integer-code coercion) get
        the user-supplied color; the rest are assigned colors from
        ``default_palette`` in the order encountered (cycling if needed).

    Examples
    --------
    >>> prop_colors = {1: "#3b82f6", 2: "#ef4444"}  # Reservoir, NonReservoir
    >>> resolve_discrete_palette(prop_colors, [1.0, 2.0, 3.0])
    {1.0: '#3b82f6', 2.0: '#ef4444', 3.0: 'tab:blue'}
    """
    palette = tuple(default_palette) if default_palette is not None else DEFAULT_DISCRETE_PALETTE
    user_colors = user_colors or {}

    out: dict[Any, Any] = {}
    fallback_idx = 0
    for cat in categories:
        # Skip NaN
        try:
            if isinstance(cat, float) and np.isnan(cat):
                continue
        except (TypeError, ValueError):
            pass

        # Direct lookup
        if cat in user_colors:
            out[cat] = user_colors[cat]
            continue
        # Integer-code coercion
        int_code = _coerce_int_code(cat)
        if int_code is not None and int_code in user_colors:
            out[cat] = user_colors[int_code]
            continue
        # Fallback
        out[cat] = palette[fallback_idx % len(palette)]
        fallback_idx += 1
    return out
