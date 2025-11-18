"""
Utility functions for well log toolkit.
"""
import re
from typing import Optional, List, Iterable


def sanitize_well_name(name: str, keep_hyphens: bool = False) -> str:
    """
    Convert well name to valid Python identifier.

    Does NOT add 'well_' prefix - that should be added explicitly where needed
    (e.g., for folder names or dictionary keys for attribute access).

    Parameters
    ----------
    name : str
        Original well name (e.g., "36/7-5 A")
    keep_hyphens : bool, default False
        If True, keeps hyphens in the result (useful for filenames).
        If False, replaces hyphens with underscores (for Python identifiers).

    Returns
    -------
    str
        Sanitized name (e.g., "36_7_5_A" or "36_7-5_A" with keep_hyphens=True)

    Examples
    --------
    >>> sanitize_well_name("36/7-5 A")
    '36_7_5_A'
    >>> sanitize_well_name("36/7-5 A", keep_hyphens=True)
    '36_7-5_A'
    >>> sanitize_well_name("Well-A")
    'Well_A'
    >>> sanitize_well_name("Well-A", keep_hyphens=True)
    'Well-A'
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Well name must be a non-empty string, got: {name}")

    if keep_hyphens:
        # Replace invalid characters except hyphens with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    else:
        # Replace all invalid characters with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


def sanitize_property_name(name: str) -> str:
    """
    Convert property name to valid Python attribute.

    Parameters
    ----------
    name : str
        Original property name (e.g., "Zoneloglinkedto'CerisaTops'")

    Returns
    -------
    str
        Sanitized name usable as Python attribute (e.g., "Zoneloglinkedto_CerisaTops_")

    Examples
    --------
    >>> sanitize_property_name("Zoneloglinkedto'CerisaTops'")
    'Zoneloglinkedto_CerisaTops_'
    >>> sanitize_property_name("PHIE-2025")
    'PHIE_2025'
    >>> sanitize_property_name("SW (v/v)")
    'SW__v_v_'
    >>> sanitize_property_name("2025_PERM")
    'prop_2025_PERM'
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Property name must be a non-empty string, got: {name}")

    # Replace invalid characters with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Ensure it doesn't start with a number (add 'prop_' prefix if needed)
    if sanitized and sanitized[0].isdigit():
        sanitized = 'prop_' + sanitized

    # Remove trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


def parse_las_line(line: str) -> tuple[str, str, str]:
    """
    Parse a LAS header line into mnemonic, unit/value, and description.

    Parameters
    ----------
    line : str
        LAS header line (e.g., "DEPT .m : DEPTH")

    Returns
    -------
    tuple[str, str, str]
        (mnemonic, unit_or_value, description)

    Examples
    --------
    >>> parse_las_line("DEPT .m : DEPTH")
    ('DEPT', 'm', 'DEPTH')
    >>> parse_las_line("WELL.  12/3-2 B   : WELL")
    ('WELL', '12/3-2 B', 'WELL')
    >>> parse_las_line("NULL .  -999.25 : NULL VALUE")
    ('NULL', '-999.25', 'NULL VALUE')
    >>> parse_las_line("PhiTLam_2025.m3/m3 : Porosity")
    ('PhiTLam_2025', 'm3/m3', 'Porosity')
    """
    # Split by colon to separate description
    parts = line.split(':', 1)
    left = parts[0].strip()
    description = parts[1].strip() if len(parts) > 1 else ''

    # Split left side by whitespace
    tokens = left.split()
    if not tokens:
        return '', '', description

    first_token = tokens[0]

    # Check if first token contains a dot (mnemonic.unit format)
    # Handle cases like "DEPT.m", "PhiTLam_2025.m3/m3", "WELL."
    if '.' in first_token:
        # Split on first dot to separate mnemonic from unit
        dot_pos = first_token.index('.')
        mnemonic = first_token[:dot_pos]
        unit_part = first_token[dot_pos + 1:]

        # Rest of tokens (if any) are additional unit/value parts
        if len(tokens) > 1:
            rest = unit_part + ' ' + ' '.join(tokens[1:]) if unit_part else ' '.join(tokens[1:])
        else:
            rest = unit_part
        rest = rest.strip()
    else:
        # No dot in first token, old behavior
        mnemonic = first_token
        rest = left[len(first_token):].strip()

        if rest.startswith('.'):
            rest = rest[1:].strip()  # Remove leading dot

    return mnemonic, rest, description


def filter_names(
    all_names: Iterable[str],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> Optional[List[str]]:
    """
    Filter a list of names based on include/exclude parameters.

    If both include and exclude are specified, exclude overrides (removes from include list).

    Parameters
    ----------
    all_names : Iterable[str]
        All available names
    include : list[str] or str, optional
        Names to include. If None, includes all names.
        Can be a single string (automatically converted to list).
    exclude : list[str] or str, optional
        Names to exclude. If both include and exclude are specified,
        exclude overrides (removes from include list).
        Can be a single string (automatically converted to list).

    Returns
    -------
    list[str] or None
        Filtered list of names, or None if no filtering should be applied
        (i.e., include all names)

    Examples
    --------
    >>> all_names = ['A', 'B', 'C', 'D']
    >>> filter_names(all_names, include=['A', 'B', 'C'])
    ['A', 'B', 'C']
    >>> filter_names(all_names, include='A')  # Single string
    ['A']
    >>> filter_names(all_names, exclude=['D'])
    ['A', 'B', 'C']
    >>> filter_names(all_names, include=['A', 'B', 'C'], exclude=['C'])
    ['A', 'B']
    >>> filter_names(all_names)  # No filtering
    None
    """
    # Convert strings to lists for convenience
    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]

    # Determine which names to include/exclude
    # If both include and exclude are specified, exclude overrides
    if include is not None:
        # Start with include list, then remove excluded
        if exclude is not None:
            return [name for name in include if name not in exclude]
        else:
            return list(include)
    elif exclude is not None:
        # No include list, just exclude from all names
        return [name for name in all_names if name not in exclude]
    else:
        # No filtering
        return None