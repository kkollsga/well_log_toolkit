"""
Utility functions for well log toolkit.
"""
import re


def sanitize_well_name(name: str) -> str:
    """
    Convert well name to valid Python attribute.
    
    Parameters
    ----------
    name : str
        Original well name (e.g., "12/3-2 B")

    Returns
    -------
    str
        Sanitized name usable as Python attribute (e.g., "well_12_3_2_B")

    Examples
    --------
    >>> sanitize_well_name("12/3-2 B")
    'well_12_3_2_B'
    >>> sanitize_well_name("Well-A")
    'well_Well_A'
    >>> sanitize_well_name("Test_Well_123")
    'well_Test_Well_123'
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Well name must be a non-empty string, got: {name}")
    
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = 'well_' + sanitized
    elif not sanitized.startswith('well_'):
        sanitized = 'well_' + sanitized
    
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
    """
    # Split by colon to separate description
    parts = line.split(':', 1)
    left = parts[0].strip()
    description = parts[1].strip() if len(parts) > 1 else ''
    
    # Split left side by whitespace
    tokens = left.split()
    if not tokens:
        return '', '', description
    
    mnemonic = tokens[0].rstrip('.')  # Remove trailing dot from mnemonic
    
    # Everything after mnemonic (and optional dot) is unit/value
    # Handle unit format: ".m" or ". m3/m3" or value: "12/3-2 B"
    rest = left[len(tokens[0]):].strip()
    
    if rest.startswith('.'):
        rest = rest[1:].strip()  # Remove leading dot
    
    return mnemonic, rest, description