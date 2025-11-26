"""BIDS-style result key helpers for analysis results.

This module provides utilities for building and parsing structured result keys
that follow a BIDS-inspired key-value format:

    parc-{parcellation}_source-{source}_desc-{description}

Examples
--------
>>> from lacuna.core.keys import build_result_key, parse_result_key
>>> key = build_result_key("Schaefer100", "FunctionalNetworkMapping", "correlation_map")
>>> key
'parc-Schaefer100_source-FunctionalNetworkMapping_desc-correlation_map'
>>> parse_result_key(key)
{'parc': 'Schaefer100', 'source': 'FunctionalNetworkMapping', 'desc': 'correlation_map'}
"""

from __future__ import annotations

# Mapping from analysis class names to short abbreviations for result keys
SOURCE_ABBREVIATIONS: dict[str, str] = {
    "MaskData": "MaskData",
    "FunctionalNetworkMapping": "FunctionalNetworkMapping",
    "StructuralNetworkMapping": "StructuralNetworkMapping",
    "RegionalDamage": "RegionalDamage",
    "ParcelAggregation": "ParcelAggregation",
}


def build_result_key(parc: str, source: str, desc: str) -> str:
    """
    Build a BIDS-style result key from components.

    Creates a structured key string in the format:
    ``parc-{parc}_source-{source}_desc-{desc}``

    Parameters
    ----------
    parc : str
        Parcellation/atlas name (e.g., "Schaefer100", "Tian_S4").
    source : str
        Source analysis class name (e.g., "MaskData", "FunctionalNetworkMapping").
    desc : str
        Description/key within the source (e.g., "correlation_map", "mask_img").

    Returns
    -------
    str
        BIDS-style result key.

    Examples
    --------
    >>> build_result_key("Schaefer100", "FunctionalNetworkMapping", "correlation_map")
    'parc-Schaefer100_source-FunctionalNetworkMapping_desc-correlation_map'

    >>> build_result_key("Tian_S4", "MaskData", "mask_img")
    'parc-Tian_S4_source-MaskData_desc-mask_img'

    >>> build_result_key("AAL", "RegionalDamage", "damage_score")
    'parc-AAL_source-RegionalDamage_desc-damage_score'
    """
    return f"parc-{parc}_source-{source}_desc-{desc}"


def parse_result_key(key: str) -> dict[str, str]:
    """
    Parse a BIDS-style result key into its components.

    Extracts key-value pairs from a structured key string in the format:
    ``parc-{parc}_source-{source}_desc-{desc}``

    Parameters
    ----------
    key : str
        BIDS-style result key to parse.

    Returns
    -------
    dict[str, str]
        Dictionary with parsed components. Keys are "parc", "source", "desc".
        Missing components will not be present in the returned dict.

    Raises
    ------
    ValueError
        If key is empty or has invalid format.

    Examples
    --------
    >>> parse_result_key("parc-Schaefer100_source-FunctionalNetworkMapping_desc-correlation_map")
    {'parc': 'Schaefer100', 'source': 'FunctionalNetworkMapping', 'desc': 'correlation_map'}

    >>> parse_result_key("parc-Tian_S4_source-MaskData_desc-mask_img")
    {'parc': 'Tian_S4', 'source': 'MaskData', 'desc': 'mask_img'}
    """
    if not key:
        raise ValueError("Result key cannot be empty")

    parts: dict[str, str] = {}

    # Split by underscore-prefixed keys (parc-, source-, desc-)
    # Handle values that contain underscores by only splitting on known prefixes
    segments = key.split("_")
    current_key: str | None = None
    current_value_parts: list[str] = []

    for segment in segments:
        # Check if this starts a new key
        if segment.startswith("parc-"):
            if current_key is not None:
                parts[current_key] = "_".join(current_value_parts)
            current_key = "parc"
            current_value_parts = [segment[5:]]  # Remove "parc-" prefix
        elif segment.startswith("source-"):
            if current_key is not None:
                parts[current_key] = "_".join(current_value_parts)
            current_key = "source"
            current_value_parts = [segment[7:]]  # Remove "source-" prefix
        elif segment.startswith("desc-"):
            if current_key is not None:
                parts[current_key] = "_".join(current_value_parts)
            current_key = "desc"
            current_value_parts = [segment[5:]]  # Remove "desc-" prefix
        else:
            # This is a continuation of the current value (contains underscore)
            if current_key is not None:
                current_value_parts.append(segment)
            else:
                raise ValueError(
                    f"Invalid result key format: '{key}'. "
                    "Expected format: parc-{parc}_source-{source}_desc-{desc}"
                )

    # Don't forget the last key-value pair
    if current_key is not None:
        parts[current_key] = "_".join(current_value_parts)

    return parts


def get_source_abbreviation(class_name: str) -> str:
    """
    Validate and return the source name for an analysis class.

    This function validates that the class name is a known analysis type
    and returns it unchanged (source names use PascalCase class names).

    Parameters
    ----------
    class_name : str
        Analysis class name (e.g., "FunctionalNetworkMapping").

    Returns
    -------
    str
        The validated class name for use in result keys.

    Raises
    ------
    KeyError
        If class_name is not a known analysis class.

    Examples
    --------
    >>> get_source_abbreviation("FunctionalNetworkMapping")
    'FunctionalNetworkMapping'
    >>> get_source_abbreviation("MaskData")
    'MaskData'
    """
    if class_name not in SOURCE_ABBREVIATIONS:
        available = ", ".join(sorted(SOURCE_ABBREVIATIONS.keys()))
        raise KeyError(
            f"Unknown analysis class '{class_name}'. "
            f"Known classes: {available}"
        )
    return SOURCE_ABBREVIATIONS[class_name]
