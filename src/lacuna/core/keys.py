"""BIDS-style result key helpers for analysis results.

This module provides utilities for building and parsing structured result keys
that follow a BIDS-inspired key-value format:

    atlas-{atlas}_source-{source}[_desc-{description}]

The desc component is optional and only included when needed (e.g., omitted for
InputMask source since the mask itself is the primary data).

Examples
--------
>>> from lacuna.core.keys import build_result_key, parse_result_key
>>> key = build_result_key("Schaefer100", "FunctionalNetworkMapping", "rmap")
>>> key
'atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap'
>>> parse_result_key(key)
{'atlas': 'Schaefer100', 'source': 'FunctionalNetworkMapping', 'desc': 'rmap'}
"""

from __future__ import annotations

# Mapping from analysis class names to short abbreviations for result keys
SOURCE_ABBREVIATIONS: dict[str, str] = {
    "SubjectData": "InputMask",
    "InputMask": "InputMask",
    "FunctionalNetworkMapping": "FunctionalNetworkMapping",
    "StructuralNetworkMapping": "StructuralNetworkMapping",
    "RegionalDamage": "RegionalDamage",
    "ParcelAggregation": "ParcelAggregation",
}


def build_result_key(atlas: str, source: str, desc: str | None = None) -> str:
    """
    Build a BIDS-style result key from components.

    Creates a structured key string in the format:
    ``atlas-{atlas}_source-{source}[_desc-{desc}]``

    The desc component is optional and automatically omitted when source is
    InputMask/SubjectData (the mask itself is the primary data, no additional
    description needed).

    Parameters
    ----------
    atlas : str
        Atlas/parcellation name (e.g., "Schaefer100", "Tian_S4").
    source : str
        Source analysis class name (e.g., "SubjectData", "FunctionalNetworkMapping").
        Will be converted to appropriate source abbreviation (e.g., SubjectData -> InputMask).
    desc : str, optional
        Description/key within the source (e.g., "rmap").
        Ignored for InputMask source (automatically omitted).

    Returns
    -------
    str
        BIDS-style result key.

    Examples
    --------
    >>> build_result_key("Schaefer100", "FunctionalNetworkMapping", "rmap")
    'atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap'

    >>> build_result_key("TianSubcortex_3TS1", "SubjectData", "maskimg")
    'atlas-TianSubcortex_3TS1_source-InputMask'

    >>> build_result_key("TianSubcortex_3TS1", "SubjectData")
    'atlas-TianSubcortex_3TS1_source-InputMask'

    >>> build_result_key("Schaefer2018_200Parcels7Networks", "RegionalDamage", "damagescore")
    'atlas-Schaefer2018_200Parcels7Networks_source-RegionalDamage_desc-damagescore'
    """
    # Convert source class name to appropriate abbreviation
    source_abbrev = SOURCE_ABBREVIATIONS.get(source, source)

    # For InputMask source, always omit desc (the mask itself is the data)
    if source_abbrev == "InputMask":
        return f"atlas-{atlas}_source-{source_abbrev}"

    if desc:
        return f"atlas-{atlas}_source-{source_abbrev}_desc-{desc}"
    return f"atlas-{atlas}_source-{source_abbrev}"


def parse_result_key(key: str) -> dict[str, str]:
    """
    Parse a BIDS-style result key into its components.

    Extracts key-value pairs from a structured key string in the format:
    ``atlas-{atlas}_source-{source}[_desc-{desc}]``

    Parameters
    ----------
    key : str
        BIDS-style result key to parse.

    Returns
    -------
    dict[str, str]
        Dictionary with parsed components. Keys are "atlas", "source", "desc".
        Missing components will not be present in the returned dict.

    Raises
    ------
    ValueError
        If key is empty or has invalid format.

    Examples
    --------
    >>> parse_result_key("atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap")
    {'atlas': 'Schaefer100', 'source': 'FunctionalNetworkMapping', 'desc': 'rmap'}

    >>> parse_result_key("atlas-Tian_S4_source-InputMask")
    {'atlas': 'Tian_S4', 'source': 'InputMask'}
    """
    if not key:
        raise ValueError("Result key cannot be empty")

    parts: dict[str, str] = {}

    # Split by underscore-prefixed keys (atlas-, source-, desc-)
    # Handle values that contain underscores by only splitting on known prefixes
    segments = key.split("_")
    current_key: str | None = None
    current_value_parts: list[str] = []

    for segment in segments:
        # Check if this starts a new key
        if segment.startswith("atlas-"):
            if current_key is not None:
                parts[current_key] = "_".join(current_value_parts)
            current_key = "atlas"
            current_value_parts = [segment[6:]]  # Remove "atlas-" prefix
        elif segment.startswith("parc-"):
            # Legacy support: treat parc- as atlas-
            if current_key is not None:
                parts[current_key] = "_".join(current_value_parts)
            current_key = "atlas"
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
                    "Expected format: atlas-{atlas}_source-{source}[_desc-{desc}]"
                )

    # Don't forget the last key-value pair
    if current_key is not None:
        parts[current_key] = "_".join(current_value_parts)

    return parts


def get_source_abbreviation(class_name: str) -> str:
    """
    Validate and return the source name for an analysis class.

    This function validates that the class name is a known analysis type
    and returns the appropriate source abbreviation for result keys.

    Parameters
    ----------
    class_name : str
        Analysis class name (e.g., "FunctionalNetworkMapping", "SubjectData").

    Returns
    -------
    str
        The source abbreviation for use in result keys.

    Raises
    ------
    KeyError
        If class_name is not a known analysis class.

    Examples
    --------
    >>> get_source_abbreviation("FunctionalNetworkMapping")
    'FunctionalNetworkMapping'
    >>> get_source_abbreviation("SubjectData")
    'InputMask'
    """
    if class_name not in SOURCE_ABBREVIATIONS:
        available = ", ".join(sorted(SOURCE_ABBREVIATIONS.keys()))
        raise KeyError(f"Unknown analysis class '{class_name}'. " f"Known classes: {available}")
    return SOURCE_ABBREVIATIONS[class_name]


# Mapping for BIDS export filenames - use short lowercase abbreviations
EXPORT_SOURCE_ABBREVIATIONS = {
    "SubjectData": "inputmask",
    "MaskData": "inputmask",
    "InputMask": "inputmask",
    "FunctionalNetworkMapping": "fnm",
    "StructuralNetworkMapping": "snm",
    "RegionalDamage": "regionaldamage",
    "ParcelAggregation": "parcelaggregation",
}

# Mapping of desc values to their source analysis - for constructing combined desc
DESC_TO_SOURCE_MAPPING = {
    "inputmask": "inputmask",
    "maskimg": "inputmask",  # Legacy
    "mask_img": "inputmask",  # Legacy
    # FNM outputs
    "rmap": "fnm",
    "correlationmap": "fnm",  # Legacy alias
    "correlation_map": "fnm",  # Legacy alias
    "zmap": "fnm",
    "z_map": "fnm",
    "tmap": "fnm",
    "t_map": "fnm",
    # SNM outputs
    "disconnectionmap": "snm",
    "disconnection_map": "snm",
    "streamlinecount": "snm",
    "streamline_count": "snm",
}

# BIDS suffix mapping - map internal suffixes to BIDS-compliant suffixes
BIDS_SUFFIX_MAPPING = {
    "values": "parcelstats",  # Tabular parcel statistics
    "parcels": "parcelstats",  # Tabular parcel data
    "map": "",  # VoxelMap NIfTI - no suffix needed (e.g., fnmrmap.nii.gz)
    "connmatrix": "connmatrix",  # Connectivity matrix (valid BIDS derivative)
    "metrics": "stats",  # Scalar metrics as tabular
}


def to_bids_label(value: str) -> str:
    """
    Convert a value to a BIDS-compliant label (single component).

    BIDS labels cannot contain underscores (reserved for key-value separation).
    This function converts to lowercase and removes underscores for simple values.

    For parcellation names that have underscores as semantic separators
    (e.g., "Schaefer2018_100Parcels7Networks"), use ``split_atlas_name()`` instead
    to preserve the atlas/description structure.

    Parameters
    ----------
    value : str
        Value to convert to a single BIDS label component.

    Returns
    -------
    str
        BIDS-compliant lowercase label without underscores.

    Examples
    --------
    >>> to_bids_label("r_map")
    'rmap'

    >>> to_bids_label("HCP1065")
    'hcp1065'

    >>> to_bids_label("mask_img")
    'maskimg'
    """
    # Remove underscores and convert to lowercase
    return value.replace("_", "").lower()


def split_atlas_name(name: str) -> tuple[str, str | None]:
    """
    Split an atlas/parcellation name into atlas and description components.

    Many atlas names follow the pattern ``{AtlasName}_{Variant}`` where the
    underscore separates the atlas family from the specific variant. This function
    splits on the first underscore to create proper BIDS entities.

    Parameters
    ----------
    name : str
        Atlas or parcellation name (e.g., "Schaefer2018_100Parcels7Networks").

    Returns
    -------
    tuple[str, str | None]
        Tuple of (atlas_name, description) where description may be None if no
        underscore is present. Both values are lowercase for BIDS compliance.

    Examples
    --------
    >>> split_atlas_name("Schaefer2018_100Parcels7Networks")
    ('schaefer2018', '100parcels7networks')

    >>> split_atlas_name("TianSubcortex_3TS1")
    ('tiansubcortex', '3ts1')

    >>> split_atlas_name("HCP1065_thr0p1")
    ('hcp1065', 'thr0p1')

    >>> split_atlas_name("HCP1065")
    ('hcp1065', None)

    >>> split_atlas_name("Schaefer2018_1000Parcels7Networks")
    ('schaefer2018', '1000parcels7networks')
    """
    if "_" in name:
        # Split on first underscore only - rest belongs to description
        parts = name.split("_", 1)
        return (parts[0].lower(), parts[1].replace("_", "").lower())
    else:
        return (name.lower(), None)


def format_bids_export_filename(
    result_key: str,
    suffix: str,
) -> str:
    """
    Format a BIDS-compliant export filename from a result key.

    Takes a result key like ``atlas-Schaefer2018_100Parcels7Networks_source-InputMask``
    and converts it to a BIDS-compliant filename component.

    The transformation:
    1. Uses ``atlas-`` entity (BIDS standard)
    2. Splits atlas names on underscore: ``Schaefer2018_100Parcels7Networks`` becomes
       ``atlas-schaefer2018_desc-100parcels7networks``
    3. Uses proper BIDS suffixes (``stats`` for tabular data, ``stat`` for maps)
    4. For FNM/SNM VoxelMaps (no parcellation), uses format like ``fnmrmap_stat``

    Underscores in the output only separate BIDS key-value pairs.

    Parameters
    ----------
    result_key : str
        BIDS-style result key (e.g., ``atlas-Schaefer100_source-InputMask``)
        or simple key (e.g., ``rmap``).
    suffix : str
        Internal suffix for the file type (e.g., ``values``, ``map``, ``connmatrix``).
        Will be converted to BIDS-compliant suffix.

    Returns
    -------
    str
        BIDS-compliant filename component ready to be appended after subject/session.

    Examples
    --------
    >>> format_bids_export_filename(
    ...     "atlas-Schaefer2018_100Parcels7Networks_source-InputMask",
    ...     "values"
    ... )
    'atlas-schaefer2018_desc-100parcels7networks_source-inputmask_parcelstats'

    >>> format_bids_export_filename(
    ...     "atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap",
    ...     "values"
    ... )
    'atlas-schaefer100_source-fnm_parcelstats'

    >>> format_bids_export_filename("rmap", "map")
    'fnmrmap'

    >>> format_bids_export_filename(
    ...     "atlas-HCP1065_thr0p1_source-InputMask",
    ...     "values"
    ... )
    'atlas-hcp1065_desc-thr0p1_source-inputmask_parcelstats'

    For FNM/SNM outputs without parcellation (VoxelMaps), the source is
    prepended to the desc without the desc- prefix, and no suffix is added:

    >>> format_bids_export_filename("rmap", "map")
    'fnmrmap'

    >>> format_bids_export_filename("disconnection_map", "map")
    'snmdisconnectionmap'
    """
    # Convert internal suffix to BIDS suffix (may be empty for VoxelMaps)
    bids_suffix = BIDS_SUFFIX_MAPPING.get(suffix, suffix)

    # Check if this is a BIDS-style key (contains known prefixes)
    # Support both new atlas- and legacy parc- prefix
    has_bids_prefix = any(
        prefix in result_key for prefix in ("atlas-", "parc-", "source-", "desc-")
    )

    if not has_bids_prefix:
        # Simple key (e.g., "correlation_map") - determine source prefix
        bids_desc = to_bids_label(result_key)
        # Check if this desc is associated with a known source (FNM/SNM)
        source_prefix = DESC_TO_SOURCE_MAPPING.get(result_key, "")
        if not source_prefix:
            # Try without underscores
            source_prefix = DESC_TO_SOURCE_MAPPING.get(bids_desc, "")

        if source_prefix and source_prefix in ("fnm", "snm"):
            # FNM/SNM VoxelMap outputs: fnmrmap (no desc- prefix)
            if bids_suffix:
                return f"{source_prefix}{bids_desc}_{bids_suffix}"
            else:
                return f"{source_prefix}{bids_desc}"
        else:
            if bids_suffix:
                return f"desc-{bids_desc}_{bids_suffix}"
            else:
                return f"desc-{bids_desc}"

    # Parse BIDS-style key
    parsed = parse_result_key(result_key)

    parts = []

    # Add atlas- entity if present, splitting on underscore for desc
    if "atlas" in parsed:
        atlas_name, atlas_desc = split_atlas_name(parsed["atlas"])
        parts.append(f"atlas-{atlas_name}")
        if atlas_desc:
            parts.append(f"desc-{atlas_desc}")

    # Get source abbreviation
    export_source = None
    if "source" in parsed:
        source = parsed["source"]
        # Use export abbreviation if available, otherwise lowercase
        export_source = EXPORT_SOURCE_ABBREVIATIONS.get(source, source.lower())

    # Handle desc - for FNM/SNM without parcellation, prepend source to desc
    if "desc" in parsed:
        desc = parsed["desc"]
        bids_desc = to_bids_label(desc)
        desc_source = DESC_TO_SOURCE_MAPPING.get(desc)

        if "atlas" not in parsed and export_source in ("fnm", "snm"):
            # No parcellation - this is a VoxelMap output from FNM/SNM
            # Use format: fnmrmap or snmdisconnectionmap (no desc- prefix)
            parts.append(f"{export_source}{bids_desc}")
        else:
            # For parcellation results, add source first, then desc if not redundant
            if export_source and not any("source-" in p for p in parts):
                parts.append(f"source-{export_source}")
            if desc_source != export_source:
                # desc provides additional info beyond source, include it
                if not any(p.startswith(f"desc-{bids_desc}") for p in parts):
                    parts.append(f"desc-{bids_desc}")
            # else: desc is redundant with source, omit it
    elif export_source:
        # No desc but have source - add source entity for parcellation results
        parts.append(f"source-{export_source}")

    # For parcellation results with source but no desc added yet, add source entity
    if "atlas" in parsed and export_source and not any("source-" in p for p in parts):
        parts.append(f"source-{export_source}")

    # Add the BIDS suffix (only if non-empty)
    if bids_suffix:
        parts.append(bids_suffix)

    return "_".join(parts)
