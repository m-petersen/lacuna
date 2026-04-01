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

from dataclasses import dataclass

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

    >>> build_result_key("tian2020parcels16", "SubjectData", "maskimg")
    'atlas-tian2020parcels16_source-InputMask'

    >>> build_result_key("tian2020parcels16", "SubjectData")
    'atlas-tian2020parcels16_source-InputMask'

    >>> build_result_key("schaefer2018parcels200networks7", "RegionalDamage", "damagescore")
    'atlas-schaefer2018parcels200networks7_source-RegionalDamage_desc-damagescore'
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
    # RegionalDamage outputs
    "damagebin": "regionaldamage",
    "damage_bin": "regionaldamage",
    "damagepct": "regionaldamage",
    "damage_pct": "regionaldamage",
    # FNM outputs - correlation and derived maps
    "rmap": "fnm",
    "correlationmap": "fnm",  # Legacy alias
    "correlation_map": "fnm",  # Legacy alias
    "zmap": "fnm",
    "z_map": "fnm",
    "tmap": "fnm",
    "t_map": "fnm",
    # FNM outputs - p-value maps
    "pmap": "fnm",
    "p_map": "fnm",
    "pfdrmap": "fnm",
    "pfdr_map": "fnm",
    # FNM outputs - threshold maps
    "tthresholdmap": "fnm",
    "t_threshold_map": "fnm",
    "pfdrthresholdmap": "fnm",
    "pfdr_threshold_map": "fnm",
    # SNM outputs - maps
    "disconnectionpct": "snm",
    "disconnection_pct": "snm",
    "disconnectiontdi": "snm",
    "disconnection_tdi": "snm",
    "streamlinecount": "snm",
    "streamline_count": "snm",
    # SNM outputs - connectivity matrices
    "maskconnectivitymatrix": "snm",
    "mask_connectivity_matrix": "snm",
    "intactconnectivitymatrix": "snm",
    "intact_connectivity_matrix": "snm",
    "fullconnectivitymatrix": "snm",
    "full_connectivity_matrix": "snm",
    "disconnectivitypercent": "snm",
    "disconnectivity_percent": "snm",
    # SNM outputs - parcel data
    "roidisconnection": "snm",
    "roi_disconnection": "snm",
    "roidisconnectionpct": "snm",
    # SNM outputs - metrics
    "matrixstatistics": "snm",
    "matrix_statistics": "snm",
}

# BIDS suffix mapping - map internal suffixes to BIDS-compliant suffixes
BIDS_SUFFIX_MAPPING = {
    "values": "parcelstats",  # Tabular parcel statistics
    "parcels": "parcelstats",  # Tabular parcel data
    "map": "",  # VoxelMap NIfTI - no suffix needed (e.g., fnmrmap.nii.gz)
    "connmatrix": "connmatrix",  # Connectivity matrix (valid BIDS derivative)
    "metrics": "stats",  # Scalar metrics as tabular
}

# Method abbreviation mapping: analysis class name -> short BIDS method entity
METHOD_ABBREVIATIONS: dict[str, str] = {
    "FunctionalNetworkMapping": "fnm",
    "StructuralNetworkMapping": "snm",
    "RegionalDamage": "rd",
    "ParcelAggregation": "pa",
}

# Desc overrides: internal desc -> canonical BIDS desc
# Used to homogenize variant names for the same concept
BIDS_DESC_OVERRIDE: dict[str, str] = {
    "disconnectivity_percent": "disconnectionpct",
    "disconnectivitypercent": "disconnectionpct",
    "disconnectivitypct": "disconnectionpct",
    "disconnection_pct": "disconnectionpct",
    "roi_disconnection": "disconnectionpct",
    "roidisconnection": "disconnectionpct",
    "roidisconnectionpct": "disconnectionpct",
    "damage_pct": "damagepct",
    "damage_bin": "damagebin",
}


@dataclass
class BidsFilename:
    """Structured representation of a BIDS-compliant filename.

    Entity ordering: method > space > atlas > desc > suffix.
    None fields are omitted from the string representation.
    """

    method: str | None = None
    space: str | None = None
    atlas: str | None = None
    desc: str | None = None
    suffix: str | None = None

    def __str__(self) -> str:
        parts: list[str] = []
        if self.method:
            parts.append(f"method-{self.method}")
        if self.space:
            parts.append(f"space-{self.space}")
        if self.atlas:
            parts.append(f"atlas-{self.atlas}")
        if self.desc:
            parts.append(f"desc-{self.desc}")
        # Suffix is appended as-is (not a key-value pair)
        if self.suffix:
            parts.append(self.suffix)
        return "_".join(parts)

    @classmethod
    def from_result_key(
        cls,
        result_key: str,
        suffix: str,
        namespace: str | None = None,
    ) -> BidsFilename:
        """Build a BidsFilename from an internal result key.

        Parameters
        ----------
        result_key : str
            Internal result key — either a simple name (e.g. ``"rmap"``)
            or a BIDS-style key (``atlas-X_source-Y_desc-Z``).
        suffix : str
            Internal suffix (``"map"``, ``"values"``, ``"connmatrix"``,
            ``"metrics"``).  Converted via ``BIDS_SUFFIX_MAPPING``.
        namespace : str, optional
            Analysis class name that produced the result (e.g.
            ``"FunctionalNetworkMapping"``).  Used to derive *method*
            when the key has no ``source-`` entity.
        """
        bids_suffix = BIDS_SUFFIX_MAPPING.get(suffix, suffix)

        has_bids_prefix = any(
            prefix in result_key for prefix in ("atlas-", "parc-", "source-", "desc-")
        )

        if not has_bids_prefix:
            # Simple key — derive method from DESC_TO_SOURCE_MAPPING or namespace
            bids_desc = to_bids_label(result_key)
            bids_desc = BIDS_DESC_OVERRIDE.get(result_key, BIDS_DESC_OVERRIDE.get(bids_desc, bids_desc))

            method = None
            source_prefix = DESC_TO_SOURCE_MAPPING.get(result_key, "")
            if not source_prefix:
                source_prefix = DESC_TO_SOURCE_MAPPING.get(bids_desc, "")
            if source_prefix in ("fnm", "snm"):
                method = source_prefix
            elif namespace:
                method = METHOD_ABBREVIATIONS.get(namespace)

            return cls(
                method=method,
                desc=bids_desc if bids_desc else None,
                suffix=bids_suffix,
            )

        # BIDS-style key
        parsed = parse_result_key(result_key)

        # Derive method from source
        method = None
        if "source" in parsed:
            source = parsed["source"]
            export_abbrev = EXPORT_SOURCE_ABBREVIATIONS.get(source, source.lower())
            if export_abbrev in ("fnm", "snm"):
                method = export_abbrev
            elif export_abbrev == "regionaldamage":
                method = "rd"
            elif export_abbrev == "parcelaggregation":
                method = "pa"
            elif export_abbrev == "inputmask":
                method = None
        elif namespace:
            method = METHOD_ABBREVIATIONS.get(namespace)

        # Atlas names are already BIDS slugs (registry uses slug keys)
        atlas_slug = None
        if "atlas" in parsed:
            atlas_slug = parsed["atlas"]

        # Resolve desc with override
        desc = None
        if "desc" in parsed:
            raw_desc = parsed["desc"]
            bids_desc = to_bids_label(raw_desc)
            desc = BIDS_DESC_OVERRIDE.get(raw_desc, BIDS_DESC_OVERRIDE.get(bids_desc, bids_desc))

        return cls(
            method=method,
            atlas=atlas_slug,
            desc=desc,
            suffix=bids_suffix,
        )


def to_bids_label(value: str) -> str:
    """
    Convert a value to a BIDS-compliant label (single component).

    BIDS labels cannot contain underscores (reserved for key-value separation).
    This function converts to lowercase and removes underscores for simple values.

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

    >>> to_bids_label("Schaefer2018")
    'schaefer2018'

    >>> to_bids_label("mask_img")
    'maskimg'
    """
    # Remove underscores and convert to lowercase
    return value.replace("_", "").lower()


