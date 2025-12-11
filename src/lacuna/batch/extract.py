"""Batch result extraction utilities.

This module provides the unified `extract()` function for extracting analysis results
from batch processing results. The function supports filtering by analysis type,
parcellation, source, and description.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import nibabel as nib
import pandas as pd

from lacuna.core.keys import parse_result_key

if TYPE_CHECKING:
    from lacuna.batch.runner import BatchResults
    from lacuna.core.mask_data import MaskData


__all__ = ["extract"]


def _get_identifier(subject: MaskData) -> str:
    """Get the identifier of a subject.

    Args:
        subject: The subject to get the identifier of.

    Returns:
        The identifier of the subject. Uses subject_id from metadata if present,
        otherwise falls back to a string representation.
    """
    # Try to get subject_id from metadata
    if hasattr(subject, "metadata") and subject.metadata:
        subject_id = subject.metadata.get("subject_id")
        if subject_id:
            return str(subject_id)
    # Fallback to object id
    return f"subject_{id(subject)}"


def _unwrap_value(val: Any) -> Any:
    """Unwrap a value by calling get_data() if available.

    Args:
        val: The value to unwrap.

    Returns:
        The unwrapped value, or the original value if no get_data() method.
    """
    # Skip nibabel images - they're already "raw" data
    # and their get_data() is deprecated in favor of get_fdata()
    if isinstance(val, nib.Nifti1Image):
        return val

    if hasattr(val, "get_data") and callable(val.get_data):
        return val.get_data()
    return val


def extract(
    batch_results: BatchResults,
    *,
    parc: str | None = None,
    source: str | None = None,
    desc: str | None = None,
    unwrap: bool = False,
    as_dataframe: bool = False,
) -> dict[str, Any] | pd.DataFrame:
    """Extract analysis results from batch processing results.

    Unified extraction function that replaces the legacy `extract_voxelmaps()`,
    `extract_parcel_table()`, and `extract_scalars()` functions. Provides
    flexible filtering using BIDS-style key components.

    Args:
        batch_results: The batch results to extract from.
        parc: Filter by parcellation name (e.g., "AAL116", "Schaefer100").
            Matches the 'parc' component of result keys.
        source: Filter by source type (e.g., "ParcelAggregation", "RegionalDamage").
            Matches the 'source' component of result keys.
        desc: Filter by description (e.g., "parcel_means", "damage_score").
            Matches the 'desc' component of result keys.
        unwrap: If True, call `get_data()` on result objects to return raw values.
            If False (default), return wrapper objects (VoxelMap, ParcelData, etc.).
        as_dataframe: If True, return results as a pandas DataFrame with columns
            for subject identifier and all extracted result keys.

    Returns:
        If as_dataframe is False:
            Dictionary mapping subject identifiers to dicts of {result_key: value}.
        If as_dataframe is True:
            DataFrame with 'subject' column and columns for each result key.

    Examples:
        Extract atlas damage for AAL116 parcellation:
        >>> results = extract(batch_results, parc="AAL116")

        Extract all ParcelAggregation results:
        >>> results = extract(batch_results, source="ParcelAggregation")

        Extract results as DataFrame:
        >>> df = extract(batch_results, parc="Schaefer100", as_dataframe=True)

        Extract with unwrapping (raw values):
        >>> results = extract(batch_results, parc="AAL116", unwrap=True)

    Raises:
        ValueError: If batch_results is empty or no results match the filters.
    """
    if not batch_results:
        msg = "batch_results is empty"
        raise ValueError(msg)

    # Build filter criteria
    filters: dict[str, str] = {}
    if parc is not None:
        filters["parc"] = parc
    if source is not None:
        filters["source"] = source
    if desc is not None:
        filters["desc"] = desc

    # Extract matching results for each subject
    extracted: dict[str, dict[str, Any]] = {}

    for subject, results in batch_results.items():
        identifier = _get_identifier(subject)
        subject_results: dict[str, Any] = {}

        # Results may be nested {namespace: {key: value}} or flat {key: value}
        # Flatten if needed
        all_results: dict[str, Any] = {}
        for key, value in results.items():
            if isinstance(value, dict):
                # This is a namespace containing keys -> flatten
                all_results.update(value)
            else:
                # Already flat
                all_results[key] = value

        for key, value in all_results.items():
            # Parse the key to check against filters
            try:
                components = parse_result_key(key)
            except ValueError:
                # Skip keys that don't parse (non-standard keys)
                continue

            # Check if key matches all filters
            matches = True
            for filter_key, filter_value in filters.items():
                if components.get(filter_key) != filter_value:
                    matches = False
                    break

            if matches:
                if unwrap:
                    subject_results[key] = _unwrap_value(value)
                else:
                    subject_results[key] = value

        if subject_results:
            extracted[identifier] = subject_results

    if not extracted:
        filter_desc = ", ".join(f"{k}={v}" for k, v in filters.items())
        msg = f"No results found matching filters: {filter_desc}"
        raise ValueError(msg)

    if as_dataframe:
        # Convert to DataFrame format
        rows: list[dict[str, Any]] = []
        for identifier, results in extracted.items():
            row: dict[str, Any] = {"subject": identifier}
            row.update(results)
            rows.append(row)
        return pd.DataFrame(rows)

    return extracted
