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
    analysis: str | None = None,
    parc: str | None = None,
    source: str | None = None,
    desc: str | None = None,
    unwrap: bool = False,
    as_dataframe: bool = False,
) -> dict[str, Any] | pd.DataFrame:
    """Extract analysis results from batch processing results.

    Unified extraction function that replaces the legacy `extract_voxelmaps()`,
    `extract_parcel_table()`, and `extract_scalars()` functions. Provides
    flexible filtering using BIDS-style key components or analysis namespace.

    Args:
        batch_results: The batch results to extract from.
        analysis: Filter by analysis namespace (e.g., "FunctionalNetworkMapping",
            "RegionalDamage"). This filters by the top-level namespace in results.
        parc: Filter by parcellation name (e.g., "AAL116", "Schaefer100").
            Matches the 'parc' component of BIDS-style result keys.
        source: Filter by source type (e.g., "ParcelAggregation", "RegionalDamage").
            Matches the 'source' component of BIDS-style result keys.
        desc: Filter by description (e.g., "parcel_means", "correlation_map").
            Matches the 'desc' component of BIDS-style result keys, OR matches
            plain result keys directly (e.g., "correlation_map").
        unwrap: If True, call `get_data()` on result objects to return raw values.
            If False (default), return wrapper objects (VoxelMap, ParcelData, etc.).
        as_dataframe: If True, return results as a pandas DataFrame with columns
            for subject identifier and all extracted result keys.

    Returns:
        If as_dataframe is False:
            Dictionary mapping subject identifiers to extracted values.
            If only one result key matches per subject, returns {subject: value}.
            If multiple keys match, returns {subject: {key: value}}.
        If as_dataframe is True:
            DataFrame with 'subject' column and columns for each result key.

    Examples:
        Extract correlation maps from FunctionalNetworkMapping:
        >>> results = extract(batch_results, analysis="FunctionalNetworkMapping",
        ...                   desc="correlation_map")

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

    # Build BIDS-style filter criteria
    bids_filters: dict[str, str] = {}
    if parc is not None:
        bids_filters["parc"] = parc
    if source is not None:
        bids_filters["source"] = source
    if desc is not None:
        bids_filters["desc"] = desc

    # Extract matching results for each subject
    extracted: dict[str, dict[str, Any]] = {}

    for subject, results in batch_results.items():
        identifier = _get_identifier(subject)
        subject_results: dict[str, Any] = {}

        # Results are nested {namespace: {key: value}}
        # Filter by analysis namespace if specified
        namespaces_to_search = results.items()
        if analysis is not None:
            # Only search in the specified namespace
            if analysis in results:
                namespaces_to_search = [(analysis, results[analysis])]
            else:
                # Namespace not found for this subject
                continue

        for _namespace, namespace_results in namespaces_to_search:
            if not isinstance(namespace_results, dict):
                # Skip non-dict values (shouldn't happen but be safe)
                continue

            for key, value in namespace_results.items():
                # Try to parse as BIDS-style key
                try:
                    components = parse_result_key(key)
                    is_bids_key = True
                except ValueError:
                    # Plain key (e.g., "correlation_map")
                    is_bids_key = False
                    components = {}

                # Check if key matches filters
                matches = True

                if is_bids_key:
                    # Check BIDS-style filters
                    for filter_key, filter_value in bids_filters.items():
                        if components.get(filter_key) != filter_value:
                            matches = False
                            break
                else:
                    # Plain key - only match on desc filter
                    # (parc and source don't apply to plain keys)
                    if desc is not None and key != desc:
                        matches = False
                    # If parc or source filters are set, plain keys don't match
                    if parc is not None or source is not None:
                        matches = False

                if matches:
                    if unwrap:
                        subject_results[key] = _unwrap_value(value)
                    else:
                        subject_results[key] = value

        if subject_results:
            extracted[identifier] = subject_results

    if not extracted:
        filter_parts = []
        if analysis:
            filter_parts.append(f"analysis={analysis}")
        for k, v in bids_filters.items():
            filter_parts.append(f"{k}={v}")
        filter_desc = ", ".join(filter_parts) if filter_parts else "(no filters)"
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

    # Simplify output: if each subject has only one result key, return the value directly
    # Check if all subjects have exactly one result
    all_single = all(len(v) == 1 for v in extracted.values())
    if all_single:
        # Return {subject: value} instead of {subject: {key: value}}
        return {
            identifier: next(iter(results.values())) for identifier, results in extracted.items()
        }

    return extracted
