"""Batch result extraction utilities.

This module provides the unified `extract()` function for extracting analysis results
from batch processing results. The function supports filtering by analysis type
and glob patterns.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

import nibabel as nib

if TYPE_CHECKING:
    from lacuna.batch.runner import BatchResults
    from lacuna.core.subject_data import SubjectData


__all__ = ["extract"]


def _get_identifier(subject: SubjectData) -> str:
    """Get the identifier of a subject.

    Builds a BIDS-style identifier from available metadata entities.
    Includes subject_id, session_id, and label for proper disambiguation
    when processing multiple lesion types for the same subject.

    Args:
        subject: The subject to get the identifier of.

    Returns:
        The identifier of the subject. Uses BIDS entities from metadata
        (subject_id, session_id, label) if present, otherwise falls back
        to a string representation.

    Examples:
        - "sub-001" (only subject_id)
        - "sub-001_ses-01" (subject + session)
        - "sub-001_ses-01_label-WMH" (subject + session + label)
        - "sub-001_label-acuteinfarct" (subject + label, no session)
    """
    if not hasattr(subject, "metadata") or not subject.metadata:
        return f"subject_{id(subject)}"

    metadata = subject.metadata
    parts = []

    # Build identifier from BIDS entities in standard order
    subject_id = metadata.get("subject_id")
    if subject_id:
        parts.append(str(subject_id))

    session_id = metadata.get("session_id")
    if session_id:
        parts.append(str(session_id))

    label = metadata.get("label")
    if label:
        parts.append(f"label-{label}")

    if parts:
        return "_".join(parts)

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
    batch_results: BatchResults | list,
    *,
    analysis: str | None = None,
    pattern: str | None = None,
    unwrap: bool = False,
) -> dict[str, Any]:
    """Extract analysis results from batch processing results.

    Unified extraction function that replaces the legacy `extract_voxelmaps()`,
    `extract_parcel_table()`, and `extract_scalars()` functions. Provides
    flexible filtering using glob patterns.

    Args:
        batch_results: The batch results to extract from. Can be either:
            - list[SubjectData]: Direct output from batch_process()
            - dict[SubjectData, dict]: BatchResults format with {subject: results}
        analysis: Filter by analysis namespace (e.g., "FunctionalNetworkMapping",
            "RegionalDamage"). This filters by the top-level namespace in results.
        pattern: Glob pattern to match result keys (e.g., "*correlationmap*",
            "parc-Schaefer*_desc-*"). Supports fnmatch-style wildcards:
            - ``*`` matches any sequence of characters
            - ``?`` matches any single character
            - ``[seq]`` matches any character in seq
        unwrap: If True, call `get_data()` on result objects to return raw values.
            If False (default), return wrapper objects (VoxelMap, ParcelData, etc.).

    Returns:
        Dictionary mapping subject identifiers to extracted values.
        If only one result key matches per subject, returns {subject: value}.
        If multiple keys match, returns {subject: {key: value}}.

    Examples:
        Extract using glob pattern:
        >>> results = extract(batch_results, analysis="FunctionalNetworkMapping",
        ...                   pattern="*correlationmap*")

        Extract with pattern matching parcellation:
        >>> results = extract(batch_results, pattern="*Schaefer*")

        Extract with unwrapping (raw values):
        >>> results = extract(batch_results, pattern="*Schaefer*", unwrap=True)

    Raises:
        ValueError: If batch_results is empty or no results match the filters.
    """
    if not batch_results:
        msg = "batch_results is empty"
        raise ValueError(msg)

    # Handle both list[SubjectData] (from batch_process) and dict (BatchResults)
    # Convert list to dict format: {subject: subject.results}
    if isinstance(batch_results, list):
        batch_dict: dict[Any, dict[str, Any]] = {}
        for item in batch_results:
            if hasattr(item, "results"):
                batch_dict[item] = item.results
            else:
                # Skip items without results
                continue
        batch_results = batch_dict

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
                # Check if key matches pattern (or match all if no pattern)
                if pattern is None or fnmatch(key, pattern):
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
        if pattern:
            filter_parts.append(f"pattern={pattern!r}")
        filter_desc = ", ".join(filter_parts) if filter_parts else "(no filters)"
        msg = f"No results found matching filters: {filter_desc}"
        raise ValueError(msg)

    # Simplify output: if each subject has only one result key, return the value directly
    # Check if all subjects have exactly one result
    all_single = all(len(v) == 1 for v in extracted.values())
    if all_single:
        # Return {subject: value} instead of {subject: {key: value}}
        return {
            identifier: next(iter(results.values())) for identifier, results in extracted.items()
        }

    return extracted
