"""Batch result extraction utilities.

This module provides convenience functions for extracting specific result types
from batch processing outputs into analysis-friendly formats.

User Story 3: Batch Result Extraction
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, overload

import nibabel as nib
import pandas as pd

if TYPE_CHECKING:
    from lacuna.core.data_types import ParcelData, VoxelMap
    from lacuna.core.mask_data import MaskData


def _get_identifier(item: MaskData | ParcelData | Any) -> str:
    """Get a unique identifier for an item.

    Attempts to extract subject_id from various sources:
    1. metadata["subject_id"]
    2. name attribute
    3. Fallback to index

    Parameters
    ----------
    item : MaskData, ParcelData, or other
        Item to extract identifier from

    Returns
    -------
    str
        Unique identifier for the item
    """
    # Try metadata first
    if hasattr(item, "metadata") and isinstance(item.metadata, dict):
        if "subject_id" in item.metadata:
            return str(item.metadata["subject_id"])

    # Try name attribute
    if hasattr(item, "name") and item.name:
        return str(item.name)

    # Fallback - will be replaced by caller with index
    return ""


def extract_voxelmaps(
    results: list[MaskData],
    source: str | None = None,
    analysis: str | None = None,
    key: str | None = None,
) -> dict[str, VoxelMap]:
    """Extract VoxelMap results from batch processing output.

    Parameters
    ----------
    results : list[MaskData]
        Batch processing results (MaskData with results attached)
    source : str, optional
        Combined source string in format "Analysis.key" (e.g., "FunctionalNetworkMapping.correlation_map").
        This is the preferred format matching result key notation.
    analysis : str, optional
        Analysis name to extract from (e.g., "FunctionalNetworkMapping").
        Can be used instead of source for separate specification.
    key : str, optional
        Result key to extract (e.g., "correlation_map").
        Used with analysis parameter.

    Returns
    -------
    dict[str, VoxelMap]
        Dictionary mapping subject identifiers to VoxelMap results

    Examples
    --------
    >>> # Preferred: use source with dotted notation
    >>> voxelmaps = extract_voxelmaps(
    ...     results,
    ...     source="FunctionalNetworkMapping.correlation_map"
    ... )
    >>>
    >>> # Alternative: separate analysis and key
    >>> voxelmaps = extract_voxelmaps(
    ...     results,
    ...     analysis="FunctionalNetworkMapping",
    ...     key="correlation_map"
    ... )
    >>> for subject_id, vmap in voxelmaps.items():
    ...     nib.save(vmap.data, f"{subject_id}_correlation.nii.gz")
    """
    from lacuna.core.data_types import VoxelMap

    def _to_voxelmap(value: Any, key_name: str, analysis_name: str) -> VoxelMap | None:
        """Convert value to VoxelMap if possible."""
        if isinstance(value, VoxelMap):
            return value
        elif isinstance(value, nib.Nifti1Image):
            # Wrap Nifti1Image in VoxelMap
            return VoxelMap(
                name=key_name,
                data=value,
                space="unknown",  # Best effort - space info not available
                resolution=float(value.header.get_zooms()[0]),  # type: ignore
                metadata={"source_analysis": analysis_name},
            )
        return None

    def _is_voxelmap_like(value: Any) -> bool:
        """Check if value is VoxelMap or can be converted to one."""
        return isinstance(value, (VoxelMap, nib.Nifti1Image))

    # Parse source string if provided (e.g., "FunctionalNetworkMapping.correlation_map")
    if source is not None:
        if "." in source:
            parts = source.split(".", 1)
            analysis = parts[0]
            key = parts[1]
        else:
            # Single word - treat as analysis name
            analysis = source

    extracted = {}

    for idx, item in enumerate(results):
        identifier = _get_identifier(item)
        if not identifier:
            identifier = f"subject_{idx:03d}"

        # Skip if no results attribute
        if not hasattr(item, "results") or item.results is None:
            continue

        # Navigate to the VoxelMap
        try:
            if analysis and key:
                analysis_results = item.results.get(analysis, {})
                value = analysis_results.get(key)
                voxelmap = _to_voxelmap(value, key, analysis) if value else None
            elif analysis:
                # Get first VoxelMap-like from analysis
                analysis_results = item.results.get(analysis, {})
                voxelmap = None
                for k, v in analysis_results.items():
                    if _is_voxelmap_like(v):
                        voxelmap = _to_voxelmap(v, k, analysis)
                        break
            else:
                # Search all analyses
                voxelmap = None
                for analysis_name, analysis_results in item.results.items():
                    if isinstance(analysis_results, dict):
                        for k, v in analysis_results.items():
                            if _is_voxelmap_like(v):
                                voxelmap = _to_voxelmap(v, k, analysis_name)
                                break
                    if voxelmap:
                        break

            if voxelmap is not None:
                extracted[identifier] = voxelmap
        except (KeyError, AttributeError):
            continue

    return extracted


@overload
def extract_parcel_table(
    results: list[MaskData],
    analysis: str,
    key: str,
    parc_filter: str | list[str] | None = None,
) -> pd.DataFrame: ...


@overload
def extract_parcel_table(
    results: list[ParcelData],
    analysis: None = None,
    key: None = None,
    parc_filter: str | list[str] | None = None,
) -> pd.DataFrame: ...


def extract_parcel_table(
    results: list,
    analysis: str | None = None,
    key: str | None = None,
    parc_filter: str | list[str] | None = None,
) -> pd.DataFrame:
    """Extract parcel data into a pandas DataFrame.

    Creates a DataFrame with subjects as rows and regions as columns.

    Parameters
    ----------
    results : list[MaskData] or list[ParcelData]
        Batch processing results. Can be:
        - list[MaskData] with ParcelData in results attribute
        - list[ParcelData] directly
    analysis : str, optional
        Analysis name to extract from (required for MaskData input)
    key : str, optional
        Result key to extract (required for MaskData input)
    parc_filter : str or list[str], optional
        Filter to specific parcellation(s). If provided, only regions
        from matching parcellations are included.

    Returns
    -------
    pd.DataFrame
        DataFrame with subject IDs as index and region names as columns

    Examples
    --------
    >>> # From batch results
    >>> df = extract_parcel_table(
    ...     results,
    ...     analysis="ParcelAggregation",
    ...     key="parcel_means"
    ... )
    >>> print(df.head())

    >>> # From list of ParcelData directly
    >>> df = extract_parcel_table(parcel_data_list)
    """
    from lacuna.core.data_types import ParcelData

    if not results:
        return pd.DataFrame()

    rows = {}

    for idx, item in enumerate(results):
        identifier = _get_identifier(item)
        if not identifier:
            identifier = f"subject_{idx:03d}"

        # Handle ParcelData directly
        if isinstance(item, ParcelData):
            parcel_data = item
        else:
            # Extract from MaskData results
            if not hasattr(item, "results") or item.results is None:
                continue

            try:
                if analysis and key:
                    analysis_results = item.results.get(analysis, {})
                    parcel_data = analysis_results.get(key)
                else:
                    parcel_data = None

                if not isinstance(parcel_data, ParcelData):
                    continue
            except (KeyError, AttributeError):
                continue

        # Apply parcel filter if specified
        data_dict = parcel_data.data
        if parc_filter:
            if isinstance(parc_filter, str):
                parc_filter = [parc_filter]
            data_dict = {
                k: v for k, v in data_dict.items()
                if any(pf.lower() in k.lower() for pf in parc_filter)
            }

        rows[identifier] = data_dict

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame.from_dict(rows, orient="index")


def extract_scalars(
    results: list[MaskData],
    source: str | None = None,
    analysis: str | None = None,
    key: str | None = None,
) -> pd.Series:
    """Extract scalar metric results into a pandas Series.

    Parameters
    ----------
    results : list[MaskData]
        Batch processing results
    source : str, optional
        Combined source string in format "Analysis.key" (e.g., "FunctionalNetworkMapping.summary_statistics").
        This is the preferred format matching result key notation.
    analysis : str, optional
        Analysis name to extract from
    key : str, optional
        Result key to extract

    Returns
    -------
    pd.Series
        Series with subject IDs as index and scalar values

    Examples
    --------
    >>> volumes = extract_scalars(
    ...     results,
    ...     source="RegionalDamage.lesion_volume"
    ... )
    >>> print(volumes.describe())
    """
    from lacuna.core.data_types import ScalarMetric

    # Parse source string if provided (e.g., "FunctionalNetworkMapping.summary_statistics")
    if source is not None:
        if "." in source:
            parts = source.split(".", 1)
            analysis = parts[0]
            key = parts[1]
        else:
            analysis = source

    extracted = {}

    for idx, item in enumerate(results):
        identifier = _get_identifier(item)
        if not identifier:
            identifier = f"subject_{idx:03d}"

        if not hasattr(item, "results") or item.results is None:
            continue

        try:
            if analysis and key:
                analysis_results = item.results.get(analysis, {})
                scalar_metric = analysis_results.get(key)
            else:
                scalar_metric = None

            if isinstance(scalar_metric, ScalarMetric):
                extracted[identifier] = scalar_metric.data
            elif isinstance(scalar_metric, (int, float)):
                extracted[identifier] = scalar_metric
        except (KeyError, AttributeError):
            continue

    return pd.Series(extracted, dtype=float)


__all__ = [
    "extract_voxelmaps",
    "extract_parcel_table",
    "extract_scalars",
]
