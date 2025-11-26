"""Batch result extraction utilities.

This module provides convenience functions for extracting specific result types
from batch processing outputs into analysis-friendly formats.

User Story 3: Batch Result Extraction
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, overload

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
    analysis: str | None = None,
    key: str | None = None,
) -> dict[str, VoxelMap]:
    """Extract VoxelMap results from batch processing output.

    Parameters
    ----------
    results : list[MaskData]
        Batch processing results (MaskData with results attached)
    analysis : str, optional
        Analysis name to extract from (e.g., "FunctionalNetworkMapping")
    key : str, optional
        Result key to extract (e.g., "correlation_map")

    Returns
    -------
    dict[str, VoxelMap]
        Dictionary mapping subject identifiers to VoxelMap results

    Examples
    --------
    >>> voxelmaps = extract_voxelmaps(
    ...     results,
    ...     analysis="FunctionalNetworkMapping",
    ...     key="correlation_map"
    ... )
    >>> for subject_id, vmap in voxelmaps.items():
    ...     nib.save(vmap.data, f"{subject_id}_correlation.nii.gz")
    """
    from lacuna.core.data_types import VoxelMap

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
                voxelmap = analysis_results.get(key)
            elif analysis:
                # Get first VoxelMap from analysis
                analysis_results = item.results.get(analysis, {})
                voxelmap = None
                for v in analysis_results.values():
                    if isinstance(v, VoxelMap):
                        voxelmap = v
                        break
            else:
                # Search all analyses
                voxelmap = None
                for analysis_results in item.results.values():
                    if isinstance(analysis_results, dict):
                        for v in analysis_results.values():
                            if isinstance(v, VoxelMap):
                                voxelmap = v
                                break
                    if voxelmap:
                        break

            if voxelmap is not None and isinstance(voxelmap, VoxelMap):
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
    analysis: str | None = None,
    key: str | None = None,
) -> pd.Series:
    """Extract scalar metric results into a pandas Series.

    Parameters
    ----------
    results : list[MaskData]
        Batch processing results
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
    ...     analysis="RegionalDamage",
    ...     key="lesion_volume"
    ... )
    >>> print(volumes.describe())
    """
    from lacuna.core.data_types import ScalarMetric

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
