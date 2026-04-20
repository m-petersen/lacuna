"""Scoring functions for atlas-lesion overlap computation.

All functions accept an optional parcel_mask parameter for regional scoring.
Zero atlas values are always excluded from scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib

    from lacuna.atlas.types import VoxelAtlas


def score_focal(
    atlas: VoxelAtlas,
    lesion_mask: np.ndarray,
    aggregation: str = "mean",
    parcel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score NT atlas values within a binary lesion mask.

    Parameters
    ----------
    atlas : VoxelAtlas
        Z-scored NT atlas.
    lesion_mask : np.ndarray
        Binary 3D lesion mask. Shape must match atlas maps.
    aggregation : str
        "mean" or "sum". Zeros always excluded.
    parcel_mask : np.ndarray or None
        If provided, restrict scoring to voxels within this parcel.

    Returns
    -------
    dict[str, float]
        {target_name: score}. NaN if no valid voxels.
    """
    effective_mask = lesion_mask.astype(bool)
    if parcel_mask is not None:
        effective_mask = effective_mask & parcel_mask.astype(bool)

    scores = {}
    for target in atlas.targets:
        data = atlas.get_map(target).get_fdata()
        values = data[effective_mask]
        # Exclude zeros
        nonzero = values[values != 0]
        if len(nonzero) == 0:
            scores[target] = float("nan")
        elif aggregation == "mean":
            scores[target] = float(np.mean(nonzero))
        elif aggregation == "sum":
            scores[target] = float(np.sum(nonzero))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return scores


def score_structural_endpoints(
    atlas: VoxelAtlas,
    endpoints_start: np.ndarray,
    endpoints_end: np.ndarray,
    intersecting_ids: np.ndarray,
    parcel_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], int]:
    """Score NT values at streamline endpoints for lesion-intersecting streamlines.

    For each intersecting streamline, computes the mean of NT values at its
    two endpoints. Sums these per-streamline means across all intersecting
    streamlines.

    Parameters
    ----------
    atlas : VoxelAtlas
        Z-scored NT atlas.
    endpoints_start : np.ndarray
        Voxel coordinates of streamline start points. Shape (n_streamlines, 3).
    endpoints_end : np.ndarray
        Voxel coordinates of streamline end points. Shape (n_streamlines, 3).
    intersecting_ids : np.ndarray
        Indices of streamlines that intersect the lesion.
    parcel_mask : np.ndarray or None
        If provided, only include streamlines with at least one endpoint
        in this parcel.

    Returns
    -------
    tuple[dict[str, float], int]
        ({target_name: score}, streamline_count).
        Streamline count is the number of intersecting streamlines used.
    """
    if len(intersecting_ids) == 0:
        return {t: 0.0 for t in atlas.targets}, 0

    # Get endpoints for intersecting streamlines
    starts = endpoints_start[intersecting_ids]
    ends = endpoints_end[intersecting_ids]

    # Optional parcel filtering
    if parcel_mask is not None:
        parcel_bool = parcel_mask.astype(bool)
        in_parcel = (
            parcel_bool[starts[:, 0], starts[:, 1], starts[:, 2]]
            | parcel_bool[ends[:, 0], ends[:, 1], ends[:, 2]]
        )
        starts = starts[in_parcel]
        ends = ends[in_parcel]
        if len(starts) == 0:
            return {t: 0.0 for t in atlas.targets}, 0

    count = len(starts)
    scores = {}
    for target in atlas.targets:
        data = atlas.get_map(target).get_fdata()
        val_start = data[starts[:, 0], starts[:, 1], starts[:, 2]]
        val_end = data[ends[:, 0], ends[:, 1], ends[:, 2]]
        # Mean of two endpoints per streamline, then sum across streamlines
        per_streamline = (val_start + val_end) / 2.0
        scores[target] = float(np.sum(per_streamline))

    return scores, count


def score_functional_overlap(
    atlas: VoxelAtlas,
    connectivity_map: nib.Nifti1Image,
    aggregation: str = "mean",
    parcel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score NT atlas weighted by a functional connectivity map.

    Only positive connectivity values are used. The score is the
    weighted mean of atlas values, with connectivity values as weights.

    Parameters
    ----------
    atlas : VoxelAtlas
        Z-scored NT atlas.
    connectivity_map : nib.Nifti1Image
        Voxelwise connectivity z-map (e.g., from fLNM). Same shape as atlas.
    aggregation : str
        "mean" (weighted mean) or "sum" (weighted sum).
    parcel_mask : np.ndarray or None
        If provided, restrict scoring to this parcel.

    Returns
    -------
    dict[str, float]
        {target_name: score}. NaN if no valid voxels.
    """
    z_data = connectivity_map.get_fdata()

    # Threshold to positive values
    positive_mask = z_data > 0
    if parcel_mask is not None:
        positive_mask = positive_mask & parcel_mask.astype(bool)

    scores = {}
    for target in atlas.targets:
        nt_data = atlas.get_map(target).get_fdata()

        # Valid voxels: positive connectivity AND nonzero NT
        valid = positive_mask & (nt_data != 0)

        if not valid.any():
            scores[target] = float("nan")
            continue

        weights = z_data[valid]
        values = nt_data[valid]
        weighted_product = weights * values

        if aggregation == "mean":
            scores[target] = float(np.sum(weighted_product) / np.sum(weights))
        elif aggregation == "sum":
            scores[target] = float(np.sum(weighted_product))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return scores


def score_react_temporal(
    nt_timeseries: dict[str, np.ndarray],
    lesion_timeseries: np.ndarray,
) -> dict[str, float]:
    """Correlate lesion BOLD timeseries with NT-weighted timeseries.

    Used for REACT-enriched global functional scoring.

    Parameters
    ----------
    nt_timeseries : dict[str, np.ndarray]
        Per-target NT timeseries from REACT stage 1. Each is 1D (n_timepoints,).
    lesion_timeseries : np.ndarray
        Mean BOLD timeseries within the lesion mask. 1D (n_timepoints,).

    Returns
    -------
    dict[str, float]
        {target_name: pearson_correlation}
    """
    scores = {}
    lesion_centered = lesion_timeseries - lesion_timeseries.mean()
    lesion_std = lesion_centered.std()

    for target, nt_ts in nt_timeseries.items():
        nt_centered = nt_ts - nt_ts.mean()
        nt_std = nt_centered.std()
        if lesion_std == 0 or nt_std == 0:
            scores[target] = 0.0
        else:
            r = np.dot(lesion_centered, nt_centered) / (
                len(lesion_centered) * lesion_std * nt_std
            )
            scores[target] = float(r)
    return scores
