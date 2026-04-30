"""ACE (Atlas Connectivity Enrichment).

A framework for enriching spatial atlas maps with functional
connectivity information. Represents an abstraction of the REACT approach 
(Dipasquale et al., 2019, NeuroImage).

Stage 1: Regress BOLD spatial patterns onto atlas maps -> atlas-weighted timeseries
Stage 2: Regress BOLD timeseries onto atlas timeseries -> enriched spatial maps
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from lacuna.atlas.types import VoxelAtlas

logger = logging.getLogger(__name__)


def compute_stage1_mask(atlas: VoxelAtlas) -> np.ndarray:
    """Compute stage 1 mask: intersection of nonzero voxels across all atlas maps.

    Parameters
    ----------
    atlas : VoxelAtlas
        The NT atlas.

    Returns
    -------
    np.ndarray
        Boolean 3D mask where all atlas maps have nonzero values.
    """
    mask = None
    for target in atlas.targets:
        data = atlas.get_map(target).get_fdata()
        target_mask = data != 0
        if mask is None:
            mask = target_mask
        else:
            mask = mask & target_mask
    return mask


def ace_stage1(
    bold_data: np.ndarray,
    atlas_matrix: np.ndarray,
    stage1_mask: np.ndarray,
) -> np.ndarray:
    """ACE Stage 1: extract atlas-weighted timeseries from fMRI.

    Parameters
    ----------
    bold_data : np.ndarray
        fMRI data, shape (n_timepoints, n_voxels). Already masked to brain.
    atlas_matrix : np.ndarray
        NT atlas values, shape (n_targets, n_voxels). Same voxel ordering.
    stage1_mask : np.ndarray
        Boolean mask within the brain mask indicating voxels to use.
        Shape (n_voxels,).

    Returns
    -------
    np.ndarray
        NT timeseries, shape (n_timepoints, n_targets).
    """
    # Mask to stage1 voxels
    x = atlas_matrix[:, stage1_mask].T  # (n_voxels_s1, n_targets)
    y = bold_data[:, stage1_mask].T  # (n_voxels_s1, n_timepoints)

    # Demean
    scaler_x = StandardScaler(with_mean=True, with_std=False)
    x = scaler_x.fit_transform(x)
    scaler_y = StandardScaler(with_mean=True, with_std=False)
    y = scaler_y.fit_transform(y)

    # Regress: voxels as observations, PET maps as predictors, timepoints as DVs
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    beta1 = model.coef_  # (n_timepoints, n_targets)

    return beta1


def ace_stage2(
    bold_data: np.ndarray,
    beta1: np.ndarray,
    stage2_mask: np.ndarray,
    normalize_data: bool = False,
) -> np.ndarray:
    """ACE Stage 2: project atlas timeseries back to voxel space.

    Parameters
    ----------
    bold_data : np.ndarray
        fMRI data, shape (n_timepoints, n_voxels). Already masked to brain.
    beta1 : np.ndarray
        NT timeseries from stage 1, shape (n_timepoints, n_targets).
    stage2_mask : np.ndarray
        Boolean mask within the brain mask indicating voxels to use.
        Shape (n_voxels,).
    normalize_data : bool
        If True, normalize BOLD data to unit standard deviation (optional).

    Returns
    -------
    np.ndarray
        Enriched maps, shape (n_voxels, n_targets). Voxels outside stage2_mask
        are zero.
    """
    n_voxels = bold_data.shape[1]
    n_targets = beta1.shape[1]

    # Prepare predictors (NT timeseries): standardize
    scaler_x = StandardScaler(with_mean=True, with_std=True)
    x = scaler_x.fit_transform(beta1)  # (n_timepoints, n_targets)

    # Prepare dependent variable (BOLD timeseries): demean, optionally normalize
    y = bold_data[:, stage2_mask]  # (n_timepoints, n_voxels_s2)
    scaler_y = StandardScaler(with_mean=True, with_std=normalize_data)
    y = scaler_y.fit_transform(y)

    # Regress: timepoints as observations, NT timeseries as predictors, voxels as DVs
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)

    beta2 = np.zeros((n_voxels, n_targets), dtype=np.float32)
    beta2[stage2_mask] = model.coef_  # (n_voxels_s2, n_targets)

    return beta2


def compute_ace_atlas(
    atlas: VoxelAtlas,
    subjects: "Iterable[np.ndarray]",
    n_subjects: int,
    brain_mask: np.ndarray,
    mask_shape: tuple[int, int, int],
    *,
    on_subject_done: "Callable[[int, np.ndarray], None] | None" = None,
) -> VoxelAtlas:
    """Run full ACE pipeline streaming subjects one at a time.

    Memory peak is one subject's BOLD plus a (n_voxels, n_targets) stage-2
    accumulator (~50 MB at GSP1000-shape). The cross-subject state is the
    accumulator alone — each subject's BOLD can be released after the
    inner loop iteration.

    Stage-1 timeseries are delivered exclusively via ``on_subject_done``.
    Callers that need the timeseries (e.g. ``lacuna prepare ace`` writing
    one ``.npy`` per subject) supply a callback. Callers that only need
    the enriched stage-2 atlas (e.g. ad-hoc analyses, tests) omit it and
    stage-1 outputs are discarded.

    Parameters
    ----------
    atlas : VoxelAtlas
        The prepared (averaged, z-scored) NT atlas.
    subjects : Iterable[np.ndarray]
        Per-subject fMRI data, yielded one at a time. Each element is
        ``(n_timepoints, n_voxels)`` where ``n_voxels`` corresponds to
        the flattened ``brain_mask``.
    n_subjects : int
        How many subjects ``subjects`` will yield. Used as the divisor for
        the cross-subject Fisher-z mean and for progress logging. The
        caller is responsible for ensuring iteration produces this count.
    brain_mask : np.ndarray
        Boolean 3D brain mask. Shape must match atlas maps.
    mask_shape : tuple
        3D shape of the brain volume.
    on_subject_done : Callable[[int, np.ndarray], None], optional
        If provided, called once per subject with ``(subject_index,
        stage1_timeseries)`` immediately after that subject is processed.
        Use this to stream stage-1 ``(n_timepoints, n_targets)`` arrays to
        disk. When omitted, stage-1 outputs are computed (they're needed
        for stage-2) and then discarded.

    Returns
    -------
    VoxelAtlas
        The Fisher-z averaged enriched stage-2 atlas.
    """
    import nibabel as nib

    n_targets = len(atlas.targets)

    # Build atlas matrix within brain mask
    flat_mask = brain_mask.ravel() if brain_mask.ndim == 3 else brain_mask
    atlas_matrix = np.empty((n_targets, int(np.sum(flat_mask))), dtype=np.float64)
    for i, target in enumerate(atlas.targets):
        data = atlas.get_map(target).get_fdata().ravel()
        atlas_matrix[i] = data[flat_mask]

    # Compute stage 1 mask within brain voxels
    stage1_nonzero = np.all(atlas_matrix != 0, axis=0)

    # Stage 2 mask: all brain voxels
    stage2_mask = np.ones(int(np.sum(flat_mask)), dtype=bool)

    # Check collinearity
    x_check = atlas_matrix[:, stage1_nonzero].T
    cond = np.linalg.cond(x_check.T @ x_check)
    if cond > 1000:
        logger.warning(
            "High condition number (%.1f) in NT atlas regressor matrix. "
            "Consider grouped regression or reviewing map selection.",
            cond,
        )

    # Process each subject
    stage2_accumulator = np.zeros((int(np.sum(flat_mask)), n_targets), dtype=np.float64)

    for i, bold in enumerate(subjects):
        logger.info("ACE: processing subject %d/%d", i + 1, n_subjects)

        # Stage 1
        beta1 = ace_stage1(bold, atlas_matrix, stage1_nonzero)
        if on_subject_done is not None:
            on_subject_done(i, beta1)

        # Stage 2
        beta2 = ace_stage2(bold, beta1, stage2_mask)

        # Fisher-z transform and accumulate
        beta2_clipped = np.clip(beta2, -0.9999, 0.9999)
        beta2_z = np.arctanh(beta2_clipped)
        stage2_accumulator += beta2_z

    # Average across subjects
    mean_stage2_z = stage2_accumulator / n_subjects

    # Build enriched atlas maps
    reference_affine = atlas.get_map(atlas.targets[0]).affine
    enriched_maps = {}
    for j, target in enumerate(atlas.targets):
        vol = np.zeros(np.prod(mask_shape), dtype=np.float32)
        vol[flat_mask] = mean_stage2_z[:, j].astype(np.float32)
        vol = vol.reshape(mask_shape)
        enriched_maps[target] = nib.Nifti1Image(vol, reference_affine)

    stage2_atlas = VoxelAtlas(
        maps=enriched_maps,
        space=atlas.space,
        resolution=atlas.resolution,
        domain=atlas.domain,
        metadata={
            **atlas.metadata,
            "enriched": True,
            "method": "ACE",
            "n_subjects": n_subjects,
        },
    )

    return stage2_atlas
