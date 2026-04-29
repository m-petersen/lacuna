"""CLI implementation for the 'lacuna prepare' subcommand.

Handles precomputation of non-subject-specific data needed by analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_prepare_lntf(args) -> None:
    """Build the NT atlas from fetched representative PET maps and cache it.

    Reads .nii.gz files from ``args.source_dir`` (typically the output of
    ``lacuna fetch ntatlas``), z-scores each map, and writes the
    serialized atlas to ``args.cache_dir``.
    """
    from lacuna.atlas.store import build_nt_atlas, save_atlas

    source_dir = Path(args.source_dir)
    cache_dir = Path(args.cache_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"PET atlas source directory not found: {source_dir}\n"
            f"Run 'lacuna fetch ntatlas --output-dir {source_dir}' first."
        )

    logger.info("Building NT atlas from %s", source_dir)
    atlas = build_nt_atlas(source_dir)
    save_atlas(atlas, cache_dir)
    logger.info("NT atlas saved to %s (%d targets)", cache_dir, len(atlas.targets))
    print(f"NT atlas prepared: {len(atlas.targets)} targets saved to {cache_dir}")


def run_prepare_sntf(args) -> None:
    """Precompute endpoint NT weights for all streamlines."""
    from lacuna.atlas.store import load_atlas

    atlas_dir = Path(args.atlas_cache_dir)
    if not (atlas_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"NT atlas not found at {atlas_dir}.\n"
            f"Run 'lacuna prepare lntf --source-dir <pet_dir> --cache-dir {atlas_dir}' first."
        )

    atlas = load_atlas(atlas_dir)
    logger.info("Loaded NT atlas with %d targets", len(atlas.targets))

    cache_dir = Path(args.cache_dir)

    logger.info("Computing endpoint NT weights for tractogram...")
    _precompute_endpoint_weights(atlas, Path(args.connectome_path), cache_dir)
    print(f"Endpoint weights saved to {cache_dir}")


def _precompute_endpoint_weights(atlas, tractogram_path, cache_dir):
    """Sample NT atlas values at all streamline endpoints, cache as float16 matrix."""
    import nibabel as nib
    import numpy as np
    from lacuna.utils.mrtrix import run_mrtrix_command

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get endpoints via tckresample
    endpoints_path = cache_dir / "endpoints.tck"
    run_mrtrix_command(
        f"tckresample {tractogram_path} {endpoints_path} -endpoints",
        verbose=True,
    )

    # Load endpoints
    endpoints_tractogram = nib.streamlines.load(str(endpoints_path))
    streamlines = endpoints_tractogram.streamlines
    n_streamlines = len(streamlines)

    # Convert to voxel coordinates
    ref_img = atlas.get_map(atlas.targets[0])
    inv_affine = np.linalg.inv(ref_img.affine)
    shape = ref_img.shape[:3]

    starts = np.zeros((n_streamlines, 3), dtype=np.int32)
    ends = np.zeros((n_streamlines, 3), dtype=np.int32)
    for i, sl in enumerate(streamlines):
        s = (inv_affine[:3, :3] @ sl[0] + inv_affine[:3, 3]).astype(np.int32)
        e = (inv_affine[:3, :3] @ sl[-1] + inv_affine[:3, 3]).astype(np.int32)
        starts[i] = np.clip(s, 0, np.array(shape) - 1)
        ends[i] = np.clip(e, 0, np.array(shape) - 1)

    # Sample NT values at endpoints
    n_targets = len(atlas.targets)
    weights = np.zeros((n_targets, n_streamlines), dtype=np.float16)
    for j, target in enumerate(atlas.targets):
        data = atlas.get_map(target).get_fdata()
        val_start = data[starts[:, 0], starts[:, 1], starts[:, 2]]
        val_end = data[ends[:, 0], ends[:, 1], ends[:, 2]]
        weights[j] = ((val_start + val_end) / 2.0).astype(np.float16)

    # Save
    np.save(cache_dir / "endpoint_weights.npy", weights)
    np.save(cache_dir / "endpoints_start.npy", starts)
    np.save(cache_dir / "endpoints_end.npy", ends)
    with open(cache_dir / "targets.txt", "w") as f:
        f.write("\n".join(atlas.targets))

    logger.info("Saved endpoint weights: shape %s", weights.shape)


def run_prepare_ace(args) -> None:
    """Run ACE (Atlas Connectivity Enrichment) on normative fMRI data."""
    from lacuna.atlas.store import load_atlas

    atlas_dir = Path(args.atlas_cache_dir)
    atlas = load_atlas(atlas_dir)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading normative fMRI connectome: %s", args.connectome_name)
    raise NotImplementedError(
        "ACE preparation requires connectome loading integration. "
        "Implement after connectome HDF5 structure is confirmed."
    )
