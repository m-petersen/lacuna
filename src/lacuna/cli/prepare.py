"""CLI implementation for the 'lacuna prepare' subcommand.

Handles precomputation of non-subject-specific data needed by analyses.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_prepare_sntf(args) -> None:
    """Precompute endpoint NT weights for all streamlines."""
    from lacuna.assets.envelope import AssetType, asset_present
    from lacuna.atlas.store import load_atlas

    atlas_dir = Path(args.ntatlas_dir)
    if not asset_present(atlas_dir, AssetType.NTATLAS):
        raise FileNotFoundError(
            f"NT atlas not found at {atlas_dir}.\n"
            f"Run 'lacuna fetch ntatlas --output-dir {atlas_dir}' first."
        )

    atlas = load_atlas(atlas_dir)
    logger.info("Loaded NT atlas with %d targets", len(atlas.targets))

    cache_dir = Path(args.cache_dir)

    logger.info("Computing endpoint NT weights for tractogram...")
    _precompute_endpoint_weights(atlas, Path(args.connectome_path), cache_dir)
    print(f"Endpoint weights saved to {cache_dir}")


def _precompute_endpoint_weights(atlas, tractogram_path, cache_dir):
    """Build the SNTF cache for a (atlas, tractogram) pair.

    Produces, in ``cache_dir``:

    * ``endpoints.tck``       — full-tractogram endpoints (output of `tckresample`).
    * ``start_weights.npy``   — (n_targets, n_streamlines) float32: NT values at the
                                start endpoint of each streamline, per target.
    * ``end_weights.npy``     — same, for the end endpoint.
    * ``targets.txt``         — newline-separated target names matching the
                                row order of the weights arrays.
    * ``streamline_indices.txt`` — float per line, ``i`` for streamline ``i``.
                                  Pass to ``tckedit -tck_weights_in`` so
                                  ``-tck_weights_out`` returns the surviving
                                  original streamline IDs after lesion filtering.
    * ``connectome_meta.json`` — content fingerprint of the source tractogram
                                so ``lacuna run sntf`` can detect a mismatched
                                ``--connectome-path``.
    """
    import json

    import nibabel as nib
    import numpy as np
    from lacuna.utils.mrtrix import run_mrtrix_command
    from lacuna.utils.tractogram_id import compute_tractogram_fingerprint

    cache_dir.mkdir(parents=True, exist_ok=True)

    endpoints_path = cache_dir / "endpoints.tck"
    run_mrtrix_command(
        ["tckresample", str(tractogram_path), str(endpoints_path), "-endpoints", "-force"],
        verbose=True,
    )

    streamlines = nib.streamlines.load(str(endpoints_path)).streamlines
    n_streamlines = len(streamlines)

    ref_img = atlas.get_map(atlas.targets[0])
    inv_affine = np.linalg.inv(ref_img.affine)
    shape = np.array(ref_img.shape[:3])

    starts = np.empty((n_streamlines, 3), dtype=np.int32)
    ends = np.empty((n_streamlines, 3), dtype=np.int32)
    for i, sl in enumerate(streamlines):
        starts[i] = np.clip(
            (inv_affine[:3, :3] @ sl[0] + inv_affine[:3, 3]).astype(np.int32),
            0, shape - 1,
        )
        ends[i] = np.clip(
            (inv_affine[:3, :3] @ sl[-1] + inv_affine[:3, 3]).astype(np.int32),
            0, shape - 1,
        )

    n_targets = len(atlas.targets)
    start_weights = np.empty((n_targets, n_streamlines), dtype=np.float32)
    end_weights = np.empty((n_targets, n_streamlines), dtype=np.float32)
    for j, target in enumerate(atlas.targets):
        data = atlas.get_map(target).get_fdata()
        start_weights[j] = data[starts[:, 0], starts[:, 1], starts[:, 2]]
        end_weights[j] = data[ends[:, 0], ends[:, 1], ends[:, 2]]

    np.save(cache_dir / "start_weights.npy", start_weights)
    np.save(cache_dir / "end_weights.npy", end_weights)
    (cache_dir / "targets.txt").write_text("\n".join(atlas.targets) + "\n")
    # Pure float index file for use with tckedit -tck_weights_in.
    indices_path = cache_dir / "streamline_indices.txt"
    np.savetxt(indices_path, np.arange(n_streamlines, dtype=np.float32), fmt="%.0f")

    fingerprint = compute_tractogram_fingerprint(tractogram_path)
    (cache_dir / "connectome_meta.json").write_text(json.dumps(fingerprint, indent=2))

    logger.info(
        "Cached endpoint weights for %d streamlines × %d targets",
        n_streamlines, n_targets,
    )


def run_prepare_ace(args) -> None:
    """Run ACE (Atlas Connectivity Enrichment) on normative fMRI data."""
    from lacuna.atlas.store import load_atlas
    from lacuna.cli.main import register_functional_connectome_from_path

    atlas_dir = Path(args.ntatlas_dir)
    atlas = load_atlas(atlas_dir)

    connectome_path = Path(args.connectome_path)
    if not connectome_path.exists():
        raise FileNotFoundError(
            f"Connectome path does not exist: {connectome_path}\n\n"
            "To download a functional connectome:\n"
            "  lacuna fetch gsp1000"
        )
    connectome_name = register_functional_connectome_from_path(connectome_path)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading normative fMRI connectome: %s", connectome_name)
    raise NotImplementedError(
        "ACE preparation requires connectome loading integration. "
        "Implement after connectome HDF5 structure is confirmed."
    )
