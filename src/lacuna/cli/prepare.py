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
    _precompute_endpoint_weights(
        atlas, atlas_dir, Path(args.connectome_path), cache_dir,
    )
    print(f"Endpoint weights saved to {cache_dir}")


def _precompute_endpoint_weights(atlas, atlas_dir, tractogram_path, cache_dir):
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
    * ``lacuna_asset.json`` — shared envelope describing the cache's identity
                              and the (tractogram, ntatlas) inputs it was
                              built from. Read at run time to catch a
                              mismatched ``--connectome-path`` or atlas swap.
    """
    import nibabel as nib
    import numpy as np
    from lacuna.assets.envelope import (
        AssetEnvelope,
        AssetType,
        RequiresEntry,
        fingerprint,
        write_envelope,
    )
    from lacuna.utils.mrtrix import run_mrtrix_command

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

    atlas_dir = Path(atlas_dir).resolve()
    tractogram_path_resolved = Path(tractogram_path).resolve()
    env = AssetEnvelope(
        asset_type=AssetType.SNTF_CACHE,
        identity=fingerprint(cache_dir, AssetType.SNTF_CACHE),
        requires=[
            RequiresEntry(
                role="tractogram",
                asset_type=AssetType.STRUCTURAL_CONNECTOME,
                identity=fingerprint(
                    tractogram_path_resolved, AssetType.STRUCTURAL_CONNECTOME,
                ),
                path_hint=str(tractogram_path_resolved),
            ),
            RequiresEntry(
                role="ntatlas",
                asset_type=AssetType.NTATLAS,
                identity=fingerprint(atlas_dir, AssetType.NTATLAS),
                path_hint=str(atlas_dir),
            ),
        ],
        provenance={
            "command": "lacuna prepare sntf",
            "n_streamlines": int(n_streamlines),
            "n_targets": int(n_targets),
        },
        data={
            "n_streamlines": int(n_streamlines),
            "targets": list(atlas.targets),
        },
    )
    write_envelope(env, cache_dir)

    logger.info(
        "Cached endpoint weights for %d streamlines × %d targets",
        n_streamlines, n_targets,
    )


def run_prepare_ace(args) -> None:
    """Build an ACE-enriched atlas cache from an NT atlas + functional connectome."""
    import shutil

    import numpy as np

    from lacuna.assets.connectomes.functional_io import (
        iter_subject_timeseries,
        list_connectome_batch_files,
        read_mask_info,
    )
    from lacuna.assets.envelope import (
        AssetEnvelope,
        AssetType,
        RequiresEntry,
        asset_present,
        fingerprint,
        write_envelope,
    )
    from lacuna.atlas.ace import compute_ace_atlas
    from lacuna.atlas.store import load_atlas, save_atlas

    atlas_dir = Path(args.ntatlas_dir)
    conn_path = Path(args.connectome_path)
    cache_dir = Path(args.cache_dir)
    max_subjects = getattr(args, "max_subjects", None)

    # Pre-flight asset checks
    if not asset_present(atlas_dir, AssetType.NTATLAS):
        raise FileNotFoundError(
            f"NT atlas not found at {atlas_dir}.\n"
            f"Run 'lacuna fetch ntatlas --output-dir {atlas_dir}' first."
        )
    batch_files = list_connectome_batch_files(conn_path)  # raises FileNotFoundError on empty

    # Load atlas + connectome metadata
    atlas = load_atlas(atlas_dir)
    mask_info = read_mask_info(batch_files[0])
    brain_mask = _reconstruct_3d_mask(mask_info)

    # Atlas/connectome compatibility — fail fast before the expensive part
    ref = atlas.get_map(atlas.targets[0])
    atlas_shape_3d = tuple(ref.shape[:3])
    if atlas_shape_3d != tuple(mask_info["mask_shape"]):
        raise ValueError(
            "atlas shape does not match the connectome's brain-mask shape.\n"
            f"  atlas:     {atlas_shape_3d}\n"
            f"  connectome: {tuple(mask_info['mask_shape'])}\n"
            "Re-fetch the atlas and connectome at matching resolution/space."
        )
    if not np.allclose(ref.affine, mask_info["mask_affine"], atol=1e-3):
        raise ValueError(
            "atlas affine does not match the connectome's brain-mask affine.\n"
            f"  atlas affine:\n{ref.affine}\n"
            f"  connectome affine:\n{mask_info['mask_affine']}"
        )

    # Pre-scan HDF5 attrs (cheap — no timeseries reads) so the user sees
    # an honest subject count and RAM estimate before the long iteration.
    import h5py

    total_in_conn = 0
    n_timepoints_attr = 0
    for bf in batch_files:
        with h5py.File(bf, "r") as hf:
            total_in_conn += int(hf.attrs.get("n_subjects", 0))
            if n_timepoints_attr == 0:
                n_timepoints_attr = int(hf.attrs.get("n_timepoints", 0))
    n_to_load = (
        min(max_subjects, total_in_conn)
        if max_subjects is not None
        else total_in_conn
    )
    if n_to_load == 0:
        raise ValueError(
            f"No subjects found in connectome at {conn_path}; cannot prepare ACE."
        )
    if max_subjects is not None and n_to_load < total_in_conn:
        logger.info(
            "Streaming %d of %d subjects (capped via --max-subjects).",
            n_to_load, total_in_conn,
        )
    else:
        logger.info("Streaming %d subjects from connectome.", n_to_load)

    # Wipe stale stage1 BEFORE the stream starts; the streaming callback
    # writes each subject's .npy as soon as compute_ace_atlas yields it.
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = cache_dir / "stage1_timeseries"
    if stage1_dir.exists():
        shutil.rmtree(stage1_dir)
    stage1_dir.mkdir()

    subject_ids: list[str] = []

    def _yield_bold():
        """Yield BOLD arrays from the connectome iterator and capture IDs.

        Lives inline so the closure over ``subject_ids`` and ``max_subjects``
        is obvious. Truncates at ``max_subjects`` when set.
        """
        n_yielded = 0
        for subj_id, ts in iter_subject_timeseries(conn_path):
            if max_subjects is not None and n_yielded >= max_subjects:
                break
            subject_ids.append(subj_id)
            n_yielded += 1
            yield ts

    def _save_stage1(i: int, beta1: np.ndarray) -> None:
        np.save(stage1_dir / f"subject-{i:04d}.npy", beta1)

    stage2_atlas = compute_ace_atlas(
        atlas,
        _yield_bold(),
        n_to_load,
        brain_mask,
        mask_info["mask_shape"],
        on_subject_done=_save_stage1,
    )
    n_subjects = len(subject_ids)

    # Write stage2_atlas (envelope last). Wipe stale stage2 too.
    stage2_dir = cache_dir / "stage2_atlas"
    if stage2_dir.exists():
        shutil.rmtree(stage2_dir)
    save_atlas(stage2_atlas, stage2_dir)

    (cache_dir / "subject_ids.txt").write_text("\n".join(subject_ids) + "\n")

    n_timepoints = n_timepoints_attr
    # Source ntatlas is build-time provenance, NOT a runtime requirement:
    # FNTF in enriched mode consumes <ace_dir>/stage2_atlas (which has its
    # own envelope), and the original ntatlas may be gone by then. Recording
    # its identity here keeps the audit trail.
    source_ntatlas_identity = fingerprint(atlas_dir, AssetType.NTATLAS)
    env = AssetEnvelope(
        asset_type=AssetType.ACE_CACHE,
        identity=fingerprint(cache_dir, AssetType.ACE_CACHE),
        requires=[
            RequiresEntry(
                role="connectome",
                asset_type=AssetType.FUNCTIONAL_CONNECTOME,
                identity=fingerprint(conn_path, AssetType.FUNCTIONAL_CONNECTOME),
                path_hint=str(conn_path.resolve()),
            ),
        ],
        provenance={
            "command": "lacuna prepare ace",
            "n_subjects": n_subjects,
            "max_subjects": max_subjects,
            "space": atlas.space,
            "n_targets": len(atlas.targets),
            "source_ntatlas_path": str(atlas_dir.resolve()),
            "source_ntatlas_identity": source_ntatlas_identity.to_dict(),
        },
        data={
            "n_targets": len(atlas.targets),
            "n_timepoints": n_timepoints,
        },
    )
    write_envelope(env, cache_dir)

    print(f"ACE cache written to {cache_dir}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Targets:  {len(atlas.targets)}")


def _reconstruct_3d_mask(mask_info: dict) -> "np.ndarray":
    """Build a 3D boolean mask from the connectome's flat indices."""
    import numpy as np

    ix, iy, iz = mask_info["mask_indices"]
    mask = np.zeros(mask_info["mask_shape"], dtype=bool)
    mask[ix, iy, iz] = True
    return mask
