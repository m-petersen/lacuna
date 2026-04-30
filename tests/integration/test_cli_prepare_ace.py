"""End-to-end integration test for `lacuna prepare ace`."""

from __future__ import annotations

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.envelope import (
    AssetType,
    ENVELOPE_FILENAME,
    read_envelope,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.data.ntatlas import load_collection


def _write_synthetic_collection(src: Path, targets: list[str]) -> None:
    coll = load_collection()
    target_to_map_id = {
        map_id.split("_", 1)[0][len("target-") :]: map_id
        for ids in coll["systems"].values()
        for map_id in ids
    }
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    shape = (5, 5, 5)
    for target in targets:
        map_id = target_to_map_id[target]
        rng = np.random.default_rng(hash(map_id) % 2**32)
        data = rng.random(shape).astype(np.float32) + 0.1
        fname = f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(src / fname))


def _write_synthetic_gsp_batch(path, *, n_subjects, n_timepoints, atlas_shape, atlas_affine):
    flat = np.zeros(np.prod(atlas_shape), dtype=bool)
    flat[: int(np.prod(atlas_shape) // 2)] = True
    n_voxels = int(flat.sum())
    indices_3d = np.array(np.unravel_index(np.where(flat)[0], atlas_shape))
    with h5py.File(path, "w") as hf:
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((n_subjects, n_timepoints, n_voxels)).astype(np.float32)
        hf.create_dataset("timeseries", data=ts)
        hf.create_dataset("mask_indices", data=indices_3d)
        hf.create_dataset("mask_affine", data=atlas_affine)
        hf.attrs["n_subjects"] = n_subjects
        hf.attrs["n_timepoints"] = n_timepoints
        hf.attrs["n_voxels"] = n_voxels
        hf.attrs["mask_shape"] = atlas_shape


@pytest.mark.integration
def test_prepare_ace_cli_e2e(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas"
    save_atlas(atlas, atlas_dir)

    conn_dir = tmp_path / "conn"
    conn_dir.mkdir()
    ref = atlas.get_map(atlas.targets[0])
    _write_synthetic_gsp_batch(
        conn_dir / "batch_0001.h5",
        n_subjects=3,
        n_timepoints=8,
        atlas_shape=ref.shape,
        atlas_affine=ref.affine,
    )

    cache_dir = tmp_path / "ace_cache"

    from lacuna.cli import main

    rc = main([
        "prepare", "ace",
        "--ntatlas-dir", str(atlas_dir),
        "--connectome-path", str(conn_dir),
        "--cache-dir", str(cache_dir),
        "--max-subjects", "2",
    ])
    assert rc == 0

    # Cache directory shape
    assert (cache_dir / ENVELOPE_FILENAME).exists()
    assert (cache_dir / "stage2_atlas" / ENVELOPE_FILENAME).exists()
    stage1_files = sorted((cache_dir / "stage1_timeseries").glob("subject-*.npy"))
    assert len(stage1_files) == 2

    # Envelope contents
    env = read_envelope(cache_dir)
    assert env.asset_type == AssetType.ACE_CACHE
    assert env.provenance["max_subjects"] == 2
    assert env.provenance["n_subjects"] == 2
    assert "source_ntatlas_path" in env.provenance
    roles = {r.role for r in env.requires}
    assert roles == {"connectome"}
