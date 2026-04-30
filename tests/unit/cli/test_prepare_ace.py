"""Unit tests for lacuna.cli.prepare.run_prepare_ace."""

from __future__ import annotations

from argparse import Namespace
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
from lacuna.cli.prepare import run_prepare_ace
from lacuna.data.ntatlas import load_collection


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _write_synthetic_collection(src: Path, targets: list[str]) -> None:
    """Drop synthetic NIfTIs into ``src`` named so ``build_nt_atlas`` finds them."""
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


def _write_synthetic_gsp_batch(
    path: Path,
    *,
    n_subjects: int,
    n_timepoints: int,
    atlas_shape: tuple[int, int, int] = (5, 5, 5),
    atlas_affine: np.ndarray | None = None,
) -> None:
    """Write one HDF5 batch matching gsp1000_to_hdf5's schema, sized to the atlas."""
    if atlas_affine is None:
        atlas_affine = np.eye(4) * 2
        atlas_affine[3, 3] = 1
    flat = np.zeros(np.prod(atlas_shape), dtype=bool)
    flat[: int(np.prod(atlas_shape) // 2)] = True
    n_voxels = int(flat.sum())
    indices_3d = np.array(np.unravel_index(np.where(flat)[0], atlas_shape))

    with h5py.File(path, "w") as hf:
        ts = np.empty((n_subjects, n_timepoints, n_voxels), dtype=np.float32)
        rng = np.random.default_rng(0)
        for s in range(n_subjects):
            ts[s] = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        hf.create_dataset("timeseries", data=ts)
        hf.create_dataset("mask_indices", data=indices_3d)
        hf.create_dataset("mask_affine", data=atlas_affine)
        hf.attrs["n_subjects"] = n_subjects
        hf.attrs["n_timepoints"] = n_timepoints
        hf.attrs["n_voxels"] = n_voxels
        hf.attrs["mask_shape"] = atlas_shape


@pytest.fixture
def ace_inputs(tmp_path):
    """Synthetic NT atlas + synthetic GSP1000-shaped HDF5 batch sized to match."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas"
    save_atlas(atlas, atlas_dir)

    conn_dir = tmp_path / "conn"
    conn_dir.mkdir()
    atlas_ref = atlas.get_map(atlas.targets[0])
    _write_synthetic_gsp_batch(
        conn_dir / "batch_0001.h5",
        n_subjects=3,
        n_timepoints=8,
        atlas_shape=atlas_ref.shape,
        atlas_affine=atlas_ref.affine,
    )
    return atlas_dir, conn_dir


def _ace_args(atlas_dir, conn_dir, cache_dir, *, max_subjects=None):
    return Namespace(
        ntatlas_dir=str(atlas_dir),
        connectome_path=str(conn_dir),
        cache_dir=str(cache_dir),
        max_subjects=max_subjects,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_ace_writes_full_cache(ace_inputs, tmp_path):
    atlas_dir, conn_dir = ace_inputs
    cache_dir = tmp_path / "ace_cache"

    run_prepare_ace(_ace_args(atlas_dir, conn_dir, cache_dir))

    # Top-level envelope
    assert (cache_dir / ENVELOPE_FILENAME).exists()
    env = read_envelope(cache_dir)
    assert env.asset_type == AssetType.ACE_CACHE

    # Stage 2 atlas (saved via save_atlas, which writes its own envelope)
    assert (cache_dir / "stage2_atlas" / ENVELOPE_FILENAME).exists()
    assert (cache_dir / "stage2_atlas" / "maps").is_dir()

    # Stage 1 timeseries: one .npy per subject (3 subjects in fixture)
    stage1_files = sorted((cache_dir / "stage1_timeseries").glob("subject-*.npy"))
    assert len(stage1_files) == 3
    assert stage1_files[0].name == "subject-0000.npy"

    # Subject IDs file
    ids = (cache_dir / "subject_ids.txt").read_text().splitlines()
    assert len(ids) == 3
    # IDs come from iter_subject_timeseries: f"{batch_stem}-{row:04d}"
    assert ids[0] == "batch_0001-0000"


def test_prepare_ace_envelope_pins_only_connectome_in_requires(ace_inputs, tmp_path):
    """Only the connectome is a runtime requirement.

    The source ntatlas is build-time provenance — FNTF in enriched mode
    consumes ``<ace_dir>/stage2_atlas`` (which has its own envelope), so
    the source ntatlas's identity is recorded in ``provenance``, NOT
    ``requires``.
    """
    atlas_dir, conn_dir = ace_inputs
    cache_dir = tmp_path / "ace_cache"

    run_prepare_ace(_ace_args(atlas_dir, conn_dir, cache_dir))

    env = read_envelope(cache_dir)
    roles = {r.role for r in env.requires}
    assert roles == {"connectome"}

    conn_req = next(r for r in env.requires if r.role == "connectome")
    assert conn_req.asset_type == AssetType.FUNCTIONAL_CONNECTOME
    assert "sha256" in conn_req.identity.fields

    # Source ntatlas information is in provenance, not requires.
    assert env.provenance["source_ntatlas_path"] == str(atlas_dir.resolve())
    assert env.provenance["source_ntatlas_identity"]["fields"]["n_targets"] == 2


def test_prepare_ace_max_subjects_truncates(ace_inputs, tmp_path):
    atlas_dir, conn_dir = ace_inputs
    cache_dir = tmp_path / "ace_cache"

    run_prepare_ace(_ace_args(atlas_dir, conn_dir, cache_dir, max_subjects=2))

    stage1_files = sorted((cache_dir / "stage1_timeseries").glob("subject-*.npy"))
    assert len(stage1_files) == 2  # exactly 2, not "≤ 2"

    env = read_envelope(cache_dir)
    assert env.provenance["n_subjects"] == 2
    assert env.provenance["max_subjects"] == 2


def test_prepare_ace_provenance_records_metadata(ace_inputs, tmp_path):
    atlas_dir, conn_dir = ace_inputs
    cache_dir = tmp_path / "ace_cache"

    run_prepare_ace(_ace_args(atlas_dir, conn_dir, cache_dir))

    env = read_envelope(cache_dir)
    assert env.provenance["command"] == "lacuna prepare ace"
    assert env.provenance["n_subjects"] == 3
    assert env.provenance["max_subjects"] is None
    assert env.provenance["space"] == "MNI152NLin6Asym"
    assert env.provenance["n_targets"] == 2


def test_prepare_ace_shape_mismatch_raises(tmp_path):
    """Atlas shape != connectome mask shape → fail before running ACE."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas"
    save_atlas(atlas, atlas_dir)

    conn_dir = tmp_path / "conn"
    conn_dir.mkdir()
    # Deliberately wrong mask shape (atlas is (5,5,5), batch claims (4,4,4)):
    _write_synthetic_gsp_batch(
        conn_dir / "batch_0001.h5",
        n_subjects=2, n_timepoints=8, atlas_shape=(4, 4, 4),
    )
    cache_dir = tmp_path / "ace_cache"

    with pytest.raises(ValueError, match="shape"):
        run_prepare_ace(_ace_args(atlas_dir, conn_dir, cache_dir))


def test_prepare_ace_missing_atlas_raises(tmp_path):
    cache_dir = tmp_path / "ace_cache"
    conn_dir = tmp_path / "conn"
    conn_dir.mkdir()
    _write_synthetic_gsp_batch(
        conn_dir / "batch_0001.h5", n_subjects=1, n_timepoints=4,
    )
    with pytest.raises(FileNotFoundError, match="NT atlas"):
        run_prepare_ace(_ace_args(tmp_path / "no_atlas", conn_dir, cache_dir))


def test_prepare_ace_empty_connectome_raises(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas"
    save_atlas(atlas, atlas_dir)

    empty_conn = tmp_path / "empty_conn"
    empty_conn.mkdir()
    cache_dir = tmp_path / "ace_cache"
    with pytest.raises(FileNotFoundError):
        run_prepare_ace(_ace_args(atlas_dir, empty_conn, cache_dir))


def test_prepare_ace_affine_mismatch_raises(tmp_path):
    """Same shape but mismatched affine → fail before running ACE."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas"
    save_atlas(atlas, atlas_dir)

    conn_dir = tmp_path / "conn"
    conn_dir.mkdir()
    atlas_ref = atlas.get_map(atlas.targets[0])
    # Same shape, different affine (atlas was built with np.eye(4)*2):
    _write_synthetic_gsp_batch(
        conn_dir / "batch_0001.h5",
        n_subjects=2,
        n_timepoints=8,
        atlas_shape=atlas_ref.shape,
        atlas_affine=np.eye(4),
    )
    cache_dir = tmp_path / "ace_cache"

    with pytest.raises(ValueError, match="affine"):
        run_prepare_ace(_ace_args(atlas_dir, conn_dir, cache_dir))
