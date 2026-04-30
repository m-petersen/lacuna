"""Verify --export-provenance toggles the per-subject sidecar."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.data.ntatlas import load_collection


def _write_synthetic_collection(src, targets, shape=(91, 109, 91)):
    coll = load_collection()
    target_to_map_id = {
        map_id.split("_", 1)[0][len("target-") :]: map_id
        for ids in coll["systems"].values()
        for map_id in ids
    }
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    for target in targets:
        map_id = target_to_map_id[target]
        rng = np.random.default_rng(hash(map_id) % 2**32)
        data = rng.random(shape).astype(np.float32) + 0.1
        fname = f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(src / fname))


@pytest.fixture
def lntf_setup(tmp_path, simple_bids_dataset):
    """Provide (bids_dir, atlas_dir) for the lntf CLI."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas_cache"
    save_atlas(atlas, atlas_dir)
    return simple_bids_dataset, atlas_dir


def _invoke(bids_dir, out_dir, atlas_dir, *extra_args):
    from lacuna.cli import main

    return main(
        [
            "run",
            "lntf",
            str(bids_dir),
            str(out_dir),
            "--ntatlas-dir",
            str(atlas_dir),
            "--mask-space",
            "MNI152NLin6Asym",
            *extra_args,
        ],
    )


@pytest.mark.integration
def test_export_provenance_off_by_default(lntf_setup, tmp_path):
    bids, atlas = lntf_setup
    out = tmp_path / "out"
    rc = _invoke(bids, out, atlas)
    assert rc == 0, f"CLI exited with {rc}"
    assert not list(out.rglob("*_desc-provenance.json")), (
        "Found a provenance sidecar despite the flag being off"
    )


@pytest.mark.integration
def test_export_provenance_writes_sidecar_when_enabled(lntf_setup, tmp_path):
    bids, atlas = lntf_setup
    out = tmp_path / "out"
    rc = _invoke(bids, out, atlas, "--export-provenance")
    assert rc == 0, f"CLI exited with {rc}"
    sidecars = list(out.rglob("*_desc-provenance.json"))
    assert sidecars, "Expected a *_desc-provenance.json sidecar; found none"
