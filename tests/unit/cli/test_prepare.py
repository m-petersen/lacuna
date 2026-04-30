"""Unit tests for lacuna.cli.prepare."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.envelope import (
    AssetType,
    ENVELOPE_FILENAME,
    read_envelope,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.cli.prepare import _precompute_endpoint_weights
from lacuna.data.ntatlas import load_collection


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


def _write_synthetic_tck(path: Path, n: int) -> None:
    streamlines = nib.streamlines.ArraySequence(
        [np.zeros((2, 3), dtype=np.float32) for _ in range(n)]
    )
    nib.streamlines.save(
        nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)),
        str(path),
    )


def test_precompute_endpoint_weights_writes_envelope(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    atlas_dir = tmp_path / "atlas"
    save_atlas(atlas, atlas_dir)

    tck = tmp_path / "t.tck"
    _write_synthetic_tck(tck, 10)

    cache_dir = tmp_path / "sntf_cache"

    # Stub tckresample so we don't need MRtrix3 in tests.
    def _fake_run(cmd, **_):
        out = cmd[2]  # tckresample IN OUT -endpoints -force
        _write_synthetic_tck(Path(out), 10)

    with patch("lacuna.utils.mrtrix.run_mrtrix_command", side_effect=_fake_run):
        _precompute_endpoint_weights(atlas, atlas_dir, tck, cache_dir)

    assert (cache_dir / ENVELOPE_FILENAME).exists()
    assert not (cache_dir / "connectome_meta.json").exists()

    env = read_envelope(cache_dir)
    assert env.asset_type == AssetType.SNTF_CACHE

    roles = {r.role for r in env.requires}
    assert roles == {"tractogram", "ntatlas"}

    tract_req = next(r for r in env.requires if r.role == "tractogram")
    assert tract_req.asset_type == AssetType.STRUCTURAL_CONNECTOME
    assert "sha256_first_mib" in tract_req.identity.fields

    atlas_req = next(r for r in env.requires if r.role == "ntatlas")
    assert atlas_req.asset_type == AssetType.NTATLAS
    assert atlas_req.identity.fields["n_targets"] == 2
