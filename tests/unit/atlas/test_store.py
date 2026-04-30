"""Tests for lacuna.atlas.store: build, save, and load NT atlas."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.envelope import AssetType, ENVELOPE_FILENAME, read_envelope
from lacuna.atlas.store import (
    _zscore_excluding_zeros,
    build_nt_atlas,
    load_atlas,
    save_atlas,
)
from lacuna.data.ntatlas import all_map_ids, load_collection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_map(
    src: Path,
    map_id: str,
    shape: tuple[int, int, int] = (10, 12, 10),
    add_zeros: bool = False,
) -> Path:
    """Write a synthetic PET NIfTI matching the expected fetch_ntatlas filename."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash(map_id) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    if add_zeros:
        data[:, :, : shape[2] // 4] = 0.0
    out = src / f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(out))
    return out


def _populate_full_collection(src: Path) -> list[str]:
    """Write a synthetic NIfTI for every map in the bundled collection."""
    src.mkdir(parents=True, exist_ok=True)
    ids = all_map_ids()
    for mid in ids:
        _write_map(src, mid)
    return ids


# ---------------------------------------------------------------------------
# _zscore_excluding_zeros
# ---------------------------------------------------------------------------


class TestZscoreExcludingZeros:
    def test_zero_voxels_stay_zero(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = _zscore_excluding_zeros(data)
        assert result[0] == 0.0

    def test_nonzero_voxels_are_zscored(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = _zscore_excluding_zeros(data)
        nonzero = data[data != 0]
        expected = (nonzero - nonzero.mean()) / nonzero.std()
        np.testing.assert_allclose(result[1:], expected, rtol=1e-5)

    def test_constant_nonzero_returns_zeros(self):
        data = np.array([0.0, 3.0, 3.0, 3.0])
        result = _zscore_excluding_zeros(data)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_all_zeros_returns_zeros(self):
        data = np.zeros(5)
        result = _zscore_excluding_zeros(data)
        np.testing.assert_array_equal(result, np.zeros(5))


# ---------------------------------------------------------------------------
# build_nt_atlas
# ---------------------------------------------------------------------------


class TestBuildNtAtlas:
    def test_full_collection(self, tmp_path):
        ids = _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        assert len(atlas.targets) == len(ids)
        # Every target appears exactly once
        assert len(set(atlas.targets)) == len(atlas.targets)

    def test_targets_match_collection(self, tmp_path):
        _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        # Must include some well-known representative targets
        for expected in ("D1", "DAT", "5HT1a", "MOR", "GABAa"):
            assert expected in atlas.targets

    def test_output_is_zscored(self, tmp_path):
        _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        data = np.asarray(atlas.get_map("D1").dataobj)
        nonzero = data[data != 0]
        np.testing.assert_allclose(nonzero.mean(), 0.0, atol=1e-4)
        np.testing.assert_allclose(nonzero.std(), 1.0, atol=1e-4)

    def test_metadata_has_systems(self, tmp_path):
        _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        systems = atlas.metadata["systems"]
        assert "Dopamine" in systems
        assert "D1" in systems["Dopamine"]
        assert "DAT" in systems["Dopamine"]

    def test_metadata_has_commit_pin(self, tmp_path):
        _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        coll = load_collection()
        assert atlas.metadata["nispace_commit"] == coll["nispace_commit"]
        assert atlas.metadata["collection"] == coll["collection_name"]

    def test_metadata_target_to_map_id(self, tmp_path):
        _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        mapping = atlas.metadata["target_to_map_id"]
        assert mapping["D1"].startswith("target-D1_tracer-sch23390")

    def test_correct_space_and_domain(self, tmp_path):
        _populate_full_collection(tmp_path)
        atlas = build_nt_atlas(tmp_path)
        assert atlas.space == "MNI152NLin6Asym"
        assert atlas.domain == "neurotransmitter"

    def test_partial_dir_skips_missing(self, tmp_path):
        # Only write the Dopamine system maps
        coll = load_collection()
        for mid in coll["systems"]["Dopamine"]:
            _write_map(tmp_path, mid)
        atlas = build_nt_atlas(tmp_path)
        # Should have all 4 dopamine targets, no others
        assert set(atlas.targets) == {"FDOPA", "D1", "D23", "DAT"}
        # Systems metadata only contains loaded systems
        assert atlas.metadata["systems"] == {
            "Dopamine": ["FDOPA", "D1", "D23", "DAT"]
        }

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No PET maps found"):
            build_nt_atlas(tmp_path)


# ---------------------------------------------------------------------------
# save_atlas / load_atlas roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadAtlas:
    @pytest.fixture
    def small_atlas_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        coll = load_collection()
        for mid in coll["systems"]["Dopamine"]:
            _write_map(src, mid)
        return src

    def test_roundtrip_targets_match(self, small_atlas_dir, tmp_path):
        atlas = build_nt_atlas(small_atlas_dir)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        loaded = load_atlas(cache)
        assert loaded.targets == atlas.targets

    def test_roundtrip_values_match(self, small_atlas_dir, tmp_path):
        atlas = build_nt_atlas(small_atlas_dir)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        loaded = load_atlas(cache)
        for target in atlas.targets:
            orig = np.asarray(atlas.get_map(target).dataobj)
            reloaded = np.asarray(loaded.get_map(target).dataobj)
            np.testing.assert_allclose(orig, reloaded, rtol=1e-5)

    def test_roundtrip_metadata(self, small_atlas_dir, tmp_path):
        atlas = build_nt_atlas(small_atlas_dir)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        loaded = load_atlas(cache)
        assert loaded.space == atlas.space
        assert loaded.domain == atlas.domain
        assert loaded.metadata["systems"] == atlas.metadata["systems"]

    def test_manifest_json_created(self, small_atlas_dir, tmp_path):
        atlas = build_nt_atlas(small_atlas_dir)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        assert (cache / "manifest.json").exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_atlas(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# save_atlas writes the lacuna_asset.json envelope
# ---------------------------------------------------------------------------


def _toy_atlas():
    """A tiny in-memory atlas for envelope-write tests."""
    affine = np.eye(4)
    maps = {
        "A": nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine),
        "B": nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), affine),
    }
    from lacuna.atlas.types import VoxelAtlas
    return VoxelAtlas(
        maps=maps,
        space="MNI152NLin6Asym",
        resolution=2.0,
        domain="neurotransmitter",
        metadata={"systems": {"toy": ["A", "B"]}},
    )


def test_save_atlas_writes_envelope(tmp_path):
    atlas = _toy_atlas()
    save_atlas(atlas, tmp_path)
    assert (tmp_path / ENVELOPE_FILENAME).exists()
    env = read_envelope(tmp_path)
    assert env.asset_type == AssetType.NTATLAS
    assert env.data["targets"] == ["A", "B"]
    assert env.data["space"] == "MNI152NLin6Asym"
    assert env.identity.fields["n_targets"] == 2
