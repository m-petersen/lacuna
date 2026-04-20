"""Tests for lacuna.atlas.store: build, save, and load NT atlas."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.store import (
    _average_excluding_zeros,
    _zscore_excluding_zeros,
    build_nt_atlas,
    load_atlas,
    save_atlas,
)
from lacuna.atlas.types import VoxelAtlas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_pet_map(
    tmp_path: Path,
    target: str,
    tracer: str,
    pub: str,
    shape: tuple[int, int, int] = (10, 12, 10),
    add_zeros: bool = False,
) -> Path:
    """Create a synthetic PET NIfTI map with BIDS-like filename."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    if add_zeros:
        data[:, :, : shape[2] // 4] = 0.0
    fname = (
        f"target-{target}_tracer-{tracer}_n-10_dx-hc"
        f"_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    )
    out = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(out))
    return out


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestAverageExcludingZeros:
    def test_basic_average(self):
        a = np.array([1.0, 2.0, 0.0])
        b = np.array([3.0, 0.0, 0.0])
        result = _average_excluding_zeros([a, b])
        # position 0: (1+3)/2 = 2, position 1: 2/1 = 2, position 2: all-zero -> 0
        np.testing.assert_allclose(result, [2.0, 2.0, 0.0])

    def test_single_array(self):
        a = np.array([0.0, 5.0, 3.0])
        result = _average_excluding_zeros([a])
        np.testing.assert_array_equal(result, a)

    def test_all_zeros_stays_zero(self):
        a = np.zeros(5)
        b = np.zeros(5)
        result = _average_excluding_zeros([a, b])
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_shape_preserved(self):
        shape = (4, 5, 6)
        arrays = [np.ones(shape) * float(i) for i in range(1, 4)]
        result = _average_excluding_zeros(arrays)
        assert result.shape == shape


class TestZscoreExcludingZeros:
    def test_zero_voxels_stay_zero(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = _zscore_excluding_zeros(data)
        assert result[0] == 0.0

    def test_nonzero_voxels_are_zscored(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = _zscore_excluding_zeros(data)
        nonzero = data[data != 0]
        expected_mean = nonzero.mean()
        expected_std = nonzero.std()
        expected = (nonzero - expected_mean) / expected_std
        np.testing.assert_allclose(result[1:], expected, rtol=1e-5)

    def test_constant_nonzero_returns_zeros(self):
        # std=0 => all z-scores are 0
        data = np.array([0.0, 3.0, 3.0, 3.0])
        result = _zscore_excluding_zeros(data)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_all_zeros_returns_zeros(self):
        data = np.zeros(5)
        result = _zscore_excluding_zeros(data)
        np.testing.assert_array_equal(result, np.zeros(5))


# ---------------------------------------------------------------------------
# build_nt_atlas tests
# ---------------------------------------------------------------------------


class TestBuildNtAtlas:
    def test_single_target_single_map(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        atlas = build_nt_atlas(tmp_path)
        assert "5HT1A" in atlas.targets

    def test_groups_by_target(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        _create_pet_map(tmp_path, "DAT", "FP-CIT", "publicb")
        atlas = build_nt_atlas(tmp_path)
        assert set(atlas.targets) == {"5HT1A", "DAT"}

    def test_averages_multiple_maps(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        _create_pet_map(tmp_path, "5HT1A", "CUMI101", "publicb")
        atlas = build_nt_atlas(tmp_path)
        # Should produce one map for 5HT1A
        assert "5HT1A" in atlas.targets
        img = atlas.get_map("5HT1A")
        assert img is not None

    def test_zeros_excluded_from_average(self, tmp_path):
        """A map with zeros in a region should contribute only where nonzero."""
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica", add_zeros=True)
        _create_pet_map(tmp_path, "5HT1A", "CUMI101", "publicb")
        atlas = build_nt_atlas(tmp_path)
        # Should not raise; atlas is well-defined
        img = atlas.get_map("5HT1A")
        assert img is not None

    def test_output_is_zscored(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        _create_pet_map(tmp_path, "5HT1A", "CUMI101", "publicb")
        atlas = build_nt_atlas(tmp_path)
        data = np.asarray(atlas.get_map("5HT1A").dataobj)
        nonzero = data[data != 0]
        # z-scored: mean ~0, std ~1
        np.testing.assert_allclose(nonzero.mean(), 0.0, atol=1e-4)
        np.testing.assert_allclose(nonzero.std(), 1.0, atol=1e-4)

    def test_correct_space_and_domain(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        atlas = build_nt_atlas(tmp_path)
        assert atlas.space == "MNI152NLin6Asym"
        assert atlas.domain == "neurotransmitter"

    def test_map_config_exclude_target(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        _create_pet_map(tmp_path, "DAT", "FP-CIT", "publicb")
        atlas = build_nt_atlas(tmp_path, map_config={"exclude": ["DAT"]})
        assert "DAT" not in atlas.targets
        assert "5HT1A" in atlas.targets

    def test_map_config_select_publication(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        _create_pet_map(tmp_path, "5HT1A", "CUMI101", "publicb")
        _create_pet_map(tmp_path, "DAT", "FP-CIT", "publica")
        _create_pet_map(tmp_path, "DAT", "DATscan", "publicb")
        # Only keep pub "publica"
        atlas = build_nt_atlas(tmp_path, map_config={"publications": ["publica"]})
        # Both targets should still appear (each has a publica map)
        assert "5HT1A" in atlas.targets
        assert "DAT" in atlas.targets

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No.*nii.gz"):
            build_nt_atlas(tmp_path)

    def test_all_excluded_raises(self, tmp_path):
        _create_pet_map(tmp_path, "5HT1A", "WAY100635", "publica")
        with pytest.raises(ValueError):
            build_nt_atlas(tmp_path, map_config={"exclude": ["5HT1A"]})


# ---------------------------------------------------------------------------
# save_atlas / load_atlas roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadAtlas:
    def _build_simple_atlas(self, tmp_path: Path) -> VoxelAtlas:
        src = tmp_path / "src"
        src.mkdir()
        _create_pet_map(src, "5HT1A", "WAY100635", "publica")
        _create_pet_map(src, "DAT", "FP-CIT", "publicb")
        return build_nt_atlas(src)

    def test_roundtrip_targets_match(self, tmp_path):
        atlas = self._build_simple_atlas(tmp_path)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        loaded = load_atlas(cache)
        assert loaded.targets == atlas.targets

    def test_roundtrip_values_match(self, tmp_path):
        atlas = self._build_simple_atlas(tmp_path)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        loaded = load_atlas(cache)
        for target in atlas.targets:
            orig = np.asarray(atlas.get_map(target).dataobj)
            reloaded = np.asarray(loaded.get_map(target).dataobj)
            np.testing.assert_allclose(orig, reloaded, rtol=1e-5)

    def test_roundtrip_space_domain(self, tmp_path):
        atlas = self._build_simple_atlas(tmp_path)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        loaded = load_atlas(cache)
        assert loaded.space == atlas.space
        assert loaded.domain == atlas.domain

    def test_manifest_json_created(self, tmp_path):
        atlas = self._build_simple_atlas(tmp_path)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        assert (cache / "manifest.json").exists()

    def test_nifti_files_created(self, tmp_path):
        atlas = self._build_simple_atlas(tmp_path)
        cache = tmp_path / "cache"
        save_atlas(atlas, cache)
        maps_dir = cache / "maps"
        for target in atlas.targets:
            assert (maps_dir / f"{target}.nii.gz").exists()

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_atlas(tmp_path / "nonexistent")

    def test_load_missing_manifest_raises(self, tmp_path):
        cache = tmp_path / "cache"
        cache.mkdir()
        with pytest.raises(FileNotFoundError):
            load_atlas(cache)
