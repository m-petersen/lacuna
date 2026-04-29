"""Tests for LocalNeurotransmitterFingerprinting analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.local_neurotransmitter_fingerprinting import (
    LocalNeurotransmitterFingerprinting,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import ParcelData
from lacuna.core.subject_data import SubjectData
from lacuna.data.ntatlas import load_collection


def _lntf_parcel(result) -> ParcelData:
    """Extract the single ParcelData produced by lntf from a Result."""
    bag = result.results["LocalNeurotransmitterFingerprinting"]
    assert len(bag) == 1
    parcel = next(iter(bag.values()))
    assert isinstance(parcel, ParcelData)
    return parcel


def _write_synthetic_collection(src, targets):
    """Write synthetic NIfTI files for the given target names using collection map IDs."""
    coll = load_collection()
    target_to_map_id = {}
    for system_targets in coll["systems"].values():
        for map_id in system_targets:
            t = map_id.split("_", 1)[0][len("target-"):]
            target_to_map_id[t] = map_id

    affine = np.eye(4) * 2
    affine[3, 3] = 1
    shape = (91, 109, 91)
    for target in targets:
        map_id = target_to_map_id[target]
        rng = np.random.default_rng(hash(map_id) % 2**32)
        data = rng.random(shape).astype(np.float32) + 0.1
        fname = f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(src / fname))


@pytest.fixture
def atlas_cache(tmp_path):
    """Build and cache a small NT atlas with D1 and 5HT1a."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


@pytest.fixture
def lesion_subject():
    """A SubjectData with a small lesion in MNI152NLin6Asym 2mm."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    mask_data = np.zeros((91, 109, 91), dtype=np.int8)
    mask_data[45:50, 55:60, 45:50] = 1  # small lesion
    mask_img = nib.Nifti1Image(mask_data, affine)
    return SubjectData(
        mask_img=mask_img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": "sub-001"},
    )


class TestLocalNTFConstruction:
    def test_basic_construction(self, atlas_cache):
        lntf = LocalNeurotransmitterFingerprinting(atlas_cache_dir=atlas_cache)
        assert lntf.TARGET_SPACE is None

    def test_targets_parameter(self, atlas_cache):
        lntf = LocalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            targets=["D1"],
        )
        assert lntf._target_spec == ["D1"]


class TestLocalNTFRun:
    def test_returns_parcel_data(self, atlas_cache, lesion_subject):
        lntf = LocalNeurotransmitterFingerprinting(atlas_cache_dir=atlas_cache)
        parcel = _lntf_parcel(lntf.run(lesion_subject))
        assert "D1" in parcel.data
        assert "5HT1a" in parcel.data

    def test_scores_are_finite(self, atlas_cache, lesion_subject):
        lntf = LocalNeurotransmitterFingerprinting(atlas_cache_dir=atlas_cache)
        parcel = _lntf_parcel(lntf.run(lesion_subject))
        for target in ["D1", "5HT1a"]:
            assert np.isfinite(parcel.data[target])

    def test_target_subsetting(self, atlas_cache, lesion_subject):
        lntf = LocalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            targets=["D1"],
        )
        parcel = _lntf_parcel(lntf.run(lesion_subject))
        assert "D1" in parcel.data
        assert "5HT1a" not in parcel.data
