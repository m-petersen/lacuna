"""Tests for LocalNeurotransmitterFingerprinting analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.local_neurotransmitter_fingerprinting import LocalNeurotransmitterFingerprinting
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.atlas.types import VoxelAtlas
from lacuna.core.data_types import ScalarMetric
from lacuna.core.subject_data import SubjectData


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.fixture
def atlas_cache(tmp_path):
    """Build and cache a small NT atlas."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
    _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
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


class TestLocalNTMConstruction:
    def test_basic_construction(self, atlas_cache):
        lntf = LocalNeurotransmitterFingerprinting(atlas_cache_dir=atlas_cache)
        assert lntf.TARGET_SPACE is None

    def test_targets_parameter(self, atlas_cache):
        lntf = LocalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            targets=["D1"],
        )
        assert lntf._target_spec == ["D1"]


class TestLocalNTMRun:
    def test_produces_scalar_metrics(self, atlas_cache, lesion_subject):
        lntf = LocalNeurotransmitterFingerprinting(atlas_cache_dir=atlas_cache)
        result = lntf.run(lesion_subject)
        lntm_results = result.results["LocalNeurotransmitterFingerprinting"]
        assert "D1" in lntm_results
        assert "5HT1a" in lntm_results
        assert isinstance(lntm_results["D1"], ScalarMetric)

    def test_scores_are_finite(self, atlas_cache, lesion_subject):
        lntf = LocalNeurotransmitterFingerprinting(atlas_cache_dir=atlas_cache)
        result = lntf.run(lesion_subject)
        lntm_results = result.results["LocalNeurotransmitterFingerprinting"]
        for target in ["D1", "5HT1a"]:
            score = lntm_results[target].get_data()
            assert np.isfinite(score)

    def test_target_subsetting(self, atlas_cache, lesion_subject):
        lntf = LocalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            targets=["D1"],
        )
        result = lntf.run(lesion_subject)
        lntm_results = result.results["LocalNeurotransmitterFingerprinting"]
        assert "D1" in lntm_results
        assert "5HT1a" not in lntm_results
