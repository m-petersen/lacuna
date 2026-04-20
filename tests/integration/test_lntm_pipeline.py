"""Integration test: full lntm pipeline from raw PET maps to scores."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis import LocalNeurotransmitterMapping
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import ScalarMetric
from lacuna.core.subject_data import SubjectData


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = (
        f"target-{target}_tracer-{tracer}_n-10_dx-hc"
        f"_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    )
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_subject(affine, shape=(91, 109, 91)):
    mask_data = np.zeros(shape, dtype=np.int8)
    mask_data[40:50, 50:60, 40:50] = 1
    return SubjectData(
        mask_img=nib.Nifti1Image(mask_data, affine),
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": "sub-001"},
    )


@pytest.fixture()
def pet_atlas(tmp_path):
    """Build a cached NT atlas from fake PET maps."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
    _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
    _create_pet_map(pet_dir, "5HT1a", "way100635", "savli2012")

    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


@pytest.mark.integration
class TestLNTMIntegration:
    def test_full_pipeline(self, pet_atlas):
        """Test: raw PET maps -> build atlas -> save -> load -> score lesion."""
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        subject = _make_subject(affine)

        lntm = LocalNeurotransmitterMapping(atlas_cache_dir=pet_atlas)
        result = lntm.run(subject)

        lntm_results = result.results["LocalNeurotransmitterMapping"]
        assert "D1" in lntm_results
        assert "5HT1a" in lntm_results
        assert isinstance(lntm_results["D1"], ScalarMetric)
        assert np.isfinite(lntm_results["D1"].data)

    def test_target_subsetting(self, pet_atlas):
        """Test that targets= filters output correctly."""
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        subject = _make_subject(affine)

        lntm = LocalNeurotransmitterMapping(
            atlas_cache_dir=pet_atlas,
            targets=["D1"],
        )
        result = lntm.run(subject)
        lntm_results = result.results["LocalNeurotransmitterMapping"]
        assert "D1" in lntm_results
        assert "5HT1a" not in lntm_results

    def test_pipeline_chaining(self, pet_atlas):
        """Test lntm works in a Pipeline with other analyses."""
        from lacuna.core.pipeline import Pipeline

        affine = np.eye(4) * 2
        affine[3, 3] = 1
        subject = _make_subject(affine)

        pipe = Pipeline(name="test_ntm")
        pipe.add(LocalNeurotransmitterMapping(atlas_cache_dir=pet_atlas))
        result = pipe.run(subject)

        assert "LocalNeurotransmitterMapping" in result.results
