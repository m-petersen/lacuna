"""Integration test: full lntf pipeline from raw PET maps to scores."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis import LocalNeurotransmitterFingerprinting
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import LabeledScalars
from lacuna.core.subject_data import SubjectData
from lacuna.data.ntatlas import load_collection


def _lntf_fingerprint(result):
    bag = result.results["LocalNeurotransmitterFingerprinting"]
    fp = next(iter(bag.values()))
    assert isinstance(fp, LabeledScalars)
    return fp


def _write_synthetic_collection(src, targets, shape=(91, 109, 91)):
    """Write synthetic NIfTIs for the given targets using collection map IDs."""
    coll = load_collection()
    target_to_map_id = {
        map_id.split("_", 1)[0][len("target-"):]: map_id
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
    """Build a cached NT atlas from synthetic PET maps."""
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])

    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


@pytest.mark.integration
class TestLNTFIntegration:
    def test_full_pipeline(self, pet_atlas):
        """Test: raw PET maps -> build atlas -> save -> load -> score lesion."""
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        subject = _make_subject(affine)

        lntf = LocalNeurotransmitterFingerprinting(ntatlas_dir=pet_atlas)
        parcel = _lntf_fingerprint(lntf.run(subject))
        assert "D1" in parcel.data
        assert "5HT1a" in parcel.data
        assert np.isfinite(parcel.data["D1"])

    def test_target_subsetting(self, pet_atlas):
        """Test that targets= filters output correctly."""
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        subject = _make_subject(affine)

        lntf = LocalNeurotransmitterFingerprinting(
            ntatlas_dir=pet_atlas,
            targets=["D1"],
        )
        parcel = _lntf_fingerprint(lntf.run(subject))
        assert "D1" in parcel.data
        assert "5HT1a" not in parcel.data

    def test_pipeline_chaining(self, pet_atlas):
        """Test lntf works in a Pipeline with other analyses."""
        from lacuna.core.pipeline import Pipeline

        affine = np.eye(4) * 2
        affine[3, 3] = 1
        subject = _make_subject(affine)

        pipe = Pipeline(name="test_ntf")
        pipe.add(LocalNeurotransmitterFingerprinting(ntatlas_dir=pet_atlas))
        result = pipe.run(subject)

        assert "LocalNeurotransmitterFingerprinting" in result.results
