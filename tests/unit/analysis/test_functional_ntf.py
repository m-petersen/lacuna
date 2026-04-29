"""Tests for FunctionalNeurotransmitterFingerprinting analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.functional_neurotransmitter_fingerprinting import (
    FunctionalNeurotransmitterFingerprinting,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import ScalarMetric
from lacuna.data.ntatlas import load_collection


def _write_synthetic_collection(src, targets):
    """Write synthetic NIfTIs for the given targets using collection map IDs."""
    coll = load_collection()
    target_to_map_id = {
        map_id.split("_", 1)[0][len("target-"):]: map_id
        for ids in coll["systems"].values()
        for map_id in ids
    }
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
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


class TestFunctionalNTMConstruction:
    def test_basic_construction(self, atlas_cache):
        fntf = FunctionalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            connectome_name="GSP1000",
        )
        assert fntf._target_spec == "all"
        assert fntf.enriched is False

    def test_enriched_parameter(self, atlas_cache):
        fntf = FunctionalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            connectome_name="GSP1000",
            enriched=True,
        )
        assert fntf.enriched is True

    def test_targets_parameter(self, atlas_cache):
        fntf = FunctionalNeurotransmitterFingerprinting(
            atlas_cache_dir=atlas_cache,
            connectome_name="GSP1000",
            targets=["D1"],
        )
        assert fntf._target_spec == ["D1"]
