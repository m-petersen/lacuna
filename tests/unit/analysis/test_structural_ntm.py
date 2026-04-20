"""Tests for StructuralNeurotransmitterMapping analysis."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.structural_neurotransmitter_mapping import (
    StructuralNeurotransmitterMapping,
)
from lacuna.assets.connectomes import (
    register_structural_connectome,
    unregister_structural_connectome,
)
from lacuna.atlas.store import build_nt_atlas, save_atlas
from lacuna.core.data_types import ScalarMetric


def _create_pet_map(tmp_path, target, tracer, pub, shape=(91, 109, 91)):
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(hash((target, tracer, pub)) % 2**32)
    data = rng.random(shape).astype(np.float32) + 0.1
    fname = f"target-{target}_tracer-{tracer}_n-10_dx-hc_pub-{pub}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    path = tmp_path / fname
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture
def atlas_cache(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _create_pet_map(pet_dir, "D1", "sch23390", "kaller2017")
    _create_pet_map(pet_dir, "5HT1a", "cumi101", "beliveau2017")
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


@pytest.fixture
def fake_connectome(tmp_path):
    """Register a fake structural connectome for testing."""
    tck_path = tmp_path / "fake.tck"
    tck_path.touch()
    name = "test_sntm_connectome"
    register_structural_connectome(
        name=name,
        space="MNI152NLin2009cAsym",
        tractogram_path=tck_path,
        description="Fake connectome for sntm testing",
    )
    yield name
    unregister_structural_connectome(name)


class TestStructuralNTMConstruction:
    def test_basic_construction(self, atlas_cache, fake_connectome):
        sntm = StructuralNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name=fake_connectome,
            check_dependencies=False,
        )
        assert sntm._target_spec == "all"

    def test_targets_parameter(self, atlas_cache, fake_connectome):
        sntm = StructuralNeurotransmitterMapping(
            atlas_cache_dir=atlas_cache,
            connectome_name=fake_connectome,
            targets=["D1"],
            check_dependencies=False,
        )
        assert sntm._target_spec == ["D1"]
