"""Tests for REACT stage 1 and stage 2 implementation."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.react import (
    compute_react_atlas,
    compute_stage1_mask,
    react_stage1,
    react_stage2,
)
from lacuna.atlas.types import VoxelAtlas


@pytest.fixture
def small_atlas():
    """Small atlas for fast REACT tests."""
    shape = (10, 10, 10)
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    rng = np.random.default_rng(42)
    maps = {}
    for target in ["D1", "5HT1a"]:
        data = rng.random(shape).astype(np.float32)
        maps[target] = nib.Nifti1Image(data, affine)
    return VoxelAtlas(
        maps=maps, space="MNI152NLin6Asym", resolution=2.0, domain="neurotransmitter"
    )


@pytest.fixture
def fake_bold_subjects():
    """Two fake fMRI subjects: (n_timepoints, n_voxels_flat)."""
    rng = np.random.default_rng(42)
    n_timepoints = 50
    n_voxels = 1000
    return [
        rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        for _ in range(2)
    ]


class TestComputeStage1Mask:
    def test_intersection_of_nonzero(self, small_atlas):
        mask = compute_stage1_mask(small_atlas)
        assert mask.shape == (10, 10, 10)
        assert mask.dtype == bool
        assert mask.sum() > 0


class TestReactStage1:
    def test_output_shape(self, small_atlas, fake_bold_subjects):
        n_timepoints = 50
        n_voxels = 1000
        stage1_mask = np.ones(n_voxels, dtype=bool)
        atlas_matrix = (
            np.random.default_rng(42).standard_normal((2, n_voxels)).astype(np.float32)
        )

        beta1 = react_stage1(fake_bold_subjects[0], atlas_matrix, stage1_mask)
        assert beta1.shape == (n_timepoints, 2)

    def test_output_not_all_zero(self, small_atlas, fake_bold_subjects):
        n_voxels = 1000
        stage1_mask = np.ones(n_voxels, dtype=bool)
        atlas_matrix = (
            np.random.default_rng(42).standard_normal((2, n_voxels)).astype(np.float32)
        )
        beta1 = react_stage1(fake_bold_subjects[0], atlas_matrix, stage1_mask)
        assert not np.allclose(beta1, 0)


class TestReactStage2:
    def test_output_shape(self):
        n_timepoints = 50
        n_voxels = 1000
        n_targets = 2
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        beta1 = rng.standard_normal((n_timepoints, n_targets)).astype(np.float32)
        stage2_mask = np.ones(n_voxels, dtype=bool)

        beta2 = react_stage2(bold, beta1, stage2_mask)
        assert beta2.shape == (n_voxels, n_targets)

    def test_output_not_all_zero(self):
        n_timepoints = 50
        n_voxels = 1000
        n_targets = 2
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        beta1 = rng.standard_normal((n_timepoints, n_targets)).astype(np.float32)
        stage2_mask = np.ones(n_voxels, dtype=bool)
        beta2 = react_stage2(bold, beta1, stage2_mask)
        assert not np.allclose(beta2, 0)


class TestComputeReactAtlas:
    def test_produces_voxel_atlas(self, small_atlas):
        """Test full REACT pipeline with synthetic data."""
        shape = (10, 10, 10)
        n_voxels = np.prod(shape)
        n_timepoints = 50
        rng = np.random.default_rng(42)

        subjects_data = [
            rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
            for _ in range(3)
        ]

        brain_mask = np.ones(shape, dtype=bool)

        result = compute_react_atlas(
            atlas=small_atlas,
            subjects_data=subjects_data,
            brain_mask=brain_mask,
            mask_shape=shape,
        )
        assert isinstance(result["stage2_atlas"], VoxelAtlas)
        assert set(result["stage2_atlas"].targets) == {"5HT1a", "D1"}
        assert "stage1_timeseries" in result
        assert len(result["stage1_timeseries"]) == 3
