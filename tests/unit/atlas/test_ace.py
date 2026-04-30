"""Tests for ACE (Atlas Connectivity Enrichment) stage 1 and stage 2."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.ace import (
    ace_stage1,
    ace_stage2,
    compute_ace_atlas,
    compute_stage1_mask,
)
from lacuna.atlas.types import VoxelAtlas


@pytest.fixture
def small_atlas():
    """Small atlas for fast ACE tests."""
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
        assert np.sum(mask) > 0


class TestAceStage1:
    def test_output_shape(self, small_atlas, fake_bold_subjects):
        n_timepoints = 50
        n_voxels = 1000
        stage1_mask = np.ones(n_voxels, dtype=bool)
        atlas_matrix = (
            np.random.default_rng(42).standard_normal((2, n_voxels)).astype(np.float32)
        )

        beta1 = ace_stage1(fake_bold_subjects[0], atlas_matrix, stage1_mask)
        assert beta1.shape == (n_timepoints, 2)

    def test_output_not_all_zero(self, small_atlas, fake_bold_subjects):
        n_voxels = 1000
        stage1_mask = np.ones(n_voxels, dtype=bool)
        atlas_matrix = (
            np.random.default_rng(42).standard_normal((2, n_voxels)).astype(np.float32)
        )
        beta1 = ace_stage1(fake_bold_subjects[0], atlas_matrix, stage1_mask)
        assert not np.allclose(beta1, 0)


class TestAceStage2:
    def test_output_shape(self):
        n_timepoints = 50
        n_voxels = 1000
        n_targets = 2
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        beta1 = rng.standard_normal((n_timepoints, n_targets)).astype(np.float32)
        stage2_mask = np.ones(n_voxels, dtype=bool)

        beta2 = ace_stage2(bold, beta1, stage2_mask)
        assert beta2.shape == (n_voxels, n_targets)

    def test_output_not_all_zero(self):
        n_timepoints = 50
        n_voxels = 1000
        n_targets = 2
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
        beta1 = rng.standard_normal((n_timepoints, n_targets)).astype(np.float32)
        stage2_mask = np.ones(n_voxels, dtype=bool)
        beta2 = ace_stage2(bold, beta1, stage2_mask)
        assert not np.allclose(beta2, 0)


class TestComputeAceAtlas:
    def test_produces_voxel_atlas(self, small_atlas):
        """Test full ACE pipeline with synthetic data."""
        shape = (10, 10, 10)
        n_voxels = np.prod(shape)
        n_timepoints = 50
        rng = np.random.default_rng(42)

        subjects_data = [
            rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
            for _ in range(3)
        ]

        brain_mask = np.ones(shape, dtype=bool)

        stage2_atlas = compute_ace_atlas(
            atlas=small_atlas,
            subjects=subjects_data,
            n_subjects=len(subjects_data),
            brain_mask=brain_mask,
            mask_shape=shape,
        )
        assert isinstance(stage2_atlas, VoxelAtlas)
        assert set(stage2_atlas.targets) == {"5HT1a", "D1"}

    def test_streaming_callback_receives_each_stage1(self, small_atlas):
        """on_subject_done is called once per subject with (index, beta1)."""
        shape = (10, 10, 10)
        n_voxels = np.prod(shape)
        n_timepoints = 50
        rng = np.random.default_rng(42)
        subjects_data = [
            rng.standard_normal((n_timepoints, n_voxels)).astype(np.float32)
            for _ in range(3)
        ]
        brain_mask = np.ones(shape, dtype=bool)
        seen: list[tuple[int, tuple[int, ...]]] = []

        def cb(i, beta1):
            seen.append((i, tuple(beta1.shape)))

        stage2_atlas = compute_ace_atlas(
            atlas=small_atlas,
            subjects=subjects_data,
            n_subjects=len(subjects_data),
            brain_mask=brain_mask,
            mask_shape=shape,
            on_subject_done=cb,
        )
        assert isinstance(stage2_atlas, VoxelAtlas)
        assert seen == [(0, (n_timepoints, 2)), (1, (n_timepoints, 2)), (2, (n_timepoints, 2))]
