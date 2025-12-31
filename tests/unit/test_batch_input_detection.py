"""Unit tests for batch input type detection.

These tests verify the _detect_input_type helper function correctly
identifies input types for batch_process.

Phase 4: VoxelMap Batch Processing (T022-T023)
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.parcellations.registry import (
    register_parcellations_from_directory,
    unregister_parcellation,
)
from lacuna.batch.api import _detect_input_type
from lacuna.core.data_types import VoxelMap
from lacuna.core.subject_data import SubjectData


@pytest.fixture
def local_test_atlas(tmp_path):
    """Create and register a local test atlas for CI."""
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Match sample_mask_data shape (10, 10, 10)
    shape = (10, 10, 10)
    affine = np.eye(4)

    # Create atlas with 2 regions
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[2:5, 2:5, 2:5] = 1
    atlas_data[5:8, 5:8, 5:8] = 2

    atlas_img = nib.Nifti1Image(atlas_data, affine)
    atlas_path = atlas_dir / "test_batch_atlas.nii.gz"
    nib.save(atlas_img, atlas_path)

    # Create labels file
    labels_path = atlas_dir / "test_batch_atlas_labels.txt"
    labels_path.write_text("1 Region_A\n2 Region_B\n")

    # Register the atlas
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=1)

    yield "test_batch_atlas"

    # Cleanup
    try:
        unregister_parcellation("test_batch_atlas")
    except KeyError:
        pass


@pytest.fixture
def sample_mask_data():
    """Create a sample SubjectData for testing."""
    shape = (10, 10, 10)
    affine = np.eye(4)
    data = np.zeros(shape, dtype=np.int8)
    data[4:6, 4:6, 4:6] = 1  # Small lesion
    img = nib.Nifti1Image(data, affine)

    return SubjectData(
        mask_img=img,
        space="MNI152NLin6Asym",
        resolution=1.0,
        metadata={"subject_id": "test_subject"},
    )


@pytest.fixture
def sample_voxelmap():
    """Create a sample VoxelMap for testing."""
    shape = (10, 10, 10)
    affine = np.eye(4)
    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine)

    return VoxelMap(
        name="test_voxelmap",
        data=img,
        space="MNI152NLin6Asym",
        resolution=1.0,
    )


class TestDetectInputType:
    """Unit tests for _detect_input_type function."""

    def test_detects_mask_data_only(self, sample_mask_data):
        """Should return 'subject_data' for list containing only SubjectData."""
        inputs = [sample_mask_data] * 3
        result = _detect_input_type(inputs)
        assert result == "subject_data"

    def test_detects_voxelmap_only(self, sample_voxelmap):
        """Should return 'voxel_map' for list containing only VoxelMap."""
        inputs = [sample_voxelmap] * 3
        result = _detect_input_type(inputs)
        assert result == "voxel_map"

    def test_detects_mixed_types(self, sample_mask_data, sample_voxelmap):
        """Should raise TypeError for list containing both types."""
        inputs = [sample_mask_data, sample_voxelmap]
        with pytest.raises(TypeError, match="does not support mixed input types"):
            _detect_input_type(inputs)

    def test_detects_mixed_types_interleaved(self, sample_mask_data, sample_voxelmap):
        """Should raise TypeError even with interleaved types."""
        inputs = [sample_mask_data, sample_voxelmap, sample_mask_data, sample_voxelmap]
        with pytest.raises(TypeError, match="does not support mixed input types"):
            _detect_input_type(inputs)

    def test_empty_list_raises_value_error(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _detect_input_type([])

    def test_single_mask_data(self, sample_mask_data):
        """Should correctly identify single SubjectData."""
        result = _detect_input_type([sample_mask_data])
        assert result == "subject_data"

    def test_single_voxelmap(self, sample_voxelmap):
        """Should correctly identify single VoxelMap."""
        result = _detect_input_type([sample_voxelmap])
        assert result == "voxel_map"

    def test_unknown_types_raise_type_error(self):
        """Unknown types should raise TypeError with informative message."""
        inputs = ["not_a_valid_type", 123, None]
        with pytest.raises(TypeError, match="requires all inputs to be SubjectData or VoxelMap"):
            _detect_input_type(inputs)


class TestBatchProcessMixedTypeError:
    """Unit tests for mixed type rejection in batch_process."""

    def test_mixed_types_raise_type_error(
        self, sample_mask_data, sample_voxelmap, local_test_atlas
    ):
        """batch_process should raise TypeError for mixed inputs."""
        from lacuna.analysis import ParcelAggregation
        from lacuna.batch import batch_process

        mixed_inputs = [sample_mask_data, sample_voxelmap]
        analysis = ParcelAggregation(
            source="maskimg", aggregation="mean", parcel_names=[local_test_atlas]
        )

        with pytest.raises(TypeError, match="mixed"):
            batch_process(inputs=mixed_inputs, analysis=analysis, show_progress=False)

    def test_error_message_is_descriptive(
        self, sample_mask_data, sample_voxelmap, local_test_atlas
    ):
        """TypeError message should explain the issue clearly."""
        from lacuna.analysis import ParcelAggregation
        from lacuna.batch import batch_process

        mixed_inputs = [sample_mask_data, sample_voxelmap]
        analysis = ParcelAggregation(
            source="maskimg", aggregation="mean", parcel_names=[local_test_atlas]
        )

        with pytest.raises(TypeError) as exc_info:
            batch_process(inputs=mixed_inputs, analysis=analysis, show_progress=False)

        error_message = str(exc_info.value).lower()
        assert "mixed" in error_message
        assert "subjectdata" in error_message or "voxelmap" in error_message
