"""Unit tests for batch input type detection.

These tests verify the _detect_input_type helper function correctly
identifies input types for batch_process.

Phase 4: VoxelMap Batch Processing (T022-T023)
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.batch.api import _detect_input_type
from lacuna.core.data_types import VoxelMap
from lacuna.core.mask_data import MaskData


@pytest.fixture
def sample_mask_data():
    """Create a sample MaskData for testing."""
    shape = (10, 10, 10)
    affine = np.eye(4)
    data = np.zeros(shape, dtype=np.int8)
    data[4:6, 4:6, 4:6] = 1  # Small lesion
    img = nib.Nifti1Image(data, affine)

    return MaskData(
        mask_img=img,
        space="MNI152NLin6Asym",
        resolution=1.0,
        metadata={"subject_id": "test_subject"}
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
        """Should return 'mask_data' for list containing only MaskData."""
        inputs = [sample_mask_data] * 3
        result = _detect_input_type(inputs)
        assert result == "mask_data"

    def test_detects_voxelmap_only(self, sample_voxelmap):
        """Should return 'voxel_map' for list containing only VoxelMap."""
        inputs = [sample_voxelmap] * 3
        result = _detect_input_type(inputs)
        assert result == "voxel_map"

    def test_detects_mixed_types(self, sample_mask_data, sample_voxelmap):
        """Should return 'mixed' for list containing both types."""
        inputs = [sample_mask_data, sample_voxelmap]
        result = _detect_input_type(inputs)
        assert result == "mixed"

    def test_detects_mixed_types_interleaved(self, sample_mask_data, sample_voxelmap):
        """Should return 'mixed' even with interleaved types."""
        inputs = [sample_mask_data, sample_voxelmap, sample_mask_data, sample_voxelmap]
        result = _detect_input_type(inputs)
        assert result == "mixed"

    def test_empty_list_raises_value_error(self):
        """Should raise ValueError for empty list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _detect_input_type([])

    def test_single_mask_data(self, sample_mask_data):
        """Should correctly identify single MaskData."""
        result = _detect_input_type([sample_mask_data])
        assert result == "mask_data"

    def test_single_voxelmap(self, sample_voxelmap):
        """Should correctly identify single VoxelMap."""
        result = _detect_input_type([sample_voxelmap])
        assert result == "voxel_map"

    def test_unknown_types_default_to_mask_data(self):
        """Unknown types should default to mask_data behavior."""
        # If we pass something else, it shouldn't crash
        # and should default to mask_data since nothing else was found
        inputs = ["not_a_valid_type", 123, None]
        result = _detect_input_type(inputs)
        # No VoxelMap or MaskData found, defaults to mask_data
        assert result == "mask_data"


class TestBatchProcessMixedTypeError:
    """Unit tests for mixed type rejection in batch_process."""

    def test_mixed_types_raise_type_error(self, sample_mask_data, sample_voxelmap):
        """batch_process should raise TypeError for mixed inputs."""
        from lacuna.batch import batch_process
        from lacuna.analysis import ParcelAggregation

        mixed_inputs = [sample_mask_data, sample_voxelmap]
        analysis = ParcelAggregation(
            source="mask_img",
            aggregation="mean",
            parcel_names=["Schaefer2018_100Parcels7Networks"]
        )

        with pytest.raises(TypeError, match="mixed"):
            batch_process(inputs=mixed_inputs, analysis=analysis, show_progress=False)

    def test_error_message_is_descriptive(self, sample_mask_data, sample_voxelmap):
        """TypeError message should explain the issue clearly."""
        from lacuna.batch import batch_process
        from lacuna.analysis import ParcelAggregation

        mixed_inputs = [sample_mask_data, sample_voxelmap]
        analysis = ParcelAggregation(
            source="mask_img",
            aggregation="mean",
            parcel_names=["Schaefer2018_100Parcels7Networks"]
        )

        with pytest.raises(TypeError) as exc_info:
            batch_process(inputs=mixed_inputs, analysis=analysis, show_progress=False)

        error_message = str(exc_info.value).lower()
        assert "mixed" in error_message
        assert "maskdata" in error_message or "voxelmap" in error_message


class TestDeprecatedMaskDataListParameter:
    """Unit tests for the deprecated mask_data_list parameter."""

    def test_mask_data_list_parameter_still_works(self, sample_mask_data):
        """mask_data_list parameter should still work for backward compat."""
        from lacuna.batch import batch_process
        from lacuna.analysis import ParcelAggregation
        import warnings

        analysis = ParcelAggregation(
            source="mask_img",
            aggregation="mean",
            parcel_names=["Schaefer2018_100Parcels7Networks"]
        )

        # Should emit deprecation warning but work
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = batch_process(
                mask_data_list=[sample_mask_data],
                analysis=analysis,
                show_progress=False
            )
            # Should have triggered a deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "mask_data_list" in str(w[0].message)

    def test_cannot_specify_both_inputs_and_mask_data_list(self, sample_mask_data):
        """Should raise ValueError if both inputs and mask_data_list provided."""
        from lacuna.batch import batch_process
        from lacuna.analysis import ParcelAggregation
        import warnings

        analysis = ParcelAggregation(
            source="mask_img",
            aggregation="mean",
            parcel_names=["Schaefer2018_100Parcels7Networks"]
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress deprecation warning
                batch_process(
                    inputs=[sample_mask_data],
                    mask_data_list=[sample_mask_data],
                    analysis=analysis,
                    show_progress=False
                )
