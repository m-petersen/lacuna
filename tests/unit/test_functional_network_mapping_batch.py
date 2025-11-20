"""
Unit tests for FunctionalNetworkMapping batch processing methods.

Tests the vectorized batch processing implementation, specifically
verifying the fixes for:
1. _load_mask_info() returning a tuple
2. _get_lesion_voxel_indices() called with correct arguments
"""

import h5py
import nibabel as nib
import numpy as np
import pytest

from lacuna import LesionData
from lacuna.analysis import FunctionalNetworkMapping


@pytest.fixture
def mock_connectome_batch(tmp_path):
    """Create a mock connectome batch file for testing."""
    connectome_path = tmp_path / "mock_connectome_batch.h5"

    # Create realistic test data
    n_subjects = 5
    n_timepoints = 100
    n_voxels = 1000

    # Create random timeseries data
    timeseries = np.random.randn(n_subjects, n_timepoints, n_voxels).astype(np.float32)

    # Create mask indices (3, n_voxels) format
    mask_indices = np.array(
        [
            np.repeat(range(10), 100),  # x coordinates
            np.tile(np.repeat(range(10), 10), 10),  # y coordinates
            np.tile(range(10), 100),  # z coordinates
        ]
    )

    # Create affine matrix (2mm MNI152 space)
    mask_affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    mask_shape = (91, 109, 91)

    # Write to HDF5
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=timeseries)
        f.create_dataset("mask_indices", data=mask_indices)
        f.create_dataset("mask_affine", data=mask_affine)
        f.attrs["mask_shape"] = mask_shape

    return connectome_path


@pytest.fixture
def mock_lesion_mni152(tmp_path):
    """Create a mock lesion in MNI152 2mm space that overlaps with connectome mask."""
    # Create a small lesion (5x5x5 voxels) that overlaps with mask indices
    # The mock_connectome_batch has mask indices from 0-9 in each dimension
    lesion_data = np.zeros((91, 109, 91), dtype=np.uint8)
    lesion_data[2:7, 2:7, 2:7] = 1  # Overlaps with mask coordinates 0-9

    # MNI152 2mm affine
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    lesion_img = nib.Nifti1Image(lesion_data, affine)

    # Save to file and load with LesionData.from_nifti
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(lesion_img, lesion_path)
    lesion = LesionData.from_nifti(str(lesion_path), metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    return lesion


def test_load_mask_info_returns_tuple(mock_connectome_batch):
    """Test that _load_mask_info() returns a tuple of (mask_indices, mask_affine, mask_shape)."""
    analysis = FunctionalNetworkMapping(
        connectome_path=str(mock_connectome_batch), method="boes", verbose=False
    )

    # Call _load_mask_info()
    result = analysis._load_mask_info()

    # Should return a tuple
    assert isinstance(result, tuple), "Expected tuple return value"
    assert len(result) == 3, "Expected 3 elements in tuple"

    mask_indices, mask_affine, mask_shape = result

    # Verify types
    assert isinstance(mask_indices, tuple), "mask_indices should be tuple"
    assert len(mask_indices) == 3, "mask_indices should have 3 coordinate arrays"
    assert isinstance(mask_affine, np.ndarray), "mask_affine should be ndarray"
    assert isinstance(mask_shape, tuple), "mask_shape should be tuple"

    # Verify values
    assert mask_affine.shape == (4, 4), "mask_affine should be 4x4"
    assert len(mask_shape) == 3, "mask_shape should be 3D"
    assert mask_shape == (91, 109, 91), "Expected MNI152 2mm shape"


def test_load_mask_info_sets_internal_state(mock_connectome_batch):
    """Test that _load_mask_info() also sets self._mask_info for backward compatibility."""
    analysis = FunctionalNetworkMapping(
        connectome_path=str(mock_connectome_batch), method="boes", verbose=False
    )

    # Should start as None
    assert analysis._mask_info is None

    # Call _load_mask_info()
    result = analysis._load_mask_info()

    # Should now be set
    assert analysis._mask_info is not None
    assert isinstance(analysis._mask_info, dict)
    assert "mask_indices" in analysis._mask_info
    assert "mask_affine" in analysis._mask_info
    assert "mask_shape" in analysis._mask_info

    # Return value should match internal state
    mask_indices, mask_affine, mask_shape = result
    np.testing.assert_array_equal(mask_affine, analysis._mask_info["mask_affine"])
    assert mask_shape == analysis._mask_info["mask_shape"]


def test_get_lesion_voxel_indices_signature(mock_connectome_batch, mock_lesion_mni152):
    """Test that _get_lesion_voxel_indices() accepts only LesionData argument."""
    analysis = FunctionalNetworkMapping(
        connectome_path=str(mock_connectome_batch), method="boes", verbose=False
    )

    # Load mask info first (required for _get_lesion_voxel_indices)
    analysis._load_mask_info()

    # Should accept LesionData object
    voxel_indices = analysis._get_lesion_voxel_indices(mock_lesion_mni152)

    # Should return array of indices
    assert isinstance(voxel_indices, np.ndarray)
    assert voxel_indices.ndim == 1
    assert len(voxel_indices) >= 0  # May be zero if no overlap


def test_run_batch_with_single_lesion(mock_connectome_batch, mock_lesion_mni152):
    """Test run_batch() method with a single lesion."""
    analysis = FunctionalNetworkMapping(
        connectome_path=str(mock_connectome_batch),
        method="boes",
        verbose=False,
        compute_t_map=False,  # Skip t-map for faster test
    )

    # Call run_batch with single lesion
    results = analysis.run_batch([mock_lesion_mni152])

    # Should return list with one result
    assert isinstance(results, list)
    assert len(results) == 1

    # Result should be LesionData with analysis results
    result = results[0]
    assert isinstance(result, LesionData)
    assert "FunctionalNetworkMapping" in result.results


def test_run_batch_with_multiple_lesions(mock_connectome_batch, mock_lesion_mni152):
    """Test run_batch() method with multiple lesions."""
    analysis = FunctionalNetworkMapping(
        connectome_path=str(mock_connectome_batch),
        method="boes",
        verbose=False,
        compute_t_map=False,
    )

    # Create multiple lesions (same lesion with different IDs)
    lesions = []
    for i in range(3):
        lesion_copy = LesionData(
            lesion_img=mock_lesion_mni152.lesion_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2, "subject_id": f"subject_{i}"},
        )
        lesions.append(lesion_copy)

    # Call run_batch
    results = analysis.run_batch(lesions)

    # Should return list with 3 results
    assert isinstance(results, list)
    assert len(results) == 3

    # All results should have analysis output
    for result in results:
        assert isinstance(result, LesionData)
        assert "FunctionalNetworkMapping" in result.results


def test_run_batch_preserves_metadata(mock_connectome_batch, mock_lesion_mni152):
    """Test that run_batch() preserves lesion metadata."""
    analysis = FunctionalNetworkMapping(
        connectome_path=str(mock_connectome_batch),
        method="boes",
        verbose=False,
        compute_t_map=False,
    )

    # Add custom metadata
    test_metadata = {"space": "MNI152NLin6Asym", "resolution": 2, "subject_id": "test_123", "custom_field": "test_value"}
    lesion = LesionData(lesion_img=mock_lesion_mni152.lesion_img, metadata=test_metadata)

    # Process
    results = analysis.run_batch([lesion])

    # Check metadata preserved
    result = results[0]
    assert result.metadata["subject_id"] == "test_123"
    assert result.metadata["custom_field"] == "test_value"


def test_load_mask_info_error_handling(tmp_path):
    """Test error handling when HDF5 file is malformed."""
    # Create invalid connectome (missing required datasets)
    bad_connectome = tmp_path / "bad_connectome.h5"
    with h5py.File(bad_connectome, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(5, 100, 1000))
        # Missing mask_indices, mask_affine, mask_shape

    analysis = FunctionalNetworkMapping(
        connectome_path=str(bad_connectome), method="boes", verbose=False
    )

    # Should raise KeyError when trying to load mask info
    with pytest.raises(KeyError):
        analysis._load_mask_info()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
