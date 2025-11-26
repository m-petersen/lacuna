"""Critical regression tests for FunctionalNetworkMapping batch processing.

These tests verify the bug fixes for:
1. _load_mask_info() returning None instead of tuple
2. _get_lesion_voxel_indices() being called with wrong number of arguments
3. _aggregate_results() not capturing add_result() return value (empty results)
"""

import h5py
import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis import FunctionalNetworkMapping
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)


@pytest.fixture
def simple_connectome(tmp_path):
    """Create minimal valid connectome for testing."""
    connectome_path = tmp_path / "connectome.h5"

    # Minimal valid data
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(5, 100, 100))
        f.create_dataset("mask_indices", data=np.random.randint(0, 50, (3, 100)))
        f.create_dataset("mask_affine", data=np.eye(4))
        f.attrs["mask_shape"] = (91, 109, 91)

    return connectome_path


def test_bug_fix_load_mask_info_returns_tuple(simple_connectome):
    """
    REGRESSION TEST: Verify _load_mask_info() returns tuple.

    Previously: _load_mask_info() returned None
    Bug: TypeError: cannot unpack non-iterable NoneType object
    Fix: Method now returns (mask_indices, mask_affine, mask_shape) tuple
    """
    register_functional_connectome(
        name="test_bug_load_mask",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=simple_connectome,
        n_subjects=5,
        description="Test"
    )

    try:
        analysis = FunctionalNetworkMapping(
            connectome_name="test_bug_load_mask", method="boes", log_level=0
        )

        # Call the method - should not raise TypeError
        result = analysis._load_mask_info()

        # Verify it returns a tuple with 3 elements
        assert result is not None, "Bug regression: _load_mask_info() returned None"
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

        # Verify unpacking works (this is what run_batch() does)
        mask_indices, mask_affine, mask_shape = result
        assert mask_indices is not None
        assert mask_affine is not None
        assert mask_shape is not None
    finally:
        unregister_functional_connectome("test_bug_load_mask")


def test_bug_fix_get_lesion_voxel_indices_signature(simple_connectome):
    """
    REGRESSION TEST: Verify _get_lesion_voxel_indices() has correct signature.

    Previously: run_batch() called with 5 arguments (self, img, indices, shape, affine)
    Bug: TypeError: takes 2 positional arguments but 5 were given
    Fix: Method takes only 2 arguments (self, mask_data), uses self._mask_info internally
    """
    register_functional_connectome(
        name="test_bug_voxel_indices",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=simple_connectome,
        n_subjects=5,
        description="Test"
    )

    try:
        analysis = FunctionalNetworkMapping(
            connectome_name="test_bug_voxel_indices", method="boes", log_level=0
        )

        # Load mask info first (required)
        analysis._load_mask_info()

        # Create a dummy lesion
        mask_data_array = np.zeros((91, 109, 91), dtype=np.uint8)
        mask_data_array[45, 50, 45] = 1
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mask_img = nib.Nifti1Image(mask_data_array, affine)
        lesion = MaskData(mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

        # Call with MaskData object only - should not raise TypeError
        try:
            result = analysis._get_lesion_voxel_indices(lesion)
            # Success - method accepts correct signature
            assert isinstance(result, np.ndarray), "Expected ndarray return value"
        except TypeError as e:
            if "positional arguments" in str(e):
                pytest.fail(f"Bug regression: Method signature is wrong - {e}")
            raise
    finally:
        unregister_functional_connectome("test_bug_voxel_indices")


def test_both_fixes_together(simple_connectome):
    """
    INTEGRATION TEST: Verify both fixes work together in run_batch().

    This simulates what happens when VectorizedStrategy calls run_batch():
    1. run_batch() unpacks _load_mask_info() return value
    2. run_batch() calls _get_lesion_voxel_indices() with correct arguments
    """
    register_functional_connectome(
        name="test_bug_both_fixes",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=simple_connectome,
        n_subjects=5,
        description="Test"
    )

    try:
        analysis = FunctionalNetworkMapping(
            connectome_name="test_bug_both_fixes", method="boes", log_level=0, compute_t_map=False
        )

        # Create a dummy lesion
        mask_data_array = np.zeros((91, 109, 91), dtype=np.uint8)
        mask_data_array[40:50, 50:60, 40:50] = 1  # Larger lesion
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        mask_img = nib.Nifti1Image(mask_data_array, affine)
        lesion = MaskData(mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

        # This should work without TypeErrors
        # (May raise ValidationError if no overlap, but that's expected)
        try:
            results = analysis.run_batch([lesion])
            # Success - both fixes working
            assert isinstance(results, list)
        except TypeError as e:
            pytest.fail(f"Bug regression in run_batch(): {e}")
        except Exception:
            # Other errors are OK for this test (e.g., validation errors)
            pass
    finally:
        unregister_functional_connectome("test_bug_both_fixes")


def test_bug_fix_aggregate_results_returns_with_data(tmp_path):
    """
    REGRESSION TEST: Verify _aggregate_results() captures add_result() return value.

    Previously: mask_data.add_result() was called but return value not captured
    Bug: Results dictionary empty, saved_files empty, output directory empty
    Fix: mask_data_with_results = mask_data.add_result() and return it
    """
    # Create mock connectome with KNOWN coordinates
    n_voxels = 125  # 5x5x5 cube
    x_coords = np.repeat(range(43, 48), 25)
    y_coords = np.tile(np.repeat(range(52, 57), 5), 5)
    z_coords = np.tile(range(43, 48), 25)
    mask_indices = np.array([x_coords, y_coords, z_coords])

    connectome_path = tmp_path / "connectome.h5"
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(10, 100, n_voxels))
        f.create_dataset("mask_indices", data=mask_indices)
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        f.create_dataset("mask_affine", data=affine)
        f.attrs["mask_shape"] = (91, 109, 91)

    # Create mock lesion IN THE SAME REGION
    mask_data_array = np.zeros((91, 109, 91), dtype=np.uint8)
    mask_data_array[43:48, 52:57, 43:48] = 1  # Same 5x5x5 cube

    mask_img = nib.Nifti1Image(mask_data_array, affine)

    # Save and load with MaskData
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(mask_img, lesion_path)
    lesion = MaskData.from_nifti(
        str(lesion_path), metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Register connectome
    register_functional_connectome(
        name="test_bug_aggregate",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=connectome_path,
        n_subjects=10,
        description="Test"
    )

    try:
        # Create analysis
        analysis = FunctionalNetworkMapping(
            connectome_name="test_bug_aggregate",
            method="boes",
            log_level=0,
            compute_t_map=True,
            t_threshold=2.0,
        )

        # Run batch processing
        results = analysis.run_batch([lesion])

        # Verify results structure
        assert len(results) == 1, "Expected 1 result"
        result = results[0]

        # The critical check: results dictionary should NOT be empty
        assert "FunctionalNetworkMapping" in result.results, (
            "Bug regression: FunctionalNetworkMapping key missing from results. "
            "This indicates add_result() return value was not captured."
        )

        flnm_results = result.results["FunctionalNetworkMapping"]

        # Verify expected keys are present
        expected_keys = ["correlation_map", "z_map", "summary_statistics"]
        for key in expected_keys:
            assert key in flnm_results, f"Missing expected key: {key}"

        # Verify NIfTI images are present
        assert isinstance(
            flnm_results["correlation_map"], nib.Nifti1Image
        ), "correlation_map should be NIfTI image"
        assert isinstance(flnm_results["z_map"], nib.Nifti1Image), "z_map should be NIfTI image"

        # Since compute_t_map=True, these should also be present
        assert "t_map" in flnm_results, "t_map should be present when compute_t_map=True"
        assert (
            "t_threshold_map" in flnm_results
        ), "t_threshold_map should be present when t_threshold is set"
    finally:
        unregister_functional_connectome("test_bug_aggregate")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
