"""
Contract tests for space handling requirements.

T032-T035: Tests for explicit space/resolution requirements and error messages.
"""

import nibabel as nib
import numpy as np
import pytest


@pytest.mark.contract
def test_space_requirement_error(synthetic_mask_img):
    """T032: Test that MaskData raises error when space is missing."""
    from lacuna.core.mask_data import MaskData

    # Missing 'space' parameter should raise error
    with pytest.raises(ValueError) as exc_info:
        MaskData(
            synthetic_mask_img,
            metadata={"subject_id": "sub-001"}  # No space or resolution
        )

    error_msg = str(exc_info.value)
    assert "space" in error_msg.lower()
    assert "parameter" in error_msg.lower()  # Changed from "metadata"


@pytest.mark.contract
def test_resolution_requirement_error(synthetic_mask_img):
    """T033: Test that MaskData raises error when resolution is missing."""
    from lacuna.core.mask_data import MaskData

    # Missing 'resolution' parameter should raise error
    with pytest.raises(ValueError) as exc_info:
        MaskData(
            synthetic_mask_img,
            space="MNI152NLin6Asym",  # space provided but not resolution
            metadata={"subject_id": "sub-001"}
        )

    error_msg = str(exc_info.value)
    assert "resolution" in error_msg.lower()
    assert "parameter" in error_msg.lower()  # Changed from "metadata"


@pytest.mark.contract
def test_separated_space_resolution_attributes():
    """T034: Test that VoxelMapResult stores space and resolution separately."""
    from lacuna.core.data_types import VoxelMap

    # Create test image
    data = np.random.rand(64, 64, 64).astype(np.float32)
    affine = np.eye(4)
    test_img = nib.Nifti1Image(data, affine)

    result = VoxelMap(
        name="test_map",
        data=test_img,
        space="MNI152NLin2009cAsym",
        resolution=2.0
    )

    # Space and resolution should be separate attributes
    assert hasattr(result, "space")
    assert hasattr(result, "resolution")
    assert result.space == "MNI152NLin2009cAsym"
    assert result.resolution == 2.0
    assert isinstance(result.space, str)
    assert isinstance(result.resolution, (int, float))


@pytest.mark.contract
def test_supported_spaces_error_message(synthetic_mask_img):
    """T035: Test that error message lists only supported spaces."""
    from lacuna.core.mask_data import MaskData

    # Use unsupported space
    with pytest.raises(ValueError) as exc_info:
        MaskData(
            synthetic_mask_img,
            metadata={
                "subject_id": "sub-001",
                "space": "UnsupportedSpace",
                "resolution": 2
            }
        )

    error_msg = str(exc_info.value)
    # Should mention supported spaces
    assert "MNI152NLin6Asym" in error_msg
    assert "MNI152NLin2009aAsym" in error_msg
    assert "MNI152NLin2009cAsym" in error_msg


@pytest.fixture
def synthetic_mask_img():
    """Create a synthetic 3D binary mask for testing."""
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)

    # Create small spherical lesion
    center = (32, 32, 32)
    radius = 5
    for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)):
        for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)):
            for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius + 1)):
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                if dist <= radius:
                    data[x, y, z] = 1

    affine = np.eye(4)
    affine[0, 0] = 2.0  # 2mm voxels
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0

    return nib.Nifti1Image(data, affine)
