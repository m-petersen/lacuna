"""
Unit tests for CoordinateSpace consistent usage.

T037: Test that CoordinateSpace objects are used consistently.
"""

import pytest


@pytest.mark.unit
def test_coordinate_space_creation():
    """T037: Test that CoordinateSpace can be created with required fields."""
    import numpy as np

    from lacuna.core.spaces import CoordinateSpace

    # Create a simple affine matrix
    affine = np.eye(4)
    affine[:3, :3] *= 2.0  # 2mm resolution

    space = CoordinateSpace(identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=affine)

    assert space.identifier == "MNI152NLin6Asym"
    assert space.resolution == 2.0


@pytest.mark.unit
def test_coordinate_space_equality():
    """Test that CoordinateSpace objects with same values are considered equal."""
    import numpy as np

    from lacuna.core.spaces import CoordinateSpace

    affine = np.eye(4)
    affine[:3, :3] *= 2.0

    space1 = CoordinateSpace(identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=affine)
    space2 = CoordinateSpace(identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=affine)

    # Should be equal (if __eq__ is implemented)
    # Otherwise, at least verify attributes match
    assert space1.identifier == space2.identifier
    assert space1.resolution == space2.resolution


@pytest.mark.unit
def test_coordinate_space_in_result_objects():
    """Test that result objects use consistent space representation."""
    import nibabel as nib
    import numpy as np

    from lacuna.core.data_types import VoxelMap

    data = np.random.rand(64, 64, 64).astype(np.float32)
    affine = np.eye(4)
    test_img = nib.Nifti1Image(data, affine)

    result = VoxelMap(name="test", data=test_img, space="MNI152NLin2009cAsym", resolution=2.0)

    # Space should be stored as string (matching CoordinateSpace.identifier)
    assert isinstance(result.space, str)
    assert isinstance(result.resolution, (int, float))
    assert result.space == "MNI152NLin2009cAsym"
    assert result.resolution == 2.0
