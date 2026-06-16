"""
Unit tests for CoordinateSpace consistent usage and space detection.

T037: Test that CoordinateSpace objects are used consistently.
"""

import nibabel as nib
import numpy as np
import pytest


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


# ===========================================================================
# detect_space_from_header: radiological-convention affine detection
# ===========================================================================


class TestDetectSpaceRadiological:
    """detect_space_from_header should recognize radiological-convention images.

    Radiological images have negative strides (e.g. srow_x = [-1, 0, 0, 90])
    but cover the same physical volume as neurological images
    (srow_x = [1, 0, 0, -91]). Both should be detected as the same space.
    """

    @pytest.mark.parametrize(
        "space,resolution,shape,neuro_origin",
        [
            ("MNI152NLin6Asym", 1, (182, 218, 182), (-91, -126, -72)),
            ("MNI152NLin6Asym", 2, (91, 109, 91), (-90, -126, -72)),
            ("MNI152NLin2009cAsym", 1, (193, 229, 193), (-96, -132, -78)),
            ("MNI152NLin2009cAsym", 2, (97, 115, 97), (-96.5, -132.5, -78.5)),
        ],
    )
    def test_neurological_convention_detected(self, space, resolution, shape, neuro_origin):
        """Standard neurological affine is detected correctly."""
        from lacuna.core.spaces import detect_space_from_header

        affine = np.diag([resolution, resolution, resolution, 1.0])
        affine[:3, 3] = neuro_origin
        img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)

        result = detect_space_from_header(img)
        assert result is not None
        assert result == (space, resolution)

    @pytest.mark.parametrize(
        "space,resolution,shape,neuro_origin",
        [
            ("MNI152NLin6Asym", 1, (182, 218, 182), (-91, -126, -72)),
            ("MNI152NLin6Asym", 2, (91, 109, 91), (-90, -126, -72)),
            ("MNI152NLin2009cAsym", 1, (193, 229, 193), (-96, -132, -78)),
            ("MNI152NLin2009cAsym", 2, (97, 115, 97), (-96.5, -132.5, -78.5)),
        ],
    )
    def test_radiological_convention_detected(self, space, resolution, shape, neuro_origin):
        """Radiological affine (flipped X) is detected as the same space."""
        from lacuna.core.spaces import detect_space_from_header

        # Flip X: negate voxel size, adjust origin to far end
        radio_origin_x = -neuro_origin[0] - resolution  # e.g. 91-2 = 89... no
        # For radiological: origin_x = -(neuro_origin_x + resolution * (shape_x - 1))
        # which equals the world-coordinate of the LAST voxel in neurological
        radio_origin_x = neuro_origin[0] + resolution * (shape[0] - 1)
        affine = np.diag([-resolution, resolution, resolution, 1.0])
        affine[:3, 3] = [radio_origin_x, neuro_origin[1], neuro_origin[2]]
        img = nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)

        result = detect_space_from_header(img)
        assert result is not None, (
            f"Radiological {space} {resolution}mm not detected. " f"Affine:\n{affine}"
        )
        assert result == (space, resolution)

    def test_wrong_shape_not_detected(self):
        """An image with a non-standard shape should not be detected."""
        from lacuna.core.spaces import detect_space_from_header

        affine = np.diag([-2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [90, -126, -72]
        img = nib.Nifti1Image(np.zeros((100, 100, 100), dtype=np.uint8), affine)

        assert detect_space_from_header(img) is None

    def test_random_affine_not_detected(self):
        """A random affine that doesn't match any known space returns None."""
        from lacuna.core.spaces import detect_space_from_header

        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-50, -50, -50]
        img = nib.Nifti1Image(np.zeros((91, 109, 91), dtype=np.uint8), affine)

        assert detect_space_from_header(img) is None

    def test_oblique_affine_not_detected(self):
        """An oblique (rotated) affine should not match via the fallback."""
        from lacuna.core.spaces import detect_space_from_header

        # Oblique: off-diagonal elements
        affine = np.array(
            [
                [-1.9, 0.3, 0.0, 90],
                [0.3, 1.9, 0.0, -126],
                [0.0, 0.0, 2.0, -72],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(np.zeros((91, 109, 91), dtype=np.uint8), affine)

        assert detect_space_from_header(img) is None
