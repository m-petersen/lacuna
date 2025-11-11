"""Contract tests for space detection API (lacuna.core.spaces).

Tests the public interface contracts defined in:
specs/001-neuroimaging-space-handling/contracts/space_detection.md
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


class TestCoordinateSpaceContract:
    """Test CoordinateSpace dataclass contract."""

    def test_coordinate_space_is_immutable(self):
        """Contract: CoordinateSpace instances are immutable."""
        from lacuna.core.spaces import CoordinateSpace

        space = CoordinateSpace(
            identifier="MNI152NLin6Asym", resolution=2, reference_affine=np.eye(4)
        )

        with pytest.raises(AttributeError):
            space.identifier = "MNI152NLin2009cAsym"

    def test_coordinate_space_validates_on_creation(self):
        """Contract: CoordinateSpace validates identifier and resolution."""
        from lacuna.core.spaces import CoordinateSpace

        # Invalid identifier
        with pytest.raises(ValueError, match="identifier must be in"):
            CoordinateSpace(identifier="InvalidSpace", resolution=2, reference_affine=np.eye(4))

        # Invalid resolution
        with pytest.raises(ValueError, match="resolution must be one of"):
            CoordinateSpace(identifier="MNI152NLin6Asym", resolution=3, reference_affine=np.eye(4))


class TestDetectSpaceFromFilename:
    """Test detect_space_from_filename() contract."""

    def test_detects_bids_format(self):
        """FR-003: Detect space from BIDS filename."""
        from lacuna.core.spaces import detect_space_from_filename

        result = detect_space_from_filename("sub-01_space-MNI152NLin6Asym_res-2_mask.nii.gz")
        assert result == ("MNI152NLin6Asym", 2)

    def test_returns_none_for_unknown(self):
        """Contract: Return None gracefully for unknown patterns."""
        from lacuna.core.spaces import detect_space_from_filename

        result = detect_space_from_filename("arbitrary_name.nii.gz")
        assert result is None

    def test_never_raises_exception(self):
        """Contract: Never raises exception on parsing failure."""
        from lacuna.core.spaces import detect_space_from_filename

        # Should not raise
        result = detect_space_from_filename("malformed%%%file.nii")
        assert result is None


class TestDetectSpaceFromHeader:
    """Test detect_space_from_header() contract."""

    def test_detects_exact_affine_match(self):
        """FR-004: Detect space from affine matrix."""
        from lacuna.core.spaces import REFERENCE_AFFINES, detect_space_from_header

        # Create mock image with known affine
        affine = REFERENCE_AFFINES[("MNI152NLin6Asym", 2)]
        img = nib.Nifti1Image(np.zeros((91, 109, 91)), affine)

        result = detect_space_from_header(img)
        assert result == ("MNI152NLin6Asym", 2)

    def test_returns_none_for_unknown_affine(self):
        """Contract: Return None for unrecognized affine."""
        from lacuna.core.spaces import detect_space_from_header

        # Custom affine not in registry
        affine = np.diag([3.5, 3.5, 3.5, 1])
        img = nib.Nifti1Image(np.zeros((50, 50, 50)), affine)

        result = detect_space_from_header(img)
        assert result is None


class TestGetImageSpace:
    """Test get_image_space() unified detection contract."""

    def test_returns_valid_coordinate_space(self):
        """Contract: get_image_space always returns CoordinateSpace instance."""
        from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace, get_image_space

        affine = REFERENCE_AFFINES[("MNI152NLin6Asym", 2)]
        img = nib.Nifti1Image(np.zeros((91, 109, 91)), affine)
        filepath = Path("sub-001_space-MNI152NLin6Asym_res-2.nii.gz")

        space = get_image_space(img, filepath=filepath)

        assert isinstance(space, CoordinateSpace)
        assert space.identifier == "MNI152NLin6Asym"
        assert space.resolution == 2
        assert space.reference_affine.shape == (4, 4)

    def test_raises_on_detection_failure(self):
        """Contract: Either returns space or raises clear exception."""
        from lacuna.core.spaces import SpaceDetectionError, get_image_space

        # Image without space info
        img = nib.Nifti1Image(np.zeros((50, 50, 50)), np.eye(4))

        with pytest.raises(SpaceDetectionError) as exc_info:
            get_image_space(img)

        assert "Could not detect coordinate space" in str(exc_info.value)
        assert "Please specify space explicitly" in str(exc_info.value)

    def test_raises_on_mismatch(self):
        """Contract: Raise error when declared space doesn't match detected."""
        from lacuna.core.spaces import REFERENCE_AFFINES, SpaceMismatchError, get_image_space

        affine = REFERENCE_AFFINES[("MNI152NLin6Asym", 2)]
        img = nib.Nifti1Image(np.zeros((91, 109, 91)), affine)

        with pytest.raises(SpaceMismatchError):
            get_image_space(img, declared_space="MNI152NLin2009cAsym", require_match=True)


class TestQuerySupportedSpaces:
    """Test query_supported_spaces() contract (FR-019)."""

    def test_returns_list_of_spaces(self):
        """Contract: Returns list of supported space identifiers."""
        from lacuna.core.spaces import query_supported_spaces

        spaces = query_supported_spaces()

        assert isinstance(spaces, list)
        assert len(spaces) > 0
        assert all(isinstance(s, str) for s in spaces)

    def test_list_is_sorted(self):
        """Contract: List is sorted alphabetically."""
        from lacuna.core.spaces import query_supported_spaces

        spaces = query_supported_spaces()
        assert spaces == sorted(spaces)

    def test_contains_expected_spaces(self):
        """Contract: Contains standard MNI spaces."""
        from lacuna.core.spaces import query_supported_spaces

        spaces = query_supported_spaces()
        assert "MNI152NLin6Asym" in spaces
        assert "MNI152NLin2009cAsym" in spaces
