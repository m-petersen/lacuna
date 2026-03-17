"""Unit tests for space variant validation utilities.

Tests consistent handling of coordinate spaces across the codebase.

Key design:
- Supported spaces: MNI152NLin6Asym, MNI152NLin2009cAsym, MNI152NLin2009bAsym (internal)
- No space aliases — spaces_are_equivalent is strict equality
- 2009b and 2009c are distinct spaces (different voxel grids, require regridding)
"""

import nibabel as nib
import numpy as np
import pytest


class TestSpacesAreEquivalent:
    """Test spaces_are_equivalent helper for comparing spaces."""

    def test_bAsym_not_equal_cAsym(self):
        """bAsym and cAsym have different voxel grids — NOT equivalent."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert not spaces_are_equivalent("MNI152NLin2009bAsym", "MNI152NLin2009cAsym")
        assert not spaces_are_equivalent("MNI152NLin2009cAsym", "MNI152NLin2009bAsym")

    def test_identical_spaces_equal(self):
        """Identical space strings are equivalent."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin6Asym")
        assert spaces_are_equivalent("MNI152NLin2009cAsym", "MNI152NLin2009cAsym")
        assert spaces_are_equivalent("MNI152NLin2009bAsym", "MNI152NLin2009bAsym")

    def test_different_spaces_not_equal(self):
        """Different spaces (NLin6 vs NLin2009c) are not equivalent."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert not spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin2009cAsym")

    def test_native_not_equal_to_mni(self):
        """Native space not equivalent to any MNI space."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert not spaces_are_equivalent("native", "MNI152NLin6Asym")
        assert not spaces_are_equivalent("native", "MNI152NLin2009cAsym")


class TestValidateSpaceCompatibility:
    """Test validate_space_compatibility helper for consistent validation."""

    def test_accepts_identical_spaces(self):
        """Should accept identical spaces."""
        from lacuna.core.spaces import validate_space_compatibility

        # Should not raise
        validate_space_compatibility(
            actual_space="MNI152NLin6Asym", expected_space="MNI152NLin6Asym", context="test"
        )

    def test_rejects_bAsym_vs_cAsym(self):
        """bAsym and cAsym have different voxel grids — should raise."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="Space mismatch"):
            validate_space_compatibility(
                actual_space="MNI152NLin2009bAsym",
                expected_space="MNI152NLin2009cAsym",
                context="test",
            )

    def test_rejects_incompatible_spaces(self):
        """Should raise error for incompatible spaces."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="Space mismatch"):
            validate_space_compatibility(
                actual_space="MNI152NLin6Asym",
                expected_space="MNI152NLin2009cAsym",
                context="test analysis",
            )

    def test_error_message_includes_context(self):
        """Error message should include context for debugging."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="test analysis"):
            validate_space_compatibility(
                actual_space="native", expected_space="MNI152NLin6Asym", context="test analysis"
            )

    def test_error_message_shows_both_spaces(self):
        """Error message should show both actual and expected spaces."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="native.*MNI152NLin6Asym"):
            validate_space_compatibility(
                actual_space="native", expected_space="MNI152NLin6Asym", context="test"
            )

    def test_suggests_transformation_if_possible(self):
        """Error message should suggest transformation if available."""
        from lacuna.core.spaces import validate_space_compatibility

        # NLin6 ↔ NLin2009c transformation exists
        with pytest.raises(ValueError, match="transform"):
            validate_space_compatibility(
                actual_space="MNI152NLin6Asym",
                expected_space="MNI152NLin2009cAsym",
                context="test",
                suggest_transform=True,
            )


class TestValidateSpaceAndResolution:
    """Test validate_space_and_resolution for SubjectData and metadata validation."""

    def test_accepts_valid_space_and_resolution(self):
        """Should accept valid space + resolution combination."""
        from lacuna.core.spaces import validate_space_and_resolution

        # Should not raise
        validate_space_and_resolution(space="MNI152NLin6Asym", resolution=2.0)

    def test_rejects_missing_resolution_when_space_specified(self):
        """Should raise error if space specified but resolution is None."""
        from lacuna.core.spaces import validate_space_and_resolution

        with pytest.raises(ValueError, match="[Rr]esolution.*required"):
            validate_space_and_resolution(space="MNI152NLin6Asym", resolution=None)

    def test_rejects_invalid_resolution(self):
        """Should raise error for invalid resolution values."""
        from lacuna.core.spaces import validate_space_and_resolution

        with pytest.raises(ValueError, match="resolution"):
            validate_space_and_resolution(
                space="MNI152NLin6Asym", resolution=3.0  # Invalid - must be 0.5, 1, or 2
            )

    def test_accepts_none_space_with_none_resolution(self):
        """Should accept both None if data is in native/unknown space."""
        from lacuna.core.spaces import validate_space_and_resolution

        # Should not raise
        validate_space_and_resolution(space=None, resolution=None)

    def test_validates_space_identifier(self):
        """Should validate space identifier is recognized."""
        from lacuna.core.spaces import validate_space_and_resolution

        with pytest.raises(ValueError, match="Unknown.*space"):
            validate_space_and_resolution(space="InvalidSpaceName", resolution=2.0)


class TestParcelAggregationSpaceHandling:
    """Test that ParcelAggregation correctly handles space differences."""

    @pytest.fixture
    def mask_data_bAsym(self):
        """Create SubjectData in MNI152NLin2009bAsym space."""
        from lacuna.core.subject_data import SubjectData

        # 2mm MNI affine
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(np.ones((91, 109, 91)), affine)

        return SubjectData(
            mask_img=img, metadata={"space": "MNI152NLin2009bAsym", "resolution": 2.0}
        )

    @pytest.fixture
    def atlas_img_cAsym(self):
        """Create atlas image in MNI152NLin2009cAsym space."""
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Atlas with integer labels
        data = np.zeros((91, 109, 91))
        data[40:50, 50:60, 40:50] = 1  # Region 1
        data[50:60, 50:60, 40:50] = 2  # Region 2

        return nib.Nifti1Image(data, affine)

    def test_transformation_needed_for_different_grids(
        self, mask_data_bAsym, atlas_img_cAsym, monkeypatch
    ):
        """ParcelAggregation SHOULD transform when grids differ.

        bAsym and cAsym have different voxel grids, so regridding is needed.
        """
        from lacuna.analysis.parcel_aggregation import ParcelAggregation

        transform_called = {"value": False}

        def mock_transform_image(*args, **kwargs):
            transform_called["value"] = True
            return atlas_img_cAsym

        monkeypatch.setattr("lacuna.spatial.transform.transform_image", mock_transform_image)

        analysis = ParcelAggregation()

        analysis._ensure_atlas_matches_input_space(
            atlas_img=atlas_img_cAsym,
            atlas_space="MNI152NLin2009cAsym",
            atlas_resolution=2.0,
            input_space="MNI152NLin2009bAsym",
            input_resolution=2.0,
            input_affine=mask_data_bAsym.mask_img.affine,
        )

        # cAsym and bAsym have different grids — transform IS needed
        assert transform_called["value"], (
            "transform_image was NOT called even though cAsym and bAsym "
            "have different voxel grids and require regridding"
        )


class TestResolutionValidationInBaseAnalysis:
    """Test that base analysis properly validates resolution is present."""

    @pytest.fixture
    def mask_data_missing_resolution(self):
        """Test removed: Cannot create SubjectData without resolution anymore."""
        pytest.skip("SubjectData now requires resolution at initialization (T006)")
        return None

    def test_detects_missing_resolution(self, mask_data_missing_resolution):
        """Test removed: SubjectData now validates resolution at initialization."""
        pytest.skip("Resolution validation moved to SubjectData.__init__ in T006")
