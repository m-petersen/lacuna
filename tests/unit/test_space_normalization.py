"""Unit tests for space variant canonicalization and validation utilities.

Tests consistent handling of space aliases (e.g., MNI152NLin2009aAsym → cAsym)
across the codebase to avoid unnecessary transformations.
"""

import pytest
import numpy as np
import nibabel as nib


class TestCanonicalizeSpaceVariant:
    """Test canonicalize_space_variant function."""

    def test_normalizes_aAsym_to_cAsym(self):
        """MNI152NLin2009aAsym should canonicalize to cAsym (canonical form)."""
        from lacuna.core.spaces import canonicalize_space_variant

        result = canonicalize_space_variant("MNI152NLin2009aAsym")
        assert result == "MNI152NLin2009cAsym"

    def test_normalizes_bAsym_to_cAsym(self):
        """MNI152NLin2009bAsym should canonicalize to cAsym (canonical form)."""
        from lacuna.core.spaces import canonicalize_space_variant

        result = canonicalize_space_variant("MNI152NLin2009bAsym")
        assert result == "MNI152NLin2009cAsym"

    def test_cAsym_is_identity(self):
        """cAsym is already canonical - should return unchanged."""
        from lacuna.core.spaces import canonicalize_space_variant

        result = canonicalize_space_variant("MNI152NLin2009cAsym")
        assert result == "MNI152NLin2009cAsym"

    def test_other_spaces_unchanged(self):
        """Non-aliased spaces should return unchanged."""
        from lacuna.core.spaces import canonicalize_space_variant

        # MNI152NLin6Asym is not an alias
        result = canonicalize_space_variant("MNI152NLin6Asym")
        assert result == "MNI152NLin6Asym"

        # Custom/native spaces unchanged
        result = canonicalize_space_variant("native")
        assert result == "native"

    def test_case_sensitive(self):
        """Space identifiers are case-sensitive."""
        from lacuna.core.spaces import canonicalize_space_variant

        # Should not match due to case
        result = canonicalize_space_variant("mni152nlin2009aasym")
        assert result == "mni152nlin2009aasym"  # Unchanged


class TestSpacesAreEquivalent:
    """Test spaces_are_equivalent helper for comparing spaces with aliases."""

    def test_aAsym_equals_cAsym(self):
        """aAsym and cAsym are anatomically equivalent."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert spaces_are_equivalent("MNI152NLin2009aAsym", "MNI152NLin2009cAsym")
        assert spaces_are_equivalent("MNI152NLin2009cAsym", "MNI152NLin2009aAsym")

    def test_bAsym_equals_cAsym(self):
        """bAsym and cAsym are anatomically equivalent."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert spaces_are_equivalent("MNI152NLin2009bAsym", "MNI152NLin2009cAsym")
        assert spaces_are_equivalent("MNI152NLin2009cAsym", "MNI152NLin2009bAsym")

    def test_aAsym_equals_bAsym(self):
        """aAsym and bAsym are anatomically equivalent (both normalize to cAsym)."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert spaces_are_equivalent("MNI152NLin2009aAsym", "MNI152NLin2009bAsym")

    def test_identical_spaces_equal(self):
        """Identical space strings are equivalent."""
        from lacuna.core.spaces import spaces_are_equivalent

        assert spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin6Asym")
        assert spaces_are_equivalent("MNI152NLin2009cAsym", "MNI152NLin2009cAsym")

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

    def test_accepts_equivalent_spaces(self):
        """Should accept equivalent spaces without error."""
        from lacuna.core.spaces import validate_space_compatibility

        # Should not raise
        validate_space_compatibility(
            actual_space="MNI152NLin2009aAsym",
            expected_space="MNI152NLin2009cAsym",
            context="test"
        )

    def test_accepts_identical_spaces(self):
        """Should accept identical spaces."""
        from lacuna.core.spaces import validate_space_compatibility

        # Should not raise
        validate_space_compatibility(
            actual_space="MNI152NLin6Asym",
            expected_space="MNI152NLin6Asym",
            context="test"
        )

    def test_rejects_incompatible_spaces(self):
        """Should raise error for incompatible spaces."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="Space mismatch"):
            validate_space_compatibility(
                actual_space="MNI152NLin6Asym",
                expected_space="MNI152NLin2009cAsym",
                context="test analysis"
            )

    def test_error_message_includes_context(self):
        """Error message should include context for debugging."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="test analysis"):
            validate_space_compatibility(
                actual_space="native",
                expected_space="MNI152NLin6Asym",
                context="test analysis"
            )

    def test_error_message_shows_both_spaces(self):
        """Error message should show both actual and expected spaces."""
        from lacuna.core.spaces import validate_space_compatibility

        with pytest.raises(ValueError, match="native.*MNI152NLin6Asym"):
            validate_space_compatibility(
                actual_space="native",
                expected_space="MNI152NLin6Asym",
                context="test"
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
                suggest_transform=True
            )


class TestValidateSpaceAndResolution:
    """Test validate_space_and_resolution for LesionData and metadata validation."""

    def test_accepts_valid_space_and_resolution(self):
        """Should accept valid space + resolution combination."""
        from lacuna.core.spaces import validate_space_and_resolution

        # Should not raise
        validate_space_and_resolution(
            space="MNI152NLin6Asym",
            resolution=2.0
        )

    def test_rejects_missing_resolution_when_space_specified(self):
        """Should raise error if space specified but resolution is None."""
        from lacuna.core.spaces import validate_space_and_resolution

        with pytest.raises(ValueError, match="[Rr]esolution.*required"):
            validate_space_and_resolution(
                space="MNI152NLin6Asym",
                resolution=None
            )

    def test_rejects_invalid_resolution(self):
        """Should raise error for invalid resolution values."""
        from lacuna.core.spaces import validate_space_and_resolution

        with pytest.raises(ValueError, match="resolution"):
            validate_space_and_resolution(
                space="MNI152NLin6Asym",
                resolution=3.0  # Invalid - must be 0.5, 1, or 2
            )

    def test_accepts_none_space_with_none_resolution(self):
        """Should accept both None if data is in native/unknown space."""
        from lacuna.core.spaces import validate_space_and_resolution

        # Should not raise
        validate_space_and_resolution(
            space=None,
            resolution=None
        )

    def test_validates_space_identifier(self):
        """Should validate space identifier is recognized."""
        from lacuna.core.spaces import validate_space_and_resolution

        with pytest.raises(ValueError, match="Unknown.*space"):
            validate_space_and_resolution(
                space="InvalidSpaceName",
                resolution=2.0
            )


class TestAtlasAggregationSpaceHandling:
    """Test that AtlasAggregation correctly handles space aliases."""

    @pytest.fixture
    def lesion_data_aAsym(self):
        """Create LesionData in MNI152NLin2009aAsym space."""
        from lacuna.core.lesion_data import LesionData

        # 2mm MNI affine
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        img = nib.Nifti1Image(np.ones((91, 109, 91)), affine)

        return LesionData(
            lesion_img=img,
            metadata={
                "space": "MNI152NLin2009aAsym",
                "resolution": 2.0
            }
        )

    @pytest.fixture
    def atlas_img_cAsym(self):
        """Create atlas image in MNI152NLin2009cAsym space."""
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        # Atlas with integer labels
        data = np.zeros((91, 109, 91))
        data[40:50, 50:60, 40:50] = 1  # Region 1
        data[50:60, 50:60, 40:50] = 2  # Region 2

        return nib.Nifti1Image(data, affine)

    def test_no_transformation_for_equivalent_spaces(
        self, lesion_data_aAsym, atlas_img_cAsym, monkeypatch
    ):
        """AtlasAggregation should NOT transform atlas when spaces are equivalent.

        This is the key bug: Previously atlas_aggregation.py line 398 did:
            if atlas_space == input_space:
                return atlas_img
        
        But "MNI152NLin2009aAsym" != "MNI152NLin2009cAsym" (string comparison)
        So it attempted transformation even though they're anatomically equivalent.
        
        After fix, should normalize both spaces and recognize equivalence.
        """
        from lacuna.analysis.atlas_aggregation import AtlasAggregation

        # Create mock that would fail if transformation attempted
        transform_called = {"value": False}

        def mock_transform_image(*args, **kwargs):
            transform_called["value"] = True
            # Return original to avoid actual transformation
            return atlas_img_cAsym

        # Monkeypatch transform at the point where it's imported
        monkeypatch.setattr(
            "lacuna.spatial.transform.transform_image",
            mock_transform_image
        )

        # Create analysis instance
        analysis = AtlasAggregation()

        # Call _ensure_atlas_matches_input_space (private method)
        result = analysis._ensure_atlas_matches_input_space(
            atlas_img=atlas_img_cAsym,
            atlas_space="MNI152NLin2009cAsym",
            atlas_resolution=2.0,
            input_space="MNI152NLin2009aAsym",
            input_resolution=2.0,
            input_affine=lesion_data_aAsym.lesion_img.affine
        )

        # Assert: Transformation should NOT have been called
        # (spaces are equivalent after normalization)
        assert not transform_called["value"], (
            "transform_image was called even though spaces are equivalent "
            "(aAsym and cAsym are anatomically identical)"
        )

        # Result should be original atlas unchanged
        assert result is atlas_img_cAsym


class TestResolutionValidationInBaseAnalysis:
    """Test that base analysis properly validates resolution is present."""

    @pytest.fixture
    def lesion_data_missing_resolution(self):
        """Create LesionData with space but no resolution."""
        from lacuna.core.lesion_data import LesionData

        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        img = nib.Nifti1Image(np.ones((91, 109, 91)), affine)

        return LesionData(
            lesion_img=img,
            metadata={
                "space": "MNI152NLin6Asym"
                # Note: no resolution key
            }
        )

    def test_detects_missing_resolution(self, lesion_data_missing_resolution):
        """BaseAnalysis should detect and raise error for missing resolution.

        Currently base.py line 394 gets resolution with .get() returning None,
        and line 398 checks if both are not None - so missing resolution
        is silently ignored instead of raising an error.
        """
        from lacuna.analysis.base import BaseAnalysis

        class TestAnalysis(BaseAnalysis):
            TARGET_SPACE = "MNI152NLin2009cAsym"
            TARGET_RESOLUTION = 2.0

            def _validate_inputs(self, lesion_data):
                """Implement required abstract method."""
                pass

            def _run_analysis(self, lesion_data):
                return []

        analysis = TestAnalysis()

        # Should raise clear error about missing resolution
        with pytest.raises(ValueError, match="[Rr]esolution.*required"):
            analysis.run(lesion_data_missing_resolution)
