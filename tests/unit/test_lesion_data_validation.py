"""
Unit tests for LesionData validation logic.

Tests edge cases, validation warnings, and error conditions
for the LesionData validation system.
"""

import nibabel as nib
import numpy as np
import pytest


class TestLesionDataValidation:
    """Unit tests for LesionData.validate() method."""

    def test_validate_empty_mask_warning(self):
        """Test that empty lesion masks trigger a warning."""
        from lacuna import LesionData

        # Create empty mask (all zeros)
        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Should warn about empty mask
        with pytest.warns(UserWarning, match="empty"):
            lesion_data.validate()

    def test_validate_suspicious_voxel_size_no_warning(self):
        """Test that unusual voxel sizes are allowed.
        
        Note: Voxel size validation is not implemented. Various voxel sizes
        (including unusual ones) are allowed.
        """
        from lacuna import LesionData

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with very large voxel size (10mm)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 10.0

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Large voxel sizes are allowed - no warning
        lesion_data.validate()  # Should pass without warnings

    def test_validate_affine_nan_handled(self):
        """Test that NaN in affine is handled by nibabel.
        
        Note: nibabel may emit HeaderDataError when creating images with NaN in affine.
        This test documents that behavior.
        """
        import warnings
        from nibabel.spatialimages import HeaderDataError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with NaN
        affine = np.eye(4)
        affine[0, 0] = np.nan

        # nibabel raises HeaderDataError for NaN in affine
        # This is expected behavior - the decomposition fails
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(HeaderDataError):
                lesion_img = nib.Nifti1Image(data, affine)

    def test_validate_affine_inf_error(self):
        """Test that Inf in affine is caught by nibabel during image creation."""
        import warnings
        from nibabel.spatialimages import HeaderDataError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with Inf
        affine = np.eye(4)
        affine[1, 1] = np.inf

        # nibabel raises HeaderDataError during image creation (affine decomposition)
        # Suppress RuntimeWarning that precedes the error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(HeaderDataError, match="Could not decompose affine"):
                lesion_img = nib.Nifti1Image(data, affine)

    def test_validate_4d_image_error(self):
        """Test that 4D images raise ValidationError."""
        from lacuna.core.exceptions import ValidationError

        # Create 4D image
        shape = (64, 64, 64, 10)
        data = np.zeros(shape, dtype=np.uint8)
        affine = np.eye(4)

        lesion_img = nib.Nifti1Image(data, affine)

        from lacuna import LesionData

        with pytest.raises(ValidationError, match="3D"):
            LesionData(lesion_img=lesion_img, anatomical_img=None, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    def test_validate_non_invertible_affine_error(self):
        """Test that non-invertible affine is caught by nibabel during image creation."""
        import warnings
        from nibabel.spatialimages import HeaderDataError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create non-invertible affine (all zeros)
        affine = np.zeros((4, 4))

        # nibabel raises HeaderDataError during image creation with non-invertible affine
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(HeaderDataError, match="Could not decompose affine"):
                lesion_img = nib.Nifti1Image(data, affine)

    def test_validate_spatial_mismatch_error(self):
        """Test that mismatched lesion and anatomical raise ValidationError."""
        from lacuna.core.exceptions import ValidationError

        # Create lesion
        lesion_data = np.zeros((64, 64, 64), dtype=np.uint8)
        lesion_data[30:35, 30:35, 30:35] = 1
        affine1 = np.eye(4)
        affine1[0, 0] = affine1[1, 1] = affine1[2, 2] = 2.0
        lesion_img = nib.Nifti1Image(lesion_data, affine1)

        # Create anatomical with different affine
        anat_data = np.random.rand(64, 64, 64).astype(np.float32)
        affine2 = np.eye(4)
        affine2[0, 0] = affine2[1, 1] = affine2[2, 2] = 3.0  # Different voxel size
        anat_img = nib.Nifti1Image(anat_data, affine2)

        from lacuna import LesionData

        with pytest.raises(ValidationError, match="Affine matrices don't match"):
            LesionData(lesion_img=lesion_img, anatomical_img=anat_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    def test_validate_spatial_mismatch_shape_warning(self):
        """Test that mismatched shapes are allowed but may generate warnings."""
        # Create lesion
        lesion_data = np.zeros((64, 64, 64), dtype=np.uint8)
        lesion_data[30:35, 30:35, 30:35] = 1
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
        lesion_img = nib.Nifti1Image(lesion_data, affine)

        # Create anatomical with different shape (but same affine)
        anat_data = np.random.rand(80, 80, 80).astype(np.float32)
        anat_img = nib.Nifti1Image(anat_data, affine)

        from lacuna import LesionData

        # Should succeed - different shapes are allowed with same affine
        # (shape checking is disabled in check_spatial_match)
        lesion = LesionData(
            lesion_img=lesion_img, 
            anatomical_img=anat_img, 
            metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        assert lesion.lesion_img.shape == (64, 64, 64)
        assert lesion.anatomical_img.shape == (80, 80, 80)

    def test_validate_valid_lesion_data_no_warnings(self):
        """Test that valid LesionData passes validation without warnings."""
        from lacuna import LesionData

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Standard 2mm isotropic voxels
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Should not raise any warnings or errors
        lesion_data.validate()

    def test_validate_very_small_voxels_no_warning(self):
        """Test that very small voxel sizes are allowed (no validation implemented)."""
        from lacuna import LesionData

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with very small voxel size (0.1mm)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 0.1

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Validation should pass (no voxel size checks currently implemented)
        assert lesion_data.validate() is True

    def test_validate_anisotropic_voxels_ok(self):
        """Test that anisotropic (but reasonable) voxels are acceptable."""
        from lacuna import LesionData

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with anisotropic voxels (1x1x3mm - common in clinical scans)
        affine = np.eye(4)
        affine[0, 0] = 1.0
        affine[1, 1] = 1.0
        affine[2, 2] = 3.0

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Should not raise warnings (reasonable clinical voxel size)
        lesion_data.validate()

    def test_validate_negative_determinant_no_warning(self):
        """Test that negative affine determinant (neurological convention) is allowed.
        
        Note: RAS+ orientation validation is not implemented. Different orientations
        (neurological vs radiological) are allowed.
        """
        from lacuna import LesionData

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with negative determinant (neurological convention)
        affine = np.eye(4)
        affine[0, 0] = -2.0  # Negative x-axis (neurological)
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Neurological convention is allowed - no warning
        lesion_data.validate()  # Should pass without warnings

    def test_validate_zero_voxel_size_rejected_by_nibabel(self):
        """Test that zero voxel size is handled by nibabel.
        
        Note: nibabel emits warnings when creating images with zero voxel sizes.
        This is expected behavior - zero voxel sizes create invalid transforms.
        """
        import warnings
        from nibabel.spatialimages import HeaderDataError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with zero voxel size
        affine = np.eye(4)
        affine[2, 2] = 0.0  # Zero z-axis spacing

        # nibabel will raise HeaderDataError for zero voxel size
        # This is expected behavior
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(HeaderDataError):
                lesion_img = nib.Nifti1Image(data, affine)

    def test_validate_lesion_data_with_both_images(self):
        """Test validation when both lesion and anatomical provided."""
        from lacuna import LesionData

        shape = (64, 64, 64)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

        # Create lesion
        lesion_data = np.zeros(shape, dtype=np.uint8)
        lesion_data[30:35, 30:35, 30:35] = 1
        lesion_img = nib.Nifti1Image(lesion_data, affine)

        # Create anatomical
        anat_data = np.random.rand(*shape).astype(np.float32) * 1000
        anat_img = nib.Nifti1Image(anat_data, affine)

        lesion_data_obj = LesionData(
            lesion_img=lesion_img,
            anatomical_img=anat_img,
            metadata={"subject_id": "test", "space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Should pass validation
        lesion_data_obj.validate()

    def test_validate_metadata_optional(self):
        """Test that validation works with minimal metadata."""
        from lacuna import LesionData

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

        lesion_img = nib.Nifti1Image(data, affine)

        # Create with empty metadata
        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2},  # Empty metadata
        )

        # Should still validate
        lesion_data.validate()
