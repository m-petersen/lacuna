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
        from ldk import LesionData

        # Create empty mask (all zeros)
        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

        lesion_img = nib.Nifti1Image(data, affine)

        lesion_data = LesionData(
            lesion_img=lesion_img,
            anatomical_img=None,
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should warn about empty mask
        with pytest.warns(UserWarning, match="empty"):
            lesion_data.validate()

    def test_validate_suspicious_voxel_size_warning(self):
        """Test that suspicious voxel sizes trigger warnings."""
        from ldk import LesionData

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
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should warn about suspicious voxel size
        with pytest.warns(UserWarning, match="voxel size"):
            lesion_data.validate()

    def test_validate_affine_nan_error(self):
        """Test that NaN in affine raises ValidationError."""
        from ldk.core.exceptions import ValidationError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with NaN
        affine = np.eye(4)
        affine[0, 0] = np.nan

        lesion_img = nib.Nifti1Image(data, affine)

        # Should raise ValidationError during construction
        from ldk import LesionData

        with pytest.raises(ValidationError, match="NaN"):
            LesionData(lesion_img=lesion_img, anatomical_img=None, metadata={"space": "MNI152_2mm"})

    def test_validate_affine_inf_error(self):
        """Test that Inf in affine raises ValidationError."""
        from ldk.core.exceptions import ValidationError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with Inf
        affine = np.eye(4)
        affine[1, 1] = np.inf

        lesion_img = nib.Nifti1Image(data, affine)

        from ldk import LesionData

        with pytest.raises(ValidationError, match="Inf"):
            LesionData(lesion_img=lesion_img, anatomical_img=None, metadata={"space": "MNI152_2mm"})

    def test_validate_4d_image_error(self):
        """Test that 4D images raise ValidationError."""
        from ldk.core.exceptions import ValidationError

        # Create 4D image
        shape = (64, 64, 64, 10)
        data = np.zeros(shape, dtype=np.uint8)
        affine = np.eye(4)

        lesion_img = nib.Nifti1Image(data, affine)

        from ldk import LesionData

        with pytest.raises(ValidationError, match="3D"):
            LesionData(lesion_img=lesion_img, anatomical_img=None, metadata={"space": "MNI152_2mm"})

    def test_validate_non_invertible_affine_error(self):
        """Test that non-invertible affine raises ValidationError."""
        from ldk.core.exceptions import ValidationError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create non-invertible affine (all zeros)
        affine = np.zeros((4, 4))

        lesion_img = nib.Nifti1Image(data, affine)

        from ldk import LesionData

        with pytest.raises(ValidationError, match="invertible"):
            LesionData(lesion_img=lesion_img, anatomical_img=None, metadata={"space": "MNI152_2mm"})

    def test_validate_spatial_mismatch_error(self):
        """Test that mismatched lesion and anatomical raise ValidationError."""
        from ldk.core.exceptions import ValidationError

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

        from ldk import LesionData

        with pytest.raises(ValidationError, match="spatial"):
            LesionData(lesion_img=lesion_img, anatomical_img=anat_img, metadata={"space": "MNI152_2mm"})

    def test_validate_spatial_mismatch_shape_error(self):
        """Test that mismatched shapes raise ValidationError."""
        from ldk.core.exceptions import ValidationError

        # Create lesion
        lesion_data = np.zeros((64, 64, 64), dtype=np.uint8)
        lesion_data[30:35, 30:35, 30:35] = 1
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
        lesion_img = nib.Nifti1Image(lesion_data, affine)

        # Create anatomical with different shape
        anat_data = np.random.rand(80, 80, 80).astype(np.float32)
        anat_img = nib.Nifti1Image(anat_data, affine)

        from ldk import LesionData

        with pytest.raises(ValidationError, match="spatial"):
            LesionData(lesion_img=lesion_img, anatomical_img=anat_img, metadata={"space": "MNI152_2mm"})

    def test_validate_valid_lesion_data_no_warnings(self):
        """Test that valid LesionData passes validation without warnings."""
        from ldk import LesionData

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
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should not raise any warnings or errors
        lesion_data.validate()

    def test_validate_very_small_voxels_warning(self):
        """Test that very small voxel sizes trigger warnings."""
        from ldk import LesionData

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
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should warn about suspicious voxel size
        with pytest.warns(UserWarning, match="voxel size"):
            lesion_data.validate()

    def test_validate_anisotropic_voxels_ok(self):
        """Test that anisotropic (but reasonable) voxels are acceptable."""
        from ldk import LesionData

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
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should not raise warnings (reasonable clinical voxel size)
        lesion_data.validate()

    def test_validate_negative_determinant_warning(self):
        """Test that negative affine determinant (neurological convention) triggers warning."""
        from ldk import LesionData

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
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should warn about non-RAS+ orientation
        with pytest.warns(UserWarning, match="RAS\\+|orientation"):
            lesion_data.validate()

    def test_validate_zero_voxel_size_error(self):
        """Test that zero voxel size raises ValidationError."""
        from ldk.core.exceptions import ValidationError

        shape = (64, 64, 64)
        data = np.zeros(shape, dtype=np.uint8)
        data[30:35, 30:35, 30:35] = 1

        # Create affine with zero voxel size
        affine = np.eye(4)
        affine[2, 2] = 0.0  # Zero z-axis spacing

        lesion_img = nib.Nifti1Image(data, affine)

        from ldk import LesionData

        with pytest.raises(ValidationError, match="voxel size|invertible"):
            LesionData(lesion_img=lesion_img, anatomical_img=None, metadata={"space": "MNI152_2mm"})

    def test_validate_lesion_data_with_both_images(self):
        """Test validation when both lesion and anatomical provided."""
        from ldk import LesionData

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
            metadata={"subject_id": "test", "space": "MNI152_2mm"},
        )

        # Should pass validation
        lesion_data_obj.validate()

    def test_validate_metadata_optional(self):
        """Test that validation works with minimal metadata."""
        from ldk import LesionData

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
            metadata={"space": "MNI152_2mm"},  # Empty metadata
        )

        # Should still validate
        lesion_data.validate()
