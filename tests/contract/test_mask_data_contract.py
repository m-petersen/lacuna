"""
Contract tests for MaskData API requirements.

Tests the core contract of MaskData:
- Binary mask validation (0/1 values only)
- Required metadata fields (space, resolution)
- Removed deprecated features (anatomical_img, registration)
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.mask_data import MaskData


class TestBinaryMaskValidation:
    """Test that MaskData enforces binary masks (0/1 values only)."""

    def test_binary_mask_accepted(self):
        """Binary mask (only 0 and 1 values) should be accepted."""
        # Create binary mask
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        # Should not raise
        result = MaskData(
            mask_img=mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2},
        )

        assert result is not None

    def test_continuous_mask_rejected(self):
        """Continuous mask (float values) should be rejected."""
        # Create continuous mask
        mask_data = np.random.rand(10, 10, 10)
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(ValueError, match="binary mask with only 0 and 1 values"):
            MaskData(
                mask_img=mask_img,
                metadata={"space": "MNI152NLin6Asym", "resolution": 2},
            )

    def test_integer_mask_rejected(self):
        """Integer mask (values > 1) should be rejected."""
        # Create integer mask with labels
        mask_data = np.zeros((10, 10, 10), dtype=np.float32)
        mask_data[3:7, 3:7, 3:7] = 5  # Label 5
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(ValueError, match="binary mask with only 0 and 1 values"):
            MaskData(
                mask_img=mask_img,
                metadata={"space": "MNI152NLin6Asym", "resolution": 2},
            )

    def test_error_message_suggests_binarization(self):
        """Error message should suggest binarization."""
        mask_data = np.random.rand(10, 10, 10)
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(ValueError, match="Please binarize your lesion mask"):
            MaskData(
                mask_img=mask_img,
                metadata={"space": "MNI152NLin6Asym", "resolution": 2},
            )


class TestAnatomicalImgRejection:
    """Test that anatomical_img parameter is not accepted (removed feature)."""

    def test_anatomical_img_parameter_does_not_exist(self):
        """MaskData should not accept anatomical_img parameter."""
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        anat_data = np.random.rand(10, 10, 10)
        anat_img = nib.Nifti1Image(anat_data, affine=np.eye(4))

        # Should raise TypeError for unexpected keyword argument
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaskData(
                mask_img=mask_img,
                anatomical_img=anat_img,  # Should be rejected
                metadata={"space": "MNI152NLin6Asym", "resolution": 2},
            )


class TestSpaceInference:
    """Test that space is always taken from metadata (no provenance fallback)."""

    def test_space_required_in_metadata(self):
        """Space must be provided as parameter or in metadata."""
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(ValueError, match="Coordinate space must be specified"):
            MaskData(
                mask_img=mask_img,
                metadata={},  # No space or resolution
            )

    def test_resolution_required_in_metadata(self):
        """Resolution must be provided as parameter or in metadata."""
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(ValueError, match="Spatial resolution must be specified"):
            MaskData(
                mask_img=mask_img,
                space="MNI152NLin6Asym",  # Space provided but not resolution
                metadata={},
            )

    def test_unsupported_space_rejected(self):
        """Unsupported template spaces should be rejected."""
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(ValueError, match="Invalid space"):
            MaskData(
                mask_img=mask_img,
                metadata={"space": "native", "resolution": 2},  # Unsupported
            )

    def test_space_property_matches_metadata(self):
        """Space should be accessible from metadata."""
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        mask_data_obj = MaskData(
            mask_img=mask_img,
            metadata={"space": "MNI152NLin2009cAsym", "resolution": 2},
        )

        # Space and resolution accessible via metadata
        assert mask_data_obj.metadata["space"] == "MNI152NLin2009cAsym"
        assert mask_data_obj.metadata["resolution"] == 2

    def test_supported_spaces_in_error_message(self):
        """Error message should list supported spaces."""
        mask_data = np.zeros((10, 10, 10))
        mask_data[3:7, 3:7, 3:7] = 1
        mask_img = nib.Nifti1Image(mask_data, affine=np.eye(4))

        with pytest.raises(
            ValueError,
            match="MNI152NLin6Asym.*MNI152NLin2009aAsym.*MNI152NLin2009cAsym",
        ):
            MaskData(
                mask_img=mask_img,
                metadata={},  # Missing space and resolution
            )
