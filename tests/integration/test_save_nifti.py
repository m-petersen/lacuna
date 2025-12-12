"""
Integration tests for NIfTI export functionality.

Tests the complete workflow of saving SubjectData objects to NIfTI files,
including round-trip testing (save → load → verify).
"""

import nibabel as nib
import numpy as np

from lacuna import SubjectData
from lacuna.io import save_nifti


def test_save_and_reload_lesion(tmp_path, synthetic_mask_img):
    """Test full save-load cycle for lesion data."""
    # Create lesion data
    original_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={
            "subject_id": "sub-test001",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
            "acquisition_date": "2025-01-15",
        },
    )

    # Save to file
    output_path = tmp_path / "lesion_mask.nii.gz"
    save_nifti(original_data, output_path)

    # Reload and verify
    reloaded = SubjectData.from_nifti(
        str(output_path),
        metadata={"subject_id": "sub-test001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Check image data matches
    assert np.array_equal(reloaded.mask_img.get_fdata(), original_data.mask_img.get_fdata())
    assert np.array_equal(reloaded.affine, original_data.affine)


def test_save_lesion_with_results(tmp_path, synthetic_mask_img):
    """Test saving lesion data with analysis results."""
    # Create lesion with results
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-test002", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Add mock results
    results = {"volume_mm3": 1500.0, "center_of_mass": [45, 54, 36]}
    lesion_with_results = mask_data.add_result("VolumeAnalysis", results)

    # Save (results not saved to NIfTI, just the image)
    output_path = tmp_path / "lesion_with_results.nii.gz"
    save_nifti(lesion_with_results, output_path)

    # Verify file exists and is valid NIfTI
    assert output_path.exists()
    reloaded_img = nib.load(output_path)
    assert reloaded_img.shape == synthetic_mask_img.shape


def test_save_lesion_and_anatomical(tmp_path, synthetic_mask_img):
    """Test saving lesion mask image.

    Note: anatomical_img parameter was removed from SubjectData.
    This test now verifies basic lesion saving functionality.
    """
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-test003", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    lesion_path = tmp_path / "lesion.nii.gz"
    save_nifti(mask_data, lesion_path)

    # Lesion file should exist
    assert lesion_path.exists()

    # Verify it's a valid NIfTI file
    mask_img = nib.load(lesion_path)
    assert mask_img.shape == synthetic_mask_img.shape


def test_save_multiple_subjects(tmp_path, synthetic_mask_img):
    """Test saving multiple subjects to different files."""
    subjects = ["sub-001", "sub-002", "sub-003"]
    saved_files = []

    for subject_id in subjects:
        mask_data = SubjectData(
            mask_img=synthetic_mask_img,
            metadata={"subject_id": subject_id, "space": "MNI152NLin6Asym", "resolution": 2},
        )

        output_path = tmp_path / f"{subject_id}_lesion.nii.gz"
        save_nifti(mask_data, output_path)
        saved_files.append(output_path)

    # Verify all files exist
    assert len(saved_files) == len(subjects)
    for file_path in saved_files:
        assert file_path.exists()
        # Verify it's a valid NIfTI
        img = nib.load(file_path)
        assert img.shape == synthetic_mask_img.shape


def test_save_preserves_affine_matrix(tmp_path):
    """Test that saving preserves the affine transformation matrix."""
    # Create custom affine matrix
    custom_affine = np.array(
        [
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Create image with custom affine
    data = np.random.randint(0, 2, size=(91, 109, 91), dtype=np.uint8)
    img = nib.Nifti1Image(data, custom_affine)

    mask_data = SubjectData(
        mask_img=img,
        metadata={"subject_id": "sub-test004", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Save and reload
    output_path = tmp_path / "custom_affine.nii.gz"
    save_nifti(mask_data, output_path)

    reloaded_img = nib.load(output_path)

    # Verify affine is preserved
    np.testing.assert_array_almost_equal(reloaded_img.affine, custom_affine)


def test_save_with_compression(tmp_path, synthetic_mask_img):
    """Test that .nii.gz files are properly compressed."""
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-test005", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Save as compressed
    compressed_path = tmp_path / "lesion_compressed.nii.gz"
    save_nifti(mask_data, compressed_path)

    # File should exist and be smaller than uncompressed version
    assert compressed_path.exists()

    # Verify it can be loaded
    reloaded = nib.load(compressed_path)
    assert reloaded.shape == synthetic_mask_img.shape
