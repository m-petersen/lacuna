"""
Integration test for loading single subject lesion data.

Tests the complete workflow of loading a single subject's lesion data
from NIfTI files with proper metadata extraction and validation.
"""

import nibabel as nib
import numpy as np
import pytest


def test_load_single_subject_lesion_only(tmp_path):
    """Test loading single subject with lesion mask only."""
    from lacuna import SubjectData

    # Create synthetic lesion
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35] = 1

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    mask_img = nib.Nifti1Image(data, affine)

    # Save to file
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(mask_img, lesion_path)

    # Load using SubjectData
    mask_data = SubjectData.from_nifti(
        lesion_path,
        metadata={
            "subject_id": "sub-test-001",
            "age": 45,
            "sex": "M",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
        },
    )

    # Verify loaded correctly
    assert mask_data.mask_img.shape == shape
    assert mask_data.metadata["subject_id"] == "sub-test-001"
    assert mask_data.metadata["age"] == 45
    assert mask_data.get_volume_mm3() > 0


def test_load_single_subject_with_anatomical(tmp_path):
    """Test loading single subject lesion mask.

    Note: anatomical_path parameter was removed from SubjectData.from_nifti().
    This test now verifies basic lesion loading with metadata.
    """
    from lacuna import SubjectData

    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    # Create lesion
    mask_data = np.zeros(shape, dtype=np.uint8)
    mask_data[30:35, 30:35, 30:35] = 1
    mask_img = nib.Nifti1Image(mask_data, affine)
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(mask_img, lesion_path)

    # Load lesion with site metadata
    mask_data = SubjectData.from_nifti(
        lesion_path,
        metadata={
            "subject_id": "sub-test-002",
            "site": "hospital_a",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
        },
    )

    # Verify
    assert mask_data.mask_img.shape == shape
    assert np.array_equal(mask_data.affine, affine)
    assert mask_data.metadata["subject_id"] == "sub-test-002"


def test_single_subject_workflow_with_analysis(tmp_path):
    """Test complete workflow: load → analyze → save."""
    from lacuna import SubjectData
    from lacuna.core.provenance import create_provenance_record
    from lacuna.io import save_nifti

    # Create and save lesion
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35] = 1
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    mask_img = nib.Nifti1Image(data, affine)
    input_path = tmp_path / "input_lesion.nii.gz"
    nib.save(mask_img, input_path)

    # Load
    mask_data = SubjectData.from_nifti(
        input_path,
        metadata={"subject_id": "sub-workflow", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Simulate analysis
    volume_mm3 = mask_data.get_volume_mm3()
    n_voxels = int(np.sum(mask_data.mask_img.get_fdata()))

    results = {
        "volume_mm3": volume_mm3,
        "n_voxels": n_voxels,
        "coordinate_space": mask_data.get_coordinate_space(),
    }

    prov = create_provenance_record(
        function="volume_calculation",
        parameters={"method": "voxel_count"},
        version="0.1.0",
    )

    # Add results and provenance
    mask_data_analyzed = mask_data.add_result("VolumeAnalysis", results).add_provenance(prov)

    # Verify results attached
    assert "VolumeAnalysis" in mask_data_analyzed.results
    assert mask_data_analyzed.results["VolumeAnalysis"]["volume_mm3"] == volume_mm3
    assert len(mask_data_analyzed.provenance) == 1

    # Save output
    output_path = tmp_path / "output_lesion.nii.gz"
    save_nifti(mask_data_analyzed, output_path)

    # Verify saved correctly
    assert output_path.exists()
    loaded_img = nib.load(output_path)
    assert np.array_equal(loaded_img.get_fdata(), mask_data_analyzed.mask_img.get_fdata())


def test_single_subject_validation_catches_issues(tmp_path):
    """Test that validation catches common issues."""
    from lacuna import SubjectData
    from lacuna.core.exceptions import ValidationError

    # Create lesion with 4D data (should be caught)
    shape = (64, 64, 64, 3)  # 4D is invalid!
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35, :] = 1

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    mask_img = nib.Nifti1Image(data, affine)
    lesion_path = tmp_path / "bad_lesion.nii.gz"
    nib.save(mask_img, lesion_path)

    # Should raise ValidationError for 4D image
    with pytest.raises(ValidationError, match="3D"):
        SubjectData.from_nifti(lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2})


def test_single_subject_empty_mask_warning(tmp_path):
    """Test that empty lesion mask triggers warning."""
    from lacuna import SubjectData

    # Create empty lesion (all zeros)
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    mask_img = nib.Nifti1Image(data, affine)
    lesion_path = tmp_path / "empty_lesion.nii.gz"
    nib.save(mask_img, lesion_path)

    # Should warn about empty mask
    with pytest.warns(UserWarning, match="empty"):
        mask_data = SubjectData.from_nifti(
            lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )
        mask_data.validate()


def test_single_subject_metadata_persistence(tmp_path):
    """Test that metadata persists through operations."""
    from lacuna import SubjectData

    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35] = 1
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    mask_img = nib.Nifti1Image(data, affine)
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(mask_img, lesion_path)

    # Load with metadata
    metadata = {
        "subject_id": "sub-meta-001",
        "age": 55,
        "sex": "F",
        "diagnosis": "stroke",
        "days_post_stroke": 7,
        "space": "MNI152NLin6Asym",
        "resolution": 2,
    }

    mask_data = SubjectData.from_nifti(lesion_path, metadata=metadata)

    # Copy should preserve metadata
    mask_data_copy = mask_data.copy()
    assert mask_data_copy.metadata == metadata

    # Serialization should preserve metadata
    data_dict = mask_data.to_dict()
    assert data_dict["metadata"] == metadata
