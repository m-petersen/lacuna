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
    from ldk import LesionData

    # Create synthetic lesion
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35] = 1

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    lesion_img = nib.Nifti1Image(data, affine)

    # Save to file
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(lesion_img, lesion_path)

    # Load using LesionData
    lesion_data = LesionData.from_nifti(
        lesion_path, metadata={"subject_id": "sub-test-001", "age": 45, "sex": "M"}
    )

    # Verify loaded correctly
    assert lesion_data.lesion_img.shape == shape
    assert lesion_data.anatomical_img is None
    assert lesion_data.metadata["subject_id"] == "sub-test-001"
    assert lesion_data.metadata["age"] == 45
    assert lesion_data.get_volume_mm3() > 0


def test_load_single_subject_with_anatomical(tmp_path):
    """Test loading single subject with both lesion and anatomical images."""
    from ldk import LesionData

    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    # Create lesion
    lesion_data = np.zeros(shape, dtype=np.uint8)
    lesion_data[30:35, 30:35, 30:35] = 1
    lesion_img = nib.Nifti1Image(lesion_data, affine)
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(lesion_img, lesion_path)

    # Create anatomical
    anat_data = np.random.rand(*shape).astype(np.float32) * 1000
    anat_img = nib.Nifti1Image(anat_data, affine)
    anat_path = tmp_path / "T1w.nii.gz"
    nib.save(anat_img, anat_path)

    # Load both
    lesion_data = LesionData.from_nifti(
        lesion_path=lesion_path,
        anatomical_path=anat_path,
        metadata={"subject_id": "sub-test-002", "site": "hospital_a"},
    )

    # Verify
    assert lesion_data.lesion_img.shape == shape
    assert lesion_data.anatomical_img.shape == shape
    assert np.array_equal(lesion_data.affine, affine)
    assert lesion_data.metadata["subject_id"] == "sub-test-002"


def test_single_subject_workflow_with_analysis(tmp_path):
    """Test complete workflow: load → analyze → save."""
    from ldk import LesionData
    from ldk.core.provenance import create_provenance_record
    from ldk.io import save_nifti

    # Create and save lesion
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35] = 1
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    lesion_img = nib.Nifti1Image(data, affine)
    input_path = tmp_path / "input_lesion.nii.gz"
    nib.save(lesion_img, input_path)

    # Load
    lesion_data = LesionData.from_nifti(input_path, metadata={"subject_id": "sub-workflow", "space": "MNI152_2mm"})

    # Simulate analysis
    volume_mm3 = lesion_data.get_volume_mm3()
    n_voxels = int(np.sum(lesion_data.lesion_img.get_fdata()))

    results = {
        "volume_mm3": volume_mm3,
        "n_voxels": n_voxels,
        "coordinate_space": lesion_data.get_coordinate_space(),
    }

    prov = create_provenance_record(
        function="volume_calculation",
        parameters={"method": "voxel_count"},
        version="0.1.0",
    )

    # Add results and provenance
    lesion_data_analyzed = lesion_data.add_result("VolumeAnalysis", results).add_provenance(prov)

    # Verify results attached
    assert "VolumeAnalysis" in lesion_data_analyzed.results
    assert lesion_data_analyzed.results["VolumeAnalysis"]["volume_mm3"] == volume_mm3
    assert len(lesion_data_analyzed.provenance) == 1

    # Save output
    output_path = tmp_path / "output_lesion.nii.gz"
    save_nifti(lesion_data_analyzed, output_path)

    # Verify saved correctly
    assert output_path.exists()
    loaded_img = nib.load(output_path)
    assert np.array_equal(loaded_img.get_fdata(), lesion_data_analyzed.lesion_img.get_fdata())


def test_single_subject_validation_catches_issues(tmp_path):
    """Test that validation catches common issues."""
    from ldk import LesionData
    from ldk.core.exceptions import ValidationError

    # Create lesion with 4D data (should be caught)
    shape = (64, 64, 64, 3)  # 4D is invalid!
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35, :] = 1

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    lesion_img = nib.Nifti1Image(data, affine)
    lesion_path = tmp_path / "bad_lesion.nii.gz"
    nib.save(lesion_img, lesion_path)

    # Should raise ValidationError for 4D image
    with pytest.raises(ValidationError, match="3D"):
        LesionData.from_nifti(lesion_path, metadata={"space": "MNI152_2mm"})


def test_single_subject_empty_mask_warning(tmp_path):
    """Test that empty lesion mask triggers warning."""
    from ldk import LesionData

    # Create empty lesion (all zeros)
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    lesion_img = nib.Nifti1Image(data, affine)
    lesion_path = tmp_path / "empty_lesion.nii.gz"
    nib.save(lesion_img, lesion_path)

    # Should warn about empty mask
    with pytest.warns(UserWarning, match="empty"):
        lesion_data = LesionData.from_nifti(lesion_path, metadata={"space": "MNI152_2mm"})
        lesion_data.validate()


def test_single_subject_metadata_persistence(tmp_path):
    """Test that metadata persists through operations."""
    from ldk import LesionData

    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[30:35, 30:35, 30:35] = 1
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    lesion_img = nib.Nifti1Image(data, affine)
    lesion_path = tmp_path / "lesion.nii.gz"
    nib.save(lesion_img, lesion_path)

    # Load with metadata
    metadata = {
        "subject_id": "sub-meta-001",
        "age": 55,
        "sex": "F",
        "diagnosis": "stroke",
        "days_post_stroke": 7,
    }

    lesion_data = LesionData.from_nifti(lesion_path, metadata=metadata)

    # Copy should preserve metadata
    lesion_data_copy = lesion_data.copy()
    assert lesion_data_copy.metadata == metadata

    # Serialization should preserve metadata
    data_dict = lesion_data.to_dict()
    assert data_dict["metadata"] == metadata
