"""
Contract tests for MaskData class.

These tests define the expected behavior of the core MaskData API contract.
Following TDD - these tests should FAIL until MaskData is implemented.
"""

import nibabel as nib
import numpy as np
import pytest


def test_mask_data_import():
    """Test that MaskData can be imported from lacuna.core."""
    from lacuna.core.mask_data import MaskData

    assert MaskData is not None


def test_mask_data_init_with_minimal_args(synthetic_mask_img, lesion_metadata):
    """Test MaskData initialization with minimal required arguments."""
    from lacuna.core.mask_data import MaskData

    lesion = MaskData(synthetic_mask_img, metadata=lesion_metadata)

    assert lesion is not None
    assert lesion.mask_img is synthetic_mask_img
    assert lesion.affine is not None
    assert isinstance(lesion.metadata, dict)
    assert "subject_id" in lesion.metadata
    assert isinstance(lesion.provenance, list)
    assert len(lesion.provenance) == 0
    assert isinstance(lesion.results, dict)
    assert len(lesion.results) == 0


def test_mask_data_init_with_metadata(synthetic_mask_img, lesion_metadata):
    """Test MaskData initialization with custom metadata."""
    from lacuna.core.mask_data import MaskData

    metadata = {
        "subject_id": "sub-001",
        "session_id": "ses-01",
        "age": 45,
        "space": "MNI152NLin6Asym",
        "resolution": 2,
    }

    lesion = MaskData(synthetic_mask_img, metadata=metadata)

    assert lesion.metadata["subject_id"] == "sub-001"
    assert lesion.metadata["session_id"] == "ses-01"
    assert lesion.metadata["age"] == 45


@pytest.mark.skip(reason="anatomical_img feature pending removal (T008)")
def test_mask_data_init_with_anatomical(synthetic_mask_img, lesion_metadata):
    """Test MaskData initialization with anatomical image."""
    from lacuna.core.mask_data import MaskData

    # Create matching anatomical image
    anat_data = np.random.rand(*synthetic_mask_img.shape).astype(np.float32)
    anat_img = nib.Nifti1Image(anat_data, synthetic_mask_img.affine)

    lesion = MaskData(synthetic_mask_img, anatomical_img=anat_img, metadata=lesion_metadata)

    assert lesion.anatomical_img is anat_img
    assert np.array_equal(lesion.anatomical_img.affine, lesion.affine)


def test_mask_data_init_validates_3d_image(synthetic_4d_img):
    """Test that MaskData rejects 4D images."""
    from lacuna.core.exceptions import ValidationError
    from lacuna.core.mask_data import MaskData

    with pytest.raises(ValidationError, match="3D"):
        MaskData(synthetic_4d_img)


@pytest.mark.skip(reason="anatomical_img feature pending removal (T008)")
def test_mask_data_init_validates_affine_mismatch():
    """Test that MaskData rejects mismatched anatomical affine."""
    from lacuna.core.exceptions import SpatialMismatchError
    from lacuna.core.mask_data import MaskData

    # Create lesion
    mask_data = np.ones((64, 64, 64), dtype=np.uint8)
    lesion_affine = np.eye(4)
    lesion_affine[0, 0] = 2.0
    mask_img = nib.Nifti1Image(mask_data, lesion_affine)

    # Create anatomical with different affine
    anat_data = np.random.rand(64, 64, 64).astype(np.float32)
    anat_affine = np.eye(4)
    anat_affine[0, 0] = 3.0  # Different voxel size
    anat_img = nib.Nifti1Image(anat_data, anat_affine)

    with pytest.raises(SpatialMismatchError):
        MaskData(mask_img, anatomical_img=anat_img)


def test_mask_data_from_nifti(tmp_path, synthetic_mask_img):
    """Test MaskData.from_nifti classmethod."""
    from lacuna.core.mask_data import MaskData

    # Save test image
    filepath = tmp_path / "test_lesion.nii.gz"
    nib.save(synthetic_mask_img, filepath)

    # Load via from_nifti (must provide space)
    lesion = MaskData.from_nifti(filepath, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    assert lesion is not None
    assert lesion.mask_img is not None
    assert np.array_equal(lesion.mask_img.get_fdata(), synthetic_mask_img.get_fdata())


def test_mask_data_from_nifti_with_metadata(tmp_path, synthetic_mask_img):
    """Test from_nifti with custom metadata."""
    from lacuna.core.mask_data import MaskData

    filepath = tmp_path / "test_lesion.nii.gz"
    nib.save(synthetic_mask_img, filepath)

    metadata = {
        "subject_id": "sub-test",
        "condition": "stroke",
        "space": "MNI152NLin6Asym",
        "resolution": 2,
    }
    lesion = MaskData.from_nifti(filepath, metadata=metadata)

    assert lesion.metadata["subject_id"] == "sub-test"
    assert lesion.metadata["condition"] == "stroke"


def test_mask_data_from_nifti_nonexistent_file():
    """Test from_nifti with nonexistent file raises error."""
    from lacuna.core.exceptions import NiftiLoadError
    from lacuna.core.mask_data import MaskData

    with pytest.raises((NiftiLoadError, FileNotFoundError)):
        MaskData.from_nifti(
            "/nonexistent/file.nii.gz", metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )


def test_mask_data_validate(synthetic_mask_img, lesion_metadata):
    """Test MaskData.validate method."""
    from lacuna.core.mask_data import MaskData

    lesion = MaskData(synthetic_mask_img, metadata=lesion_metadata)

    # Should pass validation
    assert lesion.validate() is True


def test_mask_data_get_volume_mm3(synthetic_mask_img, lesion_metadata):
    """Test get_volume_mm3 method."""
    from lacuna.core.mask_data import MaskData

    lesion = MaskData(synthetic_mask_img, metadata=lesion_metadata)
    volume = lesion.get_volume_mm3()

    assert isinstance(volume, float)
    assert volume > 0  # Synthetic lesion has nonzero voxels


def test_mask_data_get_coordinate_space(synthetic_mask_img, lesion_metadata):
    """Test get_coordinate_space method."""
    from lacuna.core.mask_data import MaskData

    lesion = MaskData(synthetic_mask_img, metadata=lesion_metadata)
    space = lesion.get_coordinate_space()

    assert isinstance(space, str)
    assert space == "MNI152NLin6Asym"  # From lesion_metadata fixture


def test_mask_data_copy(synthetic_mask_img, lesion_metadata):
    """Test MaskData.copy method creates independent copy."""
    from lacuna.core.mask_data import MaskData

    lesion = MaskData(
        synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )
    lesion_copy = lesion.copy()

    # Should be different objects
    assert lesion_copy is not lesion

    # But with same data
    assert lesion_copy.metadata["subject_id"] == "sub-001"
    assert np.array_equal(lesion_copy.affine, lesion.affine)


def test_mask_data_to_dict(synthetic_mask_img, lesion_metadata):
    """Test to_dict serialization."""
    from lacuna.core.mask_data import MaskData

    metadata = {"subject_id": "sub-001", "age": 45, "space": "MNI152NLin6Asym", "resolution": 2}
    lesion = MaskData(synthetic_mask_img, metadata=metadata)

    data_dict = lesion.to_dict()

    assert isinstance(data_dict, dict)
    assert "metadata" in data_dict
    assert "provenance" in data_dict
    assert "results" in data_dict
    assert data_dict["metadata"]["subject_id"] == "sub-001"


def test_mask_data_from_dict(synthetic_mask_img, lesion_metadata):
    """Test from_dict deserialization."""
    from lacuna.core.mask_data import MaskData

    # Create original
    metadata = {"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2}
    lesion = MaskData(synthetic_mask_img, metadata=metadata)

    # Serialize and deserialize
    data_dict = lesion.to_dict()
    lesion_restored = MaskData.from_dict(data_dict, synthetic_mask_img)

    assert lesion_restored.metadata["subject_id"] == "sub-001"


def test_mask_data_properties_are_readonly(synthetic_mask_img, lesion_metadata):
    """Test that properties cannot be directly modified."""
    from lacuna.core.mask_data import MaskData

    lesion = MaskData(synthetic_mask_img, metadata=lesion_metadata)

    # These should raise AttributeError if trying to set
    with pytest.raises(AttributeError):
        lesion.metadata = {"new": "dict"}

    with pytest.raises(AttributeError):
        lesion.provenance = []


# Fixtures


@pytest.fixture
def synthetic_mask_img():
    """Create a synthetic 3D lesion mask for testing."""
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)

    # Create small spherical lesion
    center = (32, 32, 32)
    radius = 5
    for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)):
        for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)):
            for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius + 1)):
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                if dist <= radius:
                    data[x, y, z] = 1

    affine = np.eye(4)
    affine[0, 0] = 2.0  # 2mm voxels
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0

    return nib.Nifti1Image(data, affine)


@pytest.fixture
def lesion_metadata():
    """Standard metadata with required space field."""
    return {"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2}


@pytest.fixture
def synthetic_4d_img():
    """Create a synthetic 4D image (should be rejected)."""
    shape = (64, 64, 64, 10)
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


# T017-T019: Contract tests for result attribute access
def test_mask_data_attribute_result_access(synthetic_mask_img, lesion_metadata):
    """Test that MaskData.AnalysisName returns results['AnalysisName']."""
    from lacuna.core.data_types import VoxelMap
    from lacuna.core.mask_data import MaskData

    mask_data = MaskData(synthetic_mask_img, metadata=lesion_metadata)

    # Manually add a result to simulate analysis output
    test_result = VoxelMap(
        name="TestAnalysis",
        space="MNI152NLin6Asym",
        resolution=2,
        data=synthetic_mask_img,  # Use actual NIfTI image
    )
    # Access internal _results since .results returns a deepcopy
    mask_data._results["TestAnalysis"] = {"default": test_result}

    # T017: Test attribute access returns same as dictionary access
    attr_result = mask_data.TestAnalysis
    dict_result = mask_data.results["TestAnalysis"]

    # Both should have the same content (though dict_result is a copy)
    assert isinstance(attr_result, dict)
    assert isinstance(dict_result, dict)
    assert "default" in attr_result
    assert "default" in dict_result


def test_mask_data_dictionary_result_access(synthetic_mask_img, lesion_metadata):
    """Test dictionary-based result access works as expected."""
    from lacuna.core.data_types import ParcelData
    from lacuna.core.mask_data import MaskData

    mask_data = MaskData(synthetic_mask_img, metadata=lesion_metadata)

    # Add multiple results with different keys
    result1 = ParcelData(
        name="AtlasA",
        data={"region1": 0.5, "region2": 0.7},
    )
    result2 = ParcelData(
        name="AtlasB",
        data={"region1": 0.3, "region2": 0.9},
    )

    mask_data._results["RegionalDamage"] = {
        "atlas_AtlasA": result1,
        "atlas_AtlasB": result2,
    }

    # T018: Test dictionary access pattern
    regional_results = mask_data.results["RegionalDamage"]
    assert isinstance(regional_results, dict)
    assert "atlas_AtlasA" in regional_results
    assert "atlas_AtlasB" in regional_results
    # Note: results returns a deepcopy, so we check value equality not identity
    assert regional_results["atlas_AtlasA"].name == result1.name
    assert regional_results["atlas_AtlasB"].name == result2.name


def test_mask_data_attribute_error_missing_result(synthetic_mask_img, lesion_metadata):
    """Test AttributeError with helpful message when result doesn't exist."""
    from lacuna.core.mask_data import MaskData

    mask_data = MaskData(synthetic_mask_img, metadata=lesion_metadata)

    # T019: Test that accessing non-existent result raises AttributeError
    with pytest.raises(AttributeError) as exc_info:
        _ = mask_data.NonExistentAnalysis

    # Verify the error message is helpful
    error_message = str(exc_info.value)
    assert "NonExistentAnalysis" in error_message
    assert "results" in error_message.lower()
