"""
Contract tests for LesionData class.

These tests define the expected behavior of the core LesionData API contract.
Following TDD - these tests should FAIL until LesionData is implemented.
"""
import nibabel as nib
import numpy as np
import pytest


def test_lesion_data_import():
    """Test that LesionData can be imported from ldk.core."""
    from ldk.core.lesion_data import LesionData
    assert LesionData is not None


def test_lesion_data_init_with_minimal_args(synthetic_lesion_img, lesion_metadata):
    """Test LesionData initialization with minimal required arguments."""
    from ldk.core.lesion_data import LesionData
    
    lesion = LesionData(synthetic_lesion_img, metadata=lesion_metadata)
    
    assert lesion is not None
    assert lesion.lesion_img is synthetic_lesion_img
    assert lesion.affine is not None
    assert isinstance(lesion.metadata, dict)
    assert "subject_id" in lesion.metadata
    assert isinstance(lesion.provenance, list)
    assert len(lesion.provenance) == 0
    assert isinstance(lesion.results, dict)
    assert len(lesion.results) == 0


def test_lesion_data_init_with_metadata(synthetic_lesion_img, lesion_metadata):
    """Test LesionData initialization with custom metadata."""
    from ldk.core.lesion_data import LesionData
    
    metadata = {
        "subject_id": "sub-001",
        "session_id": "ses-01",
        "age": 45,
    }
    
    lesion = LesionData(synthetic_lesion_img, metadata=metadata)
    
    assert lesion.metadata["subject_id"] == "sub-001"
    assert lesion.metadata["session_id"] == "ses-01"
    assert lesion.metadata["age"] == 45


def test_lesion_data_init_with_anatomical(synthetic_lesion_img, lesion_metadata):
    """Test LesionData initialization with anatomical image."""
    from ldk.core.lesion_data import LesionData
    
    # Create matching anatomical image
    anat_data = np.random.rand(*synthetic_lesion_img.shape).astype(np.float32)
    anat_img = nib.Nifti1Image(anat_data, synthetic_lesion_img.affine)
    
    lesion = LesionData(synthetic_lesion_img, anatomical_img=anat_img, metadata=lesion_metadata)
    
    assert lesion.anatomical_img is anat_img
    assert np.array_equal(lesion.anatomical_img.affine, lesion.affine)


def test_lesion_data_init_validates_3d_image(synthetic_4d_img):
    """Test that LesionData rejects 4D images."""
    from ldk.core.lesion_data import LesionData
    from ldk.core.exceptions import ValidationError
    
    with pytest.raises(ValidationError, match="3D"):
        LesionData(synthetic_4d_img)


def test_lesion_data_init_validates_affine_mismatch():
    """Test that LesionData rejects mismatched anatomical affine."""
    from ldk.core.lesion_data import LesionData
    from ldk.core.exceptions import SpatialMismatchError
    
    # Create lesion
    lesion_data = np.ones((64, 64, 64), dtype=np.uint8)
    lesion_affine = np.eye(4)
    lesion_affine[0, 0] = 2.0
    lesion_img = nib.Nifti1Image(lesion_data, lesion_affine)
    
    # Create anatomical with different affine
    anat_data = np.random.rand(64, 64, 64).astype(np.float32)
    anat_affine = np.eye(4)
    anat_affine[0, 0] = 3.0  # Different voxel size
    anat_img = nib.Nifti1Image(anat_data, anat_affine)
    
    with pytest.raises(SpatialMismatchError):
        LesionData(lesion_img, anatomical_img=anat_img)


def test_lesion_data_from_nifti(tmp_path, synthetic_lesion_img):
    """Test LesionData.from_nifti classmethod."""
    from ldk.core.lesion_data import LesionData
    
    # Save test image
    filepath = tmp_path / "test_lesion.nii.gz"
    nib.save(synthetic_lesion_img, filepath)
    
    # Load via from_nifti
    lesion = LesionData.from_nifti(filepath)
    
    assert lesion is not None
    assert lesion.lesion_img is not None
    assert np.array_equal(lesion.lesion_img.get_fdata(), synthetic_lesion_img.get_fdata())


def test_lesion_data_from_nifti_with_metadata(tmp_path, synthetic_lesion_img):
    """Test from_nifti with custom metadata."""
    from ldk.core.lesion_data import LesionData
    
    filepath = tmp_path / "test_lesion.nii.gz"
    nib.save(synthetic_lesion_img, filepath)
    
    metadata = {"subject_id": "sub-test", "condition": "stroke"}
    lesion = LesionData.from_nifti(filepath, metadata=metadata)
    
    assert lesion.metadata["subject_id"] == "sub-test"
    assert lesion.metadata["condition"] == "stroke"


def test_lesion_data_from_nifti_nonexistent_file():
    """Test from_nifti with nonexistent file raises error."""
    from ldk.core.lesion_data import LesionData
    from ldk.core.exceptions import NiftiLoadError
    
    with pytest.raises((NiftiLoadError, FileNotFoundError)):
        LesionData.from_nifti("/nonexistent/file.nii.gz")


def test_lesion_data_validate(synthetic_lesion_img, lesion_metadata):
    """Test LesionData.validate method."""
    from ldk.core.lesion_data import LesionData
    
    lesion = LesionData(synthetic_lesion_img, metadata=lesion_metadata)
    
    # Should pass validation
    assert lesion.validate() is True


def test_lesion_data_get_volume_mm3(synthetic_lesion_img, lesion_metadata):
    """Test get_volume_mm3 method."""
    from ldk.core.lesion_data import LesionData
    
    lesion = LesionData(synthetic_lesion_img, metadata=lesion_metadata)
    volume = lesion.get_volume_mm3()
    
    assert isinstance(volume, float)
    assert volume > 0  # Synthetic lesion has nonzero voxels


def test_lesion_data_get_coordinate_space(synthetic_lesion_img, lesion_metadata):
    """Test get_coordinate_space method."""
    from ldk.core.lesion_data import LesionData
    
    lesion = LesionData(synthetic_lesion_img, metadata=lesion_metadata)
    space = lesion.get_coordinate_space()
    
    assert isinstance(space, str)
    assert space == "native"  # Default for new lesions


def test_lesion_data_copy(synthetic_lesion_img, lesion_metadata):
    """Test LesionData.copy method creates independent copy."""
    from ldk.core.lesion_data import LesionData
    
    lesion = LesionData(synthetic_lesion_img, metadata={"subject_id": "sub-001", "space": "MNI152_2mm"})
    lesion_copy = lesion.copy()
    
    # Should be different objects
    assert lesion_copy is not lesion
    
    # But with same data
    assert lesion_copy.metadata["subject_id"] == "sub-001"
    assert np.array_equal(lesion_copy.affine, lesion.affine)


def test_lesion_data_to_dict(synthetic_lesion_img, lesion_metadata):
    """Test to_dict serialization."""
    from ldk.core.lesion_data import LesionData
    
    metadata = {"subject_id": "sub-001", "age": 45}
    lesion = LesionData(synthetic_lesion_img, metadata=metadata)
    
    data_dict = lesion.to_dict()
    
    assert isinstance(data_dict, dict)
    assert "metadata" in data_dict
    assert "provenance" in data_dict
    assert "results" in data_dict
    assert data_dict["metadata"]["subject_id"] == "sub-001"


def test_lesion_data_from_dict(synthetic_lesion_img, lesion_metadata):
    """Test from_dict deserialization."""
    from ldk.core.lesion_data import LesionData
    
    # Create original
    metadata = {"subject_id": "sub-001", "space": "MNI152_2mm"}
    lesion = LesionData(synthetic_lesion_img, metadata=metadata)
    
    # Serialize and deserialize
    data_dict = lesion.to_dict()
    lesion_restored = LesionData.from_dict(data_dict, synthetic_lesion_img)
    
    assert lesion_restored.metadata["subject_id"] == "sub-001"


def test_lesion_data_properties_are_readonly(synthetic_lesion_img, lesion_metadata):
    """Test that properties cannot be directly modified."""
    from ldk.core.lesion_data import LesionData
    
    lesion = LesionData(synthetic_lesion_img, metadata=lesion_metadata)
    
    # These should raise AttributeError if trying to set
    with pytest.raises(AttributeError):
        lesion.metadata = {"new": "dict"}
    
    with pytest.raises(AttributeError):
        lesion.provenance = []


# Fixtures

@pytest.fixture
def synthetic_lesion_img():
    """Create a synthetic 3D lesion mask for testing."""
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    
    # Create small spherical lesion
    center = (32, 32, 32)
    radius = 5
    for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)):
        for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)):
            for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius + 1)):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
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
    return {"subject_id": "sub-001", "space": "MNI152_2mm"}


@pytest.fixture
def synthetic_4d_img():
    """Create a synthetic 4D image (should be rejected)."""
    shape = (64, 64, 64, 10)
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)
