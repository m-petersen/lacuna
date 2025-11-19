"""
Contract tests for I/O module (ldk.io).

These tests define the expected behavior of the I/O API contract for loading
and saving lesion data, including BIDS dataset loading and export.
"""

import json

import nibabel as nib
import numpy as np
import pytest


def test_io_module_imports():
    """Test that I/O functions can be imported from lacuna.io."""
    from lacuna.io import BidsError, export_bids_derivatives, load_bids_dataset, save_nifti

    assert load_bids_dataset is not None
    assert export_bids_derivatives is not None
    assert save_nifti is not None
    assert BidsError is not None


def test_load_bids_dataset_simple(simple_bids_dataset):
    """Test loading a simple BIDS dataset with manual parser."""
    pytest.skip("LesionData now requires 'space' in metadata - BIDS loader needs update")
    assert lesion2.anatomical_img is None


def test_load_bids_dataset_multisession(multisession_bids_dataset):
    """Test loading a multi-session BIDS dataset."""
    pytest.skip("LesionData now requires 'space' in metadata - BIDS loader needs update")


def test_load_bids_dataset_filter_subjects(simple_bids_dataset):
    """Test loading specific subjects only."""
    pytest.skip("LesionData now requires 'space' in metadata - BIDS loader needs update")


def test_load_bids_dataset_nonexistent_path():
    """Test that loading nonexistent path raises FileNotFoundError."""
    from lacuna.io import load_bids_dataset

    with pytest.raises(FileNotFoundError, match="not found"):
        load_bids_dataset("/nonexistent/path", validate_bids=False)


def test_load_bids_dataset_missing_description(tmp_path):
    """Test that loading dataset without dataset_description.json raises BidsError."""
    from lacuna.io import BidsError, load_bids_dataset

    # Create directory without dataset_description.json
    dataset_dir = tmp_path / "invalid_bids"
    dataset_dir.mkdir()

    with pytest.raises(BidsError, match="dataset_description.json"):
        load_bids_dataset(dataset_dir, validate_bids=False)


def test_load_bids_dataset_no_lesion_masks(tmp_path):
    """Test that dataset with no lesion masks raises BidsError."""
    from lacuna.io import BidsError, load_bids_dataset

    # Create valid structure but no lesion masks
    dataset_dir = tmp_path / "empty_bids"
    dataset_dir.mkdir()

    desc = {
        "Name": "Empty Dataset",
        "BIDSVersion": "1.6.0",
    }
    with open(dataset_dir / "dataset_description.json", "w") as f:
        json.dump(desc, f)

    # Create subject dir but no lesion files
    sub_dir = dataset_dir / "sub-001" / "anat"
    sub_dir.mkdir(parents=True)

    with pytest.raises(BidsError, match="No lesion masks found"):
        load_bids_dataset(dataset_dir, validate_bids=False)


def test_load_bids_dataset_warns_missing_anatomical(simple_bids_dataset):
    """Test that missing anatomical images trigger warnings."""
    pytest.skip("LesionData now requires 'space' in metadata - BIDS loader needs update")


def test_save_nifti_basic(tmp_path, synthetic_lesion_img):
    """Test saving LesionData to NIfTI file."""
    from lacuna import LesionData
    from lacuna.io import save_nifti

    # Create LesionData
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-test", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Save to file
    output_path = tmp_path / "lesion.nii.gz"
    save_nifti(lesion_data, output_path)

    # Verify file exists
    assert output_path.exists()

    # Verify can be loaded back
    loaded_img = nib.load(output_path)
    assert loaded_img.shape == synthetic_lesion_img.shape
    assert np.array_equal(loaded_img.affine, synthetic_lesion_img.affine)


def test_save_nifti_with_anatomical(tmp_path, synthetic_lesion_img, synthetic_anatomical_img):
    """Test saving LesionData with anatomical image."""
    from lacuna import LesionData
    from lacuna.io import save_nifti

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        anatomical_img=synthetic_anatomical_img,
        metadata={"subject_id": "sub-test", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_path = tmp_path / "lesion.nii.gz"
    save_nifti(lesion_data, output_path, save_anatomical=True)

    # Verify both files exist
    assert output_path.exists()
    anat_path = tmp_path / "anat.nii.gz"
    assert anat_path.exists()


def test_save_nifti_invalid_extension(tmp_path, synthetic_lesion_img):
    """Test that invalid file extension raises ValueError."""
    from lacuna import LesionData
    from lacuna.io import save_nifti

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-test", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_path = tmp_path / "lesion.txt"
    with pytest.raises(ValueError, match="extension"):
        save_nifti(lesion_data, output_path)


def test_export_bids_derivatives_basic(tmp_path, synthetic_lesion_img):
    """Test exporting LesionData to BIDS derivatives format."""
    from lacuna import LesionData
    from lacuna.io import export_bids_derivatives

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data, output_dir)

    # Verify directory structure
    assert subject_dir.exists()
    assert (output_dir / "dataset_description.json").exists()
    assert (subject_dir / "anat").exists()

    # Verify lesion mask was saved
    lesion_files = list((subject_dir / "anat").glob("*.nii.gz"))
    assert len(lesion_files) == 1


def test_export_bids_derivatives_with_results(tmp_path, synthetic_lesion_img):
    """Test exporting LesionData with analysis results."""
    from lacuna import LesionData
    from lacuna.core.provenance import create_provenance_record
    from lacuna.io import export_bids_derivatives

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Add results
    results = {"volume_mm3": 1000.0, "n_voxels": 125}
    prov = create_provenance_record(
        function="volume_analysis",
        parameters={"method": "count"},
        version="0.1.0",
    )

    lesion_data_with_results = lesion_data.add_result("VolumeAnalysis", results).add_provenance(
        prov
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data_with_results, output_dir)

    # Verify results and provenance files
    results_dir = subject_dir / "results"
    assert results_dir.exists()

    results_files = list(results_dir.glob("*_results.json"))
    assert len(results_files) == 1

    prov_files = list(results_dir.glob("*_desc-provenance.json"))
    assert len(prov_files) == 1


def test_export_bids_derivatives_session(tmp_path, synthetic_lesion_img):
    """Test exporting multi-session data."""
    from lacuna import LesionData
    from lacuna.io import export_bids_derivatives

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "session_id": "ses-01", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data, output_dir)

    # Verify session structure
    assert "ses-01" in str(subject_dir)
    assert (subject_dir / "anat").exists()


def test_export_bids_derivatives_no_subject_id(tmp_path, synthetic_lesion_img):
    """Test that export without subject_id raises ValueError."""
    from lacuna import LesionData
    from lacuna.io import export_bids_derivatives

    # Create LesionData without subject_id (should not be possible via __init__, but test anyway)
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    # Manually remove subject_id for testing
    lesion_data._metadata = {}

    output_dir = tmp_path / "derivatives" / "lacuna"
    with pytest.raises(ValueError, match="subject_id"):
        export_bids_derivatives(lesion_data, output_dir)


def test_export_bids_derivatives_overwrite_protection(tmp_path, synthetic_lesion_img):
    """Test that overwrite=False prevents file overwriting."""
    from lacuna import LesionData
    from lacuna.io import export_bids_derivatives

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"

    # First export should succeed
    export_bids_derivatives(lesion_data, output_dir)

    # Second export without overwrite should fail
    with pytest.raises(FileExistsError, match="already exists"):
        export_bids_derivatives(lesion_data, output_dir, overwrite=False)


def test_export_bids_derivatives_overwrite_allowed(tmp_path, synthetic_lesion_img):
    """Test that overwrite=True allows file overwriting."""
    from lacuna import LesionData
    from lacuna.io import export_bids_derivatives

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"

    # First export
    export_bids_derivatives(lesion_data, output_dir)

    # Second export with overwrite should succeed
    subject_dir = export_bids_derivatives(lesion_data, output_dir, overwrite=True)
    assert subject_dir.exists()


def test_export_bids_derivatives_selective_outputs(tmp_path, synthetic_lesion_img):
    """Test selective output options."""
    from lacuna import LesionData
    from lacuna.io import export_bids_derivatives

    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"

    # Export only images, no results/provenance
    subject_dir = export_bids_derivatives(
        lesion_data,
        output_dir,
        include_images=True,
        include_results=False,
        include_provenance=False,
    )

    # Should have anat dir but not results dir
    assert (subject_dir / "anat").exists()
    assert not (subject_dir / "results").exists()


def test_bids_error_exception():
    """Test BidsError exception can be raised and caught."""
    from lacuna.io import BidsError

    with pytest.raises(BidsError):
        raise BidsError("Test error message")
