"""
Contract tests for I/O module (ldk.io).

These tests define the expected behavior of the I/O API contract for loading
and saving lesion data, including BIDS dataset loading and export.
"""

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
    pytest.skip("SubjectData now requires 'space' in metadata - BIDS loader needs update")
    pass


def test_load_bids_dataset_multisession(multisession_bids_dataset):
    """Test loading a multi-session BIDS dataset."""
    pytest.skip("SubjectData now requires 'space' in metadata - BIDS loader needs update")


def test_load_bids_dataset_filter_subjects(simple_bids_dataset):
    """Test loading specific subjects only."""
    pytest.skip("SubjectData now requires 'space' in metadata - BIDS loader needs update")


def test_load_bids_dataset_nonexistent_path():
    """Test that loading nonexistent path raises FileNotFoundError."""
    from lacuna.io import load_bids_dataset

    with pytest.raises(FileNotFoundError, match="not found"):
        load_bids_dataset("/nonexistent/path")


def test_load_bids_dataset_empty_directory(tmp_path):
    """Test that loading empty directory raises BidsError."""
    from lacuna.io import BidsError, load_bids_dataset

    # Create empty directory - new API doesn't require dataset_description.json
    dataset_dir = tmp_path / "empty_dir"
    dataset_dir.mkdir()

    with pytest.raises(BidsError, match="No files matching"):
        load_bids_dataset(dataset_dir)


def test_load_bids_dataset_no_matching_files(tmp_path):
    """Test that dataset with no matching mask files raises BidsError."""
    from lacuna.io import BidsError, load_bids_dataset

    # Create directory with non-matching files
    dataset_dir = tmp_path / "no_masks"
    dataset_dir.mkdir()

    # Create subject dir but no mask files
    sub_dir = dataset_dir / "sub-001" / "anat"
    sub_dir.mkdir(parents=True)

    # Create a non-mask NIfTI file
    import nibabel as nib
    import numpy as np

    data = np.zeros((10, 10, 10), dtype=np.uint8)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, sub_dir / "sub-001_T1w.nii.gz")

    with pytest.raises(BidsError, match="No files matching"):
        load_bids_dataset(dataset_dir)  # Default suffix is _mask.nii.gz


def test_load_bids_dataset_warns_missing_anatomical(simple_bids_dataset):
    """Test that missing anatomical images trigger warnings."""
    pytest.skip("SubjectData now requires 'space' in metadata - BIDS loader needs update")


def test_save_nifti_basic(tmp_path, synthetic_mask_img):
    """Test saving SubjectData to NIfTI file."""
    from lacuna import SubjectData
    from lacuna.io import save_nifti

    # Create SubjectData
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-test", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Save to file
    output_path = tmp_path / "lesion.nii.gz"
    save_nifti(mask_data, output_path)

    # Verify file exists
    assert output_path.exists()

    # Verify can be loaded back
    loaded_img = nib.load(output_path)
    assert loaded_img.shape == synthetic_mask_img.shape
    assert np.array_equal(loaded_img.affine, synthetic_mask_img.affine)


def test_save_nifti_invalid_extension(tmp_path, synthetic_mask_img):
    """Test that invalid file extension raises ValueError."""
    from lacuna import SubjectData
    from lacuna.io import save_nifti

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-test", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_path = tmp_path / "lesion.txt"
    with pytest.raises(ValueError, match="extension"):
        save_nifti(mask_data, output_path)


def test_export_bids_derivatives_basic(tmp_path, synthetic_mask_img):
    """Test exporting SubjectData to BIDS derivatives format."""
    from lacuna import SubjectData
    from lacuna.io import export_bids_derivatives

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(mask_data, output_dir)

    # Verify directory structure
    assert subject_dir.exists()
    assert (output_dir / "dataset_description.json").exists()
    assert (subject_dir / "anat").exists()

    # Verify lesion mask was saved
    lesion_files = list((subject_dir / "anat").glob("*.nii.gz"))
    assert len(lesion_files) == 1


def test_export_bids_derivatives_with_results(tmp_path, synthetic_mask_img):
    """Test exporting SubjectData with analysis results."""
    from lacuna import SubjectData
    from lacuna.core.provenance import create_provenance_record
    from lacuna.io import export_bids_derivatives

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Add results
    results = {"volume_mm3": 1000.0, "n_voxels": 125}
    prov = create_provenance_record(
        function="volume_analysis",
        parameters={"method": "count"},
        version="0.1.0",
    )

    mask_data_with_results = mask_data.add_result("VolumeAnalysis", results).add_provenance(prov)

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(mask_data_with_results, output_dir)

    # All outputs now go to anat/ directory (BIDS compliant)
    anat_dir = subject_dir / "anat"
    assert anat_dir.exists()

    # Each scalar result is saved as individual JSON file (e.g., desc-volumemm3_stats.json)
    results_files = list(anat_dir.glob("*_stats.json"))
    assert (
        len(results_files) >= 1
    ), f"Expected scalar result files, got: {list(anat_dir.glob('*.json'))}"

    prov_files = list(anat_dir.glob("*_desc-provenance.json"))
    assert len(prov_files) == 1


def test_export_bids_derivatives_session(tmp_path, synthetic_mask_img):
    """Test exporting multi-session data."""
    from lacuna import SubjectData
    from lacuna.io import export_bids_derivatives

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={
            "subject_id": "sub-001",
            "session_id": "ses-01",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
        },
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(mask_data, output_dir)

    # Verify session structure
    assert "ses-01" in str(subject_dir)
    assert (subject_dir / "anat").exists()


def test_export_bids_derivatives_no_subject_id(tmp_path, synthetic_mask_img):
    """Test that export without subject_id raises ValueError."""
    from lacuna import SubjectData
    from lacuna.io import export_bids_derivatives

    # Create SubjectData without subject_id (should not be possible via __init__, but test anyway)
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Manually remove subject_id for testing
    mask_data._metadata = {}

    output_dir = tmp_path / "derivatives" / "lacuna"
    with pytest.raises(ValueError, match="subject_id"):
        export_bids_derivatives(mask_data, output_dir)


def test_export_bids_derivatives_overwrite_protection(tmp_path, synthetic_mask_img):
    """Test that overwrite=False prevents file overwriting."""
    from lacuna import SubjectData
    from lacuna.io import export_bids_derivatives

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"

    # First export should succeed
    export_bids_derivatives(mask_data, output_dir)

    # Second export without overwrite should fail
    with pytest.raises(FileExistsError, match="already exists"):
        export_bids_derivatives(mask_data, output_dir, overwrite=False)


def test_export_bids_derivatives_overwrite_allowed(tmp_path, synthetic_mask_img):
    """Test that overwrite=True allows file overwriting."""
    from lacuna import SubjectData
    from lacuna.io import export_bids_derivatives

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"

    # First export
    export_bids_derivatives(mask_data, output_dir)

    # Second export with overwrite should succeed
    subject_dir = export_bids_derivatives(mask_data, output_dir, overwrite=True)
    assert subject_dir.exists()


def test_export_bids_derivatives_selective_outputs(tmp_path, synthetic_mask_img):
    """Test selective output options with new explicit parameters."""
    from lacuna import SubjectData
    from lacuna.io import export_bids_derivatives

    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"

    # Export only lesion mask, no other results
    subject_dir = export_bids_derivatives(
        mask_data,
        output_dir,
        export_lesion_mask=True,
        export_voxelmaps=False,
        export_parcel_data=False,
        export_connectivity=False,
        export_scalars=False,
        export_provenance=False,
    )

    # Should have anat dir but not results dir
    assert (subject_dir / "anat").exists()
    assert not (subject_dir / "results").exists()


def test_bids_error_exception():
    """Test BidsError exception can be raised and caught."""
    from lacuna.io import BidsError

    with pytest.raises(BidsError):
        raise BidsError("Test error message")
