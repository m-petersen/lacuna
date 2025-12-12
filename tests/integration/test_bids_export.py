"""
Integration tests for BIDS derivatives export functionality.

Tests the complete workflow of exporting SubjectData with results to BIDS-compliant
derivatives structure, including multi-subject and multi-session scenarios.
"""

import json

import numpy as np

from lacuna import SubjectData
from lacuna.core.provenance import create_provenance_record
from lacuna.io import export_bids_derivatives


def test_export_single_subject_workflow(tmp_path, synthetic_mask_img):
    """Test complete export workflow for single subject."""
    # Create lesion data
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Export to BIDS derivatives
    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(mask_data, output_dir)

    # Verify directory structure
    assert subject_dir.exists()
    assert subject_dir.name == "sub-001"
    assert (output_dir / "dataset_description.json").exists()
    assert (subject_dir / "anat").exists()

    # Verify dataset_description.json content
    with open(output_dir / "dataset_description.json") as f:
        ds_desc = json.load(f)
    assert ds_desc["Name"] == "Lacuna Derivatives"
    assert "GeneratedBy" in ds_desc


def test_export_with_analysis_results(tmp_path, synthetic_mask_img):
    """Test exporting lesion with analysis results."""
    # Create lesion with results
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-002", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    # Add analysis results - each result key gets its own file
    results = {
        "volume_mm3": 2500.0,
        "n_voxels": 312,
        "regions_affected": ["Frontal_Sup_L", "Frontal_Mid_L"],
    }
    lesion_with_results = mask_data.add_result("RegionalDamage", results)

    # Add provenance
    prov = create_provenance_record(
        function="compute_regional_damage",
        parameters={"atlas": "Schaefer2018_100Parcels7Networks"},
        version="0.1.0",
    )
    lesion_with_results = lesion_with_results.add_provenance(prov)

    # Export
    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_with_results, output_dir)

    # Verify anat directory exists (all outputs go to anat/ per BIDS spec)
    anat_dir = subject_dir / "anat"
    assert anat_dir.exists()

    # Verify results JSON files exist (one per result key)
    # Pattern: sub-002_desc-{key}_stats.json (namespace removed, BIDS suffix)
    results_files = list(anat_dir.glob("*_stats.json"))
    assert (
        len(results_files) >= 1
    ), f"Expected results files, found: {list(anat_dir.glob('*.json'))}"

    # Verify one of the results has correct content
    volume_files = list(anat_dir.glob("*_desc-volumemm3_stats.json"))
    if volume_files:
        with open(volume_files[0]) as f:
            saved_value = json.load(f)
        assert saved_value == 2500.0

    # Verify provenance JSON exists
    prov_files = list(anat_dir.glob("*_desc-provenance.json"))
    assert len(prov_files) == 1


def test_export_multi_session(tmp_path, synthetic_mask_img):
    """Test exporting multi-session data."""
    sessions = ["ses-pre", "ses-post"]
    subject_id = "sub-003"

    for session in sessions:
        mask_data = SubjectData(
            mask_img=synthetic_mask_img,
            metadata={
                "subject_id": subject_id,
                "session_id": session,
                "space": "MNI152NLin6Asym",
                "resolution": 2,
            },
        )

        output_dir = tmp_path / "derivatives" / "lacuna"
        subject_dir = export_bids_derivatives(mask_data, output_dir)

        # Verify session is in the path
        assert session in str(subject_dir)
        assert (subject_dir / "anat").exists()

    # Verify both sessions exist
    subject_base = tmp_path / "derivatives" / "lacuna" / subject_id
    assert (subject_base / "ses-pre").exists()
    assert (subject_base / "ses-post").exists()


def test_export_multiple_subjects(tmp_path, synthetic_mask_img):
    """Test exporting multiple subjects to same derivatives directory."""
    subjects = ["sub-101", "sub-102", "sub-103", "sub-104", "sub-105"]
    output_dir = tmp_path / "derivatives" / "lacuna"

    for subject_id in subjects:
        mask_data = SubjectData(
            mask_img=synthetic_mask_img,
            metadata={"subject_id": subject_id, "space": "MNI152NLin6Asym", "resolution": 2},
        )
        export_bids_derivatives(mask_data, output_dir)

    # Verify all subjects exist
    for subject_id in subjects:
        subject_dir = output_dir / subject_id
        assert subject_dir.exists()
        assert (subject_dir / "anat").exists()

    # Verify shared dataset_description
    assert (output_dir / "dataset_description.json").exists()


def test_export_and_reload_workflow(tmp_path, synthetic_mask_img):
    """Test complete save-export-load workflow."""
    # Create original lesion
    original = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={
            "subject_id": "sub-201",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
            "study": "test_study",
        },
    )

    # Export to BIDS
    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(original, output_dir)

    # Find exported mask file (now uses _mask.nii.gz pattern)
    mask_files = list((subject_dir / "anat").glob("*_mask.nii.gz"))
    assert len(mask_files) == 1
    mask_file = mask_files[0]

    # Reload mask data
    import nibabel as nib

    reloaded_img = nib.load(mask_file)

    # Verify data matches
    np.testing.assert_array_equal(reloaded_img.get_fdata(), original.mask_img.get_fdata())
    np.testing.assert_array_equal(reloaded_img.affine, original.affine)


def test_export_lesion_mask_only(tmp_path, synthetic_mask_img):
    """Test exporting with only lesion mask enabled."""
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-301", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(
        mask_data,
        output_dir,
        export_lesion_mask=True,
        export_voxelmaps=False,
        export_parcel_data=False,
        export_connectivity=False,
        export_scalars=False,
    )

    anat_dir = subject_dir / "anat"
    assert anat_dir.exists()

    # Should have mask file
    files = list(anat_dir.glob("*_mask.nii.gz"))
    assert len(files) >= 1


def test_export_preserves_bids_naming(tmp_path, synthetic_mask_img):
    """Test that BIDS naming conventions are followed."""
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={
            "subject_id": "sub-401",
            "session_id": "ses-01",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
        },
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(mask_data, output_dir)

    # Find mask file and check naming
    mask_files = list((subject_dir / "anat").glob("*_mask.nii.gz"))
    assert len(mask_files) == 1

    filename = mask_files[0].name
    # Should contain subject and session
    assert "sub-401" in filename
    assert "ses-01" in filename
    # Should contain space entity
    assert "space-MNI152" in filename or "MNI152" in filename
    # Should end with _mask.nii.gz
    assert filename.endswith("_mask.nii.gz")


def test_export_selective_outputs(tmp_path, synthetic_mask_img):
    """Test selective output export options."""
    # Create lesion with results
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-501", "space": "MNI152NLin6Asym", "resolution": 2},
    )
    lesion_with_results = mask_data.add_result("TestAnalysis", {"metric": 42.0})

    output_dir = tmp_path / "derivatives" / "lacuna"

    # Export only lesion mask, no scalars/parcel data
    subject_dir = export_bids_derivatives(
        lesion_with_results,
        output_dir,
        export_lesion_mask=True,
        export_voxelmaps=False,
        export_parcel_data=False,
        export_connectivity=False,
        export_scalars=False,
        export_provenance=False,
    )

    # Should have anat dir with mask file
    assert (subject_dir / "anat").exists()
    mask_files = list((subject_dir / "anat").glob("*_mask.nii.gz"))
    assert len(mask_files) >= 1


def test_export_creates_complete_derivatives_structure(tmp_path, synthetic_mask_img):
    """Test that complete BIDS derivatives structure is created."""
    mask_data = SubjectData(
        mask_img=synthetic_mask_img,
        metadata={"subject_id": "sub-601", "space": "MNI152NLin6Asym", "resolution": 2},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(mask_data, output_dir)

    # Check complete structure
    expected_paths = [
        output_dir,
        output_dir / "dataset_description.json",
        subject_dir,
        subject_dir / "anat",
    ]

    for path in expected_paths:
        assert path.exists(), f"Expected path does not exist: {path}"
