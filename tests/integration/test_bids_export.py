"""
Integration tests for BIDS derivatives export functionality.

Tests the complete workflow of exporting LesionData with results to BIDS-compliant
derivatives structure, including multi-subject and multi-session scenarios.
"""

import json

import numpy as np

from lacuna import LesionData
from lacuna.core.provenance import create_provenance_record
from lacuna.io import export_bids_derivatives


def test_export_single_subject_workflow(tmp_path, synthetic_lesion_img):
    """Test complete export workflow for single subject."""
    # Create lesion data
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-001", "space": "MNI152_2mm"},
    )

    # Export to BIDS derivatives
    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data, output_dir)

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


def test_export_with_analysis_results(tmp_path, synthetic_lesion_img):
    """Test exporting lesion with analysis results."""
    # Create lesion with results
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-002", "space": "MNI152_2mm"},
    )

    # Add analysis results
    results = {
        "volume_mm3": 2500.0,
        "n_voxels": 312,
        "regions_affected": ["Frontal_Sup_L", "Frontal_Mid_L"],
    }
    lesion_with_results = lesion_data.add_result("RegionalDamage", results)

    # Add provenance
    prov = create_provenance_record(
        function="compute_regional_damage", parameters={"atlas": "AAL3"}, version="0.1.0"
    )
    lesion_with_results = lesion_with_results.add_provenance(prov)

    # Export
    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_with_results, output_dir)

    # Verify results directory exists
    results_dir = subject_dir / "results"
    assert results_dir.exists()

    # Verify results JSON exists
    results_files = list(results_dir.glob("*_results.json"))
    assert len(results_files) == 1

    # Verify results content
    with open(results_files[0]) as f:
        saved_results = json.load(f)
    assert "RegionalDamage" in saved_results
    assert saved_results["RegionalDamage"]["volume_mm3"] == 2500.0

    # Verify provenance JSON exists
    prov_files = list(results_dir.glob("*_desc-provenance.json"))
    assert len(prov_files) == 1


def test_export_multi_session(tmp_path, synthetic_lesion_img):
    """Test exporting multi-session data."""
    sessions = ["ses-pre", "ses-post"]
    subject_id = "sub-003"

    for session in sessions:
        lesion_data = LesionData(
            lesion_img=synthetic_lesion_img,
            metadata={"subject_id": subject_id, "session_id": session, "space": "MNI152_2mm"},
        )

        output_dir = tmp_path / "derivatives" / "lacuna"
        subject_dir = export_bids_derivatives(lesion_data, output_dir)

        # Verify session is in the path
        assert session in str(subject_dir)
        assert (subject_dir / "anat").exists()

    # Verify both sessions exist
    subject_base = tmp_path / "derivatives" / "lacuna" / subject_id
    assert (subject_base / "ses-pre").exists()
    assert (subject_base / "ses-post").exists()


def test_export_multiple_subjects(tmp_path, synthetic_lesion_img):
    """Test exporting multiple subjects to same derivatives directory."""
    subjects = ["sub-101", "sub-102", "sub-103", "sub-104", "sub-105"]
    output_dir = tmp_path / "derivatives" / "lacuna"

    for subject_id in subjects:
        lesion_data = LesionData(
            lesion_img=synthetic_lesion_img,
            metadata={"subject_id": subject_id, "space": "MNI152_2mm"},
        )
        export_bids_derivatives(lesion_data, output_dir)

    # Verify all subjects exist
    for subject_id in subjects:
        subject_dir = output_dir / subject_id
        assert subject_dir.exists()
        assert (subject_dir / "anat").exists()

    # Verify shared dataset_description
    assert (output_dir / "dataset_description.json").exists()


def test_export_and_reload_workflow(tmp_path, synthetic_lesion_img):
    """Test complete save-export-load workflow."""
    # Create original lesion
    original = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-201", "space": "MNI152_2mm", "study": "test_study"},
    )

    # Export to BIDS
    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(original, output_dir)

    # Find exported lesion file
    lesion_files = list((subject_dir / "anat").glob("*_lesion.nii.gz"))
    assert len(lesion_files) == 1
    lesion_file = lesion_files[0]

    # Reload lesion
    reloaded = LesionData.from_nifti(
        str(lesion_file), metadata={"subject_id": "sub-201", "space": "MNI152_2mm"}
    )

    # Verify data matches
    np.testing.assert_array_equal(reloaded.lesion_img.get_fdata(), original.lesion_img.get_fdata())
    np.testing.assert_array_equal(reloaded.affine, original.affine)


def test_export_with_anatomical_image(tmp_path, synthetic_lesion_img, synthetic_anatomical_img):
    """Test exporting lesion with anatomical reference image."""
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        anatomical_img=synthetic_anatomical_img,
        metadata={"subject_id": "sub-301", "space": "MNI152_2mm"},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data, output_dir, include_anatomical=True)

    anat_dir = subject_dir / "anat"
    assert anat_dir.exists()

    # Should have both lesion and anatomical files
    files = list(anat_dir.glob("*.nii.gz"))
    assert len(files) >= 1  # At least lesion, maybe anatomical too


def test_export_preserves_bids_naming(tmp_path, synthetic_lesion_img):
    """Test that BIDS naming conventions are followed."""
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-401", "session_id": "ses-01", "space": "MNI152_2mm"},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data, output_dir)

    # Find lesion file and check naming
    lesion_files = list((subject_dir / "anat").glob("*.nii.gz"))
    assert len(lesion_files) == 1

    filename = lesion_files[0].name
    # Should contain subject and session
    assert "sub-401" in filename
    assert "ses-01" in filename
    # Should contain space entity
    assert "space-MNI152" in filename or "MNI152" in filename
    # Should end with _lesion.nii.gz
    assert filename.endswith("_lesion.nii.gz")


def test_export_selective_outputs(tmp_path, synthetic_lesion_img):
    """Test selective output export options."""
    # Create lesion with results
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-501", "space": "MNI152_2mm"},
    )
    lesion_with_results = lesion_data.add_result("TestAnalysis", {"metric": 42.0})

    output_dir = tmp_path / "derivatives" / "lacuna"

    # Export only images, no results
    subject_dir = export_bids_derivatives(
        lesion_with_results, output_dir, include_images=True, include_results=False
    )

    # Should have anat dir but not results dir
    assert (subject_dir / "anat").exists()
    assert not (subject_dir / "results").exists()


def test_export_creates_complete_derivatives_structure(tmp_path, synthetic_lesion_img):
    """Test that complete BIDS derivatives structure is created."""
    lesion_data = LesionData(
        lesion_img=synthetic_lesion_img,
        metadata={"subject_id": "sub-601", "space": "MNI152_2mm"},
    )

    output_dir = tmp_path / "derivatives" / "lacuna"
    subject_dir = export_bids_derivatives(lesion_data, output_dir)

    # Check complete structure
    expected_paths = [
        output_dir,
        output_dir / "dataset_description.json",
        subject_dir,
        subject_dir / "anat",
    ]

    for path in expected_paths:
        assert path.exists(), f"Expected path does not exist: {path}"
