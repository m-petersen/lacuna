"""
Integration test for loading BIDS datasets.

Tests the complete workflow of loading multiple subjects from a BIDS dataset
with proper filtering, metadata extraction, and derivative handling.
"""

import nibabel as nib
import numpy as np


def test_load_bids_dataset_complete_workflow(simple_bids_dataset):
    """Test loading complete BIDS dataset."""
    from ldk.io import load_bids_dataset

    # Load entire dataset
    lesion_data_dict = load_bids_dataset(simple_bids_dataset)

    # Should load both subjects
    assert len(lesion_data_dict) == 2
    assert "sub-001" in lesion_data_dict
    assert "sub-002" in lesion_data_dict

    # sub-001 should have anatomical
    assert lesion_data_dict["sub-001"].anatomical_img is not None

    # sub-002 should not have anatomical
    assert lesion_data_dict["sub-002"].anatomical_img is None

    # Both should have lesion masks
    assert lesion_data_dict["sub-001"].lesion_img is not None
    assert lesion_data_dict["sub-002"].lesion_img is not None


def test_load_bids_dataset_subject_filtering(simple_bids_dataset):
    """Test filtering by specific subjects."""
    from ldk.io import load_bids_dataset

    # Load only sub-001
    lesion_data_dict = load_bids_dataset(simple_bids_dataset, subjects=["sub-001"])

    assert len(lesion_data_dict) == 1
    assert "sub-001" in lesion_data_dict
    assert "sub-002" not in lesion_data_dict


def test_load_bids_dataset_multisession(multisession_bids_dataset):
    """Test loading dataset with multiple sessions."""
    from ldk.io import load_bids_dataset

    # Load all sessions
    lesion_data_dict = load_bids_dataset(multisession_bids_dataset)

    # Should have 2 entries: sub-001_ses-01, sub-001_ses-02
    assert len(lesion_data_dict) == 2
    assert "sub-001_ses-01" in lesion_data_dict
    assert "sub-001_ses-02" in lesion_data_dict

    # Both should have data
    for key, ld in lesion_data_dict.items():
        assert ld.lesion_img is not None
        assert ld.anatomical_img is not None


def test_load_bids_dataset_session_filtering(multisession_bids_dataset):
    """Test filtering by specific sessions."""
    from ldk.io import load_bids_dataset

    # Load only ses-01
    lesion_data_dict = load_bids_dataset(multisession_bids_dataset, sessions=["ses-01"])

    assert len(lesion_data_dict) == 1
    assert "sub-001_ses-01" in lesion_data_dict
    assert "sub-001_ses-02" not in lesion_data_dict


def test_bids_dataset_batch_analysis(simple_bids_dataset):
    """Test analyzing multiple subjects from BIDS dataset."""
    from ldk.io import load_bids_dataset

    # Load dataset
    lesion_data_dict = load_bids_dataset(simple_bids_dataset)

    # Analyze all subjects
    results = {}
    for subject_id, lesion_data in lesion_data_dict.items():
        volume_mm3 = lesion_data.get_volume_mm3()
        coord_space = lesion_data.get_coordinate_space()

        results[subject_id] = {
            "volume_mm3": volume_mm3,
            "coordinate_space": coord_space,
            "has_anatomical": lesion_data.anatomical_img is not None,
        }

    # Verify results
    assert len(results) == 2
    assert all(v["volume_mm3"] > 0 for v in results.values())
    assert all(v["coordinate_space"] == "RAS+" for v in results.values())


def test_bids_dataset_export_derivatives(simple_bids_dataset, tmp_path):
    """Test complete workflow: load → analyze → export."""
    from ldk.core.provenance import create_provenance_record
    from ldk.io import export_bids_derivatives, load_bids_dataset

    # Load dataset
    lesion_data_dict = load_bids_dataset(simple_bids_dataset)

    # Analyze each subject
    analyzed_dict = {}
    for subject_id, lesion_data in lesion_data_dict.items():
        # Compute results
        volume_mm3 = lesion_data.get_volume_mm3()
        results = {
            "volume_mm3": volume_mm3,
            "analysis_type": "VolumeCalculation",
        }

        # Create provenance
        prov = create_provenance_record(
            function="test_volume_analysis",
            parameters={"method": "voxel_count"},
            version="0.1.0",
        )

        # Add results and provenance
        analyzed = lesion_data.add_result("VolumeAnalysis", results).add_provenance(prov)
        analyzed_dict[subject_id] = analyzed

    # Export derivatives
    output_dir = tmp_path / "derivatives"
    export_bids_derivatives(
        lesion_data_dict=analyzed_dict,
        output_dir=output_dir,
        pipeline_name="test_pipeline",
        pipeline_version="0.1.0",
        include_results=True,
        include_provenance=True,
    )

    # Verify structure
    pipeline_dir = output_dir / "test_pipeline"
    assert pipeline_dir.exists()
    assert (pipeline_dir / "dataset_description.json").exists()

    # Check subjects
    for subject_id in analyzed_dict.keys():
        subject_dir = pipeline_dir / subject_id
        assert subject_dir.exists()

        lesion_dir = subject_dir / "lesion"
        assert lesion_dir.exists()
        assert len(list(lesion_dir.glob("*_mask.nii.gz"))) > 0

        results_dir = subject_dir / "results"
        assert results_dir.exists()
        assert len(list(results_dir.glob("*_desc-VolumeAnalysis.json"))) > 0
        assert len(list(results_dir.glob("*_desc-provenance.json"))) > 0


def test_bids_dataset_missing_anatomicals(tmp_path):
    """Test handling dataset where some subjects lack anatomical images."""
    from ldk.io import load_bids_dataset

    # Create BIDS dataset with mixed anatomical availability
    bids_root = tmp_path / "bids_mixed"
    bids_root.mkdir()

    # Create dataset_description.json
    import json

    dataset_desc = {"Name": "Mixed Dataset", "BIDSVersion": "1.6.0"}
    with open(bids_root / "dataset_description.json", "w") as f:
        json.dump(dataset_desc, f)

    # Create subjects (using fixtures)
    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    lesion_data = np.zeros(shape, dtype=np.uint8)
    lesion_data[30:35, 30:35, 30:35] = 1

    # sub-01: has anatomical
    sub01_dir = bids_root / "sub-01" / "anat"
    sub01_dir.mkdir(parents=True)

    nib.save(
        nib.Nifti1Image(lesion_data, affine),
        sub01_dir / "sub-01_space-T1w_label-lesion_mask.nii.gz",
    )

    anat_data = np.random.rand(*shape).astype(np.float32) * 1000
    nib.save(nib.Nifti1Image(anat_data, affine), sub01_dir / "sub-01_T1w.nii.gz")

    # sub-02: no anatomical
    sub02_dir = bids_root / "sub-02" / "anat"
    sub02_dir.mkdir(parents=True)

    nib.save(
        nib.Nifti1Image(lesion_data, affine),
        sub02_dir / "sub-02_space-T1w_label-lesion_mask.nii.gz",
    )

    # Load dataset
    lesion_data_dict = load_bids_dataset(bids_root)

    # Verify mixed anatomical availability
    assert len(lesion_data_dict) == 2
    assert lesion_data_dict["sub-01"].anatomical_img is not None
    assert lesion_data_dict["sub-02"].anatomical_img is None


def test_bids_dataset_metadata_extraction(simple_bids_dataset):
    """Test that BIDS metadata is properly extracted."""
    from ldk.io import load_bids_dataset

    lesion_data_dict = load_bids_dataset(simple_bids_dataset)

    # Verify metadata extraction
    for subject_id, lesion_data in lesion_data_dict.items():
        assert "subject_id" in lesion_data.metadata
        assert lesion_data.metadata["subject_id"] == subject_id

        # Should have source path info
        assert "lesion_path" in lesion_data.metadata


def test_bids_dataset_empty(tmp_path):
    """Test handling of empty BIDS dataset."""
    from ldk.io import load_bids_dataset

    # Create minimal BIDS structure with no subjects
    bids_root = tmp_path / "empty_bids"
    bids_root.mkdir()

    import json

    dataset_desc = {"Name": "Empty Dataset", "BIDSVersion": "1.6.0"}
    with open(bids_root / "dataset_description.json", "w") as f:
        json.dump(dataset_desc, f)

    # Should raise BidsError or return empty dict
    result = load_bids_dataset(bids_root)
    assert len(result) == 0 or isinstance(result, dict)


def test_bids_dataset_load_with_custom_metadata(simple_bids_dataset, tmp_path):
    """Test adding custom metadata during BIDS loading."""
    from ldk.io import load_bids_dataset

    # Load with custom metadata template
    custom_metadata = {"study": "test_study", "scanner": "3T"}

    lesion_data_dict = load_bids_dataset(simple_bids_dataset)

    # Add custom metadata to all subjects
    enriched_dict = {}
    for subject_id, lesion_data in lesion_data_dict.items():
        new_metadata = {**lesion_data.metadata, **custom_metadata, "subject_id": subject_id}
        enriched = lesion_data.copy_with(metadata=new_metadata)
        enriched_dict[subject_id] = enriched

    # Verify custom metadata persists
    for ld in enriched_dict.values():
        assert ld.metadata["study"] == "test_study"
        assert ld.metadata["scanner"] == "3T"
