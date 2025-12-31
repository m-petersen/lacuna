"""
Unit tests for BIDS derivatives writer.

Tests the export_bids_derivatives and export_bids_derivatives_batch functions
that write analysis results in BIDS-compliant format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


class TestExportBidsDerivatives:
    """Tests for export_bids_derivatives function."""

    @pytest.fixture
    def mock_subject_data(self, tmp_path):
        """Create a mock SubjectData object for testing."""
        import nibabel as nib

        from lacuna.core.subject_data import SubjectData

        # Create a simple mask image
        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        mask_img = nib.Nifti1Image(mask_array, affine)

        # Create SubjectData with required metadata
        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

    @pytest.fixture
    def mock_subject_data_with_session(self, tmp_path):
        """Create a mock SubjectData with session info."""
        import nibabel as nib

        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        mask_img = nib.Nifti1Image(mask_array, affine)

        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001", "session_id": "ses-01"},
        )

    def test_creates_dataset_description(self, mock_subject_data, tmp_path):
        """Test that export creates dataset_description.json."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(mock_subject_data, output_dir)

        desc_file = output_dir / "dataset_description.json"
        assert desc_file.exists()

        with open(desc_file) as f:
            desc = json.load(f)

        assert "Name" in desc
        assert "BIDSVersion" in desc
        assert "GeneratedBy" in desc
        assert desc["BIDSVersion"] == "1.6.0"

    def test_creates_subject_directory(self, mock_subject_data, tmp_path):
        """Test that export creates subject directory structure."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        result_path = export_bids_derivatives(mock_subject_data, output_dir)

        assert result_path.exists()
        assert (result_path / "anat").exists()

    def test_creates_session_directory(self, mock_subject_data_with_session, tmp_path):
        """Test that export creates session subdirectory when session_id is present."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        result_path = export_bids_derivatives(mock_subject_data_with_session, output_dir)

        assert result_path.exists()
        assert "ses-01" in str(result_path)

    def test_exports_lesion_mask_nifti(self, mock_subject_data, tmp_path):
        """Test that lesion mask is exported as NIfTI."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(mock_subject_data, output_dir)

        # Find lesion mask file
        anat_dir = output_dir / "sub-001" / "anat"
        mask_files = list(anat_dir.glob("*label-lesion_mask.nii.gz"))

        assert len(mask_files) == 1
        assert "space-MNI152NLin6Asym" in mask_files[0].name

    def test_skip_lesion_mask_export(self, mock_subject_data, tmp_path):
        """Test that lesion mask export can be skipped."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(mock_subject_data, output_dir, export_lesion_mask=False)

        anat_dir = output_dir / "sub-001" / "anat"
        mask_files = list(anat_dir.glob("*label-lesion_mask.nii.gz"))

        assert len(mask_files) == 0

    def test_raises_without_subject_id(self, tmp_path):
        """Test that export raises ValueError if subject_id is missing."""
        import nibabel as nib

        from lacuna.core.subject_data import SubjectData
        from lacuna.io.bids import export_bids_derivatives

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        # SubjectData defaults subject_id to "sub-unknown" if not provided
        # We need to explicitly set an empty subject_id by overwriting after creation
        subject_data = SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={},  # Will default to "sub-unknown"
        )
        # Manually remove subject_id to test the validation
        subject_data._metadata = {}

        output_dir = tmp_path / "derivatives"
        with pytest.raises(ValueError, match="subject_id"):
            export_bids_derivatives(subject_data, output_dir)

    def test_raises_if_file_exists_without_overwrite(self, mock_subject_data, tmp_path):
        """Test that export raises FileExistsError when overwrite=False."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"

        # Export once
        export_bids_derivatives(mock_subject_data, output_dir)

        # Second export should fail without overwrite
        with pytest.raises(FileExistsError):
            export_bids_derivatives(mock_subject_data, output_dir, overwrite=False)

    def test_overwrites_with_overwrite_true(self, mock_subject_data, tmp_path):
        """Test that export overwrites when overwrite=True."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"

        # Export twice with overwrite
        export_bids_derivatives(mock_subject_data, output_dir)
        export_bids_derivatives(mock_subject_data, output_dir, overwrite=True)

        # Should not raise

    def test_returns_subject_directory_path(self, mock_subject_data, tmp_path):
        """Test that export returns the subject directory path."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        result = export_bids_derivatives(mock_subject_data, output_dir)

        assert isinstance(result, Path)
        assert result.parent == output_dir


class TestExportBidsDerivativesBatch:
    """Tests for export_bids_derivatives_batch function."""

    @pytest.fixture
    def mock_subjects(self, tmp_path):
        """Create multiple mock SubjectData objects."""
        import nibabel as nib

        from lacuna.core.subject_data import SubjectData

        subjects = []
        for i in range(3):
            mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
            mask_array[4:6, 4:6, 4:6] = 1
            affine = np.eye(4)
            mask_img = nib.Nifti1Image(mask_array, affine)

            subject = SubjectData(
                mask_img=mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
                metadata={"subject_id": f"sub-{i+1:03d}"},
            )
            subjects.append(subject)

        return subjects

    def test_exports_all_subjects(self, mock_subjects, tmp_path):
        """Test that batch export creates directories for all subjects."""
        from lacuna.io.bids import export_bids_derivatives_batch

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives_batch(mock_subjects, output_dir)

        # Check all subjects exist
        assert (output_dir / "sub-001").exists()
        assert (output_dir / "sub-002").exists()
        assert (output_dir / "sub-003").exists()

    def test_creates_single_dataset_description(self, mock_subjects, tmp_path):
        """Test that batch export creates one dataset_description.json."""
        from lacuna.io.bids import export_bids_derivatives_batch

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives_batch(mock_subjects, output_dir)

        desc_files = list(output_dir.glob("**/dataset_description.json"))
        assert len(desc_files) == 1

    def test_returns_output_dir(self, mock_subjects, tmp_path):
        """Test that batch export returns the output directory."""
        from lacuna.io.bids import export_bids_derivatives_batch

        output_dir = tmp_path / "derivatives"
        result = export_bids_derivatives_batch(mock_subjects, output_dir)

        assert result == output_dir


class TestExportWithResults:
    """Tests for exporting SubjectData with analysis results."""

    @pytest.fixture
    def subject_with_results(self, tmp_path):
        """Create SubjectData with mock analysis results."""
        import nibabel as nib

        from lacuna.core.data_types import ParcelData, ScalarMetric, VoxelMap
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        mask_img = nib.Nifti1Image(mask_array, affine)

        subject = SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

        # Add mock VoxelMap result (uses 'data' not 'image')
        voxel_data = np.random.rand(10, 10, 10).astype(np.float32)
        voxel_img = nib.Nifti1Image(voxel_data, affine)
        voxel_map = VoxelMap(
            name="correlation",
            data=voxel_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
        )

        # Add mock ParcelData result (uses 'data' not 'values')
        parcel_data = ParcelData(
            name="regional_damage",
            data={"region_1": 0.5, "region_2": 0.3},
        )

        # Add mock ScalarMetric result (uses 'data' not 'value')
        scalar = ScalarMetric(name="test_metric", data=42.0)

        # Store results
        subject._results = {
            "test_analysis": {
                "voxelmap": voxel_map,
                "parceldata": parcel_data,
                "scalar": scalar,
            }
        }

        return subject

    def test_exports_voxelmap_as_nifti(self, subject_with_results, tmp_path):
        """Test that VoxelMap results are exported as NIfTI."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(subject_with_results, output_dir)

        anat_dir = output_dir / "sub-001" / "anat"
        nifti_files = list(anat_dir.glob("*.nii.gz"))

        # Should have lesion mask + voxelmap
        assert len(nifti_files) >= 2

    def test_skip_voxelmap_export(self, subject_with_results, tmp_path):
        """Test that VoxelMap export can be skipped."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(
            subject_with_results,
            output_dir,
            export_voxelmaps=False,
            export_parcel_data=False,
            export_scalars=False,
        )

        anat_dir = output_dir / "sub-001" / "anat"
        nifti_files = list(anat_dir.glob("*.nii.gz"))

        # Should have only lesion mask
        assert len(nifti_files) == 1
        assert "label-lesion" in nifti_files[0].name

    def test_exports_parcel_data_as_tsv(self, subject_with_results, tmp_path):
        """Test that ParcelData results are exported as TSV."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(subject_with_results, output_dir)

        anat_dir = output_dir / "sub-001" / "anat"
        tsv_files = list(anat_dir.glob("*.tsv"))

        assert len(tsv_files) >= 1

    def test_exports_scalar_as_json(self, subject_with_results, tmp_path):
        """Test that ScalarMetric results are exported as JSON."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(subject_with_results, output_dir)

        anat_dir = output_dir / "sub-001" / "anat"
        # ScalarMetric exports to *_stats.json files
        json_files = list(anat_dir.glob("*_stats.json"))

        assert len(json_files) >= 1


class TestExportProvenance:
    """Tests for provenance export functionality."""

    @pytest.fixture
    def subject_with_provenance(self, tmp_path):
        """Create SubjectData with provenance history."""
        import nibabel as nib

        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        affine = np.eye(4)
        mask_img = nib.Nifti1Image(mask_array, affine)

        subject = SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

        # Add provenance as a dict (the standard format)
        provenance_entry = {
            "operation": "test_analysis",
            "parameters": {"param1": "value1"},
            "timestamp": "2024-01-01T00:00:00Z",
        }
        subject._provenance = [provenance_entry]

        return subject

    def test_exports_provenance_as_json(self, subject_with_provenance, tmp_path):
        """Test that provenance is exported as JSON."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(subject_with_provenance, output_dir)

        anat_dir = output_dir / "sub-001" / "anat"
        prov_files = list(anat_dir.glob("*provenance.json"))

        assert len(prov_files) == 1

    def test_skip_provenance_export(self, subject_with_provenance, tmp_path):
        """Test that provenance export can be skipped."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives"
        export_bids_derivatives(
            subject_with_provenance,
            output_dir,
            export_provenance=False,
        )

        anat_dir = output_dir / "sub-001" / "anat"
        prov_files = list(anat_dir.glob("*provenance.json"))

        assert len(prov_files) == 0
