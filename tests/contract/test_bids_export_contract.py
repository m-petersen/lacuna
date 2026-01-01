"""Contract tests for BIDS derivative export.

These tests define the expected behavior for BIDS derivative export
for all container types with proper sidecars.

Contract:  BIDS Derivative Export
"""

import json

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.data_types import ConnectivityMatrix, ParcelData, VoxelMap
from lacuna.core.subject_data import SubjectData


@pytest.fixture
def sample_mask_data_for_export():
    """Create SubjectData with results for BIDS export testing."""
    shape = (10, 10, 10)
    affine = np.eye(4) * 2
    affine[3, 3] = 1

    data = np.zeros(shape, dtype=np.int8)
    data[4:6, 4:6, 4:6] = 1
    img = nib.Nifti1Image(data, affine)

    mask_data = SubjectData(
        mask_img=img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": "sub-001", "session_id": "ses-01"},
    )

    # Add VoxelMap result
    voxel_data = np.random.rand(*shape).astype(np.float32)
    voxel_img = nib.Nifti1Image(voxel_data, affine)
    voxel_map = VoxelMap(
        name="rmap",
        data=voxel_img,
        space="MNI152NLin6Asym",
        resolution=2.0,
    )

    # Add ParcelData result
    parcel_data = ParcelData(
        name="parcelmeans",
        data={"region_A": 0.5, "region_B": 0.3, "region_C": 0.8},
        parcel_names=["TestAtlas"],
    )

    # Add results to mask_data
    results = {
        "FunctionalNetworkMapping": {
            "rmap": voxel_map,
            "parcelmeans": parcel_data,
        }
    }

    return mask_data.add_result("FunctionalNetworkMapping", results["FunctionalNetworkMapping"])


@pytest.fixture
def sample_voxelmap():
    """Create a standalone VoxelMap for export testing."""
    shape = (10, 10, 10)
    affine = np.eye(4) * 2
    affine[3, 3] = 1

    data = np.random.rand(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, affine)

    return VoxelMap(
        name="zmap",
        data=img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": "sub-002"},
    )


@pytest.fixture
def sample_parcel_data():
    """Create standalone ParcelData for export testing."""
    return ParcelData(
        name="damage_scores",
        data={
            "Left_Hippocampus": 0.45,
            "Right_Hippocampus": 0.12,
            "Left_Amygdala": 0.78,
        },
        parcel_names=["TianSubcortex_3TS1"],
        metadata={"subject_id": "sub-003"},
    )


@pytest.fixture
def sample_connectivity_matrix():
    """Create ConnectivityMatrix for export testing."""
    matrix = np.random.rand(10, 10).astype(np.float32)
    matrix = (matrix + matrix.T) / 2  # Make symmetric

    return ConnectivityMatrix(
        name="structural_connectivity",
        matrix=matrix,
        region_labels=[f"region_{i}" for i in range(10)],
        matrix_type="structural",
        metadata={"subject_id": "sub-004"},
    )


class TestBidsExportContract:
    """Contract tests for BIDS derivative export."""

    def test_export_creates_dataset_description(self, sample_mask_data_for_export, tmp_path):
        """Export should create dataset_description.json."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives" / "lacuna"
        export_bids_derivatives(sample_mask_data_for_export, output_dir, overwrite=True)

        desc_file = output_dir / "dataset_description.json"
        assert desc_file.exists()

        with open(desc_file) as f:
            desc = json.load(f)

        assert "Name" in desc
        assert "BIDSVersion" in desc
        assert "GeneratedBy" in desc

    def test_export_creates_subject_directory(self, sample_mask_data_for_export, tmp_path):
        """Export should create subject directory structure."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives" / "lacuna"
        result_path = export_bids_derivatives(
            sample_mask_data_for_export, output_dir, overwrite=True
        )

        assert result_path.exists()
        assert "sub-001" in str(result_path)

    def test_export_saves_lesion_mask_nifti(self, sample_mask_data_for_export, tmp_path):
        """Export should save lesion mask as NIfTI with BIDS naming."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives" / "lacuna"
        result_path = export_bids_derivatives(
            sample_mask_data_for_export, output_dir, overwrite=True
        )

        # Look for NIfTI file
        nifti_files = list(result_path.rglob("*.nii.gz"))
        assert len(nifti_files) >= 1

        # Check BIDS naming pattern
        found_lesion = any("mask" in f.name or "lesion" in f.name for f in nifti_files)
        assert found_lesion

    def test_export_respects_overwrite_false(self, sample_mask_data_for_export, tmp_path):
        """Export should raise error when overwrite=False and files exist."""
        from lacuna.io.bids import export_bids_derivatives

        output_dir = tmp_path / "derivatives" / "lacuna"

        # First export
        export_bids_derivatives(sample_mask_data_for_export, output_dir, overwrite=True)

        # Second export should raise
        with pytest.raises(FileExistsError):
            export_bids_derivatives(sample_mask_data_for_export, output_dir, overwrite=False)


class TestVoxelMapExportContract:
    """Contract tests for VoxelMap BIDS export."""

    def test_export_voxelmap_creates_nifti(self, sample_voxelmap, tmp_path):
        """VoxelMap export should create NIfTI file with BIDS naming."""
        from lacuna.io.bids import export_voxelmap

        output_path = export_voxelmap(
            sample_voxelmap,
            tmp_path,
            subject_id="sub-002",
            desc="zmap",
        )

        assert output_path.exists()
        assert output_path.suffix == ".gz" or output_path.name.endswith(".nii")

    def test_export_voxelmap_creates_sidecar(self, sample_voxelmap, tmp_path):
        """VoxelMap export should create JSON sidecar."""
        from lacuna.io.bids import export_voxelmap

        output_path = export_voxelmap(
            sample_voxelmap,
            tmp_path,
            subject_id="sub-002",
            desc="zmap",
        )

        sidecar_path = output_path.with_suffix("").with_suffix(".json")
        assert sidecar_path.exists()

        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert "Space" in sidecar or "space" in sidecar
        assert "Resolution" in sidecar or "resolution" in sidecar


class TestParcelDataExportContract:
    """Contract tests for ParcelData BIDS export."""

    def test_export_parcel_data_creates_tsv(self, sample_parcel_data, tmp_path):
        """ParcelData export should create TSV file."""
        from lacuna.io.bids import export_parcel_data

        output_path = export_parcel_data(
            sample_parcel_data,
            tmp_path,
            subject_id="sub-003",
            desc="damage",
        )

        assert output_path.exists()
        assert output_path.suffix == ".tsv"

    def test_export_parcel_data_creates_sidecar(self, sample_parcel_data, tmp_path):
        """ParcelData export should create JSON sidecar."""
        from lacuna.io.bids import export_parcel_data

        output_path = export_parcel_data(
            sample_parcel_data,
            tmp_path,
            subject_id="sub-003",
            desc="damage",
        )

        sidecar_path = output_path.with_suffix(".json")
        assert sidecar_path.exists()

    def test_export_parcel_data_tsv_has_columns(self, sample_parcel_data, tmp_path):
        """ParcelData TSV should have region and value columns."""
        import pandas as pd

        from lacuna.io.bids import export_parcel_data

        output_path = export_parcel_data(
            sample_parcel_data,
            tmp_path,
            subject_id="sub-003",
            desc="damage",
        )

        df = pd.read_csv(output_path, sep="\t")
        assert "region" in df.columns or "Region" in df.columns
        assert "value" in df.columns or "Value" in df.columns


class TestConnectivityMatrixExportContract:
    """Contract tests for ConnectivityMatrix BIDS export."""

    def test_export_connectivity_matrix_creates_tsv(self, sample_connectivity_matrix, tmp_path):
        """ConnectivityMatrix export should create TSV file."""
        from lacuna.io.bids import export_connectivity_matrix

        output_path = export_connectivity_matrix(
            sample_connectivity_matrix,
            tmp_path,
            subject_id="sub-004",
            desc="structural",
        )

        assert output_path.exists()
        assert output_path.suffix == ".tsv"

    def test_export_connectivity_matrix_preserves_labels(
        self, sample_connectivity_matrix, tmp_path
    ):
        """ConnectivityMatrix export should preserve region labels."""
        import pandas as pd

        from lacuna.io.bids import export_connectivity_matrix

        output_path = export_connectivity_matrix(
            sample_connectivity_matrix,
            tmp_path,
            subject_id="sub-004",
            desc="structural",
        )

        df = pd.read_csv(output_path, sep="\t", index_col=0)
        assert list(df.columns) == sample_connectivity_matrix.region_labels


class TestBatchExportContract:
    """Contract tests for batch BIDS export."""

    def test_batch_export_creates_dataset_description(self, sample_mask_data_for_export, tmp_path):
        """Batch export should create single dataset_description.json."""
        from lacuna.io.bids import export_bids_derivatives_batch

        results = [sample_mask_data_for_export]
        output_dir = tmp_path / "derivatives" / "lacuna"

        export_bids_derivatives_batch(results, output_dir, overwrite=True)

        desc_file = output_dir / "dataset_description.json"
        assert desc_file.exists()

    def test_batch_export_handles_multiple_subjects(self, tmp_path):
        """Batch export should handle multiple subjects."""
        from lacuna.io.bids import export_bids_derivatives_batch

        # Create multiple subjects
        subjects = []
        for i in range(3):
            shape = (10, 10, 10)
            affine = np.eye(4) * 2
            affine[3, 3] = 1

            data = np.zeros(shape, dtype=np.int8)
            data[4:6, 4:6, 4:6] = 1
            img = nib.Nifti1Image(data, affine)

            mask_data = SubjectData(
                mask_img=img,
                space="MNI152NLin6Asym",
                resolution=2.0,
                metadata={"subject_id": f"sub-{i:03d}"},
            )
            subjects.append(mask_data)

        output_dir = tmp_path / "derivatives" / "lacuna"
        export_bids_derivatives_batch(subjects, output_dir, overwrite=True)

        # Check all subjects created
        for i in range(3):
            subject_dir = output_dir / f"sub-{i:03d}"
            assert subject_dir.exists()
