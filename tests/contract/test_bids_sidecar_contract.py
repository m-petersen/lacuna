"""Contract tests for BIDS sidecar parsing."""

import json

import nibabel as nib
import numpy as np
import pytest

from lacuna.io.bids import load_bids_dataset


class TestBidsSidecarContract:
    """Contract tests for BIDS sidecar parsing during loading."""

    @pytest.fixture
    def bids_dataset_with_sidecars(self, tmp_path):
        """Create a minimal BIDS dataset with JSON sidecars."""
        # Create dataset_description.json (required for valid BIDS)
        dataset_description = {
            "Name": "Test Dataset",
            "BIDSVersion": "1.6.0",
        }
        with open(tmp_path / "dataset_description.json", "w") as f:
            json.dump(dataset_description, f)

        # Create subject directory
        anat_dir = tmp_path / "sub-001" / "anat"
        anat_dir.mkdir(parents=True)

        # Create lesion mask NIfTI
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        img = nib.Nifti1Image(data, affine)
        nib.save(img, anat_dir / "sub-001_mask-lesion.nii.gz")

        # Create T1w anatomical
        anat_data = np.random.rand(10, 10, 10).astype(np.float32)
        anat_img = nib.Nifti1Image(anat_data, affine)
        nib.save(anat_img, anat_dir / "sub-001_T1w.nii.gz")

        # Create JSON sidecar with space/resolution info
        sidecar = {
            "Space": "MNI152NLin6Asym",
            "Resolution": 2,  # Numeric value
            "Description": "Lesion mask in standard space",
        }
        with open(anat_dir / "sub-001_mask-lesion.json", "w") as f:
            json.dump(sidecar, f)

        return tmp_path

    @pytest.fixture
    def bids_dataset_with_string_resolution(self, tmp_path):
        """Create a BIDS dataset with string resolution like '2mm'."""
        # Create dataset_description.json (required for valid BIDS)
        dataset_description = {
            "Name": "Test Dataset String Resolution",
            "BIDSVersion": "1.6.0",
        }
        with open(tmp_path / "dataset_description.json", "w") as f:
            json.dump(dataset_description, f)

        anat_dir = tmp_path / "sub-003" / "anat"
        anat_dir.mkdir(parents=True)

        # Create lesion mask NIfTI
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        img = nib.Nifti1Image(data, affine)
        nib.save(img, anat_dir / "sub-003_mask-lesion.nii.gz")

        # Create T1w anatomical
        anat_data = np.random.rand(10, 10, 10).astype(np.float32)
        anat_img = nib.Nifti1Image(anat_data, affine)
        nib.save(anat_img, anat_dir / "sub-003_T1w.nii.gz")

        # Create JSON sidecar with string resolution
        sidecar = {
            "Space": "MNI152NLin6Asym",
            "Resolution": "2mm",  # String format
            "Description": "Lesion mask in standard space",
        }
        with open(anat_dir / "sub-003_mask-lesion.json", "w") as f:
            json.dump(sidecar, f)

        return tmp_path

    def test_sidecar_space_extracted(self, bids_dataset_with_sidecars):
        """Contract: JSON sidecar Space field is extracted to SubjectData.space."""
        result = load_bids_dataset(bids_dataset_with_sidecars, validate_bids=False)
        mask_data = result["sub-001"]
        assert mask_data.space == "MNI152NLin6Asym"

    def test_sidecar_resolution_extracted(self, bids_dataset_with_sidecars):
        """Contract: JSON sidecar Resolution field is extracted to SubjectData.resolution."""
        result = load_bids_dataset(bids_dataset_with_sidecars, validate_bids=False)
        mask_data = result["sub-001"]
        assert mask_data.resolution == 2.0

    def test_sidecar_string_resolution_parsed(self, bids_dataset_with_string_resolution):
        """Contract: String resolution '2mm' is parsed to numeric 2.0."""
        result = load_bids_dataset(bids_dataset_with_string_resolution, validate_bids=False)
        mask_data = result["sub-003"]
        assert mask_data.resolution == 2.0

    def test_sidecar_metadata_preserved(self, bids_dataset_with_sidecars):
        """Contract: BIDS metadata is preserved in SubjectData.metadata."""
        result = load_bids_dataset(bids_dataset_with_sidecars, validate_bids=False)
        mask_data = result["sub-001"]
        assert mask_data.metadata["subject_id"] == "sub-001"
        assert "lesion_path" in mask_data.metadata
