"""
Unit tests for the bidsify module.

Tests the core logic for converting NIfTI files to BIDS format.
"""

import json

import nibabel as nib
import numpy as np
import pytest


class TestBidsifyFunction:
    """Test the bidsify function logic."""

    def test_bidsify_single_file(self, tmp_path):
        """bidsify should convert a single NIfTI file to BIDS structure."""
        from lacuna.io.bidsify import bidsify

        # Create input directory with one NIfTI file
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[3:7, 3:7, 3:7] = 1
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient001.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
        )

        # Check structure
        assert output_dir.exists()
        assert (output_dir / "sub-patient001").exists()
        assert (output_dir / "dataset_description.json").exists()

    def test_bidsify_multiple_files(self, tmp_path):
        """bidsify should convert multiple NIfTI files."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        for name in ["subj_a", "subj_b", "subj_c"]:
            nib.save(img, input_dir / f"{name}.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
        )

        assert (output_dir / "sub-subja").exists()
        assert (output_dir / "sub-subjb").exists()
        assert (output_dir / "sub-subjc").exists()

    def test_bidsify_with_session(self, tmp_path):
        """bidsify with session should create session subdirectory."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
            session="01",
        )

        assert (output_dir / "sub-patient" / "ses-01" / "anat").exists()

    def test_bidsify_with_label(self, tmp_path):
        """bidsify with label should include label entity in filename."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
            label="lesion",
        )

        mask_files = list(output_dir.rglob("*mask.nii.gz"))
        assert len(mask_files) == 1
        assert "label-lesion" in mask_files[0].name

    def test_bidsify_filename_becomes_subject_id(self, tmp_path):
        """bidsify should use filename (without extension) as subject ID."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "MyPatient123.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
        )

        # Subject ID should be filename with non-alphanumeric stripped
        assert (output_dir / "sub-MyPatient123").exists()

    def test_bidsify_creates_participants_tsv(self, tmp_path):
        """bidsify should create participants.tsv file."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
        )

        assert (output_dir / "participants.tsv").exists()

    def test_bidsify_dataset_description_content(self, tmp_path):
        """bidsify should create valid dataset_description.json."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
        )

        with open(output_dir / "dataset_description.json") as f:
            desc = json.load(f)

        assert "Name" in desc
        assert "BIDSVersion" in desc

    def test_bidsify_correct_bids_filename_format(self, tmp_path):
        """bidsify should create correctly formatted BIDS filenames."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "test.nii.gz")

        output_dir = tmp_path / "output"

        bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space="MNI152NLin6Asym",
            session="01",
            label="lesion",
        )

        # Expected format: sub-<id>_ses-<id>_space-<space>_label-<label>_mask.nii.gz
        mask_files = list(output_dir.rglob("*mask.nii.gz"))
        assert len(mask_files) == 1
        filename = mask_files[0].name
        assert filename.startswith("sub-test_")
        assert "ses-01" in filename
        assert "space-MNI152NLin6Asym" in filename
        assert "label-lesion" in filename
        assert filename.endswith("_mask.nii.gz")


class TestBidsifyValidation:
    """Test input validation for bidsify."""

    def test_bidsify_invalid_space_raises_error(self, tmp_path):
        """bidsify should raise error for invalid space."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "test.nii.gz")

        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="[Ss]pace|[Ii]nvalid"):
            bidsify(
                input_dir=input_dir,
                output_dir=output_dir,
                space="InvalidSpace",
            )

    def test_bidsify_nonexistent_input_raises_error(self, tmp_path):
        """bidsify should raise error for non-existent input directory."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"

        with pytest.raises(FileNotFoundError):
            bidsify(
                input_dir=input_dir,
                output_dir=output_dir,
                space="MNI152NLin6Asym",
            )

    def test_bidsify_empty_input_raises_error(self, tmp_path):
        """bidsify should raise error for empty input directory."""
        from lacuna.io.bidsify import bidsify

        input_dir = tmp_path / "input"
        input_dir.mkdir()  # Empty directory

        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="[Nn]o.*[Nn]ifti|[Ee]mpty"):
            bidsify(
                input_dir=input_dir,
                output_dir=output_dir,
                space="MNI152NLin6Asym",
            )


class TestSubjectIdSanitization:
    """Test subject ID sanitization logic."""

    def test_sanitize_subject_id_removes_special_chars(self):
        """sanitize_subject_id should remove special characters."""
        from lacuna.io.bidsify import sanitize_subject_id

        assert sanitize_subject_id("patient_001") == "patient001"
        assert sanitize_subject_id("sub-test") == "subtest"
        assert sanitize_subject_id("file.name") == "filename"

    def test_sanitize_subject_id_handles_underscores(self):
        """sanitize_subject_id should handle underscores."""
        from lacuna.io.bidsify import sanitize_subject_id

        assert sanitize_subject_id("patient_a_b") == "patientab"

    def test_sanitize_subject_id_preserves_alphanumeric(self):
        """sanitize_subject_id should preserve alphanumeric characters."""
        from lacuna.io.bidsify import sanitize_subject_id

        assert sanitize_subject_id("Patient123ABC") == "Patient123ABC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
