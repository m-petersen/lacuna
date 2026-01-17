"""Unit tests for the tutorial data module.

Tests the bundled tutorial BIDS dataset access functions.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.data.tutorials import (
    get_subject_mask_path,
    get_tutorial_bids_dir,
    get_tutorial_subjects,
    setup_tutorial_data,
)


class TestGetTutorialBidsDir:
    """Test get_tutorial_bids_dir function."""

    def test_returns_existing_path(self):
        """get_tutorial_bids_dir() returns a path that exists."""
        bids_dir = get_tutorial_bids_dir()
        assert bids_dir.exists()
        assert bids_dir.is_dir()

    def test_contains_bids_files(self):
        """Tutorial BIDS directory contains required BIDS files."""
        bids_dir = get_tutorial_bids_dir()

        # Should have dataset_description.json
        assert (bids_dir / "dataset_description.json").exists()

        # Should have participants.tsv
        assert (bids_dir / "participants.tsv").exists()

    def test_contains_subject_directories(self):
        """Tutorial BIDS directory contains subject directories."""
        bids_dir = get_tutorial_bids_dir()
        subject_dirs = list(bids_dir.glob("sub-*"))
        assert len(subject_dirs) >= 1


class TestGetTutorialSubjects:
    """Test get_tutorial_subjects function."""

    def test_returns_list_of_subjects(self):
        """get_tutorial_subjects() returns a list of subject IDs."""
        subjects = get_tutorial_subjects()
        assert isinstance(subjects, list)
        assert len(subjects) >= 1

    def test_subjects_have_correct_prefix(self):
        """Subject IDs have 'sub-' prefix."""
        subjects = get_tutorial_subjects()
        for subject in subjects:
            assert subject.startswith("sub-")

    def test_returns_expected_subjects(self):
        """Tutorial data contains expected 3 subjects."""
        subjects = get_tutorial_subjects()
        assert len(subjects) == 3
        assert "sub-01" in subjects
        assert "sub-02" in subjects
        assert "sub-03" in subjects


class TestGetSubjectMaskPath:
    """Test get_subject_mask_path function."""

    def test_returns_existing_path(self):
        """get_subject_mask_path() returns path to existing file."""
        subjects = get_tutorial_subjects()
        mask_path = get_subject_mask_path(subjects[0])
        assert mask_path.exists()
        assert mask_path.is_file()

    def test_path_is_nifti(self):
        """Mask path points to a NIfTI file."""
        subjects = get_tutorial_subjects()
        mask_path = get_subject_mask_path(subjects[0])
        assert mask_path.suffix == ".gz"
        assert ".nii" in mask_path.name

    def test_mask_is_valid_nifti(self):
        """Mask file is a valid NIfTI that can be loaded."""
        subjects = get_tutorial_subjects()
        mask_path = get_subject_mask_path(subjects[0])
        mask_img = nib.load(mask_path)
        assert mask_img is not None
        assert len(mask_img.shape) == 3

    def test_mask_is_binary(self):
        """Mask contains only 0 and 1 values."""
        subjects = get_tutorial_subjects()
        mask_path = get_subject_mask_path(subjects[0])
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        unique_values = np.unique(mask_data)
        assert set(unique_values).issubset({0, 1})

    def test_mask_is_mni_space(self):
        """Mask is in expected MNI space with correct dimensions."""
        subjects = get_tutorial_subjects()
        mask_path = get_subject_mask_path(subjects[0])
        mask_img = nib.load(mask_path)

        # MNI152NLin6Asym 1mm = 182x218x182
        assert mask_img.shape == (182, 218, 182)

    def test_invalid_subject_raises_error(self):
        """get_subject_mask_path() raises error for invalid subject."""
        with pytest.raises(FileNotFoundError):
            get_subject_mask_path("sub-invalid")

    def test_all_subjects_have_masks(self):
        """All tutorial subjects have mask files."""
        subjects = get_tutorial_subjects()
        for subject in subjects:
            mask_path = get_subject_mask_path(subject)
            assert mask_path.exists()


class TestSetupTutorialData:
    """Test setup_tutorial_data function."""

    def test_copies_to_target_directory(self, tmp_path):
        """setup_tutorial_data() copies data to target directory."""
        target = tmp_path / "tutorial_copy"
        result = setup_tutorial_data(target)

        assert result.exists()
        assert result.is_dir()
        assert result == target

    def test_copied_data_has_subjects(self, tmp_path):
        """Copied data contains subject directories."""
        target = tmp_path / "tutorial_copy"
        setup_tutorial_data(target)

        subjects = list(target.glob("sub-*"))
        assert len(subjects) == 3

    def test_copied_data_has_bids_files(self, tmp_path):
        """Copied data contains BIDS metadata files."""
        target = tmp_path / "tutorial_copy"
        setup_tutorial_data(target)

        assert (target / "dataset_description.json").exists()
        assert (target / "participants.tsv").exists()

    def test_existing_directory_raises_error(self, tmp_path):
        """setup_tutorial_data() raises error if target exists."""
        target = tmp_path / "tutorial_copy"
        target.mkdir()

        with pytest.raises(FileExistsError):
            setup_tutorial_data(target)

    def test_overwrite_replaces_existing(self, tmp_path):
        """setup_tutorial_data(overwrite=True) replaces existing."""
        target = tmp_path / "tutorial_copy"
        target.mkdir()
        (target / "old_file.txt").write_text("old content")

        setup_tutorial_data(target, overwrite=True)

        # Old file should be gone
        assert not (target / "old_file.txt").exists()
        # New data should be present
        assert (target / "dataset_description.json").exists()
