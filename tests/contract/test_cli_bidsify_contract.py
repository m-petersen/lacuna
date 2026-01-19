"""
Contract tests for the lacuna bidsify CLI subcommand.

These tests define the expected behavior and API contracts for the bidsify
command which converts a directory of NIfTI files to BIDS format.
"""

import subprocess
import sys

import pytest


class TestBidsifySubcommandContract:
    """Test bidsify subcommand exists and has expected interface."""

    def test_bidsify_help_exits_zero(self):
        """lacuna bidsify --help should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "bidsify", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "bidsify" in result.stdout.lower() or "bids" in result.stdout.lower()

    def test_bidsify_requires_input_directory(self):
        """lacuna bidsify should require input directory argument."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "bidsify"],
            capture_output=True,
            text=True,
        )
        # Should fail without required arguments
        assert result.returncode != 0

    def test_bidsify_help_shows_space_option(self):
        """bidsify help should show --space option."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "bidsify", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--space" in result.stdout

    def test_bidsify_help_shows_session_option(self):
        """bidsify help should show --session option."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "bidsify", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--session" in result.stdout or "-ses" in result.stdout

    def test_bidsify_help_shows_label_option(self):
        """bidsify help should show --label option."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "bidsify", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--label" in result.stdout

    def test_bidsify_space_only_allows_valid_values(self):
        """bidsify --space should only accept MNI152NLin6Asym or MNI152NLin2009cAsym."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "bidsify", "--help"],
            capture_output=True,
            text=True,
        )
        # Both valid spaces should be mentioned in help
        assert "MNI152NLin6Asym" in result.stdout
        assert "MNI152NLin2009cAsym" in result.stdout


class TestBidsifyFunctionality:
    """Test bidsify command creates proper BIDS structure."""

    def test_bidsify_creates_bids_directory_structure(self, tmp_path):
        """bidsify should create BIDS-compliant directory structure."""
        import nibabel as nib
        import numpy as np

        # Create input directory with NIfTI files
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create a simple NIfTI file
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[3:7, 3:7, 3:7] = 1
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient001.nii.gz")

        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "bidsify",
                str(input_dir),
                str(output_dir),
                "--space",
                "MNI152NLin6Asym",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_dir.exists()
        # Should create sub-patient001 directory
        assert (output_dir / "sub-patient001").exists()

    def test_bidsify_creates_dataset_description(self, tmp_path):
        """bidsify should create dataset_description.json."""
        import nibabel as nib
        import numpy as np

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "test.nii.gz")

        output_dir = tmp_path / "output"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "bidsify",
                str(input_dir),
                str(output_dir),
                "--space",
                "MNI152NLin6Asym",
            ],
            capture_output=True,
            text=True,
        )

        assert (output_dir / "dataset_description.json").exists()

    def test_bidsify_with_session_creates_session_directory(self, tmp_path):
        """bidsify --session should create session subdirectory."""
        import nibabel as nib
        import numpy as np

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "subj.nii.gz")

        output_dir = tmp_path / "output"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "bidsify",
                str(input_dir),
                str(output_dir),
                "--space",
                "MNI152NLin6Asym",
                "--session",
                "01",
            ],
            capture_output=True,
            text=True,
        )

        assert (output_dir / "sub-subj" / "ses-01").exists()

    def test_bidsify_with_label_includes_label_in_filename(self, tmp_path):
        """bidsify --label should include label entity in output filename."""
        import nibabel as nib
        import numpy as np

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "patient.nii.gz")

        output_dir = tmp_path / "output"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "bidsify",
                str(input_dir),
                str(output_dir),
                "--space",
                "MNI152NLin6Asym",
                "--label",
                "lesion",
            ],
            capture_output=True,
            text=True,
        )

        # Find the mask file
        mask_files = list(output_dir.rglob("*mask.nii.gz"))
        assert len(mask_files) >= 1
        assert "label-lesion" in mask_files[0].name

    def test_bidsify_rejects_invalid_space(self, tmp_path):
        """bidsify should reject invalid space values."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        import nibabel as nib
        import numpy as np

        data = np.zeros((10, 10, 10), dtype=np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)
        nib.save(img, input_dir / "test.nii.gz")

        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "bidsify",
                str(input_dir),
                str(output_dir),
                "--space",
                "InvalidSpace",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0


class TestTutorialCommandContract:
    """Test tutorial subcommand."""

    def test_tutorial_help_exits_zero(self):
        """lacuna tutorial --help should exit with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "tutorial", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "tutorial" in result.stdout.lower()

    def test_tutorial_copies_to_directory(self, tmp_path):
        """lacuna tutorial should copy tutorial data to specified directory."""
        output_dir = tmp_path / "tutorial_data"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "tutorial",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert output_dir.exists()
        # Should have BIDS structure
        assert (output_dir / "dataset_description.json").exists()
        # Should have subjects
        assert len(list(output_dir.glob("sub-*"))) >= 1

    def test_tutorial_without_output_uses_default(self, tmp_path, monkeypatch):
        """lacuna tutorial without output_dir should use lacuna_tutorial in current directory."""
        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lacuna",
                "tutorial",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0
        # Tutorial data should be in lacuna_tutorial subdirectory
        tutorial_dir = tmp_path / "lacuna_tutorial"
        assert tutorial_dir.exists() or (tmp_path / "dataset_description.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
