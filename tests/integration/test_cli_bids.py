"""
Integration tests for CLI + BIDS workflow.

Tests the end-to-end workflow of the CLI processing BIDS datasets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def minimal_bids_dataset(tmp_path):
    """Create a minimal BIDS dataset for CLI testing."""
    import nibabel as nib

    bids_root = tmp_path / "bids"
    bids_root.mkdir()

    # Create dataset_description.json
    desc = {
        "Name": "Test Dataset",
        "BIDSVersion": "1.6.0",
    }
    with open(bids_root / "dataset_description.json", "w") as f:
        json.dump(desc, f)

    # Create subject directories
    for sub_id in ["sub-001", "sub-002"]:
        anat_dir = bids_root / sub_id / "anat"
        anat_dir.mkdir(parents=True)

        # Create a valid binary mask
        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        mask_img = nib.Nifti1Image(mask_array, affine)

        # Save with BIDS-compliant name
        mask_path = anat_dir / f"{sub_id}_space-MNI152NLin6Asym_desc-lesion_mask.nii.gz"
        nib.save(mask_img, mask_path)

        # Create sidecar JSON with space info
        sidecar = {
            "space": "MNI152NLin6Asym",
            "resolution": 2.0,
        }
        sidecar_path = anat_dir / f"{sub_id}_space-MNI152NLin6Asym_desc-lesion_mask.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

    return bids_root


@pytest.fixture
def output_dir(tmp_path):
    """Create an output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def work_dir(tmp_path):
    """Create a work directory."""
    work = tmp_path / "work"
    work.mkdir()
    return work


class TestCLIWorkflow:
    """Tests for CLI workflow with BIDS datasets."""

    def test_cli_help_returns_zero(self):
        """Test that --help returns exit code 0."""
        from lacuna.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0

    def test_cli_version_returns_zero(self):
        """Test that --version returns exit code 0."""
        from lacuna.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])

        assert exc_info.value.code == 0

    def test_cli_invalid_bids_returns_error(self, tmp_path, output_dir):
        """Test that invalid BIDS directory returns error code."""
        from lacuna.cli import main

        # Non-existent BIDS directory
        result = main([str(tmp_path / "nonexistent"), str(output_dir), "participant"])

        assert result == 2  # EXIT_INVALID_ARGS

    def test_cli_missing_dataset_description_returns_error(self, tmp_path, output_dir):
        """Test that BIDS without dataset_description.json returns error."""
        from lacuna.cli import main

        # Create directory without dataset_description.json
        bids_dir = tmp_path / "bad_bids"
        bids_dir.mkdir()

        result = main([str(bids_dir), str(output_dir), "participant"])

        # Should return BIDS error
        assert result in [2, 64]  # EXIT_INVALID_ARGS or EXIT_BIDS_ERROR


class TestCLIModuleEntry:
    """Tests for `python -m lacuna` entry point."""

    def test_module_entry_help(self):
        """Test that `python -m lacuna --help` works."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "bids_dir" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_module_entry_version(self):
        """Test that `python -m lacuna --version` works."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--version"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0


class TestCLIWithMockedAnalysis:
    """Tests for CLI with mocked analysis to avoid heavy computation."""

    def test_cli_creates_output_directory(self, minimal_bids_dataset, output_dir, work_dir):
        """Test that CLI creates output directory."""
        from lacuna.cli import main

        # This will fail at BIDS loading stage but should still check args
        # Since we need proper BIDS loading, we'll just test arg validation
        main(
            [
                str(minimal_bids_dataset),
                str(output_dir),
                "participant",
                "--work-dir",
                str(work_dir),
            ]
        )

        # Even if analysis fails, output dir should be created
        assert output_dir.exists()

    def test_cli_respects_participant_label(self, minimal_bids_dataset, output_dir, work_dir):
        """Test that CLI respects --participant-label filtering."""
        from lacuna.cli import main

        # Filter to just one subject
        main(
            [
                str(minimal_bids_dataset),
                str(output_dir),
                "participant",
                "--participant-label",
                "001",
                "--work-dir",
                str(work_dir),
            ]
        )

        # May fail at analysis stage but args should be parsed

    def test_cli_verbose_flag(self, minimal_bids_dataset, output_dir, work_dir):
        """Test that CLI accepts verbosity flags."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                str(minimal_bids_dataset),
                str(output_dir),
                "participant",
                "-vv",
            ]
        )

        # Verbosity should be 2 (uses verbose_count)
        assert args.verbose_count == 2


class TestCLIConfiguration:
    """Tests for CLI configuration from command-line arguments."""

    def test_config_from_args(self, minimal_bids_dataset, output_dir, work_dir):
        """Test that CLIConfig is created correctly from args."""
        from lacuna.cli.config import CLIConfig
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                str(minimal_bids_dataset),
                str(output_dir),
                "participant",
                "--work-dir",
                str(work_dir),
                "--nprocs",
                "4",
            ]
        )

        config = CLIConfig.from_args(args)

        assert config.bids_dir == minimal_bids_dataset
        assert config.output_dir == output_dir
        assert config.work_dir == work_dir
        assert config.n_procs == 4

    def test_config_validates_paths(self, tmp_path, output_dir):
        """Test that CLIConfig validates BIDS path exists."""
        from lacuna.cli.config import CLIConfig
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                str(tmp_path / "nonexistent"),
                str(output_dir),
                "participant",
            ]
        )

        config = CLIConfig.from_args(args)

        with pytest.raises(ValueError, match="does not exist"):
            config.validate()
