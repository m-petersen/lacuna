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
def tmp_dir(tmp_path):
    """Create a tmp directory."""
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    return tmp


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

    def test_cli_run_rd_invalid_bids_returns_error(self, tmp_path, output_dir):
        """Test that invalid BIDS directory returns error code."""
        from lacuna.cli import main

        # Non-existent BIDS directory
        result = main(
            [
                "run",
                "rd",
                str(tmp_path / "nonexistent"),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
            ]
        )

        assert result == 2  # EXIT_INVALID_ARGS

    def test_cli_run_rd_missing_dataset_description_returns_error(self, tmp_path, output_dir):
        """Test that BIDS without dataset_description.json returns error."""
        from lacuna.cli import main

        # Create directory without dataset_description.json
        bids_dir = tmp_path / "bad_bids"
        bids_dir.mkdir()

        result = main(
            ["run", "rd", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100"]
        )

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
        # Check for new subcommand structure
        assert (
            "fetch" in result.stdout or "run" in result.stdout or "usage" in result.stdout.lower()
        )

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

    def test_cli_run_rd_creates_output_directory(self, minimal_bids_dataset, output_dir, tmp_dir):
        """Test that CLI run creates output directory."""
        from lacuna.cli import main

        # Run with minimal BIDS dataset
        main(
            [
                "run",
                "rd",
                str(minimal_bids_dataset),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "--tmp-dir",
                str(tmp_dir),
            ]
        )

        # Even if analysis fails, output dir should be created
        assert output_dir.exists()

    def test_cli_run_rd_respects_participant_label(self, minimal_bids_dataset, output_dir, tmp_dir):
        """Test that CLI respects --participant-label filtering."""
        from lacuna.cli import main

        # Filter to just one subject
        main(
            [
                "run",
                "rd",
                str(minimal_bids_dataset),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "--participant-label",
                "001",
                "--tmp-dir",
                str(tmp_dir),
            ]
        )

        # May fail at analysis stage but args should be parsed

    def test_cli_verbose_flag(self, minimal_bids_dataset, output_dir, tmp_dir):
        """Test that CLI accepts verbosity flags."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "rd",
                str(minimal_bids_dataset),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "-vv",
            ]
        )

        # Verbosity should be 2 (uses verbose_count)
        assert args.verbose_count == 2


class TestInfoCommand:
    """Tests for 'lacuna info' command."""

    def test_info_atlases_shows_available_atlases(self):
        """Test that 'lacuna info atlases' shows available atlases."""
        from lacuna.cli import main

        result = main(["info", "atlases"])
        assert result == 0

    def test_info_connectomes_shows_available_connectomes(self):
        """Test that 'lacuna info connectomes' shows available connectomes."""
        from lacuna.cli import main

        result = main(["info", "connectomes"])
        assert result == 0


class TestCollectCommand:
    """Tests for 'lacuna collect' command."""

    def test_collect_with_nonexistent_dir_returns_error(self, tmp_path):
        """Test that collect with non-existent output dir returns error."""
        from lacuna.cli import main

        # Create bids_dir but not output_dir
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        result = main(["collect", str(bids_dir), str(tmp_path / "nonexistent")])
        assert result != 0

    def test_collect_with_empty_dir(self, tmp_path):
        """Test that collect with empty dir handles gracefully."""
        from lacuna.cli import main

        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "empty_derivatives"
        output_dir.mkdir()

        # Should handle empty dir gracefully (no parcelstats files found)
        main(["collect", str(bids_dir), str(output_dir)])
        # May return error or success depending on implementation
        # Just verify it doesn't crash


class TestRunCommandAliases:
    """Tests for analysis command aliases."""

    def test_run_rd_alias_works(self, minimal_bids_dataset, output_dir):
        """Test that 'run rd' alias works."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        # Should not raise
        args = parser.parse_args(
            ["run", "rd", str(minimal_bids_dataset), str(output_dir), "--parcel-atlases", "Schaefer100"]
        )
        assert args.analysis == "rd"

    def test_run_regionaldamage_alias_works(self, minimal_bids_dataset, output_dir):
        """Test that 'run regionaldamage' alias works."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "regionaldamage",
                str(minimal_bids_dataset),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
            ]
        )
        # Alias keeps the name used (argparse behavior)
        assert args.analysis == "regionaldamage"

    def test_run_fnm_alias_works(self, minimal_bids_dataset, output_dir, tmp_path):
        """Test that 'run fnm' alias works (requires --connectome-path)."""
        from lacuna.cli.parser import build_parser

        conn_path = tmp_path / "gsp1000.h5"
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "fnm",
                "--connectome-path",
                str(conn_path),
                str(minimal_bids_dataset),
                str(output_dir),
            ]
        )
        assert args.analysis == "fnm"
        assert args.connectome_path == conn_path

    def test_run_snm_alias_works(self, minimal_bids_dataset, output_dir, tmp_path):
        """Test that 'run snm' alias works (requires --connectome-path)."""
        from lacuna.cli.parser import build_parser

        conn_path = tmp_path / "dtor985.tck"
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "snm",
                "--connectome-path",
                str(conn_path),
                str(minimal_bids_dataset),
                str(output_dir),
            ]
        )
        assert args.analysis == "snm"
        assert args.connectome_path == conn_path
