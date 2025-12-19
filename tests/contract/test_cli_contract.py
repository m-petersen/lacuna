"""Contract tests for CLI interface.

These tests verify that the CLI interface follows the BIDS-Apps specification
and behaves correctly with various argument combinations.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestCLIContract:
    """Contract tests for CLI interface."""

    def test_cli_help_exits_zero(self):
        """Test that --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "lacuna" in result.stdout.lower()

    def test_cli_version_exits_zero(self):
        """Test that --version exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "lacuna" in result.stdout.lower()

    def test_cli_missing_args_exits_nonzero(self):
        """Test that missing required args exits with non-zero code."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna"],
            capture_output=True,
            text=True,
        )
        # argparse exits with code 2 for missing required args
        assert result.returncode == 2

    def test_cli_invalid_analysis_level_exits_nonzero(self):
        """Test that invalid analysis level exits with non-zero code."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "/data", "/output", "group"],
            capture_output=True,
            text=True,
        )
        # argparse exits with code 2 for invalid choices
        assert result.returncode == 2

    def test_cli_nonexistent_bids_dir_exits_nonzero(self, tmp_path):
        """Test that non-existent BIDS dir exits with code 2."""
        nonexistent = tmp_path / "nonexistent"
        output = tmp_path / "output"

        result = subprocess.run(
            [sys.executable, "-m", "lacuna", str(nonexistent), str(output), "participant"],
            capture_output=True,
            text=True,
        )
        # Exit code 2 for invalid arguments
        assert result.returncode == 2

    def test_cli_help_shows_bids_apps_args(self):
        """Test that help shows BIDS-Apps required arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Check for BIDS-Apps required arguments
        assert "bids_dir" in result.stdout
        assert "output_dir" in result.stdout
        assert "participant" in result.stdout

    def test_cli_help_shows_analysis_options(self):
        """Test that help shows analysis options."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Check for analysis options
        assert "--functional-connectome" in result.stdout
        assert "--structural-connectome" in result.stdout
        assert "--parcel-atlases" in result.stdout

    def test_cli_help_shows_performance_options(self):
        """Test that help shows performance options."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Check for performance options
        assert "--nprocs" in result.stdout
        assert "--work-dir" in result.stdout


class TestCLIMainFunction:
    """Tests for the main() function directly."""

    def test_main_with_help_returns_zero(self):
        """Test that main() with --help returns 0."""
        from lacuna.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_with_missing_args_returns_two(self, capsys):
        """Test that main() with missing args returns 2."""
        from lacuna.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    def test_main_with_nonexistent_bids_returns_two(self, tmp_path):
        """Test that main() with non-existent BIDS returns 2."""
        from lacuna.cli.main import main

        nonexistent = tmp_path / "nonexistent"
        output = tmp_path / "output"

        result = main([str(nonexistent), str(output), "participant"])
        assert result == 2


class TestExitCodes:
    """Tests for CLI exit codes."""

    def test_exit_codes_are_defined(self):
        """Test that exit codes are defined correctly."""
        from lacuna.cli.main import (
            EXIT_ANALYSIS_ERROR,
            EXIT_BIDS_ERROR,
            EXIT_GENERAL_ERROR,
            EXIT_INVALID_ARGS,
            EXIT_SUCCESS,
        )

        assert EXIT_SUCCESS == 0
        assert EXIT_GENERAL_ERROR == 1
        assert EXIT_INVALID_ARGS == 2
        assert EXIT_BIDS_ERROR == 64
        assert EXIT_ANALYSIS_ERROR == 65
