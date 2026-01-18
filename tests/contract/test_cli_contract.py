"""Contract tests for CLI interface.

These tests verify that the CLI interface follows the expected subcommand
structure and behaves correctly with various argument combinations.

CLI Structure:
    lacuna                          # Main entry point
    ├── fetch                       # Download normative data
    ├── run                         # Run analyses
    │   ├── rd (regionaldamage)     # Regional damage analysis
    │   ├── fnm (functionalnetworkmapping)  # Functional network mapping
    │   └── snm (structuralnetworkmapping)  # Structural network mapping
    ├── collect                     # Aggregate results
    └── info                        # Show available resources
"""

from __future__ import annotations

import subprocess
import sys

import pytest


class TestCLIHelpContract:
    """Contract tests for CLI help system."""

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

    def test_cli_no_args_shows_help(self):
        """Test that running without args shows help (subcommand-based CLI)."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna"],
            capture_output=True,
            text=True,
        )
        # New CLI shows help and returns 0 when no subcommand given
        assert result.returncode == 0
        # Should show available commands
        assert "fetch" in result.stdout or "run" in result.stdout


class TestSubcommandStructure:
    """Contract tests for subcommand structure."""

    @pytest.mark.parametrize("command", ["fetch", "run", "collect", "info"])
    def test_subcommand_help_exits_zero(self, command):
        """Test that each subcommand has working --help."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", command, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert command in result.stdout.lower() or "usage" in result.stdout.lower()


class TestRunSubcommandContract:
    """Contract tests for 'lacuna run' subcommand."""

    def test_run_shows_help_without_analysis_type(self):
        """Test that 'lacuna run' without analysis type shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "run"],
            capture_output=True,
            text=True,
        )
        # Subcommand-based CLI shows help when subcommand is missing
        assert result.returncode == 0
        # Should show available analyses
        assert "rd" in result.stdout or "fnm" in result.stdout or "snm" in result.stdout

    @pytest.mark.parametrize(
        "short,full",
        [
            ("rd", "regionaldamage"),
            ("fnm", "functionalnetworkmapping"),
            ("snm", "structuralnetworkmapping"),
        ],
    )
    def test_run_analysis_aliases(self, short, full):
        """Test that analysis aliases have working --help."""
        # Short form
        result_short = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", short, "--help"],
            capture_output=True,
            text=True,
        )
        assert result_short.returncode == 0

        # Full form
        result_full = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", full, "--help"],
            capture_output=True,
            text=True,
        )
        assert result_full.returncode == 0

    def test_run_rd_help_shows_atlas_options(self):
        """Test that 'lacuna run rd --help' shows atlas options."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", "rd", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show parcel atlas option
        assert "parcel" in result.stdout.lower()

    def test_run_fnm_help_shows_connectome_options(self):
        """Test that 'lacuna run fnm --help' shows connectome options."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", "fnm", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show connectome option
        assert "connectome" in result.stdout.lower()

    def test_run_snm_help_shows_tractogram_options(self):
        """Test that 'lacuna run snm --help' shows tractogram options."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", "snm", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show tractogram option
        assert "tractogram" in result.stdout.lower() or "connectome" in result.stdout.lower()

    def test_run_requires_bids_dir(self, tmp_path):
        """Test that run requires a valid BIDS directory argument."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", "rd"],
            capture_output=True,
            text=True,
        )
        # Should fail because bids_dir is required
        assert result.returncode != 0

    def test_run_missing_bids_dir_exits_error(self, tmp_path):
        """Test that non-existent BIDS dir exits with error."""
        nonexistent = tmp_path / "nonexistent"
        output = tmp_path / "output"

        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "run", "rd", str(nonexistent), str(output)],
            capture_output=True,
            text=True,
        )
        # Exit code should be non-zero (invalid args)
        assert result.returncode != 0


class TestCollectSubcommandContract:
    """Contract tests for 'lacuna collect' subcommand."""

    def test_collect_help_exits_zero(self):
        """Test that collect --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "collect", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_collect_requires_derivatives_dir(self):
        """Test that collect requires a derivatives directory."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "collect"],
            capture_output=True,
            text=True,
        )
        # Should fail because derivatives_dir is required
        assert result.returncode != 0


class TestInfoSubcommandContract:
    """Contract tests for 'lacuna info' subcommand."""

    def test_info_help_exits_zero(self):
        """Test that info --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "info", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    @pytest.mark.parametrize("resource", ["atlases", "connectomes"])
    def test_info_resource_commands_exist(self, resource):
        """Test that info subcommands for resources exist."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "info", resource, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_info_atlases_runs_successfully(self):
        """Test that 'lacuna info atlases' runs and shows output."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "info", "atlases"],
            capture_output=True,
            text=True,
        )
        # Should exit successfully
        assert result.returncode == 0
        # Should show atlas info
        assert (
            "atlas" in result.stdout.lower()
            or "parcel" in result.stdout.lower()
            or "schaefer" in result.stdout.lower()
        )


class TestFetchSubcommandContract:
    """Contract tests for 'lacuna fetch' subcommand."""

    def test_fetch_help_exits_zero(self):
        """Test that fetch --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "lacuna", "fetch", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestCLIMainFunction:
    """Tests for the main() function directly."""

    def test_main_with_help_returns_zero(self):
        """Test that main() with --help returns 0."""
        from lacuna.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_with_no_args_shows_help(self, capsys):
        """Test that main() with no args shows help (subcommand-based CLI)."""
        from lacuna.cli.main import main

        result = main([])
        # New CLI shows help and returns 0 when no subcommand given
        assert result == 0

    def test_main_info_atlases_returns_zero(self):
        """Test that main() with 'info atlases' returns 0."""
        from lacuna.cli.main import main

        result = main(["info", "atlases"])
        assert result == 0


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
