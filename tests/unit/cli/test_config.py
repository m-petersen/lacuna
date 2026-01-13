"""Unit tests for CLI configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from lacuna.cli.config import CLIConfig


class TestCLIConfigBasic:
    """Tests for basic CLIConfig functionality."""

    def test_create_config_with_required_args(self, tmp_path):
        """Test creating config with required arguments."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis_level="participant",
        )

        assert config.bids_dir == bids_dir
        assert config.output_dir == output_dir
        assert config.analysis_level == "participant"

    def test_default_values(self, tmp_path):
        """Test that default values are set correctly."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis_level="participant",
        )

        assert config.participant_label is None
        assert config.session_id is None
        assert config.functional_connectome is None
        assert config.structural_connectome is None
        assert config.parcel_atlases is None
        assert config.n_procs == 1
        assert config.verbose_count == 0


class TestCLIConfigLogLevel:
    """Tests for log_level property."""

    @pytest.mark.parametrize(
        "verbose_count,expected_level",
        [
            (0, 25),  # WORKFLOW level
            (1, 20),  # INFO
            (2, 15),  # Between INFO and DEBUG
            (3, 10),  # DEBUG (minimum)
            (10, 10),  # Should cap at DEBUG
        ],
    )
    def test_log_level_from_verbose_count(self, tmp_path, verbose_count, expected_level):
        """Test that log_level is computed correctly from verbose_count."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            verbose_count=verbose_count,
        )

        assert config.log_level == expected_level


class TestCLIConfigFromArgs:
    """Tests for from_args class method."""

    def test_from_args_creates_config(self, tmp_path):
        """Test creating config from parsed arguments."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        args = argparse.Namespace(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis_level="participant",
            participant_label=["001", "002"],
            functional_connectome=None,
            structural_connectome=None,
            parcel_atlases=["Schaefer100"],
            nprocs=4,
            tmp_dir=Path("tmp"),
            verbose_count=1,
        )

        config = CLIConfig.from_args(args)

        assert config.bids_dir == bids_dir
        assert config.output_dir == output_dir
        assert config.participant_label == ["001", "002"]
        assert config.parcel_atlases == ["Schaefer100"]
        assert config.n_procs == 4
        assert config.verbose_count == 1


class TestCLIConfigValidation:
    """Tests for validate method."""

    def test_validate_passes_with_valid_config(self, tmp_path):
        """Test that validation passes with valid configuration."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        output_dir = tmp_path / "output"

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=output_dir,
            analysis_level="participant",
        )

        # Should not raise
        config.validate()

    def test_validate_fails_if_bids_dir_missing(self, tmp_path):
        """Test that validation fails if BIDS directory doesn't exist."""
        config = CLIConfig(
            bids_dir=tmp_path / "nonexistent",
            output_dir=tmp_path / "output",
            analysis_level="participant",
        )

        with pytest.raises(ValueError, match="Input path does not exist"):
            config.validate()

    def test_validate_fails_if_output_equals_bids(self, tmp_path):
        """Test that validation fails if output_dir equals bids_dir."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=bids_dir,  # Same as input
            analysis_level="participant",
        )

        with pytest.raises(ValueError, match="cannot be same as input"):
            config.validate()

    def test_validate_fails_if_functional_connectome_missing(self, tmp_path):
        """Test that validation fails if functional connectome doesn't exist."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            functional_connectome=str(tmp_path / "nonexistent.h5"),  # String path
        )

        with pytest.raises(ValueError, match="Functional connectome not found"):
            config.validate()

    def test_validate_fails_if_structural_connectome_missing(self, tmp_path):
        """Test that validation fails if structural tractogram doesn't exist."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            structural_connectome=str(tmp_path / "nonexistent.tck"),  # Non-existent path
        )

        with pytest.raises(ValueError, match="Structural tractogram not found"):
            config.validate()

    def test_validate_succeeds_with_valid_structural_connectome(self, tmp_path):
        """Test that validation succeeds with valid structural connectome."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        connectome = tmp_path / "connectome.tck"
        connectome.touch()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            structural_connectome=str(connectome),  # String path
        )

        # Should not raise
        config.validate()

    def test_validate_fails_with_invalid_nprocs(self, tmp_path):
        """Test that validation fails with n_procs < 1."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            n_procs=0,
        )

        with pytest.raises(ValueError, match="--nprocs must be -1 .* or >= 1"):
            config.validate()

    def test_validate_passes_with_functional_connectome(self, tmp_path):
        """Test that validation passes with existing functional connectome."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        connectome = tmp_path / "connectome.h5"
        connectome.touch()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            functional_connectome=str(connectome),  # String path
        )

        # Should not raise
        config.validate()

    def test_validate_passes_with_structural_connectome(self, tmp_path):
        """Test that validation passes with structural connectome."""
        bids_dir = tmp_path / "bids"
        bids_dir.mkdir()
        connectome = tmp_path / "connectome.tck"
        connectome.touch()

        config = CLIConfig(
            bids_dir=bids_dir,
            output_dir=tmp_path / "output",
            analysis_level="participant",
            structural_connectome=str(connectome),  # String path
        )

        # Should not raise
        config.validate()


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_tmp_dir_from_env_var(self, tmp_path, monkeypatch):
        """Test that tmp_dir defaults from LACUNA_TMP_DIR env var."""
        from lacuna.cli.parser import build_parser

        tmp_dir_path = tmp_path / "custom_tmp"
        monkeypatch.setenv("LACUNA_TMP_DIR", str(tmp_dir_path))

        parser = build_parser()
        args = parser.parse_args([str(tmp_path), str(tmp_path / "out"), "participant"])

        assert args.tmp_dir == tmp_dir_path

    def test_tmp_dir_cli_overrides_env_var(self, tmp_path, monkeypatch):
        """Test that --tmp-dir CLI flag overrides LACUNA_TMP_DIR."""
        from lacuna.cli.parser import build_parser

        monkeypatch.setenv("LACUNA_TMP_DIR", "/from/env")
        cli_tmp = tmp_path / "cli_tmp"

        parser = build_parser()
        args = parser.parse_args(
            [
                str(tmp_path),
                str(tmp_path / "out"),
                "participant",
                "--tmp-dir",
                str(cli_tmp),
            ]
        )

        assert args.tmp_dir == cli_tmp

    def test_tmp_dir_defaults_to_local_tmp(self, tmp_path, monkeypatch):
        """Test that tmp_dir defaults to './tmp' when no env var."""
        from lacuna.cli.parser import build_parser

        # Ensure env var is not set
        monkeypatch.delenv("LACUNA_TMP_DIR", raising=False)

        parser = build_parser()
        args = parser.parse_args([str(tmp_path), str(tmp_path / "out"), "participant"])

        assert args.tmp_dir == Path("tmp")
