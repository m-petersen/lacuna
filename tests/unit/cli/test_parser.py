"""Unit tests for CLI parser.

Tests the parser module for the new subcommand-based CLI structure.
"""

from __future__ import annotations

import argparse

import pytest

from lacuna.cli.parser import build_parser


class TestBuildParser:
    """Tests for the build_parser function."""

    def test_build_parser_returns_argument_parser(self):
        """Test that build_parser returns an ArgumentParser."""
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_custom_prog_name(self):
        """Test parser with custom prog name."""
        parser = build_parser(prog="custom_prog")
        assert parser.prog == "custom_prog"


class TestParserHelpAndVersion:
    """Tests for help and version flags."""

    def test_parser_shows_help(self, capsys):
        """Test that parser shows help on -h."""
        parser = build_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-h"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Lacuna" in captured.out
        # New structure shows subcommands, not bids_dir
        assert "fetch" in captured.out
        assert "run" in captured.out
        assert "collect" in captured.out
        assert "info" in captured.out

    def test_parser_shows_version(self, capsys):
        """Test that parser shows version on --version."""
        parser = build_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "lacuna" in captured.out.lower()


class TestSubcommandParsing:
    """Tests for subcommand parsing."""

    @pytest.mark.parametrize("command", ["fetch", "run", "collect", "info"])
    def test_subcommand_help_exits_zero(self, command, capsys):
        """Test that each subcommand shows help."""
        parser = build_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([command, "--help"])

        assert exc_info.value.code == 0


class TestRunSubcommandParsing:
    """Tests for 'lacuna run' subcommand parsing."""

    def test_run_rd_subcommand_parsing(self, tmp_path):
        """Test parsing 'run rd' subcommand."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["run", "rd", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100"]
        )

        assert args.command == "run"
        assert args.analysis == "rd"
        assert args.bids_dir == bids_dir
        assert args.output_dir == output_dir
        assert args.parcel_atlases == ["Schaefer100"]

    def test_run_fnm_subcommand_parsing(self, tmp_path):
        """Test parsing 'run fnm' subcommand."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        conn_path = tmp_path / "gsp1000.h5"

        args = parser.parse_args(
            ["run", "fnm", str(bids_dir), str(output_dir), "--connectome-path", str(conn_path)]
        )

        assert args.command == "run"
        assert args.analysis == "fnm"

    def test_run_snm_subcommand_parsing(self, tmp_path):
        """Test parsing 'run snm' subcommand."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        conn_path = tmp_path / "dtor985.tck"

        args = parser.parse_args(
            ["run", "snm", str(bids_dir), str(output_dir), "--connectome-path", str(conn_path)]
        )

        assert args.command == "run"
        assert args.analysis == "snm"

    def test_run_with_participant_label(self, tmp_path):
        """Test run with --participant-label option."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            [
                "run",
                "rd",
                str(bids_dir),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "--participant-label",
                "001",
                "002",
            ]
        )

        assert args.participant_label == ["001", "002"]

    def test_run_with_parcel_atlases(self, tmp_path):
        """Test run with --parcel-atlases option."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            [
                "run",
                "rd",
                str(bids_dir),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "Schaefer200",
            ]
        )

        assert args.parcel_atlases == ["Schaefer100", "Schaefer200"]

    def test_run_with_nprocs(self, tmp_path):
        """Test run with --nprocs option."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["run", "rd", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100", "--nprocs", "8"]
        )

        assert args.nprocs == 8

    def test_run_with_tmp_dir(self, tmp_path):
        """Test run with --tmp-dir option."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        tmp_dir = tmp_path / "tmp"

        args = parser.parse_args(
            [
                "run",
                "rd",
                str(bids_dir),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "--tmp-dir",
                str(tmp_dir),
            ]
        )

        assert args.tmp_dir == tmp_dir

    def test_run_with_verbose(self, tmp_path):
        """Test run with verbose flags."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["run", "rd", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100", "-v"]
        )

        assert args.verbose_count == 1

    def test_run_with_multiple_verbose(self, tmp_path):
        """Test run with multiple verbose flags."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["run", "rd", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100", "-vv"]
        )

        assert args.verbose_count == 2

    def test_run_default_values(self, tmp_path):
        """Test run default values."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["run", "rd", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100"]
        )

        assert args.participant_label is None
        assert args.nprocs == -1  # Default to all CPUs
        assert args.verbose_count == 0


class TestRunAliases:
    """Tests for analysis type aliases."""

    def test_regionaldamage_alias(self, tmp_path):
        """Test that 'regionaldamage' alias works."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["run", "regionaldamage", str(bids_dir), str(output_dir), "--parcel-atlases", "Schaefer100"]
        )

        # Argparse uses the subcommand name as-is, alias maps to full name
        assert args.analysis == "regionaldamage"

    def test_functionalnetworkmapping_alias(self, tmp_path):
        """Test that 'functionalnetworkmapping' alias works."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        conn_path = tmp_path / "gsp1000.h5"

        args = parser.parse_args(
            [
                "run",
                "functionalnetworkmapping",
                str(bids_dir),
                str(output_dir),
                "--connectome-path",
                str(conn_path),
            ]
        )

        assert args.analysis == "functionalnetworkmapping"

    def test_structuralnetworkmapping_alias(self, tmp_path):
        """Test that 'structuralnetworkmapping' alias works."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        conn_path = tmp_path / "dtor985.tck"

        args = parser.parse_args(
            [
                "run",
                "structuralnetworkmapping",
                str(bids_dir),
                str(output_dir),
                "--connectome-path",
                str(conn_path),
            ]
        )

        assert args.analysis == "structuralnetworkmapping"


class TestCollectSubcommand:
    """Tests for 'lacuna collect' subcommand."""

    def test_collect_parsing(self, tmp_path):
        """Test parsing collect subcommand."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(["collect", str(bids_dir), str(output_dir)])

        assert args.command == "collect"
        assert args.bids_dir == bids_dir
        assert args.output_dir == output_dir

    def test_collect_with_pattern_filter(self, tmp_path):
        """Test collect with --pattern option."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        args = parser.parse_args(
            ["collect", str(bids_dir), str(output_dir), "--pattern", "*lesion*"]
        )

        assert args.pattern == "*regionaldamage*lesion*"


class TestInfoSubcommand:
    """Tests for 'lacuna info' subcommand."""

    def test_info_atlases_parsing(self):
        """Test parsing 'info atlases'."""
        parser = build_parser()

        args = parser.parse_args(["info", "atlases"])

        assert args.command == "info"
        assert args.topic == "atlases"

    def test_info_connectomes_parsing(self):
        """Test parsing 'info connectomes'."""
        parser = build_parser()

        args = parser.parse_args(["info", "connectomes"])

        assert args.command == "info"
        assert args.topic == "connectomes"


class TestComplexScenarios:
    """Tests for complex command line scenarios."""

    def test_full_run_command_line(self, tmp_path):
        """Test parsing a full run command with all options."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        tmp_dir = tmp_path / "tmp"

        conn_path = tmp_path / "gsp1000.h5"
        args = parser.parse_args(
            [
                "run",
                "fnm",
                str(bids_dir),
                str(output_dir),
                "--participant-label",
                "001",
                "002",
                "--connectome-path",
                str(conn_path),
                "--parcel-atlases",
                "Schaefer100",
                "Schaefer200",
                "--nprocs",
                "4",
                "--tmp-dir",
                str(tmp_dir),
                "-vv",
            ]
        )

        assert args.command == "run"
        assert args.analysis == "fnm"
        assert args.bids_dir == bids_dir
        assert args.output_dir == output_dir
        assert args.participant_label == ["001", "002"]
        assert args.connectome_path == conn_path
        assert args.parcel_atlases == ["Schaefer100", "Schaefer200"]
        assert args.nprocs == 4
        assert args.tmp_dir == tmp_dir
        assert args.verbose_count == 2

    def test_short_tmp_dir_option(self, tmp_path):
        """Test that -w works as short form for --tmp-dir."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        tmp_dir = tmp_path / "tmp"

        args = parser.parse_args(
            [
                "run",
                "rd",
                str(bids_dir),
                str(output_dir),
                "--parcel-atlases",
                "Schaefer100",
                "-w",
                str(tmp_dir),
            ]
        )

        assert args.tmp_dir == tmp_dir


class TestConnectomePathFlag:
    """Tests for --connectome-path required flag for FNM/SNM."""

    def test_fnm_connectome_path_flag(self, tmp_path):
        """Test FNM with --connectome-path."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        conn_path = tmp_path / "connectome.h5"

        args = parser.parse_args(
            ["run", "fnm", str(bids_dir), str(output_dir), "--connectome-path", str(conn_path)]
        )

        assert args.connectome_path == conn_path

    def test_snm_connectome_path_flag(self, tmp_path):
        """Test SNM with --connectome-path."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"
        conn_path = tmp_path / "tractogram.tck"

        args = parser.parse_args(
            ["run", "snm", str(bids_dir), str(output_dir), "--connectome-path", str(conn_path)]
        )

        assert args.connectome_path == conn_path

    def test_fnm_requires_connectome_path(self, tmp_path, capsys):
        """Test that FNM requires --connectome-path."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["run", "fnm", str(bids_dir), str(output_dir)])

        # Should fail due to missing required argument
        assert exc_info.value.code != 0

    def test_snm_requires_connectome_path(self, tmp_path, capsys):
        """Test that SNM requires --connectome-path."""
        parser = build_parser()
        bids_dir = tmp_path / "bids"
        output_dir = tmp_path / "output"

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["run", "snm", str(bids_dir), str(output_dir)])

        # Should fail due to missing required argument
        assert exc_info.value.code != 0
