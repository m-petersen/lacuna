"""Unit tests for CLI argument parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from lacuna.cli.parser import build_parser


class TestBuildParser:
    """Tests for build_parser function."""

    def test_build_parser_returns_argument_parser(self):
        """Test that build_parser returns an ArgumentParser."""
        parser = build_parser()
        assert parser is not None
        assert hasattr(parser, "parse_args")

    def test_parser_accepts_required_positional_args(self, tmp_path):
        """Test that parser accepts required positional arguments."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant"])

        assert args.bids_dir == Path("/data/bids")
        assert args.output_dir == Path("/output")
        assert args.analysis_level == "participant"

    def test_parser_rejects_invalid_analysis_level(self, tmp_path):
        """Test that parser rejects invalid analysis levels."""
        parser = build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["/data/bids", "/output", "group"])

    def test_parser_accepts_participant_label(self):
        """Test that parser accepts --participant-label."""
        parser = build_parser()

        args = parser.parse_args(
            ["/data/bids", "/output", "participant", "--participant-label", "001", "002", "003"]
        )

        assert args.participant_label == ["001", "002", "003"]

    def test_parser_strips_sub_prefix(self):
        """Test that parser strips 'sub-' prefix from participant labels."""
        parser = build_parser()

        args = parser.parse_args(
            [
                "/data/bids",
                "/output",
                "participant",
                "--participant-label",
                "sub-001",
                "002",
                "sub-003",
            ]
        )

        assert args.participant_label == ["001", "002", "003"]

    def test_parser_accepts_functional_connectome(self):
        """Test that parser accepts --functional-connectome."""
        parser = build_parser()

        args = parser.parse_args(
            [
                "/data/bids",
                "/output",
                "participant",
                "--functional-connectome",
                "/path/to/connectome.h5",
            ]
        )

        assert args.functional_connectome == Path("/path/to/connectome.h5")

    def test_parser_accepts_structural_connectome(self):
        """Test that parser accepts --structural-connectome."""
        parser = build_parser()

        args = parser.parse_args(
            [
                "/data/bids",
                "/output",
                "participant",
                "--structural-connectome",
                "/path/to/tractogram.tck",
                "--structural-tdi",
                "/path/to/tdi.nii.gz",
            ]
        )

        assert args.structural_connectome == Path("/path/to/tractogram.tck")
        assert args.structural_tdi == Path("/path/to/tdi.nii.gz")

    def test_parser_accepts_parcel_atlases(self):
        """Test that parser accepts --parcel-atlases."""
        parser = build_parser()

        args = parser.parse_args(
            [
                "/data/bids",
                "/output",
                "participant",
                "--parcel-atlases",
                "Schaefer100",
                "Schaefer200",
            ]
        )

        assert args.parcel_atlases == ["Schaefer100", "Schaefer200"]

    def test_parser_accepts_nprocs(self):
        """Test that parser accepts --nprocs."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant", "--nprocs", "8"])

        assert args.nprocs == 8

    def test_parser_accepts_work_dir(self):
        """Test that parser accepts --work-dir."""
        parser = build_parser()

        args = parser.parse_args(
            ["/data/bids", "/output", "participant", "--work-dir", "/scratch/work"]
        )

        assert args.work_dir == Path("/scratch/work")

    def test_parser_accepts_short_work_dir(self):
        """Test that parser accepts -w for work-dir."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant", "-w", "/scratch/work"])

        assert args.work_dir == Path("/scratch/work")

    def test_parser_accepts_verbose(self):
        """Test that parser accepts -v for verbosity."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant", "-v"])

        assert args.verbose_count == 1

    def test_parser_accepts_multiple_verbose(self):
        """Test that parser accepts multiple -v flags."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant", "-vv"])

        assert args.verbose_count == 2

    def test_parser_accepts_skip_bids_validation(self):
        """Test that parser accepts --skip-bids-validation."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant", "--skip-bids-validation"])

        assert args.skip_bids_validation is True

    def test_parser_default_values(self):
        """Test parser default values."""
        parser = build_parser()

        args = parser.parse_args(["/data/bids", "/output", "participant"])

        assert args.participant_label is None
        assert args.skip_bids_validation is False
        assert args.functional_connectome is None
        assert args.structural_connectome is None
        assert args.parcel_atlases is None
        assert args.nprocs == 1
        assert args.verbose_count == 0

    def test_parser_custom_prog_name(self):
        """Test that parser accepts custom program name."""
        parser = build_parser(prog="my-lacuna")

        assert parser.prog == "my-lacuna"


class TestParserHelpAndVersion:
    """Tests for help and version output."""

    def test_parser_shows_help(self, capsys):
        """Test that parser shows help on -h."""
        parser = build_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-h"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Lacuna" in captured.out
        assert "bids_dir" in captured.out
        assert "output_dir" in captured.out

    def test_parser_shows_version(self, capsys):
        """Test that parser shows version on --version."""
        parser = build_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "lacuna" in captured.out


class TestComplexScenarios:
    """Tests for complex CLI scenarios."""

    def test_full_command_line(self):
        """Test parsing a full command line with all options."""
        parser = build_parser()

        args = parser.parse_args(
            [
                "/data/bids",
                "/output",
                "participant",
                "--participant-label",
                "001",
                "002",
                "--functional-connectome",
                "/connectomes/gsp1000.h5",
                "--structural-connectome",
                "/connectomes/dtor985.tck",
                "--structural-tdi",
                "/connectomes/dtor985_tdi.nii.gz",
                "--parcel-atlases",
                "Schaefer100",
                "Schaefer200",
                "--nprocs",
                "4",
                "--work-dir",
                "/scratch/work",
                "--skip-bids-validation",
                "-vv",
            ]
        )

        assert args.bids_dir == Path("/data/bids")
        assert args.output_dir == Path("/output")
        assert args.analysis_level == "participant"
        assert args.participant_label == ["001", "002"]
        assert args.functional_connectome == Path("/connectomes/gsp1000.h5")
        assert args.structural_connectome == Path("/connectomes/dtor985.tck")
        assert args.structural_tdi == Path("/connectomes/dtor985_tdi.nii.gz")
        assert args.parcel_atlases == ["Schaefer100", "Schaefer200"]
        assert args.nprocs == 4
        assert args.work_dir == Path("/scratch/work")
        assert args.skip_bids_validation is True
        assert args.verbose_count == 2
