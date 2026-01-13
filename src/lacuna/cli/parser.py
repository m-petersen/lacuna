"""
Lacuna CLI argument parser module.

This module provides the argument parser for the Lacuna CLI, following
the BIDS-Apps specification for neuroimaging pipelines.

Functions:
    build_parser: Build and return the argument parser.
"""

from __future__ import annotations

import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path


def _drop_sub(value: str) -> str:
    """Remove 'sub-' prefix from subject ID if present."""
    return value.removeprefix("sub-")


def build_parser(prog: str | None = None) -> ArgumentParser:
    """
    Build the CLI argument parser.

    Parameters
    ----------
    prog : str, optional
        Program name for help text. Defaults to 'lacuna'.

    Returns
    -------
    ArgumentParser
        Configured argument parser following BIDS-Apps specification.
    """
    from lacuna import __version__

    parser = ArgumentParser(
        prog=prog or "lacuna",
        description=f"Lacuna: Lesion Network Mapping Analysis v{__version__}",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # BIDS-Apps Required Positional Arguments
    parser.add_argument(
        "bids_dir",
        type=Path,
        help=(
            "Root folder of BIDS dataset (sub-XXXXX folders at top level), "
            "OR path to a single NIfTI mask file for quick analysis"
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for derivatives",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        help=(
            'Processing level: "participant" runs per-subject analysis, '
            '"group" aggregates subject-level parcelstats into group TSV files'
        ),
    )

    # Configuration file
    g_config = parser.add_argument_group("Configuration")
    g_config.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="YAML",
        help=(
            "Path to YAML configuration file. Use 'lacuna --generate-config' "
            "to create a template. Command-line options override config file."
        ),
    )
    g_config.add_argument(
        "--generate-config",
        action="store_true",
        help="Print a template configuration file to stdout and exit",
    )

    # BIDS Filtering Options
    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        nargs="+",
        type=_drop_sub,
        metavar="LABEL",
        help="Subject IDs to process (without sub- prefix)",
    )
    g_bids.add_argument(
        "--session-id",
        "--session_id",
        nargs="+",
        metavar="SESSION",
        help="Session IDs to process (without ses- prefix)",
    )
    g_bids.add_argument(
        "--pattern",
        type=str,
        metavar="GLOB",
        help="Glob pattern to filter mask files (e.g., '*label-WMH*')",
    )
    # Mask Space Options
    g_space = parser.add_argument_group("Mask space options")
    g_space.add_argument(
        "--mask-space",
        type=str,
        metavar="SPACE",
        help=(
            "Coordinate space of input masks (e.g., 'MNI152NLin6Asym'). "
            "Required if not detectable from filename or sidecar JSON. "
            "Resolution is auto-detected from voxel size."
        ),
    )

    # Analysis Options
    g_analysis = parser.add_argument_group("Analysis options")
    g_analysis.add_argument(
        "--functional-connectome",
        type=Path,
        metavar="PATH",
        help=(
            "Path to functional connectome directory or HDF5 file. "
            "Enables FunctionalNetworkMapping analysis."
        ),
    )
    g_analysis.add_argument(
        "--structural-tractogram",
        type=Path,
        metavar="PATH",
        help=(
            "Path to whole-brain tractogram (.tck). "
            "Enables StructuralNetworkMapping analysis. Requires MRtrix3."
        ),
    )
    g_analysis.add_argument(
        "--structural-tdi",
        type=Path,
        metavar="PATH",
        help="Path to pre-computed whole-brain TDI NIfTI (optional, speeds up processing)",
    )
    g_analysis.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help=(
            "Atlas names for RegionalDamage analysis. "
            "Use 'lacuna list-parcellations' to see available atlases."
        ),
    )
    g_analysis.add_argument(
        "--custom-parcellation",
        nargs=2,
        action="append",
        metavar=("NIFTI", "LABELS"),
        help=(
            "Custom parcellation: NIfTI file path and labels file path. "
            "Can be specified multiple times. Space/resolution auto-detected. "
            "Example: --custom-parcellation /path/atlas.nii.gz /path/labels.txt"
        ),
    )
    g_analysis.add_argument(
        "--skip-regional-damage",
        action="store_true",
        help="Skip RegionalDamage analysis (enabled by default)",
    )

    # Performance Options
    g_perf = parser.add_argument_group("Performance options")
    g_perf.add_argument(
        "--nprocs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel processes (-1 for all CPUs)",
    )
    g_perf.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of subjects to process together in vectorized mode. "
            "Higher values improve FNM performance but use more memory. "
            "Use -1 for all subjects at once. (default: 1 = sequential)"
        ),
    )
    g_perf.add_argument(
        "-w",
        "--tmp-dir",
        dest="tmp_dir",
        type=Path,
        default=Path(os.getenv("LACUNA_TMP_DIR", "tmp")),
        metavar="PATH",
        help="Temporary directory for intermediate files (default: $LACUNA_TMP_DIR or ./tmp)",
    )

    # Other Options
    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    g_other.add_argument(
        "--version",
        action="version",
        version=f"lacuna {__version__}",
    )
    g_other.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    )

    return parser


def build_fetch_parser(subparsers) -> None:
    """
    Add the fetch subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add fetch parser to.
    """
    from argparse import RawDescriptionHelpFormatter

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download and setup connectomes for analysis",
        description=(
            "Download, process, and register connectomes for lesion network mapping.\n\n"
            "Available connectomes:\n"
            "  gsp1000  - GSP1000 functional connectome (~100GB, requires Dataverse API key)\n"
            "  dtor985  - dTOR985 structural tractogram (~10GB, no authentication needed)\n\n"
            "Examples:\n"
            "  lacuna fetch gsp1000 --api-key $DATAVERSE_API_KEY --batches 50\n"
            "  lacuna fetch dtor985 --output-dir /data/connectomes\n"
            "  lacuna fetch --list"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Positional argument for connectome name
    fetch_parser.add_argument(
        "connectome",
        nargs="?",
        choices=["gsp1000", "dtor985"],
        help="Connectome to fetch",
    )

    # List flag
    fetch_parser.add_argument(
        "--list",
        action="store_true",
        help="List available connectomes",
    )

    # Output options
    fetch_parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="PATH",
        help="Output directory for processed files (default: ~/.cache/lacuna/connectomes/<name>)",
    )

    # GSP1000-specific options
    g_gsp = fetch_parser.add_argument_group("GSP1000 options")
    g_gsp.add_argument(
        "--api-key",
        type=str,
        metavar="KEY",
        help="Harvard Dataverse API key (or set DATAVERSE_API_KEY env var)",
    )
    g_gsp.add_argument(
        "--batches",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of HDF5 batch files to create. More batches = lower RAM usage.\n"
            "Recommendations: 16GB → 100, 32GB+ → 50. Ignored in test mode."
        ),
    )
    g_gsp.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            "Download only 1 tarball (~2GB) to test the full pipeline.\n"
            "Verifies download, extraction, conversion, and registration work."
        ),
    )

    # dTOR985-specific options
    g_dtor = fetch_parser.add_argument_group("dTOR985 options")
    g_dtor.add_argument(
        "--no-keep-original",
        action="store_true",
        help="Remove original .trk file after conversion to save disk space",
    )

    # Common options
    g_common = fetch_parser.add_argument_group("Common options")
    g_common.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    g_common.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive guided setup wizard",
    )
    g_common.add_argument(
        "--clean",
        action="store_true",
        help="Remove cached data for a specific connectome",
    )
    g_common.add_argument(
        "--clean-all",
        action="store_true",
        help="Remove all cached connectome data",
    )


def build_main_parser(prog: str | None = None) -> ArgumentParser:
    """
    Build the main CLI parser with subcommands.

    This creates a parser that supports both the main analysis workflow
    and utility subcommands like 'fetch'.

    Parameters
    ----------
    prog : str, optional
        Program name for help text. Defaults to 'lacuna'.

    Returns
    -------
    ArgumentParser
        Configured argument parser with subcommands.
    """
    from lacuna import __version__

    # Create main parser
    parser = ArgumentParser(
        prog=prog or "lacuna",
        description=f"Lacuna: Lesion Network Mapping Analysis v{__version__}",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"lacuna {__version__}",
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Use 'lacuna <command> --help' for more information on a command.",
    )

    # Add fetch subcommand
    build_fetch_parser(subparsers)

    # Add run subcommand (main analysis - makes BIDS args available)
    run_parser = subparsers.add_parser(
        "run",
        help="Run lesion network mapping analysis (BIDS-Apps workflow)",
        description="Run the main lesion network mapping analysis pipeline.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    _add_bids_arguments(run_parser)

    return parser


def _add_bids_arguments(parser: ArgumentParser) -> None:
    """Add BIDS-Apps arguments to a parser."""

    # BIDS-Apps Required Positional Arguments
    parser.add_argument(
        "bids_dir",
        type=Path,
        help=(
            "Root folder of BIDS dataset (sub-XXXXX folders at top level), "
            "OR path to a single NIfTI mask file for quick analysis"
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for derivatives",
    )
    parser.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        help=(
            'Processing level: "participant" runs per-subject analysis, '
            '"group" aggregates subject-level parcelstats into group TSV files'
        ),
    )

    # Configuration file
    g_config = parser.add_argument_group("Configuration")
    g_config.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="YAML",
        help=(
            "Path to YAML configuration file. Use 'lacuna --generate-config' "
            "to create a template. Command-line options override config file."
        ),
    )
    g_config.add_argument(
        "--generate-config",
        action="store_true",
        help="Print a template configuration file to stdout and exit",
    )

    # BIDS Filtering Options
    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        nargs="+",
        type=_drop_sub,
        metavar="LABEL",
        help="Subject IDs to process (without sub- prefix)",
    )
    g_bids.add_argument(
        "--session-id",
        "--session_id",
        nargs="+",
        metavar="SESSION",
        help="Session IDs to process (without ses- prefix)",
    )
    g_bids.add_argument(
        "--pattern",
        type=str,
        metavar="GLOB",
        help="Glob pattern to filter mask files (e.g., '*label-WMH*')",
    )
    # Mask Space Options
    g_space = parser.add_argument_group("Mask space options")
    g_space.add_argument(
        "--mask-space",
        type=str,
        metavar="SPACE",
        help=(
            "Coordinate space of input masks (e.g., 'MNI152NLin6Asym'). "
            "Required if not detectable from filename or sidecar JSON. "
            "Resolution is auto-detected from voxel size."
        ),
    )

    # Analysis Options
    g_analysis = parser.add_argument_group("Analysis options")
    g_analysis.add_argument(
        "--functional-connectome",
        type=Path,
        metavar="PATH",
        help=(
            "Path to functional connectome directory or HDF5 file. "
            "Enables FunctionalNetworkMapping analysis."
        ),
    )
    g_analysis.add_argument(
        "--structural-tractogram",
        type=Path,
        metavar="PATH",
        help=(
            "Path to whole-brain tractogram (.tck). "
            "Enables StructuralNetworkMapping analysis. Requires MRtrix3."
        ),
    )
    g_analysis.add_argument(
        "--structural-tdi",
        type=Path,
        metavar="PATH",
        help="Path to pre-computed whole-brain TDI NIfTI (optional, speeds up processing)",
    )
    g_analysis.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help=(
            "Atlas names for RegionalDamage analysis. "
            "Use 'lacuna list-parcellations' to see available atlases."
        ),
    )
    g_analysis.add_argument(
        "--custom-parcellation",
        nargs=2,
        action="append",
        metavar=("NIFTI", "LABELS"),
        help=(
            "Custom parcellation: NIfTI file path and labels file path. "
            "Can be specified multiple times. Space/resolution auto-detected. "
            "Example: --custom-parcellation /path/atlas.nii.gz /path/labels.txt"
        ),
    )
    g_analysis.add_argument(
        "--skip-regional-damage",
        action="store_true",
        help="Skip RegionalDamage analysis (enabled by default)",
    )

    # Performance Options
    g_perf = parser.add_argument_group("Performance options")
    g_perf.add_argument(
        "--nprocs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel processes (-1 for all CPUs)",
    )
    g_perf.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of subjects to process together in vectorized mode. "
            "Higher values improve FNM performance but use more memory. "
            "Use -1 for all subjects at once. (default: 1 = sequential)"
        ),
    )
    g_perf.add_argument(
        "-w",
        "--tmp-dir",
        dest="tmp_dir",
        type=Path,
        default=Path(os.getenv("LACUNA_TMP_DIR", "tmp")),
        metavar="PATH",
        help="Temporary directory for intermediate files (default: $LACUNA_TMP_DIR or ./tmp)",
    )

    # Other Options
    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    g_other.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    )
