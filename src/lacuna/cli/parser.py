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
        choices=["participant"],
        help='Processing level (only "participant" supported)',
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
    g_bids.add_argument(
        "--skip-bids-validation",
        action="store_true",
        help="Skip BIDS dataset validation",
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
        "-w",
        "--work-dir",
        type=Path,
        default=Path(os.getenv("LACUNA_WORK_DIR", "work")),
        metavar="PATH",
        help="Working directory (default: $LACUNA_WORK_DIR or ./work)",
    )

    # Other Options
    g_other = parser.add_argument_group("Other options")
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
