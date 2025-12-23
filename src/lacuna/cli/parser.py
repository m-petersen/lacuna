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

    # Space/Resolution Options (required when not in filename)
    g_space = parser.add_argument_group("Space and resolution options")
    g_space.add_argument(
        "--space",
        type=str,
        metavar="SPACE",
        help="Coordinate space (e.g., 'MNI152NLin6Asym'). Required if not in filename.",
    )
    g_space.add_argument(
        "--resolution",
        type=float,
        metavar="MM",
        help="Voxel resolution in mm (e.g., 2.0). Required if not in filename.",
    )

    # Analysis Options
    g_analysis = parser.add_argument_group("Analysis options")
    g_analysis.add_argument(
        "--functional-connectome",
        type=str,
        metavar="NAME_OR_PATH",
        help=(
            "Functional connectome name (from registry) or path to HDF5/directory. "
            "Enables FunctionalNetworkMapping analysis."
        ),
    )
    g_analysis.add_argument(
        "--structural-connectome",
        type=str,
        metavar="NAME_OR_PATH",
        help=(
            "Structural connectome name (from registry) or path to tractogram (.tck). "
            "Enables StructuralNetworkMapping analysis."
        ),
    )
    g_analysis.add_argument(
        "--structural-tdi",
        type=Path,
        metavar="PATH",
        help="Path to whole-brain TDI NIfTI (required with --structural-connectome path)",
    )
    g_analysis.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help=(
            "Atlas names for RegionalDamage analysis (from registry). "
            "If not specified, uses default atlas."
        ),
    )
    g_analysis.add_argument(
        "--skip-regional-damage",
        action="store_true",
        help="Skip RegionalDamage analysis (enabled by default)",
    )
    g_analysis.add_argument(
        "--atlas-dir",
        type=Path,
        metavar="PATH",
        help="Additional directory containing atlas files",
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
