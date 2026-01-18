"""
Lacuna CLI argument parser module.

This module provides the argument parser for the Lacuna CLI with a clean
subcommand-based structure:

- lacuna fetch: Download and setup connectomes
- lacuna run <analysis>: Run analyses (rd, fnm, snm)
- lacuna collect: Aggregate results across subjects
- lacuna info: Display available resources (atlases, connectomes)

Functions:
    build_parser: Build and return the main argument parser.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path


def _drop_sub(value: str) -> str:
    """Remove 'sub-' prefix from subject ID if present."""
    return value.removeprefix("sub-")


def build_parser(prog: str | None = None) -> ArgumentParser:
    """
    Build the main CLI parser with subcommands.

    Creates a parser that supports:
    - lacuna fetch: Download connectomes
    - lacuna run <analysis>: Run analyses
    - lacuna collect: Aggregate results
    - lacuna info: Show available resources

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
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  fetch    Download and setup connectomes for analysis\n"
            "  run      Run lesion network mapping analyses\n"
            "  collect  Aggregate results across subjects\n"
            "  info     Display available resources (atlases, connectomes)\n\n"
            "Examples:\n"
            "  lacuna fetch gsp1000 --api-key \\$DATAVERSE_API_KEY\n"
            "  lacuna run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks\n"
            "  lacuna run fnm /bids /output --connectome-path /path/to/gsp1000_batches\n"
            "  lacuna collect /bids /output\n"
            "  lacuna info atlases\n"
        ),
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
        description="Use 'lacuna <command> --help' for more information.",
        metavar="<command>",
    )

    # Add subcommands
    _build_fetch_parser(subparsers)
    _build_run_parser(subparsers)
    _build_collect_parser(subparsers)
    _build_info_parser(subparsers)

    return parser


def _build_fetch_parser(subparsers) -> None:
    """
    Add the fetch subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add fetch parser to.
    """
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download and setup connectomes for analysis",
        description=(
            "Download, process, and register connectomes for lesion network mapping.\n\n"
            "Available connectomes:\n"
            "  gsp1000  - GSP1000 functional connectome (~100GB, requires Dataverse API key)\n"
            "  dtor985  - dTOR985 structural tractogram (~10GB, requires Figshare API key)\n\n"
            "Examples:\n"
            "  lacuna fetch gsp1000 --api-key \\$DATAVERSE_API_KEY --batches 50\n"
            "  lacuna fetch dtor985 --api-key \\$FIGSHARE_API_KEY --output-dir /data/connectomes\n"
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

    # Common options
    g_common = fetch_parser.add_argument_group("Common options")
    g_common.add_argument(
        "--api-key",
        type=str,
        metavar="KEY",
        help=(
            "API key for authenticated downloads.\n"
            "For GSP1000: Dataverse API key (or set DATAVERSE_API_KEY env var)\n"
            "For dTOR985: Figshare API key (or set FIGSHARE_API_KEY env var)\n"
            "Get Figshare key from: https://figshare.com/account/applications"
        ),
    )
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

    # GSP1000-specific options
    g_gsp = fetch_parser.add_argument_group("GSP1000 options")
    g_gsp.add_argument(
        "--batches",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of HDF5 batch files to create. More batches = lower RAM usage.\n"
            "Recommendations: 16GB -> 100, 32GB+ -> 50. Ignored in test mode."
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
    g_gsp.add_argument(
        "--skip-checksum",
        action="store_true",
        help=(
            "Skip checksum verification during download.\n"
            "Use if you get checksum mismatch errors (server metadata may be outdated)."
        ),
    )

    # dTOR985-specific options
    g_dtor = fetch_parser.add_argument_group("dTOR985 options")
    g_dtor.add_argument(
        "--no-keep-original",
        action="store_true",
        help="Remove original .trk file after conversion to save disk space",
    )


def _build_run_parser(subparsers) -> None:
    """
    Add the run subcommand parser with analysis subcommands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add run parser to.
    """
    run_parser = subparsers.add_parser(
        "run",
        help="Run lesion network mapping analyses",
        description=(
            "Run lesion network mapping analyses on BIDS datasets.\n\n"
            "Available analyses:\n"
            "  rd   (regionaldamage)           - Lesion overlap with parcellations\n"
            "  fnm  (functionalnetworkmapping) - Functional connectivity disruption\n"
            "  snm  (structuralnetworkmapping) - White matter disconnection\n\n"
            "Examples:\n"
            "  lacuna run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks\n"
            "  lacuna run fnm /bids /output --connectome-path /path/to/gsp1000_batches --method boes\n"
            "  lacuna run snm /bids /output --connectome-path /path/to/tractogram.tck --mrtrix-threads 4"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Create analysis subparsers
    analysis_subparsers = run_parser.add_subparsers(
        dest="analysis",
        title="analyses",
        description="Use 'lacuna run <analysis> --help' for analysis-specific options.",
        metavar="<analysis>",
    )

    # Add analysis subcommands
    _build_rd_parser(analysis_subparsers)
    _build_fnm_parser(analysis_subparsers)
    _build_snm_parser(analysis_subparsers)


def _add_shared_run_arguments(parser: ArgumentParser) -> None:
    """Add arguments shared across all run subcommands."""
    # Positional arguments
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

    # BIDS Filtering Options
    g_bids = parser.add_argument_group("BIDS filtering options")
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
            "Required if not detectable from filename or sidecar JSON."
        ),
    )

    # Performance Options
    g_perf = parser.add_argument_group("Performance options")
    g_perf.add_argument(
        "--nprocs",
        type=int,
        default=-1,
        metavar="N",
        help="Number of parallel processes (-1 for all CPUs)",
    )
    g_perf.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Number of masks to process together per batch. "
            "Use -1 for all masks at once (fastest). "
            "Lower values reduce peak memory."
        ),
    )
    g_perf.add_argument(
        "-w",
        "--tmp-dir",
        dest="tmp_dir",
        type=Path,
        default=Path(os.getenv("LACUNA_TMP_DIR", "tmp")),
        metavar="PATH",
        help="Temporary directory for intermediate files",
    )

    # Other Options
    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    g_other.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate results in output",
    )
    g_other.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    )


def _build_rd_parser(subparsers) -> None:
    """Add the RegionalDamage (rd) analysis parser."""
    rd_parser = subparsers.add_parser(
        "rd",
        aliases=["regionaldamage"],
        help="Compute lesion overlap with brain parcellations",
        description=(
            "RegionalDamage Analysis\n\n"
            "Computes lesion overlap with brain parcellations (atlases).\n"
            "For each parcel, calculates the percentage of voxels overlapping\n"
            "with the lesion mask.\n\n"
            "Use 'lacuna info atlases' to see available atlases.\n\n"
            "Examples:\n"
            "  lacuna run rd /bids /output\n"
            "  lacuna run rd /bids /output --parcel-atlases Schaefer2018_100Parcels7Networks\n"
            "  lacuna run rd /bids /output --parcel-atlases Schaefer2018_400Parcels17Networks TianSubcortex_3TS2"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Add shared arguments
    _add_shared_run_arguments(rd_parser)

    # RegionalDamage-specific options
    g_rd = rd_parser.add_argument_group("RegionalDamage options")
    g_rd.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        required=True,
        metavar="ATLAS",
        help="Atlas names to use. Use 'lacuna info atlases' to list available atlases.",
    )
    g_rd.add_argument(
        "--threshold",
        type=float,
        metavar="VALUE",
        help="Threshold for binary mask conversion (for probabilistic masks)",
    )
    g_rd.add_argument(
        "--custom-parcellation",
        nargs=2,
        action="append",
        metavar=("NIFTI", "LABELS"),
        help=(
            "Custom parcellation: NIfTI file path and labels file path. "
            "Can be specified multiple times."
        ),
    )


def _build_fnm_parser(subparsers) -> None:
    """Add the FunctionalNetworkMapping (fnm) analysis parser."""
    fnm_parser = subparsers.add_parser(
        "fnm",
        aliases=["functionalnetworkmapping"],
        help="Compute functional network disconnection maps",
        description=(
            "Functional Network Mapping Analysis\n\n"
            "Computes functional connectivity disruption using a normative\n"
            "functional connectome. Generates correlation, z-score, t-score,\n"
            "and p-value maps.\n\n"
            "Use 'lacuna info connectomes' to see available connectomes.\n\n"
            "Methods:\n"
            "  boes - Mean timeseries across all lesion voxels (default)\n"
            "  pini - PCA-based selection of representative voxels\n\n"
            "Examples:\n"
            "  lacuna run fnm /bids /output --connectome-path ~/.cache/lacuna/gsp1000/\n"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Add shared arguments
    _add_shared_run_arguments(fnm_parser)

    # FNM-specific options
    g_fnm = fnm_parser.add_argument_group("FunctionalNetworkMapping options")
    g_fnm.add_argument(
        "--connectome-path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to HDF5 connectome file or directory of batch files (from 'lacuna fetch gsp1000')",
    )
    g_fnm.add_argument(
        "--method",
        type=str,
        choices=["boes", "pini"],
        default="boes",
        help="Timeseries extraction method (default: boes)",
    )
    g_fnm.add_argument(
        "--pini-percentile",
        type=int,
        default=20,
        metavar="N",
        help="For PINI method: PC1 loading percentile threshold (default: 20)",
    )
    g_fnm.add_argument(
        "--no-p-map",
        action="store_true",
        dest="no_p_map",
        help="Disable p-value map computation (enabled by default)",
    )
    g_fnm.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="FDR correction alpha (default: 0.05, use 0 to disable)",
    )
    g_fnm.add_argument(
        "--t-threshold",
        type=float,
        metavar="VALUE",
        help="Create binary mask for |t| > threshold",
    )
    g_fnm.add_argument(
        "--output-resolution",
        type=int,
        choices=[1, 2],
        metavar="MM",
        help="Output resolution in mm (default: match input)",
    )
    g_fnm.add_argument(
        "--no-return-input-space",
        action="store_true",
        help="Keep outputs in connectome space (default: transform to input space)",
    )

    # Parcel aggregation for FNM outputs
    g_parcels = fnm_parser.add_argument_group("Parcel aggregation options")
    g_parcels.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help="Aggregate FNM outputs to these atlases. Use 'lacuna info atlases' to list.",
    )


def _build_snm_parser(subparsers) -> None:
    """Add the StructuralNetworkMapping (snm) analysis parser."""
    snm_parser = subparsers.add_parser(
        "snm",
        aliases=["structuralnetworkmapping"],
        help="Compute structural disconnection maps",
        description=(
            "Structural Network Mapping Analysis\n\n"
            "Computes white matter disconnection using tractography.\n"
            "Generates disconnection maps showing regions affected by\n"
            "streamline interruption through lesioned tissue.\n\n"
            "Requires MRtrix3 to be installed and in PATH.\n\n"
            "Download a tractogram with 'lacuna fetch dtor985' first.\n\n"
            "Examples:\n"
            "  lacuna run snm /bids /output --connectome-path ~/.cache/lacuna/dtor985/tractogram.tck\n"
            "  lacuna run snm /bids /output --connectome-path /data/dtor985.tck --mrtrix-threads 4"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Add shared arguments
    _add_shared_run_arguments(snm_parser)

    # SNM-specific options
    g_snm = snm_parser.add_argument_group("StructuralNetworkMapping options")
    g_snm.add_argument(
        "--connectome-path",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to .tck tractogram file (from 'lacuna fetch dtor985')",
    )
    g_snm.add_argument(
        "--parcellation",
        type=str,
        metavar="NAME",
        help="Atlas for connectivity matrices. Use 'lacuna info atlases' to list.",
    )
    g_snm.add_argument(
        "--compute-roi-disconnection",
        action="store_true",
        help="Compute per-ROI disconnection values",
    )
    g_snm.add_argument(
        "--output-resolution",
        type=int,
        choices=[1, 2],
        default=2,
        metavar="MM",
        help="Output resolution in mm (default: 2)",
    )
    g_snm.add_argument(
        "--no-cache-tdi",
        action="store_true",
        dest="no_cache_tdi",
        help="Disable TDI caching (enabled by default)",
    )
    g_snm.add_argument(
        "--mrtrix-threads",
        type=int,
        default=1,
        metavar="N",
        dest="mrtrix_threads",
        help="Number of threads for MRtrix3 commands (default: 1)",
    )
    g_snm.add_argument(
        "--no-return-input-space",
        action="store_true",
        help="Keep outputs in connectome space (default: transform to input space)",
    )
    g_snm.add_argument(
        "--show-mrtrix-output",
        action="store_true",
        help="Display MRtrix3 command output",
    )


def _build_collect_parser(subparsers) -> None:
    """
    Add the collect subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add collect parser to.
    """
    collect_parser = subparsers.add_parser(
        "collect",
        help="Aggregate parcelstats across subjects",
        description=(
            "Aggregate subject-level parcelstats TSV files into group-level tables.\n\n"
            "Scans the output directory for *_parcelstats.tsv files and combines\n"
            "them into group-level TSV files.\n\n"
            "Examples:\n"
            "  lacuna collect /bids /output\n"
            "  lacuna collect /bids /output --label lesion\n"
            "  lacuna collect /bids /output --analysis RegionalDamage"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Positional arguments
    collect_parser.add_argument(
        "bids_dir",
        type=Path,
        help="Root folder of BIDS dataset (for metadata)",
    )
    collect_parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory containing derivatives to aggregate",
    )

    # Filtering options
    g_filter = collect_parser.add_argument_group("Filtering options")
    g_filter.add_argument(
        "--label",
        type=str,
        metavar="NAME",
        help="Filter by lesion label (e.g., 'lesion', 'WMH')",
    )
    g_filter.add_argument(
        "--analysis",
        type=str,
        metavar="NAME",
        help="Filter by analysis type (e.g., 'RegionalDamage', 'FunctionalNetworkMapping')",
    )

    # Output options
    g_output = collect_parser.add_argument_group("Output options")
    g_output.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing group files",
    )
    g_output.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    )


def _build_info_parser(subparsers) -> None:
    """
    Add the info subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add info parser to.
    """
    info_parser = subparsers.add_parser(
        "info",
        help="Display available resources (atlases, connectomes)",
        description=(
            "Display detailed information about available resources.\n\n"
            "Topics:\n"
            "  atlases     - Available brain parcellations (atlases)\n"
            "  connectomes - Registered connectomes (functional and structural)\n\n"
            "Examples:\n"
            "  lacuna info atlases\n"
            "  lacuna info connectomes"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    info_parser.add_argument(
        "topic",
        choices=["atlases", "connectomes"],
        help="Topic to display information about",
    )
