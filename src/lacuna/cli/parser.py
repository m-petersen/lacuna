"""
Lacuna CLI argument parser module.

This module provides the argument parser for the Lacuna CLI with a clean
subcommand-based structure:

- lacuna fetch: Download and setup connectomes
- lacuna run <analysis>: Run analyses (rd, fnm, snm, afnm)
- lacuna bidsify: Convert NIfTI files to BIDS format
- lacuna parcellate: Reduce a connectome to a parcel-level connectivity matrix
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
        description=f"Lacuna v{__version__}",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  bidsify    Convert a directory of NIfTI masks into BIDS layout\n"
            "  parcellate Reduce a connectome to a parcel-level connectivity matrix\n"
            "  fetch     Download and setup connectomes for analysis\n"
            "  prepare   Precompute atlas data for NTM analyses\n"
            "  run       Run lesion analyses\n"
            "  collect   Aggregate results across subjects\n"
            "  info      Display available resources (atlases, connectomes)\n"
            "  tutorial  Setup tutorial data for learning Lacuna\n"
            "  check     Validate inputs and check output completeness\n\n"
            "Examples:\n"
            "  lacuna tutorial ./my_tutorial\n"
            "  lacuna bidsify /raw /bids --space MNI152NLin6Asym\n"
            "  lacuna fetch gsp1000 --api-key \\$DATAVERSE_API_KEY\n"
            "  lacuna run rd /bids /output --parcel-atlases schaefer2018parcels100networks7\n"
            "  lacuna run fnm /bids /output --connectome-path /path/to/gsp1000_batches\n"
            "  lacuna collect /output\n"
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
    _build_prepare_parser(subparsers)
    _build_collect_parser(subparsers)
    _build_info_parser(subparsers)
    _build_bidsify_parser(subparsers)
    _build_parcellate_parser(subparsers)
    _build_tutorial_parser(subparsers)
    _build_check_parser(subparsers)

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
        help="Download and setup connectomes",
        description=(
            "Download, process, and register connectomes for lesion network mapping.\n\n"
            "Available connectomes:\n"
            "  gsp1000  - GSP1000 functional connectome (~100GB, requires Dataverse API key)\n"
            "  dtor985  - dTOR985 structural tractogram (~10GB, requires Figshare API key)\n"
            "  hcp1065  - HCP1065 structural tractogram (~1.5GB, no API key required)\n"
            "  ntatlas  - Neurotransmitter PET atlas maps (~30MB, no API key required)\n\n"
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
        choices=["gsp1000", "dtor985", "hcp1065", "ntatlas"],
        help="Connectome to fetch (gsp1000, dtor985, hcp1065, or ntatlas)",
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
            "Download only 1 tarball (~10GB) to test the full pipeline.\n"
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
    g_gsp.add_argument(
        "--no-keep-original",
        action="store_true",
        help="Remove original files after HDF5 conversion to save disk space",
    )

    # dTOR985-specific options
    g_dtor = fetch_parser.add_argument_group("dTOR985 options")
    g_dtor.add_argument(
        "--no-keep-original-trk",
        action="store_true",
        help="Remove original .trk file after conversion to .tck to save disk space",
    )

    # HCP1065-specific options
    g_hcp = fetch_parser.add_argument_group("HCP1065 options")
    g_hcp.add_argument(
        "--no-keep-original-zip",
        action="store_true",
        help="Remove original .zip file and extracted tracts after merging to .tck",
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
            "  rd   (localdamage)                      - Lesion overlap with parcellations\n"
            "  fnm  (functionalnetworkmapping)            - Functional lesion connectivity maps\n"
            "  snm  (structuralnetworkmapping)            - White matter disconnection\n"
            "  afnm (acceleratedfunctionalnetworkmapping) - Accelerated functional LNM (M @ C)\n"
            "  lntf (localneurotransmitterfingerprinting)        - Local NT density within lesion\n"
            "  sntf (structuralneurotransmitterfingerprinting)   - NT at disconnected endpoints\n"
            "  fntf (functionalneurotransmitterfingerprinting)   - NT weighted by connectivity\n\n"
            "Examples:\n"
            "  lacuna run rd /bids /output --parcel-atlases schaefer2018parcels100networks7\n"
            "  lacuna run fnm /bids /output --connectome-path /path/to/gsp1000_batches --method boes\n"
            "  lacuna run snm /bids /output --connectome-path /path/to/tractogram.tck --nprocs 4\n"
            "  lacuna run afnm /bids /output --matrix-path /path/to/gsp1000_schaefer400.tsv \\\n"
            "      --parcel-atlases schaefer2018parcels400networks17"
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
    _build_afnm_parser(analysis_subparsers)
    _build_lntm_parser(analysis_subparsers)
    _build_sntm_parser(analysis_subparsers)
    _build_fntm_parser(analysis_subparsers)


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
        choices=["MNI152NLin6Asym", "MNI152NLin2009cAsym"],
        metavar="SPACE",
        help=(
            "Coordinate space of input masks "
            "(MNI152NLin6Asym or MNI152NLin2009cAsym). "
            "Required if not detectable from filename or sidecar JSON."
        ),
    )

    # Other Options
    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    g_other.add_argument(
        "--on-empty",
        choices=["warn", "skip", "error"],
        default="warn",
        help=(
            "Behavior when a subject has an empty mask (no non-zero voxels) or "
            "no overlap with the analysis network/atlas: "
            "warn (default, process with zero-valued outputs), "
            "skip (exclude from processing), or "
            "error (raise error and halt processing)."
        ),
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
    """Add the LocalDamage (rd) analysis parser."""
    rd_parser = subparsers.add_parser(
        "rd",
        aliases=["localdamage"],
        help="Compute lesion overlap with brain parcellations",
        description=(
            "LocalDamage Analysis\n\n"
            "Computes lesion overlap with brain parcellations (atlases).\n"
            "For each parcel, calculates the percentage of voxels overlapping\n"
            "with the lesion mask.\n\n"
            "Use 'lacuna info atlases' to see available atlases.\n\n"
            "Examples:\n"
            "  lacuna run rd /bids /output\n"
            "  lacuna run rd /bids /output --parcel-atlases schaefer2018parcels100networks7\n"
            "  lacuna run rd /bids /output --parcel-atlases schaefer2018parcels400networks17 tian2020parcels32"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Add shared arguments
    _add_shared_run_arguments(rd_parser)

    # Performance Options
    g_perf = rd_parser.add_argument_group("Performance options")
    g_perf.add_argument(
        "--nprocs",
        type=int,
        default=-1,
        metavar="N",
        help="Number of parallel processes for subject processing (-1 for all CPUs)",
    )
    g_perf.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Number of subjects to process before writing outputs (-1 for all). "
            "Lower values produce incremental output and reduce peak memory."
        ),
    )

    # LocalDamage-specific options
    g_rd = rd_parser.add_argument_group("LocalDamage options")
    g_rd.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help="Atlas names to use. Use 'lacuna info atlases' to list available atlases.",
    )
    g_rd.add_argument(
        "--custom-parcellation",
        nargs=4,
        action="append",
        metavar=("NAME", "NIFTI", "LABELS", "SPACE"),
        help=(
            "Custom parcellation: a short name for output labelling, NIfTI file "
            "path, labels file path, and coordinate space (e.g., MNI152NLin6Asym). "
            "Can be specified multiple times."
        ),
    )


def _build_fnm_parser(subparsers) -> None:
    """Add the FunctionalNetworkMapping (fnm) analysis parser."""
    fnm_parser = subparsers.add_parser(
        "fnm",
        aliases=["functionalnetworkmapping"],
        help="Compute functional lesion connectivity maps",
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

    # Performance Options
    g_perf = fnm_parser.add_argument_group("Performance options")
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
            "Number of lesion masks to vectorize together (-1 for all). "
            "Controls memory usage when processing many subjects."
        ),
    )

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
    g_parcels.add_argument(
        "--custom-parcellation",
        nargs=4,
        action="append",
        metavar=("NAME", "NIFTI", "LABELS", "SPACE"),
        help=(
            "Custom parcellation: a short name for output labelling, NIfTI file "
            "path, labels file path, and coordinate space (e.g., MNI152NLin6Asym). "
            "Can be specified multiple times."
        ),
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
            "  lacuna run snm /bids /output --connectome-path /data/dtor985.tck --nprocs 4"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Add shared arguments
    _add_shared_run_arguments(snm_parser)

    # Performance Options
    g_perf = snm_parser.add_argument_group("Performance options")
    g_perf.add_argument(
        "--nprocs",
        type=int,
        default=-1,
        metavar="N",
        help="Number of threads for MRtrix3 processing (-1 for all CPUs)",
    )
    g_perf.add_argument(
        "--batch-size",
        type=int,
        default=-1,
        metavar="N",
        help=(
            "Number of subjects to process before writing outputs (-1 for all). "
            "Lower values produce incremental output and reduce peak memory."
        ),
    )
    g_perf.add_argument(
        "-w",
        "--tmp-dir",
        dest="tmp_dir",
        type=Path,
        default=Path(os.getenv("LACUNA_TMP_DIR", "tmp")),
        metavar="PATH",
        help="Temporary directory for MRtrix3 intermediate files",
    )

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
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help="Atlas name(s) for parcellation-based analyses. Use 'lacuna info atlases' to list.",
    )
    g_snm.add_argument(
        "--compute-disconnectivity-matrix",
        action="store_true",
        help="Compute disconnectivity matrices (requires --parcel-atlases or --custom-parcellation)",
    )
    g_snm.add_argument(
        "--compute-roi-disconnection",
        action="store_true",
        help="Compute per-ROI disconnection values (requires --parcel-atlases or --custom-parcellation)",
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
        "--no-return-input-space",
        action="store_true",
        help="Keep outputs in connectome space (default: transform to input space)",
    )
    g_snm.add_argument(
        "--show-mrtrix-output",
        action="store_true",
        help="Display MRtrix3 command output",
    )
    g_snm.add_argument(
        "--custom-parcellation",
        nargs=4,
        action="append",
        metavar=("NAME", "NIFTI", "LABELS", "SPACE"),
        help=(
            "Custom parcellation: a short name for output labelling, NIfTI file "
            "path, labels file path, and coordinate space (e.g., MNI152NLin6Asym). "
            "Can be specified multiple times."
        ),
    )


def _build_afnm_parser(subparsers) -> None:
    """Add the AcceleratedFunctionalNetworkMapping (afnm) analysis parser."""
    afnm_parser = subparsers.add_parser(
        "afnm",
        aliases=["acceleratedfunctionalnetworkmapping"],
        help="Compute accelerated functional network maps (M @ C)",
        description=(
            "Accelerated Functional Network Mapping Analysis\n\n"
            "Accelerated lesion network mapping via matrix multiplication:\n"
            "  AFNMAP = M \u00d7 C\n"
            "where M is the lesion-by-parcel weight matrix and C is a precomputed\n"
            "group-average parcel-level functional connectivity matrix (produced by\n"
            "'lacuna parcellate'). Contrast with voxel-level FNM (`lacuna run fnm`).\n\n"
            "Examples:\n"
            "  lacuna run afnm /bids /output \\\n"
            "      --matrix-path /data/parcellated/GSP1000_schaefer400.tsv \\\n"
            "      --parcel-atlases schaefer2018parcels400networks17"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    _add_shared_run_arguments(afnm_parser)

    g_perf = afnm_parser.add_argument_group("Performance options")
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
        help="Number of lesion masks to vectorize together (-1 for all).",
    )

    g_afnm = afnm_parser.add_argument_group("AcceleratedFunctionalNetworkMapping options")
    g_afnm.add_argument(
        "--matrix-path",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Path to the parcel-level group FC matrix TSV (from "
            "'lacuna parcellate --modality functional')."
        ),
    )
    g_afnm.add_argument(
        "--lesion-weighting",
        type=str,
        choices=["fractional", "binary", "voxel_count"],
        default="fractional",
        help=(
            "How to weight the lesion\u2192parcel row vector m: "
            "'fractional' (default, 1/n_regions_touched), 'binary' (0/1), or "
            "'voxel_count' (fraction of parcel voxels covered by the lesion)."
        ),
    )

    g_parc = afnm_parser.add_argument_group("Parcellation selection")
    g_parc.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help="Atlas name matching the parcellation used to build --matrix-path.",
    )
    g_parc.add_argument(
        "--custom-parcellation",
        nargs=4,
        action="append",
        metavar=("NAME", "NIFTI", "LABELS", "SPACE"),
        help=(
            "Custom parcellation: a short name for output labelling, NIfTI file "
            "path, labels file path, and coordinate space. Must match the "
            "parcellation used to build --matrix-path."
        ),
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
            "Scans a derivatives directory for *_parcelstats.tsv files and combines\n"
            "them into group-level TSV files.\n\n"
            "Examples:\n"
            "  lacuna collect /output\n"
            "  lacuna collect /output --pattern '*roidisconnection*'\n"
            "  lacuna collect /output --output-dir /results --pattern '*acuteinfarct*'"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Positional arguments
    collect_parser.add_argument(
        "derivatives_dir",
        type=Path,
        help="Directory containing BIDS derivatives to aggregate (e.g., lacuna run output)",
    )

    # Filtering options
    g_filter = collect_parser.add_argument_group("Filtering options")
    g_filter.add_argument(
        "--pattern",
        type=str,
        metavar="GLOB",
        help="Glob pattern to filter parcelstats files (e.g., '*acuteinfarct*', '*lesion*')",
    )

    # Output options
    g_output = collect_parser.add_argument_group("Output options")
    g_output.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Directory for group-level TSV files (default: same as derivatives_dir)",
    )
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


def _build_bidsify_parser(subparsers) -> None:
    """Add the top-level `bidsify` subcommand parser."""
    bidsify_parser = subparsers.add_parser(
        "bidsify",
        help="Convert NIfTI files to BIDS format",
        description=(
            "Convert a directory of NIfTI mask files to BIDS format.\n\n"
            "Input filenames become subject IDs (special characters removed).\n"
            "For example: patient_001.nii.gz -> sub-patient001/\n\n"
            "Examples:\n"
            "  lacuna bidsify /raw/masks /bids --space MNI152NLin6Asym\n"
            "  lacuna bidsify /raw /bids --space MNI152NLin6Asym --session 01 --label lesion\n"
            "  lacuna bidsify ./masks ./bids_masks --space MNI152NLin2009cAsym"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    bidsify_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing NIfTI mask files (.nii or .nii.gz)",
    )
    bidsify_parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for BIDS dataset",
    )

    bidsify_parser.add_argument(
        "--space",
        "-s",
        type=str,
        required=True,
        choices=["MNI152NLin6Asym", "MNI152NLin2009cAsym"],
        help="Coordinate space of the masks (MNI152NLin6Asym or MNI152NLin2009cAsym)",
    )

    g_opts = bidsify_parser.add_argument_group("Optional BIDS entities")
    g_opts.add_argument(
        "--session",
        "-ses",
        type=str,
        metavar="LABEL",
        help="Session label (e.g., '01', 'baseline'). Creates ses-<label> subdirectory.",
    )
    g_opts.add_argument(
        "--label",
        "-l",
        type=str,
        metavar="NAME",
        help="Label for the mask entity (e.g., 'lesion', 'tumor')",
    )

    g_other = bidsify_parser.add_argument_group("Other options")
    g_other.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress messages",
    )


def _build_parcellate_parser(subparsers) -> None:
    """Add the top-level `parcellate` subcommand parser."""
    parcellate_parser = subparsers.add_parser(
        "parcellate",
        help="Reduce a connectome to a parcel-level connectivity matrix",
        description=(
            "Reduce a connectome to a parcel-level N\u00d7N ConnectivityMatrix.\n\n"
            "Inputs:\n"
            "  - A voxelwise functional connectome (HDF5, same format as 'lacuna run fnm'), or\n"
            "  - A structural tractogram (.tck, same format as 'lacuna run snm').\n"
            "The --modality flag is required; modality is never inferred from path or extension.\n\n"
            "Output is a BIDS-style TSV + JSON sidecar ConnectivityMatrix (same format that\n"
            "'lacuna run snm' uses for disconnectivity matrices) written under the output\n"
            "directory. Each selected parcellation produces its own output file.\n\n"
            "Examples:\n"
            "  lacuna parcellate --connectome-path ~/.cache/lacuna/gsp1000/ \\\n"
            "      --modality functional --parcel-atlases schaefer2018parcels400networks17 \\\n"
            "      --output /data/parcellated/\n"
            "  lacuna parcellate --connectome-path /data/dtor985.tck \\\n"
            "      --modality structural --parcel-atlases schaefer2018parcels400networks17 \\\n"
            "      --output /data/parcellated/"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    g_io = parcellate_parser.add_argument_group("Inputs / outputs")
    g_io.add_argument(
        "--connectome-path",
        type=Path,
        required=True,
        metavar="PATH",
        help=(
            "Path to the whole-brain connectome: voxelwise HDF5 file or directory "
            "(for --modality functional), or tractogram .tck (for --modality structural)."
        ),
    )
    g_io.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["functional", "structural"],
        help="Connectome modality. Required; no inference from path or extension.",
    )
    g_io.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory for the parcellated matrix (TSV + JSON sidecar).",
    )

    g_parc = parcellate_parser.add_argument_group("Parcellation selection")
    g_parc.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help="Atlas names to use. Use 'lacuna info atlases' to list available atlases.",
    )
    g_parc.add_argument(
        "--custom-parcellation",
        nargs=4,
        action="append",
        metavar=("NAME", "NIFTI", "LABELS", "SPACE"),
        help=(
            "Custom parcellation: a short name for output labelling, NIfTI file "
            "path, labels file path, and coordinate space (e.g., MNI152NLin6Asym). "
            "Can be specified multiple times."
        ),
    )

    g_other = parcellate_parser.add_argument_group("Other options")
    g_other.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    g_other.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG).",
    )


def _build_tutorial_parser(subparsers) -> None:
    """
    Add the tutorial subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object to add tutorial parser to.
    """
    tutorial_parser = subparsers.add_parser(
        "tutorial",
        help="Setup tutorial data for learning Lacuna",
        description=(
            "Copy the bundled tutorial dataset to a directory.\n\n"
            "The tutorial dataset includes:\n"
            "  - 3 synthetic subjects (sub-01, sub-02, sub-03)\n"
            "  - Binary lesion masks in MNI152NLin6Asym space\n"
            "  - BIDS-compliant structure ready for analysis\n\n"
            "Use --raw to output a flat directory of NIfTI mask files\n"
            "(named by subject ID) instead of the BIDS structure.\n"
            "This is useful for demonstrating the bidsify workflow:\n"
            "  lacuna tutorial -> lacuna bidsify -> lacuna fetch -> lacuna run\n\n"
            "Examples:\n"
            "  lacuna tutorial ./my_tutorial\n"
            "  lacuna tutorial /data/lacuna_tutorial --force\n"
            "  lacuna tutorial ./raw_masks --raw"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    # Optional positional argument for output directory
    tutorial_parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Output directory for tutorial data (default: ./lacuna_tutorial)",
    )

    # Options
    tutorial_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing directory if it exists",
    )

    tutorial_parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Output raw NIfTI mask files in a flat directory (no BIDS structure). "
            "Use this to practice the bidsify workflow."
        ),
    )


def _add_shared_check_arguments(parser: ArgumentParser) -> None:
    """Add arguments shared by all check analysis subcommands."""
    parser.add_argument(
        "bids_dir",
        type=Path,
        help="Root folder of BIDS dataset (sub-XXXXX folders at top level)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to check for existing results",
    )

    g_bids = parser.add_argument_group("BIDS filtering options")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        nargs="+",
        type=_drop_sub,
        metavar="LABEL",
        help="Subject IDs to check (without sub- prefix)",
    )
    g_bids.add_argument(
        "--session-id",
        "--session_id",
        nargs="+",
        metavar="SESSION",
        help="Session IDs to check (without ses- prefix)",
    )
    g_bids.add_argument(
        "--pattern",
        type=str,
        metavar="GLOB",
        help="Glob pattern to filter mask files (e.g., '*label-WMH*')",
    )
    g_bids.add_argument(
        "--mask-space",
        type=str,
        choices=["MNI152NLin6Asym", "MNI152NLin2009cAsym"],
        metavar="SPACE",
        help=(
            "Coordinate space of input masks "
            "(MNI152NLin6Asym or MNI152NLin2009cAsym). "
            "Required if not detectable from filename or sidecar JSON."
        ),
    )

    g_out = parser.add_argument_group("Output options")
    g_out.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help=(
            "Suppress all output except missing/empty subject IDs (one per line). "
            "Useful for shell scripting."
        ),
    )
    g_out.add_argument(
        "--output-file",
        type=Path,
        metavar="PATH",
        help="Write missing/empty subject IDs to a file (one per line).",
    )
    g_out.add_argument(
        "--check-content",
        action="store_true",
        help=(
            "Inspect output file content to detect empty (all-zero) results. "
            "Without this flag, only file existence is checked. "
            "Adds overhead of reading sidecar JSON / TSV files."
        ),
    )


def _build_check_parser(subparsers) -> None:
    """Add the check subcommand parser."""
    check_parser = subparsers.add_parser(
        "check",
        help="Validate inputs and check output completeness",
        description=(
            "Validate input masks before a run, or check output completeness after.\n\n"
            "Use 'lacuna check input' to catch common mask issues (non-binary,\n"
            "empty, missing space) before committing to a long batch run.\n"
            "Use 'lacuna check rd|fnm|snm|afnm' to identify subjects with missing outputs.\n\n"
            "Available checks:\n"
            "  input - Validate input masks (binary, non-empty, space)\n"
            "  rd    - Check for parcelstats TSV files (LocalDamage)\n"
            "  fnm   - Check for functional rmap NIfTI files\n"
            "  snm   - Check for disconnection NIfTI files\n"
            "  afnm  - Check for accelerated functional LNM parcel outputs\n\n"
            "Examples:\n"
            "  lacuna check input /bids\n"
            "  lacuna check rd /bids /output\n"
            "  lacuna check rd /bids /output --parcel-atlases schaefer2018parcels400networks7\n"
            "  lacuna check rd /bids /output --output-file missing.txt\n"
            "  lacuna check fnm /bids /output --quiet\n"
            "  lacuna check snm /bids /output --participant-label 001 002"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    check_subparsers = check_parser.add_subparsers(
        dest="analysis",
        title="analyses",
        description="Use 'lacuna check <analysis> --help' for analysis-specific options.",
        metavar="<analysis>",
    )

    _build_check_input_parser(check_subparsers)
    _build_check_rd_parser(check_subparsers)
    _build_check_fnm_parser(check_subparsers)
    _build_check_snm_parser(check_subparsers)
    _build_check_afnm_parser(check_subparsers)


def _build_check_rd_parser(subparsers) -> None:
    """Add the check rd subcommand parser."""
    rd_parser = subparsers.add_parser(
        "rd",
        aliases=["localdamage"],
        help="Check for LocalDamage parcelstats outputs",
        description=(
            "Check which subjects have LocalDamage parcelstats TSV outputs.\n\n"
            "By default, any '*method-rd*parcelstats.tsv' file in a\n"
            "subject's output directory counts as complete. If --parcel-atlases is\n"
            "given, each named atlas is checked individually.\n\n"
            "Examples:\n"
            "  lacuna check rd /bids /output\n"
            "  lacuna check rd /bids /output --parcel-atlases schaefer2018parcels400networks7\n"
            "  lacuna check rd /bids /output --output-file missing.txt"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_check_arguments(rd_parser)

    g_rd = rd_parser.add_argument_group("LocalDamage options")
    g_rd.add_argument(
        "--parcel-atlases",
        nargs="+",
        type=str,
        metavar="ATLAS",
        help=(
            "Atlas name(s) to check individually. If omitted, any parcelstats "
            "TSV file counts as complete. Use 'lacuna info atlases' to list."
        ),
    )


def _build_check_fnm_parser(subparsers) -> None:
    """Add the check fnm subcommand parser."""
    fnm_parser = subparsers.add_parser(
        "fnm",
        aliases=["functionalnetworkmapping"],
        help="Check for FunctionalNetworkMapping rmap outputs",
        description=(
            "Check which subjects have FunctionalNetworkMapping rmap NIfTI outputs.\n\n"
            "A subject is considered complete if a '*method-fnm*desc-rmap*.nii.gz' file\n"
            "exists in their output directory.\n\n"
            "Examples:\n"
            "  lacuna check fnm /bids /output\n"
            "  lacuna check fnm /bids /output --quiet"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_check_arguments(fnm_parser)


def _build_check_snm_parser(subparsers) -> None:
    """Add the check snm subcommand parser."""
    snm_parser = subparsers.add_parser(
        "snm",
        aliases=["structuralnetworkmapping"],
        help="Check for StructuralNetworkMapping disconnection outputs",
        description=(
            "Check which subjects have StructuralNetworkMapping disconnection outputs.\n\n"
            "A subject is considered complete if a '*method-snm*desc-disconnectionpct*.nii.gz'\n"
            "file exists in their output directory.\n\n"
            "Examples:\n"
            "  lacuna check snm /bids /output\n"
            "  lacuna check snm /bids /output --output-file missing.txt"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_check_arguments(snm_parser)


def _build_check_afnm_parser(subparsers) -> None:
    """Add the check afnm subcommand parser."""
    afnm_parser = subparsers.add_parser(
        "afnm",
        aliases=["acceleratedfunctionalnetworkmapping"],
        help="Check for AcceleratedFunctionalNetworkMapping parcel outputs",
        description=(
            "Check which subjects have AcceleratedFunctionalNetworkMapping parcel outputs.\n\n"
            "A subject is considered complete if a '*method-afnm*parcelstats.tsv' file\n"
            "exists in their output directory.\n\n"
            "Examples:\n"
            "  lacuna check afnm /bids /output\n"
            "  lacuna check afnm /bids /output --quiet"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_check_arguments(afnm_parser)


def _build_check_input_parser(subparsers) -> None:
    """Add the check input subcommand parser."""
    input_parser = subparsers.add_parser(
        "input",
        help="Validate input masks before running analyses",
        description=(
            "Validate input mask files in a BIDS directory.\n\n"
            "Checks each mask for common issues that would cause analysis failures:\n"
            "  - File loadable by nibabel\n"
            "  - 3D image (not 4D or 2D)\n"
            "  - Binary (only values 0 and 1)\n"
            "  - Non-empty (at least 1 non-zero voxel)\n"
            "  - Coordinate space detectable\n"
            "  - Filename space consistent with affine\n"
            "  - Lesion not suspiciously small (<10 voxels)\n"
            "  - Consistent dimensions and spaces across dataset\n\n"
            "Examples:\n"
            "  lacuna check input /bids\n"
            "  lacuna check input /bids --mask-space MNI152NLin6Asym\n"
            "  lacuna check input /bids --output-file problems.txt"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    input_parser.add_argument(
        "bids_dir",
        type=Path,
        help="Root folder of BIDS dataset (sub-XXXXX folders at top level)",
    )

    g_bids = input_parser.add_argument_group("BIDS filtering options")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        nargs="+",
        type=_drop_sub,
        metavar="LABEL",
        help="Subject IDs to check (without sub- prefix)",
    )
    g_bids.add_argument(
        "--session-id",
        "--session_id",
        nargs="+",
        metavar="SESSION",
        help="Session IDs to check (without ses- prefix)",
    )
    g_bids.add_argument(
        "--pattern",
        type=str,
        metavar="GLOB",
        help="Glob pattern to filter mask files (e.g., '*label-WMH*')",
    )
    g_bids.add_argument(
        "--mask-space",
        type=str,
        choices=["MNI152NLin6Asym", "MNI152NLin2009cAsym"],
        metavar="SPACE",
        help=(
            "Coordinate space of input masks. Providing this suppresses "
            "'space not detectable' errors for files without space- in the filename."
        ),
    )

    g_out = input_parser.add_argument_group("Output options")
    g_out.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Print only paths with errors (one per line).",
    )
    g_out.add_argument(
        "--output-file",
        type=Path,
        metavar="PATH",
        help="Write paths with errors to a file (one per line).",
    )


def _build_prepare_parser(subparsers) -> None:
    """Add the prepare subcommand parser."""
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Precompute non-subject-specific data for analyses",
        description=(
            "Precompute non-subject-specific data needed by analyses.\n\n"
            "Available targets:\n"
            "  lntf - Prepare NT atlas (average per target, z-score)\n"
            "  sntf - Precompute structural endpoint NT weights\n"
            "  ace  - Run ACE (Atlas Connectivity Enrichment) on normative data\n\n"
            "Examples:\n"
            "  lacuna prepare lntf\n"
            "  lacuna prepare lntf --source-dir /path/to/pet_maps\n"
            "  lacuna prepare sntf --connectome-path /path/to/tractogram.tck\n"
            "  lacuna prepare ace --connectome-name GSP1000"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )

    prepare_subparsers = prepare_parser.add_subparsers(
        dest="prepare_target",
        title="targets",
        description="Use 'lacuna prepare <target> --help' for target-specific options.",
        metavar="<target>",
    )

    # prepare lntf
    prepare_lntm = prepare_subparsers.add_parser(
        "lntf",
        help="Prepare NT atlas (average per target, z-score)",
    )
    prepare_lntm.add_argument(
        "--source-dir", type=str, default=None,
        help="Directory with raw PET NIfTI maps",
    )
    prepare_lntm.add_argument(
        "--cache-dir", type=str, default=None,
        help="Output cache directory",
    )
    prepare_lntm.add_argument(
        "--map-config", type=str, default=None,
        help="YAML map selection config file",
    )

    # prepare sntf
    prepare_sntm = prepare_subparsers.add_parser(
        "sntf",
        help="Precompute structural endpoint NT weights",
    )
    prepare_sntm.add_argument(
        "--connectome-path", type=str, required=True,
        help="Path to structural tractogram (.tck)",
    )
    prepare_sntm.add_argument(
        "--cache-dir", type=str, default=None,
        help="Output cache directory",
    )

    # prepare ace
    prepare_ace = prepare_subparsers.add_parser(
        "ace",
        help="Run ACE (Atlas Connectivity Enrichment) on normative data",
    )
    prepare_ace.add_argument(
        "--connectome-name", type=str, required=True,
        help="Normative fMRI connectome name (e.g., GSP1000)",
    )
    prepare_ace.add_argument(
        "--cache-dir", type=str, default=None,
        help="Output cache directory",
    )


def _add_ntm_common_args(parser: ArgumentParser) -> None:
    """Add arguments shared by all NTM analyses."""
    g_ntm = parser.add_argument_group("Neurotransmitter mapping options")
    g_ntm.add_argument(
        "--targets", type=str, default="all",
        help="Target preset or comma-separated list (default: all)",
    )
    g_ntm.add_argument(
        "--enriched", action="store_true",
        help="Use ACE-enriched atlas instead of static",
    )
    g_ntm.add_argument(
        "--ace-cache-dir", type=str, default=None,
        help="Directory with ACE outputs (required if --enriched)",
    )
    g_ntm.add_argument(
        "--atlas-cache-dir", type=str, required=True,
        help="Directory with prepared NT atlas (from lacuna prepare lntf)",
    )


def _build_lntm_parser(subparsers) -> None:
    """Add the LocalNeurotransmitterFingerprinting (lntf) analysis parser."""
    lntm_parser = subparsers.add_parser(
        "lntf",
        aliases=["localneurotransmitterfingerprinting"],
        help="Compute local NT density within the lesion",
        description=(
            "Local neurotransmitter fingerprinting: score NT atlas values directly\n"
            "within the lesion mask. Answers: 'what neurotransmitter landscape\n"
            "did the lesion wipe out?'\n\n"
            "Examples:\n"
            "  lacuna run lntf /bids /output\n"
            "  lacuna run lntf /bids /output --targets dopaminergic\n"
            "  lacuna run lntf /bids /output --parcel-atlases schaefer2018parcels100networks7"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_run_arguments(lntm_parser)
    _add_ntm_common_args(lntm_parser)

    g_lntm = lntm_parser.add_argument_group("LNTF-specific options")
    g_lntm.add_argument(
        "--aggregation", choices=["mean", "sum"], default="mean",
        help="Scoring aggregation method (default: mean)",
    )


def _build_sntm_parser(subparsers) -> None:
    """Add the StructuralNeurotransmitterFingerprinting (sntf) analysis parser."""
    sntm_parser = subparsers.add_parser(
        "sntf",
        aliases=["structuralneurotransmitterfingerprinting"],
        help="Compute NT at disconnected streamline endpoints",
        description=(
            "Structural neurotransmitter fingerprinting: score NT atlas values at\n"
            "endpoints of lesion-disconnected streamlines. Answers: 'what\n"
            "NT-weighted structural connectivity does the lesion disrupt?'\n\n"
            "Examples:\n"
            "  lacuna run sntf /bids /output --connectome-path /path/to/tractogram.tck\n"
            "  lacuna run sntf /bids /output --connectome-path /path/to/tractogram.tck --targets serotonergic"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_run_arguments(sntm_parser)
    _add_ntm_common_args(sntm_parser)

    g_sntm = sntm_parser.add_argument_group("SNTF-specific options")
    g_sntm.add_argument(
        "--connectome-path", type=str, required=True,
        help="Path to structural tractogram (.tck)",
    )
    g_sntm.add_argument(
        "--precomputed-weights-dir", type=str, default=None,
        help="Directory with precomputed endpoint NT weights (from lacuna prepare sntf)",
    )


def _build_fntm_parser(subparsers) -> None:
    """Add the FunctionalNeurotransmitterFingerprinting (fntf) analysis parser."""
    fntm_parser = subparsers.add_parser(
        "fntf",
        aliases=["functionalneurotransmitterfingerprinting"],
        help="Compute NT weighted by functional connectivity",
        description=(
            "Functional neurotransmitter fingerprinting: score NT atlas values\n"
            "weighted by functional connectivity of the lesion. Answers:\n"
            "'what NT systems are functionally connected to the lesion?'\n\n"
            "Examples:\n"
            "  lacuna run fntf /bids /output --connectome-name GSP1000\n"
            "  lacuna run fntf /bids /output --connectome-name GSP1000 --enriched --ace-cache-dir /path/to/ace"
        ),
        formatter_class=RawDescriptionHelpFormatter,
    )
    _add_shared_run_arguments(fntm_parser)
    _add_ntm_common_args(fntm_parser)

    g_fntm = fntm_parser.add_argument_group("FNTF-specific options")
    g_fntm.add_argument(
        "--connectome-name", type=str, required=True,
        help="Functional connectome name (e.g., GSP1000)",
    )
    g_fntm.add_argument(
        "--method", choices=["boes", "pini"], default="boes",
        help="Lesion timeseries extraction method (default: boes)",
    )
