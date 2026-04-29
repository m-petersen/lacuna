"""
Lacuna CLI main module.

This module provides the main entry point for the Lacuna CLI, orchestrating
the workflow from argument parsing through analysis execution to output writing.

Commands:
    lacuna fetch     - Download and setup connectomes
    lacuna run       - Run analyses (ld, fnm, snm)
    lacuna collect   - Aggregate results across subjects
    lacuna info      - Display available resources

Functions:
    main: Main CLI entry point that parses arguments and runs the workflow.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import Namespace

from lacuna.core.exceptions import ValidationError
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)

# Exit codes following BIDS-Apps convention
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_INVALID_ARGS = 2
EXIT_IO_ERROR = 74
EXIT_BIDS_ERROR = 64
EXIT_ANALYSIS_ERROR = 65


@dataclass
class RunConfig:
    """Configuration for run commands."""

    bids_dir: Path
    output_dir: Path
    analysis: str
    participant_label: list[str] | None = None
    session_id: list[str] | None = None
    pattern: str | None = None
    space: str | None = None
    n_procs: int = -1
    batch_size: int = -1
    tmp_dir: Path | None = None
    overwrite: bool = False
    keep_intermediate: bool = False
    on_empty: str = "warn"  # "warn", "skip", or "error"
    verbose_count: int = 0
    # Analysis-specific options stored as dict
    analysis_options: dict[str, Any] | None = None

    @property
    def is_single_file(self) -> bool:
        """Check if input is a single NIfTI file rather than BIDS directory."""
        return self.bids_dir.is_file() and self.bids_dir.suffix in (".nii", ".gz")

    @property
    def log_level(self) -> int:
        """Convert verbose_count to log level."""
        return max(25 - 5 * self.verbose_count, 10)

    @property
    def verbose(self) -> bool:
        """Check if verbose output is enabled."""
        return self.verbose_count >= 1

    @classmethod
    def from_args(cls, args: Namespace) -> RunConfig:
        """Create RunConfig from parsed arguments."""
        # Collect analysis-specific options based on analysis type
        analysis_options: dict[str, Any] = {}

        # Common analysis options
        # Note: SNM uses parcellation_name (set below), not parcel_names
        if (
            hasattr(args, "parcel_atlases")
            and args.parcel_atlases
            and args.analysis not in ("snm", "structuralnetworkmapping")
        ):
            analysis_options["parcel_names"] = args.parcel_atlases
        if hasattr(args, "custom_parcellation") and args.custom_parcellation:
            analysis_options["custom_parcellation"] = args.custom_parcellation
        if hasattr(args, "keep_intermediate") and args.keep_intermediate:
            analysis_options["keep_intermediate"] = args.keep_intermediate

        # FNM/SNM connectome path - always provided as path
        if hasattr(args, "connectome_path") and args.connectome_path:
            analysis_options["_connectome_path"] = args.connectome_path
        if hasattr(args, "method") and args.method:
            analysis_options["method"] = args.method
        if hasattr(args, "pini_percentile"):
            analysis_options["pini_percentile"] = args.pini_percentile
        # Handle --no-p-map flag (default is to compute p-map)
        if hasattr(args, "no_p_map") and args.no_p_map:
            analysis_options["compute_p_map"] = False
        if hasattr(args, "fdr_alpha"):
            fdr_alpha = args.fdr_alpha
            analysis_options["fdr_alpha"] = fdr_alpha if fdr_alpha > 0 else None
        if hasattr(args, "t_threshold") and args.t_threshold is not None:
            analysis_options["t_threshold"] = args.t_threshold
        if hasattr(args, "output_resolution") and args.output_resolution is not None:
            analysis_options["output_resolution"] = args.output_resolution
        if hasattr(args, "no_return_input_space") and args.no_return_input_space:
            analysis_options["return_in_input_space"] = False

        # SNM-specific options
        if (
            hasattr(args, "parcel_atlases")
            and args.parcel_atlases
            and args.analysis in ("snm", "structuralnetworkmapping")
        ):
            analysis_options["parcellation_name"] = args.parcel_atlases
        if hasattr(args, "compute_disconnectivity_matrix") and args.compute_disconnectivity_matrix:
            analysis_options["compute_disconnectivity_matrix"] = True
        if hasattr(args, "compute_roi_disconnection") and args.compute_roi_disconnection:
            analysis_options["compute_roi_disconnection"] = True
        # Handle --no-cache-tdi flag (default is to cache)
        if hasattr(args, "no_cache_tdi") and args.no_cache_tdi:
            analysis_options["cache_tdi"] = False
        # Pass nprocs to analysis as n_jobs
        nprocs = getattr(args, "nprocs", -1)
        if nprocs != 1 and args.analysis in (
            "snm",
            "structuralnetworkmapping",
            "fnm",
            "functionalnetworkmapping",
        ):
            analysis_options["n_jobs"] = nprocs
        if hasattr(args, "show_mrtrix_output") and args.show_mrtrix_output:
            analysis_options["show_mrtrix_output"] = True

        # AFNM-specific options
        if args.analysis in ("afnm", "acceleratedfunctionalnetworkmapping"):
            if getattr(args, "matrix_path", None) is not None:
                analysis_options["matrix_path"] = args.matrix_path
            if getattr(args, "lesion_weighting", None) is not None:
                analysis_options["lesion_weighting"] = args.lesion_weighting

        # NTM-shared options (lntf, sntf, fntf)
        ntm_analyses = (
            "lntf", "localneurotransmitterfingerprinting",
            "sntf", "structuralneurotransmitterfingerprinting",
            "fntf", "functionalneurotransmitterfingerprinting",
        )
        if args.analysis in ntm_analyses:
            if getattr(args, "atlas_cache_dir", None) is not None:
                analysis_options["atlas_cache_dir"] = args.atlas_cache_dir
            if getattr(args, "ace_cache_dir", None) is not None:
                analysis_options["ace_cache_dir"] = args.ace_cache_dir

        # LNTF-specific
        if args.analysis in ("lntf", "localneurotransmitterfingerprinting"):
            if getattr(args, "aggregation", None) is not None:
                analysis_options["aggregation"] = args.aggregation

        # SNTF-specific
        if args.analysis in ("sntf", "structuralneurotransmitterfingerprinting"):
            if getattr(args, "precomputed_weights_dir", None) is not None:
                analysis_options["precomputed_weights_dir"] = args.precomputed_weights_dir
            if getattr(args, "endpoint_combine", None) is not None:
                analysis_options["endpoint_combine"] = args.endpoint_combine
            if getattr(args, "aggregation", None) is not None:
                analysis_options["aggregation"] = args.aggregation

        # FNTF-specific
        if args.analysis in ("fntf", "functionalneurotransmitterfingerprinting"):
            if getattr(args, "connectome_name", None) is not None:
                analysis_options["connectome_name"] = args.connectome_name
            if getattr(args, "method", None) is not None:
                analysis_options["method"] = args.method

        return cls(
            bids_dir=args.bids_dir,
            output_dir=args.output_dir,
            analysis=args.analysis,
            participant_label=getattr(args, "participant_label", None),
            session_id=getattr(args, "session_id", None),
            pattern=getattr(args, "pattern", None),
            space=getattr(args, "mask_space", None),
            n_procs=getattr(args, "nprocs", -1),
            batch_size=getattr(args, "batch_size", -1),
            tmp_dir=getattr(args, "tmp_dir", None),
            overwrite=getattr(args, "overwrite", False),
            keep_intermediate=getattr(args, "keep_intermediate", False),
            on_empty=getattr(args, "on_empty", "warn"),
            verbose_count=getattr(args, "verbose_count", 0),
            analysis_options=analysis_options,
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.bids_dir.exists():
            raise ValueError(f"Input path does not exist: {self.bids_dir}")
        if self.output_dir.resolve() == self.bids_dir.resolve():
            raise ValueError("Output directory cannot be same as input path")
        if self.is_single_file and not self.space:
            raise ValueError("--mask-space is required when processing a single NIfTI file")
        if self.n_procs < -1 or self.n_procs == 0:
            raise ValueError(f"--nprocs must be -1 (all CPUs) or >= 1, got {self.n_procs}")

        # SNM flag dependency validation
        if self.analysis in ("snm", "structuralnetworkmapping"):
            opts = self.analysis_options
            has_atlas = "parcellation_name" in opts or opts.get("custom_parcellation")
            has_disconn = opts.get("compute_disconnectivity_matrix", False)
            has_roi = opts.get("compute_roi_disconnection", False)

            if has_atlas and not (has_disconn or has_roi):
                raise ValueError(
                    "--parcel-atlases/--custom-parcellation requires at least one of "
                    "--compute-disconnectivity-matrix or --compute-roi-disconnection"
                )
            if (has_disconn or has_roi) and not has_atlas:
                raise ValueError(
                    "--compute-disconnectivity-matrix and --compute-roi-disconnection "
                    "require --parcel-atlases or --custom-parcellation."
                )
            if has_atlas and "parcellation_name" in opts:
                # parcellation_name is now a list for SNM
                names = opts["parcellation_name"]
                if isinstance(names, str):
                    names = [names]
                opts["parcellation_name"] = [self._validate_atlas_name(n) for n in names]

        # Validate atlas names for RD and FNM (parcel_names list)
        if "parcel_names" in self.analysis_options:
            self.analysis_options["parcel_names"] = [
                self._validate_atlas_name(n) for n in self.analysis_options["parcel_names"]
            ]

    @staticmethod
    def _validate_atlas_name(name: str) -> str:
        """Validate that an atlas name exists in the registry."""
        from lacuna.assets.parcellations import PARCELLATION_REGISTRY

        if name in PARCELLATION_REGISTRY:
            return name

        available = sorted(PARCELLATION_REGISTRY.keys())
        raise ValueError(
            f"Atlas '{name}' not found. "
            f"Available atlases: {', '.join(available[:5])}...\n"
            f"Use 'lacuna info atlases' to see all options."
        )


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Parses command-line arguments and routes to appropriate command handler.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    from lacuna.cli.parser import build_parser

    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    # Route to appropriate command handler
    if args.command == "fetch":
        return _handle_fetch_command(args)
    elif args.command == "run":
        return _handle_run_command(args)
    elif args.command == "collect":
        return _handle_collect_command(args)
    elif args.command == "info":
        return _handle_info_command(args)
    elif args.command == "bidsify":
        return _handle_bidsify_command(args)
    elif args.command == "parcellate":
        return _handle_parcellate_command(args)
    elif args.command == "tutorial":
        return _handle_tutorial_command(args)
    elif args.command == "check":
        return _handle_check_command(args)
    elif args.command == "prepare":
        return _handle_prepare_command(args)
    else:
        # No command specified - show help
        parser.print_help()
        return EXIT_SUCCESS


def _handle_fetch_command(args: Namespace) -> int:
    """Handle the fetch subcommand."""
    from lacuna.cli.fetch_cmd import handle_fetch_command

    return handle_fetch_command(args)


def _handle_run_command(args: Namespace) -> int:
    """Handle the run subcommand."""
    if not args.analysis:
        # No analysis specified - show run help
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        # Parse just "run" to get the run subparser
        parser.parse_args(["run", "--help"])
        return EXIT_SUCCESS

    # Suppress nilearn warnings
    import warnings

    warnings.filterwarnings("ignore", module="nilearn")
    warnings.filterwarnings("ignore", message=".*Non-finite values.*")
    warnings.filterwarnings("ignore", message=".*Casting data from.*")

    try:
        config = RunConfig.from_args(args)
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return EXIT_INVALID_ARGS

    # Configure logging
    _setup_logging(config.log_level)

    logger.info("Lacuna CLI starting")
    logger.info(f"Input: {config.bids_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Analysis: {config.analysis}")

    try:
        return _run_analysis_workflow(config)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if config.verbose_count >= 2:
            import traceback

            traceback.print_exc()
        return EXIT_GENERAL_ERROR


def _handle_collect_command(args: Namespace) -> int:
    """Handle the collect subcommand."""
    from lacuna.io.bids import BidsError, aggregate_parcelstats

    # Setup logging
    log_level = max(25 - 5 * getattr(args, "verbose_count", 0), 10)
    _setup_logging(log_level)

    derivatives_dir = args.derivatives_dir
    output_dir = getattr(args, "output_dir", None) or derivatives_dir
    overwrite = getattr(args, "overwrite", False)
    pattern = getattr(args, "pattern", None)

    # Build glob pattern - if user provides one, wrap it for either suffix.
    if pattern:
        if "_parcelstats.tsv" in pattern or "_profilestats.tsv" in pattern:
            glob_patterns = [pattern]
        else:
            stem = pattern.strip("*")
            glob_patterns = [
                f"*{stem}*_parcelstats.tsv",
                f"*{stem}*_profilestats.tsv",
            ]
    else:
        glob_patterns = ["*_parcelstats.tsv", "*_profilestats.tsv"]

    logger.info("Running collect (group-level aggregation)")
    logger.info(f"Scanning derivatives directory: {derivatives_dir}")
    if output_dir != derivatives_dir:
        logger.info(f"Output directory: {output_dir}")
    logger.info(f"Patterns: {', '.join(glob_patterns)}")

    # Pre-scan to inform user what was found
    from lacuna.io.bids import _extract_output_type

    matched_files = [
        f
        for gp in glob_patterns
        for f in Path(derivatives_dir).rglob(gp)
        if not f.name.startswith("group_")
    ]
    n_subjects = len(
        {
            f.parent.parent.parent.name
            for f in matched_files
            if f.parent.parent.parent.name.startswith("sub-")
        }
    )
    n_output_types = len({_extract_output_type(f.name) for f in matched_files})
    logger.info(
        f"Found {len(matched_files)} file(s) across {n_subjects} subject(s) "
        f"and {n_output_types} output type(s)"
    )

    try:
        created_files: dict = {}
        for gp in glob_patterns:
            try:
                created_files.update(
                    aggregate_parcelstats(
                        derivatives_dir=derivatives_dir,
                        output_dir=output_dir,
                        pattern=gp,
                        overwrite=overwrite,
                    )
                )
            except BidsError:
                # No files matched this pattern — try the next one.
                continue
        if not created_files and not matched_files:
            raise BidsError(
                f"No parcelstats/profilestats files found in {derivatives_dir}"
            )

        if not created_files:
            if not overwrite:
                logger.info(
                    "No new files created (all outputs already exist). Use --overwrite to replace."
                )
            else:
                logger.warning("No parcelstats files found to aggregate")
            return EXIT_SUCCESS

        logger.info(f"Created {len(created_files)} group-level TSV file(s):")
        for _output_type, path in created_files.items():
            logger.info(f"  - {path.name}")

        return EXIT_SUCCESS

    except BidsError as e:
        logger.error(f"Collect failed: {e}")
        return EXIT_ANALYSIS_ERROR
    except Exception as e:
        logger.error(f"Unexpected error during collect: {e}")
        if getattr(args, "verbose_count", 0) >= 2:
            import traceback

            traceback.print_exc()
        return EXIT_GENERAL_ERROR


def _handle_info_command(args: Namespace) -> int:
    """Handle the info subcommand."""
    topic = args.topic

    if topic == "atlases":
        return _show_atlases_info()
    elif topic == "connectomes":
        return _show_connectomes_info()

    return EXIT_SUCCESS


def _show_atlases_info() -> int:
    """Display information about available atlases."""
    from lacuna.assets.parcellations import list_parcellations
    from lacuna.data import get_atlas_citation

    atlases = list_parcellations()

    print("\nAvailable Brain Parcellations (Atlases)")
    print("=" * 60)

    if not atlases:
        print("  No atlases registered.")
        print("\n  Use 'lacuna fetch' to download connectomes which include atlases.")
        return EXIT_SUCCESS

    # Group by type
    combined = [a for a in atlases if "tian" in a.name and "schaefer" in a.name]
    schaefer = [a for a in atlases if a.name.startswith("schaefer") and a not in combined]
    tian = [a for a in atlases if a.name.startswith("tian") and a not in combined]
    other = [a for a in atlases if a not in schaefer + tian + combined]

    def print_atlas_group(title: str, atlas_list: list, citation_key: str | None = None):
        if not atlas_list:
            return
        print(f"\n{title}:")
        for atlas in sorted(atlas_list, key=lambda x: x.name):
            space = getattr(atlas, "space", "unknown")
            resolution = getattr(atlas, "resolution", "?")
            print(f"  {atlas.name:<45} ({space}, {resolution}mm)")
        if citation_key:
            citation = get_atlas_citation(citation_key)
            if not citation.startswith("No citation"):
                print("\n  Citation:")
                for line in citation.strip().splitlines():
                    print(f"    {line}")

    # Use well-known citation keys (these match entries in get_atlas_citation)
    schaefer_key = "tpl-MNI152NLin6Asym_res-01_atlas-Schaefer2018_desc-parcels100networks7_dseg"
    tian_key = "tpl-MNI152NLin6Asym_res-01_atlas-Tian2020_desc-parcels16_dseg"

    print_atlas_group(
        "Schaefer Cortical Parcellations", schaefer, schaefer_key if schaefer else None
    )
    print_atlas_group("Tian Subcortical Parcellations", tian, tian_key if tian else None)
    print_atlas_group("Combined Cortical + Subcortical", combined)
    print_atlas_group("Other Parcellations", other)

    print("\n" + "=" * 60)
    print(f"Total: {len(atlases)} atlas(es) available")
    print()

    return EXIT_SUCCESS


def _show_connectomes_info() -> int:
    """Display information about available connectomes."""
    from lacuna.assets.connectomes import (
        list_functional_connectomes,
        list_structural_connectomes,
    )
    from lacuna.io.downloaders import CONNECTOME_SOURCES

    func_connectomes = list_functional_connectomes()
    struct_connectomes = list_structural_connectomes()

    print("\nRegistered Connectomes")
    print("=" * 60)

    print("\nFunctional Connectomes:")
    if func_connectomes:
        for func_conn in func_connectomes:
            print(
                f"  {func_conn.name:<30} (space={func_conn.space}, resolution={func_conn.resolution}mm)"
            )
    else:
        print("  None registered. Use 'lacuna fetch gsp1000' to download GSP1000.")

    print("\nStructural Connectomes:")
    if struct_connectomes:
        for struct_conn in struct_connectomes:
            print(f"  {struct_conn.name:<30} (space={struct_conn.space})")
    else:
        print("  None registered. Use 'lacuna fetch dtor985' or 'lacuna fetch hcp1065'.")

    print("\n" + "=" * 60)
    print("\nFetchable Connectomes (use 'lacuna fetch <name>'):")
    for name, source in CONNECTOME_SOURCES.items():
        print(f"\n  {name:<8} - {source.display_name} (~{source.estimated_size_gb:.0f}GB)")
        space_note = source.space
        if name == "hcp1065":
            space_note += " (native: MNI152NLin2009aAsym)"
        print(f"             {source.n_subjects} subjects, {space_note} space")
        if source.citation:
            print("             Citation:")
            for line in source.citation.strip().splitlines():
                print(f"               {line}")
    print()

    return EXIT_SUCCESS


def _handle_parcellate_command(args: Namespace) -> int:
    """Handle `lacuna parcellate` (modality dispatch)."""
    modality = getattr(args, "modality", None)
    if modality == "functional":
        try:
            from lacuna.prepare.parcellate import run_parcellate_functional_cli
        except ImportError:
            print(
                "Error: 'lacuna parcellate --modality functional' is not yet implemented.",
                file=sys.stderr,
            )
            return EXIT_GENERAL_ERROR
        return run_parcellate_functional_cli(args)
    if modality == "structural":
        print(
            "Error: 'lacuna parcellate --modality structural' is not yet implemented.",
            file=sys.stderr,
        )
        return EXIT_GENERAL_ERROR
    print(f"Error: unknown modality {modality!r}", file=sys.stderr)
    return EXIT_GENERAL_ERROR


def _handle_bidsify_command(args: Namespace) -> int:
    """Handle the `bidsify` subcommand."""
    from lacuna.io.bidsify import bidsify

    try:
        input_dir = args.input_dir
        output_dir = args.output_dir
        space = args.space
        session = getattr(args, "session", None)
        label = getattr(args, "label", None)
        verbose = getattr(args, "verbose", False)

        if verbose:
            print(f"Converting NIfTI files from {input_dir} to BIDS format...")
            print(f"Output directory: {output_dir}")
            print(f"Space: {space}")
            if session:
                print(f"Session: {session}")
            if label:
                print(f"Label: {label}")

        result_dir = bidsify(
            input_dir=input_dir,
            output_dir=output_dir,
            space=space,
            session=session,
            label=label,
        )

        if verbose:
            print(f"\nBIDS dataset created at: {result_dir}")
            # Count subjects
            subjects = list(result_dir.glob("sub-*"))
            print(f"Converted {len(subjects)} subject(s)")

        return EXIT_SUCCESS

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_IO_ERROR
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return EXIT_GENERAL_ERROR


def _handle_tutorial_command(args: Namespace) -> int:
    """Handle the tutorial subcommand."""
    from pathlib import Path

    from lacuna.data.tutorials import setup_tutorial_data, setup_tutorial_raw_masks

    # Get output directory
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        output_dir = Path.cwd() / "lacuna_tutorial"
    else:
        output_dir = Path(output_dir)

    force = getattr(args, "force", False)
    raw = getattr(args, "raw", False)

    print(f"\nSetting up tutorial data at: {output_dir}")

    try:
        if raw:
            result_dir = setup_tutorial_raw_masks(output_dir, overwrite=force)
        else:
            result_dir = setup_tutorial_data(output_dir, overwrite=force)
        print(f"✓ Tutorial data copied to: {result_dir}")
        return EXIT_SUCCESS

    except FileExistsError:
        print(f"Error: Directory already exists: {output_dir}")
        print("       Use --force to overwrite, or choose a different location.")
        return EXIT_GENERAL_ERROR
    except Exception as e:
        print(f"Error setting up tutorial data: {e}")
        return EXIT_GENERAL_ERROR


def _check_input_mask(filepath: Path, mask_space: str | None) -> dict:
    """Validate a single input mask file.

    Returns a dict with keys: path, status ("ok"|"warning"|"error"),
    issues (list[str]), shape, voxel_size.
    """
    import nibabel as nib
    import numpy as np

    result: dict = {
        "path": filepath,
        "status": "ok",
        "issues": [],
        "shape": None,
        "voxel_size": None,
    }

    # 1. Load file
    try:
        img = nib.load(filepath)
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"cannot load: {e}")
        return result

    result["shape"] = img.shape[:3]
    result["voxel_size"] = tuple(round(float(v), 2) for v in img.header.get_zooms()[:3])

    # 2. 3D check
    if len(img.shape) != 3:
        result["status"] = "error"
        result["issues"].append(f"not 3D (shape: {img.shape})")
        return result

    data = np.asarray(img.dataobj)

    # 3. Binary check
    unique = np.unique(data)
    if not np.all(np.isin(unique, [0, 1])):
        result["status"] = "error"
        vals = ", ".join(str(v) for v in unique[:6])
        if len(unique) > 6:
            vals += ", ..."
        result["issues"].append(f"not binary (values: {vals})")

    # 4. Empty check
    n_nonzero = int(np.count_nonzero(data))
    if n_nonzero == 0:
        result["status"] = "error"
        result["issues"].append("empty mask (0 non-zero voxels)")
    elif n_nonzero < 10:
        # 6. Small lesion warning
        if result["status"] == "ok":
            result["status"] = "warning"
        result["issues"].append(f"very small ({n_nonzero} voxels)")

    # 5. Space detection and consistency
    from lacuna.core.spaces import detect_space_from_header
    from lacuna.io.bids import _parse_bids_entities

    entities = _parse_bids_entities(filepath.name)
    filename_space = entities.get("space")
    detected = detect_space_from_header(img)
    detected_space = detected[0] if detected else None

    if not filename_space and detected_space is None and mask_space is None:
        if result["status"] == "ok":
            result["status"] = "error"
        result["issues"].append("space not detectable (no space- in filename, unknown affine)")
    elif filename_space and detected_space:
        # Check that the space declared in the filename matches the affine/shape
        from lacuna.core.spaces import spaces_are_equivalent

        if not spaces_are_equivalent(filename_space, detected_space):
            if result["status"] == "ok":
                result["status"] = "error"
            result["issues"].append(
                f"space mismatch: filename says '{filename_space}' but affine matches '{detected_space}'"
            )

    # Record the effective space for cross-dataset consistency checking
    result["space"] = filename_space or detected_space or mask_space

    return result


def _discover_mask_files(
    bids_dir: Path,
    pattern: str,
    participant_label: list[str] | None,
) -> list[Path]:
    """Discover mask files in a BIDS directory, applying filters."""
    import fnmatch as _fnmatch
    import re

    suffix = "_mask.nii.gz"
    all_files = sorted(bids_dir.rglob(f"*{suffix}"))

    matching_files = []
    for fp in all_files:
        stem = fp.name[:-7] if fp.name.endswith(".nii.gz") else fp.name
        if (
            _fnmatch.fnmatch(stem, f"*{pattern}*")
            or _fnmatch.fnmatch(stem, pattern)
            or _fnmatch.fnmatch(stem, f"{pattern}*")
            or _fnmatch.fnmatch(stem, f"*{pattern}")
        ):
            matching_files.append(fp)

    if participant_label:
        normalized = {(s[4:] if s.startswith("sub-") else s) for s in participant_label}
        matching_files = [
            fp
            for fp in matching_files
            if (m := re.search(r"sub-([^/_]+)", str(fp))) and m.group(1) in normalized
        ]

    return matching_files


def _handle_check_input(args: Namespace) -> int:
    """Handle the 'lacuna check input' subcommand."""
    from collections import Counter

    bids_dir: Path = args.bids_dir
    participant_label: list[str] | None = getattr(args, "participant_label", None)
    session_id: list[str] | None = getattr(args, "session_id", None)
    pattern: str | None = getattr(args, "pattern", None)
    mask_space: str | None = getattr(args, "mask_space", None)
    quiet: bool = getattr(args, "quiet", False)
    output_file: Path | None = getattr(args, "output_file", None)

    if not bids_dir.exists():
        print(f"Error: BIDS directory does not exist: {bids_dir}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    bids_pattern = _build_pattern(session_id, pattern)
    mask_files = _discover_mask_files(bids_dir, bids_pattern, participant_label)

    if not mask_files:
        print("No mask files found in BIDS dataset.", file=sys.stderr)
        return EXIT_BIDS_ERROR

    if not quiet:
        print(f"\nChecking {len(mask_files)} input mask(s) in {bids_dir}...\n")

    # Check each mask
    from tqdm import tqdm

    results = [
        _check_input_mask(fp, mask_space)
        for fp in tqdm(mask_files, desc="Checking masks", unit="mask", disable=quiet)
    ]

    # 7. Consistency check: flag shapes that differ from the majority
    shape_counts: Counter = Counter()
    for r in results:
        if r["shape"] is not None and r["voxel_size"] is not None:
            shape_counts[(r["shape"], r["voxel_size"])] += 1

    if shape_counts:
        majority_key = shape_counts.most_common(1)[0][0]
        for r in results:
            key = (r["shape"], r["voxel_size"])
            if key != (None, None) and key != majority_key:
                if r["status"] == "ok":
                    r["status"] = "warning"
                r["issues"].append(
                    f"dimensions {r['shape']} differ from majority {majority_key[0]}"
                )

    # 8. Cross-dataset space consistency: warn if masks have different detected spaces
    space_counts: Counter = Counter()
    for r in results:
        if r.get("space"):
            space_counts[r["space"]] += 1

    if len(space_counts) > 1:
        majority_space = space_counts.most_common(1)[0][0]
        for r in results:
            if r.get("space") and r["space"] != majority_space:
                if r["status"] == "ok":
                    r["status"] = "warning"
                r["issues"].append(f"space '{r['space']}' differs from majority '{majority_space}'")

    # Collect error paths
    error_paths = [str(r["path"].relative_to(bids_dir)) for r in results if r["status"] == "error"]

    # Write to file if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(error_paths) + ("\n" if error_paths else ""))

    if quiet:
        for p in error_paths:
            print(p)
        return EXIT_SUCCESS if not error_paths else EXIT_GENERAL_ERROR

    # Human-readable table
    max_path = max(
        len("File"),
        max(len(str(r["path"].relative_to(bids_dir))) for r in results),
    )
    max_path = min(max_path, 70)  # cap column width
    sep_width = max_path + 12 + 30

    print(f"{'File':<{max_path}}  {'Status':<9}Issues")
    print("-" * sep_width)

    for r in results:
        rel = str(r["path"].relative_to(bids_dir))
        if len(rel) > max_path:
            rel = "..." + rel[-(max_path - 3) :]
        status = r["status"].upper() if r["status"] != "ok" else "ok"
        issues = "; ".join(r["issues"]) if r["issues"] else ""
        print(f"{rel:<{max_path}}  {status:<9}{issues}")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_warn = sum(1 for r in results if r["status"] == "warning")
    n_err = sum(1 for r in results if r["status"] == "error")

    print(f"\n{'=' * sep_width}")
    print(f"Summary: {n_ok} ok, {n_warn} warnings, {n_err} errors")
    print(f"{'=' * sep_width}")

    # Problems summary: list only files with issues so they're easy to find
    problem_results = [r for r in results if r["status"] != "ok"]
    if problem_results:
        print(f"\nProblems ({len(problem_results)}):\n")
        for r in problem_results:
            rel = str(r["path"].relative_to(bids_dir))
            status = r["status"].upper()
            issues = "; ".join(r["issues"])
            print(f"  [{status}] {rel}")
            print(f"         {issues}")
        print()

    if output_file and error_paths:
        print(f"Error file paths written to: {output_file}\n")

    return EXIT_SUCCESS if n_err == 0 else EXIT_GENERAL_ERROR


def _is_output_empty(filepath: Path, analysis_type: str) -> bool:
    """Check whether an output file contains only zeros / empty-mask data.

    For NIfTI files (fnm, snm): reads the companion JSON sidecar and checks
    Metadata.empty_mask. Falls back to loading the NIfTI if no sidecar exists.

    For TSV files (ld): reads the file and checks if all numeric values are 0.

    Parameters
    ----------
    filepath : Path
        Path to the output file to check.
    analysis_type : str
        Analysis type: 'ld', 'localdamage', 'fnm', 'functionalnetworkmapping',
        'snm', or 'structuralnetworkmapping'.

    Returns
    -------
    bool
        True if output is empty (all zeros), False otherwise.
    """
    import json

    norm = analysis_type.lower()

    if norm in ("ld", "ld", "localdamage"):
        # Read TSV and check if all numeric columns are zero
        try:
            import pandas as pd

            df = pd.read_csv(filepath, sep="\t")
            numeric_cols = df.select_dtypes(include="number")
            if numeric_cols.empty:
                return True  # No numeric data
            return (numeric_cols == 0).all().all()
        except Exception:
            return False  # If we can't read it, assume non-empty

    else:  # FNM or SNM -- try sidecar first
        sidecar_path = filepath.with_suffix("").with_suffix(".json")
        if sidecar_path.exists():
            try:
                with open(sidecar_path) as f:
                    data = json.load(f)
                meta = data.get("Metadata", {})
                if meta.get("empty_mask") is True:
                    return True
                # If sidecar exists but no empty_mask key, assume non-empty
                return False
            except Exception:
                pass  # Fall through to NIfTI loading

        # No sidecar or failed to read -- fall back to loading NIfTI
        try:
            import nibabel as nib
            import numpy as np

            img = nib.load(filepath)
            return not np.any(img.get_fdata() > 0)
        except Exception:
            return False  # If we can't read it, assume non-empty


def _check_subject_complete(
    anat_dir: Path,
    analysis: str,
    parcel_atlases: list[str] | None,
    check_content: bool = False,
    label: str | None = None,
) -> tuple[str, list[str]]:
    """Check whether expected output files exist for a subject/session/label.

    Parameters
    ----------
    anat_dir : Path
        Anatomy directory for the subject/session.
    analysis : str
        Analysis type: 'ld', 'fnm', 'snm', etc.
    parcel_atlases : list[str] | None
        For RD: list of expected atlas names. If None, any parcelstats file counts as complete.
    check_content : bool
        If True, inspect file content to detect empty (all-zero) outputs.
    label : str | None
        Lesion label (e.g. 'WMH', 'acuteinfarct'). When provided, glob patterns
        are narrowed to only match outputs for this specific label.

    Returns
    -------
    tuple of (status, missing)
        status : "complete" | "empty" | "missing"
        missing : list of descriptions of what was not found
    """
    norm = analysis.lower()
    label_glob = f"*label-{label}_*" if label else "*"

    if not anat_dir.exists():
        if norm in ("ld", "ld", "localdamage"):
            sentinel = f"{label_glob}method-ld*parcelstats.tsv"
        elif norm in ("fnm", "functionalnetworkmapping"):
            sentinel = f"{label_glob}method-fnm*desc-rmap*.nii.gz"
        elif norm in ("snm", "structuralnetworkmapping"):
            sentinel = f"{label_glob}method-snm*desc-disconnectionpct*.nii.gz"
        elif norm in ("afnm", "acceleratedfunctionalnetworkmapping"):
            sentinel = f"{label_glob}method-afnm*parcelstats.tsv"
        else:
            sentinel = f"<unknown analysis '{analysis}'>"
        return "missing", [sentinel]

    if norm in ("ld", "ld", "localdamage"):
        all_matches = list(anat_dir.glob(f"{label_glob}method-ld*parcelstats.tsv"))
        if parcel_atlases:
            missing = []
            for atlas in parcel_atlases:
                atlas_fragment = atlas.replace("_", "").lower()
                hits = [f for f in all_matches if atlas_fragment in f.name.lower()]
                if not hits:
                    missing.append(atlas)
            if missing:
                return "missing", missing
            # All atlases found -- check content if requested
            if check_content and all(_is_output_empty(f, norm) for f in all_matches):
                return "empty", []
            return "complete", []
        else:
            if not all_matches:
                return "missing", [f"{label_glob}method-ld*parcelstats.tsv"]
            # Files exist -- check content if requested
            if check_content and all(_is_output_empty(f, norm) for f in all_matches):
                return "empty", []
            return "complete", []

    elif norm in ("fnm", "functionalnetworkmapping"):
        hits = list(anat_dir.glob(f"{label_glob}method-fnm*desc-rmap*.nii.gz"))
        if not hits:
            return "missing", [f"{label_glob}method-fnm*desc-rmap*.nii.gz"]
        # Files exist -- check content if requested
        if check_content and all(_is_output_empty(f, norm) for f in hits):
            return "empty", []
        return "complete", []

    elif norm in ("snm", "structuralnetworkmapping"):
        hits = list(anat_dir.glob(f"{label_glob}method-snm*desc-disconnectionpct*.nii.gz"))
        if not hits:
            return "missing", [f"{label_glob}method-snm*desc-disconnectionpct*.nii.gz"]
        # Files exist -- check content if requested
        if check_content and all(_is_output_empty(f, norm) for f in hits):
            return "empty", []
        return "complete", []

    elif norm in ("afnm", "acceleratedfunctionalnetworkmapping"):
        hits = list(anat_dir.glob(f"{label_glob}method-afnm*parcelstats.tsv"))
        if not hits:
            return "missing", [f"{label_glob}method-afnm*parcelstats.tsv"]
        if check_content and all(_is_output_empty(f, norm) for f in hits):
            return "empty", []
        return "complete", []

    return "missing", [f"unknown analysis '{analysis}'"]


def _discover_bids_subjects(
    bids_dir: Path,
    pattern: str,
    participant_label: list[str] | None,
) -> list[dict]:
    """Discover subjects in a BIDS directory without loading NIfTI images.

    Returns a list of metadata dicts with subject_id, session_id, and label.
    Deduplicates by (subject_id, session_id, label) — one entry per unique
    combination, so subjects with multiple lesion labels get separate entries.
    """
    import fnmatch as _fnmatch
    import re

    from lacuna.io.bids import _parse_bids_entities

    suffix = "_mask.nii.gz"
    all_files = list(bids_dir.rglob(f"*{suffix}"))

    # Filter by filename pattern
    matching_files = []
    for fp in all_files:
        stem = fp.name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        if (
            _fnmatch.fnmatch(stem, f"*{pattern}*")
            or _fnmatch.fnmatch(stem, pattern)
            or _fnmatch.fnmatch(stem, f"{pattern}*")
            or _fnmatch.fnmatch(stem, f"*{pattern}")
        ):
            matching_files.append(fp)

    # Filter by participant label
    if participant_label:
        normalized = {(s[4:] if s.startswith("sub-") else s) for s in participant_label}
        matching_files = [
            fp
            for fp in matching_files
            if (m := re.search(r"sub-([^/_]+)", str(fp))) and m.group(1) in normalized
        ]

    # Deduplicate by (subject_id, session_id, label)
    seen: set[tuple] = set()
    results = []
    for fp in sorted(matching_files):
        meta = _parse_bids_entities(fp.name)
        sub_id = meta.get("subject_id")
        ses_id = meta.get("session_id")
        label = meta.get("label")
        key = (sub_id, ses_id, label)
        if sub_id and key not in seen:
            seen.add(key)
            results.append({"subject_id": sub_id, "session_id": ses_id, "label": label})

    return results


def _handle_check_command(args: Namespace) -> int:
    """Handle the check subcommand."""
    analysis: str | None = getattr(args, "analysis", None)
    if not analysis:
        from lacuna.cli.parser import build_parser

        build_parser().parse_args(["check", "--help"])
        return EXIT_SUCCESS

    # Input check is a separate path — no output_dir needed
    if analysis == "input":
        return _handle_check_input(args)

    bids_dir: Path = args.bids_dir
    output_dir: Path = args.output_dir
    participant_label: list[str] | None = getattr(args, "participant_label", None)
    session_id: list[str] | None = getattr(args, "session_id", None)
    pattern: str | None = getattr(args, "pattern", None)
    parcel_atlases: list[str] | None = getattr(args, "parcel_atlases", None)
    quiet: bool = getattr(args, "quiet", False)
    output_file: Path | None = getattr(args, "output_file", None)
    check_content: bool = getattr(args, "check_content", False)

    if not bids_dir.exists():
        print(f"Error: BIDS directory does not exist: {bids_dir}", file=sys.stderr)
        return EXIT_INVALID_ARGS

    bids_pattern = _build_pattern(session_id, pattern)
    subject_metas = _discover_bids_subjects(bids_dir, bids_pattern, participant_label)

    if not subject_metas:
        print("No subjects found in BIDS dataset.", file=sys.stderr)
        return EXIT_BIDS_ERROR

    if not quiet:
        print(
            f"\nChecking {analysis} outputs in {output_dir} for {len(subject_metas)} subject/label(s)...\n"
        )

    # Check each subject/label combination
    rows: list[dict] = []
    for meta in subject_metas:
        sub_id: str = meta["subject_id"]
        ses_id: str | None = meta["session_id"]
        label: str | None = meta.get("label")

        anat_dir = output_dir / sub_id
        if ses_id:
            anat_dir = anat_dir / ses_id
        anat_dir = anat_dir / "anat"

        status, missing = _check_subject_complete(
            anat_dir, analysis, parcel_atlases, check_content, label=label
        )
        rows.append(
            {
                "subject_id": sub_id,
                "session_id": ses_id,
                "label": label,
                "status": status,
                "missing": missing,
            }
        )

    # Deduplicated sorted lists of missing/empty bare subject IDs
    missing_subject_ids = sorted(
        {r["subject_id"].removeprefix("sub-") for r in rows if r["status"] == "missing"}
    )
    empty_subject_ids = sorted(
        {r["subject_id"].removeprefix("sub-") for r in rows if r["status"] == "empty"}
    )

    # Write to file if requested (always, regardless of --quiet)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            "\n".join(missing_subject_ids) + ("\n" if missing_subject_ids else "")
        )

    if quiet:
        for sub_id in missing_subject_ids:
            print(sub_id)
        if check_content:
            for sub_id in empty_subject_ids:
                print(f"{sub_id}\tempty")
        if missing_subject_ids:
            return EXIT_GENERAL_ERROR
        if empty_subject_ids:
            return 2  # EXIT_EMPTY_OUTPUTS
        return EXIT_SUCCESS

    # Human-readable table
    has_sessions = any(r["session_id"] for r in rows)
    has_labels = any(r.get("label") for r in rows)
    sub_width = max(len("Subject"), max(len(r["subject_id"]) for r in rows)) + 2
    label_width = (
        max(len("Label"), max((len(r.get("label") or "-") for r in rows), default=5)) + 2
        if has_labels
        else 0
    )
    sep_width = sub_width + (12 if has_sessions else 0) + label_width + 22

    header = f"{'Subject':<{sub_width}}"
    if has_sessions:
        header += f"{'Session':<12}"
    if has_labels:
        header += f"{'Label':<{label_width}}"
    header += "Status"
    print(header)
    print("-" * sep_width)

    for r in rows:
        if r["status"] == "complete":
            status_str = "complete"
        elif r["status"] == "empty":
            status_str = "EMPTY  (all-zero output)"
        else:
            detail = ", ".join(r["missing"])
            status_str = "MISSING" + (f"  ({detail})" if detail else "")

        line = f"{r['subject_id']:<{sub_width}}"
        if has_sessions:
            line += f"{(r['session_id'] or '-'):<12}"
        if has_labels:
            line += f"{(r.get('label') or '-'):<{label_width}}"
        line += status_str
        print(line)

    n_complete = sum(1 for r in rows if r["status"] == "complete")
    n_empty = sum(1 for r in rows if r["status"] == "empty")
    n_missing = sum(1 for r in rows if r["status"] == "missing")
    n_total = len(rows)
    print(f"\n{'=' * sep_width}")
    summary = f"Summary: {n_complete} / {n_total} complete"
    if n_missing > 0:
        summary += f", {n_missing} missing"
    if n_empty > 0:
        summary += f", {n_empty} empty"
    print(summary)
    print(f"{'=' * sep_width}\n")

    if missing_subject_ids:
        # Detail what is missing per subject
        print("Missing outputs:")
        for r in rows:
            if r["status"] != "missing":
                continue
            sub = r["subject_id"]
            parts = [sub]
            if r.get("session_id"):
                parts.append(r["session_id"])
            if r.get("label"):
                parts.append(f"label-{r['label']}")
            detail = ", ".join(r["missing"])
            line = " / ".join(parts)
            if detail:
                line += f":  {detail}"
            print(f"  {line}")

        label_str = " ".join(missing_subject_ids)
        print("\nRerun missing subjects:")
        print(f"  lacuna run {analysis} {bids_dir} {output_dir} --participant-label {label_str}\n")
        if output_file:
            print(f"Missing subject IDs written to: {output_file}\n")
        return EXIT_GENERAL_ERROR

    if empty_subject_ids:
        print(f"Note: {len(empty_subject_ids)} subject(s) produced empty (all-zero) outputs.")
        print(
            "This typically means the input mask had no overlap with the analysis atlas/network.\n"
        )
        return 2  # EXIT_EMPTY_OUTPUTS

    print("All subjects are complete.\n")
    return EXIT_SUCCESS


def _register_connectome_from_path(
    analysis_options: dict[str, Any], analysis_class_name: str
) -> None:
    """Register a connectome from the provided --connectome-path.

    Users provide paths to connectomes via --connectome-path (after downloading
    with 'lacuna fetch'). This function validates the path and registers it
    so the analysis can use it.

    Parameters
    ----------
    analysis_options : dict
        The analysis options dictionary (modified in place).
    analysis_class_name : str
        The name of the analysis class being run.

    Raises
    ------
    FileNotFoundError
        If the connectome path does not exist.
    ValueError
        If the path has an invalid format for the analysis type.
    """
    # Get the path from --connectome-path
    connectome_path_str = analysis_options.pop("_connectome_path", None)
    if not connectome_path_str:
        # No connectome needed for this analysis (e.g., LocalDamage)
        return

    connectome_path = Path(connectome_path_str)
    if not connectome_path.exists():
        raise FileNotFoundError(
            f"Connectome path does not exist: {connectome_path}\n\n"
            "To download a connectome:\n"
            "  lacuna fetch gsp1000    # Functional connectome\n"
            "  lacuna fetch dtor985    # Structural connectome"
        )

    # Register based on analysis type
    if analysis_class_name in (
        "StructuralNetworkMapping",
        "StructuralNeurotransmitterFingerprinting",
    ):
        from lacuna.assets.connectomes import (
            list_structural_connectomes,
            register_structural_connectome,
        )

        # Validate it's a .tck file
        if connectome_path.suffix.lower() != ".tck":
            raise ValueError(
                f"Structural network mapping requires .tck tractogram files.\n"
                f"Got: {connectome_path.name} (suffix: '{connectome_path.suffix}')\n\n"
                "Hint: Use 'lacuna fetch dtor985' to download a tractogram,\n"
                "      or convert with MRtrix3's tckconvert if needed."
            )

        # Check if already registered (avoid duplicate registration)
        registered_names = [c.name for c in list_structural_connectomes()]
        auto_name = f"cli_{connectome_path.stem}"

        if auto_name not in registered_names:
            logger.info(f"Registering structural connectome: {connectome_path.name}")
            # Try to infer space from filename or default to MNI152NLin2009cAsym
            space = "MNI152NLin2009cAsym"  # Common default for tractograms
            if "MNI152NLin6Asym" in str(connectome_path):
                space = "MNI152NLin6Asym"
            elif "MNI152NLin2009bAsym" in str(connectome_path):
                space = "MNI152NLin2009bAsym"

            register_structural_connectome(
                name=auto_name,
                space=space,
                tractogram_path=connectome_path,
                description=f"Registered from CLI: {connectome_path}",
            )

        analysis_options["connectome_name"] = auto_name

    elif analysis_class_name in (
        "FunctionalNetworkMapping",
        "FunctionalNeurotransmitterFingerprinting",
    ):
        from lacuna.assets.connectomes import (
            list_functional_connectomes,
            register_functional_connectome,
        )

        # Validate it's an HDF5 file or directory
        valid_extensions = {".h5", ".hdf5"}
        is_hdf5 = connectome_path.suffix.lower() in valid_extensions
        is_directory = connectome_path.is_dir()

        if not is_hdf5 and not is_directory:
            raise ValueError(
                f"Functional connectomes require HDF5 files (.h5/.hdf5) or batch directories.\n"
                f"Got: {connectome_path.name} (suffix: '{connectome_path.suffix}')\n\n"
                "Hint: Use 'lacuna fetch gsp1000' to download a functional connectome."
            )

        # Check if already registered (avoid duplicate registration)
        registered_names = [c.name for c in list_functional_connectomes()]
        auto_name = f"cli_{connectome_path.stem}"

        if auto_name not in registered_names:
            logger.info(f"Registering functional connectome: {connectome_path.name}")
            # Try to infer space from filename or default to MNI152NLin6Asym
            space = "MNI152NLin6Asym"  # Common default for GSP
            if "MNI152NLin2009" in str(connectome_path):
                space = "MNI152NLin2009cAsym"

            # Infer resolution from path or default to 2mm
            resolution = 2
            if "_1mm" in str(connectome_path) or "res-01" in str(connectome_path):
                resolution = 1

            register_functional_connectome(
                name=auto_name,
                space=space,
                resolution=resolution,
                data_path=connectome_path,
                description=f"Registered from CLI: {connectome_path}",
            )

        analysis_options["connectome_name"] = auto_name


def _register_custom_parcellations(
    analysis_options: dict[str, Any], analysis_class_name: str
) -> None:
    """Register custom parcellations from --custom-parcellation arguments.

    Registers each custom parcellation in the global registry and appends the
    registered name to the appropriate parcel_names/parcellation_name list so
    the analysis picks it up.

    Parameters
    ----------
    analysis_options : dict
        The analysis options dictionary (modified in place).
    analysis_class_name : str
        The name of the analysis class being run.
    """
    custom_parcellations = analysis_options.pop("custom_parcellation", None)
    if not custom_parcellations:
        return

    import nibabel as nib

    from lacuna.assets.parcellations.registry import register_parcellation_from_files

    registered_names = []
    for name, nifti_path, labels_path, space in custom_parcellations:
        nifti_path = Path(nifti_path)
        if not nifti_path.exists():
            raise FileNotFoundError(f"Custom parcellation file not found: {nifti_path}")

        # Detect resolution from voxel size
        img = nib.load(nifti_path)
        voxel_sizes = img.header.get_zooms()[:3]
        resolution = round(min(voxel_sizes))

        logger.info(f"Registering custom parcellation: {name} (space={space}, res={resolution}mm)")
        register_parcellation_from_files(
            name=name,
            parcellation_path=nifti_path,
            labels_path=labels_path,
            space=space,
            resolution=resolution,
            description=f"Custom parcellation registered from CLI: {nifti_path.name}",
        )
        registered_names.append(name)

    # Append to the appropriate parcel names key
    if analysis_class_name == "StructuralNetworkMapping":
        existing = analysis_options.get("parcellation_name") or []
        if isinstance(existing, str):
            existing = [existing]
        analysis_options["parcellation_name"] = existing + registered_names
    else:
        # LocalDamage and FNM post-processing both use parcel_names
        existing = analysis_options.get("parcel_names") or []
        analysis_options["parcel_names"] = existing + registered_names


def _run_analysis_workflow(config: RunConfig) -> int:
    """Run the analysis workflow based on configuration."""
    from lacuna import SubjectData
    from lacuna.io import load_bids_dataset

    # Ensure directories exist
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.tmp_dir:
        config.tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("LACUNA_TMP_DIR", str(config.tmp_dir.resolve()))

    # Map analysis names to class names
    analysis_name_map = {
        "ld": "LocalDamage",
        "localdamage": "LocalDamage",
        "ld": "LocalDamage",
        "fnm": "FunctionalNetworkMapping",
        "functionalnetworkmapping": "FunctionalNetworkMapping",
        "snm": "StructuralNetworkMapping",
        "structuralnetworkmapping": "StructuralNetworkMapping",
        "afnm": "AcceleratedFunctionalNetworkMapping",
        "acceleratedfunctionalnetworkmapping": "AcceleratedFunctionalNetworkMapping",
        "lntf": "LocalNeurotransmitterFingerprinting",
        "localneurotransmitterfingerprinting": "LocalNeurotransmitterFingerprinting",
        "sntf": "StructuralNeurotransmitterFingerprinting",
        "structuralneurotransmitterfingerprinting": "StructuralNeurotransmitterFingerprinting",
        "fntf": "FunctionalNeurotransmitterFingerprinting",
        "functionalneurotransmitterfingerprinting": "FunctionalNeurotransmitterFingerprinting",
    }

    analysis_class_name = analysis_name_map.get(config.analysis.lower())
    if not analysis_class_name:
        logger.error(f"Unknown analysis: {config.analysis}")
        return EXIT_INVALID_ARGS

    # Register connectome from path for FNM/SNM analyses
    _register_connectome_from_path(config.analysis_options, analysis_class_name)

    # Register custom parcellations before building analysis steps
    _register_custom_parcellations(config.analysis_options, analysis_class_name)

    # Extract parcel_names for FNM post-processing (FNM doesn't accept parcel_names)
    fnm_parcel_names = None
    if analysis_class_name == "FunctionalNetworkMapping":
        fnm_parcel_names = config.analysis_options.pop("parcel_names", None)

    # Build analysis steps
    steps = {analysis_class_name: config.analysis_options or {}}

    # For FNM with parcel atlases, add parcel aggregation as second step
    if fnm_parcel_names:
        # Aggregate FNM output maps (r, z, t maps)
        steps["ParcelAggregation"] = {
            "source": {
                "FunctionalNetworkMapping": [
                    "rmap",
                    "zmap",
                    "tmap",
                ]
            },
            "aggregation": "mean",
            "parcel_names": fnm_parcel_names,
        }

    # Add verbose flag to analysis options
    if config.verbose and analysis_class_name in steps:
        steps[analysis_class_name]["verbose"] = True

    logger.info(f"Running analysis: {analysis_class_name}")

    try:
        if config.is_single_file:
            # Single file mode
            subject_data = SubjectData.from_nifti(
                config.bids_dir,
                space=config.space,
                resolution=None,  # Auto-detect
                metadata={"subject_id": f"sub-{config.bids_dir.stem.split('_')[0]}"},
            )
            subjects_list = [subject_data]
            logger.info("Loaded single mask file")
        else:
            # BIDS dataset mode
            pattern = _build_pattern(
                config.session_id,
                config.pattern,
            )
            subjects_dict = load_bids_dataset(
                bids_root=config.bids_dir,
                pattern=pattern,
                space=config.space,
                resolution=None,  # Auto-detect
                subjects=config.participant_label,  # Filter at file discovery level
            )

            if not subjects_dict:
                logger.error("No subjects found in BIDS dataset")
                return EXIT_BIDS_ERROR

            subjects_list = list(subjects_dict.values())

            _log_discovery_summary(subjects_list, config)

        # Process subjects
        if len(subjects_list) > 1 and config.batch_size != 1:
            # Batch processing
            result = _process_batch(subjects_list, steps, config, config.batch_size)
        else:
            # Sequential processing
            from tqdm import tqdm

            processed_count = 0
            empty_mask_subjects = []
            for subject_data in tqdm(
                subjects_list,
                desc="Processing subjects",
                disable=not config.verbose,
            ):
                if subject_data.is_empty_mask:
                    sid = subject_data.metadata.get("subject_id", "unknown")
                    empty_mask_subjects.append(sid)
                    if config.on_empty == "skip":
                        logger.info(f"Skipping empty mask: {sid}")
                        continue
                    elif config.on_empty == "error":
                        raise ValidationError(
                            f"Empty mask for {sid}: no non-zero voxels. "
                            f"Use --on-empty warn or skip to handle gracefully."
                        )
                result = _process_single_subject(subject_data, steps, config, export=True)
                if result == EXIT_SUCCESS:
                    processed_count += 1
                else:
                    logger.warning("Subject processing failed, continuing...")

            logger.info(f"Successfully processed {processed_count} subject(s)")
            if empty_mask_subjects:
                action_str = {
                    "skip": "skipped",
                    "warn": "processed with zero-valued outputs",
                    "error": "halted processing",
                }.get(config.on_empty, "processed")
                logger.warning(
                    f"{len(empty_mask_subjects)} subject(s) had empty masks "
                    f"({action_str}): {', '.join(empty_mask_subjects)}"
                )
            result = EXIT_SUCCESS if processed_count > 0 else EXIT_ANALYSIS_ERROR

        if result == EXIT_SUCCESS:
            logger.info(f"Results saved to: {config.output_dir}")
            logger.info("Lacuna CLI completed successfully")

        return result

    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        if config.verbose_count >= 2:
            import traceback

            traceback.print_exc()
        return EXIT_BIDS_ERROR


def _process_single_subject(
    subject_data: SubjectData,
    steps: dict,
    config: RunConfig,
    export: bool = True,
) -> int:
    """Process a single subject."""
    from lacuna.core.pipeline import analyze
    from lacuna.io import export_bids_derivatives

    try:
        if steps:
            result = analyze(
                data=subject_data,
                steps=steps,
                n_jobs=config.n_procs,
                show_progress=False,
                verbose=config.verbose,
            )
            # analyze with single input returns single output
            assert isinstance(result, SubjectData)
        else:
            result = subject_data

        if export:
            export_bids_derivatives(
                subject_data=result,
                output_dir=config.output_dir,
                export_lesion_mask=False,
                export_provenance=False,
                overwrite=True,
            )

        return EXIT_SUCCESS

    except Exception as e:
        subject_id = subject_data.metadata.get("subject_id", "unknown")
        logger.error(f"Failed to process {subject_id}: {e}")
        return EXIT_ANALYSIS_ERROR


def _process_batch(
    subjects_list: list,
    steps: dict,
    config: RunConfig,
    batch_size: int,
) -> int:
    """Process subjects in batches."""
    from lacuna.analysis import get_analysis
    from lacuna.batch import batch_process
    from lacuna.io import export_bids_derivatives

    n_subjects = len(subjects_list)
    actual_batch_size = n_subjects if batch_size == -1 else min(batch_size, n_subjects)

    logger.info(f"Batch processing: {n_subjects} masks in batches of {actual_batch_size}")

    # Report and optionally filter empty masks
    empty_mask_subjects = [
        s.metadata.get("subject_id", "unknown") for s in subjects_list if s.is_empty_mask
    ]
    if empty_mask_subjects:
        if config.on_empty == "skip":
            subjects_list = [s for s in subjects_list if not s.is_empty_mask]
            logger.warning(
                f"{len(empty_mask_subjects)} subject(s) with empty masks "
                f"skipped: {', '.join(empty_mask_subjects)}"
            )
            n_subjects = len(subjects_list)
            if n_subjects == 0:
                logger.error("No subjects remaining after skipping empty masks")
                return EXIT_ANALYSIS_ERROR
            actual_batch_size = n_subjects if batch_size == -1 else min(batch_size, n_subjects)
        elif config.on_empty == "error":
            msg = (
                f"{len(empty_mask_subjects)} subject(s) have empty masks: "
                f"{', '.join(empty_mask_subjects)}. "
                f"Use --on-empty warn or skip to handle gracefully."
            )
            logger.error(msg)
            raise ValidationError(msg)
        else:  # "warn"
            logger.warning(
                f"{len(empty_mask_subjects)} subject(s) have empty masks "
                f"(zero-valued outputs will be produced): {', '.join(empty_mask_subjects)}"
            )

    # Build analysis instances
    analyses = []
    for analysis_name, kwargs in steps.items():
        analysis_cls = get_analysis(analysis_name)
        kwargs = (kwargs or {}).copy()
        if "verbose" not in kwargs:
            kwargs["verbose"] = config.verbose
        analyses.append((analysis_name, analysis_cls(**kwargs)))

    processed_count = 0
    failed_count = 0

    for batch_start in range(0, n_subjects, actual_batch_size):
        batch_end = min(batch_start + actual_batch_size, n_subjects)
        batch = subjects_list[batch_start:batch_end]

        batch_num = batch_start // actual_batch_size + 1
        total_batches = (n_subjects + actual_batch_size - 1) // actual_batch_size

        if n_subjects > actual_batch_size:
            logger.info(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} masks) ---")

        try:
            current_data = batch
            for analysis_name, analysis in analyses:
                if config.verbose:
                    logger.info(f"\n─── {analysis_name} ───")

                lesion_batch_size = None if batch_size == -1 else batch_size
                current_data = batch_process(
                    inputs=current_data,
                    analysis=analysis,
                    n_jobs=config.n_procs,
                    show_progress=config.verbose,
                    strategy=None,
                    lesion_batch_size=lesion_batch_size,
                    progress_desc=analysis_name,
                )

            for result in current_data:
                try:
                    export_bids_derivatives(
                        subject_data=result,
                        output_dir=config.output_dir,
                        export_lesion_mask=False,
                        export_provenance=False,
                        overwrite=True,
                    )
                    processed_count += 1
                except Exception as e:
                    subject_id = result.metadata.get("subject_id", "unknown")
                    logger.warning(f"Failed to export {subject_id}: {e}")
                    failed_count += 1

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            failed_count += len(batch)

    if processed_count == 0:
        logger.error("No subjects were successfully processed")
        return EXIT_ANALYSIS_ERROR

    if failed_count > 0:
        logger.warning(f"Completed with {failed_count} failures out of {n_subjects} subjects")

    logger.info(f"Successfully processed {processed_count} subject(s)")
    return EXIT_SUCCESS


def _build_pattern(
    sessions: list[str] | None,
    extra_pattern: str | None,
) -> str:
    """Build a glob pattern from session and extra pattern filters.

    Note: Subject filtering is now handled by load_bids_dataset's subjects param.
    """
    pattern_parts = ["*"]  # Start with wildcard

    if sessions:
        if len(sessions) == 1:
            pattern_parts.append(f"ses-{sessions[0]}*")

    if extra_pattern:
        pattern_parts.append(extra_pattern)

    return "".join(pattern_parts) if pattern_parts else "*"


def _filter_by_participants(
    subjects_list: list,
    participant_labels: list[str],
) -> list:
    """Filter subjects list to only include specified participants.

    Parameters
    ----------
    subjects_list : list
        List of SubjectData objects.
    participant_labels : list of str
        Participant labels to keep (without 'sub-' prefix).

    Returns
    -------
    list
        Filtered list of SubjectData objects.
    """
    # Normalize labels: allow with or without 'sub-' prefix
    normalized_labels = set()
    for label in participant_labels:
        if label.startswith("sub-"):
            normalized_labels.add(label)
            normalized_labels.add(label[4:])  # without prefix
        else:
            normalized_labels.add(label)
            normalized_labels.add(f"sub-{label}")  # with prefix

    filtered = []
    for subject_data in subjects_list:
        subject_id = subject_data.metadata.get("subject_id", "")
        # Check if subject_id matches any label (with or without prefix)
        if subject_id in normalized_labels:
            filtered.append(subject_data)

    return filtered


def _format_subject_id(subject_data) -> str:
    """Format a human-readable identifier for a subject."""
    parts = []
    metadata = subject_data.metadata

    subject_id = metadata.get("subject_id", "unknown")
    parts.append(subject_id)

    session_id = metadata.get("session_id")
    if session_id:
        parts.append(session_id)

    label = metadata.get("label")
    if label:
        parts.append(label)

    return "/".join(parts)


def _log_discovery_summary(subjects_list: list, config: RunConfig) -> None:
    """Log a summary of discovered subjects."""
    if not subjects_list:
        return

    unique_subjects = set()
    unique_sessions = set()
    unique_labels = set()

    for subject_data in subjects_list:
        metadata = subject_data.metadata
        if "subject_id" in metadata:
            unique_subjects.add(metadata["subject_id"])
        if "session_id" in metadata:
            unique_sessions.add(metadata["session_id"])
        if "label" in metadata:
            unique_labels.add(metadata["label"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("DISCOVERY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total mask images: {len(subjects_list)}")
    logger.info(f"  Unique subjects:   {len(unique_subjects)}")
    if unique_sessions:
        logger.info(f"  Unique sessions:   {len(unique_sessions)}")
    if unique_labels:
        logger.info(f"  Labels:            {', '.join(sorted(unique_labels))}")

    filters = []
    if config.participant_label:
        filters.append(f"subjects={config.participant_label}")
    if config.session_id:
        filters.append(f"sessions={config.session_id}")
    if config.pattern:
        filters.append(f"pattern='{config.pattern}'")

    if filters:
        logger.info(f"  Filters:           {', '.join(filters)}")

    logger.info("=" * 60)
    logger.info("")

    if len(subjects_list) <= 20:
        logger.info("Masks to process:")
        for i, subject_data in enumerate(subjects_list, 1):
            logger.info(f"  {i:3d}. {_format_subject_id(subject_data)}")
        logger.info("")


def _handle_prepare_command(args: Namespace) -> int:
    """Handle the prepare subcommand."""
    target = getattr(args, "prepare_target", None)
    if not target:
        from lacuna.cli.parser import build_parser

        build_parser().parse_args(["prepare", "--help"])
        return EXIT_SUCCESS

    _setup_logging(logging.INFO)

    from lacuna.cli.prepare import run_prepare_ace, run_prepare_sntf

    try:
        if target == "sntf":
            run_prepare_sntf(args)
        elif target == "ace":
            run_prepare_ace(args)
        else:
            logger.error("Unknown prepare target: %s", target)
            return EXIT_INVALID_ARGS
    except FileNotFoundError as e:
        logger.error(str(e))
        return EXIT_GENERAL_ERROR
    except NotImplementedError as e:
        logger.error(str(e))
        return EXIT_GENERAL_ERROR

    return EXIT_SUCCESS


def _setup_logging(level: int) -> None:
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
