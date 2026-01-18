"""
Lacuna CLI main module.

This module provides the main entry point for the Lacuna CLI, orchestrating
the workflow from argument parsing through analysis execution to output writing.

Commands:
    lacuna fetch     - Download and setup connectomes
    lacuna run       - Run analyses (rd, fnm, snm)
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

from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)

# Exit codes following BIDS-Apps convention
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_INVALID_ARGS = 2
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
        if hasattr(args, "parcel_atlases") and args.parcel_atlases:
            analysis_options["parcel_names"] = args.parcel_atlases
        if hasattr(args, "threshold") and args.threshold is not None:
            analysis_options["threshold"] = args.threshold
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
        if hasattr(args, "parcellation") and args.parcellation:
            analysis_options["parcellation_name"] = args.parcellation
        if hasattr(args, "compute_roi_disconnection") and args.compute_roi_disconnection:
            analysis_options["compute_disconnectivity_matrix"] = True
        # Handle --no-cache-tdi flag (default is to cache)
        if hasattr(args, "no_cache_tdi") and args.no_cache_tdi:
            analysis_options["cache_tdi"] = False
        if hasattr(args, "mrtrix_threads"):
            analysis_options["n_jobs"] = args.mrtrix_threads
        if hasattr(args, "show_mrtrix_output") and args.show_mrtrix_output:
            analysis_options["show_mrtrix_output"] = True

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

    output_dir = args.output_dir
    overwrite = getattr(args, "overwrite", False)
    pattern = getattr(args, "pattern", None)

    # Build glob pattern - if user provides pattern, wrap it to match parcelstats files
    if pattern:
        # User provides a pattern fragment to match within parcelstats filenames
        # e.g., "*400*inputmask*" -> "*400*inputmask*_parcelstats.tsv"
        if "_parcelstats.tsv" in pattern:
            glob_pattern = pattern
        else:
            # Ensure pattern ends with _parcelstats.tsv
            glob_pattern = f"*{pattern.strip('*')}*_parcelstats.tsv"
    else:
        glob_pattern = "*_parcelstats.tsv"

    logger.info("Running collect (group-level aggregation)")
    logger.info(f"Scanning derivatives directory: {output_dir}")
    logger.info(f"Pattern: {glob_pattern}")

    try:
        created_files = aggregate_parcelstats(
            derivatives_dir=output_dir,
            output_dir=output_dir,
            pattern=glob_pattern,
            overwrite=overwrite,
        )

        if not created_files:
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

    atlases = list_parcellations()

    print("\nAvailable Brain Parcellations (Atlases)")
    print("=" * 60)

    if not atlases:
        print("  No atlases registered.")
        print("\n  Use 'lacuna fetch' to download connectomes which include atlases.")
        return EXIT_SUCCESS

    # Group by type
    schaefer = [a for a in atlases if a.name.startswith("Schaefer")]
    tian = [a for a in atlases if a.name.startswith("Tian")]
    combined = [a for a in atlases if "Tian" in a.name and "Schaefer" in a.name]
    other = [a for a in atlases if a not in schaefer + tian + combined]

    def print_atlas_group(title: str, atlas_list: list):
        if not atlas_list:
            return
        print(f"\n{title}:")
        for atlas in sorted(atlas_list, key=lambda x: x.name):
            space = getattr(atlas, "space", "unknown")
            resolution = getattr(atlas, "resolution", "?")
            print(f"  {atlas.name:<45} ({space}, {resolution}mm)")

    print_atlas_group("Schaefer Cortical Parcellations", schaefer)
    print_atlas_group("Tian Subcortical Parcellations", tian)
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
        print("  None registered. Use 'lacuna fetch dtor985' to download dTOR985.")

    print("\n" + "=" * 60)
    print("\nFetchable Connectomes (use 'lacuna fetch <name>'):")
    print("  gsp1000  - GSP1000 Functional Connectome (~200GB)")
    print("             1000 healthy subjects, MNI152NLin6Asym space")
    print("  dtor985  - dTOR985 Structural Tractogram (~10GB)")
    print("             985 healthy subjects, MNI152NLin2009bAsym space")
    print()

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
        # No connectome needed for this analysis (e.g., RegionalDamage)
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
    if analysis_class_name == "StructuralNetworkMapping":
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

    elif analysis_class_name == "FunctionalNetworkMapping":
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
        "rd": "RegionalDamage",
        "regionaldamage": "RegionalDamage",
        "fnm": "FunctionalNetworkMapping",
        "functionalnetworkmapping": "FunctionalNetworkMapping",
        "snm": "StructuralNetworkMapping",
        "structuralnetworkmapping": "StructuralNetworkMapping",
    }

    analysis_class_name = analysis_name_map.get(config.analysis.lower())
    if not analysis_class_name:
        logger.error(f"Unknown analysis: {config.analysis}")
        return EXIT_INVALID_ARGS

    # Register connectome from path for FNM/SNM analyses
    _register_connectome_from_path(config.analysis_options, analysis_class_name)

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
                config.participant_label,
                config.session_id,
                config.pattern,
            )
            subjects_dict = load_bids_dataset(
                bids_root=config.bids_dir,
                pattern=pattern,
                space=config.space,
                resolution=None,  # Auto-detect
            )

            if not subjects_dict:
                logger.error("No subjects found in BIDS dataset")
                return EXIT_BIDS_ERROR

            subjects_list = list(subjects_dict.values())

            # Filter by participant labels if specified
            if config.participant_label:
                subjects_list = _filter_by_participants(subjects_list, config.participant_label)
                if not subjects_list:
                    logger.error(
                        f"No subjects found matching participant labels: {config.participant_label}"
                    )
                    return EXIT_BIDS_ERROR

            _log_discovery_summary(subjects_list, config)

        # Process subjects
        if len(subjects_list) > 1 and config.batch_size != 1:
            # Batch processing
            result = _process_batch(subjects_list, steps, config, config.batch_size)
        else:
            # Sequential processing
            from tqdm import tqdm

            processed_count = 0
            for subject_data in tqdm(
                subjects_list,
                desc="Processing subjects",
                disable=not config.verbose,
            ):
                result = _process_single_subject(subject_data, steps, config, export=True)
                if result == EXIT_SUCCESS:
                    processed_count += 1
                else:
                    logger.warning("Subject processing failed, continuing...")

            logger.info(f"Successfully processed {processed_count} subject(s)")
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
    subjects: list[str] | None,
    sessions: list[str] | None,
    extra_pattern: str | None,
) -> str:
    """Build a glob pattern from filters.

    Note: For multiple subjects, this returns a broad pattern.
    Filtering by specific subjects is done in _filter_by_participants().
    """
    pattern_parts = []

    if subjects:
        if len(subjects) == 1:
            pattern_parts.append(f"*sub-{subjects[0]}*")
        else:
            # Multiple subjects - use broad pattern, filter later
            pattern_parts.append("*sub-*")
    else:
        pattern_parts.append("*")

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


def _setup_logging(level: int) -> None:
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
