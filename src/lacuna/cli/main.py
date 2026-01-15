"""
Lacuna CLI main module.

This module provides the main entry point for the Lacuna CLI, orchestrating
the workflow from argument parsing through analysis execution to output writing.

Functions:
    main: Main CLI entry point that parses arguments and runs the workflow.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lacuna.cli.config import CLIConfig

logger = logging.getLogger(__name__)

# Exit codes following BIDS-Apps convention
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_INVALID_ARGS = 2
EXIT_BIDS_ERROR = 64
EXIT_ANALYSIS_ERROR = 65


def _format_subject_id(subject_data) -> str:
    """
    Format a human-readable identifier for a subject.

    Combines subject_id, session_id, and label into a compact string.

    Parameters
    ----------
    subject_data : SubjectData
        Subject data with metadata.

    Returns
    -------
    str
        Formatted identifier like 'sub-001/ses-01/lesion' or 'sub-001/lesion'
    """
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


def _log_discovery_summary(subjects_list: list, config) -> None:
    """
    Log a summary of discovered subjects and masks.

    Parameters
    ----------
    subjects_list : list[SubjectData]
        List of discovered subject data.
    config : CLIConfig
        Configuration with filter criteria.
    """
    if not subjects_list:
        return

    # Count unique subjects, sessions, and labels
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

    # Build summary message
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

    # Log filters if any were applied
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

    # Log individual masks if verbose or small number
    if len(subjects_list) <= 20:
        logger.info("Masks to process:")
        for i, subject_data in enumerate(subjects_list, 1):
            logger.info(f"  {i:3d}. {_format_subject_id(subject_data)}")
        logger.info("")


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Parses command-line arguments, loads input data (BIDS dataset or single file),
    runs analyses, and writes BIDS-derivatives output.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, uses sys.argv[1:].

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    from lacuna.cli.config import CLIConfig, generate_config_template, load_yaml_config
    from lacuna.cli.parser import build_parser

    if argv is None:
        argv = sys.argv[1:]

    # Check if this is a subcommand (fetch, etc.) or BIDS workflow
    subcommands = {"fetch", "run"}
    if argv and argv[0] in subcommands:
        return _handle_subcommand(argv)

    # BIDS-Apps workflow
    parser = build_parser()

    # Handle --generate-config before full parsing
    if "--generate-config" in argv:
        print(generate_config_template())
        return EXIT_SUCCESS

    args = parser.parse_args(argv)

    # Load YAML config if provided
    yaml_config = None
    if args.config:
        try:
            yaml_config = load_yaml_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Config file error: {e}")
            return EXIT_INVALID_ARGS

    # Build configuration from arguments + YAML
    try:
        config = CLIConfig.from_args(args, yaml_config)
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return EXIT_INVALID_ARGS

    # Configure logging
    _setup_logging(config.log_level)

    # Suppress nilearn warnings (they are verbose and not user-actionable)
    import warnings

    warnings.filterwarnings("ignore", module="nilearn")
    warnings.filterwarnings("ignore", message=".*Non-finite values.*")
    warnings.filterwarnings("ignore", message=".*Casting data from.*")

    logger.info("Lacuna CLI starting")
    logger.info(f"Input: {config.bids_dir}")
    logger.info(f"Output directory: {config.output_dir}")

    try:
        return _run_workflow(config)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if config.verbose_count >= 2:
            import traceback

            traceback.print_exc()
        return EXIT_GENERAL_ERROR


def _handle_subcommand(argv: list[str]) -> int:
    """
    Handle subcommands (fetch, run, etc.).

    Parameters
    ----------
    argv : list of str
        Command-line arguments starting with subcommand.

    Returns
    -------
    int
        Exit code.
    """
    from lacuna.cli.parser import build_main_parser

    parser = build_main_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        from lacuna.cli.fetch_cmd import handle_fetch_command

        return handle_fetch_command(args)

    elif args.command == "run":
        # The 'run' subcommand uses the standard BIDS workflow
        from lacuna.cli.config import CLIConfig, load_yaml_config

        yaml_config = None
        if hasattr(args, "config") and args.config:
            try:
                yaml_config = load_yaml_config(args.config)
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Config file error: {e}")
                return EXIT_INVALID_ARGS

        try:
            config = CLIConfig.from_args(args, yaml_config)
            config.validate()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return EXIT_INVALID_ARGS

        _setup_logging(config.log_level)
        return _run_workflow(config)

    else:
        parser.print_help()
        return EXIT_SUCCESS


def _run_workflow(config: CLIConfig) -> int:
    """
    Run the main analysis workflow.

    Parameters
    ----------
    config : CLIConfig
        Validated configuration.

    Returns
    -------
    int
        Exit code.
    """
    # Handle group-level analysis separately
    if config.analysis_level == "group":
        return _run_group_workflow(config)

    from lacuna import SubjectData
    from lacuna.io import load_bids_dataset

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure tmp directory exists
    config.tmp_dir.mkdir(parents=True, exist_ok=True)

    # Set/log environment variables
    os.environ.setdefault("LACUNA_TMP_DIR", str(config.tmp_dir.resolve()))
    logger.debug(f"LACUNA_TMP_DIR: {os.environ.get('LACUNA_TMP_DIR')}")

    # Log TemplateFlow configuration (managed by templateflow library)
    templateflow_home = os.environ.get("TEMPLATEFLOW_HOME")
    if templateflow_home:
        logger.debug(f"TEMPLATEFLOW_HOME: {templateflow_home}")
    else:
        logger.debug("TEMPLATEFLOW_HOME not set, using default location")

    # Step 1: Register connectomes (do this once before processing subjects)
    # New format: config.connectomes (dict of named connectomes from YAML)
    # CLI-provided: config.functional_connectome, config.structural_connectome
    registered_connectomes: dict[str, str] = {}

    # Register connectomes from YAML config
    for conn_name, conn_config in config.connectomes.items():
        registered_name = _register_connectome_from_config(conn_name, conn_config)
        if registered_name:
            registered_connectomes[conn_name] = registered_name

    # Register CLI-provided connectomes
    functional_connectome_name = _resolve_connectome(
        config.functional_connectome,
        connectome_type="functional",
        space=config.space,
    )
    structural_connectome_name = _resolve_connectome(
        config.structural_connectome,
        connectome_type="structural",
        space=config.space,
    )

    # Step 2: Build analysis steps (do this once)
    steps = _build_analysis_steps(
        config,
        registered_connectomes=registered_connectomes,
        functional_connectome_name=functional_connectome_name,
        structural_connectome_name=structural_connectome_name,
    )

    if not steps:
        logger.warning("No analyses configured. Only exporting input masks.")
    else:
        logger.info(f"Running analyses: {', '.join(steps.keys())}")

    # Step 3: Process subjects
    # For memory efficiency, process each subject individually when multiple
    # subjects are specified: load → analyze → export → release memory
    try:
        if config.is_single_file:
            # Single NIfTI file mode
            subject_data = SubjectData.from_nifti(
                config.bids_dir,
                space=config.space,
                resolution=config.resolution,
                metadata={
                    "subject_id": f"sub-{config.bids_dir.stem.split('_')[0]}",
                },
            )
            result = _process_single_subject(subject_data, steps, config, export=True)
            if result != EXIT_SUCCESS:
                return result
            logger.info("Loaded and processed single mask file")

        elif config.participant_label and len(config.participant_label) > 1:
            # Multiple subjects specified by participant label
            from lacuna.io.bids import BidsError

            # First, load all subjects
            all_subjects = []
            for subject_id in config.participant_label:
                pattern = _build_pattern(
                    [subject_id],  # Single subject
                    config.session_id,
                    config.pattern,
                )
                try:
                    subjects_dict = load_bids_dataset(
                        bids_root=config.bids_dir,
                        pattern=pattern,
                        space=config.space,
                        resolution=config.resolution,
                    )
                    if subjects_dict:
                        all_subjects.extend(subjects_dict.values())
                    else:
                        logger.warning(f"No data found for subject: {subject_id}")
                except BidsError:
                    logger.warning(f"No data found for subject: {subject_id}")

            if not all_subjects:
                logger.error(f"No subjects matching {config.participant_label} found")
                return EXIT_BIDS_ERROR

            _log_discovery_summary(all_subjects, config)

            # Process using batch or sequential mode
            if config.batch_size != 1 and steps:
                # Batch processing mode
                result = _process_batch(all_subjects, steps, config, config.batch_size)
                if result != EXIT_SUCCESS:
                    return result
            else:
                # Sequential processing
                from tqdm import tqdm

                processed_count = 0
                for subject_data in tqdm(
                    all_subjects,
                    desc="Processing subjects",
                    disable=not config.verbose,
                ):
                    result = _process_single_subject(subject_data, steps, config, export=True)
                    if result != EXIT_SUCCESS:
                        logger.warning("Subject processing failed, continuing...")
                    else:
                        processed_count += 1

                logger.info(f"Successfully processed {processed_count} subject(s)")

        else:
            # Single subject or all subjects - load all at once
            pattern = _build_pattern(
                config.participant_label,
                config.session_id,
                config.pattern,
            )
            subjects_dict = load_bids_dataset(
                bids_root=config.bids_dir,
                pattern=pattern,
                space=config.space,
                resolution=config.resolution,
            )

            if not subjects_dict:
                logger.error("No subjects found in BIDS dataset")
                return EXIT_BIDS_ERROR

            # Filter by multiple sessions if specified
            if config.session_id and len(config.session_id) > 1:
                filtered_dict = {}
                for key, value in subjects_dict.items():
                    session_id = value.metadata.get("session_id", "")
                    session_id_clean = session_id.replace("ses-", "")
                    if session_id_clean in config.session_id:
                        filtered_dict[key] = value
                if not filtered_dict:
                    logger.error(f"No sessions matching {config.session_id} found")
                    return EXIT_BIDS_ERROR
                subjects_dict = filtered_dict

            subjects_list = list(subjects_dict.values())
            _log_discovery_summary(subjects_list, config)

            # Process subjects - use batch processing if batch_size != 1
            if len(subjects_list) > 1:
                if config.batch_size != 1 and steps:
                    # Batch processing mode - process multiple subjects together
                    result = _process_batch(subjects_list, steps, config, config.batch_size)
                    if result != EXIT_SUCCESS:
                        return result
                else:
                    # Sequential processing - one subject at a time
                    from tqdm import tqdm

                    for subject_data in tqdm(
                        subjects_list,
                        desc="Processing subjects",
                        disable=not config.verbose,
                    ):
                        result = _process_single_subject(subject_data, steps, config, export=True)
                        if result != EXIT_SUCCESS:
                            logger.warning("Subject processing failed, continuing...")
            else:
                # Single subject
                result = _process_single_subject(subjects_list[0], steps, config, export=True)
                if result != EXIT_SUCCESS:
                    return result

    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        return EXIT_BIDS_ERROR

    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("Lacuna CLI completed successfully")
    return EXIT_SUCCESS


def _process_single_subject(
    subject_data,
    steps: dict,
    config: CLIConfig,
    export: bool = True,
) -> int:
    """
    Process a single subject: analyze and optionally export.

    This function processes one subject at a time to minimize memory usage.
    After export, results can be garbage collected.

    Parameters
    ----------
    subject_data : SubjectData
        Input subject data.
    steps : dict
        Analysis steps to run.
    config : CLIConfig
        Configuration.
    export : bool
        Whether to export results immediately.

    Returns
    -------
    int
        Exit code.
    """
    from lacuna.core.pipeline import analyze
    from lacuna.io import export_bids_derivatives

    try:
        if steps:
            result = analyze(
                data=subject_data,
                steps=steps,
                n_jobs=config.n_procs,
                show_progress=False,  # Outer loop shows progress
                verbose=config.verbose,
            )
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
    config: CLIConfig,
    batch_size: int,
) -> int:
    """
    Process subjects in batches using optimized batch operations.

    Automatically selects the best strategy for each analysis:
    - "vectorized" for FNM (processes all masks together in matrix operations)
    - "parallel" for other analyses (parallel processing across subjects)

    Parameters
    ----------
    subjects_list : list[SubjectData]
        List of subjects to process.
    steps : dict
        Analysis steps to run.
    config : CLIConfig
        Configuration.
    batch_size : int
        Number of subjects per batch. Use -1 for all subjects at once.

    Returns
    -------
    int
        Exit code.
    """
    from lacuna.batch import batch_process
    from lacuna.io import export_bids_derivatives

    # Determine actual batch size
    n_subjects = len(subjects_list)
    if batch_size == -1:
        actual_batch_size = n_subjects
    else:
        actual_batch_size = min(batch_size, n_subjects)

    logger.info(f"Batch processing: {n_subjects} subjects in batches of {actual_batch_size}")

    # Build analysis instances from steps
    from lacuna.analysis import get_analysis

    analyses = []
    for analysis_name, kwargs in steps.items():
        analysis_cls = get_analysis(analysis_name)
        if kwargs is None:
            kwargs = {}
        else:
            kwargs = kwargs.copy()
        if "verbose" not in kwargs:
            kwargs["verbose"] = config.verbose
        analyses.append((analysis_name, analysis_cls(**kwargs)))

    # Process in batches
    processed_count = 0
    failed_count = 0

    for batch_start in range(0, n_subjects, actual_batch_size):
        batch_end = min(batch_start + actual_batch_size, n_subjects)
        batch = subjects_list[batch_start:batch_end]

        # Log batch info with subject identifiers
        batch_num = batch_start // actual_batch_size + 1
        total_batches = (n_subjects + actual_batch_size - 1) // actual_batch_size

        if n_subjects > actual_batch_size:
            logger.info("")
            logger.info(f"--- Batch {batch_num}/{total_batches} ({len(batch)} masks) ---")
            for subject_data in batch:
                logger.info(f"    • {_format_subject_id(subject_data)}")

        try:
            # Run each analysis in sequence using batch_process
            # batch_process auto-selects strategy based on analysis.batch_strategy
            current_data = batch
            for analysis_name, analysis in analyses:
                strategy = getattr(analysis, "batch_strategy", "parallel")
                logger.info(f"Running {analysis_name} ({strategy} strategy)")
                current_data = batch_process(
                    inputs=current_data,
                    analysis=analysis,
                    n_jobs=config.n_procs,
                    show_progress=config.verbose,
                    strategy=None,  # Auto-select based on analysis.batch_strategy
                )

            # Export results
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
    """
    Build a glob pattern from subject/session filters.

    Parameters
    ----------
    subjects : list of str, optional
        Subject IDs to filter.
    sessions : list of str, optional
        Session IDs to filter.
    extra_pattern : str, optional
        Additional pattern to include.

    Returns
    -------
    str
        Glob pattern for matching files.
    """
    # Start with wildcards
    pattern_parts = []

    if subjects:
        # Match any of the specified subjects
        if len(subjects) == 1:
            pattern_parts.append(f"*sub-{subjects[0]}*")
        else:
            # Multiple subjects should be loaded individually by caller
            # This branch is kept for backward compatibility but caller
            # should iterate over subjects instead
            pattern_parts.append("*sub-*")
    else:
        pattern_parts.append("*")

    if sessions:
        if len(sessions) == 1:
            pattern_parts.append(f"ses-{sessions[0]}*")

    if extra_pattern:
        pattern_parts.append(extra_pattern)

    # Combine pattern parts
    return "".join(pattern_parts) if pattern_parts else "*"


def _build_analysis_steps(
    config: CLIConfig,
    *,
    registered_connectomes: dict[str, str] | None = None,
    functional_connectome_name: str | None = None,
    structural_connectome_name: str | None = None,
) -> dict[str, dict | None]:
    """
    Build analysis steps dictionary from configuration.

    Uses full analysis configurations from YAML if available, otherwise
    falls back to CLI arguments.

    Parameters
    ----------
    config : CLIConfig
        CLI configuration.
    registered_connectomes : dict, optional
        Mapping of YAML connectome names to registered names.
    functional_connectome_name : str, optional
        Registered functional connectome name (from CLI).
    structural_connectome_name : str, optional
        Registered structural connectome name (from CLI).

    Returns
    -------
    dict
        Steps dictionary for analyze() function.
    """
    registered_connectomes = registered_connectomes or {}
    steps: dict[str, dict | None] = {}

    def _resolve_connectome_ref(analysis_config: dict) -> str | None:
        """Resolve a connectome reference in analysis config to registered name."""
        # Check for 'connectome' key that references a name in connectomes section
        conn_ref = analysis_config.get("connectome")
        if conn_ref and conn_ref in registered_connectomes:
            return registered_connectomes[conn_ref]
        # Check for already-resolved connectome_name
        conn_name = analysis_config.get("connectome_name")
        if conn_name:
            return str(conn_name)
        return None

    # Check if we have full YAML analysis configs
    if config.analyses:
        # Use YAML-based analysis configurations

        # RegionalDamage
        if "RegionalDamage" in config.analyses:
            rd_config = config.analyses["RegionalDamage"].copy()
            # CLI parcel_atlases override YAML if provided
            if config.parcel_atlases and not rd_config.get("parcel_names"):
                rd_config["parcel_names"] = config.parcel_atlases
            steps["RegionalDamage"] = rd_config if rd_config else None

        # FunctionalNetworkMapping
        if "FunctionalNetworkMapping" in config.analyses:
            fnm_config = config.analyses["FunctionalNetworkMapping"].copy()
            # Remove path keys (already used for registration)
            fnm_config.pop("connectome_path", None)

            # Resolve connectome reference
            conn_name = _resolve_connectome_ref(fnm_config)
            if conn_name is None and functional_connectome_name:
                conn_name = functional_connectome_name
            if conn_name:
                fnm_config.pop("connectome", None)  # Remove reference key
                fnm_config["connectome_name"] = conn_name
                steps["FunctionalNetworkMapping"] = fnm_config
            else:
                logger.warning("FunctionalNetworkMapping configured but no connectome provided")

        # StructuralNetworkMapping
        if "StructuralNetworkMapping" in config.analyses:
            snm_config = config.analyses["StructuralNetworkMapping"].copy()
            # Remove path keys (already used for registration)
            snm_config.pop("tractogram_path", None)
            snm_config.pop("tdi_path", None)

            # Resolve connectome reference
            conn_name = _resolve_connectome_ref(snm_config)
            if conn_name is None and structural_connectome_name:
                conn_name = structural_connectome_name
            if conn_name:
                snm_config.pop("connectome", None)  # Remove reference key
                snm_config["connectome_name"] = conn_name
                steps["StructuralNetworkMapping"] = snm_config
            else:
                logger.warning("StructuralNetworkMapping configured but no connectome provided")

        # ParcelAggregation (explicit YAML config)
        if "ParcelAggregation" in config.analyses:
            pa_config = config.analyses["ParcelAggregation"].copy()
            # Use parcel_atlases from CLI if not in YAML
            if config.parcel_atlases and not pa_config.get("parcel_names"):
                pa_config["parcel_names"] = config.parcel_atlases
            steps["ParcelAggregation"] = pa_config

        # Auto-add ParcelAggregation for FNM/SNM if parcel_atlases specified but no explicit config
        elif config.parcel_atlases and (
            "FunctionalNetworkMapping" in steps or "StructuralNetworkMapping" in steps
        ):
            sources: dict[str, str | list[str]] = {}
            if "FunctionalNetworkMapping" in steps:
                sources["FunctionalNetworkMapping"] = ["rmap", "tmap", "zmap"]
            if "StructuralNetworkMapping" in steps:
                sources["StructuralNetworkMapping"] = "disconnection_map"
            if sources:
                steps["ParcelAggregation"] = {
                    "source": sources,
                    "aggregation": "mean",
                    "parcel_names": config.parcel_atlases,
                }

    else:
        # Fallback: CLI-only mode (no YAML analyses config)

        # RegionalDamage (enabled by default unless skipped)
        if not config.skip_regional_damage:
            rd_kwargs: dict | None = None
            if config.parcel_atlases:
                rd_kwargs = {"parcel_names": config.parcel_atlases}
            steps["RegionalDamage"] = rd_kwargs

        # FunctionalNetworkMapping (if connectome provided)
        if functional_connectome_name:
            steps["FunctionalNetworkMapping"] = {
                "connectome_name": functional_connectome_name,
            }

        # StructuralNetworkMapping (if connectome provided)
        if structural_connectome_name:
            steps["StructuralNetworkMapping"] = {
                "connectome_name": structural_connectome_name,
            }

        # ParcelAggregation for FNM/SNM outputs (if parcel_atlases specified)
        if config.parcel_atlases:
            sources_fallback: dict[str, str | list[str]] = {}
            if functional_connectome_name:
                sources_fallback["FunctionalNetworkMapping"] = [
                    "rmap",
                    "tmap",
                    "zmap",
                ]
            if structural_connectome_name:
                sources_fallback["StructuralNetworkMapping"] = "disconnection_map"
            if sources_fallback:
                steps["ParcelAggregation"] = {
                    "source": sources_fallback,
                    "aggregation": "mean",
                    "parcel_names": config.parcel_atlases,
                }

    return steps


def _resolve_connectome(
    connectome_path: str | Path | None,
    connectome_type: str,
    space: str | None = None,
) -> str | None:
    """
    Resolve a connectome path to a registered name.

    Parameters
    ----------
    connectome_path : str or Path, optional
        Path to connectome file/directory.
    connectome_type : str
        Type of connectome ("functional" or "structural").
    space : str, optional
        Coordinate space for registration.

    Returns
    -------
    str or None
        Registered connectome name, or None if not provided.
    """
    if not connectome_path:
        return None

    connectome_path = Path(connectome_path)

    # Validate path exists
    if not connectome_path.exists():
        logger.error(f"{connectome_type.capitalize()} connectome not found: {connectome_path}")
        return None

    return _register_connectome_from_path(
        connectome_path,
        connectome_type=connectome_type,
        space=space or "MNI152NLin6Asym",
    )


def _register_connectome_from_path(
    connectome_path: Path,
    connectome_type: str,
    space: str,
) -> str | None:
    """
    Register a connectome from a file path.

    Parameters
    ----------
    connectome_path : Path
        Path to connectome file/directory.
    connectome_type : str
        Type of connectome ("functional" or "structural").
    space : str
        Coordinate space.

    Returns
    -------
    str or None
        Registered connectome name, or None if registration failed.
    """
    try:
        name = f"_cli_{connectome_path.stem}"

        if connectome_type == "functional":
            from lacuna.assets.connectomes import register_functional_connectome

            # Try to extract resolution from HDF5 file
            resolution = _get_resolution_from_connectome(connectome_path)

            register_functional_connectome(
                name=name,
                space=space,
                resolution=resolution,
                data_path=connectome_path,
            )
        elif connectome_type == "structural":
            from lacuna.assets.connectomes import register_structural_connectome

            register_structural_connectome(
                name=name,
                space=space,
                tractogram_path=connectome_path,
            )

        logger.info(f"Registered {connectome_type} connectome: {name} (space={space})")
        return name

    except Exception as e:
        logger.error(f"Failed to register {connectome_type} connectome: {e}")
        return None


def _get_resolution_from_connectome(connectome_path: Path) -> float:
    """
    Extract resolution from a functional connectome HDF5 file.

    Parameters
    ----------
    connectome_path : Path
        Path to connectome HDF5 file.

    Returns
    -------
    float
        Resolution in mm (defaults to 2.0 if cannot be determined).
    """
    try:
        import h5py
        import numpy as np

        with h5py.File(connectome_path, "r") as f:
            if "mask_affine" in f:
                affine = np.array(f["mask_affine"])
                # Resolution is typically the diagonal elements of the affine
                resolution = abs(affine[0, 0])
                if 0.5 <= resolution <= 10.0:  # Sanity check
                    return float(resolution)
    except Exception:
        pass

    # Default to 2mm if cannot be determined
    logger.debug(f"Could not determine resolution from {connectome_path}, using default 2.0mm")
    return 2.0


def _register_connectome_from_config(name: str, conn_config) -> str | None:
    """
    Register a connectome from a ConnectomeConfig.

    Parameters
    ----------
    name : str
        Name to register the connectome under.
    conn_config : ConnectomeConfig
        Connectome configuration from YAML.

    Returns
    -------
    str or None
        Registered connectome name, or None if registration failed.
    """
    try:
        registered_name = f"_yaml_{name}"

        if conn_config.type == "functional":
            from lacuna.assets.connectomes import register_functional_connectome

            # Use provided resolution or extract from file
            resolution = conn_config.resolution
            if resolution is None:
                resolution = _get_resolution_from_connectome(conn_config.path)

            register_functional_connectome(
                name=registered_name,
                space=conn_config.space,
                resolution=resolution,
                data_path=conn_config.path,
            )
        elif conn_config.type == "structural":
            from lacuna.assets.connectomes import register_structural_connectome

            register_structural_connectome(
                name=registered_name,
                space=conn_config.space,
                tractogram_path=conn_config.path,
                template_path=conn_config.template_path,
            )
        else:
            logger.error(f"Unknown connectome type: {conn_config.type}")
            return None

        logger.info(
            f"Registered {conn_config.type} connectome: {registered_name} "
            f"(space={conn_config.space})"
        )
        return registered_name

    except Exception as e:
        logger.error(f"Failed to register connectome '{name}': {e}")
        return None


def _setup_logging(level: int) -> None:
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _run_group_workflow(config: CLIConfig) -> int:
    """
    Run group-level analysis workflow.

    Aggregates subject-level parcelstats TSV files into group-level DataFrames.

    Parameters
    ----------
    config : CLIConfig
        Validated configuration.

    Returns
    -------
    int
        Exit code.
    """
    from lacuna.io.bids import BidsError, aggregate_parcelstats

    logger.info("Running group-level analysis")
    logger.info(f"Scanning derivatives directory: {config.output_dir}")

    try:
        # Aggregate parcelstats files
        created_files = aggregate_parcelstats(
            derivatives_dir=config.output_dir,
            output_dir=config.output_dir,
            overwrite=config.overwrite,
        )

        if not created_files:
            logger.warning("No parcelstats files found to aggregate")
            return EXIT_SUCCESS

        logger.info(f"Created {len(created_files)} group-level TSV file(s):")
        for _output_type, path in created_files.items():
            logger.info(f"  - {path.name}")

        return EXIT_SUCCESS

    except BidsError as e:
        logger.error(f"Group analysis failed: {e}")
        return EXIT_ANALYSIS_ERROR
    except Exception as e:
        logger.error(f"Unexpected error during group analysis: {e}")
        if config.verbose_count >= 2:
            import traceback

            traceback.print_exc()
        return EXIT_GENERAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
