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

    parser = build_parser()

    # Handle --generate-config before full parsing
    if argv is None:
        argv = sys.argv[1:]
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
    from lacuna import SubjectData
    from lacuna.core.pipeline import analyze
    from lacuna.io import export_bids_derivatives, load_bids_dataset

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure work directory exists
    config.work_dir.mkdir(parents=True, exist_ok=True)

    # Set/log environment variables
    os.environ.setdefault("LACUNA_WORK_DIR", str(config.work_dir.resolve()))
    logger.debug(f"LACUNA_WORK_DIR: {os.environ.get('LACUNA_WORK_DIR')}")

    # Log TemplateFlow configuration (managed by templateflow library)
    templateflow_home = os.environ.get("TEMPLATEFLOW_HOME")
    if templateflow_home:
        logger.debug(f"TEMPLATEFLOW_HOME: {templateflow_home}")
    else:
        logger.debug("TEMPLATEFLOW_HOME not set, using default location")

    # Step 1: Load input data
    logger.info("Loading input data...")
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
            subjects_list = [subject_data]
            logger.info("Loaded single mask file")
        else:
            # BIDS dataset mode
            # Build pattern from subject/session filters
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
            subjects_list = list(subjects_dict.values())
            logger.info(f"Loaded {len(subjects_list)} subject(s)")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return EXIT_BIDS_ERROR

    # Step 2: Register custom connectomes if paths provided
    functional_connectome_name = _resolve_connectome(
        config.functional_connectome,
        connectome_type="functional",
        space=config.space,
    )
    structural_connectome_name = _resolve_connectome(
        config.structural_connectome,
        connectome_type="structural",
        space=config.space,
        tdi_path=config.structural_tdi,
    )

    # Step 3: Build analysis steps
    steps = _build_analysis_steps(
        config,
        functional_connectome_name=functional_connectome_name,
        structural_connectome_name=structural_connectome_name,
    )

    if not steps:
        logger.warning("No analyses configured. Only exporting input masks.")
    else:
        logger.info(f"Running analyses: {', '.join(steps.keys())}")

    # Step 4: Run analyses
    try:
        if steps:
            results = analyze(
                data=subjects_list if len(subjects_list) > 1 else subjects_list[0],
                steps=steps,
                n_jobs=config.n_procs,
                show_progress=True,
                log_level=config.verbose_count,
            )
            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]
        else:
            # No analysis steps, just pass through the input
            results = subjects_list
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return EXIT_ANALYSIS_ERROR

    # Step 5: Export results
    logger.info("Exporting BIDS derivatives...")
    try:
        for result in results:
            export_bids_derivatives(
                subject_data=result,
                output_dir=config.output_dir,
                overwrite=True,
            )
    except Exception as e:
        logger.error(f"Failed to export derivatives: {e}")
        return EXIT_ANALYSIS_ERROR

    logger.info(f"Results saved to: {config.output_dir}")
    logger.info("Lacuna CLI completed successfully")
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
            # For multiple subjects, we'll use a simple pattern that matches any
            # The load_bids_dataset will match all, but we could filter after
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
    functional_connectome_name: str | None,
    structural_connectome_name: str | None,
) -> dict[str, dict | None]:
    """
    Build analysis steps dictionary from configuration.

    Parameters
    ----------
    config : CLIConfig
        CLI configuration.
    functional_connectome_name : str, optional
        Registered functional connectome name.
    structural_connectome_name : str, optional
        Registered structural connectome name.

    Returns
    -------
    dict
        Steps dictionary for analyze() function.
    """
    steps: dict[str, dict | None] = {}

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

    return steps


def _resolve_connectome(
    connectome_path: str | Path | None,
    connectome_type: str,
    space: str | None = None,
    tdi_path: Path | None = None,
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
    tdi_path : Path, optional
        TDI path for structural connectomes.

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
        tdi_path=tdi_path,
    )


def _register_connectome_from_path(
    connectome_path: Path,
    connectome_type: str,
    space: str,
    tdi_path: Path | None = None,
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
    tdi_path : Path, optional
        TDI path for structural connectomes.

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
                tdi_path=tdi_path,
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


def _setup_logging(level: int) -> None:
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
