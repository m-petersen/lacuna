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
    from lacuna.cli.config import CLIConfig
    from lacuna.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)

    # Build configuration from arguments
    try:
        config = CLIConfig.from_args(args)
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
            subjects_dict = load_bids_dataset(
                bids_root=config.bids_dir,
                subjects=config.participant_label,
                sessions=config.session_id,
                pattern=config.pattern,
                space=config.space,
                resolution=config.resolution,
                validate_bids=not config.skip_bids_validation,
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
        resolution=config.resolution,
    )
    structural_connectome_name = _resolve_connectome(
        config.structural_connectome,
        connectome_type="structural",
        space=config.space,
        resolution=config.resolution,
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
    connectome_ref: str | None,
    connectome_type: str,
    space: str | None = None,
    resolution: float | None = None,
    tdi_path: Path | None = None,
) -> str | None:
    """
    Resolve a connectome reference (name or path) to a registered name.

    Parameters
    ----------
    connectome_ref : str, optional
        Connectome name (from registry) or path to file.
    connectome_type : str
        Type of connectome ("functional" or "structural").
    space : str, optional
        Coordinate space for registration.
    resolution : float, optional
        Resolution for registration.
    tdi_path : Path, optional
        TDI path for structural connectomes.

    Returns
    -------
    str or None
        Registered connectome name, or None if not provided.
    """
    if not connectome_ref:
        return None

    connectome_path = Path(connectome_ref)

    # Check if it's an existing file/directory (path mode)
    if connectome_path.exists():
        return _register_connectome_from_path(
            connectome_path,
            connectome_type=connectome_type,
            space=space or "MNI152NLin6Asym",
            resolution=resolution or 2.0,
            tdi_path=tdi_path,
        )

    # Otherwise treat as a registry name
    # Validate it exists in registry
    if connectome_type == "functional":
        from lacuna.assets.connectomes import list_functional_connectomes

        available = list_functional_connectomes()
        if connectome_ref not in [c["name"] for c in available]:
            logger.warning(
                f"Functional connectome '{connectome_ref}' not found in registry. "
                "Available: " + ", ".join(c["name"] for c in available)
            )
    elif connectome_type == "structural":
        from lacuna.assets.connectomes import list_structural_connectomes

        available = list_structural_connectomes()
        if connectome_ref not in [c["name"] for c in available]:
            logger.warning(
                f"Structural connectome '{connectome_ref}' not found in registry. "
                "Available: " + ", ".join(c["name"] for c in available)
            )

    return connectome_ref


def _register_connectome_from_path(
    connectome_path: Path,
    connectome_type: str,
    space: str,
    resolution: float,
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
    resolution : float
        Resolution in mm.
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

        logger.info(
            f"Registered {connectome_type} connectome: {name} "
            f"(space={space}, res={resolution}mm)"
        )
        return name

    except Exception as e:
        logger.error(f"Failed to register {connectome_type} connectome: {e}")
        return None


def _setup_logging(level: int) -> None:
    """Configure logging based on verbosity level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    sys.exit(main())
