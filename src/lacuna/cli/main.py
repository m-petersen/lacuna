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

    Parses command-line arguments, loads BIDS data, runs analyses,
    and writes BIDS-derivatives output.

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
    logger.info(f"BIDS directory: {config.bids_dir}")
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

    # Step 1: Load BIDS dataset
    logger.info("Loading BIDS dataset...")
    try:
        subjects_dict = load_bids_dataset(
            bids_root=config.bids_dir,
            subjects=config.participant_label,
            sessions=config.session_id,
            validate_bids=not config.skip_bids_validation,
        )
    except Exception as e:
        logger.error(f"Failed to load BIDS dataset: {e}")
        return EXIT_BIDS_ERROR

    if not subjects_dict:
        logger.error("No subjects found in BIDS dataset")
        return EXIT_BIDS_ERROR

    logger.info(f"Loaded {len(subjects_dict)} subject(s)")

    # Step 2: Register custom connectomes if paths provided
    functional_connectome_name = None
    structural_connectome_name = None

    if config.functional_connectome:
        functional_connectome_name = _register_connectome(
            config.functional_connectome, connectome_type="functional"
        )
        if not functional_connectome_name:
            return EXIT_ANALYSIS_ERROR

    if config.structural_connectome:
        structural_connectome_name = _register_structural_connectome(
            config.structural_connectome,
            config.structural_tdi,
        )
        if not structural_connectome_name:
            return EXIT_ANALYSIS_ERROR

    # Step 3: Run analyses
    logger.info("Running analyses...")
    subjects_list = list(subjects_dict.values())

    try:
        # TODO: Add parcel_atlases support when analyze() is extended
        results = analyze(
            data=subjects_list if len(subjects_list) > 1 else subjects_list[0],
            functional_connectome=functional_connectome_name,
            structural_connectome=structural_connectome_name,
            log_level=config.verbose_count,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return EXIT_ANALYSIS_ERROR

    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    # Step 4: Export results
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


def _register_connectome(connectome_path: Path, connectome_type: str) -> str | None:
    """
    Register a custom connectome from a file path.

    Parameters
    ----------
    connectome_path : Path
        Path to connectome file.
    connectome_type : str
        Type of connectome ("functional" or "structural").

    Returns
    -------
    str or None
        Registered connectome name, or None if registration failed.
    """
    try:
        import h5py

        # Read metadata from HDF5 file
        with h5py.File(connectome_path, "r") as f:
            space = f.attrs.get("space", "MNI152NLin6Asym")
            resolution = f.attrs.get("resolution", 2.0)

        # Generate a name from the file
        name = f"_cli_{connectome_path.stem}"

        # Register with the connectome registry
        from lacuna.assets.connectomes import register_functional_connectome

        register_functional_connectome(
            name=name,
            space=space,
            resolution=float(resolution),
            data_path=connectome_path,
        )

        logger.info(
            f"Registered {connectome_type} connectome: {name} (space={space}, res={resolution}mm)"
        )
        return name

    except Exception as e:
        logger.error(f"Failed to register {connectome_type} connectome: {e}")
        return None


def _register_structural_connectome(tractogram_path: Path, tdi_path: Path | None) -> str | None:
    """
    Register a custom structural connectome from file paths.

    Parameters
    ----------
    tractogram_path : Path
        Path to tractogram (.tck) file.
    tdi_path : Path
        Path to TDI NIfTI file.

    Returns
    -------
    str or None
        Registered connectome name, or None if registration failed.
    """
    try:
        # Generate a name from the file
        name = f"_cli_{tractogram_path.stem}"

        # Read metadata from TDI if available, otherwise use defaults
        space = "MNI152NLin6Asym"  # Default space
        resolution = 2.0  # Default resolution

        if tdi_path and tdi_path.exists():
            import nibabel as nib

            tdi_img = nib.load(tdi_path)
            # Try to infer resolution from voxel size
            voxel_sizes = tdi_img.header.get_zooms()[:3]
            if voxel_sizes:
                resolution = float(voxel_sizes[0])  # Assume isotropic

        # Register with the structural connectome registry
        from lacuna.assets.connectomes import register_structural_connectome

        register_structural_connectome(
            name=name,
            space=space,
            resolution=resolution,
            tractogram_path=tractogram_path,
            tdi_path=tdi_path,
        )

        logger.info(f"Registered structural connectome: {name} (space={space}, res={resolution}mm)")
        return name

    except Exception as e:
        logger.error(f"Failed to register structural connectome: {e}")
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
