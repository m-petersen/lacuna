"""
Lacuna CLI configuration module.

This module provides the CLIConfig dataclass for holding parsed
CLI arguments and validating them.

Classes:
    CLIConfig: Configuration from CLI arguments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


@dataclass
class CLIConfig:
    """
    Configuration from CLI arguments.

    This dataclass holds all configuration parsed from command-line
    arguments and provides validation.

    Attributes
    ----------
    bids_dir : Path
        Input BIDS dataset directory.
    output_dir : Path
        Output derivatives directory.
    analysis_level : str
        Processing level ("participant" only for now).
    participant_label : list of str, optional
        Subject IDs to process.
    session_id : list of str, optional
        Session IDs to process.
    skip_bids_validation : bool
        Whether to skip BIDS dataset validation.
    functional_connectome : Path, optional
        Path to functional connectome HDF5 file.
    structural_connectome : Path, optional
        Path to structural tractogram (.tck).
    structural_tdi : Path, optional
        Path to whole-brain TDI NIfTI.
    parcel_atlases : list of str, optional
        Atlas names for parcel aggregation.
    atlas_dir : Path, optional
        Additional directory containing atlas files.
    n_procs : int
        Number of parallel processes.
    work_dir : Path
        Working directory for intermediate files.
    verbose_count : int
        Logging verbosity level (0-2).
    """

    # BIDS-Apps required arguments
    bids_dir: Path
    output_dir: Path
    analysis_level: str

    # BIDS filtering
    participant_label: list[str] | None = None
    session_id: list[str] | None = None
    skip_bids_validation: bool = False

    # Analysis options
    functional_connectome: Path | None = None
    structural_connectome: Path | None = None
    structural_tdi: Path | None = None
    parcel_atlases: list[str] | None = None
    atlas_dir: Path | None = None

    # Performance options
    n_procs: int = 1
    work_dir: Path = field(default_factory=lambda: Path("work"))

    # Other options
    verbose_count: int = 0

    @property
    def log_level(self) -> int:
        """
        Convert verbose_count to log level.

        Returns
        -------
        int
            Logging level (25=WORKFLOW, 20=INFO, 10=DEBUG).
        """
        # 25 = custom WORKFLOW level, 20 = INFO, 10 = DEBUG
        return max(25 - 5 * self.verbose_count, 10)

    @classmethod
    def from_args(cls, args: Namespace) -> CLIConfig:
        """
        Create CLIConfig from parsed arguments.

        Parameters
        ----------
        args : Namespace
            Parsed arguments from argparse.

        Returns
        -------
        CLIConfig
            Configuration instance.
        """
        return cls(
            bids_dir=args.bids_dir,
            output_dir=args.output_dir,
            analysis_level=args.analysis_level,
            participant_label=args.participant_label,
            session_id=getattr(args, "session_id", None),
            skip_bids_validation=args.skip_bids_validation,
            functional_connectome=args.functional_connectome,
            structural_connectome=args.structural_connectome,
            structural_tdi=getattr(args, "structural_tdi", None),
            parcel_atlases=args.parcel_atlases,
            atlas_dir=getattr(args, "atlas_dir", None),
            n_procs=args.nprocs,
            work_dir=args.work_dir,
            verbose_count=args.verbose_count,
        )

    def validate(self) -> None:
        """
        Validate configuration.

        Raises
        ------
        ValueError
            If configuration is invalid.
        """
        # BIDS directory must exist
        if not self.bids_dir.exists():
            raise ValueError(f"BIDS directory does not exist: {self.bids_dir}")

        # Output directory cannot be same as input
        if self.output_dir.resolve() == self.bids_dir.resolve():
            raise ValueError("Output directory cannot be same as input BIDS directory")

        # Analysis level must be 'participant'
        if self.analysis_level != "participant":
            raise ValueError(
                f"Invalid analysis level '{self.analysis_level}'. "
                "Only 'participant' is supported."
            )

        # Functional connectome path must exist if provided
        if self.functional_connectome and not self.functional_connectome.exists():
            raise ValueError(f"Functional connectome not found: {self.functional_connectome}")

        # Structural connectome validation
        if self.structural_connectome:
            if not self.structural_connectome.exists():
                raise ValueError(f"Structural connectome not found: {self.structural_connectome}")
            if not self.structural_tdi:
                raise ValueError("--structural-tdi required when using --structural-connectome")
            if not self.structural_tdi.exists():
                raise ValueError(f"Structural TDI not found: {self.structural_tdi}")

        # Atlas directory must exist if provided
        if self.atlas_dir and not self.atlas_dir.exists():
            raise ValueError(f"Atlas directory not found: {self.atlas_dir}")

        # n_procs must be positive
        if self.n_procs < 1:
            raise ValueError(f"--nprocs must be at least 1, got {self.n_procs}")
