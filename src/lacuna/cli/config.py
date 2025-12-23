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
        Input BIDS dataset directory OR path to single NIfTI mask.
    output_dir : Path
        Output derivatives directory.
    analysis_level : str
        Processing level ("participant" only for now).
    participant_label : list of str, optional
        Subject IDs to process.
    session_id : list of str, optional
        Session IDs to process.
    pattern : str, optional
        Glob pattern to filter mask files.
    skip_bids_validation : bool
        Whether to skip BIDS dataset validation.
    space : str, optional
        Coordinate space (required if not in filename).
    resolution : float, optional
        Voxel resolution in mm (required if not in filename).
    functional_connectome : str, optional
        Functional connectome name or path.
    structural_connectome : str, optional
        Structural connectome name or path.
    structural_tdi : Path, optional
        Path to whole-brain TDI NIfTI.
    parcel_atlases : list of str, optional
        Atlas names for RegionalDamage analysis.
    skip_regional_damage : bool
        Whether to skip RegionalDamage analysis.
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
    pattern: str | None = None
    skip_bids_validation: bool = False

    # Space/Resolution
    space: str | None = None
    resolution: float | None = None

    # Analysis options
    functional_connectome: str | None = None
    structural_connectome: str | None = None
    structural_tdi: Path | None = None
    parcel_atlases: list[str] | None = None
    skip_regional_damage: bool = False
    atlas_dir: Path | None = None

    # Performance options
    n_procs: int = 1
    work_dir: Path = field(default_factory=lambda: Path("work"))

    # Other options
    verbose_count: int = 0

    @property
    def is_single_file(self) -> bool:
        """Check if input is a single NIfTI file rather than BIDS directory."""
        return self.bids_dir.is_file() and self.bids_dir.suffix in (".nii", ".gz")

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
            pattern=getattr(args, "pattern", None),
            skip_bids_validation=args.skip_bids_validation,
            space=getattr(args, "space", None),
            resolution=getattr(args, "resolution", None),
            functional_connectome=getattr(args, "functional_connectome", None),
            structural_connectome=getattr(args, "structural_connectome", None),
            structural_tdi=getattr(args, "structural_tdi", None),
            parcel_atlases=args.parcel_atlases,
            skip_regional_damage=getattr(args, "skip_regional_damage", False),
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
        # Input must exist
        if not self.bids_dir.exists():
            raise ValueError(f"Input path does not exist: {self.bids_dir}")

        # Output directory cannot be same as input
        if self.output_dir.resolve() == self.bids_dir.resolve():
            raise ValueError("Output directory cannot be same as input path")

        # Analysis level must be 'participant'
        if self.analysis_level != "participant":
            raise ValueError(
                f"Invalid analysis level '{self.analysis_level}'. "
                "Only 'participant' is supported."
            )

        # For single file input, space and resolution are required
        if self.is_single_file:
            if not self.space:
                raise ValueError(
                    "--space is required when processing a single NIfTI file"
                )
            if not self.resolution:
                raise ValueError(
                    "--resolution is required when processing a single NIfTI file"
                )

        # Structural connectome validation (path vs name)
        if self.structural_connectome:
            connectome_path = Path(self.structural_connectome)
            if connectome_path.exists():
                # It's a path, TDI is required
                if not self.structural_tdi:
                    raise ValueError(
                        "--structural-tdi required when --structural-connectome is a file path"
                    )
                if not self.structural_tdi.exists():
                    raise ValueError(f"Structural TDI not found: {self.structural_tdi}")

        # Functional connectome path validation (if it's a path)
        if self.functional_connectome:
            connectome_path = Path(self.functional_connectome)
            if connectome_path.exists() or "/" in self.functional_connectome:
                # Looks like a path, validate it exists
                if not connectome_path.exists():
                    raise ValueError(
                        f"Functional connectome not found: {self.functional_connectome}"
                    )

        # Atlas directory must exist if provided
        if self.atlas_dir and not self.atlas_dir.exists():
            raise ValueError(f"Atlas directory not found: {self.atlas_dir}")

        # n_procs validation
        if self.n_procs < -1 or self.n_procs == 0:
            raise ValueError(
                f"--nprocs must be -1 (all CPUs) or >= 1, got {self.n_procs}"
            )
