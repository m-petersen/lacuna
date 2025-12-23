"""
Lacuna CLI configuration module.

This module provides the CLIConfig dataclass for holding parsed
CLI arguments and validating them.

Classes:
    CLIConfig: Configuration from CLI arguments.

Functions:
    load_yaml_config: Load configuration from YAML file.
    generate_config_template: Generate a template YAML configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argparse import Namespace

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    ValueError
        If YAML parsing fails.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for config file support. " "Install with: pip install pyyaml"
        ) from e

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        try:
            config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}") from e

    return config


def generate_config_template() -> str:
    """
    Generate a template YAML configuration file.

    Returns
    -------
    str
        YAML template content.
    """
    template_path = Path(__file__).parent / "default_config.yaml"
    if template_path.exists():
        return template_path.read_text()

    # Fallback inline template
    return """\
# Lacuna Configuration File
# See documentation for full options

input: /path/to/bids_dataset
output: /path/to/output
mask_space: MNI152NLin6Asym

regional_damage:
  enabled: true
  atlases:
    - Schaefer2018_100Parcels7Networks

functional_network_mapping:
  enabled: false
  connectome_path: null

structural_network_mapping:
  enabled: false
  tractogram_path: null

n_jobs: 1
"""


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
    def from_args(cls, args: Namespace, yaml_config: dict[str, Any] | None = None) -> CLIConfig:
        """
        Create CLIConfig from parsed arguments and optional YAML config.

        YAML config values are used as defaults; CLI arguments override them.

        Parameters
        ----------
        args : Namespace
            Parsed arguments from argparse.
        yaml_config : dict, optional
            Configuration loaded from YAML file.

        Returns
        -------
        CLIConfig
            Configuration instance.
        """
        yaml_config = yaml_config or {}

        # Helper to get value: CLI arg takes precedence over YAML
        def get_val(cli_name: str, yaml_key: str, default=None, yaml_section: str | None = None):
            cli_val = getattr(args, cli_name, None)
            if cli_val is not None:
                return cli_val
            if yaml_section:
                section = yaml_config.get(yaml_section, {}) or {}
                return section.get(yaml_key, default)
            return yaml_config.get(yaml_key, default)

        # Extract analysis configs from YAML
        regional_damage = yaml_config.get("regional_damage", {}) or {}
        functional_mapping = yaml_config.get("functional_network_mapping", {}) or {}
        structural_mapping = yaml_config.get("structural_network_mapping", {}) or {}

        # Determine if regional damage is skipped
        skip_rd = getattr(args, "skip_regional_damage", False)
        if not skip_rd and regional_damage.get("enabled") is False:
            skip_rd = True

        # Get parcel atlases from CLI or YAML
        parcel_atlases = getattr(args, "parcel_atlases", None)
        if parcel_atlases is None:
            parcel_atlases = regional_damage.get("atlases")

        # Get connectome paths from CLI or YAML
        func_conn = getattr(args, "functional_connectome", None)
        if func_conn is None and functional_mapping.get("enabled"):
            func_conn = functional_mapping.get("connectome_path")
        if func_conn:
            func_conn = str(func_conn)

        struct_conn = getattr(args, "structural_tractogram", None)
        if struct_conn is None and structural_mapping.get("enabled"):
            struct_conn = structural_mapping.get("tractogram_path")

        struct_tdi = getattr(args, "structural_tdi", None)
        if struct_tdi is None:
            tdi_path = structural_mapping.get("tdi_path")
            if tdi_path:
                struct_tdi = Path(tdi_path)

        # Get space from CLI or YAML
        space = getattr(args, "mask_space", None) or yaml_config.get("mask_space")

        # Get subjects/sessions from CLI or YAML
        participants = getattr(args, "participant_label", None)
        if participants is None:
            participants = yaml_config.get("subjects") or None
            if participants == []:
                participants = None

        sessions = getattr(args, "session_id", None)
        if sessions is None:
            sessions = yaml_config.get("sessions") or None
            if sessions == []:
                sessions = None

        # Get work_dir from CLI or YAML
        work_dir = args.work_dir
        if yaml_config.get("work_dir"):
            work_dir = Path(yaml_config["work_dir"])

        return cls(
            bids_dir=args.bids_dir,
            output_dir=args.output_dir,
            analysis_level=args.analysis_level,
            participant_label=participants,
            session_id=sessions,
            pattern=getattr(args, "pattern", None) or yaml_config.get("pattern"),
            skip_bids_validation=getattr(args, "skip_bids_validation", False)
            or yaml_config.get("skip_bids_validation", False),
            space=space,
            resolution=None,  # Resolution is auto-detected from image affine
            functional_connectome=func_conn,
            structural_connectome=struct_conn,
            structural_tdi=struct_tdi,
            parcel_atlases=parcel_atlases,
            skip_regional_damage=skip_rd,
            atlas_dir=getattr(args, "atlas_dir", None),
            n_procs=getattr(args, "nprocs", None) or yaml_config.get("n_jobs", 1),
            work_dir=work_dir,
            verbose_count=getattr(args, "verbose_count", 0) or yaml_config.get("verbosity", 0),
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

        # For single file input, space is required (resolution is auto-detected from affine)
        if self.is_single_file:
            if not self.space:
                raise ValueError(
                    "--mask-space is required when processing a single NIfTI file "
                    "(cannot be inferred from BIDS filename)"
                )

        # Structural connectome validation
        if self.structural_connectome:
            connectome_path = Path(self.structural_connectome)
            if not connectome_path.exists():
                raise ValueError(f"Structural tractogram not found: {self.structural_connectome}")

        # Functional connectome path validation
        if self.functional_connectome:
            connectome_path = Path(self.functional_connectome)
            if not connectome_path.exists():
                raise ValueError(f"Functional connectome not found: {self.functional_connectome}")

        # Atlas directory must exist if provided
        if self.atlas_dir and not self.atlas_dir.exists():
            raise ValueError(f"Atlas directory not found: {self.atlas_dir}")

        # n_procs validation
        if self.n_procs < -1 or self.n_procs == 0:
            raise ValueError(f"--nprocs must be -1 (all CPUs) or >= 1, got {self.n_procs}")
