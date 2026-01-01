"""
Lacuna CLI configuration module.

This module provides the CLIConfig dataclass for holding parsed
CLI arguments and validating them.

Classes:
    CLIConfig: Configuration from CLI arguments.
    ConnectomeConfig: Configuration for a connectome resource.

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


@dataclass
class ConnectomeConfig:
    """
    Configuration for a connectome resource.

    Attributes
    ----------
    path : Path
        Path to the connectome file (HDF5 for functional, .tck for structural).
    space : str
        Coordinate space (e.g., MNI152NLin6Asym).
    resolution : float, optional
        Resolution in mm (auto-detected from file if not provided).
    type : str
        Connectome type: "functional" or "structural".
    tdi_path : Path, optional
        Path to whole-brain TDI NIfTI (for structural connectomes only).
    """

    path: Path
    space: str
    type: str = "functional"
    resolution: float | None = None
    tdi_path: Path | None = None

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> ConnectomeConfig:
        """
        Create ConnectomeConfig from a dictionary.

        Parameters
        ----------
        name : str
            Name of the connectome (used for inferring type if not specified).
        config : dict
            Configuration dictionary with path, space, etc.

        Returns
        -------
        ConnectomeConfig
            Configuration instance.
        """
        path = Path(config["path"])

        # Infer type from extension or explicit config
        conn_type = config.get("type")
        if conn_type is None:
            if path.suffix == ".tck" or (path.is_dir() and "tck" in str(path).lower()):
                conn_type = "structural"
            else:
                conn_type = "functional"

        tdi_path = None
        if config.get("tdi_path"):
            tdi_path = Path(config["tdi_path"])

        return cls(
            path=path,
            space=config.get("space", "MNI152NLin6Asym"),
            type=conn_type,
            resolution=config.get("resolution"),
            tdi_path=tdi_path,
        )


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
    space : str, optional
        Coordinate space (required if not in filename).
    resolution : float, optional
        Voxel resolution in mm (required if not in filename).
    functional_connectome : str, optional
        Functional connectome name or path (from CLI).
    structural_connectome : str, optional
        Structural connectome name or path (from CLI).
    structural_tdi : Path, optional
        Path to whole-brain TDI NIfTI (from CLI).
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
    connectomes : dict
        Connectome resources keyed by name (new extensible format).
    analyses : dict
        Full analysis configurations from YAML.
    """

    # BIDS-Apps required arguments
    bids_dir: Path
    output_dir: Path
    analysis_level: str

    # BIDS filtering
    participant_label: list[str] | None = None
    session_id: list[str] | None = None
    pattern: str | None = None

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

    # Connectome resources keyed by name (new extensible format)
    connectomes: dict[str, ConnectomeConfig] = field(default_factory=dict)

    # Full analysis configurations from YAML
    analyses: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    @property
    def verbose(self) -> bool:
        """
        Check if verbose output is enabled for analysis classes.

        Returns
        -------
        bool
            True if verbose_count >= 1, False otherwise.
        """
        return self.verbose_count >= 1

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

        # Parse connectomes section (new extensible format)
        connectomes: dict[str, ConnectomeConfig] = {}
        connectomes_config = yaml_config.get("connectomes", {}) or {}
        for conn_name, conn_dict in connectomes_config.items():
            if conn_dict and isinstance(conn_dict, dict) and conn_dict.get("path"):
                connectomes[conn_name] = ConnectomeConfig.from_dict(conn_name, conn_dict)

        # Parse analyses section (new extensible format)
        analyses: dict[str, dict[str, Any]] = {}
        analyses_config = yaml_config.get("analyses", {}) or {}
        for analysis_name, analysis_dict in analyses_config.items():
            if analysis_dict is None:
                analyses[analysis_name] = {}
            elif isinstance(analysis_dict, dict):
                analyses[analysis_name] = analysis_dict.copy()

        # Determine if regional damage is skipped
        skip_rd = getattr(args, "skip_regional_damage", False)

        # Get parcel atlases from CLI
        parcel_atlases = getattr(args, "parcel_atlases", None)

        # Get connectome paths from CLI
        func_conn = getattr(args, "functional_connectome", None)
        if func_conn:
            func_conn = str(func_conn)

        struct_conn = getattr(args, "structural_tractogram", None)

        struct_tdi = getattr(args, "structural_tdi", None)

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
            connectomes=connectomes,
            analyses=analyses,
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

        # Validate connectomes (new format)
        for conn_name, conn_config in self.connectomes.items():
            if not conn_config.path.exists():
                raise ValueError(f"Connectome '{conn_name}' path not found: {conn_config.path}")
            if conn_config.tdi_path and not conn_config.tdi_path.exists():
                raise ValueError(f"Connectome '{conn_name}' TDI not found: {conn_config.tdi_path}")

        # n_procs validation
        if self.n_procs < -1 or self.n_procs == 0:
            raise ValueError(f"--nprocs must be -1 (all CPUs) or >= 1, got {self.n_procs}")
