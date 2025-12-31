"""
Base classes and registry for connectome downloaders.

This module provides:
- ConnectomeSource: Configuration for a fetchable connectome
- FetchConfig: User-specified fetch configuration
- FetchProgress: Progress tracking for downloads
- FetchResult: Outcome of a fetch operation
- CONNECTOME_SOURCES: Registry of available connectomes
- get_api_key: API key resolution helper
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    import argparse


@dataclass
class ConnectomeSource:
    """Configuration for a fetchable connectome source."""

    name: str
    """Unique identifier (e.g., 'gsp1000', 'dtor985')."""

    display_name: str
    """Human-readable name (e.g., 'GSP1000 Functional Connectome')."""

    type: Literal["functional", "structural"]
    """Connectome type determining processing pipeline."""

    description: str
    """User-facing description of the connectome."""

    source_type: Literal["dataverse", "figshare"]
    """Download source requiring specific authentication/handling."""

    # Dataverse-specific
    persistent_id: str | None = None
    """DOI for Dataverse datasets (e.g., 'doi:10.7910/DVN/ILXIKS')."""

    dataverse_server: str = "https://dataverse.harvard.edu"
    """Dataverse server URL."""

    # Figshare-specific
    download_url: str | None = None
    """Direct download URL for Figshare files."""

    # Processing
    default_batches: int = 10
    """Default number of HDF5 batches (functional only)."""

    requires_mask: bool = False
    """Whether brain mask is needed for processing."""

    mask_url: str | None = None
    """URL to download brain mask if required."""

    # Metadata
    n_subjects: int = 0
    """Number of subjects in the connectome."""

    space: str = "MNI152NLin6Asym"
    """Coordinate space."""

    estimated_size_gb: float = 0.0
    """Estimated download size in GB for user information."""


@dataclass
class FetchConfig:
    """Configuration for a connectome fetch operation."""

    connectome: str
    """Connectome name to fetch (e.g., 'gsp1000', 'dtor985')."""

    output_dir: Path
    """Directory for processed output files."""

    # Authentication
    api_key: str | None = None
    """Dataverse API key (for GSP1000). Can also use DATAVERSE_API_KEY env var."""

    # Processing options
    batches: int = 10
    """Number of HDF5 batch files for functional connectomes."""

    keep_original: bool = True
    """Keep original downloaded files after processing."""

    # Registration
    register: bool = True
    """Automatically register connectome after processing."""

    register_name: str | None = None
    """Custom name for registration. Defaults to source name (e.g., 'GSP1000')."""

    # Behavior
    force: bool = False
    """Overwrite existing files and registrations."""

    resume: bool = True
    """Resume interrupted downloads."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> FetchConfig:
        """Create config from CLI arguments."""
        return cls(
            connectome=getattr(args, "connectome", ""),
            output_dir=Path(getattr(args, "output_dir", ".")),
            api_key=getattr(args, "api_key", None),
            batches=getattr(args, "batches", 10),
            keep_original=not getattr(args, "no_keep_original", False),
            register=not getattr(args, "no_register", False),
            register_name=getattr(args, "register_name", None),
            force=getattr(args, "force", False),
            resume=getattr(args, "resume", True),
        )

    def get_api_key(self) -> str | None:
        """Get API key from config, env var, or config file."""
        if self.api_key:
            return self.api_key
        if key := os.environ.get("DATAVERSE_API_KEY"):
            return key
        # Check config file
        return _load_config_file_key()


@dataclass
class FetchProgress:
    """Progress information for fetch operations."""

    phase: Literal["download", "processing", "registration"]
    """Current operation phase."""

    current_file: str
    """Name of file currently being processed."""

    files_completed: int
    """Number of files completed."""

    files_total: int
    """Total number of files to process."""

    bytes_transferred: int = 0
    """Bytes transferred in current download."""

    bytes_total: int = 0
    """Total bytes for current download."""

    message: str = ""
    """Human-readable status message."""

    @property
    def percent_complete(self) -> float:
        """Overall percentage completion."""
        if self.files_total == 0:
            return 0.0
        return (self.files_completed / self.files_total) * 100

    @property
    def download_percent(self) -> float:
        """Current file download percentage."""
        if self.bytes_total == 0:
            return 0.0
        return (self.bytes_transferred / self.bytes_total) * 100


@dataclass
class FetchResult:
    """Result of a connectome fetch operation."""

    success: bool
    """Whether the operation completed successfully."""

    connectome_name: str
    """Name of the fetched connectome."""

    output_dir: Path
    """Directory containing processed files."""

    output_files: list[Path] = field(default_factory=list)
    """List of created output files."""

    registered: bool = False
    """Whether the connectome was registered."""

    register_name: str | None = None
    """Name used for registration, or None if not registered."""

    duration_seconds: float = 0.0
    """Total operation time in seconds."""

    download_time_seconds: float = 0.0
    """Time spent downloading."""

    processing_time_seconds: float = 0.0
    """Time spent processing."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal warnings encountered."""

    error: str | None = None
    """Error message if success=False."""

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.success:
            return (
                f"✅ Successfully fetched {self.connectome_name}\n"
                f"   Output: {self.output_dir}\n"
                f"   Files: {len(self.output_files)}\n"
                f"   Registered as: {self.register_name or 'not registered'}\n"
                f"   Time: {self.download_time_seconds:.1f}s download, "
                f"{self.processing_time_seconds:.1f}s processing"
            )
        return f"❌ Failed to fetch {self.connectome_name}: {self.error}"


class DownloaderProtocol(Protocol):
    """Protocol for source-specific downloaders."""

    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> list[Path]:
        """Download files to output path."""
        ...


class BaseDownloader(ABC):
    """Abstract base class for downloaders."""

    def __init__(self, source: ConnectomeSource):
        """
        Initialize downloader with source configuration.

        Parameters
        ----------
        source : ConnectomeSource
            Configuration for the connectome source.
        """
        self.source = source

    @abstractmethod
    def download(
        self,
        output_path: Path,
        progress_callback: Callable[[FetchProgress], None] | None = None,
    ) -> list[Path]:
        """
        Download files to output path.

        Parameters
        ----------
        output_path : Path
            Directory to download files to.
        progress_callback : callable, optional
            Function called with FetchProgress updates.

        Returns
        -------
        list[Path]
            List of downloaded file paths.
        """
        ...


def _load_config_file_key() -> str | None:
    """Load API key from config file if it exists."""
    config_path = Path.home() / ".config" / "lacuna" / "config.yaml"
    if not config_path.exists():
        return None

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        if config and "dataverse" in config:
            api_key: str | None = config["dataverse"].get("api_key")
            return api_key
    except Exception:
        pass
    return None


def get_api_key(cli_key: str | None = None) -> str | None:
    """
    Get API key using priority order: CLI > env var > config file.

    Parameters
    ----------
    cli_key : str, optional
        API key provided via CLI argument.

    Returns
    -------
    str or None
        The API key, or None if not found.
    """
    if cli_key:
        return cli_key
    if key := os.environ.get("DATAVERSE_API_KEY"):
        return key
    return _load_config_file_key()


# Registry of available connectomes
CONNECTOME_SOURCES: dict[str, ConnectomeSource] = {
    "gsp1000": ConnectomeSource(
        name="gsp1000",
        display_name="GSP1000 Functional Connectome",
        type="functional",
        description=(
            "Brain Genomics Superstruct Project 1000-subject resting-state fMRI dataset. "
            "Provides functional connectivity templates for lesion network mapping."
        ),
        source_type="dataverse",
        persistent_id="doi:10.7910/DVN/ILXIKS",
        dataverse_server="https://dataverse.harvard.edu",
        default_batches=10,
        n_subjects=1000,
        space="MNI152NLin6Asym",
        estimated_size_gb=100.0,
    ),
    "dtor985": ConnectomeSource(
        name="dtor985",
        display_name="dTOR985 Structural Tractogram",
        type="structural",
        description=(
            "Diffusion Tensor Imaging Open Resource 985-subject whole-brain tractogram. "
            "Provides structural connectivity template for lesion network mapping."
        ),
        source_type="figshare",
        download_url="https://figshare.com/ndownloader/files/49587541",
        n_subjects=985,
        space="MNI152NLin6Asym",
        estimated_size_gb=10.0,
    ),
}
