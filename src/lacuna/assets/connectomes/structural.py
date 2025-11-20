"""Structural connectome management.

This module provides registration and loading of structural connectomes
for tractography-based lesion network mapping (sLNM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lacuna.assets.base import AssetRegistry
from lacuna.assets.connectomes.registry import StructuralConnectomeMetadata

# Global registry for structural connectomes
_structural_connectome_registry = AssetRegistry[StructuralConnectomeMetadata](
    "structural connectome"
)


@dataclass
class StructuralConnectome:
    """Loaded structural connectome for sLNM analysis.

    Provides paths to tractogram and TDI files needed for
    StructuralNetworkMapping analysis.

    Attributes
    ----------
    metadata : StructuralConnectomeMetadata
        Connectome metadata
    tractogram_path : Path
        Path to .tck streamlines file
    tdi_path : Path
        Path to whole-brain TDI image
    template_path : Path | None
        Optional template image path
    """

    metadata: StructuralConnectomeMetadata
    tractogram_path: Path
    tdi_path: Path
    template_path: Path | None = None


def register_structural_connectome(
    name: str,
    space: str,
    resolution: float,
    tractogram_path: str | Path,
    tdi_path: str | Path,
    n_subjects: int,
    template_path: str | Path | None = None,
    description: str = "",
) -> None:
    """Register a structural connectome for sLNM analysis.

    Parameters
    ----------
    name : str
        Unique identifier (e.g., "dTOR985")
    space : str
        Coordinate space (e.g., "MNI152NLin2009bAsym")
    resolution : float
        Resolution in mm (typically 1.0)
    tractogram_path : str or Path
        Path to .tck whole-brain streamlines file
    tdi_path : str or Path
        Path to whole-brain TDI NIfTI file
    n_subjects : int
        Sample size
    template_path : str or Path, optional
        Path to template image for output grid
    description : str, optional
        Human-readable description

    Raises
    ------
    FileNotFoundError
        If tractogram or TDI files don't exist
    ValueError
        If file validation fails

    Examples
    --------
    >>> from lacuna.assets.connectomes import register_structural_connectome
    >>>
    >>> register_structural_connectome(
    ...     name="HCP842_dTOR",
    ...     space="MNI152NLin2009cAsym",
    ...     resolution=1.0,
    ...     tractogram_path="/data/dtor/hcp842_tractogram.tck",
    ...     tdi_path="/data/dtor/hcp842_tdi_1mm.nii.gz",
    ...     n_subjects=842,
    ...     description="HCP dTOR tractogram (842 subjects, 1mm)"
    ... )
    """
    # Convert to paths
    tractogram_path = Path(tractogram_path).resolve()
    tdi_path = Path(tdi_path).resolve()
    template_path = Path(template_path).resolve() if template_path else None

    # Validate files exist
    if not tractogram_path.exists():
        raise FileNotFoundError(f"Tractogram file not found: {tractogram_path}")
    if not tdi_path.exists():
        raise FileNotFoundError(f"TDI file not found: {tdi_path}")
    if template_path and not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Validate file extensions
    if tractogram_path.suffix != ".tck":
        raise ValueError(f"Expected .tck file, got: {tractogram_path.suffix}")
    if tdi_path.suffix not in [".nii", ".gz"]:
        raise ValueError(f"Expected .nii/.nii.gz file, got: {tdi_path.suffix}")

    # Create metadata
    metadata = StructuralConnectomeMetadata(
        name=name,
        space=space,
        resolution=resolution,
        description=description or f"Structural connectome: {name}",
        n_subjects=n_subjects,
        tractogram_path=tractogram_path,
        tdi_path=tdi_path,
        template_path=template_path,
    )

    # Register
    _structural_connectome_registry.register(metadata)


def unregister_structural_connectome(name: str) -> None:
    """Unregister a structural connectome.

    Parameters
    ----------
    name : str
        Connectome name

    Raises
    ------
    KeyError
        If connectome not registered

    Examples
    --------
    >>> from lacuna.assets.connectomes import unregister_structural_connectome
    >>> unregister_structural_connectome("HCP842_dTOR")
    """
    _structural_connectome_registry.unregister(name)


def list_structural_connectomes(
    atlas: str | None = None,
    space: str | None = None,
) -> list[StructuralConnectomeMetadata]:
    """List registered structural connectomes.

    Parameters
    ----------
    atlas : str, optional
        Filter by atlas name
    space : str, optional
        Filter by coordinate space

    Returns
    -------
    list[StructuralConnectomeMetadata]
        Matching connectomes

    Examples
    --------
    >>> from lacuna.assets.connectomes import list_structural_connectomes
    >>>
    >>> # List all
    >>> connectomes = list_structural_connectomes()
    >>>
    >>> # Filter by atlas
    >>> schaefer_connectomes = list_structural_connectomes(
    ...     atlas="Schaefer2018_100Parcels7Networks"
    ... )
    """
    return _structural_connectome_registry.list(atlas=atlas, space=space)


def load_structural_connectome(name: str) -> StructuralConnectome:
    """Load a structural connectome for sLNM analysis.

    Parameters
    ----------
    name : str
        Connectome name

    Returns
    -------
    StructuralConnectome
        Loaded connectome with paths ready for StructuralNetworkMapping

    Raises
    ------
    KeyError
        If connectome not registered

    Examples
    --------
    >>> from lacuna.assets.connectomes import load_structural_connectome
    >>> from lacuna.analysis import StructuralNetworkMapping
    >>>
    >>> connectome = load_structural_connectome("HCP842_dTOR")
    >>> analysis = StructuralNetworkMapping(
    ...     tractogram_path=connectome.tractogram_path,
    ...     whole_brain_tdi=connectome.tdi_path,
    ...     template=connectome.template_path
    ... )
    """
    metadata = _structural_connectome_registry.get(name)

    return StructuralConnectome(
        metadata=metadata,
        tractogram_path=metadata.tractogram_path,
        tdi_path=metadata.tdi_path,
        template_path=metadata.template_path,
    )


__all__ = [
    "StructuralConnectome",
    "register_structural_connectome",
    "unregister_structural_connectome",
    "list_structural_connectomes",
    "load_structural_connectome",
]
