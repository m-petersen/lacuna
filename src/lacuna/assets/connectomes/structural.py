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

    Provides tractogram path for StructuralNetworkMapping analysis.
    TDI is computed on-the-fly during analysis (with optional caching).

    Attributes
    ----------
    metadata : StructuralConnectomeMetadata
        Connectome metadata
    tractogram_path : Path
        Path to .tck streamlines file
    template_path : Path | None
        Optional template image path
    """

    metadata: StructuralConnectomeMetadata
    tractogram_path: Path
    template_path: Path | None = None


def register_structural_connectome(
    name: str,
    space: str,
    tractogram_path: str | Path,
    template_path: str | Path | None = None,
    description: str = "",
) -> None:
    """Register a structural connectome for sLNM analysis.

    TDI is computed on-the-fly during analysis. Use cache_tdi=True (default) in
    StructuralNetworkMapping to cache computed TDIs for reuse, or cache_tdi=False
    to compute without caching.

    Note: Unlike functional connectomes, structural connectomes (tractograms) don't
    have an inherent voxel resolution - they exist in continuous 3D space. The output
    resolution is controlled by the `output_resolution` parameter in
    StructuralNetworkMapping analysis.

    Parameters
    ----------
    name : str
        Unique identifier (e.g., "dTOR985")
    space : str
        Coordinate space (e.g., "MNI152NLin2009bAsym")
    tractogram_path : str or Path
        Path to .tck whole-brain streamlines file
    template_path : str or Path, optional
        Path to template image for output grid
    description : str, optional
        Human-readable description

    Raises
    ------
    FileNotFoundError
        If tractogram file doesn't exist
    ValueError
        If file validation fails

    Examples
    --------
    >>> from lacuna.assets.connectomes import register_structural_connectome
    >>>
    >>> # Register tractogram (TDI computed on-the-fly during analysis)
    >>> register_structural_connectome(
    ...     name="dTOR985",
    ...     space="MNI152NLin2009cAsym",
    ...     tractogram_path="/data/dtor/dTOR985_tractogram.tck",
    ...     description="dTOR tractogram - TDI computed on-demand"
    ... )
    """
    # Convert to paths
    tractogram_path = Path(tractogram_path).resolve()
    template_path = Path(template_path).resolve() if template_path else None

    # Validate tractogram exists
    if not tractogram_path.exists():
        raise FileNotFoundError(f"Tractogram file not found: {tractogram_path}")

    # Validate template exists if provided
    if template_path and not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    # Validate file extensions
    if tractogram_path.suffix != ".tck":
        raise ValueError(f"Expected .tck file, got: {tractogram_path.suffix}")

    # Create metadata
    # Note: resolution=0.0 as placeholder since tractograms don't have inherent voxel
    # resolution. Output resolution is controlled by StructuralNetworkMapping.output_resolution
    metadata = StructuralConnectomeMetadata(
        name=name,
        space=space,
        resolution=0.0,  # Tractograms don't have inherent voxel resolution
        description=description or f"Structural connectome: {name}",
        tractogram_path=tractogram_path,
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
    >>> unregister_structural_connectome("dTOR985")
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
        Loaded connectome with tractogram path ready for StructuralNetworkMapping.
        TDI will be computed on-the-fly during analysis.

    Raises
    ------
    KeyError
        If connectome not registered

    Examples
    --------
    >>> from lacuna.assets.connectomes import load_structural_connectome
    >>> from lacuna.analysis import StructuralNetworkMapping
    >>>
    >>> connectome = load_structural_connectome("dTOR985")
    >>> analysis = StructuralNetworkMapping(
    ...     connectome_name="dTOR985",
    ...     cache_tdi=True  # Cache computed TDI for reuse
    ... )
    """
    metadata = _structural_connectome_registry.get(name)

    return StructuralConnectome(
        metadata=metadata,
        tractogram_path=metadata.tractogram_path,
        template_path=metadata.template_path,
    )


__all__ = [
    "StructuralConnectome",
    "register_structural_connectome",
    "unregister_structural_connectome",
    "list_structural_connectomes",
    "load_structural_connectome",
]
