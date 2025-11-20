"""Functional connectome management.

This module provides registration and loading of functional connectomes
for voxel-wise lesion network mapping (fLNM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py

from lacuna.assets.base import AssetRegistry
from lacuna.assets.connectomes.registry import FunctionalConnectomeMetadata

# Global registry for functional connectomes
_functional_connectome_registry = AssetRegistry[FunctionalConnectomeMetadata](
    "functional connectome"
)


@dataclass
class FunctionalConnectome:
    """Loaded functional connectome for fLNM analysis.

    Provides path to HDF5 file(s) with voxel-wise timeseries data
    needed for FunctionalNetworkMapping analysis.

    Attributes
    ----------
    metadata : FunctionalConnectomeMetadata
        Connectome metadata
    data_path : Path
        Path to .h5 file or directory with batch files
    is_batched : bool
        True if data_path points to directory with multiple files
    """

    metadata: FunctionalConnectomeMetadata
    data_path: Path
    is_batched: bool


def register_functional_connectome(
    name: str,
    space: str,
    resolution: float,
    data_path: str | Path,
    n_subjects: int,
    description: str = "",
) -> None:
    """Register a functional connectome for fLNM analysis.

    Supports both single HDF5 files and directories with batched files.

    HDF5 Required Structure:
    - 'timeseries': (n_subjects, n_timepoints, n_voxels) array
    - 'mask_indices': (3, n_voxels) or (n_voxels, 3) coordinates
    - 'mask_affine': (4, 4) affine matrix
    - 'mask_shape': Tuple attribute (e.g., (91, 109, 91))

    Parameters
    ----------
    name : str
        Unique identifier (e.g., "GSP1000")
    space : str
        Coordinate space (e.g., "MNI152NLin6Asym")
    resolution : float
        Resolution in mm (typically 2.0)
    data_path : str or Path
        Path to .h5 file or directory containing batch files
    n_subjects : int
        Total sample size
    description : str, optional
        Human-readable description

    Raises
    ------
    FileNotFoundError
        If data_path doesn't exist
    ValueError
        If HDF5 structure is invalid

    Examples
    --------
    >>> from lacuna.assets.connectomes import register_functional_connectome
    >>>
    >>> # Single file
    >>> register_functional_connectome(
    ...     name="GSP1000",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2.0,
    ...     data_path="/data/gsp/gsp1000_connectome.h5",
    ...     n_subjects=1000,
    ...     description="GSP1000 voxel-wise connectome"
    ... )
    >>>
    >>> # Batched directory
    >>> register_functional_connectome(
    ...     name="GSP1000_batched",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2.0,
    ...     data_path="/data/gsp/batches/",
    ...     n_subjects=1000,
    ...     description="GSP1000 voxel-wise connectome (batched)"
    ... )
    """
    # Convert to path
    data_path = Path(data_path).resolve()

    # Validate path exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # Determine if batched
    is_batched = data_path.is_dir()

    # Validate HDF5 structure
    if is_batched:
        # Find first .h5 file in directory
        h5_files = list(data_path.glob("*.h5"))
        if not h5_files:
            raise ValueError(f"No .h5 files found in directory: {data_path}")
        test_file = h5_files[0]
    else:
        if data_path.suffix != ".h5":
            raise ValueError(f"Expected .h5 file, got: {data_path.suffix}")
        test_file = data_path

    # Validate required datasets
    try:
        with h5py.File(test_file, "r") as f:
            required = ["timeseries", "mask_indices", "mask_affine"]
            missing = [k for k in required if k not in f]
            if missing:
                raise ValueError(
                    f"HDF5 file missing required datasets: {missing}. " f"Required: {required}"
                )

            # Check mask_shape attribute
            if "mask_shape" not in f.attrs:
                raise ValueError("HDF5 file must have 'mask_shape' attribute")
    except Exception as e:
        raise ValueError(f"Invalid HDF5 file structure: {e}") from e

    # Create metadata
    metadata = FunctionalConnectomeMetadata(
        name=name,
        space=space,
        resolution=resolution,
        description=description or f"Functional connectome: {name}",
        n_subjects=n_subjects,
        data_path=data_path,
        is_batched=is_batched,
    )

    # Register
    _functional_connectome_registry.register(metadata)


def unregister_functional_connectome(name: str) -> None:
    """Unregister a functional connectome.

    Parameters
    ----------
    name : str
        Connectome name

    Raises
    ------
    KeyError
        If connectome not registered
    """
    _functional_connectome_registry.unregister(name)


def list_functional_connectomes(
    space: str | None = None,
) -> list[FunctionalConnectomeMetadata]:
    """List registered functional connectomes.

    Parameters
    ----------
    space : str, optional
        Filter by coordinate space

    Returns
    -------
    list[FunctionalConnectomeMetadata]
        Matching connectomes

    Examples
    --------
    >>> from lacuna.assets.connectomes import list_functional_connectomes
    >>>
    >>> # List all
    >>> connectomes = list_functional_connectomes()
    >>>
    >>> # Filter by space
    >>> mni_connectomes = list_functional_connectomes(space="MNI152NLin6Asym")
    """
    return _functional_connectome_registry.list(space=space)


def load_functional_connectome(name: str) -> FunctionalConnectome:
    """Load a functional connectome for fLNM analysis.

    Parameters
    ----------
    name : str
        Connectome name

    Returns
    -------
    FunctionalConnectome
        Loaded connectome with path ready for FunctionalNetworkMapping

    Raises
    ------
    KeyError
        If connectome not registered

    Examples
    --------
    >>> from lacuna.assets.connectomes import load_functional_connectome
    >>> from lacuna.analysis import FunctionalNetworkMapping
    >>>
    >>> connectome = load_functional_connectome("GSP1000")
    >>> analysis = FunctionalNetworkMapping(
    ...     connectome_path=connectome.data_path,
    ...     method="boes"
    ... )
    """
    metadata = _functional_connectome_registry.get(name)

    return FunctionalConnectome(
        metadata=metadata,
        data_path=metadata.data_path,
        is_batched=metadata.is_batched,
    )


__all__ = [
    "FunctionalConnectome",
    "register_functional_connectome",
    "unregister_functional_connectome",
    "list_functional_connectomes",
    "load_functional_connectome",
]
