"""Connectome metadata classes.

This module defines metadata structures for both structural and
functional connectomes used in lesion network mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lacuna.assets.base import SpatialAssetMetadata


@dataclass(frozen=True)
class StructuralConnectomeMetadata(SpatialAssetMetadata):
    """Metadata for a structural connectome (tractography-based).

    Used for structural lesion network mapping (sLNM). Requires:
    - Tractogram file (.tck format from MRtrix3)
    - Whole-brain track density image (TDI) as reference connectivity

    Note: Unlike functional connectomes, structural connectomes (tractograms)
    don't have an inherent voxel resolution - they exist in continuous 3D space.
    The output resolution is controlled by the StructuralNetworkMapping analysis.

    Attributes
    ----------
    name : str
        Unique identifier (e.g., "dTOR985")
    space : str
        Coordinate space (typically "MNI152NLin2009bAsym")
    resolution : float
        Resolution in mm (placeholder value, not used for tractograms)
    description : str
        Human-readable description
    n_subjects : int
        Sample size used to generate connectome
    modality : str
        Imaging modality (always "dwi")
    tractogram_path : Path
        Path to .tck streamlines file
    tdi_path : Path
        Path to whole-brain TDI NIfTI file
    template_path : Path | None
        Optional path to template image defining output grid
    """

    n_subjects: int = 0
    modality: str = "dwi"
    tractogram_path: Path | None = None
    tdi_path: Path | None = None
    template_path: Path | None = None

    def __repr__(self) -> str:
        """Concise representation showing only essential fields."""
        return (
            f"StructuralConnectomeMetadata("
            f"name={self.name!r}, "
            f"space={self.space!r}, "
            f"tractogram_path={self.tractogram_path})"
        )

    def validate(self) -> None:
        """Validate space only (tractograms don't have inherent resolution).

        Raises
        ------
        ValueError
            If space is invalid
        """
        from lacuna.core.spaces import SPACE_ALIASES, SUPPORTED_SPACES

        # Check if space is supported (either directly or as alias)
        if self.space not in SUPPORTED_SPACES and self.space not in SPACE_ALIASES:
            raise ValueError(f"Unsupported space: {self.space}. Supported: {SUPPORTED_SPACES}")


@dataclass(frozen=True)
class FunctionalConnectomeMetadata(SpatialAssetMetadata):
    """Metadata for a functional connectome (voxel-wise timeseries).

    Used for functional lesion network mapping (fLNM). Requires HDF5 file(s)
    containing whole-brain voxel-wise BOLD timeseries data.

    HDF5 structure:
    - 'timeseries': (n_subjects, n_timepoints, n_voxels) array
    - 'mask_indices': (3, n_voxels) or (n_voxels, 3) brain mask coordinates
    - 'mask_affine': (4, 4) affine transformation matrix
    - 'mask_shape': Tuple as attribute (e.g., (91, 109, 91))

    Attributes
    ----------
    name : str
        Unique identifier (e.g., "GSP1000")
    space : str
        Coordinate space (typically "MNI152NLin6Asym")
    resolution : float
        Resolution in mm (typically 2.0 for functional data)
    description : str
        Human-readable description
    n_subjects : int
        Sample size in connectome
    modality : str
        Imaging modality (always "bold")
    data_path : Path
        Path to .h5 file or directory containing batch files
    is_batched : bool
        True if data_path is directory with multiple HDF5 files
    """

    n_subjects: int = 0
    modality: str = "bold"
    data_path: Path | None = None
    is_batched: bool = False

    def __repr__(self) -> str:
        """Concise representation showing only essential fields."""
        return (
            f"FunctionalConnectomeMetadata("
            f"name={self.name!r}, "
            f"space={self.space!r}, "
            f"resolution={self.resolution}, "
            f"n_subjects={self.n_subjects}, "
            f"data_path={self.data_path})"
        )


__all__ = [
    "StructuralConnectomeMetadata",
    "FunctionalConnectomeMetadata",
]
