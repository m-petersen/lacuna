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

    Attributes
    ----------
    name : str
        Unique identifier (e.g., "dTOR985")
    space : str
        Coordinate space (typically "MNI152NLin2009bAsym")
    resolution : float
        Resolution in mm (typically 1.0 for high-res tractography)
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


__all__ = [
    "StructuralConnectomeMetadata",
    "FunctionalConnectomeMetadata",
]
