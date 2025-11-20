"""Analysis output management with polymorphic output types.

This module provides a hierarchy of output classes for different types of
analysis results, with automatic spatial alignment for voxel-level outputs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib


logger = logging.getLogger(__name__)


class AnalysisResult(ABC):
    """Abstract base class for all analysis outputs.

    This is the base class for all analysis result types. It provides
    common functionality for metadata management and a consistent interface
    for accessing results.

    Subclasses implement specific result types:
    - VoxelMapResult: For 3D/4D brain maps (functional connectivity, disconnection)
    - AtlasAggregationResult: For region-level aggregated data (atlas-based analysis)
    - ConnectivityMatrixResult: For connectivity matrices
    - SurfaceResult: For surface-based data (vertices, faces)
    - TractogramResult: For tractography streamlines
    - MiscResult: For summary statistics, scalars, and other data

    Attributes
    ----------
    name : str
        Name/identifier for this result (e.g., "correlation_map", "z_map")
    metadata : dict
        Additional metadata about the analysis result
    result_type : str
        Type identifier for the result (set by subclasses)
    """

    def __init__(self, name: str, metadata: dict[str, Any] | None = None):
        """Initialize base analysis result.

        Parameters
        ----------
        name : str
            Name/identifier for this result
        metadata : dict, optional
            Additional metadata about the result
        """
        self.name = name
        self.metadata = metadata or {}
        self.result_type = self.__class__.__name__

    @abstractmethod
    def get_data(self, **kwargs) -> Any:
        """Get the primary data from this result.

        Parameters
        ----------
        **kwargs
            Subclass-specific options for data retrieval

        Returns
        -------
        Any
            The primary data (type depends on subclass)
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """Get a summary description of this result.

        Returns
        -------
        str
            Human-readable summary
        """
        pass

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.result_type}(name='{self.name}', metadata={len(self.metadata)} items)"


@dataclass
class VoxelMapResult(AnalysisResult):
    """Result container for voxel-level brain maps.

    This class stores voxel-level analysis outputs (e.g., functional connectivity maps,
    structural disconnection maps) in their native computation space.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    data : nib.Nifti1Image
        Brain map in its computation space
    space : str
        Coordinate space identifier (e.g., 'MNI152NLin6Asym')
    resolution : float
        Resolution in mm (e.g., 1.0, 2.0)
    metadata : dict
        Additional metadata about the output
    """

    name: str
    data: nib.Nifti1Image
    space: str
    resolution: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize base class."""
        super().__init__(name=self.name, metadata=self.metadata)

    def get_data(self) -> nib.Nifti1Image:
        """Get the brain map data."""
        return self.data

    def summary(self) -> str:
        """Get a summary description of this result."""
        shape = self.data.shape
        return f"{self.name}: {shape} voxels, " f"space={self.space}@{self.resolution}mm"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"VoxelMapResult("
            f"name='{self.name}', "
            f"shape={self.data.shape}, "
            f"space='{self.space}@{self.resolution}mm')"
        )


@dataclass
class AtlasAggregationResult(AnalysisResult):
    """Result container for region-of-interest (ROI) level data.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    data : dict
        Dictionary mapping ROI identifiers to values
    atlas_names : list of str, optional
        Names of atlases used in the analysis
    aggregation_method : str, optional
        Method used for aggregation (e.g., "mean", "percent")
    metadata : dict
        Additional metadata about the output
    """

    name: str
    data: dict[str, float]
    atlas_names: list[str] | None = None
    aggregation_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize base class."""
        super().__init__(name=self.name, metadata=self.metadata)

    def get_data(self, atlas_filter: str | None = None) -> dict[str, float]:
        """Get ROI data, optionally filtered by atlas name."""
        if atlas_filter is None:
            return self.data

        return {
            roi: value for roi, value in self.data.items() if atlas_filter.lower() in roi.lower()
        }

    def get_top_regions(self, n: int = 10, ascending: bool = False) -> dict[str, float]:
        """Get top N regions by value."""
        sorted_items = sorted(self.data.items(), key=lambda x: x[1], reverse=not ascending)
        return dict(sorted_items[:n])

    def summary(self) -> str:
        """Get a summary description of this result."""
        n_rois = len(self.data)
        atlas_info = f"{len(self.atlas_names)} atlases" if self.atlas_names else "unknown atlases"
        method_info = f", method={self.aggregation_method}" if self.aggregation_method else ""
        return f"{self.name}: {n_rois} regions from {atlas_info}{method_info}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"AtlasAggregationResult(name='{self.name}', n_regions={len(self.data)})"


@dataclass
class ConnectivityMatrixResult(AnalysisResult):
    """Result container for connectivity matrices.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    matrix : np.ndarray
        Connectivity matrix (N x N)
    lesioned_matrix : np.ndarray, optional
        Connectivity matrix after lesion (N x N)
    region_labels : list of str, optional
        Labels for matrix rows/columns
    matrix_type : str, optional
        Type of connectivity ("structural", "functional")
    metadata : dict
        Additional metadata about the output
    """

    name: str
    matrix: np.ndarray
    lesioned_matrix: np.ndarray | None = None
    region_labels: list[str] | None = None
    matrix_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate matrix."""
        super().__init__(name=self.name, metadata=self.metadata)

        # Validate matrix shape
        if self.matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got shape {self.matrix.shape}")
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {self.matrix.shape}")

        # Validate lesioned matrix if provided
        if self.lesioned_matrix is not None:
            if self.lesioned_matrix.shape != self.matrix.shape:
                raise ValueError(
                    f"Lesioned matrix shape {self.lesioned_matrix.shape} "
                    f"must match intact matrix shape {self.matrix.shape}"
                )

        # Validate labels if provided
        if self.region_labels is not None:
            if len(self.region_labels) != self.matrix.shape[0]:
                raise ValueError(
                    f"Number of labels ({len(self.region_labels)}) "
                    f"must match matrix size ({self.matrix.shape[0]})"
                )

    def get_data(self, lesioned: bool = False) -> np.ndarray:
        """Get connectivity matrix."""
        if lesioned:
            if self.lesioned_matrix is None:
                raise ValueError("No lesioned matrix available")
            return self.lesioned_matrix
        return self.matrix

    def compute_disconnection(
        self, method: str = "absolute", normalize: bool = False
    ) -> np.ndarray:
        """Compute disconnection between intact and lesioned matrices."""
        if self.lesioned_matrix is None:
            raise ValueError("Cannot compute disconnection without lesioned matrix")

        if method == "absolute":
            disconnection = self.matrix - self.lesioned_matrix
        elif method == "relative":
            with np.errstate(divide="ignore", invalid="ignore"):
                disconnection = np.divide(
                    self.matrix - self.lesioned_matrix, self.matrix, where=(self.matrix != 0)
                )
                disconnection = np.nan_to_num(disconnection)
        elif method == "percent":
            with np.errstate(divide="ignore", invalid="ignore"):
                disconnection = np.divide(
                    (self.matrix - self.lesioned_matrix) * 100,
                    self.matrix,
                    where=(self.matrix != 0),
                )
                disconnection = np.nan_to_num(disconnection)
        else:
            raise ValueError(
                f"Invalid method '{method}'. Must be 'absolute', 'relative', or 'percent'"
            )

        if normalize:
            min_val, max_val = disconnection.min(), disconnection.max()
            if max_val > min_val:
                disconnection = (disconnection - min_val) / (max_val - min_val)

        return disconnection

    def summary(self) -> str:
        """Get a summary description of this result."""
        n_regions = self.matrix.shape[0]
        has_lesioned = (
            "with lesioned version" if self.lesioned_matrix is not None else "intact only"
        )
        type_info = f", type={self.matrix_type}" if self.matrix_type else ""
        return f"{self.name}: {n_regions}x{n_regions} {has_lesioned}{type_info}"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConnectivityMatrixResult("
            f"name='{self.name}', "
            f"shape={self.matrix.shape}, "
            f"has_lesioned={self.lesioned_matrix is not None})"
        )


@dataclass
class SurfaceResult(AnalysisResult):
    """Result container for surface-based data.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    vertices : np.ndarray
        Vertex coordinates (N x 3)
    faces : np.ndarray
        Triangle faces (M x 3, indices into vertices)
    vertex_data : np.ndarray, optional
        Per-vertex values (N,) - e.g., correlation, thickness
    hemisphere : str, optional
        Hemisphere identifier ("L", "R", "both")
    surface_type : str, optional
        Type of surface (e.g., "pial", "white", "inflated")
    metadata : dict
        Additional metadata about the output
    """

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    vertex_data: np.ndarray | None = None
    hemisphere: str | None = None
    surface_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate surface data."""
        super().__init__(name=self.name, metadata=self.metadata)

        # Validate vertices
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(f"Vertices must be N x 3, got shape {self.vertices.shape}")

        # Validate faces
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"Faces must be M x 3, got shape {self.faces.shape}")

        # Validate vertex data if provided
        if self.vertex_data is not None:
            if self.vertex_data.shape[0] != self.vertices.shape[0]:
                raise ValueError(
                    f"Vertex data length ({self.vertex_data.shape[0]}) "
                    f"must match number of vertices ({self.vertices.shape[0]})"
                )

    def get_data(self) -> np.ndarray:
        """Get per-vertex data."""
        if self.vertex_data is None:
            raise ValueError("No vertex data available")
        return self.vertex_data

    def get_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Get surface mesh."""
        return self.vertices, self.faces

    def summary(self) -> str:
        """Get a summary description of this result."""
        n_verts = len(self.vertices)
        n_faces = len(self.faces)
        has_data = "with vertex data" if self.vertex_data is not None else "mesh only"
        hemi_info = f", hemisphere={self.hemisphere}" if self.hemisphere else ""
        type_info = f", type={self.surface_type}" if self.surface_type else ""
        return f"{self.name}: {n_verts} vertices, {n_faces} faces, {has_data}{hemi_info}{type_info}"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SurfaceResult("
            f"name='{self.name}', "
            f"n_vertices={len(self.vertices)}, "
            f"n_faces={len(self.faces)})"
        )


@dataclass
class TractogramResult(AnalysisResult):
    """Result container for tractography streamlines.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    streamlines : list or np.ndarray
        List of streamlines, each as (N_points, 3) array
    tractogram_path : Path, optional
        Path to saved tractogram file (.tck, .trk)
    n_streamlines : int, optional
        Number of streamlines (computed if not provided)
    affine : np.ndarray, optional
        Affine transformation matrix for the tractogram
    metadata : dict
        Additional metadata about the output
    """

    name: str
    streamlines: list[np.ndarray] | np.ndarray | None = None
    tractogram_path: Path | None = None
    n_streamlines: int | None = None
    affine: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate tractogram data."""
        super().__init__(name=self.name, metadata=self.metadata)

        # Must have either streamlines or path
        if self.streamlines is None and self.tractogram_path is None:
            raise ValueError("Must provide either streamlines or tractogram_path")

        # Compute n_streamlines if not provided
        if self.n_streamlines is None and self.streamlines is not None:
            self.n_streamlines = len(self.streamlines)

    def get_data(self, load_if_needed: bool = True) -> list[np.ndarray] | np.ndarray:
        """Get tractogram streamlines.

        Parameters
        ----------
        load_if_needed : bool, default=True
            If streamlines are not in memory but path is available, load them

        Returns
        -------
        streamlines
            List or array of streamlines
        """
        if self.streamlines is not None:
            return self.streamlines

        if self.tractogram_path is not None and load_if_needed:
            raise NotImplementedError(
                "Loading streamlines from file not yet implemented. "
                "Use external library (e.g., nibabel, dipy) to load tractogram."
            )

        raise ValueError("No streamlines available and load_if_needed=False")

    def summary(self) -> str:
        """Get a summary description of this result."""
        n_str = self.n_streamlines if self.n_streamlines is not None else "unknown"
        in_memory = "in memory" if self.streamlines is not None else "on disk"
        path_info = f", path={self.tractogram_path.name}" if self.tractogram_path else ""
        return f"{self.name}: {n_str} streamlines, {in_memory}{path_info}"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TractogramResult("
            f"name='{self.name}', "
            f"n_streamlines={self.n_streamlines}, "
            f"in_memory={self.streamlines is not None})"
        )


@dataclass
class MiscResult(AnalysisResult):
    """Result container for miscellaneous data.

    This class handles summary statistics, scalar values, metadata,
    and any other data that doesn't fit into specific result types.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    data : Any
        The data (can be scalar, dict, list, etc.)
    data_type : str, optional
        Type description (e.g., "scalar", "summary_stats", "metadata")
    metadata : dict
        Additional metadata about the output
    """

    name: str
    data: Any
    data_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize base class."""
        super().__init__(name=self.name, metadata=self.metadata)

        # Infer data_type if not provided
        if self.data_type is None:
            if isinstance(self.data, (int, float, bool)):
                self.data_type = "scalar"
            elif isinstance(self.data, dict):
                self.data_type = "dictionary"
            elif isinstance(self.data, (list, tuple)):
                self.data_type = "sequence"
            else:
                self.data_type = "unknown"

    def get_data(self) -> Any:
        """Get the data."""
        return self.data

    def summary(self) -> str:
        """Get a summary description of this result."""
        if self.data_type == "scalar":
            return f"{self.name}: {self.data} ({type(self.data).__name__})"
        elif self.data_type == "dictionary":
            return f"{self.name}: dict with {len(self.data)} keys"
        elif self.data_type == "sequence":
            return f"{self.name}: {type(self.data).__name__} with {len(self.data)} items"
        else:
            return f"{self.name}: {self.data_type}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MiscResult(name='{self.name}', data_type='{self.data_type}')"
