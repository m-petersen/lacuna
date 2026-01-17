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


class DataContainer(ABC):
    """Abstract base class for unified data type containers.

    This is the base class for all data container types. It provides
    common functionality for metadata management and a consistent interface
    for accessing data.

    Subclasses implement specific data types:
    - VoxelMap: For 3D/4D brain maps (functional connectivity, disconnection)
    - ParcelData: For region-level aggregated data (atlas-based analysis)
    - ConnectivityMatrix: For connectivity matrices
    - SurfaceMesh: For surface-based data (vertices, faces)
    - Tractogram: For tractography streamlines
    - ScalarMetric: For summary statistics, scalars, and other data

    Attributes
    ----------
    name : str
        Name/identifier for this data container (e.g., "rmap", "zmap")
    metadata : dict
        Additional metadata about the data
    data_type : str
        Type identifier for the container (set by subclasses)

    Examples
    --------
    Subclasses are used to store analysis results:

    >>> # VoxelMap for brain maps
    >>> voxel_result = VoxelMap(
    ...     name="rmap",
    ...     data=nifti_img,
    ...     space="MNI152NLin6Asym",
    ...     resolution=2.0
    ... )
    >>> print(voxel_result.summary())

    >>> # ParcelData for region-level data
    >>> parcel_result = ParcelData(
    ...     name="damage_scores",
    ...     data={"V1": 0.8, "V2": 0.6},
    ...     aggregation_method="mean"
    ... )
    >>> top_regions = parcel_result.get_top_regions(n=5)
    """

    def __init__(self, name: str, metadata: dict[str, Any] | None = None):
        """Initialize base data container.

        Parameters
        ----------
        name : str
            Name/identifier for this container
        metadata : dict, optional
            Additional metadata about the data
        """
        self.name = name
        self.metadata = metadata or {}
        self.data_type = self.__class__.__name__

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
        return f"{self.data_type}(name='{self.name}', metadata={len(self.metadata)} items)"


@dataclass
class VoxelMap(DataContainer):
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

    Examples
    --------
    >>> import nibabel as nib
    >>> import numpy as np
    >>> # Create a sample brain map
    >>> data = np.random.randn(91, 109, 91)
    >>> img = nib.Nifti1Image(data, np.eye(4) * 2)

    >>> voxel_map = VoxelMap(
    ...     name="functional_connectivity",
    ...     data=img,
    ...     space="MNI152NLin6Asym",
    ...     resolution=2.0,
    ...     metadata={"seed": "PCC"}
    ... )
    >>> print(voxel_map.summary())
    functional_connectivity: (91, 109, 91) voxels, space=MNI152NLin6Asym, resolution=2.0mm
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
        return f"{self.name}: {shape} voxels, space={self.space}, resolution={self.resolution}mm"

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"VoxelMap("
            f"name='{self.name}', "
            f"shape={self.data.shape}, "
            f"space='{self.space}', "
            f"resolution={self.resolution})"
        )


@dataclass
class ParcelData(DataContainer):
    """Result container for atlas-based region aggregation.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    data : dict
        Dictionary mapping ROI identifiers to values
    region_labels : list of str, optional
        Ordered list of region label names (from atlas metadata)
    parcel_names : list of str, optional
        Names of atlases used in the analysis
    aggregation_method : str, optional
        Method used for aggregation (e.g., "mean", "percent")
    metadata : dict
        Additional metadata about the output

    Examples
    --------
    >>> # Create parcel data from atlas-based analysis
    >>> parcel_data = ParcelData(
    ...     name="damage_scores",
    ...     data={
    ...         "Visual_V1": 0.85,
    ...         "Motor_Primary": 0.42,
    ...         "Prefrontal_DLPFC": 0.15
    ...     },
    ...     parcel_names=["Schaefer100"],
    ...     aggregation_method="percent"
    ... )

    >>> # Get top damaged regions
    >>> top = parcel_data.get_top_regions(n=2)
    >>> print(top)
    {'Visual_V1': 0.85, 'Motor_Primary': 0.42}

    >>> print(parcel_data.summary())
    damage_scores: 3 regions from 1 atlases, method=percent
    """

    name: str
    data: dict[str, float]
    region_labels: list[str] | None = None
    parcel_names: list[str] | None = None
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
        atlas_info = f"{len(self.parcel_names)} atlases" if self.parcel_names else "unknown atlases"
        method_info = f", method={self.aggregation_method}" if self.aggregation_method else ""
        return f"{self.name}: {n_rois} regions from {atlas_info}{method_info}"

    def __repr__(self) -> str:
        """Return string representation."""
        method_str = f", method='{self.aggregation_method}'" if self.aggregation_method else ""
        return f"ParcelData(name='{self.name}', n_regions={len(self.data)}{method_str})"


@dataclass
class ConnectivityMatrix(DataContainer):
    """Result container for connectivity matrices.

    Stores a single connectivity matrix with optional region labels.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    matrix : np.ndarray
        Connectivity matrix (N x N)
    region_labels : list of str, optional
        Labels for matrix rows/columns
    matrix_type : str, optional
        Type of connectivity ("structural", "functional")
    metadata : dict
        Additional metadata about the output

    Examples
    --------
    >>> import numpy as np
    >>> # Create a structural connectivity matrix
    >>> conn_matrix = np.array([
    ...     [1.0, 0.8, 0.3],
    ...     [0.8, 1.0, 0.5],
    ...     [0.3, 0.5, 1.0]
    ... ])

    >>> conn = ConnectivityMatrix(
    ...     name="structural_connectivity",
    ...     matrix=conn_matrix,
    ...     region_labels=["V1", "V2", "MT"],
    ...     matrix_type="structural"
    ... )
    >>> print(conn.summary())
    structural_connectivity: 3x3, type=structural
    """

    name: str
    matrix: np.ndarray
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

        # Validate labels if provided
        if self.region_labels is not None:
            if len(self.region_labels) != self.matrix.shape[0]:
                raise ValueError(
                    f"Number of labels ({len(self.region_labels)}) "
                    f"must match matrix size ({self.matrix.shape[0]})"
                )

    def get_data(self) -> np.ndarray:
        """Get connectivity matrix."""
        return self.matrix

    def summary(self) -> str:
        """Get a summary description of this result."""
        n_regions = self.matrix.shape[0]
        type_info = f", type={self.matrix_type}" if self.matrix_type else ""
        return f"{self.name}: {n_regions}x{n_regions}{type_info}"

    def __repr__(self) -> str:
        """Return string representation."""
        type_str = f", type='{self.matrix_type}'" if self.matrix_type else ""
        return (
            f"ConnectivityMatrix(" f"name='{self.name}', " f"shape={self.matrix.shape}{type_str})"
        )


@dataclass
class SurfaceMesh(DataContainer):
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
            f"SurfaceMesh("
            f"name='{self.name}', "
            f"n_vertices={len(self.vertices)}, "
            f"n_faces={len(self.faces)})"
        )


@dataclass
class Tractogram(DataContainer):
    """Result container for tractography streamlines.

    Primary storage is path-based. Optionally stores streamlines in memory
    for immediate access. Use nibabel or dipy to load tractograms from disk.

    Attributes
    ----------
    name : str
        Name/identifier for this result
    tractogram_path : Path
        Path to saved tractogram file (.tck, .trk)
    streamlines : list or np.ndarray, optional
        Optional in-memory streamlines, each as (N_points, 3) array
    metadata : dict
        Additional metadata about the output
    """

    name: str
    tractogram_path: Path
    streamlines: list[np.ndarray] | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate tractogram data."""
        super().__init__(name=self.name, metadata=self.metadata)

        # Path is required
        if self.tractogram_path is None:
            raise ValueError("tractogram_path is required")

    def get_data(self) -> list[np.ndarray] | np.ndarray | Path:
        """Get tractogram data.

        Returns
        -------
        streamlines or path
            Returns in-memory streamlines if available, otherwise returns path.
            Use nibabel.streamlines.load() to load from path.

        Examples
        --------
        >>> result = Tractogram(name="tracts", tractogram_path=Path("tracts.tck"))
        >>> data = result.get_data()  # Returns Path
        >>> # Load with nibabel:
        >>> import nibabel as nib
        >>> tractogram = nib.streamlines.load(str(data))
        """
        if self.streamlines is not None:
            return self.streamlines
        return self.tractogram_path

    def summary(self) -> str:
        """Get a summary description of this result."""
        storage = "in-memory" if self.streamlines is not None else "on-disk"
        return f"{self.name}: {storage}, path={self.tractogram_path.name}"

    def __repr__(self) -> str:
        """Return string representation."""
        in_mem = self.streamlines is not None
        return (
            f"Tractogram("
            f"name='{self.name}', "
            f"path='{self.tractogram_path.name}', "
            f"in_memory={in_mem})"
        )


@dataclass
class ScalarMetric(DataContainer):
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
        """Initialize base class and infer data_type if needed."""
        # Store the data_type before calling super().__init__
        user_data_type = self.data_type
        super().__init__(name=self.name, metadata=self.metadata)

        # Restore or infer data_type
        if user_data_type is not None:
            self.data_type = user_data_type
        elif isinstance(self.data, (int, float, bool)):
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
        return f"ScalarMetric(name='{self.name}', data_type='{self.data_type}')"
