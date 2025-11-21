"""Analysis output management with polymorphic output types.

This module provides a hierarchy of output classes for different types of
analysis results, with automatic spatial alignment for voxel-level outputs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib
    from lacuna.core.spaces import CoordinateSpace

logger = logging.getLogger(__name__)


class AnalysisResult(ABC):
    """Abstract base class for all analysis outputs.
    
    This is the base class for all analysis result types. It provides
    common functionality for metadata management and a consistent interface
    for accessing results.
    
    Subclasses implement specific result types:
    - VoxelMapResult: For 3D/4D brain maps (functional connectivity, disconnection)
    - ROIResult: For region-level aggregated data (atlas-based analysis)
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
    
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize base class."""
        super().__init__(name=self.name, metadata=self.metadata)
    
    def get_data(self) -> nib.Nifti1Image:
        """Get the brain map data."""
        return self.data
    
    def summary(self) -> str:
        """Get a summary description of this result."""
        shape = self.data.shape
        return (
            f"{self.name}: {shape} voxels, "
            f"space={self.space}@{self.resolution}mm"
        )
    
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
    """Result container for atlas-based region aggregation.
    
    Attributes
    ----------
    name : str
        Name/identifier for this result
    data : dict
        Dictionary mapping ROI identifiers to values
    region_labels : list of str, optional
        Ordered list of region label names (from atlas metadata)
    atlas_names : list of str, optional
        Names of atlases used in the analysis
    aggregation_method : str, optional
        Method used for aggregation (e.g., "mean", "percent")
    metadata : dict
        Additional metadata about the output
    """
    
    name: str
    data: Dict[str, float]
    region_labels: Optional[List[str]] = None
    atlas_names: Optional[List[str]] = None
    aggregation_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize base class."""
        super().__init__(name=self.name, metadata=self.metadata)
    
    def get_data(self, atlas_filter: Optional[str] = None) -> Dict[str, float]:
        """Get ROI data, optionally filtered by atlas name."""
        if atlas_filter is None:
            return self.data
        
        return {
            roi: value 
            for roi, value in self.data.items() 
            if atlas_filter.lower() in roi.lower()
        }
    
    def get_top_regions(self, n: int = 10, ascending: bool = False) -> Dict[str, float]:
        """Get top N regions by value."""
        sorted_items = sorted(
            self.data.items(), 
            key=lambda x: x[1], 
            reverse=not ascending
        )
        return dict(sorted_items[:n])
    
    def summary(self) -> str:
        """Get a summary description of this result."""
        n_rois = len(self.data)
        atlas_info = f"{len(self.atlas_names)} atlases" if self.atlas_names else "unknown atlases"
        method_info = f", method={self.aggregation_method}" if self.aggregation_method else ""
        return f"{self.name}: {n_rois} regions from {atlas_info}{method_info}"
    
    def __repr__(self) -> str:
        """Return string representation."""
        method_str = f", method='{self.aggregation_method}'" if self.aggregation_method else ""
        return f"AtlasAggregationResult(name='{self.name}', n_regions={len(self.data)}{method_str})"


@dataclass
class ConnectivityMatrixResult(AnalysisResult):
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
    """
    
    name: str
    matrix: np.ndarray
    region_labels: Optional[List[str]] = None
    matrix_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
            f"ConnectivityMatrixResult("
            f"name='{self.name}', "
            f"shape={self.matrix.shape}{type_str})")


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
    vertex_data: Optional[np.ndarray] = None
    hemisphere: Optional[str] = None
    surface_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize and validate surface data."""
        super().__init__(name=self.name, metadata=self.metadata)
        
        # Validate vertices
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"Vertices must be N x 3, got shape {self.vertices.shape}"
            )
        
        # Validate faces
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(
                f"Faces must be M x 3, got shape {self.faces.shape}"
            )
        
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
    streamlines: Optional[Union[List[np.ndarray], np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize and validate tractogram data."""
        super().__init__(name=self.name, metadata=self.metadata)
        
        # Path is required
        if self.tractogram_path is None:
            raise ValueError("tractogram_path is required")
    
    def get_data(self) -> Union[List[np.ndarray], np.ndarray, Path]:
        """Get tractogram data.
        
        Returns
        -------
        streamlines or path
            Returns in-memory streamlines if available, otherwise returns path.
            Use nibabel.streamlines.load() to load from path.
        
        Examples
        --------
        >>> result = TractogramResult(name="tracts", tractogram_path=Path("tracts.tck"))
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
            f"TractogramResult("
            f"name='{self.name}', "
            f"path='{self.tractogram_path.name}', "
            f"in_memory={in_mem})")


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
    data_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
