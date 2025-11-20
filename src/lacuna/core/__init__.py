"""
Core data structures and utilities for the lesion decoding toolkit.
"""

from .exceptions import (
    AnalysisError,
    BIDSValidationError,
    CoordinateSpaceError,
    LacunaError,
    NiftiLoadError,
    ProvenanceError,
    SpatialMismatchError,
    ValidationError,
)
from .mask_data import MaskData
from .output import (
    AnalysisResult,
    ConnectivityMatrixResult,
    MiscResult,
    AtlasAggregationResult,
    SurfaceResult,
    TractogramResult,
    VoxelMapResult,
)
from .provenance import (
    create_provenance_record,
    merge_provenance,
    validate_provenance_record,
)
from .validation import (
    check_spatial_match,
    ensure_ras_plus,
    validate_affine,
    validate_nifti_image,
)

__all__ = [
    # Exceptions
    "LacunaError",
    "ValidationError",
    "SpatialMismatchError",
    "CoordinateSpaceError",
    "BIDSValidationError",
    "ProvenanceError",
    "NiftiLoadError",
    "AnalysisError",
    # Core data
    "MaskData",
    # Output classes
    "AnalysisResult",
    "VoxelMapResult",
    "AtlasAggregationResult",
    "ConnectivityMatrixResult",
    "SurfaceResult",
    "TractogramResult",
    "MiscResult",
    # Validation
    "validate_nifti_image",
    "ensure_ras_plus",
    "check_spatial_match",
    "validate_affine",
    # Provenance
    "create_provenance_record",
    "validate_provenance_record",
    "merge_provenance",
]
