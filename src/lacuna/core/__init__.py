"""
Core data structures and utilities for the lesion decoding toolkit.
"""

from .data_types import (
    ConnectivityMatrix,
    DataContainer,
    ParcelData,
    ScalarMetric,
    SurfaceMesh,
    Tractogram,
    VoxelMap,
)
from .exceptions import (
    AnalysisError,
    BIDSValidationError,
    CoordinateSpaceError,
    EmptyMaskError,
    LacunaError,
    NiftiLoadError,
    ProvenanceError,
    SpatialMismatchError,
    ValidationError,
)
from .keys import (
    SOURCE_ABBREVIATIONS,
    build_result_key,
    get_source_abbreviation,
    parse_result_key,
)
from .pipeline import Pipeline, analyze
from .provenance import (
    create_provenance_record,
    merge_provenance,
    validate_provenance_record,
)
from .subject_data import SubjectData
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
    "EmptyMaskError",
    "SpatialMismatchError",
    "CoordinateSpaceError",
    "BIDSValidationError",
    "ProvenanceError",
    "NiftiLoadError",
    "AnalysisError",
    # Core data
    "SubjectData",
    # Pipeline
    "Pipeline",
    "analyze",
    # Output classes
    "DataContainer",
    "VoxelMap",
    "ParcelData",
    "ConnectivityMatrix",
    "SurfaceMesh",
    "Tractogram",
    "ScalarMetric",
    # Validation
    "validate_nifti_image",
    "ensure_ras_plus",
    "check_spatial_match",
    "validate_affine",
    # Provenance
    "create_provenance_record",
    "validate_provenance_record",
    "merge_provenance",
    # Result key helpers
    "SOURCE_ABBREVIATIONS",
    "build_result_key",
    "parse_result_key",
    "get_source_abbreviation",
]
