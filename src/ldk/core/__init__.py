"""Core module for lesion decoding toolkit."""

from .exceptions import (
    AnalysisError,
    BIDSValidationError,
    CoordinateSpaceError,
    LdkError,
    NiftiLoadError,
    ProvenanceError,
    SpatialMismatchError,
    ValidationError,
)
from .provenance import create_provenance_record, validate_provenance_record
from .validation import (
    check_spatial_match,
    ensure_ras_plus,
    validate_affine,
    validate_nifti_image,
)

__all__ = [
    # Exceptions
    "LdkError",
    "ValidationError",
    "SpatialMismatchError",
    "CoordinateSpaceError",
    "BIDSValidationError",
    "ProvenanceError",
    "NiftiLoadError",
    "AnalysisError",
    # Validation
    "validate_nifti_image",
    "ensure_ras_plus",
    "check_spatial_match",
    "validate_affine",
    # Provenance
    "create_provenance_record",
    "validate_provenance_record",
]
