"""
Base exception hierarchy for the lesion decoding toolkit.

All custom exceptions inherit from LdkError to enable precise error handling
while maintaining compatibility with standard Python exceptions.
"""


class LdkError(Exception):
    """Base exception for all ldk errors."""

    pass


class ValidationError(LdkError, ValueError):
    """Raised when data validation fails."""

    pass


class SpatialMismatchError(ValidationError):
    """Raised when spatial properties (affine, shape) don't match."""

    pass


class CoordinateSpaceError(LdkError, ValueError):
    """Raised when operations require specific coordinate space."""

    pass


class BIDSValidationError(LdkError, ValueError):
    """Raised when BIDS dataset structure is invalid."""

    pass


class ProvenanceError(LdkError, RuntimeError):
    """Raised when provenance tracking encounters issues."""

    pass


class NiftiLoadError(LdkError, IOError):
    """Raised when NIfTI file loading fails."""

    pass


class AnalysisError(LdkError, RuntimeError):
    """Raised when analysis computation fails."""

    pass
