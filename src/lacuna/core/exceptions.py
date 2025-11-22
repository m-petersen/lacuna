"""
from __future__ import annotations

Base exception hierarchy for the lesion decoding toolkit.

All custom exceptions inherit from LacunaError to enable precise error handling
while maintaining compatibility with standard Python exceptions.
"""


class LacunaError(Exception):
    """Base exception for all lacuna errors."""

    pass


class ValidationError(LacunaError, ValueError):
    """Raised when data validation fails."""

    pass


class SpatialMismatchError(ValidationError):
    """Raised when spatial properties (affine, shape) don't match."""

    pass


class CoordinateSpaceError(LacunaError, ValueError):
    """Raised when operations require specific coordinate space."""

    pass


class BIDSValidationError(LacunaError, ValueError):
    """Raised when BIDS dataset structure is invalid."""

    pass


class ProvenanceError(LacunaError, RuntimeError):
    """Raised when provenance tracking encounters issues."""

    pass


class NiftiLoadError(LacunaError, IOError):
    """Raised when NIfTI file loading fails."""

    pass


class AnalysisError(LacunaError, RuntimeError):
    """Raised when analysis computation fails."""

    pass


class AtlasNotFoundError(LacunaError, FileNotFoundError):
    """Raised when atlas cannot be found or resolved."""

    pass


class ConnectomeNotFoundError(LacunaError, FileNotFoundError):
    """Raised when connectome cannot be found or resolved."""

    pass


# Spatial coordinate space exceptions


class SpaceDetectionError(LacunaError):
    """Raised when coordinate space cannot be detected from file."""

    def __init__(self, filepath, attempted_methods):
        self.filepath = filepath
        self.attempted_methods = attempted_methods
        message = (
            f"Could not detect coordinate space for '{filepath}'. "
            f"Attempted methods: {', '.join(attempted_methods)}. "
            f"Please specify space explicitly using the 'space' parameter."
        )
        super().__init__(message)


class SpaceMismatchError(ValidationError):
    """Raised when declared space doesn't match detected space."""

    def __init__(
        self, declared_space: str, detected_space: "str | None", filepath, affine_difference: float
    ):
        self.declared_space = declared_space
        self.detected_space = detected_space
        self.filepath = filepath
        self.affine_difference = affine_difference

        message = (
            f"Space mismatch for '{filepath}': "
            f"declared='{declared_space}', detected='{detected_space}' "
            f"(affine difference: {affine_difference:.6f}). "
            f"Set require_match=False to override validation."
        )
        super().__init__(message)


class TransformNotAvailableError(LacunaError):
    """Raised when spatial transform is not available."""

    def __init__(self, source_space: str, target_space: str, supported_transforms: list):
        self.source_space = source_space
        self.target_space = target_space
        message = (
            f"No transform available from '{source_space}' to '{target_space}'. "
            f"Supported transforms: {supported_transforms}"
        )
        super().__init__(message)


class TransformDownloadError(LacunaError):
    """Raised when transform file cannot be downloaded."""

    def __init__(self, source_space: str, target_space: str, reason: str):
        self.source_space = source_space
        self.target_space = target_space
        self.reason = reason
        message = (
            f"Failed to download transform from '{source_space}' to '{target_space}': {reason}. "
            f"Check network connection or download manually from TemplateFlow."
        )
        super().__init__(message)
