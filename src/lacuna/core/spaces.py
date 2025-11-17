"""Coordinate space representation and detection for neuroimaging data.

This module defines the core coordinate space abstractions and provides
automatic detection from filenames and image headers.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from lacuna.core.exceptions import (
    SpaceDetectionError,
    SpaceMismatchError,
    TransformNotAvailableError,
)

# Supported MNI coordinate spaces
SUPPORTED_SPACES = [
    "MNI152NLin6Asym",
    "MNI152NLin2009cAsym",
    "MNI152NLin2009aAsym",  # Alias for cAsym
    "MNI152NLin2009bAsym",  # Alias for cAsym
    "native",
]

# Space aliases - anatomically identical spaces that require no transformation
# Only resampling is needed if resolutions differ
SPACE_ALIASES = {
    "MNI152NLin2009aAsym": "MNI152NLin2009cAsym",
    "MNI152NLin2009bAsym": "MNI152NLin2009cAsym",
}


def canonicalize_space_variant(space: str) -> str:
    """Canonicalize MNI space variant identifiers to their standard forms.
    
    Handles space variant aliases by converting them to their canonical representations:
    - MNI152NLin2009aAsym -> MNI152NLin2009cAsym
    - MNI152NLin2009bAsym -> MNI152NLin2009cAsym  
    - MNI152NLin2009cAsym -> MNI152NLin2009cAsym (unchanged)
    - Other spaces remain unchanged
    
    This canonicalization enables consistent space comparisons and transform lookups.
    Note: This is NOT related to spatial normalization (warping images to standard space).
    
    Args:
        space: Space identifier string
        
    Returns:
        Canonical space identifier
        
    Examples:
        >>> canonicalize_space_variant("MNI152NLin2009aAsym")
        'MNI152NLin2009cAsym'
        >>> canonicalize_space_variant("MNI152NLin6Asym")
        'MNI152NLin6Asym'
    """
    return SPACE_ALIASES.get(space, space)


def spaces_are_equivalent(space1: str, space2: str) -> bool:
    """Check if two space identifiers are anatomically equivalent.
    
    Returns True if spaces are identical or are aliases of each other
    (e.g., aAsym and cAsym are equivalent).
    
    Args:
        space1: First space identifier
        space2: Second space identifier
        
    Returns:
        True if spaces are equivalent, False otherwise
        
    Examples:
        >>> spaces_are_equivalent("MNI152NLin2009aAsym", "MNI152NLin2009cAsym")
        True
        >>> spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        False
    """
    return canonicalize_space_variant(space1) == canonicalize_space_variant(space2)


def validate_space_compatibility(
    actual_space: str,
    expected_space: str,
    context: str = "operation",
    suggest_transform: bool = False
) -> None:
    """Validate that actual space is compatible with expected space.
    
    Raises ValueError if spaces are incompatible (not equivalent).
    Handles space aliases automatically.
    
    Args:
        actual_space: The actual space of the data
        expected_space: The expected/required space
        context: Description of operation for error messages
        suggest_transform: Whether to suggest transformation in error message
        
    Raises:
        ValueError: If spaces are incompatible
        
    Examples:
        >>> validate_space_compatibility("MNI152NLin2009aAsym", "MNI152NLin2009cAsym", "test")
        # No error - spaces are equivalent
        >>> validate_space_compatibility("native", "MNI152NLin6Asym", "test")
        ValueError: Space mismatch in test: got 'native', expected 'MNI152NLin6Asym'
    """
    if not spaces_are_equivalent(actual_space, expected_space):
        msg = (
            f"Space mismatch in {context}: "
            f"got '{actual_space}', expected '{expected_space}'"
        )
        
        if suggest_transform:
            # Check if transform might be available
            norm_actual = canonicalize_space_variant(actual_space)
            norm_expected = canonicalize_space_variant(expected_space)
            
            # Basic check for common transform pairs
            if {norm_actual, norm_expected} == {"MNI152NLin6Asym", "MNI152NLin2009cAsym"}:
                msg += (
                    ". Transform available between these spaces. "
                    "Consider using transform_lesion_data() to align spaces."
                )
        
        raise ValueError(msg)


def validate_space_and_resolution(
    space: str | None,
    resolution: float | None,
    strict: bool = True
) -> None:
    """Validate space identifier and resolution are consistent.
    
    Ensures that if a space is specified, resolution is also provided,
    and both are valid values.
    
    Args:
        space: Space identifier (can be None for native/unknown space)
        resolution: Resolution in mm (can be None if space is None)
        strict: Whether to require resolution when space is provided
        
    Raises:
        ValueError: If validation fails
        
    Examples:
        >>> validate_space_and_resolution("MNI152NLin6Asym", 2.0)
        # No error
        >>> validate_space_and_resolution("MNI152NLin6Asym", None)
        ValueError: Resolution is required when space is specified
        >>> validate_space_and_resolution(None, None)
        # No error - both None is acceptable
    """
    # Both None is acceptable (native/unknown space)
    if space is None and resolution is None:
        return
    
    # If space is provided, resolution must be provided in strict mode
    if space is not None and resolution is None and strict:
        raise ValueError(
            "Resolution is required when space is specified. "
            f"Got space='{space}' but resolution=None"
        )
    
    # Validate space identifier if provided
    if space is not None and space not in SUPPORTED_SPACES:
        raise ValueError(
            f"Unknown or unsupported space identifier: '{space}'. "
            f"Supported spaces: {SUPPORTED_SPACES}"
        )
    
    # Validate resolution if provided
    if resolution is not None:
        valid_resolutions = [0.5, 1, 2]
        if resolution not in valid_resolutions:
            raise ValueError(
                f"Invalid resolution: {resolution}mm. "
                f"Must be one of {valid_resolutions}mm"
            )


# Reference affine matrices for each space/resolution pair
# These are the canonical transformations from voxel to world coordinates
# Values verified from actual template files in src/lacuna/data/templates/
REFERENCE_AFFINES = {
    ("MNI152NLin6Asym", 2): np.array(
        [
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    ("MNI152NLin2009cAsym", 1): np.array(
        [
            [1.0, 0.0, 0.0, -96.0],
            [0.0, 1.0, 0.0, -132.0],
            [0.0, 0.0, 1.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    # MNI152NLin2009aAsym - anatomically identical to cAsym 
    ("MNI152NLin2009aAsym", 1): np.array(
        [
            [1.0, 0.0, 0.0, -96.0],
            [0.0, 1.0, 0.0, -132.0],
            [0.0, 0.0, 1.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    ("MNI152NLin2009cAsym", 2): np.array(
        [
            [2.0, 0.0, 0.0, -96.0],
            [0.0, 2.0, 0.0, -132.0],
            [0.0, 0.0, 2.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    # MNI152NLin2009aAsym 2mm - anatomically identical to cAsym
    ("MNI152NLin2009aAsym", 2): np.array(
        [
            [2.0, 0.0, 0.0, -96.0],
            [0.0, 2.0, 0.0, -132.0],
            [0.0, 0.0, 2.0, -78.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    # MNI152NLin2009bAsym - anatomically identical to cAsym (dTOR tractogram, 0.5mm)
    ("MNI152NLin2009bAsym", 0.5): np.array(
        [
            [0.5, 0.0, 0.0, -98.0],
            [0.0, 0.5, 0.0, -134.0],
            [0.0, 0.0, 0.5, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    ("MNI152NLin6Asym", 1): np.array(
        [
            [1.0, 0.0, 0.0, -90.0],
            [0.0, 1.0, 0.0, -126.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


@dataclass(frozen=True)
class CoordinateSpace:
    """Immutable representation of a neuroimaging coordinate space.

    Attributes:
        identifier: Space identifier (e.g., 'MNI152NLin6Asym')
        resolution: Voxel resolution in mm (0.5, 1, or 2)
        reference_affine: 4x4 affine transformation matrix
    """

    identifier: str
    resolution: float
    reference_affine: np.ndarray

    def __post_init__(self):
        """Validate space identifier and resolution."""
        if self.identifier not in SUPPORTED_SPACES:
            raise ValueError(f"identifier must be in {SUPPORTED_SPACES}, got '{self.identifier}'")

        valid_resolutions = [0.5, 1, 2]
        if self.resolution not in valid_resolutions:
            raise ValueError(
                f"resolution must be one of {valid_resolutions}, got {self.resolution}"
            )

        if self.reference_affine.shape != (4, 4):
            raise ValueError(
                f"reference_affine must be 4x4, got shape {self.reference_affine.shape}"
            )


@dataclass(frozen=True)
class SpatialMetadata:
    """Container for spatial metadata attached to imaging data.

    Attributes:
        space: CoordinateSpace instance
        is_validated: Whether spatial consistency has been validated
        validation_tolerance: Tolerance used for affine validation (default 1e-3)
    """

    space: CoordinateSpace
    is_validated: bool = False
    validation_tolerance: float = 1e-3

    def validate_consistency(self, img: nib.Nifti1Image) -> bool:
        """Validate that image affine matches the declared space.

        Args:
            img: Nibabel image to validate

        Returns:
            True if affine matches within tolerance, False otherwise
        """
        expected_affine = self.space.reference_affine
        actual_affine = img.affine

        diff = np.abs(expected_affine - actual_affine)
        return np.all(diff < self.validation_tolerance)


def detect_space_from_filename(filepath: str | Path) -> tuple[str, int] | None:
    """Extract space identifier and resolution from filename.

    Follows BIDS naming conventions (space- and res- entities).

    Args:
        filepath: Path to neuroimaging file

    Returns:
        Tuple of (space_identifier, resolution_mm) if detected, None otherwise

    Examples:
        >>> detect_space_from_filename("sub-01_space-MNI152NLin6Asym_res-2_mask.nii.gz")
        ('MNI152NLin6Asym', 2)
    """
    filepath = Path(filepath)
    filename = filepath.name

    # BIDS format: space-{identifier}_res-{resolution}
    # Use word boundary to prevent capturing '_res' as part of space name
    space_match = re.search(r"space-([A-Za-z0-9]+)(?:_|\.)", filename)
    res_match = re.search(r"res-(\d+(?:\.\d+)?)", filename)

    if space_match and res_match:
        space = space_match.group(1)
        try:
            resolution = int(float(res_match.group(1)))
            return (space, resolution)
        except ValueError:
            return None

    return None


def detect_space_from_header(
    img: nib.Nifti1Image, tolerance: float = 1e-3
) -> tuple[str, float] | None:
    """Detect coordinate space from image affine matrix.

    Compares the image affine against known reference affines.

    Args:
        img: Nibabel image
        tolerance: Maximum difference for affine matching

    Returns:
        Tuple of (space_identifier, resolution) if matched, None otherwise

    Examples:
        >>> img = nib.load("lesion.nii.gz")
        >>> detect_space_from_header(img)
        ('MNI152NLin6Asym', 2)
    """
    img_affine = img.affine

    for (space, resolution), ref_affine in REFERENCE_AFFINES.items():
        diff = np.abs(img_affine - ref_affine)
        if np.all(diff < tolerance):
            return (space, resolution)

    return None


def get_image_space(
    img: nib.Nifti1Image,
    filepath: Path | None = None,
    declared_space: str | None = None,
    declared_resolution: float | None = None,
    require_match: bool = True,
) -> CoordinateSpace:
    """Unified space detection with validation.

    Attempts detection from filename first, then header. If declared_space
    is provided, validates against detected space.

    Args:
        img: Nibabel image
        filepath: Optional path for filename-based detection
        declared_space: Optional explicit space declaration
        declared_resolution: Optional explicit resolution declaration
        require_match: If True, raises error on mismatch

    Returns:
        CoordinateSpace instance

    Raises:
        SpaceDetectionError: If space cannot be detected
        SpaceMismatchError: If declared space doesn't match detected

    Examples:
        >>> img = nib.load("sub-01_space-MNI152NLin6Asym_res-2.nii.gz")
        >>> space = get_image_space(img, filepath=Path("sub-01_space-MNI152NLin6Asym_res-2.nii.gz"))
        >>> space.identifier
        'MNI152NLin6Asym'
    """
    detected_space = None
    detected_resolution = None
    attempted_methods = []

    # Try filename detection
    if filepath is not None:
        result = detect_space_from_filename(filepath)
        if result is not None:
            detected_space, detected_resolution = result
        attempted_methods.append("filename")

    # Try header detection
    if detected_space is None:
        result = detect_space_from_header(img)
        if result is not None:
            detected_space, detected_resolution = result
        attempted_methods.append("header")

    # Use declared space if provided
    if declared_space is not None:
        if detected_space is not None and require_match:
            if declared_space != detected_space:
                affine_diff = np.max(
                    np.abs(
                        img.affine
                        - REFERENCE_AFFINES.get(
                            (declared_space, declared_resolution or 2), img.affine
                        )
                    )
                )
                raise SpaceMismatchError(
                    declared_space=declared_space,
                    detected_space=detected_space,
                    filepath=filepath or Path("unknown"),
                    affine_difference=float(affine_diff),
                )
        detected_space = declared_space
        detected_resolution = declared_resolution or 2

    # Raise error if detection failed
    if detected_space is None:
        raise SpaceDetectionError(
            filepath=filepath or Path("unknown"), attempted_methods=attempted_methods
        )

    # Get reference affine
    reference_affine = REFERENCE_AFFINES.get((detected_space, detected_resolution), img.affine)

    return CoordinateSpace(
        identifier=detected_space, resolution=detected_resolution, reference_affine=reference_affine
    )


def query_supported_spaces() -> list[str]:
    """Query all supported coordinate space identifiers.

    Returns:
        Sorted list of space identifiers

    Examples:
        >>> spaces = query_supported_spaces()
        >>> 'MNI152NLin6Asym' in spaces
        True
    """
    return sorted(SUPPORTED_SPACES)


class SpaceValidator:
    """Validator for spatial consistency between datasets."""

    def validate_space_declaration(
        self, space: CoordinateSpace, img: nib.Nifti1Image, tolerance: float = 1e-3
    ) -> bool:
        """Validate that image matches declared space.

        Args:
            space: Declared coordinate space
            img: Image to validate
            tolerance: Affine matching tolerance

        Returns:
            True if valid, False otherwise
        """
        diff = np.abs(img.affine - space.reference_affine)
        return np.all(diff < tolerance)

    def detect_mismatch(self, space1: CoordinateSpace, space2: CoordinateSpace) -> bool:
        """Check if two spaces are different.

        Args:
            space1: First space
            space2: Second space

        Returns:
            True if spaces differ, False if same
        """
        return space1.identifier != space2.identifier or space1.resolution != space2.resolution

    def can_transform(self, source_space: CoordinateSpace, target_space: CoordinateSpace) -> bool:
        """Check if transformation is possible between spaces.

        Args:
            source_space: Source coordinate space
            target_space: Target coordinate space

        Returns:
            True if transformation is supported
        """
        # Same space - no transform needed
        if not self.detect_mismatch(source_space, target_space):
            return True

        # Resolve space aliases (aAsym/bAsym -> cAsym)
        # Aliased spaces are anatomically identical, only resampling needed
        source_id = SPACE_ALIASES.get(source_space.identifier, source_space.identifier)
        target_id = SPACE_ALIASES.get(target_space.identifier, target_space.identifier)

        # If aliases resolve to same space, only resolution difference remains
        if source_id == target_id:
            return True  # Resolution resampling is always supported

        # Check if transform pair is in registry
        supported_pairs = [
            ("MNI152NLin6Asym", "MNI152NLin2009cAsym"),
            ("MNI152NLin2009cAsym", "MNI152NLin6Asym"),
        ]

        pair = (source_id, target_id)
        return pair in supported_pairs


__all__ = [
    "CoordinateSpace",
    "SpatialMetadata",
    "SpaceValidator",
    "SUPPORTED_SPACES",
    "SPACE_ALIASES",
    "REFERENCE_AFFINES",
    "detect_space_from_filename",
    "detect_space_from_header",
    "get_image_space",
    "query_supported_spaces",
    "SpaceDetectionError",
    "SpaceMismatchError",
    "TransformNotAvailableError",
]
