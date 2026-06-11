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
# NLin6Asym and 2009cAsym are user-facing; 2009bAsym is internal (dTOR985 tractogram).
SUPPORTED_SPACES = [
    "MNI152NLin6Asym",
    "MNI152NLin2009cAsym",
    "MNI152NLin2009bAsym",  # Internal only — dTOR985 tractogram space
]

# User-facing input spaces — only these should be accepted from user-provided data.
USER_INPUT_SPACES = ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]


def spaces_are_equivalent(space1: str, space2: str) -> bool:
    """Check if two space identifiers refer to the same coordinate space.

    Parameters
    ----------
    space1 : str
        First space identifier.
    space2 : str
        Second space identifier.

    Returns
    -------
    bool
        True if spaces are the same, False otherwise.

    Examples
    --------
    >>> spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin6Asym")
    True
    >>> spaces_are_equivalent("MNI152NLin2009bAsym", "MNI152NLin2009cAsym")
    False
    """
    return space1 == space2


def validate_space_compatibility(
    actual_space: str,
    expected_space: str,
    context: str = "operation",
    suggest_transform: bool = False,
) -> None:
    """Validate that actual space is compatible with expected space.

    Raises ValueError if spaces are incompatible (not equivalent).
    Handles space aliases automatically.

    Parameters
    ----------
    actual_space : str
        The actual space of the data.
    expected_space : str
        The expected/required space.
    context : str
        Description of operation for error messages.
    suggest_transform : bool
        Whether to suggest transformation in error message.

    Raises
    ------
    ValueError
        If spaces are incompatible.

    Examples
    --------
    >>> validate_space_compatibility("MNI152NLin6Asym", "MNI152NLin6Asym", "test")
    # No error - spaces match
    >>> validate_space_compatibility("native", "MNI152NLin6Asym", "test")
    ValueError: Space mismatch in test: got 'native', expected 'MNI152NLin6Asym'
    """
    if not spaces_are_equivalent(actual_space, expected_space):
        msg = f"Space mismatch in {context}: " f"got '{actual_space}', expected '{expected_space}'"

        if suggest_transform:
            if {actual_space, expected_space} <= set(SUPPORTED_SPACES):
                msg += (
                    ". Transform available between these spaces. "
                    "Consider using transform_mask_data() to align spaces."
                )

        raise ValueError(msg)


def validate_space_and_resolution(
    space: str | None, resolution: float | None, strict: bool = True
) -> None:
    """Validate space identifier and resolution are consistent.

    Ensures that if a space is specified, resolution is also provided,
    and both are valid values.

    Parameters
    ----------
    space : str or None
        Space identifier (can be None for native/unknown space).
    resolution : float or None
        Resolution in mm (can be None if space is None).
    strict : bool
        Whether to require resolution when space is provided.

    Raises
    ------
    ValueError
        If validation fails.

    Examples
    --------
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
                f"Invalid resolution: {resolution}mm. " f"Must be one of {valid_resolutions}mm"
            )


# Reference affine matrices for each space/resolution pair
# These are the canonical transformations from voxel to world coordinates
# Values verified from actual template files
REFERENCE_AFFINES = {
    ("MNI152NLin6Asym", 1): np.array(
        [
            [1.0, 0.0, 0.0, -91.0],
            [0.0, 1.0, 0.0, -126.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    # 2mm origin is NOT the 1mm origin: TemplateFlow's res-02 grid is
    # offset (origin -90, not -91). Value taken from the canonical
    # tpl-MNI152NLin6Asym template_description.json "res" block.
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
    # 2mm origin is NOT the 1mm origin: TemplateFlow's res-02 grid is
    # offset (origin -96.5, not -96). Value taken from the canonical
    # tpl-MNI152NLin2009cAsym template_description.json "res" block.
    ("MNI152NLin2009cAsym", 2): np.array(
        [
            [2.0, 0.0, 0.0, -96.5],
            [0.0, 2.0, 0.0, -132.5],
            [0.0, 0.0, 2.0, -78.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    ("MNI152NLin2009bAsym", 0.5): np.array(
        [
            [0.5, 0.0, 0.0, -98.0],
            [0.0, 0.5, 0.0, -134.0],
            [0.0, 0.0, 0.5, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    # 2009bAsym origin verified from original MNI template: (-98, -134, -72)
    ("MNI152NLin2009bAsym", 1): np.array(
        [
            [1.0, 0.0, 0.0, -98.0],
            [0.0, 1.0, 0.0, -134.0],
            [0.0, 0.0, 1.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    ("MNI152NLin2009bAsym", 2): np.array(
        [
            [2.0, 0.0, 0.0, -98.0],
            [0.0, 2.0, 0.0, -134.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

# Reference template shapes for each space/resolution pair.
# Used by regrid operations to define the target voxel grid.
# Values verified from original MNI template files.
REFERENCE_SHAPES = {
    ("MNI152NLin6Asym", 1): (182, 218, 182),
    ("MNI152NLin6Asym", 2): (91, 109, 91),
    ("MNI152NLin2009cAsym", 1): (193, 229, 193),
    ("MNI152NLin2009cAsym", 2): (97, 115, 97),
    ("MNI152NLin2009bAsym", 0.5): (394, 466, 378),
    ("MNI152NLin2009bAsym", 1): (197, 233, 189),
    ("MNI152NLin2009bAsym", 2): (99, 117, 95),
}


@dataclass(frozen=True)
class CoordinateSpace:
    """Immutable representation of a neuroimaging coordinate space.

    Attributes
    ----------
    identifier : str
        Space identifier (e.g., 'MNI152NLin6Asym').
    resolution : float
        Voxel resolution in mm (0.5, 1, or 2).
    reference_affine : numpy.ndarray
        4x4 affine transformation matrix.
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

    Attributes
    ----------
    space : CoordinateSpace
        CoordinateSpace instance.
    is_validated : bool
        Whether spatial consistency has been validated.
    validation_tolerance : float
        Tolerance used for affine validation (default 1e-3).
    """

    space: CoordinateSpace
    is_validated: bool = False
    validation_tolerance: float = 1e-3

    def validate_consistency(self, img: nib.Nifti1Image) -> bool:
        """Validate that image affine matches the declared space.

        Parameters
        ----------
        img : nib.Nifti1Image
            Nibabel image to validate.

        Returns
        -------
        bool
            True if affine matches within tolerance, False otherwise.
        """
        expected_affine = self.space.reference_affine
        actual_affine = img.affine

        diff = np.abs(expected_affine - actual_affine)
        return np.all(diff < self.validation_tolerance)


def detect_space_from_filename(filepath: str | Path) -> tuple[str, int] | None:
    """Extract space identifier and resolution from filename.

    Follows BIDS naming conventions (space- and res- entities).

    Parameters
    ----------
    filepath : str or Path
        Path to neuroimaging file.

    Returns
    -------
    tuple of (str, int) or None
        Tuple of (space_identifier, resolution_mm) if detected, None otherwise.

    Examples
    --------
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
            # Preserve sub-millimeter resolutions: int("0.5") path would have
            # truncated res-0.5 to 0. Collapse to int only when integral so
            # dict lookups against integer keys (e.g. res 1, 2) still match.
            res = float(res_match.group(1))
            resolution = int(res) if res.is_integer() else res
            return (space, resolution)
        except ValueError:
            return None

    return None


def detect_space_from_header(
    img: nib.Nifti1Image, tolerance: float = 1e-3
) -> tuple[str, float] | None:
    """Detect coordinate space from image affine matrix.

    Compares the image affine against known reference affines.
    Handles radiological-convention images (negative strides) by
    canonicalizing to RAS+ orientation before comparison.

    Parameters
    ----------
    img : nib.Nifti1Image
        Nibabel image.
    tolerance : float
        Maximum difference for affine matching.

    Returns
    -------
    tuple of (str, float) or None
        Tuple of (space_identifier, resolution) if matched, None otherwise.

    Examples
    --------
    >>> img = nib.load("lesion.nii.gz")
    >>> detect_space_from_header(img)
    ('MNI152NLin6Asym', 2)
    """
    img_affine = img.affine

    # First try direct affine comparison (fast path for neurological convention)
    for (space, resolution), ref_affine in REFERENCE_AFFINES.items():
        diff = np.abs(img_affine - ref_affine)
        if np.all(diff < tolerance):
            return (space, resolution)

    # If direct match failed, check for radiological-convention images.
    # Radiological images have negative strides (e.g. [-1, 2, 3]) so the
    # raw affine differs, but the voxel grid covers the same physical space.
    # We match by: same shape, same voxel sizes, and field-of-view center
    # within one voxel of the reference.
    for (space, resolution), ref_affine in REFERENCE_AFFINES.items():
        ref_shape = REFERENCE_SHAPES.get((space, resolution))
        if ref_shape is None or img.shape[:3] != ref_shape:
            continue
        # Voxel sizes must match (absolute values, ignoring stride sign)
        img_voxel_sizes = np.abs(np.diag(img_affine[:3, :3]))
        ref_voxel_sizes = np.abs(np.diag(ref_affine[:3, :3]))
        if not np.allclose(img_voxel_sizes, ref_voxel_sizes, atol=tolerance):
            continue
        # Check off-diagonals are zero (no oblique transforms)
        off_diag = img_affine[:3, :3].copy()
        np.fill_diagonal(off_diag, 0)
        if not np.allclose(off_diag, 0, atol=tolerance):
            continue
        # Compare FOV centers: the center voxel maps to the same world coord
        # regardless of storage direction, with at most 1-voxel jitter from
        # the LR flip (which shifts the grid by voxel_size in the flipped axis).
        center_voxel = (np.array(img.shape[:3]) - 1) / 2.0
        img_center = img_affine[:3, :3] @ center_voxel + img_affine[:3, 3]
        ref_center = ref_affine[:3, :3] @ center_voxel + ref_affine[:3, 3]
        if np.all(np.abs(img_center - ref_center) <= resolution + tolerance):
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

    Parameters
    ----------
    img : nib.Nifti1Image
        Nibabel image.
    filepath : Path or None
        Optional path for filename-based detection.
    declared_space : str or None
        Optional explicit space declaration.
    declared_resolution : float or None
        Optional explicit resolution declaration.
    require_match : bool
        If True, raises error on mismatch.

    Returns
    -------
    CoordinateSpace
        CoordinateSpace instance.

    Raises
    ------
    SpaceDetectionError
        If space cannot be detected.
    SpaceMismatchError
        If declared space doesn't match detected.

    Examples
    --------
    >>> img = nib.load("sub-01_space-MNI152NLin6Asym_res-2.nii.gz")
    >>> space = get_image_space(img, filepath=Path("sub-01_space-MNI152NLin6Asym_res-2.nii.gz"))
    >>> space.identifier
    'MNI152NLin6Asym'
    """
    detected_space = None
    detected_resolution = None
    attempted_methods = []

    # Detect space from image affine only (not filename or sidecar metadata)
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

    Returns
    -------
    list[str]
        Sorted list of space identifiers.

    Examples
    --------
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

        Parameters
        ----------
        space : CoordinateSpace
            Declared coordinate space.
        img : nib.Nifti1Image
            Image to validate.
        tolerance : float
            Affine matching tolerance.

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        diff = np.abs(img.affine - space.reference_affine)
        return np.all(diff < tolerance)

    def detect_mismatch(self, space1: CoordinateSpace, space2: CoordinateSpace) -> bool:
        """Check if two spaces are different.

        Parameters
        ----------
        space1 : CoordinateSpace
            First space.
        space2 : CoordinateSpace
            Second space.

        Returns
        -------
        bool
            True if spaces differ, False if same.
        """
        return space1.identifier != space2.identifier or space1.resolution != space2.resolution

    def can_transform(self, source_space: CoordinateSpace, target_space: CoordinateSpace) -> bool:
        """Check if transformation is possible between spaces.

        Parameters
        ----------
        source_space : CoordinateSpace
            Source coordinate space.
        target_space : CoordinateSpace
            Target coordinate space.

        Returns
        -------
        bool
            True if transformation is supported.
        """
        # Same space - no transform needed
        if not self.detect_mismatch(source_space, target_space):
            return True

        source_id = source_space.identifier
        target_id = target_space.identifier

        # Same space, different resolution — resampling always supported
        if source_id == target_id:
            return True

        # 2009b ↔ 2009c: regrid (same MNI world coords, different voxel grid)
        if {source_id, target_id} == {"MNI152NLin2009bAsym", "MNI152NLin2009cAsym"}:
            return True

        # Nonlinear warps + chained transforms (NLin6 ↔ 2009c, NLin6 ↔ 2009b via 2009c)
        if "MNI152NLin6Asym" in {source_id, target_id}:
            other = target_id if source_id == "MNI152NLin6Asym" else source_id
            return other in {"MNI152NLin2009cAsym", "MNI152NLin2009bAsym"}

        return False


__all__ = [
    "CoordinateSpace",
    "SpatialMetadata",
    "SpaceValidator",
    "SUPPORTED_SPACES",
    "USER_INPUT_SPACES",
    "REFERENCE_AFFINES",
    "spaces_are_equivalent",
    "validate_space_compatibility",
    "detect_space_from_filename",
    "detect_space_from_header",
    "get_image_space",
    "query_supported_spaces",
    "SpaceDetectionError",
    "SpaceMismatchError",
    "TransformNotAvailableError",
]
