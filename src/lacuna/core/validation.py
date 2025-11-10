"""
Validation utilities for neuroimaging data.

Functions for validating NIfTI images, affine matrices, and coordinate spaces.
"""

import nibabel as nib
import numpy as np

from .exceptions import SpatialMismatchError, ValidationError


def validate_nifti_image(
    img: nib.Nifti1Image, require_3d: bool = True, check_affine: bool = True
) -> None:
    """
    Validate NIfTI image properties.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Image to validate.
    require_3d : bool, default=True
        Raise error if image is not 3D.
    check_affine : bool, default=True
        Verify affine matrix is invertible.

    Raises
    ------
    ValidationError
        If validation fails.

    Examples
    --------
    >>> import nibabel as nib
    >>> img = nib.load("lesion.nii.gz")
    >>> validate_nifti_image(img)
    """
    if not isinstance(img, nib.Nifti1Image):
        raise ValidationError(f"Expected Nifti1Image, got {type(img)}")

    # Check dimensionality
    shape = img.shape
    if require_3d:
        if len(shape) != 3:
            raise ValidationError(f"Image must be 3D, got shape {shape} ({len(shape)} dimensions)")

    # Check affine matrix
    if check_affine:
        affine = img.affine
        try:
            np.linalg.inv(affine)
        except np.linalg.LinAlgError as e:
            raise ValidationError("Affine matrix is not invertible") from e

        # Check for NaN or inf values
        if not np.all(np.isfinite(affine)):
            raise ValidationError("Affine matrix contains NaN or inf values")


def ensure_ras_plus(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Ensure image is in RAS+ orientation (nilearn standard).

    Parameters
    ----------
    img : nibabel.Nifti1Image
        Input image (any orientation).

    Returns
    -------
    nibabel.Nifti1Image
        Image reoriented to RAS+ (if necessary).

    Notes
    -----
    RAS+ means:
    - First axis: Right to Left
    - Second axis: Anterior to Posterior
    - Third axis: Superior to Inferior

    Examples
    --------
    >>> import nibabel as nib
    >>> img = nib.load("lesion.nii.gz")
    >>> img_ras = ensure_ras_plus(img)
    """
    # Get current orientation
    ornt = nib.io_orientation(img.affine)

    # RAS+ orientation
    ras_ornt = np.array([[0, 1], [1, 1], [2, 1]])

    # Calculate transformation
    transform = nib.orientations.ornt_transform(ornt, ras_ornt)

    # Apply if needed
    if not np.array_equal(transform, [[0, 1], [1, 1], [2, 1]]):
        img = img.as_reoriented(transform)

    return img


def check_spatial_match(
    img1: nib.Nifti1Image,
    img2: nib.Nifti1Image,
    check_shape: bool = True,
    check_affine: bool = True,
    atol: float = 1e-3,
) -> bool:
    """
    Check if two images have matching spatial properties.

    Parameters
    ----------
    img1 : nibabel.Nifti1Image
        First image.
    img2 : nibabel.Nifti1Image
        Second image.
    check_shape : bool, default=True
        Check if shapes match.
    check_affine : bool, default=True
        Check if affines match (within tolerance).
    atol : float, default=1e-3
        Absolute tolerance for affine comparison (in mm).

    Returns
    -------
    bool
        True if images match spatially.

    Raises
    ------
    SpatialMismatchError
        If spatial properties don't match.

    Examples
    --------
    >>> import nibabel as nib
    >>> lesion = nib.load("lesion.nii.gz")
    >>> anat = nib.load("anatomical.nii.gz")
    >>> check_spatial_match(lesion, anat)
    """
    # Check shapes
    if check_shape:
        if img1.shape != img2.shape:
            raise SpatialMismatchError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    # Check affines
    if check_affine:
        if not np.allclose(img1.affine, img2.affine, atol=atol):
            raise SpatialMismatchError(
                f"Affine matrices don't match (tolerance={atol} mm).\n"
                f"Image 1:\n{img1.affine}\n"
                f"Image 2:\n{img2.affine}"
            )

    return True


def validate_affine(affine: np.ndarray) -> None:
    """
    Validate an affine transformation matrix.

    Parameters
    ----------
    affine : ndarray, shape (4, 4)
        Affine matrix to validate.

    Raises
    ------
    ValidationError
        If affine is invalid.
    """
    if not isinstance(affine, np.ndarray):
        raise ValidationError(f"Affine must be numpy array, got {type(affine)}")

    if affine.shape != (4, 4):
        raise ValidationError(f"Affine must be 4x4, got shape {affine.shape}")

    if not np.all(np.isfinite(affine)):
        raise ValidationError("Affine contains NaN or inf values")

    # Check if invertible
    try:
        np.linalg.inv(affine)
    except np.linalg.LinAlgError as e:
        raise ValidationError("Affine matrix is not invertible") from e

    # Check last row is [0, 0, 0, 1]
    if not np.allclose(affine[3, :], [0, 0, 0, 1]):
        raise ValidationError(f"Affine last row must be [0, 0, 0, 1], got {affine[3, :]}")
