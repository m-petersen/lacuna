"""
Coordinate space detection, validation, and alignment.

This module provides utilities for preventing silent space mismatch errors.

Note: ANTsPy-based registration functions require the optional 'preprocess' dependency.
Install with: pip install lacuna[preprocess]
"""

import re
from pathlib import Path
from typing import Literal

try:
    import ants
    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False

import nibabel as nib
import numpy as np
from nilearn.image import resample_img

from ..core.exceptions import ValidationError

# Supported MNI152 template spaces
SUPPORTED_SPACES = [
    "MNI152NLin6Asym",
    "MNI152NLin2009cAsym",
    "MNI152NLin2009bAsym",
]

# Supported resolutions in mm
SUPPORTED_RESOLUTIONS = [1, 2]

# NIFTI qform/sform codes
NIFTI_XFORM_MNI_152 = 4


def detect_space_from_filename(filepath: str | Path) -> dict[str, str | int]:
    """
    Detect coordinate space from BIDS-compliant filename.

    Parses BIDS entities (tpl-, space-, res-) from the filename to extract
    space and resolution information.

    Parameters
    ----------
    filepath : str or Path
        Path to the file. Only the filename is used for parsing.

    Returns
    -------
    dict
        Dictionary with detected 'space' (str) and 'resolution' (int) keys.
        Returns empty dict if no space information found.
        Defaults to 1mm resolution if space found but no resolution specified.

    Examples
    --------
    >>> detect_space_from_filename("sub-01_space-MNI152NLin6Asym_res-02_mask.nii.gz")
    {'space': 'MNI152NLin6Asym', 'resolution': 2}

    >>> detect_space_from_filename("tpl-MNI152NLin2009cAsym_T1w.nii.gz")
    {'space': 'MNI152NLin2009cAsym', 'resolution': 1}
    """
    filename = Path(filepath).name
    result = {}

    # Parse space entity
    space_patterns = [r"tpl-([A-Za-z0-9]+)", r"space-([A-Za-z0-9]+)"]
    for pattern in space_patterns:
        match = re.search(pattern, filename)
        if match:
            space_candidate = match.group(1)
            if space_candidate in SUPPORTED_SPACES:
                result["space"] = space_candidate
                break

    # Parse resolution entity
    res_match = re.search(r"res-(\d+)", filename)
    if res_match:
        res_str = res_match.group(1)
        res_int = int(res_str.lstrip("0") or "0")
        if res_int in SUPPORTED_RESOLUTIONS:
            result["resolution"] = res_int

    # Default to 1mm if space found but no resolution
    if "space" in result and "resolution" not in result:
        result["resolution"] = 1

    return result


def detect_space_from_header(img: nib.Nifti1Image) -> dict[str, str | int]:
    """
    Detect coordinate space from NIfTI header metadata.

    Examines qform/sform codes and voxel dimensions to infer the coordinate
    space and resolution. Less reliable than filename parsing but useful as
    a fallback.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        The NIfTI image to inspect.

    Returns
    -------
    dict
        Dictionary with detected 'space' (str) and 'resolution' (int) keys.
        Returns empty dict if space cannot be determined from header.

    Notes
    -----
    - Detects MNI152 space if qform or sform code is 4 (NIFTI_XFORM_MNI_152)
    - Assumes MNI152NLin6Asym as default MNI152 variant (cannot distinguish)
    - Resolution determined from average voxel size
    """
    result = {}

    # Check qform/sform codes
    qform_code = img.header.get_qform(coded=True)[1]
    sform_code = img.header.get_sform(coded=True)[1]

    if qform_code == NIFTI_XFORM_MNI_152 or sform_code == NIFTI_XFORM_MNI_152:
        result["space"] = "MNI152NLin6Asym"

    # Detect resolution from voxel dimensions
    voxel_sizes = img.header.get_zooms()[:3]
    avg_voxel_size = sum(voxel_sizes) / len(voxel_sizes)

    if 0.5 <= avg_voxel_size < 1.5:
        result["resolution"] = 1
    elif 1.5 <= avg_voxel_size < 2.5:
        result["resolution"] = 2

    return result


def get_image_space(
    img: nib.Nifti1Image,
    filepath: str | Path | None = None,
) -> dict[str, str | int]:
    """
    Get coordinate space information from image with fallback strategies.

    Uses a fallback chain: filename parsing → header inspection → error.
    Prefers filename over header since BIDS entities are more reliable.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        The NIfTI image to inspect.
    filepath : str, Path, or None, optional
        Path to the image file for BIDS entity parsing.
        If None, only header inspection is used.

    Returns
    -------
    dict
        Dictionary with 'space' (str) and 'resolution' (int) keys.

    Raises
    ------
    ValidationError
        If space cannot be detected from filename or header.

    Examples
    --------
    >>> img = nib.load("sub-01_space-MNI152NLin6Asym_mask.nii.gz")
    >>> get_image_space(img, "sub-01_space-MNI152NLin6Asym_mask.nii.gz")
    {'space': 'MNI152NLin6Asym', 'resolution': 1}
    """
    result = {}

    # Strategy 1: Filename parsing
    if filepath:
        result = detect_space_from_filename(filepath)

    # Strategy 2: Header inspection (fallback)
    if not result:
        result = detect_space_from_header(img)

    # Strategy 3: Fail with helpful error
    if not result or "space" not in result:
        filepath_str = f"\nFilename: {filepath}" if filepath else ""
        raise ValidationError(
            f"Cannot detect coordinate space for image.{filepath_str}\n"
            f"\n"
            f"Please ensure:\n"
            f"  1. Filename follows BIDS convention\n"
            f"  2. Header has MNI152 qform/sform code\n"
            f"\n"
            f"Supported spaces: {', '.join(SUPPORTED_SPACES)}"
        )

    return result


def _get_transform_from_templateflow(source_space: str, target_space: str) -> Path | None:
    """
    Query TemplateFlow for transformation file.

    Parameters
    ----------
    source_space : str
        Source coordinate space.
    target_space : str
        Target coordinate space.

    Returns
    -------
    Path or None
        Path to transform file if found, None otherwise.
    """
    try:
        import templateflow.api as tflow

        xfm = tflow.get(template=target_space, mode="image", from_=source_space, extension=".h5")
        return Path(xfm)
    except (ImportError, Exception):
        return None


def _get_bundled_transform(source_space: str, target_space: str) -> Path:
    """
    Get bundled transform from package data.

    Parameters
    ----------
    source_space : str
        Source coordinate space.
    target_space : str
        Target coordinate space.

    Returns
    -------
    Path
        Path to bundled transform file.

    Raises
    ------
    FileNotFoundError
        If transform file not found in package data.
    """
    filename = f"tpl-{target_space}_from-{source_space}_mode-image_xfm.h5"
    transform_dir = Path(__file__).parent.parent / "data" / "transforms"
    transform_path = transform_dir / filename

    if not transform_path.exists():
        raise FileNotFoundError(
            f"Transform {source_space} → {target_space} not available.\n"
            f"Tried: {transform_path}\n"
            f"\n"
            f"Supported transform pairs:\n"
            f"  • MNI152NLin6Asym ↔ MNI152NLin2009cAsym\n"
            f"\n"
            f"For other spaces, install templateflow:\n"
            f"  pip install templateflow"
        )

    return transform_path


def _get_bundled_template(space: str, resolution: int) -> nib.Nifti1Image:
    """
    Load bundled template image for specified space and resolution.

    Parameters
    ----------
    space : str
        Template space (e.g., 'MNI152NLin6Asym', 'MNI152NLin2009cAsym').
    resolution : int
        Resolution in mm (1 or 2).

    Returns
    -------
    nibabel.Nifti1Image
        Template brain image.

    Raises
    ------
    FileNotFoundError
        If template not found in package data.
    """
    filename = f"tpl-{space}_res-0{resolution}_desc-brain_T1w.nii.gz"
    template_dir = Path(__file__).parent.parent / "data" / "templates"
    template_path = template_dir / filename

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template {space} @ {resolution}mm not available.\n"
            f"Tried: {template_path}\n"
            f"\n"
            f"Available templates:\n"
            f"  • MNI152NLin6Asym: 1mm, 2mm\n"
            f"  • MNI152NLin2009cAsym: 1mm, 2mm\n"
            f"  • MNI152NLin2009bAsym: 1mm"
        )

    return nib.load(template_path)


def align_to_reference(
    img: nib.Nifti1Image,
    source_space: str,
    target_space: str,
    source_resolution: int = 1,
    target_resolution: int = 1,
    use_templateflow: bool = True,
    interpolation: Literal["continuous", "nearest"] = "continuous",
) -> nib.Nifti1Image:
    """
    Align image from source to target MNI152 template space.

    Applies transformation using nitransforms. Tries TemplateFlow first
    (if enabled), then falls back to bundled transforms. Handles resolution
    changes by adjusting the reference image.

    Parameters
    ----------
    img : nibabel.Nifti1Image
        The image to align.
    source_space : str
        Source coordinate space (must be in SUPPORTED_SPACES).
    target_space : str
        Target coordinate space (must be in SUPPORTED_SPACES).
    source_resolution : int, optional
        Source image resolution in mm (default: 1).
    target_resolution : int, optional
        Target resolution in mm (default: 1).
    use_templateflow : bool, optional
        Try TemplateFlow API before bundled transforms (default: True).
    interpolation : {'continuous', 'nearest'}, optional
        Interpolation method (default: 'continuous').
        Use 'nearest' for label/mask images.

    Returns
    -------
    nibabel.Nifti1Image
        The aligned image in target space and resolution.

    Raises
    ------
    ValueError
        If source_space or target_space not supported.
    FileNotFoundError
        If transform file not found in TemplateFlow or bundled data.

    Examples
    --------
    >>> lesion_img = nib.load("lesion_NLin6.nii.gz")
    >>> aligned = align_to_reference(
    ...     lesion_img,
    ...     source_space="MNI152NLin6Asym",
    ...     target_space="MNI152NLin2009cAsym",
    ...     interpolation="nearest"  # for masks
    ... )
    """
    # Validate spaces
    if source_space not in SUPPORTED_SPACES:
        raise ValueError(f"Unsupported source space: {source_space}")
    if target_space not in SUPPORTED_SPACES:
        raise ValueError(f"Unsupported target space: {target_space}")

    if not HAS_ANTS:
        raise ImportError(
            "ANTsPy is required for align_to_reference but is not installed. "
            "Install with: pip install lacuna[preprocess]"
        )

    print(f"[ALIGN_TO_REFERENCE] Using ANTsPy backend")  # DEBUG
    print(f"[ALIGN_TO_REFERENCE] Source: {source_space} @ {source_resolution}mm")  # DEBUG
    print(f"[ALIGN_TO_REFERENCE] Target: {target_space} @ {target_resolution}mm")  # DEBUG

    # Shortcut: same space and resolution - no transformation needed
    if source_space == target_space and source_resolution == target_resolution:
        return nib.Nifti1Image(img.get_fdata(), img.affine, img.header.copy())

    # Handle same space, different resolution (resample only)
    if source_space == target_space and source_resolution != target_resolution:
        # Just resample to target resolution (no coordinate transform needed)
        target_affine = img.affine.copy()
        target_affine[0, 0] = target_resolution if target_affine[0, 0] > 0 else -target_resolution
        target_affine[1, 1] = target_resolution if target_affine[1, 1] > 0 else -target_resolution
        target_affine[2, 2] = target_resolution if target_affine[2, 2] > 0 else -target_resolution

        # Calculate target shape based on resolution change
        scale_factor = source_resolution / target_resolution
        target_shape = tuple(int(s * scale_factor) for s in img.shape[:3])

        # Create reference and resample
        reference = nib.Nifti1Image(np.zeros(target_shape), target_affine)
        return resample_img(
            img,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation=interpolation,
        )

    # Get transform file for different coordinate spaces
    transform_path = None
    if use_templateflow:
        transform_path = _get_transform_from_templateflow(source_space, target_space)
    if transform_path is None:
        transform_path = _get_bundled_transform(source_space, target_space)

    # Load the actual target template as reference
    # This ensures we get the correct dimensions for the target space
    try:
        # Try to load bundled template first
        reference = _get_bundled_template(target_space, target_resolution)
    except FileNotFoundError:
        # Fall back to TemplateFlow if bundled template not available
        if use_templateflow:
            try:
                import templateflow.api as tflow

                template_path = tflow.get(
                    template=target_space,
                    resolution=target_resolution,
                    desc="brain",
                    suffix="T1w",
                    extension=".nii.gz",
                )
                reference = nib.load(template_path)
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not load template for {target_space} @ {target_resolution}mm.\n"
                    f"Not available in bundled data or TemplateFlow.\n"
                    f"Error: {e}"
                ) from e
        else:
            raise

    # Convert nibabel images to ANTsPy format
    # Use proper nibabel to ANTsPy conversion
    def nifti_to_ants(nib_image):
        """Convert nibabel image to ANTsPy image."""
        ndim = nib_image.ndim
        q_form = nib_image.get_qform()
        spacing = nib_image.header["pixdim"][1 : ndim + 1]

        origin = np.zeros((ndim))
        origin[:3] = q_form[:3, 3]

        direction = np.diag(np.ones(ndim))
        direction[:3, :3] = q_form[:3, :3] / spacing[:3]

        return ants.from_numpy(
            data=np.asarray(nib_image.dataobj).astype(np.float32),
            origin=origin.tolist(),
            spacing=spacing.tolist(),
            direction=direction,
        )

    moving_ants = nifti_to_ants(img)
    fixed_ants = nifti_to_ants(reference)

    # Apply transform using ANTsPy
    interpolator_map = {
        "continuous": "linear",
        "nearest": "nearestNeighbor",
    }
    ants_interpolator = interpolator_map.get(interpolation, "linear")

    transformed_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=[str(transform_path)],
        interpolator=ants_interpolator,
    )

    # Convert back to nibabel format
    # Get the data and reconstruct the affine from ANTsPy image properties
    transformed_data = transformed_ants.numpy()
    transformed_affine = reference.affine.copy()  # Use reference affine

    transformed_img = nib.Nifti1Image(transformed_data, transformed_affine, reference.header.copy())

    return transformed_img


def check_space_compatibility(
    *images_with_info: tuple[nib.Nifti1Image, str, dict],
    strict: bool = False,
) -> tuple[bool, str]:
    """
    Check if multiple images are in compatible spaces.

    Validates that all images share the same coordinate space. In strict mode,
    also requires matching resolutions.

    Parameters
    ----------
    *images_with_info : tuple[nibabel.Nifti1Image, str, dict]
        Variable number of (image, description, space_info) tuples where:
        - image: The NIfTI image
        - description: Human-readable label (e.g., "Lesion mask", "Atlas")
        - space_info: Dict with 'space' and 'resolution' keys
    strict : bool, optional
        If True, require matching resolutions (default: False).

    Returns
    -------
    tuple[bool, str]
        - bool: True if compatible, False otherwise
        - str: Error message if incompatible, empty string if compatible

    Examples
    --------
    >>> lesion_info = {'space': 'MNI152NLin6Asym', 'resolution': 1}
    >>> atlas_info = {'space': 'MNI152NLin6Asym', 'resolution': 2}
    >>> compatible, msg = check_space_compatibility(
    ...     (lesion_img, "Lesion", lesion_info),
    ...     (atlas_img, "Atlas", atlas_info),
    ...     strict=False  # Allow different resolutions
    ... )
    >>> compatible
    True
    """
    if len(images_with_info) < 2:
        return True, ""

    # Get reference (first image)
    _, ref_desc, ref_info = images_with_info[0]
    ref_space = ref_info["space"]
    ref_res = ref_info["resolution"]

    # Check all others against reference
    for _img, desc, info in images_with_info[1:]:
        space = info["space"]
        res = info["resolution"]

        # Check space mismatch
        if space != ref_space:
            error_msg = (
                f"Space mismatch detected:\n"
                f"  {ref_desc}: {ref_space} {ref_res}mm\n"
                f"  {desc}: {space} {res}mm"
            )
            return False, error_msg

        # Check resolution mismatch (if strict)
        if strict and res != ref_res:
            error_msg = (
                f"Resolution mismatch (strict mode):\n"
                f"  {ref_desc}: {ref_space} {ref_res}mm\n"
                f"  {desc}: {space} {res}mm"
            )
            return False, error_msg

    return True, ""
