"""Transformation strategies for spatial coordinate space conversions."""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
from nitransforms import linear as nitl

from lacuna.core.exceptions import TransformNotAvailableError
from lacuna.core.spaces import CoordinateSpace

# Fix for Jupyter notebooks: Allow nested event loops for nitransforms
# nitransforms uses asyncio.run() which fails in Jupyter's existing event loop
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    # nest_asyncio not available - will provide helpful error if needed
    pass

if TYPE_CHECKING:
    from lacuna.core.mask_data import MaskData

logger = logging.getLogger(__name__)


# Space equivalence groups - spaces that are anatomically identical
# MNI152NLin2009aAsym, bAsym, and cAsym are anatomically identical
# They only differ in the preprocessing pipeline used to create them
EQUIVALENT_SPACES = {
    "MNI152NLin2009aAsym": "MNI152NLin2009cAsym",
    "MNI152NLin2009bAsym": "MNI152NLin2009cAsym",
    "MNI152NLin2009cAsym": "MNI152NLin2009cAsym",  # Identity for completeness
}


def _canonicalize_space_variant(space_id: str) -> str:
    """Canonicalize MNI space variant identifiers to canonical forms.

    For anatomically identical spaces (e.g., MNI152NLin2009[abc]Asym),
    returns the canonical representative (MNI152NLin2009cAsym).

    Args:
        space_id: Space identifier to canonicalize

    Returns:
        Canonical space identifier
    """
    return EQUIVALENT_SPACES.get(space_id, space_id)


class InterpolationMethod(str, Enum):
    """Supported interpolation methods for spatial transformations."""

    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"


class TransformationStrategy:
    """Strategy for applying spatial transformations between coordinate spaces.

    This class determines the optimal transformation direction and method
    for converting data between different coordinate spaces.
    """

    def determine_direction(
        self, source: CoordinateSpace, target: CoordinateSpace
    ) -> Literal["forward", "reverse", "none"]:
        """Determine transformation direction based on source and target spaces.

        Args:
            source: Source coordinate space
            target: Target coordinate space

        Returns:
            "forward" for source->target transform,
            "reverse" for target->source transform,
            "none" if no transformation needed

        Raises:
            TransformNotAvailableError: If transformation not supported
        """
        # Normalize space identifiers to handle equivalent spaces
        source_normalized = _canonicalize_space_variant(source.identifier)
        target_normalized = _canonicalize_space_variant(target.identifier)

        # Same space (after normalization) - no transform needed
        if source_normalized == target_normalized and source.resolution == target.resolution:
            return "none"

        # Same space but different resolution - needs resampling only
        if source_normalized == target_normalized and source.resolution != target.resolution:
            return "resample"

        # Check if transformation is available (using normalized spaces)
        if not can_transform_between(source, target):
            available = query_available_transforms()
            raise TransformNotAvailableError(
                source_space=source.identifier,
                target_space=target.identifier,
                supported_transforms=available,
            )

        # Determine direction based on normalized space identifiers
        # Forward: NLin6 -> NLin2009c
        if source_normalized == "MNI152NLin6Asym" and target_normalized == "MNI152NLin2009cAsym":
            return "forward"
        # Reverse: NLin2009c -> NLin6
        elif source_normalized == "MNI152NLin2009cAsym" and target_normalized == "MNI152NLin6Asym":
            return "reverse"

        # Should not reach here if can_transform_between passed
        raise TransformNotAvailableError(
            source_space=source.identifier,
            target_space=target.identifier,
            available_transforms=query_available_transforms(),
        )

    def select_interpolation(
        self, img: nib.Nifti1Image, method: InterpolationMethod | None = None
    ) -> InterpolationMethod:
        """Select appropriate interpolation method based on image data.

        Args:
            img: Image to transform
            method: Override interpolation method (if None, auto-detect)

        Returns:
            Interpolation method to use
        """
        if method is not None:
            return method

        # Auto-detect: use nearest neighbor for binary/integer data
        data = img.get_fdata()

        # Check if data is binary (only 0 and 1)
        unique_vals = np.unique(data)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            return InterpolationMethod.NEAREST

        # Check if data is integer-valued (likely label map)
        if np.allclose(data, np.round(data)):
            return InterpolationMethod.NEAREST

        # Default to linear for continuous data
        return InterpolationMethod.LINEAR

    def apply_resampling(
        self,
        img: nib.Nifti1Image,
        target_space: CoordinateSpace,
        interpolation: InterpolationMethod | None = None,
    ) -> nib.Nifti1Image:
        """Resample image to different resolution in same coordinate space.

        Args:
            img: Image to resample
            target_space: Target coordinate space (with desired resolution)
            interpolation: Interpolation method override

        Returns:
            Resampled image
        """
        from nilearn.image import resample_img

        # Select interpolation method
        interp_method = self.select_interpolation(img, interpolation)
        interp_str = "nearest" if interp_method == InterpolationMethod.NEAREST else "continuous"

        # Calculate target affine with new resolution
        source_affine = img.affine
        target_affine = source_affine.copy()

        # Get current resolution from affine diagonal
        current_res = abs(source_affine[0, 0])  # Assume isotropic for simplicity
        target_res = target_space.resolution

        # Update affine diagonal to target resolution (preserve sign)
        for i in range(3):
            if source_affine[i, i] >= 0:
                target_affine[i, i] = target_res
            else:
                target_affine[i, i] = -target_res

        # Calculate target shape based on resolution change
        scale_factor = current_res / target_res
        target_shape = tuple(int(s * scale_factor) for s in img.shape[:3])

        logger.debug(
            f"Resampling: {img.shape} @ {current_res}mm -> {target_shape} @ {target_res}mm"
        )

        return resample_img(
            img,
            target_affine=target_affine,
            target_shape=target_shape,
            interpolation=interp_str,
            force_resample=True,
            copy_header=True,
        )

    def apply_transformation(
        self,
        img: nib.Nifti1Image,
        source: CoordinateSpace,
        target: CoordinateSpace,
        transform: nitl.Affine,
        interpolation: InterpolationMethod | None = None,
    ) -> nib.Nifti1Image:
        """Apply spatial transformation to image data.

        Args:
            img: Image to transform
            source: Source coordinate space
            target: Target coordinate space
            transform: Nitransforms affine transformation object
            interpolation: Interpolation method (auto-detected if None)

        Returns:
            Transformed image in target space

        Raises:
            TransformNotAvailableError: If transformation not supported
        """
        # Determine direction
        direction = self.determine_direction(source, target)

        if direction == "none":
            # No transformation needed
            return img

        # Select interpolation method
        interp_method = self.select_interpolation(img, interpolation)

        # Apply transformation using nitransforms
        logger.info(
            f"Applying {direction} transformation: {source.identifier} "
            f"-> {target.identifier} using {interp_method.value} interpolation"
        )

        # Set reference space for the transform
        # Create a reference image in target space with appropriate resolution
        from lacuna.assets.templates import load_template

        # Try to load reference template
        template_name = f"{target.identifier}_res-{target.resolution}"
        try:
            reference_img = load_template(template_name)
        except (KeyError, FileNotFoundError):
            reference_img = None

        if reference_img is not None:
            # Load reference image
            reference_nifti = nib.load(reference_img)
            transform.reference = reference_nifti
        else:
            # If no template available, create a reference from target affine
            # This creates an empty reference with the correct space properties

            # Standard MNI template dimensions based on space and resolution
            # These dimensions match the actual templates in lacuna/data/templates
            dimension_map = {
                ("MNI152NLin6Asym", 1): (182, 218, 182),
                ("MNI152NLin6Asym", 2): (91, 109, 91),
                ("MNI152NLin2009cAsym", 1): (193, 229, 193),
                ("MNI152NLin2009cAsym", 2): (97, 115, 97),
                ("MNI152NLin2009bAsym", 0.5): (394, 466, 378),
            }

            shape = dimension_map.get(
                (target.identifier, target.resolution),
                # Fallback: estimate from resolution
                tuple(int(193 // target.resolution) for _ in range(3)),
            )

            reference_data = np.zeros(shape, dtype=np.uint8)
            reference_nifti = nib.Nifti1Image(reference_data, target.reference_affine)
            transform.reference = reference_nifti

        # Apply the transform
        # Handle asyncio event loop conflict in Jupyter notebooks
        # nitransforms uses asyncio.run() which fails if an event loop is already running

        # Check image dimensionality and handle accordingly
        img_data = img.get_fdata()
        original_shape = img_data.shape

        if img_data.ndim == 4:
            # Check if we have singleton dimensions we can squeeze
            if img_data.shape[3] == 1:
                logger.debug(f"Squeezing singleton 4th dimension from shape {original_shape}")
                img_data = np.squeeze(img_data, axis=3)
                img = nib.Nifti1Image(img_data, img.affine, img.header)
            else:
                # 4D atlas with multiple volumes - transform each volume independently
                n_volumes = img_data.shape[3]
                logger.info(
                    f"Transforming 4D image with {n_volumes} volumes from "
                    f"{source.identifier}@{source.resolution}mm to "
                    f"{target.identifier}@{target.resolution}mm (shape: {img.shape})"
                )

                # Transform each volume independently
                transformed_volumes = []
                for vol_idx in range(n_volumes):
                    logger.debug(f"Transforming volume {vol_idx + 1}/{n_volumes}")

                    # Extract single volume as 3D image
                    vol_data = img_data[..., vol_idx]
                    vol_img = nib.Nifti1Image(vol_data, img.affine, img.header)

                    # Transform this volume
                    try:
                        transformed_vol = transform.apply(
                            vol_img, order=self._get_interpolation_order(interp_method)
                        )
                    except RuntimeError as e:
                        if "asyncio.run() cannot be called from a running event loop" in str(e):
                            # We're in Jupyter - use nest_asyncio
                            try:
                                import nest_asyncio

                                nest_asyncio.apply()
                                transformed_vol = transform.apply(
                                    vol_img, order=self._get_interpolation_order(interp_method)
                                )
                            except ImportError:
                                raise RuntimeError(
                                    "Running spatial transformations in Jupyter notebooks requires nest_asyncio. "
                                    "Install with: pip install nest-asyncio"
                                ) from e
                        else:
                            raise

                    transformed_volumes.append(transformed_vol.get_fdata())

                # Stack all transformed volumes back into 4D
                transformed_4d_data = np.stack(transformed_volumes, axis=-1)
                transformed = nib.Nifti1Image(
                    transformed_4d_data,
                    transformed_vol.affine,  # Use affine from last transformed volume
                    transformed_vol.header,
                )

                logger.info(
                    f"4D transformation complete. Output shape: {transformed.shape}, "
                    f"dtype: {transformed.get_fdata().dtype}"
                )

                return transformed
        elif img_data.ndim > 4:
            raise ValueError(
                f"Cannot transform {img_data.ndim}D image. Expected 3D or 4D image. Shape: {original_shape}"
            )

        # 3D image transformation (original logic)
        logger.info(
            f"Transforming image from {source.identifier}@{source.resolution}mm "
            f"to {target.identifier}@{target.resolution}mm (shape: {img.shape})"
        )

        try:
            transformed = transform.apply(img, order=self._get_interpolation_order(interp_method))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # We're in a Jupyter notebook - use nest_asyncio to allow nested event loops
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    transformed = transform.apply(
                        img, order=self._get_interpolation_order(interp_method)
                    )
                except ImportError:
                    raise RuntimeError(
                        "Running spatial transformations in Jupyter notebooks requires nest_asyncio. "
                        "Install with: pip install nest-asyncio"
                    ) from e
            else:
                raise

        logger.info(
            f"Transformation complete. Output shape: {transformed.shape}, "
            f"dtype: {transformed.get_fdata().dtype}"
        )

        return transformed

    def _get_interpolation_order(self, method: InterpolationMethod) -> int:
        """Map interpolation method to scipy order parameter.

        Args:
            method: Interpolation method

        Returns:
            Scipy interpolation order (0-3)
        """
        mapping = {
            InterpolationMethod.NEAREST: 0,
            InterpolationMethod.LINEAR: 1,
            InterpolationMethod.CUBIC: 3,
        }
        return mapping[method]


def transform_image(
    img: nib.Nifti1Image,
    source_space: str,
    target_space: CoordinateSpace | str,
    source_resolution: int | None = None,
    interpolation: InterpolationMethod | str | None = None,
) -> nib.Nifti1Image:
    """Transform a NIfTI image between coordinate spaces.

    This is a low-level, generic function for transforming any NIfTI image
    between coordinate spaces. Use this when working with atlases, templates,
    or other non-lesion images.

    Args:
        img: NIfTI image to transform
        source_space: Source coordinate space identifier (e.g., "MNI152NLin6Asym")
        target_space: Target coordinate space object or identifier string
        source_resolution: Source resolution in mm (default: infer from affine)
        interpolation: Interpolation method (auto-detected if None).
            Can be InterpolationMethod enum or string ('nearest', 'linear', 'cubic')

    Returns:
        Transformed NIfTI image in target space

    Raises:
        TransformNotAvailableError: If transformation not supported

    Examples:
        >>> from lacuna.spatial.transform import transform_image
        >>> from lacuna.core.spaces import CoordinateSpace
        >>> import nibabel as nib
        >>> # Load atlas in NLin6 space
        >>> atlas = nib.load("atlas_NLin6.nii.gz")
        >>> # Define target space
        >>> target = CoordinateSpace("MNI152NLin2009cAsym", 2, reference_affine=...)
        >>> # Transform atlas using nearest neighbor (preserve labels)
        >>> transformed = transform_image(atlas, "MNI152NLin6Asym", target, interpolation='nearest')
    """
    from lacuna.core.spaces import REFERENCE_AFFINES

    # Convert string interpolation to enum if needed
    if isinstance(interpolation, str):
        interp_map = {
            "nearest": InterpolationMethod.NEAREST,
            "linear": InterpolationMethod.LINEAR,
            "cubic": InterpolationMethod.CUBIC,
        }
        interpolation = interp_map.get(interpolation.lower())
        if interpolation is None:
            raise ValueError(
                "Invalid interpolation string. Must be one of: 'nearest', 'linear', 'cubic'"
            )

    # Infer source resolution if not provided
    if source_resolution is None:
        source_resolution = int(round(abs(img.affine[0, 0])))

    # Create source CoordinateSpace
    source_space_obj = CoordinateSpace(
        identifier=source_space,
        resolution=source_resolution,
        reference_affine=REFERENCE_AFFINES.get((source_space, source_resolution), img.affine),
    )

    # Convert target_space to CoordinateSpace if it's a string
    if isinstance(target_space, str):
        # Infer target resolution from source if not explicitly different
        target_resolution = source_resolution
        target_space_obj = CoordinateSpace(
            identifier=target_space,
            resolution=target_resolution,
            reference_affine=REFERENCE_AFFINES.get(
                (target_space, target_resolution),
                source_space_obj.reference_affine,  # Use source as fallback
            ),
        )
    else:
        target_space_obj = target_space

    # Check if transformation needed
    strategy = TransformationStrategy()
    direction = strategy.determine_direction(source_space_obj, target_space_obj)

    if direction == "none":
        logger.info("Source and target spaces match - no transformation needed")
        return img

    # Handle resolution-only change (same space, different resolution)
    if direction == "resample":
        logger.info(
            f"Resampling from {source_space_obj.resolution}mm to {target_space_obj.resolution}mm "
            f"in {source_space_obj.identifier}"
        )
        return strategy.apply_resampling(img, target_space_obj, interpolation)

    # Load transform from asset registry
    from lacuna.assets.transforms import load_transform

    transform_name = f"{source_space_obj.identifier}_to_{target_space_obj.identifier}"
    try:
        transform_path = load_transform(transform_name)
    except (KeyError, FileNotFoundError) as e:
        from lacuna.core.exceptions import TransformNotAvailableError

        raise TransformNotAvailableError(
            source_space_obj.identifier,
            target_space_obj.identifier,
            supported_transforms=query_available_transforms(),
        ) from e

    # Load transform with nitransforms
    try:
        from nitransforms import linear as nitl

        transform = nitl.load(transform_path)
    except ImportError as e:
        raise ImportError(
            "nitransforms package is required for spatial transformations. "
            "Install with: pip install nitransforms"
        ) from e

    # Apply transformation
    return strategy.apply_transformation(
        img,
        source_space_obj,
        target_space_obj,
        transform,
        interpolation,
    )


def transform_mask_data(
    mask_data: "MaskData",
    target_space: CoordinateSpace,
    interpolation: InterpolationMethod | str | None = None,
) -> "MaskData":
    """Transform lesion data to target coordinate space.

    This is the high-level API for transforming MaskData objects between
    coordinate spaces. It handles:
    - Space detection and validation
    - Transform loading and caching
    - Transformation application
    - Provenance tracking

    Args:
        mask_data: MaskData object to transform
        target_space: Target coordinate space
        interpolation: Interpolation method (auto-detected if None).
            Can be InterpolationMethod enum or string ('nearest', 'linear', 'cubic')

    Returns:
        New MaskData object in target space

    Raises:
        TransformNotAvailableError: If transformation not supported
        SpaceDetectionError: If source space cannot be determined

    Examples:
        >>> from lacuna.core.mask_data import MaskData
        >>> from lacuna.core.spaces import CoordinateSpace, REFERENCE_AFFINES
        >>> # Load lesion in NLin6 space
        >>> lesion = MaskData.from_nifti("lesion.nii.gz", metadata={"space": "MNI152NLin6Asym", "resolution": 2})
        >>> # Transform to NLin2009c
        >>> target = CoordinateSpace("MNI152NLin2009cAsym", 2, REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)])
        >>> transformed = transform_mask_data(lesion, target)
    """
    # Import here to avoid circular imports
    from lacuna.core.mask_data import MaskData
    from lacuna.core.provenance import TransformationRecord

    # Get source space from metadata
    source_identifier = mask_data.metadata.get("space")
    source_resolution = mask_data.metadata.get("resolution", 2)

    if source_identifier is None:
        from pathlib import Path

        from lacuna.core.exceptions import SpaceDetectionError

        raise SpaceDetectionError(
            filepath=Path("unknown"),
            attempted_methods=["metadata lookup"],
        )

    # Use the generic transform_image function
    transformed_img = transform_image(
        img=mask_data.mask_img,
        source_space=source_identifier,
        target_space=target_space,
        source_resolution=source_resolution,
        interpolation=interpolation,
    )

    # If image unchanged, no transformation was needed
    if transformed_img is mask_data.mask_img:
        return mask_data

    # Create transformation record for provenance
    strategy = TransformationStrategy()
    interp_method = strategy.select_interpolation(mask_data.mask_img, interpolation)

    from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace

    source_space_obj = CoordinateSpace(
        identifier=source_identifier,
        resolution=source_resolution,
        reference_affine=REFERENCE_AFFINES.get(
            (source_identifier, source_resolution), mask_data.affine
        ),
    )
    direction = strategy.determine_direction(source_space_obj, target_space)

    transform_record = TransformationRecord(
        source_space=source_identifier,
        source_resolution=source_resolution,
        target_space=target_space.identifier,
        target_resolution=target_space.resolution,
        method=(
            "nitransforms"
            if direction == "forward" or direction == "reverse"
            else "nilearn_resample"
        ),
        interpolation=interp_method.value,
        rationale=(
            f"Automatic transformation for {direction} direction"
            if direction != "resample"
            else "Resolution change within same coordinate space"
        ),
    )

    # Create new MaskData with transformed image
    new_metadata = mask_data.metadata.copy()
    new_metadata["space"] = target_space.identifier
    new_metadata["resolution"] = target_space.resolution

    new_provenance = mask_data.provenance.copy()
    new_provenance.append(transform_record.to_dict())

    return MaskData(
        mask_img=transformed_img,
        metadata=new_metadata,
        provenance=new_provenance,
        results=mask_data.results,
    )


def query_available_transforms() -> list[tuple[str, str]]:
    """Query available spatial transformations.

    Returns a list of supported (source_space, target_space) pairs for
    spatial transformations. This includes both actual transforms and
    equivalent space mappings (e.g., MNI152NLin2009aAsym is equivalent
    to MNI152NLin2009cAsym).

    Returns
    -------
    list[tuple[str, str]]
        List of (source, target) space identifier pairs that can be transformed.

    Examples
    --------
    >>> transforms = query_available_transforms()
    >>> ('MNI152NLin6Asym', 'MNI152NLin2009cAsym') in transforms
    True
    >>> ('MNI152NLin6Asym', 'MNI152NLin2009aAsym') in transforms  # Via equivalence
    True
    """
    # Base transforms available in TemplateFlow
    base_transforms = [
        ("MNI152NLin6Asym", "MNI152NLin2009cAsym"),
        ("MNI152NLin2009cAsym", "MNI152NLin6Asym"),
    ]

    # Add equivalent space transforms
    # Since a/b/cAsym are anatomically identical, we can transform between them
    all_transforms = base_transforms.copy()

    # Add transforms from NLin6 to all NLin2009 variants
    for variant in ["MNI152NLin2009aAsym", "MNI152NLin2009bAsym"]:
        all_transforms.extend(
            [
                ("MNI152NLin6Asym", variant),
                (variant, "MNI152NLin6Asym"),
            ]
        )

    # Add identity transforms for equivalent spaces (same anatomy, no actual transform needed)
    nlin2009_variants = ["MNI152NLin2009aAsym", "MNI152NLin2009bAsym", "MNI152NLin2009cAsym"]
    for i, space1 in enumerate(nlin2009_variants):
        for space2 in nlin2009_variants[i + 1 :]:
            all_transforms.extend(
                [
                    (space1, space2),
                    (space2, space1),
                ]
            )

    return all_transforms


def can_transform_between(source: CoordinateSpace, target: CoordinateSpace) -> bool:
    """Check if transformation is possible between two coordinate spaces.

    This includes checking for:
    1. Identity (same space and resolution)
    2. Resampling (same space, different resolution)
    3. Actual transforms (different spaces)
    4. Equivalent spaces (anatomically identical spaces)

    Args:
        source: Source coordinate space
        target: Target coordinate space

    Returns:
        True if transformation is supported, False otherwise

    Examples
    --------
    >>> from lacuna.core.spaces import CoordinateSpace, REFERENCE_AFFINES
    >>> source = CoordinateSpace('MNI152NLin6Asym', 2, REFERENCE_AFFINES[('MNI152NLin6Asym', 2)])
    >>> target = CoordinateSpace('MNI152NLin2009cAsym', 2, REFERENCE_AFFINES[('MNI152NLin2009cAsym', 2)])
    >>> can_transform_between(source, target)
    True
    >>> # Also works with equivalent spaces
    >>> target_a = CoordinateSpace('MNI152NLin2009aAsym', 2, REFERENCE_AFFINES.get(('MNI152NLin2009aAsym', 2), target.reference_affine))
    >>> can_transform_between(source, target_a)
    True
    """
    # Normalize space identifiers
    source_normalized = _canonicalize_space_variant(source.identifier)
    target_normalized = _canonicalize_space_variant(target.identifier)

    # Same space (after normalization) - always possible (identity or resample)
    if source_normalized == target_normalized:
        return True

    # Check if transform pair is supported
    available_transforms = query_available_transforms()
    return (source.identifier, target.identifier) in available_transforms


__all__ = [
    "query_available_transforms",
    "can_transform_between",
    "TransformationStrategy",
    "InterpolationMethod",
    "transform_mask_data",
]
