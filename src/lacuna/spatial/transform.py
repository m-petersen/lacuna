"""Transformation strategies for spatial coordinate space conversions."""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
from nitransforms import linear as nitl

from lacuna.core.exceptions import TransformNotAvailableError
from lacuna.core.spaces import CoordinateSpace

if TYPE_CHECKING:
    from lacuna.core.lesion_data import LesionData

logger = logging.getLogger(__name__)


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
        # Same space - no transform needed
        if source.identifier == target.identifier and source.resolution == target.resolution:
            return "none"

        # Check if transformation is available
        if not can_transform_between(source, target):
            available = query_available_transforms()
            raise TransformNotAvailableError(
                source_space=source.identifier,
                target_space=target.identifier,
                available_transforms=available,
            )

        # Determine direction based on space identifiers
        # Forward: NLin6 -> NLin2009c or NLin2009b -> NLin2009c
        if source.identifier == "MNI152NLin6Asym" and target.identifier == "MNI152NLin2009cAsym":
            return "forward"
        elif (
            source.identifier == "MNI152NLin2009bAsym"
            and target.identifier == "MNI152NLin2009cAsym"
        ):
            return "forward"
        # Reverse: NLin2009c -> NLin6 or NLin2009c -> NLin2009b
        elif source.identifier == "MNI152NLin2009cAsym" and target.identifier == "MNI152NLin6Asym":
            return "reverse"
        elif (
            source.identifier == "MNI152NLin2009cAsym"
            and target.identifier == "MNI152NLin2009bAsym"
        ):
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
        from lacuna.spatial.assets import DataAssetManager

        asset_manager = DataAssetManager()
        reference_img = asset_manager.get_template(target.identifier, resolution=target.resolution)

        if reference_img is not None:
            # Load reference image
            reference_nifti = nib.load(reference_img)
            transform.reference = reference_nifti
        else:
            # If no template available, create a reference from target affine
            # This creates an empty reference with the correct space properties
            import numpy as np

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
        transformed = transform.apply(img, order=self._get_interpolation_order(interp_method))

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


def transform_lesion_data(
    lesion_data: "LesionData",
    target_space: CoordinateSpace,
    interpolation: InterpolationMethod | None = None,
) -> "LesionData":
    """Transform lesion data to target coordinate space.

    This is the high-level API for transforming LesionData objects between
    coordinate spaces. It handles:
    - Space detection and validation
    - Transform loading and caching
    - Transformation application
    - Provenance tracking

    Args:
        lesion_data: LesionData object to transform
        target_space: Target coordinate space
        interpolation: Interpolation method (auto-detected if None)

    Returns:
        New LesionData object in target space

    Raises:
        TransformNotAvailableError: If transformation not supported
        SpaceDetectionError: If source space cannot be determined

    Examples:
        >>> from lacuna.core.lesion_data import LesionData
        >>> from lacuna.core.spaces import CoordinateSpace, REFERENCE_AFFINES
        >>> # Load lesion in NLin6 space
        >>> lesion = LesionData.from_nifti("lesion.nii.gz", metadata={"space": "MNI152NLin6Asym", "resolution": 2})
        >>> # Transform to NLin2009c
        >>> target = CoordinateSpace("MNI152NLin2009cAsym", 2, REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)])
        >>> transformed = transform_lesion_data(lesion, target)
    """
    # Import here to avoid circular imports
    from lacuna.core.lesion_data import LesionData
    from lacuna.core.provenance import TransformationRecord
    from lacuna.spatial.assets import DataAssetManager

    # Get source space from metadata
    source_identifier = lesion_data.metadata.get("space")
    source_resolution = lesion_data.metadata.get("resolution", 2)

    if source_identifier is None:
        from pathlib import Path

        from lacuna.core.exceptions import SpaceDetectionError

        raise SpaceDetectionError(
            filepath=Path("unknown"),
            attempted_methods=["metadata lookup"],
        )

    # Create source CoordinateSpace
    from lacuna.core.spaces import REFERENCE_AFFINES

    source_space = CoordinateSpace(
        identifier=source_identifier,
        resolution=source_resolution,
        reference_affine=REFERENCE_AFFINES.get(
            (source_identifier, source_resolution), lesion_data.affine
        ),
    )

    # Check if transformation needed
    strategy = TransformationStrategy()
    direction = strategy.determine_direction(source_space, target_space)

    if direction == "none":
        # No transformation needed - return copy with updated metadata
        logger.info("Source and target spaces match - no transformation needed")
        return lesion_data

    # Load transform from asset manager
    asset_manager = DataAssetManager()
    transform_path = asset_manager.get_transform(source_space.identifier, target_space.identifier)

    if transform_path is None:
        from lacuna.core.exceptions import TransformNotAvailableError

        raise TransformNotAvailableError(
            source_space.identifier,
            target_space.identifier,
            available_transforms=query_available_transforms(),
        )

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
    transformed_img = strategy.apply_transformation(
        lesion_data.lesion_img,
        source_space,
        target_space,
        transform,
        interpolation,
    )

    # Create transformation record for provenance
    interp_method = strategy.select_interpolation(lesion_data.lesion_img, interpolation)
    transform_record = TransformationRecord(
        source_space=source_space.identifier,
        source_resolution=source_space.resolution,
        target_space=target_space.identifier,
        target_resolution=target_space.resolution,
        method="nitransforms",
        interpolation=interp_method.value,
        rationale=f"Automatic transformation for {direction} direction",
    )

    # Create new LesionData with transformed image
    new_metadata = lesion_data.metadata.copy()
    new_metadata["space"] = target_space.identifier
    new_metadata["resolution"] = target_space.resolution

    new_provenance = lesion_data.provenance.copy()
    new_provenance.append(transform_record.to_dict())

    return LesionData(
        lesion_img=transformed_img,
        anatomical_img=lesion_data.anatomical_img,
        metadata=new_metadata,
        provenance=new_provenance,
        results=lesion_data.results,
    )


def query_available_transforms() -> list[tuple[str, str]]:
    """Query available spatial transformations.

    Returns a list of supported (source_space, target_space) pairs for
    spatial transformations.

    Returns
    -------
    list[tuple[str, str]]
        List of (source, target) space identifier pairs that can be transformed.

    Examples
    --------
    >>> transforms = query_available_transforms()
    >>> ('MNI152NLin6Asym', 'MNI152NLin2009cAsym') in transforms
    True
    """
    return [
        ("MNI152NLin6Asym", "MNI152NLin2009cAsym"),
        ("MNI152NLin2009cAsym", "MNI152NLin6Asym"),
        ("MNI152NLin2009bAsym", "MNI152NLin2009cAsym"),
        ("MNI152NLin2009cAsym", "MNI152NLin2009bAsym"),
    ]


def can_transform_between(source: CoordinateSpace, target: CoordinateSpace) -> bool:
    """Check if transformation is possible between two coordinate spaces.

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
    """
    # Same space - no transform needed
    if source.identifier == target.identifier and source.resolution == target.resolution:
        return True

    # Check if transform pair is supported
    available_transforms = query_available_transforms()
    return (source.identifier, target.identifier) in available_transforms


__all__ = [
    "query_available_transforms",
    "can_transform_between",
    "TransformationStrategy",
    "InterpolationMethod",
    "transform_lesion_data",
]
