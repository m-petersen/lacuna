"""Transform registry with TemplateFlow integration.

This module provides registry and metadata for spatial transformations
between coordinate spaces.
"""

from __future__ import annotations

from dataclasses import dataclass

from lacuna.assets.base import AssetMetadata, AssetRegistry


@dataclass(frozen=True)
class TransformMetadata(AssetMetadata):
    """Metadata for a spatial transformation.

    Attributes
    ----------
    name : str
        Transform identifier (e.g., "MNI152NLin6Asym_to_MNI152NLin2009cAsym")
    description : str
        Human-readable description
    from_space : str
        Source coordinate space
    to_space : str
        Target coordinate space
    transform_type : str
        Type of transform ("nonlinear", "affine", "composite")
    source : str
        Source of transform (always "templateflow")
    """

    from_space: str = ""
    to_space: str = ""
    transform_type: str = "nonlinear"
    source: str = "templateflow"

    def validate(self) -> None:
        """Validate transform metadata.

        Raises
        ------
        ValueError
            If metadata is invalid
        """
        from lacuna.core.spaces import SPACE_ALIASES, SUPPORTED_SPACES

        # Check spaces
        for space in [self.from_space, self.to_space]:
            if space and space not in SUPPORTED_SPACES and space not in SPACE_ALIASES:
                raise ValueError(f"Unsupported space: {space}. " f"Supported: {SUPPORTED_SPACES}")


# Create registry and populate with known transforms
TRANSFORM_REGISTRY = AssetRegistry[TransformMetadata]("transform")

# Register known TemplateFlow transforms
# Note: Transform files exist in TemplateFlow but may not be immediately discoverable
# via the API. They are downloaded on first use.
_KNOWN_TRANSFORMS = [
    TransformMetadata(
        name="MNI152NLin6Asym_to_MNI152NLin2009cAsym",
        description="Nonlinear transform from MNI152 6th gen to 2009c",
        from_space="MNI152NLin6Asym",
        to_space="MNI152NLin2009cAsym",
        transform_type="nonlinear",
    ),
    TransformMetadata(
        name="MNI152NLin2009cAsym_to_MNI152NLin6Asym",
        description="Nonlinear transform from MNI152 2009c to 6th gen",
        from_space="MNI152NLin2009cAsym",
        to_space="MNI152NLin6Asym",
        transform_type="nonlinear",
    ),
]

for transform in _KNOWN_TRANSFORMS:
    TRANSFORM_REGISTRY.register(transform)


def list_transforms(
    from_space: str | None = None,
    to_space: str | None = None,
) -> list[TransformMetadata]:
    """List available transforms from TemplateFlow.

    Parameters
    ----------
    from_space : str, optional
        Filter by source coordinate space
    to_space : str, optional
        Filter by target coordinate space

    Returns
    -------
    list[TransformMetadata]
        Matching transforms

    Examples
    --------
    >>> from lacuna.assets.transforms import list_transforms
    >>>
    >>> # List all available transforms
    >>> transforms = list_transforms()
    >>>
    >>> # Find transforms from NLin6 to NLin2009c
    >>> transforms = list_transforms(
    ...     from_space="MNI152NLin6Asym",
    ...     to_space="MNI152NLin2009cAsym"
    ... )
    """
    # Use registry filtering
    # Note: AssetRegistry.list() doesn't support from_space/to_space directly,
    # so we need to filter manually
    all_transforms = TRANSFORM_REGISTRY.list()

    if from_space is not None:
        all_transforms = [t for t in all_transforms if t.from_space == from_space]

    if to_space is not None:
        all_transforms = [t for t in all_transforms if t.to_space == to_space]

    return all_transforms


__all__ = [
    "TransformMetadata",
    "TRANSFORM_REGISTRY",
    "list_transforms",
]
