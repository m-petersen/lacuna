"""Template registry with TemplateFlow integration.

This module provides registry and metadata for reference brain templates.
Templates are automatically downloaded from TemplateFlow on first use.
"""

from __future__ import annotations

from dataclasses import dataclass

from lacuna.assets.base import AssetRegistry, SpatialAssetMetadata


@dataclass(frozen=True)
class TemplateMetadata(SpatialAssetMetadata):
    """Metadata for a reference brain template.

    Attributes
    ----------
    name : str
        Template identifier (e.g., "MNI152NLin2009cAsym")
    space : str
        Coordinate space (same as name for templates)
    resolution : float
        Voxel resolution in mm
    description : str
        Human-readable description
    modality : str
        Image modality (e.g., "T1w", "T2w", "FLAIR")
    source : str
        Source of template (always "templateflow")
    """

    modality: str = "T1w"
    source: str = "templateflow"


# Create registry and populate with known templates
TEMPLATE_REGISTRY = AssetRegistry[TemplateMetadata]("template")

# Register known TemplateFlow templates
_KNOWN_TEMPLATES = [
    TemplateMetadata(
        name="MNI152NLin2009cAsym_res-1",
        space="MNI152NLin2009cAsym",
        resolution=1.0,
        description="MNI152 nonlinear 2009c asymmetric template, 1mm",
        modality="T1w",
    ),
    TemplateMetadata(
        name="MNI152NLin2009cAsym_res-2",
        space="MNI152NLin2009cAsym",
        resolution=2.0,
        description="MNI152 nonlinear 2009c asymmetric template, 2mm",
        modality="T1w",
    ),
    TemplateMetadata(
        name="MNI152NLin6Asym_res-1",
        space="MNI152NLin6Asym",
        resolution=1.0,
        description="MNI152 nonlinear 6th generation asymmetric template, 1mm",
        modality="T1w",
    ),
    TemplateMetadata(
        name="MNI152NLin6Asym_res-2",
        space="MNI152NLin6Asym",
        resolution=2.0,
        description="MNI152 nonlinear 6th generation asymmetric template, 2mm",
        modality="T1w",
    ),
]

for template in _KNOWN_TEMPLATES:
    TEMPLATE_REGISTRY.register(template)


def list_templates(
    space: str | None = None,
    resolution: float | None = None,
    modality: str | None = None,
) -> list[TemplateMetadata]:
    """List available templates from TemplateFlow.

    Parameters
    ----------
    space : str, optional
        Filter by coordinate space
    resolution : float, optional
        Filter by resolution in mm
    modality : str, optional
        Filter by modality (e.g., "T1w", "T2w")

    Returns
    -------
    list[TemplateMetadata]
        Matching templates

    Examples
    --------
    >>> from lacuna.assets.templates import list_templates
    >>>
    >>> # List all available templates
    >>> templates = list_templates()
    >>>
    >>> # Filter by space and resolution
    >>> mni_1mm = list_templates(space="MNI152NLin2009cAsym", resolution=1.0)
    """
    return TEMPLATE_REGISTRY.list(space=space, resolution=resolution, modality=modality)


__all__ = [
    "TemplateMetadata",
    "TEMPLATE_REGISTRY",
    "list_templates",
]
