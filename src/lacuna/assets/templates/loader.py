"""Template grid loading.

Lacuna uses MNI templates only as a resampling grid (affine + shape), never
their intensity data. This module returns tiny zero-filled "grid-only"
references bundled with the package (data/templates/), avoiding a TemplateFlow
runtime dependency and any redistribution of FSL's MNI152NLin6Asym image.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lacuna.assets.templates.registry import TEMPLATE_REGISTRY
from lacuna.spatial.transform import _canonicalize_space_variant

logger = logging.getLogger(__name__)


def load_template(name: str) -> Path:
    """Load a reference brain grid by name.

    Returns a bundled zero-filled grid-only reference (affine + shape) for the
    requested space/resolution — Lacuna never uses template intensity data.

    Supports space equivalence: anatomically identical spaces like
    MNI152NLin2009[abc]Asym are automatically normalized to their
    canonical form (cAsym).

    Parameters
    ----------
    name : str
        Template name from registry (e.g., "MNI152NLin2009cAsym_res-1")

    Returns
    -------
    Path
        Path to template NIfTI file

    Raises
    ------
    KeyError
        If template not found in registry
    FileNotFoundError
        If template download fails

    Examples
    --------
    >>> from lacuna.assets.templates import load_template
    >>>
    >>> # Load MNI template
    >>> template_path = load_template("MNI152NLin2009cAsym_res-1")
    >>> import nibabel as nib
    >>> template = nib.load(template_path)
    >>> print(template.shape)
    (193, 229, 193)
    """
    # Canonicalize space variant in template name before registry lookup
    # e.g., "MNI152NLin2009bAsym_res-2" -> "MNI152NLin2009cAsym_res-2"
    if "_res-" in name:
        space_part, res_part = name.rsplit("_res-", 1)
        canonical_space = _canonicalize_space_variant(space_part)
        canonical_name = f"{canonical_space}_res-{res_part}"
        if canonical_name != name:
            logger.info(
                f"Using space equivalence: {name} → {canonical_name} "
                f"(anatomically identical spaces)"
            )
            name = canonical_name

    # Get metadata from registry
    metadata = TEMPLATE_REGISTRY.get(name)

    # Normalize space to handle equivalence
    space_normalized = _canonicalize_space_variant(metadata.space)

    # Log if normalization occurred
    if space_normalized != metadata.space:
        logger.info(
            f"Using space equivalence: {metadata.space} → {space_normalized} "
            f"(anatomically identical spaces)"
        )

    # Lacuna uses templates only as a resampling *grid* (affine + shape), never
    # their intensity data. We therefore ship tiny zero-filled "grid-only"
    # references on the canonical grids (see scripts/generate_grid_references.py)
    # instead of downloading templates from TemplateFlow. This removes the
    # TemplateFlow runtime dependency and avoids redistributing FSL's
    # MNI152NLin6Asym intensity image.
    resolution = (
        int(metadata.resolution) if float(metadata.resolution).is_integer() else metadata.resolution
    )
    grid_dir = Path(__file__).parent.parent.parent / "data" / "templates"
    grid_path = grid_dir / f"{space_normalized}_res-{resolution}.nii.gz"

    if not grid_path.exists():
        available = sorted(p.stem.replace(".nii", "") for p in grid_dir.glob("*.nii.gz"))
        raise FileNotFoundError(
            f"No bundled grid reference for template '{name}' "
            f"(space={space_normalized}, res={resolution}; expected {grid_path.name}). "
            f"Available grids: {available}. "
            f"Regenerate with scripts/generate_grid_references.py if this grid should exist."
        )

    return grid_path


def is_template_cached(name: str) -> bool:
    """Check if template is already cached locally.

    Parameters
    ----------
    name : str
        Template name from registry

    Returns
    -------
    bool
        True if template is cached, False otherwise

    Examples
    --------
    >>> from lacuna.assets.templates import is_template_cached
    >>> is_template_cached("MNI152NLin2009cAsym_res-1")
    True
    """
    try:
        template_path = load_template(name)
        return template_path.exists()
    except (FileNotFoundError, KeyError):
        return False


__all__ = [
    "load_template",
    "is_template_cached",
]
