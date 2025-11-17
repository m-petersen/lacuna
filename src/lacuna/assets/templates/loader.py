"""Template loading with TemplateFlow integration.

This module provides functions to load reference brain templates,
automatically downloading from TemplateFlow as needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lacuna.assets.templates.registry import TEMPLATE_REGISTRY
from lacuna.spatial.transform import _canonicalize_space_variant

logger = logging.getLogger(__name__)


def load_template(name: str) -> Path:
    """Load a reference brain template by name.
    
    Downloads from TemplateFlow on first use and caches locally.
    
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
    
    try:
        import templateflow.api as tflow
    except ImportError as e:
        raise ImportError(
            "TemplateFlow is required for template loading. "
            "Install with: pip install templateflow"
        ) from e
    
    try:
        # Get template from TemplateFlow (using normalized space)
        template_path = tflow.get(
            space_normalized,
            resolution=metadata.resolution,
            desc=None,
            suffix=metadata.modality,
            extension=".nii.gz",
        )
        
        if template_path is None or (isinstance(template_path, list) and not template_path):
            raise ValueError(
                f"Template not found in TemplateFlow: {metadata.space} at {metadata.resolution}mm"
            )
        
        # TemplateFlow can return a list, take first item
        if isinstance(template_path, list):
            template_path = template_path[0]
        
        return Path(template_path)
        
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load template {name} "
            f"(space={metadata.space}, res={metadata.resolution}, modality={metadata.modality}): {e}"
        ) from e


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
