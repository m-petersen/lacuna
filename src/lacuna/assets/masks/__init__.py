"""Brain mask assets.

Binary brain masks per coordinate space and resolution, redistributed via OSF and
downloaded/cached on first use. See :func:`load_brain_mask`.
"""

from __future__ import annotations

from lacuna.assets.masks.loader import load_brain_mask
from lacuna.assets.masks.registry import (
    BRAIN_MASK_REGISTRY,
    BrainMaskMetadata,
    mask_name,
)

__all__ = [
    "load_brain_mask",
    "BRAIN_MASK_REGISTRY",
    "BrainMaskMetadata",
    "mask_name",
]
