"""Parcellation asset management for Lacuna.

This module provides parcellation registry, loading, and management functions.

Note: This module was previously named 'atlases'. Backward compatibility aliases
are provided but deprecated.
"""

import warnings
from typing import Any

from lacuna.assets.parcellations.loader import (
    BUNDLED_PARCELLATIONS_DIR,
    Parcellation,
    load_parcellation,
)
from lacuna.assets.parcellations.registry import (
    PARCELLATION_REGISTRY,
    ParcellationMetadata,
    list_parcellations,
    register_parcellation,
    register_parcellation_from_files,
    register_parcellations_from_directory,
    unregister_parcellation,
)

__all__ = [
    # Data classes
    "Parcellation",
    "ParcellationMetadata",
    # Registry
    "PARCELLATION_REGISTRY",
    # Constants
    "BUNDLED_PARCELLATIONS_DIR",
    # Functions
    "list_parcellations",
    "load_parcellation",
    "register_parcellation",
    "register_parcellation_from_files",
    "register_parcellations_from_directory",
    "unregister_parcellation",
    # Deprecated aliases (for backward compatibility)
    "Atlas",
    "AtlasMetadata",
    "ATLAS_REGISTRY",
    "BUNDLED_ATLASES_DIR",
    "list_atlases",
    "load_atlas",
    "register_atlas",
    "register_atlas_from_files",
    "register_atlases_from_directory",
    "unregister_atlas",
]

# Deprecated aliases for backward compatibility
_DEPRECATION_MSG = (
    "The 'atlases' terminology is deprecated. Use 'parcellations' instead. "
    "For example, use 'Parcellation' instead of 'Atlas', "
    "'load_parcellation()' instead of 'load_atlas()', etc. "
    "This alias will be removed in version 1.0."
)


def _deprecated_alias(new_name: str, old_name: str) -> Any:
    """Create a deprecated alias that warns on first access."""

    def getter():
        warnings.warn(
            f"{old_name} is deprecated, use {new_name} instead. {_DEPRECATION_MSG}",
            DeprecationWarning,
            stacklevel=3,
        )
        return globals()[new_name]

    return getter


# Create deprecated aliases using __getattr__ for lazy loading
def __getattr__(name: str) -> Any:
    """Provide deprecated atlas names with warnings."""
    deprecated_map = {
        "Atlas": "Parcellation",
        "AtlasMetadata": "ParcellationMetadata",
        "ATLAS_REGISTRY": "PARCELLATION_REGISTRY",
        "BUNDLED_ATLASES_DIR": "BUNDLED_PARCELLATIONS_DIR",
        "list_atlases": "list_parcellations",
        "load_atlas": "load_parcellation",
        "register_atlas": "register_parcellation",
        "register_atlas_from_files": "register_parcellation_from_files",
        "register_atlases_from_directory": "register_parcellations_from_directory",
        "unregister_atlas": "unregister_parcellation",
    }

    if name in deprecated_map:
        new_name = deprecated_map[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead. {_DEPRECATION_MSG}",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
