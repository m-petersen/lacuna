"""Atlas asset management for Lacuna.

This module provides atlas registry, loading, and management functions.
"""

from lacuna.assets.atlases.loader import Atlas, load_atlas
from lacuna.assets.atlases.registry import (
    ATLAS_REGISTRY,
    AtlasMetadata,
    list_atlases,
    register_atlas,
    register_atlas_from_files,
    register_atlases_from_directory,
    unregister_atlas,
)

__all__ = [
    # Data classes
    "Atlas",
    "AtlasMetadata",
    # Registry
    "ATLAS_REGISTRY",
    # Functions
    "list_atlases",
    "load_atlas",
    "register_atlas",
    "register_atlas_from_files",
    "register_atlases_from_directory",
    "unregister_atlas",
]
