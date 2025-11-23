"""Parcellation asset management for Lacuna.

This module provides parcellation registry, loading, and management functions.
"""

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
]
