"""Unified asset management system for Lacuna.

This module provides centralized management of all neuroimaging assets:
- Parcellations (bundled and user-registered)
- Templates (from TemplateFlow)
- Transforms (from TemplateFlow)
- Connectomes (structural and functional, user-registered)

All asset types follow a consistent registry pattern with register/list/load functions.
"""

# Import parcellation management
from lacuna.assets import parcellations  # noqa: F401
from lacuna.assets.parcellations import (
    BUNDLED_PARCELLATIONS_DIR,
    PARCELLATION_REGISTRY,
    Parcellation,
    ParcellationMetadata,
    list_parcellations,
    load_parcellation,
    register_parcellation,
    register_parcellation_from_files,
    register_parcellations_from_directory,
    unregister_parcellation,
)

# Keep atlases import for existing code (will be removed in separate commit)
from lacuna.assets.atlases import (
    ATLAS_REGISTRY,
    Atlas,
    AtlasMetadata,
    list_atlases,
    load_atlas,
    register_atlas,
    register_atlas_from_files,
    register_atlases_from_directory,
    unregister_atlas,
)

# Base classes
from lacuna.assets.base import (
    AssetMetadata,
    AssetRegistry,
    SpatialAssetMetadata,
)

# Connectomes
from lacuna.assets.connectomes import (
    FunctionalConnectome,
    FunctionalConnectomeMetadata,
    StructuralConnectome,
    StructuralConnectomeMetadata,
    list_functional_connectomes,
    list_structural_connectomes,
    load_functional_connectome,
    load_structural_connectome,
    register_functional_connectome,
    register_structural_connectome,
    unregister_functional_connectome,
    unregister_structural_connectome,
)

# Templates
from lacuna.assets.templates import (
    TemplateMetadata,
    is_template_cached,
    list_templates,
    load_template,
)

# Transforms
from lacuna.assets.transforms import (
    TransformMetadata,
    is_transform_cached,
    list_transforms,
    load_transform,
)

# Connectomes

# Templates

# Transforms

__all__ = [
    # Base classes
    "AssetMetadata",
    "SpatialAssetMetadata",
    "AssetRegistry",
    # Atlases
    "Atlas",
    "AtlasMetadata",
    "ATLAS_REGISTRY",
    "list_atlases",
    "load_atlas",
    "register_atlas",
    "register_atlas_from_files",
    "register_atlases_from_directory",
    "unregister_atlas",
    # Templates
    "TemplateMetadata",
    "list_templates",
    "load_template",
    "is_template_cached",
    # Transforms
    "TransformMetadata",
    "list_transforms",
    "load_transform",
    "is_transform_cached",
    # Structural Connectomes
    "StructuralConnectomeMetadata",
    "StructuralConnectome",
    "register_structural_connectome",
    "unregister_structural_connectome",
    "list_structural_connectomes",
    "load_structural_connectome",
    # Functional Connectomes
    "FunctionalConnectomeMetadata",
    "FunctionalConnectome",
    "register_functional_connectome",
    "unregister_functional_connectome",
    "list_functional_connectomes",
    "load_functional_connectome",
]
