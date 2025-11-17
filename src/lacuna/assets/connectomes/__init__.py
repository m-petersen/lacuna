"""Connectome asset management for Lacuna.

This module provides connectome registry and loading for both
structural (tractography-based) and functional (fMRI voxel-wise) connectomes.

Structural connectomes are used for structural lesion network mapping (sLNM)
and require tractogram + TDI files.

Functional connectomes are used for functional lesion network mapping (fLNM)
and require HDF5 files with voxel-wise timeseries data (not parcellated matrices).
"""

from lacuna.assets.connectomes.functional import (
    FunctionalConnectome,
    list_functional_connectomes,
    load_functional_connectome,
    register_functional_connectome,
    unregister_functional_connectome,
)
from lacuna.assets.connectomes.registry import (
    FunctionalConnectomeMetadata,
    StructuralConnectomeMetadata,
)
from lacuna.assets.connectomes.structural import (
    StructuralConnectome,
    list_structural_connectomes,
    load_structural_connectome,
    register_structural_connectome,
    unregister_structural_connectome,
)

__all__ = [
    # Metadata
    "StructuralConnectomeMetadata",
    "FunctionalConnectomeMetadata",
    # Structural
    "StructuralConnectome",
    "register_structural_connectome",
    "unregister_structural_connectome",
    "list_structural_connectomes",
    "load_structural_connectome",
    # Functional
    "FunctionalConnectome",
    "register_functional_connectome",
    "unregister_functional_connectome",
    "list_functional_connectomes",
    "load_functional_connectome",
]
