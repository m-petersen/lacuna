"""Atlas types and configuration for the lacuna atlas engine.

Provides VoxelAtlas for collections of named 3D brain maps, plus placeholder
classes ParcelAtlas and SurfaceAtlas reserved for future releases.  Also
exports NT target grouping constants, presets, filename parsers, target
resolution, and map-selection config parsing from :mod:`lacuna.atlas.config`.
"""

from .config import (
    ALL_TARGETS,
    NT_PRESETS,
    NT_TARGET_GROUPS,
    parse_map_selection,
    parse_publication_from_filename,
    parse_target_from_filename,
    resolve_targets,
)
from .scoring import (
    score_focal,
    score_functional_overlap,
    score_ace_temporal,
    score_structural_endpoints,
)
from .store import build_nt_atlas, load_atlas, save_atlas
from .types import ParcelAtlas, SurfaceAtlas, VoxelAtlas

__all__ = [
    # types
    "VoxelAtlas",
    "ParcelAtlas",
    "SurfaceAtlas",
    # config constants
    "NT_TARGET_GROUPS",
    "ALL_TARGETS",
    "NT_PRESETS",
    # config utilities
    "parse_target_from_filename",
    "parse_publication_from_filename",
    "resolve_targets",
    "parse_map_selection",
    # store
    "build_nt_atlas",
    "save_atlas",
    "load_atlas",
    # scoring
    "score_focal",
    "score_structural_endpoints",
    "score_functional_overlap",
    "score_ace_temporal",
]
