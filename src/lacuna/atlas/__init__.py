"""Atlas types for the lacuna atlas engine.

Provides VoxelAtlas for collections of named 3D brain maps, plus placeholder
classes ParcelAtlas and SurfaceAtlas reserved for future releases.
"""

from .types import ParcelAtlas, SurfaceAtlas, VoxelAtlas

__all__ = ["VoxelAtlas", "ParcelAtlas", "SurfaceAtlas"]
