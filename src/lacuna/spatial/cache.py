"""LRU cache for transforms and transformation results.

This module implements a least-recently-used (LRU) cache for storing:
1. Transform objects (e.g., paths to .h5 transform files)
2. Transformed image results (to avoid recomputation)

The cache automatically evicts oldest entries when size limit is exceeded.
"""

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import nibabel as nib

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lacuna" / "transforms"

# Default max cache size (1GB)
DEFAULT_MAX_SIZE_MB = 1024


class TransformCache:
    """LRU cache for spatial transformations.

    Caches both transform objects and transformed image results to improve
    performance when processing multiple subjects with the same transformations.

    Attributes:
        cache_dir: Directory for storing cached results
        max_size_mb: Maximum cache size in megabytes
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    ):
        """Initialize transform cache.

        Args:
            cache_dir: Directory for cache storage (default: ~/.cache/lacuna/transforms)
            max_size_mb: Maximum cache size in megabytes (default: 1024)
        """
        self.cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        self.max_size_mb = max_size_mb
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches (LRU)
        self._transform_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._result_cache: OrderedDict[str, Path] = OrderedDict()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_size_bytes = 0

        # Load existing cache size
        self._update_cache_size()

    def _make_transform_key(self, source_space: str, target_space: str) -> tuple[str, str]:
        """Create ordered key for transform lookup.

        Transforms are bidirectional, so we use sorted order.

        Args:
            source_space: Source coordinate space
            target_space: Target coordinate space

        Returns:
            Ordered tuple (space1, space2) where space1 <= space2
        """
        return tuple(sorted([source_space, target_space]))

    def _hash_image(self, img: nib.Nifti1Image) -> str:
        """Compute hash of image data and affine.

        Args:
            img: NIfTI image

        Returns:
            SHA256 hash string
        """
        hasher = hashlib.sha256()

        # Hash image data
        data = img.get_fdata()
        hasher.update(data.tobytes())

        # Hash affine
        hasher.update(img.affine.tobytes())

        return hasher.hexdigest()

    def _make_result_key(
        self, img: nib.Nifti1Image, source_space: str, target_space: str
    ) -> str:
        """Create cache key for transformed result.

        Args:
            img: Source image
            source_space: Source coordinate space
            target_space: Target coordinate space

        Returns:
            Cache key string
        """
        img_hash = self._hash_image(img)
        transform_key = self._make_transform_key(source_space, target_space)
        return f"{img_hash}_{transform_key[0]}_{transform_key[1]}"

    def _update_cache_size(self) -> None:
        """Update current cache size by scanning cache directory."""
        self._current_size_bytes = sum(
            f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()
        )

    def _evict_lru(self) -> None:
        """Evict least recently used entries until under size limit."""
        while self._current_size_bytes > self.max_size_bytes and self._result_cache:
            # Remove oldest entry
            key, cached_path = self._result_cache.popitem(last=False)

            # Delete file if it exists
            if cached_path.exists():
                file_size = cached_path.stat().st_size
                cached_path.unlink()
                self._current_size_bytes -= file_size
                self._evictions += 1

                logger.warning(
                    f"Evicted cached result {key} ({file_size / 1024 / 1024:.1f} MB). "
                    f"Cache size: {self._current_size_bytes / 1024 / 1024:.1f} MB / "
                    f"{self.max_size_mb:.1f} MB"
                )

    def get_transform(self, source_space: str, target_space: str) -> Any | None:
        """Retrieve cached transform object.

        Args:
            source_space: Source coordinate space
            target_space: Target coordinate space

        Returns:
            Transform object if cached, None otherwise
        """
        key = self._make_transform_key(source_space, target_space)

        if key in self._transform_cache:
            # Move to end (most recently used)
            self._transform_cache.move_to_end(key)
            self._hits += 1
            return self._transform_cache[key]

        self._misses += 1
        return None

    def put_transform(
        self, source_space: str, target_space: str, transform: Any
    ) -> None:
        """Store transform object in cache.

        Args:
            source_space: Source coordinate space
            target_space: Target coordinate space
            transform: Transform object to cache
        """
        key = self._make_transform_key(source_space, target_space)
        self._transform_cache[key] = transform
        self._transform_cache.move_to_end(key)

    def get_result(
        self, img: nib.Nifti1Image, source_space: str, target_space: str
    ) -> nib.Nifti1Image | None:
        """Retrieve cached transformation result.

        Args:
            img: Source image
            source_space: Source coordinate space
            target_space: Target coordinate space

        Returns:
            Transformed image if cached, None otherwise
        """
        key = self._make_result_key(img, source_space, target_space)

        if key in self._result_cache:
            cached_path = self._result_cache[key]

            if cached_path.exists():
                # Move to end (most recently used)
                self._result_cache.move_to_end(key)
                self._hits += 1

                # Load and return cached result
                return nib.load(str(cached_path))

            # File missing, remove from cache
            del self._result_cache[key]

        self._misses += 1
        return None

    def put_result(
        self,
        source_img: nib.Nifti1Image,
        source_space: str,
        target_space: str,
        result_img: nib.Nifti1Image,
    ) -> None:
        """Store transformation result in cache.

        Args:
            source_img: Source image
            source_space: Source coordinate space
            target_space: Target coordinate space
            result_img: Transformed result image
        """
        key = self._make_result_key(source_img, source_space, target_space)

        # Save result to disk
        cached_path = self.cache_dir / f"{key}.nii.gz"
        nib.save(result_img, str(cached_path))

        # Update cache metadata
        file_size = cached_path.stat().st_size
        self._current_size_bytes += file_size
        self._result_cache[key] = cached_path
        self._result_cache.move_to_end(key)

        # Evict if necessary
        self._evict_lru()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Hit rate (hits / total requests)
            - evictions: Number of evictions
            - size_mb: Current cache size in MB
            - num_transforms: Number of cached transforms
            - num_results: Number of cached results
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "size_mb": self._current_size_bytes / 1024 / 1024,
            "num_transforms": len(self._transform_cache),
            "num_results": len(self._result_cache),
        }


# Global cache singleton
_global_cache: TransformCache | None = None


def get_global_cache() -> TransformCache:
    """Get global transform cache singleton.

    Returns:
        Global TransformCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = TransformCache()

    return _global_cache


__all__ = [
    "TransformCache",
    "get_global_cache",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MAX_SIZE_MB",
]
