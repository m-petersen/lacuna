"""Contract tests for TransformCache infrastructure.

Tests the caching mechanism for spatial transformations.
Following TDD: tests written before implementation.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


class TestTransformCacheConstruction:
    """Tests for TransformCache initialization."""

    def test_can_create_with_default_cache_dir(self):
        """Cache can be created with default directory."""
        from lacuna.spatial.cache import TransformCache

        cache = TransformCache()
        assert cache is not None
        assert cache.cache_dir.exists()

    def test_can_create_with_custom_cache_dir(self):
        """Cache can be created with custom directory."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))
            assert cache.cache_dir == Path(tmpdir)

    def test_creates_cache_directory_if_not_exists(self):
        """Cache automatically creates directory if missing."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "transforms" / "cache"
            cache = TransformCache(cache_dir=cache_path)
            assert cache_path.exists()


class TestTransformCaching:
    """Tests for transform object caching."""

    def test_returns_none_for_missing_transform(self):
        """get_transform returns None when transform not cached."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))
            result = cache.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
            assert result is None

    def test_can_store_and_retrieve_transform(self):
        """Transform can be stored and retrieved."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            # Create mock transform object (string for simplicity)
            mock_transform = "mock_transform_h5_path"

            # Store transform
            cache.put_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym", mock_transform)

            # Retrieve transform
            result = cache.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
            assert result == mock_transform

    def test_transform_key_is_bidirectional(self):
        """Transform cache uses ordered keys (A->B same as B->A)."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            mock_transform = "mock_transform"
            cache.put_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym", mock_transform)

            # Should be retrievable in reverse order
            result = cache.get_transform("MNI152NLin2009cAsym", "MNI152NLin6Asym")
            assert result == mock_transform


class TestResultCaching:
    """Tests for transformed image result caching."""

    def test_returns_none_for_missing_result(self):
        """get_result returns None when result not cached."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            # Create a dummy image for hash computation
            data = np.zeros((10, 10, 10), dtype=np.float32)
            img = nib.Nifti1Image(data, affine=np.eye(4))

            result = cache.get_result(img, "MNI152NLin6Asym", "MNI152NLin2009cAsym")
            assert result is None

    def test_can_store_and_retrieve_result(self):
        """Transformed result can be stored and retrieved."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            # Create source image
            data = np.random.rand(10, 10, 10).astype(np.float32)
            source_img = nib.Nifti1Image(data, affine=np.eye(4))

            # Create result image
            result_data = np.random.rand(10, 10, 10).astype(np.float32)
            result_img = nib.Nifti1Image(result_data, affine=np.eye(4))

            # Store result
            cache.put_result(
                source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym", result_img
            )

            # Retrieve result
            retrieved = cache.get_result(
                source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym"
            )
            assert retrieved is not None
            assert isinstance(retrieved, nib.Nifti1Image)
            np.testing.assert_array_almost_equal(retrieved.get_fdata(), result_data)

    def test_different_images_have_different_cache_keys(self):
        """Different source images produce different cache keys."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            # Create two different images
            data1 = np.ones((10, 10, 10), dtype=np.float32)
            img1 = nib.Nifti1Image(data1, affine=np.eye(4))

            data2 = np.zeros((10, 10, 10), dtype=np.float32)
            img2 = nib.Nifti1Image(data2, affine=np.eye(4))

            # Store result for img1
            result_data = np.random.rand(10, 10, 10).astype(np.float32)
            result_img = nib.Nifti1Image(result_data, affine=np.eye(4))
            cache.put_result(img1, "MNI152NLin6Asym", "MNI152NLin2009cAsym", result_img)

            # img2 should not have cached result
            result = cache.get_result(img2, "MNI152NLin6Asym", "MNI152NLin2009cAsym")
            assert result is None


class TestCacheEviction:
    """Tests for LRU eviction policy."""

    def test_evicts_when_max_size_exceeded(self):
        """Cache evicts oldest entries when max size exceeded."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache with very small max size (1 MB)
            cache = TransformCache(cache_dir=Path(tmpdir), max_size_mb=1)

            # Store multiple large results to exceed limit
            for i in range(5):
                data = np.random.rand(100, 100, 100).astype(np.float32)  # ~4MB each
                source_img = nib.Nifti1Image(data, affine=np.eye(4))
                result_img = nib.Nifti1Image(data, affine=np.eye(4))

                cache.put_result(
                    source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym", result_img
                )

            # Cache should have evicted oldest entries
            stats = cache.get_stats()
            assert stats["evictions"] > 0


class TestCacheStats:
    """Tests for cache statistics."""

    def test_tracks_hits_and_misses(self):
        """Cache tracks hit and miss statistics."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            data = np.random.rand(10, 10, 10).astype(np.float32)
            source_img = nib.Nifti1Image(data, affine=np.eye(4))
            result_img = nib.Nifti1Image(data, affine=np.eye(4))

            # Miss
            cache.get_result(source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym")

            # Store
            cache.put_result(
                source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym", result_img
            )

            # Hit
            cache.get_result(source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym")

            stats = cache.get_stats()
            assert stats["hits"] == 1
            assert stats["misses"] == 1

    def test_calculates_hit_rate(self):
        """Cache calculates hit rate correctly."""
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))

            data = np.random.rand(10, 10, 10).astype(np.float32)
            source_img = nib.Nifti1Image(data, affine=np.eye(4))
            result_img = nib.Nifti1Image(data, affine=np.eye(4))

            # 1 miss
            cache.get_result(source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym")

            # Store
            cache.put_result(
                source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym", result_img
            )

            # 2 hits
            cache.get_result(source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym")
            cache.get_result(source_img, "MNI152NLin6Asym", "MNI152NLin2009cAsym")

            stats = cache.get_stats()
            assert stats["hit_rate"] == pytest.approx(2 / 3)  # 2 hits / 3 total


class TestGlobalCache:
    """Tests for global cache singleton."""

    def test_get_global_cache_returns_singleton(self):
        """get_global_cache returns same instance."""
        from lacuna.spatial.cache import get_global_cache

        cache1 = get_global_cache()
        cache2 = get_global_cache()

        assert cache1 is cache2

    def test_global_cache_is_usable(self):
        """Global cache can store and retrieve transforms."""
        from lacuna.spatial.cache import get_global_cache

        cache = get_global_cache()

        mock_transform = "global_mock_transform"
        cache.put_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym", mock_transform)

        result = cache.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        assert result == mock_transform
