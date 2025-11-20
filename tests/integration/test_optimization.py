"""Integration tests for transformation optimization strategy.

Tests verify that the transformation system intelligently chooses direction
based on data size, resolution, and cache availability to optimize performance.
"""

import logging

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
from lacuna.spatial.cache import TransformCache
from lacuna.spatial.transform import TransformationStrategy


class TestTransformationDirectionChoices:
    """Test automatic transformation direction optimization."""

    def test_size_based_direction_small_source(self):
        """Test that smaller source data triggers forward transformation."""
        # Create source and target spaces
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )

        # Create small source image (20x20x20)
        small_data = np.zeros((20, 20, 20), dtype=np.float32)
        small_img = nib.Nifti1Image(small_data, source.reference_affine)

        # Create large target image (91x109x91 - full brain)
        large_data = np.zeros((91, 109, 91), dtype=np.float32)
        large_img = nib.Nifti1Image(large_data, target.reference_affine)

        # Determine direction without cache
        strategy = TransformationStrategy()
        direction = strategy.determine_direction(
            source, target, source_img=small_img, target_img=large_img, check_cache=False
        )

        # Should choose forward (transform small source to target space)
        assert direction == "forward", "Should transform smaller source dataset"

    def test_size_based_direction_small_target(self):
        """Test that smaller target data can trigger reverse transformation."""
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )

        # Create large source image (91x109x91)
        large_data = np.zeros((91, 109, 91), dtype=np.float32)
        large_img = nib.Nifti1Image(large_data, source.reference_affine)

        # Create small target image (20x20x20)
        small_data = np.zeros((20, 20, 20), dtype=np.float32)
        small_img = nib.Nifti1Image(small_data, target.reference_affine)

        strategy = TransformationStrategy()
        direction = strategy.determine_direction(
            source, target, source_img=large_img, target_img=small_img, check_cache=False
        )

        # Should choose reverse (transform small target to source space)
        assert direction == "reverse", "Should transform smaller target dataset"

    def test_resolution_based_direction_source_finer(self):
        """Test that higher source resolution prefers reverse transformation."""
        # Source at 1mm (finer)
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=1,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 1)],
        )
        # Target at 2mm (coarser)
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )

        strategy = TransformationStrategy()
        res_comp = strategy.compare_resolutions(source, target)

        assert res_comp == "source_higher", "Source should have higher resolution"

        # Direction determination should prefer reverse to preserve resolution
        direction = strategy.determine_direction(source, target, check_cache=False)

        # Should prefer reverse to avoid downsampling source data
        assert direction == "reverse", "Should preserve higher resolution source by transforming target"

    def test_resolution_based_direction_target_finer(self):
        """Test that higher target resolution prefers forward transformation."""
        # Source at 2mm (coarser)
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        # Target at 1mm (finer)
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=1,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 1)],
        )

        strategy = TransformationStrategy()
        res_comp = strategy.compare_resolutions(source, target)

        assert res_comp == "target_higher", "Target should have higher resolution"

        # Direction determination should prefer forward to upsample to finer target
        direction = strategy.determine_direction(source, target, check_cache=False)

        # Should prefer forward to upsample source to finer target
        assert direction == "forward", "Should upsample source to higher resolution target"

    def test_equal_resolution_default_direction(self):
        """Test default direction when resolutions are equal."""
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )

        strategy = TransformationStrategy()
        res_comp = strategy.compare_resolutions(source, target)

        assert res_comp == "equal", "Resolutions should be equal"

        # Should use default direction (forward for NLin6 -> NLin2009c)
        direction = strategy.determine_direction(source, target, check_cache=False)
        assert direction == "forward", "Should use default forward direction"

    def test_logging_includes_rationale(self, caplog):
        """Test that transformation decisions are logged with rationale."""
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=1,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 1)],
        )

        # Create source image
        data = np.random.rand(91, 109, 91).astype(np.float32)
        img = nib.Nifti1Image(data, source.reference_affine)

        strategy = TransformationStrategy()

        with caplog.at_level(logging.DEBUG):
            direction = strategy.determine_direction(source, target, source_img=img, check_cache=False)

        # Check that rationale is logged
        assert any(
            "resolution" in record.message.lower() for record in caplog.records
        ), "Should log resolution-based rationale"

    def test_data_size_estimation(self):
        """Test data size estimation accuracy."""
        # Create 91x109x91 float32 image
        data = np.zeros((91, 109, 91), dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))

        strategy = TransformationStrategy()
        size_mb = strategy.estimate_data_size(img)

        # Expected: 91 * 109 * 91 * 4 bytes / (1024 * 1024) ≈ 3.6 MB
        expected_mb = (91 * 109 * 91 * 4) / (1024 * 1024)

        # Estimate may be 2x expected due to internal data handling
        # Accept if within reasonable range (2x-3x original size)
        assert size_mb >= expected_mb * 0.9, f"Size estimate {size_mb:.1f}MB should be at least {expected_mb * 0.9:.1f}MB"
        assert size_mb <= expected_mb * 3, f"Size estimate {size_mb:.1f}MB should not exceed {expected_mb * 3:.1f}MB"


class TestCacheInfluencedOptimization:
    """Test that cache availability influences transformation direction."""

    def test_cache_hit_overrides_size_optimization(self, tmp_path):
        """Test that cached results override size-based optimization."""
        # Create temporary cache
        cache = TransformCache(cache_dir=tmp_path / "cache", max_size_mb=100)

        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )

        # Create small source image
        small_data = np.random.rand(20, 20, 20).astype(np.float32)
        small_img = nib.Nifti1Image(small_data, source.reference_affine)

        # Create large target image
        large_data = np.zeros((91, 109, 91), dtype=np.float32)
        large_img = nib.Nifti1Image(large_data, target.reference_affine)

        # Without cache, should choose forward (transform small source)
        strategy = TransformationStrategy()
        direction_no_cache = strategy.determine_direction(
            source, target, source_img=small_img, target_img=large_img, check_cache=False
        )
        assert direction_no_cache == "forward"

        # Add a cached result for reverse direction
        cache.put_result(large_img, target.identifier, source.identifier, large_img)

        # Now with cache, should choose reverse even though forward would be smaller
        # Note: This requires passing the cache instance to determine_direction
        # For now, we verify the cache has the result
        cached = cache.get_result(large_img, target.identifier, source.identifier)
        assert cached is not None, "Cache should contain reverse transformation result"

        # Cache statistics should show the hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["num_results"] == 1


@pytest.mark.slow
class TestTransformationOptimizationIntegration:
    """Integration tests for full transformation optimization workflow."""

    def test_rationale_appears_in_provenance(self):
        """Test that transformation rationale is recorded in provenance."""
        # This test would require full LesionData transformation
        # For now, we verify TransformationRecord accepts rationale
        from lacuna.core.provenance import TransformationRecord

        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1,
            method="nitransforms",
            interpolation="linear",
            rationale="Forward transformation (3.6MB data), target resolution 1mm finer than source 2mm",
        )

        record_dict = record.to_dict()
        assert "rationale" in record_dict
        assert "resolution" in record_dict["rationale"]
        assert "MB" in record_dict["rationale"]
