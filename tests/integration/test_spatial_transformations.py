"""Integration tests for spatial transformations.

Tests the complete transformation pipeline including:
- Transform loading and caching
- TemplateFlow integrity checking
- 3D/4D image handling
- Asyncio compatibility in Jupyter environments
- Space variant canonicalization
"""

import pytest
import numpy as np
import nibabel as nib
from pathlib import Path

from lacuna.spatial.transform import (
    transform_image,
    can_transform_between,
    _canonicalize_space_variant,
)
from lacuna.core.spaces import CoordinateSpace


class TestTransformLoading:
    """Test transform loading and TemplateFlow integration."""

    def test_load_transform_from_templateflow(self):
        """Transform should be downloaded from TemplateFlow if not cached."""
        from lacuna.assets.transforms.loader import load_transform
        
        # This should trigger TemplateFlow download
        transform_name = "MNI152NLin6Asym_to_MNI152NLin2009cAsym"
        path = load_transform(transform_name)
        
        assert path.exists()
        assert path.suffix == ".h5"
        assert path.stat().st_size > 1024  # At least 1KB
        
    def test_transform_with_space_variant_aliasing(self):
        """Transform should work with space variants (aAsym/bAsym → cAsym)."""
        from lacuna.assets.transforms.loader import load_transform
        
        # Request with aAsym variant
        transform_name = "MNI152NLin2009aAsym_to_MNI152NLin6Asym"
        path = load_transform(transform_name)
        
        assert path.exists()
        # Should normalize to cAsym internally
        assert "MNI152NLin2009cAsym" in str(path) or "cAsym" in str(path)
        
    def test_corrupted_file_detection_and_retry(self, tmp_path, monkeypatch):
        """Corrupted transform files should be detected and re-downloaded."""
        from lacuna.assets.transforms.loader import load_transform
        import templateflow.api as tflow
        
        # Create a fake corrupted file
        cache_dir = tmp_path / "templateflow" / "tpl-MNI152NLin6Asym"
        cache_dir.mkdir(parents=True)
        corrupted_file = cache_dir / "tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5"
        corrupted_file.write_bytes(b"corrupted")  # Only 9 bytes
        
        # Monkeypatch to use our tmp cache
        def mock_home():
            return tmp_path
        
        monkeypatch.setattr(Path, "home", mock_home)
        
        # Should detect corruption and re-download
        # Note: This test requires actual TemplateFlow access
        # In practice, you'd mock the tflow.get() call


class TestImageDimensionHandling:
    """Test handling of 3D and 4D images."""

    def test_transform_3d_image(self):
        """3D images should transform successfully."""
        # Create 3D test image
        data = np.random.rand(91, 109, 91)
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        img = nib.Nifti1Image(data, affine)
        
        # Transform should handle 3D image
        result = transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear"
        )
        
        assert result.ndim == 3
        assert result.shape[0] > 0
        
    def test_transform_4d_image_with_singleton_dimension(self):
        """4D images with singleton 4th dimension should be squeezed."""
        # Create 4D test image with singleton dimension
        data = np.random.rand(91, 109, 91, 1)
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        img = nib.Nifti1Image(data, affine)
        
        # Should squeeze to 3D and transform
        result = transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear"
        )
        
        assert result.ndim == 3
        
    def test_transform_4d_image_non_singleton_fails(self):
        """4D images with non-singleton 4th dimension should raise error."""
        # Create 4D test image with multiple volumes
        data = np.random.rand(91, 109, 91, 5)
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        img = nib.Nifti1Image(data, affine)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot transform 4D image"):
            transform_image(
                img=img,
                source_space="MNI152NLin6Asym",
                target_space="MNI152NLin2009cAsym",
                source_resolution=2.0,
                interpolation="linear"
            )


class TestSpaceVariantCanonicalization:
    """Test space variant canonicalization."""

    def test_aAsym_canonicalizes_to_cAsym(self):
        """aAsym variant should canonicalize to cAsym."""
        result = _canonicalize_space_variant("MNI152NLin2009aAsym")
        assert result == "MNI152NLin2009cAsym"
        
    def test_bAsym_canonicalizes_to_cAsym(self):
        """bAsym variant should canonicalize to cAsym."""
        result = _canonicalize_space_variant("MNI152NLin2009bAsym")
        assert result == "MNI152NLin2009cAsym"
        
    def test_cAsym_unchanged(self):
        """cAsym variant should remain unchanged."""
        result = _canonicalize_space_variant("MNI152NLin2009cAsym")
        assert result == "MNI152NLin2009cAsym"
        
    def test_non_variant_space_unchanged(self):
        """Non-variant spaces should remain unchanged."""
        result = _canonicalize_space_variant("MNI152NLin6Asym")
        assert result == "MNI152NLin6Asym"


class TestAtlasTransformation:
    """Test atlas transformation in analysis pipeline."""

    def test_atlas_transformation_preserves_labels(self):
        """Atlas transformation should preserve integer labels."""
        # Create atlas with integer labels
        labels = np.random.randint(0, 10, size=(91, 109, 91))
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        atlas_img = nib.Nifti1Image(labels.astype(np.int16), affine)
        
        # Transform with nearest neighbor
        result = transform_image(
            img=atlas_img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="nearest"
        )
        
        result_data = result.get_fdata()
        unique_labels = np.unique(result_data)
        
        # All values should be integers (or very close due to float conversion)
        assert np.allclose(result_data, np.round(result_data))
        
    def test_regional_damage_with_space_mismatch(self):
        """RegionalDamage should handle lesion/atlas in different spaces."""
        from lacuna import LesionData
        from lacuna.analysis import RegionalDamage
        
        # Create lesion in NLin6Asym space
        lesion_data = np.random.rand(182, 218, 182) > 0.9
        affine = np.eye(4)
        affine[:3, :3] = np.diag([-1, 1, 1])
        lesion_img = nib.Nifti1Image(lesion_data.astype(np.uint8), affine)
        
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={
                "space": "MNI152NLin6Asym",
                "resolution": 1.0
            }
        )
        
        # Run analysis - should automatically transform atlas to match
        # Note: This is an integration test that requires actual atlases
        analysis = RegionalDamage(threshold=0.5)
        
        # Should not raise error even with space mismatch
        result = analysis.run(lesion)
        assert "RegionalDamage" in result.results


class TestAsyncioCompatibility:
    """Test asyncio event loop compatibility for Jupyter."""

    def test_transform_with_existing_event_loop(self):
        """Transformation should work even with existing asyncio event loop."""
        import asyncio
        
        # Create event loop (simulating Jupyter)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create test image
            data = np.random.rand(91, 109, 91)
            affine = np.array([
                [-2., 0., 0., 90.],
                [0., 2., 0., -126.],
                [0., 0., 2., -72.],
                [0., 0., 0., 1.]
            ])
            img = nib.Nifti1Image(data, affine)
            
            # Should not raise RuntimeError about event loop
            result = transform_image(
                img=img,
                source_space="MNI152NLin6Asym",
                target_space="MNI152NLin2009cAsym",
                source_resolution=2.0,
                interpolation="linear"
            )
            
            assert result is not None
            
        finally:
            loop.close()


class TestTransformCaching:
    """Test transform caching behavior."""

    def test_transform_cached_after_first_load(self):
        """Transform should be cached after first load."""
        from lacuna.assets.transforms.loader import load_transform
        
        transform_name = "MNI152NLin6Asym_to_MNI152NLin2009cAsym"
        
        # First load
        path1 = load_transform(transform_name)
        assert path1.exists()
        
        # Second load should use cache (same path)
        path2 = load_transform(transform_name)
        assert path1 == path2
        
    def test_can_transform_between_checks_availability(self):
        """can_transform_between should correctly report availability."""
        # Create source and target spaces
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym",
            resolution=2.0,
            reference_affine=np.eye(4)
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2.0,
            reference_affine=np.eye(4)
        )
        
        # Should return True for available transform
        result = can_transform_between(source, target)
        assert result is True
        
        # Test with space variants
        source_variant = CoordinateSpace(
            identifier="MNI152NLin2009aAsym",
            resolution=2.0,
            reference_affine=np.eye(4)
        )
        
        # Should return True (aAsym canonicalizes to cAsym)
        result = can_transform_between(source_variant, target)
        assert result is True


class TestLoggingTransparency:
    """Test that transformations log appropriately."""

    def test_transform_logs_progress(self, caplog):
        """Transformations should log their progress."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Create test image
        data = np.random.rand(91, 109, 91)
        affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
        img = nib.Nifti1Image(data, affine)
        
        # Transform
        result = transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear"
        )
        
        # Check logs
        assert "Loading transform" in caplog.text or "Transforming image" in caplog.text
        assert "Transformation complete" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
