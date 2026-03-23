"""Integration tests for spatial transformations.

Tests the complete transformation pipeline including:
- Transform loading and caching
- TemplateFlow integrity checking
- 3D/4D image handling
- Asyncio compatibility in Jupyter environments
- TemplateFlow space variant canonicalization (bAsym → cAsym for file lookup)
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
from lacuna.spatial.transform import (
    TransformationStrategy,
    _canonicalize_space_variant,
    can_transform_between,
    transform_image,
)


class TestTransformLoading:
    """Test transform loading and TemplateFlow integration."""

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_load_transform_from_templateflow(self):
        """Transform should be downloaded from TemplateFlow if not cached."""
        from lacuna.assets.transforms.loader import load_transform

        # This should trigger TemplateFlow download
        transform_name = "MNI152NLin6Asym_to_MNI152NLin2009cAsym"
        path = load_transform(transform_name)

        assert path.exists()
        assert path.suffix == ".h5"
        assert path.stat().st_size > 1024  # At least 1KB

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_corrupted_file_detection_and_retry(self, tmp_path, monkeypatch):
        """Corrupted transform files should be detected and re-downloaded."""

        # Create a fake corrupted file
        cache_dir = tmp_path / "templateflow" / "tpl-MNI152NLin6Asym"
        cache_dir.mkdir(parents=True)
        corrupted_file = (
            cache_dir / "tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_mode-image_xfm.h5"
        )
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

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_3d_image(self):
        """3D images should transform successfully."""
        # Create 3D test image
        data = np.random.rand(91, 109, 91)
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(data, affine)

        # Transform should handle 3D image
        result = transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear",
        )

        assert result.ndim == 3
        assert result.shape[0] > 0

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_4d_image_with_singleton_dimension(self):
        """4D images with singleton 4th dimension should be squeezed."""
        # Create 4D test image with singleton dimension
        data = np.random.rand(91, 109, 91, 1)
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(data, affine)

        # Should squeeze to 3D and transform
        result = transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear",
        )

        assert result.ndim == 3

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_4d_image_multiple_volumes(self):
        """4D images with multiple volumes should be transformed volume by volume."""
        # Create 4D test image with multiple volumes
        data = np.random.rand(91, 109, 91, 5)
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(data, affine)

        # Should transform each volume independently and return 4D result
        result = transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear",
        )

        # Result should be 4D with same number of volumes
        assert result.ndim == 4
        assert result.shape[-1] == 5


class TestSpaceVariantCanonicalization:
    """Test TemplateFlow canonicalization function.

    TemplateFlow canonicalization maps 2009 variants → 2009c for file lookup.
    The core spaces module no longer has canonicalize_space_variant.
    """

    # --- TemplateFlow canonicalization (transform.py) ---

    def test_templateflow_bAsym_to_cAsym(self):
        """TemplateFlow: bAsym → cAsym."""
        assert _canonicalize_space_variant("MNI152NLin2009bAsym") == "MNI152NLin2009cAsym"

    def test_templateflow_cAsym_unchanged(self):
        """TemplateFlow: cAsym stays cAsym."""
        assert _canonicalize_space_variant("MNI152NLin2009cAsym") == "MNI152NLin2009cAsym"

    def test_templateflow_NLin6_unchanged(self):
        """TemplateFlow: NLin6 is unchanged."""
        assert _canonicalize_space_variant("MNI152NLin6Asym") == "MNI152NLin6Asym"


class TestDetermineDirection:
    """Test TransformationStrategy.determine_direction() for all direction types."""

    def _make_space(self, identifier, resolution=2):
        affine = REFERENCE_AFFINES.get((identifier, resolution), np.eye(4))
        return CoordinateSpace(
            identifier=identifier, resolution=resolution, reference_affine=affine
        )

    def test_none_same_space_same_resolution(self):
        source = self._make_space("MNI152NLin6Asym", 2)
        target = self._make_space("MNI152NLin6Asym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "none"

    def test_resample_same_space_different_resolution(self):
        source = self._make_space("MNI152NLin2009cAsym", 1)
        target = self._make_space("MNI152NLin2009cAsym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "resample"

    def test_regrid_bAsym_to_cAsym(self):
        """2009b → 2009c requires regrid (different voxel grids, same world coords)."""
        source = self._make_space("MNI152NLin2009bAsym", 2)
        target = self._make_space("MNI152NLin2009cAsym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "regrid"

    def test_regrid_cAsym_to_bAsym(self):
        """2009c → 2009b requires regrid."""
        source = self._make_space("MNI152NLin2009cAsym", 2)
        target = self._make_space("MNI152NLin2009bAsym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "regrid"

    def test_forward_NLin6_to_2009c(self):
        source = self._make_space("MNI152NLin6Asym", 2)
        target = self._make_space("MNI152NLin2009cAsym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "forward"

    def test_reverse_2009c_to_NLin6(self):
        source = self._make_space("MNI152NLin2009cAsym", 2)
        target = self._make_space("MNI152NLin6Asym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "reverse"

    def test_chain_forward_NLin6_to_bAsym(self):
        """NLin6 → 2009b chains: NLin6→2009c (warp) → 2009b (regrid)."""
        source = self._make_space("MNI152NLin6Asym", 2)
        target = self._make_space("MNI152NLin2009bAsym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "chain_forward"

    def test_chain_reverse_bAsym_to_NLin6(self):
        """2009b → NLin6 chains: 2009b→2009c (regrid) → NLin6 (warp)."""
        source = self._make_space("MNI152NLin2009bAsym", 2)
        target = self._make_space("MNI152NLin6Asym", 2)
        assert TransformationStrategy().determine_direction(source, target) == "chain_reverse"


class TestRegridOperation:
    """Test regrid (affine-aware resampling between 2009b and 2009c)."""

    def test_regrid_preserves_world_coordinates(self):
        """A point at a known MNI coordinate should map to the same mm location after regrid."""
        from lacuna.core.spaces import REFERENCE_SHAPES

        # Create a test image in 2009cAsym with a single bright voxel
        affine_c = REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)]
        shape_c = REFERENCE_SHAPES[("MNI152NLin2009cAsym", 2)]
        data_c = np.zeros(shape_c, dtype=np.float32)

        # Place a point at voxel center — compute its world coordinate
        vox_ijk = np.array([48, 57, 48])  # Near center of 2009c 2mm grid
        world_xyz = affine_c[:3, :3] @ vox_ijk + affine_c[:3, 3]

        # Set a small sphere around that voxel
        data_c[vox_ijk[0], vox_ijk[1], vox_ijk[2]] = 1.0
        img_c = nib.Nifti1Image(data_c, affine_c)

        # Regrid to 2009bAsym
        target = CoordinateSpace(
            identifier="MNI152NLin2009bAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009bAsym", 2)],
        )
        strategy = TransformationStrategy()
        result = strategy.apply_regrid(img_c, target, interpolation=None)

        # Find the peak voxel in the result
        result_data = result.get_fdata()
        peak_vox = np.unravel_index(np.argmax(result_data), result_data.shape)
        result_world = result.affine[:3, :3] @ np.array(peak_vox) + result.affine[:3, 3]

        # World coordinates should be within one voxel (2mm) of the original
        assert np.allclose(
            world_xyz, result_world, atol=2.0
        ), f"World coordinate mismatch: original {world_xyz}, regridded {result_world}"

    def test_regrid_output_shape_matches_reference(self):
        """Regridded image should have the target space's reference shape."""
        from lacuna.core.spaces import REFERENCE_SHAPES

        affine_b = REFERENCE_AFFINES[("MNI152NLin2009bAsym", 2)]
        shape_b = REFERENCE_SHAPES[("MNI152NLin2009bAsym", 2)]
        data = np.ones(shape_b, dtype=np.float32)
        img_b = nib.Nifti1Image(data, affine_b)

        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym",
            resolution=2,
            reference_affine=REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)],
        )
        result = TransformationStrategy().apply_regrid(img_b, target)

        expected_shape = REFERENCE_SHAPES[("MNI152NLin2009cAsym", 2)]
        assert (
            result.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {result.shape}"


class TestAtlasTransformation:
    """Test atlas transformation in analysis pipeline."""

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_atlas_transformation_preserves_labels(self):
        """Atlas transformation should preserve integer labels."""
        # Create atlas with integer labels
        labels = np.random.randint(0, 10, size=(91, 109, 91))
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        atlas_img = nib.Nifti1Image(labels.astype(np.int16), affine)

        # Transform with nearest neighbor
        result = transform_image(
            img=atlas_img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="nearest",
        )

        result_data = result.get_fdata()
        np.unique(result_data)

        # All values should be integers (or very close due to float conversion)
        assert np.allclose(result_data, np.round(result_data))

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_regional_damage_with_space_mismatch(self):
        """RegionalDamage should handle lesion/atlas in different spaces."""
        from lacuna import SubjectData
        from lacuna.analysis import RegionalDamage

        # Create lesion in NLin6Asym space
        mask_data = np.random.rand(182, 218, 182) > 0.9
        affine = np.eye(4)
        affine[:3, :3] = np.diag([-1, 1, 1])
        mask_img = nib.Nifti1Image(mask_data.astype(np.uint8), affine)

        lesion = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 1.0}
        )

        # Run analysis - should automatically transform atlas to match
        # Note: This is an integration test that requires actual atlases
        analysis = RegionalDamage()

        # Should not raise error even with space mismatch
        result = analysis.run(lesion)
        assert "RegionalDamage" in result.results


class TestAsyncioCompatibility:
    """Test asyncio event loop compatibility for Jupyter."""

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_with_existing_event_loop(self):
        """Transformation should work even with existing asyncio event loop."""
        import asyncio

        # Create event loop (simulating Jupyter)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Create test image
            data = np.random.rand(91, 109, 91)
            affine = np.array(
                [
                    [-2.0, 0.0, 0.0, 90.0],
                    [0.0, 2.0, 0.0, -126.0],
                    [0.0, 0.0, 2.0, -72.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            img = nib.Nifti1Image(data, affine)

            # Should not raise RuntimeError about event loop
            result = transform_image(
                img=img,
                source_space="MNI152NLin6Asym",
                target_space="MNI152NLin2009cAsym",
                source_resolution=2.0,
                interpolation="linear",
            )

            assert result is not None

        finally:
            loop.close()


class TestTransformCaching:
    """Test transform caching behavior."""

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
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

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_can_transform_between_checks_availability(self):
        """can_transform_between should correctly report availability."""
        # Create source and target spaces
        source = CoordinateSpace(
            identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=np.eye(4)
        )
        target = CoordinateSpace(
            identifier="MNI152NLin2009cAsym", resolution=2.0, reference_affine=np.eye(4)
        )

        # Should return True for available transform
        result = can_transform_between(source, target)
        assert result is True


class TestLoggingTransparency:
    """Test that transformations log appropriately when verbose=True."""

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_logs_progress(self, caplog):
        """Transformations should log their progress when verbose=True."""
        import logging

        caplog.set_level(logging.INFO)

        # Create test image
        data = np.random.rand(91, 109, 91)
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(data, affine)

        # Transform with verbose=True to enable logging
        transform_image(
            img=img,
            source_space="MNI152NLin6Asym",
            target_space="MNI152NLin2009cAsym",
            source_resolution=2.0,
            interpolation="linear",
            verbose=True,  # Enable logging
        )

        # Check logs - should have transformation info
        assert "Warping" in caplog.text or "Transforming" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
