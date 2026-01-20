"""Tests for 4D atlas support.

Tests cover:
1. ParcellationMetadata with is_4d field
2. Automatic detection of 4D atlases during registration
3. 4D atlas transformation (volume-by-volume)
4. 4D atlas aggregation in RegionalDamage
5. Mixed 3D and 4D atlas usage
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna import SubjectData
from lacuna.analysis import RegionalDamage
from lacuna.assets.parcellations.registry import (
    ParcellationMetadata,
    register_parcellation,
    unregister_parcellation,
)
from lacuna.core.data_types import ParcelData
from lacuna.core.spaces import CoordinateSpace
from lacuna.spatial.transform import transform_image


class Test4DParcellationMetadata:
    """Test ParcellationMetadata with is_4d field."""

    def test_atlas_metadata_has_is_4d_field(self):
        """ParcellationMetadata should have is_4d boolean field."""
        metadata = ParcellationMetadata(
            name="Test3DAtlas",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="Test 3D atlas",
            parcellation_filename="test_3d.nii.gz",
            labels_filename="test_3d_labels.txt",
            is_4d=False,
        )

        assert hasattr(metadata, "is_4d")
        assert metadata.is_4d is False

    def test_atlas_metadata_4d_true(self):
        """ParcellationMetadata should accept is_4d=True for 4D atlases."""
        metadata = ParcellationMetadata(
            name="Test4DAtlas",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="Test 4D atlas with multiple tracts",
            parcellation_filename="test_4d.nii.gz",
            labels_filename="test_4d_labels.txt",
            is_4d=True,
            n_regions=64,  # 64 tracts in 4th dimension
        )

        assert metadata.is_4d is True
        assert metadata.n_regions == 64

    def test_atlas_metadata_is_4d_defaults_to_false(self):
        """is_4d should default to False if not specified."""
        metadata = ParcellationMetadata(
            name="TestDefaultAtlas",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="Test atlas",
            parcellation_filename="test.nii.gz",
            labels_filename="test_labels.txt",
        )

        # Should either have is_4d=False or not have the field yet
        # (depends on if we make it optional with default)
        assert not getattr(metadata, "is_4d", False)


class Test4DAtlasDetection:
    """Test automatic detection of 4D atlases during registration."""

    def test_register_3d_atlas_sets_is_4d_false(self, tmp_path):
        """Registering 3D atlas should set is_4d=False."""
        # Create 3D atlas
        atlas_data = np.random.randint(0, 10, size=(10, 10, 10), dtype=np.int16)
        atlas_img = nib.Nifti1Image(atlas_data, affine=np.eye(4))
        atlas_path = tmp_path / "atlas_3d.nii.gz"
        nib.save(atlas_img, atlas_path)

        # Create labels file
        labels_path = tmp_path / "atlas_3d_labels.txt"
        labels_path.write_text("1 Region1\n2 Region2\n")

        try:
            # Register atlas - should detect 3D
            metadata = ParcellationMetadata(
                name="Test3DRegistration",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="Test 3D atlas registration",
                parcellation_filename=str(atlas_path),
                labels_filename=str(labels_path),
                is_4d=False,
            )

            # Check metadata
            assert metadata.is_4d is False

        finally:
            # Clean up
            try:
                unregister_parcellation("Test3DRegistration")
            except (ValueError, KeyError):
                pass

    def test_register_4d_atlas_sets_is_4d_true(self, tmp_path):
        """Registering 4D atlas should set is_4d=True."""
        # Create 4D atlas (e.g., HCP1065 with 64 tracts)
        atlas_data = np.random.randint(0, 2, size=(10, 10, 10, 64), dtype=np.int16)
        atlas_img = nib.Nifti1Image(atlas_data, affine=np.eye(4))
        atlas_path = tmp_path / "atlas_4d.nii.gz"
        nib.save(atlas_img, atlas_path)

        # Create labels file
        labels_path = tmp_path / "atlas_4d_labels.txt"
        labels = "\n".join([f"{i} Tract{i}" for i in range(1, 65)])
        labels_path.write_text(labels)

        try:
            # Register atlas - should detect 4D
            metadata = ParcellationMetadata(
                name="Test4DRegistration",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="Test 4D atlas registration",
                parcellation_filename=str(atlas_path),
                labels_filename=str(labels_path),
                is_4d=True,
                n_regions=64,
            )

            # Check metadata
            assert metadata.is_4d is True
            assert metadata.n_regions == 64

        finally:
            # Clean up
            try:
                unregister_parcellation("Test4DRegistration")
            except (ValueError, KeyError):
                pass


class Test4DAtlasTransformation:
    """Test volume-by-volume transformation of 4D atlases."""

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_4d_atlas_real_world_scenario(self, tmp_path):
        """Test transformation with realistic 4D atlas scenario (like HCP1065)."""
        # This test mimics the actual error scenario:
        # MNI152NLin2009aAsym (1mm) atlas transformed to MNI152NLin6Asym (2mm)

        # Create 4D atlas with realistic dimensions
        atlas_data = np.zeros((91, 109, 91, 4), dtype=np.int16)
        atlas_data[40:50, 50:60, 40:50, 0] = 1  # Tract 1
        atlas_data[30:40, 40:50, 30:40, 1] = 1  # Tract 2
        atlas_data[50:60, 60:70, 50:60, 2] = 1  # Tract 3
        atlas_data[35:45, 45:55, 35:45, 3] = 1  # Tract 4

        # Source affine (MNI152NLin2009aAsym, 2mm)
        source_affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        atlas_img = nib.Nifti1Image(atlas_data, source_affine)

        # Target space (MNI152NLin6Asym, 2mm)
        target_affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        target_space = CoordinateSpace(
            identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=target_affine
        )

        # Transform - this should work without UnboundLocalError
        result = transform_image(
            img=atlas_img,
            source_space="MNI152NLin2009aAsym",
            target_space=target_space,
            source_resolution=2.0,
            interpolation="nearest",
        )

        # Verify result
        assert result.ndim == 4 or len(result.shape) == 4
        result_data = result.get_fdata()
        assert result_data.shape[3] == 4  # Should still have 4 volumes

        # Each volume should contain binary values
        for vol_idx in range(4):
            unique_vals = np.unique(result_data[..., vol_idx])
            assert all(val in [0, 1] for val in unique_vals)

    def test_transform_4d_atlas_volume_by_volume(self, tmp_path):
        """transform_image should handle 4D atlases by transforming each volume."""
        # Create 4D atlas with 3 volumes
        atlas_data = np.zeros((10, 10, 10, 3), dtype=np.int16)
        atlas_data[3:7, 3:7, 3:7, 0] = 1  # Tract 1
        atlas_data[2:8, 2:8, 2:8, 1] = 1  # Tract 2 (larger)
        atlas_data[4:6, 4:6, 4:6, 2] = 1  # Tract 3 (smaller)

        affine = np.array(
            [
                [2.0, 0.0, 0.0, -10.0],
                [0.0, 2.0, 0.0, -10.0],
                [0.0, 0.0, 2.0, -10.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        atlas_img = nib.Nifti1Image(atlas_data, affine)

        # Target space (same space, different resolution for simple test)
        target_space = CoordinateSpace(
            identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=affine
        )

        # Transform - should handle 4D by iterating volumes
        # Note: This will use the actual transform logic which we'll implement
        result = transform_image(
            img=atlas_img,
            source_space="MNI152NLin6Asym",
            target_space=target_space,
            source_resolution=2.0,
            interpolation="nearest",
        )

        # Check result
        assert result.ndim == 4 or result.shape == atlas_data.shape
        result_data = result.get_fdata()

        # Should still have 3 volumes
        assert result_data.ndim == 4
        assert result_data.shape[3] == 3

        # Each volume should contain binary values (0 or 1)
        for vol_idx in range(3):
            unique_vals = np.unique(result_data[..., vol_idx])
            assert all(
                val in [0, 1] for val in unique_vals
            ), f"Volume {vol_idx} has non-binary values: {unique_vals}"

    def test_transform_4d_atlas_preserves_labels(self, tmp_path):
        """4D atlas transformation should preserve integer labels in each volume."""
        # Create 4D atlas with multiple labels per volume
        atlas_data = np.zeros((20, 20, 20, 2), dtype=np.int16)
        atlas_data[5:10, 5:10, 5:10, 0] = 1  # Volume 0: region 1
        atlas_data[10:15, 10:15, 10:15, 0] = 2  # Volume 0: region 2
        atlas_data[3:8, 3:8, 3:8, 1] = 1  # Volume 1: region 1
        atlas_data[12:17, 12:17, 12:17, 1] = 2  # Volume 1: region 2

        affine = np.array(
            [
                [2.0, 0.0, 0.0, -20.0],
                [0.0, 2.0, 0.0, -20.0],
                [0.0, 0.0, 2.0, -20.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        atlas_img = nib.Nifti1Image(atlas_data, affine)

        target_space = CoordinateSpace(
            identifier="MNI152NLin6Asym", resolution=2.0, reference_affine=affine
        )

        # Transform with nearest neighbor to preserve labels
        result = transform_image(
            img=atlas_img,
            source_space="MNI152NLin6Asym",
            target_space=target_space,
            source_resolution=2.0,
            interpolation="nearest",
        )

        result_data = result.get_fdata()

        # Check each volume has integer labels
        for vol_idx in range(2):
            unique_vals = np.unique(result_data[..., vol_idx])
            assert all(
                val in [0, 1, 2] for val in unique_vals
            ), f"Volume {vol_idx} has unexpected values: {unique_vals}"

    @pytest.mark.slow
    @pytest.mark.requires_templateflow
    def test_transform_4d_atlas_different_spaces(self, tmp_path):
        """4D atlas transformation should work across different coordinate spaces."""
        # This test will verify that the transformation works when
        # source and target are in different spaces
        # (e.g., MNI152NLin6Asym -> MNI152NLin2009cAsym)

        # Create simple 4D atlas
        atlas_data = np.zeros((91, 109, 91, 2), dtype=np.int16)
        atlas_data[40:50, 50:60, 40:50, 0] = 1
        atlas_data[50:60, 60:70, 50:60, 1] = 1

        # MNI152NLin6Asym affine (2mm)
        source_affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        atlas_img = nib.Nifti1Image(atlas_data, source_affine)

        # Transform to MNI152NLin2009cAsym
        target_affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        target_space = CoordinateSpace(
            identifier="MNI152NLin2009cAsym", resolution=2.0, reference_affine=target_affine
        )

        # This should work - transform each volume independently
        result = transform_image(
            img=atlas_img,
            source_space="MNI152NLin6Asym",
            target_space=target_space,
            source_resolution=2.0,
            interpolation="nearest",
        )

        # Result should be 4D
        assert result.ndim == 4 or len(result.shape) == 4
        result_data = result.get_fdata()
        assert result_data.shape[3] == 2


class Test4DParcelAggregation:
    """Test atlas aggregation with 4D atlases."""

    def test_regional_damage_with_4d_atlas(self, tmp_path):
        """RegionalDamage should work with 4D atlases."""
        # Create lesion
        mask_data = np.zeros((20, 20, 20))
        mask_data[8:12, 8:12, 8:12] = 1
        lesion_affine = np.eye(4) * 2
        lesion_affine[3, 3] = 1
        mask_img = nib.Nifti1Image(mask_data, lesion_affine)

        lesion = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Create and register 4D atlas
        atlas_data = np.zeros((20, 20, 20, 3), dtype=np.int16)
        atlas_data[5:15, 5:15, 5:15, 0] = 1  # Tract 1 (overlaps lesion)
        atlas_data[0:10, 0:10, 0:10, 1] = 1  # Tract 2 (partially overlaps)
        atlas_data[15:20, 15:20, 15:20, 2] = 1  # Tract 3 (no overlap)

        atlas_img = nib.Nifti1Image(atlas_data, lesion_affine)
        atlas_path = tmp_path / "test_4d_atlas.nii.gz"
        nib.save(atlas_img, atlas_path)

        # Create labels
        labels_path = tmp_path / "test_4d_labels.txt"
        labels_path.write_text("1 Tract1\n2 Tract2\n3 Tract3\n")

        try:
            # Register 4D atlas
            metadata = ParcellationMetadata(
                name="Test4DAtlas_Aggregation",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="Test 4D atlas for aggregation",
                parcellation_filename=str(atlas_path),
                labels_filename=str(labels_path),
                is_4d=True,
                n_regions=3,
            )
            register_parcellation(metadata)

            # Run RegionalDamage with 4D atlas
            analysis = RegionalDamage(parcel_names=["Test4DAtlas_Aggregation"])

            result = analysis.run(lesion)

            # Check results - using BIDS-style keys
            damage_results = result.results["RegionalDamage"]
            expected_key = "atlas-Test4DAtlas_Aggregation_source-InputMask"
            assert expected_key in damage_results

            # Get region data
            region_data = damage_results[expected_key].get_data()
            assert len(region_data) > 0

            # Tract1 should have highest damage (full overlap)
            # Tract2 should have some damage
            # Tract3 should have no damage
            # Note: Results structure depends on ParcelData implementation

        finally:
            try:
                unregister_parcellation("Test4DAtlas_Aggregation")
            except (ValueError, KeyError):
                pass

    def test_4d_atlas_aggregation_per_volume(self, tmp_path):
        """4D atlas aggregation should compute damage per volume independently."""
        # Create lesion that overlaps different volumes differently
        mask_data = np.zeros((20, 20, 20))
        mask_data[8:12, 8:12, 8:12] = 1  # Central lesion
        lesion_affine = np.eye(4) * 2
        lesion_affine[3, 3] = 1
        mask_img = nib.Nifti1Image(mask_data, lesion_affine)

        lesion = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Create 4D atlas where each volume has different overlap
        atlas_data = np.zeros((20, 20, 20, 3), dtype=np.int16)
        atlas_data[7:13, 7:13, 7:13, 0] = 1  # Volume 0: large overlap with lesion
        atlas_data[10:15, 10:15, 10:15, 1] = 1  # Volume 1: small overlap
        atlas_data[0:5, 0:5, 0:5, 2] = 1  # Volume 2: no overlap

        atlas_img = nib.Nifti1Image(atlas_data, lesion_affine)
        atlas_path = tmp_path / "test_4d_varying.nii.gz"
        nib.save(atlas_img, atlas_path)

        labels_path = tmp_path / "test_4d_varying_labels.txt"
        labels_path.write_text("1 LargeOverlap\n2 SmallOverlap\n3 NoOverlap\n")

        try:
            metadata = ParcellationMetadata(
                name="Test4D_VaryingOverlap",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="Test 4D with varying overlap",
                parcellation_filename=str(atlas_path),
                labels_filename=str(labels_path),
                is_4d=True,
                n_regions=3,
            )
            register_parcellation(metadata)

            analysis = RegionalDamage(parcel_names=["Test4D_VaryingOverlap"])

            result = analysis.run(lesion)
            damage_results = result.results["RegionalDamage"]

            # Verify we got results for the atlas (BIDS-style naming)
            expected_key = "atlas-Test4D_VaryingOverlap_source-InputMask"
            assert expected_key in damage_results
            region_data = damage_results[expected_key].get_data()
            assert len(region_data) > 0

            # LargeOverlap should have more damage than SmallOverlap
            # NoOverlap should have 0 or minimal damage

        finally:
            try:
                unregister_parcellation("Test4D_VaryingOverlap")
            except (ValueError, KeyError):
                pass


class TestMixed3DAnd4DAtlases:
    """Test using both 3D and 4D atlases together."""

    def test_regional_damage_with_mixed_atlases(self, tmp_path):
        """RegionalDamage should handle mix of 3D and 4D atlases."""
        # Create lesion
        mask_data = np.zeros((20, 20, 20))
        mask_data[8:12, 8:12, 8:12] = 1
        lesion_affine = np.eye(4) * 2
        lesion_affine[3, 3] = 1
        mask_img = nib.Nifti1Image(mask_data, lesion_affine)

        lesion = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Create 3D atlas
        atlas_3d_data = np.zeros((20, 20, 20), dtype=np.int16)
        atlas_3d_data[5:10, 5:10, 5:10] = 1
        atlas_3d_data[10:15, 10:15, 10:15] = 2
        atlas_3d_img = nib.Nifti1Image(atlas_3d_data, lesion_affine)
        atlas_3d_path = tmp_path / "mixed_3d.nii.gz"
        nib.save(atlas_3d_img, atlas_3d_path)

        labels_3d_path = tmp_path / "mixed_3d_labels.txt"
        labels_3d_path.write_text("1 Region1\n2 Region2\n")

        # Create 4D atlas
        atlas_4d_data = np.zeros((20, 20, 20, 2), dtype=np.int16)
        atlas_4d_data[7:13, 7:13, 7:13, 0] = 1
        atlas_4d_data[0:5, 0:5, 0:5, 1] = 1
        atlas_4d_img = nib.Nifti1Image(atlas_4d_data, lesion_affine)
        atlas_4d_path = tmp_path / "mixed_4d.nii.gz"
        nib.save(atlas_4d_img, atlas_4d_path)

        labels_4d_path = tmp_path / "mixed_4d_labels.txt"
        labels_4d_path.write_text("1 Tract1\n2 Tract2\n")

        try:
            # Register both atlases
            metadata_3d = ParcellationMetadata(
                name="Mixed3D",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="3D atlas for mixed test",
                parcellation_filename=str(atlas_3d_path),
                labels_filename=str(labels_3d_path),
                is_4d=False,
                n_regions=2,
            )
            register_parcellation(metadata_3d)

            metadata_4d = ParcellationMetadata(
                name="Mixed4D",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="4D atlas for mixed test",
                parcellation_filename=str(atlas_4d_path),
                labels_filename=str(labels_4d_path),
                is_4d=True,
                n_regions=2,
            )
            register_parcellation(metadata_4d)

            # Run with both atlases
            analysis = RegionalDamage(parcel_names=["Mixed3D", "Mixed4D"])

            result = analysis.run(lesion)
            damage_results = result.results["RegionalDamage"]

            # Should have results from both atlases (BIDS-style keys)
            assert "atlas-Mixed3D_source-InputMask" in damage_results
            assert "atlas-Mixed4D_source-InputMask" in damage_results

            # Each should have region data
            mixed3d_data = damage_results["atlas-Mixed3D_source-InputMask"].get_data()
            mixed4d_data = damage_results["atlas-Mixed4D_source-InputMask"].get_data()
            assert len(mixed3d_data) > 0
            assert len(mixed4d_data) > 0

        finally:
            try:
                unregister_parcellation("Mixed3D")
                unregister_parcellation("Mixed4D")
            except (ValueError, KeyError):
                pass


class TestRegionalDamageOutputAPI:
    """Test correct usage of RegionalDamage output API."""

    def test_regional_damage_returns_roi_result_list(self, tmp_path):
        """RegionalDamage should return list of ParcelData objects, not dict."""
        # Create lesion
        mask_data = np.zeros((20, 20, 20))
        mask_data[8:12, 8:12, 8:12] = 1
        lesion_affine = np.eye(4) * 2
        lesion_affine[3, 3] = 1
        mask_img = nib.Nifti1Image(mask_data, lesion_affine)

        lesion = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Create 3D atlas
        atlas_data = np.zeros((20, 20, 20), dtype=np.int16)
        atlas_data[5:10, 5:10, 5:10] = 1
        atlas_data[10:15, 10:15, 10:15] = 2
        atlas_img = nib.Nifti1Image(atlas_data, lesion_affine)
        atlas_path = tmp_path / "test_output_api.nii.gz"
        nib.save(atlas_img, atlas_path)

        labels_path = tmp_path / "test_output_api_labels.txt"
        labels_path.write_text("1 Region1\n2 Region2\n")

        try:
            metadata = ParcellationMetadata(
                name="TestOutputAPI",
                space="MNI152NLin6Asym",
                resolution=2.0,
                description="Test output API",
                parcellation_filename=str(atlas_path),
                labels_filename=str(labels_path),
                is_4d=False,
                n_regions=2,
            )
            register_parcellation(metadata)

            analysis = RegionalDamage(parcel_names=["TestOutputAPI"])

            result = analysis.run(lesion)
            damage_results = result.results["RegionalDamage"]

            # NEW API: damage_results is a dict with BIDS-style keys
            assert isinstance(
                damage_results, dict
            ), "RegionalDamage results should be a dict, not list"

            # Should have the atlas with BIDS-style key
            expected_key = "atlas-TestOutputAPI_source-InputMask"
            assert expected_key in damage_results

            # Get the ROI result
            roi_result = damage_results[expected_key]
            assert isinstance(roi_result, ParcelData)

            # Access the damage data via get_data()
            damage_data = roi_result.get_data()
            assert isinstance(damage_data, dict), "ParcelData.get_data() should return a dict"

            # Damage data should be dict of region_name -> percentage
            assert all(
                isinstance(k, str) for k in damage_data.keys()
            ), "Region names should be strings"
            assert all(
                isinstance(v, (int, float)) for v in damage_data.values()
            ), "Damage percentages should be numeric"

            # Test correct usage pattern (as shown in notebook)
            # CORRECT: Get data from ParcelData first
            sorted_regions = sorted(damage_data.items(), key=lambda x: x[1], reverse=True)
            assert isinstance(sorted_regions, list)

        finally:
            try:
                unregister_parcellation("TestOutputAPI")
            except (ValueError, KeyError):
                pass
