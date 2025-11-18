"""
Test that atlas labels are correctly assigned to regions.

This test specifically checks for the off-by-one error in 4D atlas label assignment
where volume indices don't match label IDs.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from lacuna import LesionData
from lacuna.analysis import AtlasAggregation


class TestAtlasLabelAssignment:
    """Test correct label assignment for 3D and 4D atlases."""

    def test_4d_atlas_label_assignment_matches_volume_order(self):
        """
        Test that 4D atlas labels are correctly assigned to volume indices.

        This is a critical test to catch off-by-one errors where the volume
        index in the 4D atlas doesn't match the region ID in the labels file.

        The test creates:
        - A lesion that's entirely in the RIGHT hemisphere (x > 45)
        - A 4D atlas with 3 regions:
            - Volume 0: RIGHT hemisphere region (x > 45)
            - Volume 1: LEFT hemisphere region (x < 45)
            - Volume 2: MIDDLE region (40 < x < 50)
        - Labels file with IDs matching volumes:
            - ID 0 → "Region_Right"
            - ID 1 → "Region_Left"
            - ID 2 → "Region_Middle"

        Expected behavior:
        - Region_Right should have HIGH damage (lesion is in right)
        - Region_Left should have ZERO damage (lesion is not in left)
        - Region_Middle should have MEDIUM damage (lesion overlaps middle)

        Bug symptom (off-by-one error):
        - Volume 0 incorrectly gets label for ID 1 (Region_Left)
        - Volume 1 incorrectly gets label for ID 2 (Region_Middle)
        - Volume 2 incorrectly gets label for ID 0 (Region_Right)
        - Result: Region_Left shows high damage, Region_Right shows zero damage!
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create spatial reference (91x109x91 like MNI152 2mm)
            shape = (91, 109, 91)
            affine = np.eye(4)
            affine[:3, :3] *= 2.0  # 2mm resolution

            # Create lesion in RIGHT hemisphere (x > 45)
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[45:75, 40:60, 40:50] = 1  # Right hemisphere lesion starting at midline
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

            # Create 4D atlas with 3 regions (volumes)
            atlas_4d = np.zeros((*shape, 3), dtype=np.uint8)

            # Volume 0: RIGHT hemisphere region (x > 45) - should have HIGH overlap
            atlas_4d[55:80, 35:65, 35:55, 0] = 1

            # Volume 1: LEFT hemisphere region (x < 45) - should have ZERO overlap
            atlas_4d[10:35, 35:65, 35:55, 1] = 1

            # Volume 2: MIDDLE region (40 < x < 50) - should have MEDIUM overlap
            atlas_4d[40:50, 35:65, 35:55, 2] = 1

            atlas_img = nib.Nifti1Image(atlas_4d, affine)
            atlas_path = tmpdir / "test_4d_atlas.nii.gz"
            nib.save(atlas_img, atlas_path)

            # Create labels file with IDs matching volume indices (0, 1, 2)
            labels_path = tmpdir / "test_4d_atlas_labels.txt"
            labels_path.write_text("0 Region_Right\n1 Region_Left\n2 Region_Middle\n")

            # Load lesion data
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

            # Register atlas
            from lacuna.assets.atlases.registry import register_atlases_from_directory
            register_atlases_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)

            # Run analysis - use only this test's atlas
            analysis = AtlasAggregation(
                source="lesion_img",
                aggregation="percent",
                threshold=0.5,
                atlas_names=["test_4d_atlas"],  # Explicitly use only this test's atlas
            )
            result = analysis.run(lesion_data_obj)
            results_list = result.results["AtlasAggregation"]
            results = results_list[0].get_data()

            right_damage = results.get("test_4d_atlas_Region_Right", None)
            right_damage = results.get("test_4d_atlas_Region_Right", None)
            left_damage = results.get("test_4d_atlas_Region_Left", None)
            middle_damage = results.get("test_4d_atlas_Region_Middle", None)

            # Debug output
            print("\n=== Label Assignment Test Results ===")
            print(f"Region_Right damage: {right_damage:.2f}%")
            print(f"Region_Left damage: {left_damage:.2f}%")
            print(f"Region_Middle damage: {middle_damage:.2f}%")
            print("=====================================\n")

            # Assertions
            assert right_damage is not None, "Region_Right not found in results"
            assert left_damage is not None, "Region_Left not found in results"
            assert middle_damage is not None, "Region_Middle not found in results"

            # CRITICAL: Right hemisphere region should have HIGH damage
            # (lesion is entirely in right hemisphere, x > 45)
            assert right_damage > 10.0, (
                f"Region_Right should have HIGH damage (>10%), but got {right_damage:.2f}%. "
                "This indicates labels are incorrectly assigned to volumes!"
            )

            # LEFT hemisphere region should have ZERO or very low damage
            assert left_damage < 5.0, (
                f"Region_Left should have ZERO/LOW damage (<5%), but got {left_damage:.2f}%. "
                "This indicates an off-by-one error in label assignment!"
            )

            # Middle region should have some damage (partial overlap)
            assert 5.0 <= middle_damage <= 30.0, (
                f"Region_Middle should have MEDIUM damage (5-30%), but got {middle_damage:.2f}%"
            )

    def test_4d_atlas_with_nonzero_starting_id(self):
        """
        Test 4D atlas where label IDs start from 1 (not 0).

        Many atlases use 1-indexed labels (ID 1, 2, 3...) while volumes
        are always 0-indexed (volume 0, 1, 2...). This tests that the
        mapping is correct.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create simple setup
            shape = (50, 50, 50)
            affine = np.eye(4)

            # Lesion in bottom half (z < 25)
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[:, :, 0:25] = 1
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

            # 4D atlas with 2 regions
            atlas_4d = np.zeros((*shape, 2), dtype=np.uint8)
            atlas_4d[:, :, 0:25, 0] = 1  # Volume 0: Bottom half
            atlas_4d[:, :, 25:50, 1] = 1  # Volume 1: Top half

            atlas_img = nib.Nifti1Image(atlas_4d, affine)
            atlas_path = tmpdir / "atlas_1indexed.nii.gz"
            nib.save(atlas_img, atlas_path)

            # Labels starting from 1 (common convention)
            labels_path = tmpdir / "atlas_1indexed_labels.txt"
            labels_path.write_text("1 Bottom_Region\n2 Top_Region\n")

            # Load and analyze
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
            
            # Register atlas
            from lacuna.assets.atlases.registry import register_atlases_from_directory
            register_atlases_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)
            
            # Run analysis - use only this test's atlas
            analysis = AtlasAggregation(
                source="lesion_img",
                aggregation="percent",
                threshold=0.5,
                atlas_names=["atlas_1indexed"],  # Explicitly use only this test's atlas
            )
            result = analysis.run(lesion_data_obj)
            results_list = result.results["AtlasAggregation"]
            results = results_list[0].get_data()

            bottom_damage = results.get("atlas_1indexed_Bottom_Region", None)
            top_damage = results.get("atlas_1indexed_Top_Region", None)

            print("\n=== 1-Indexed Label Test Results ===")
            print(f"Bottom_Region damage: {bottom_damage:.2f}%")
            print(f"Top_Region damage: {top_damage:.2f}%")
            print("====================================\n")

            # Bottom region (volume 0, ID 1) should have HIGH damage
            assert bottom_damage > 50.0, (
                f"Bottom_Region (volume 0 → ID 1) should have HIGH damage, "
                f"but got {bottom_damage:.2f}%"
            )

            # Top region (volume 1, ID 2) should have ZERO damage
            assert top_damage < 5.0, (
                f"Top_Region (volume 1 → ID 2) should have ZERO damage, but got {top_damage:.2f}%"
            )

    def test_3d_atlas_label_assignment_unchanged(self):
        """
        Test that 3D atlas label assignment still works correctly.

        3D atlases use integer labels directly in the volume, so the
        label ID matches the voxel value. This should not be affected
        by 4D atlas fixes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create setup
            shape = (50, 50, 50)
            affine = np.eye(4)

            # Lesion in region 2
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[20:30, 20:30, 20:30] = 1
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

            # 3D atlas with labeled regions
            atlas_3d = np.zeros(shape, dtype=np.uint8)
            atlas_3d[0:20, 0:20, 0:20] = 1  # Region 1
            atlas_3d[15:40, 15:40, 15:40] = 2  # Region 2 (overlaps lesion more)
            atlas_3d[40:50, 40:50, 40:50] = 3  # Region 3

            atlas_img = nib.Nifti1Image(atlas_3d, affine)
            atlas_path = tmpdir / "atlas_3d.nii.gz"
            nib.save(atlas_img, atlas_path)

            # Labels matching voxel values
            labels_path = tmpdir / "atlas_3d_labels.txt"
            labels_path.write_text("1 First_Region\n2 Second_Region\n3 Third_Region\n")

            # Load and analyze
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
            
            # Register atlas
            from lacuna.assets.atlases.registry import register_atlases_from_directory
            register_atlases_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)
            
            # Use only this test's atlas
            analysis = AtlasAggregation(
                source="lesion_img",
                aggregation="percent",
                threshold=0.5,
                atlas_names=["atlas_3d"],  # Explicitly use only this test's atlas
            )
            result = analysis.run(lesion_data_obj)
            results_list = result.results["AtlasAggregation"]
            results = results_list[0].get_data()

            first_damage = results.get("atlas_3d_First_Region", None)
            second_damage = results.get("atlas_3d_Second_Region", None)
            third_damage = results.get("atlas_3d_Third_Region", None)

            # Region 2 should have high damage, others should have zero
            assert first_damage < 5.0, "First_Region should have no damage"
            assert second_damage > 5.0, "Second_Region should have damage"
            assert third_damage < 5.0, "Third_Region should have no damage"
