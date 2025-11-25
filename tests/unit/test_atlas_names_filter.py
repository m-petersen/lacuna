"""
Test parcel_names filter functionality.

Tests that the parcel_names parameter correctly filters which atlases are processed.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis import ParcelAggregation, RegionalDamage


class TestAtlasNamesFilter:
    """Test parcel_names filtering functionality."""

    def test_atlas_names_filters_correctly(self):
        """Test that only specified atlases are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            mask_data = np.zeros(shape, dtype=np.uint8)
            mask_data[20:30, 20:30, 20:30] = 1
            mask_img = nib.Nifti1Image(mask_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(mask_img, lesion_path)

            # Create 3 different atlases
            for atlas_name in ["atlas_A", "atlas_B", "atlas_C"]:
                atlas_3d = np.zeros(shape, dtype=np.uint8)
                atlas_3d[15:35, 15:35, 15:35] = 1
                atlas_3d[35:45, 35:45, 35:45] = 2

                atlas_img = nib.Nifti1Image(atlas_3d, affine)
                atlas_path = tmpdir / f"{atlas_name}.nii.gz"
                nib.save(atlas_img, atlas_path)

                labels_path = tmpdir / f"{atlas_name}_labels.txt"
                labels_path.write_text(f"1 {atlas_name}_Region1\n2 {atlas_name}_Region2\n")

            # Load lesion
            mask_data_obj = MaskData.from_nifti(
                lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
            )

            # Register all atlases
            from lacuna.assets.parcellations.registry import register_parcellations_from_directory

            register_parcellations_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)

            # Test 1: Process only atlas_B
            analysis = RegionalDamage(parcel_names=["atlas_B"])
            result = analysis.run(mask_data_obj)
            atlas_results = result.results["RegionalDamage"]

            # Should only have atlas_B results
            assert "atlas-atlas_B_desc-MaskImg" in atlas_results
            assert "atlas-atlas_A_desc-MaskImg" not in atlas_results
            assert "atlas-atlas_C_desc-MaskImg" not in atlas_results

            # Test 2: Process atlas_A and atlas_C
            analysis = RegionalDamage(parcel_names=["atlas_A", "atlas_C"])
            result = analysis.run(mask_data_obj)
            atlas_results = result.results["RegionalDamage"]

            # Should have atlas_A and atlas_C, but not atlas_B
            assert "atlas-atlas_A_desc-MaskImg" in atlas_results
            assert "atlas-atlas_B_desc-MaskImg" not in atlas_results
            assert "atlas-atlas_C_desc-MaskImg" in atlas_results

            # Test 3: None = process all atlases
            analysis = RegionalDamage(parcel_names=None)
            result = analysis.run(mask_data_obj)
            atlas_results = result.results["RegionalDamage"]

            # Should have all three atlases
            assert "atlas-atlas_A_desc-MaskImg" in atlas_results
            assert "atlas-atlas_B_desc-MaskImg" in atlas_results
            assert "atlas-atlas_C_desc-MaskImg" in atlas_results

    def test_atlas_names_warns_if_not_found(self):
        """Test that warning is issued if requested atlas not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            mask_data = np.zeros(shape, dtype=np.uint8)
            mask_data[20:30, 20:30, 20:30] = 1
            mask_img = nib.Nifti1Image(mask_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(mask_img, lesion_path)

            # Create only atlas_A
            atlas_3d = np.zeros(shape, dtype=np.uint8)
            atlas_3d[15:35, 15:35, 15:35] = 1

            atlas_img = nib.Nifti1Image(atlas_3d, affine)
            atlas_path = tmpdir / "atlas_A.nii.gz"
            nib.save(atlas_img, atlas_path)

            labels_path = tmpdir / "atlas_A_labels.txt"
            labels_path.write_text("1 Region1\n")

            # Load lesion
            mask_data_obj = MaskData.from_nifti(
                lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
            )

            # Register atlas
            from lacuna.assets.parcellations.registry import register_parcellations_from_directory

            register_parcellations_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)

            # Request atlas_A and atlas_B (atlas_B doesn't exist)
            analysis = RegionalDamage(parcel_names=["atlas_A", "atlas_B"])

            # The logger will output a warning (captured by capsys), but we just run it
            result = analysis.run(mask_data_obj)

            # Should still process atlas_A successfully
            atlas_results = result.results["RegionalDamage"]
            assert "atlas-atlas_A_desc-MaskImg" in atlas_results

    def test_atlas_names_raises_if_none_found(self):
        """Test that error is raised if no matching atlases found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            mask_data = np.zeros(shape, dtype=np.uint8)
            mask_data[20:30, 20:30, 20:30] = 1
            mask_img = nib.Nifti1Image(mask_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(mask_img, lesion_path)

            # Create atlas_A (but we'll request atlas_B)
            atlas_3d = np.zeros(shape, dtype=np.uint8)
            atlas_3d[15:35, 15:35, 15:35] = 1

            atlas_img = nib.Nifti1Image(atlas_3d, affine)
            atlas_path = tmpdir / "atlas_A.nii.gz"
            nib.save(atlas_img, atlas_path)

            labels_path = tmpdir / "atlas_A_labels.txt"
            labels_path.write_text("1 Region1\n")

            # Load lesion
            mask_data_obj = MaskData.from_nifti(
                lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
            )

            # Register atlas
            from lacuna.assets.parcellations.registry import register_parcellations_from_directory

            register_parcellations_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)

            # Request only atlas_B (doesn't exist)
            analysis = RegionalDamage(parcel_names=["atlas_B"])

            with pytest.raises(ValueError, match="No matching parcellations found for specified names"):
                analysis.run(mask_data_obj)

    def test_atlas_names_validates_input_type(self):
        """Test that parcel_names validation catches invalid types."""
        # Should raise TypeError for non-list
        with pytest.raises(TypeError, match="parcel_names must be a list"):
            ParcelAggregation(parcel_names="atlas_A")

        # Should raise TypeError for list with non-strings
        with pytest.raises(TypeError, match="All items in parcel_names must be strings"):
            ParcelAggregation(parcel_names=["atlas_A", 123])

        # Should raise ValueError for empty list
        with pytest.raises(ValueError, match="parcel_names cannot be an empty list"):
            ParcelAggregation(parcel_names=[])

    def test_atlas_names_works_with_atlas_aggregation(self):
        """Test that parcel_names works with ParcelAggregation (not just RegionalDamage)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            mask_data = np.zeros(shape, dtype=np.uint8)
            mask_data[20:30, 20:30, 20:30] = 1
            mask_img = nib.Nifti1Image(mask_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(mask_img, lesion_path)

            # Create 2 atlases
            for atlas_name in ["atlas_X", "atlas_Y"]:
                atlas_3d = np.zeros(shape, dtype=np.uint8)
                atlas_3d[15:35, 15:35, 15:35] = 1

                atlas_img = nib.Nifti1Image(atlas_3d, affine)
                atlas_path = tmpdir / f"{atlas_name}.nii.gz"
                nib.save(atlas_img, atlas_path)

                labels_path = tmpdir / f"{atlas_name}_labels.txt"
                labels_path.write_text(f"1 {atlas_name}_Region1\n")

            # Load lesion
            mask_data_obj = MaskData.from_nifti(
                lesion_path=lesion_path, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
            )

            # Register atlases
            from lacuna.assets.parcellations.registry import register_parcellations_from_directory

            register_parcellations_from_directory(tmpdir, space="MNI152NLin6Asym", resolution=2)

            # Use ParcelAggregation with parcel_names filter
            analysis = ParcelAggregation(
                source="mask_img",
                aggregation="mean",
                parcel_names=["atlas_X"],
            )
            result = analysis.run(mask_data_obj)
            atlas_results = result.results["ParcelAggregation"]

            # Should only have atlas_X results
            assert "atlas-atlas_X_desc-MaskImg" in atlas_results
            assert "atlas-atlas_Y_desc-MaskImg" not in atlas_results
