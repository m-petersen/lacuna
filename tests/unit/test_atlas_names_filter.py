"""
Test atlas_names filter functionality.

Tests that the atlas_names parameter correctly filters which atlases are processed.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ldk import LesionData
from ldk.analysis import AtlasAggregation, RegionalDamage


class TestAtlasNamesFilter:
    """Test atlas_names filtering functionality."""

    def test_atlas_names_filters_correctly(self):
        """Test that only specified atlases are processed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[20:30, 20:30, 20:30] = 1
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

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
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152_2mm"})

            # Test 1: Process only atlas_B
            analysis = RegionalDamage(atlas_dir=str(tmpdir), atlas_names=["atlas_B"])
            result = analysis.run(lesion_data_obj)
            results = result.results["RegionalDamage"]

            # Should only have atlas_B results
            assert any("atlas_B" in key for key in results.keys())
            assert not any("atlas_A" in key for key in results.keys())
            assert not any("atlas_C" in key for key in results.keys())

            # Test 2: Process atlas_A and atlas_C
            analysis = RegionalDamage(atlas_dir=str(tmpdir), atlas_names=["atlas_A", "atlas_C"])
            result = analysis.run(lesion_data_obj)
            results = result.results["RegionalDamage"]

            # Should have atlas_A and atlas_C, but not atlas_B
            assert any("atlas_A" in key for key in results.keys())
            assert not any("atlas_B" in key for key in results.keys())
            assert any("atlas_C" in key for key in results.keys())

            # Test 3: None = process all atlases
            analysis = RegionalDamage(atlas_dir=str(tmpdir), atlas_names=None)
            result = analysis.run(lesion_data_obj)
            results = result.results["RegionalDamage"]

            # Should have all three atlases
            assert any("atlas_A" in key for key in results.keys())
            assert any("atlas_B" in key for key in results.keys())
            assert any("atlas_C" in key for key in results.keys())

    def test_atlas_names_warns_if_not_found(self):
        """Test that warning is issued if requested atlas not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[20:30, 20:30, 20:30] = 1
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

            # Create only atlas_A
            atlas_3d = np.zeros(shape, dtype=np.uint8)
            atlas_3d[15:35, 15:35, 15:35] = 1

            atlas_img = nib.Nifti1Image(atlas_3d, affine)
            atlas_path = tmpdir / "atlas_A.nii.gz"
            nib.save(atlas_img, atlas_path)

            labels_path = tmpdir / "atlas_A_labels.txt"
            labels_path.write_text("1 Region1\n")

            # Load lesion
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152_2mm"})

            # Request atlas_A and atlas_B (atlas_B doesn't exist)
            analysis = RegionalDamage(atlas_dir=str(tmpdir), atlas_names=["atlas_A", "atlas_B"])

            with pytest.warns(UserWarning, match="Some requested atlases were not found.*atlas_B"):
                result = analysis.run(lesion_data_obj)

            # Should still process atlas_A successfully
            results = result.results["RegionalDamage"]
            assert any("atlas_A" in key for key in results.keys())

    def test_atlas_names_raises_if_none_found(self):
        """Test that error is raised if no matching atlases found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[20:30, 20:30, 20:30] = 1
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

            # Create atlas_A (but we'll request atlas_B)
            atlas_3d = np.zeros(shape, dtype=np.uint8)
            atlas_3d[15:35, 15:35, 15:35] = 1

            atlas_img = nib.Nifti1Image(atlas_3d, affine)
            atlas_path = tmpdir / "atlas_A.nii.gz"
            nib.save(atlas_img, atlas_path)

            labels_path = tmpdir / "atlas_A_labels.txt"
            labels_path.write_text("1 Region1\n")

            # Load lesion
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152_2mm"})

            # Request only atlas_B (doesn't exist)
            analysis = RegionalDamage(atlas_dir=str(tmpdir), atlas_names=["atlas_B"])

            with pytest.raises(ValueError, match="No matching atlases found for specified names"):
                analysis.run(lesion_data_obj)

    def test_atlas_names_validates_input_type(self):
        """Test that atlas_names validation catches invalid types."""
        # Should raise TypeError for non-list
        with pytest.raises(TypeError, match="atlas_names must be a list"):
            AtlasAggregation(atlas_dir="/tmp", atlas_names="atlas_A")

        # Should raise TypeError for list with non-strings
        with pytest.raises(TypeError, match="All items in atlas_names must be strings"):
            AtlasAggregation(atlas_dir="/tmp", atlas_names=["atlas_A", 123])

        # Should raise ValueError for empty list
        with pytest.raises(ValueError, match="atlas_names cannot be an empty list"):
            AtlasAggregation(atlas_dir="/tmp", atlas_names=[])

    def test_atlas_names_works_with_atlas_aggregation(self):
        """Test that atlas_names works with AtlasAggregation (not just RegionalDamage)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create lesion
            shape = (50, 50, 50)
            affine = np.eye(4)
            lesion_data = np.zeros(shape, dtype=np.uint8)
            lesion_data[20:30, 20:30, 20:30] = 1
            lesion_img = nib.Nifti1Image(lesion_data, affine)
            lesion_path = tmpdir / "lesion.nii.gz"
            nib.save(lesion_img, lesion_path)

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
            lesion_data_obj = LesionData.from_nifti(lesion_path=lesion_path, metadata={"space": "MNI152_2mm"})

            # Use AtlasAggregation with atlas_names filter
            analysis = AtlasAggregation(
                atlas_dir=str(tmpdir),
                source="lesion_img",
                aggregation="mean",
                atlas_names=["atlas_X"],
            )
            result = analysis.run(lesion_data_obj)
            results = result.results["AtlasAggregation"]

            # Should only have atlas_X
            assert any("atlas_X" in key for key in results.keys())
            assert not any("atlas_Y" in key for key in results.keys())
