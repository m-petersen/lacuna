"""
Unit tests for spatial compatibility checking in atlas aggregation.

Tests that AtlasAggregation properly handles atlases with mismatched
spatial dimensions, especially when multiple atlases are present.
"""

import warnings

import nibabel as nib
import numpy as np
import pytest

from ldk import LesionData
from ldk.analysis import AtlasAggregation, RegionalDamage


def test_atlas_aggregation_skips_incompatible_atlas_shapes(tmp_path):
    """Test that atlases with incompatible shapes are skipped with warning."""
    # Create lesion with specific shape
    lesion_shape = (64, 64, 64)
    lesion_array = np.zeros(lesion_shape, dtype=np.uint8)
    lesion_array[20:40, 20:40, 20:40] = 1
    lesion_img = nib.Nifti1Image(lesion_array, np.eye(4))
    lesion_data = LesionData(lesion_img=lesion_img)

    # Create atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Atlas 1: Compatible shape (64x64x64)
    atlas1_data = np.zeros(lesion_shape, dtype=np.uint8)
    atlas1_data[10:30, 10:30, 10:30] = 1
    atlas1_data[30:50, 30:50, 30:50] = 2
    nib.save(nib.Nifti1Image(atlas1_data, np.eye(4)), atlas_dir / "compatible.nii.gz")
    (atlas_dir / "compatible_labels.txt").write_text("1 Region_A\n2 Region_B\n")

    # Atlas 2: INCOMPATIBLE shape (97x115x97) - different dimensions!
    incompatible_shape = (97, 115, 97)
    atlas2_data = np.zeros(incompatible_shape, dtype=np.uint8)
    atlas2_data[10:30, 10:30, 10:30] = 1
    nib.save(nib.Nifti1Image(atlas2_data, np.eye(4)), atlas_dir / "incompatible.nii.gz")
    (atlas_dir / "incompatible_labels.txt").write_text("1 Region_C\n")

    # Atlas 3: Another compatible shape (64x64x64)
    atlas3_data = np.zeros(lesion_shape, dtype=np.uint8)
    atlas3_data[40:60, 40:60, 40:60] = 1
    nib.save(nib.Nifti1Image(atlas3_data, np.eye(4)), atlas_dir / "compatible2.nii.gz")
    (atlas_dir / "compatible2_labels.txt").write_text("1 Region_D\n")

    # Run analysis - should skip incompatible atlas with warning
    analysis = AtlasAggregation(atlas_dir=str(atlas_dir), aggregation="percent")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = analysis.run(lesion_data)

        # Should have warning about incompatible atlas
        assert len(w) >= 1
        warning_messages = [str(warning.message) for warning in w]
        assert any("incompatible" in msg.lower() for msg in warning_messages)
        assert any("incompatible" in msg for msg in warning_messages)

    # Results should only include compatible atlases
    atlas_results = result.results["AtlasAggregation"]

    # Should have results from compatible atlases only
    assert "compatible_Region_A" in atlas_results
    assert "compatible_Region_B" in atlas_results
    assert "compatible2_Region_D" in atlas_results

    # Should NOT have results from incompatible atlas
    assert "incompatible_Region_C" not in atlas_results


def test_regional_damage_with_mixed_atlas_sizes(tmp_path):
    """Test RegionalDamage with mixture of compatible and incompatible atlases."""
    # Create binary lesion
    lesion_shape = (91, 109, 91)  # Standard MNI152 2mm dimensions
    lesion_array = np.zeros(lesion_shape, dtype=np.uint8)
    lesion_array[40:50, 50:60, 40:50] = 1
    lesion_img = nib.Nifti1Image(lesion_array, np.eye(4))
    lesion_data = LesionData(lesion_img=lesion_img)

    # Create atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Compatible atlas (MNI152 2mm)
    atlas1_data = np.zeros(lesion_shape, dtype=np.uint8)
    atlas1_data[35:55, 45:65, 35:55] = 1
    nib.save(nib.Nifti1Image(atlas1_data, np.eye(4)), atlas_dir / "mni2mm.nii.gz")
    (atlas_dir / "mni2mm_labels.txt").write_text("1 Frontal_Region\n")

    # Incompatible atlas (MNI152 1mm - different dimensions)
    incompatible_shape = (182, 218, 182)
    atlas2_data = np.zeros(incompatible_shape, dtype=np.uint8)
    atlas2_data[70:100, 90:120, 70:100] = 1
    nib.save(nib.Nifti1Image(atlas2_data, np.eye(4)), atlas_dir / "mni1mm.nii.gz")
    (atlas_dir / "mni1mm_labels.txt").write_text("1 Parietal_Region\n")

    # Run RegionalDamage - should work despite mixed atlases
    analysis = RegionalDamage(atlas_dir=str(atlas_dir))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = analysis.run(lesion_data)

        # Should warn about incompatible atlas
        assert len(w) >= 1
        assert any("incompatible" in str(warning.message).lower() for warning in w)

    # Should have results from compatible atlas only
    # Note: RegionalDamage stores results under its own class name "RegionalDamage"
    atlas_results = result.results.get("RegionalDamage", {})
    assert "mni2mm_Frontal_Region" in atlas_results
    assert "mni1mm_Parietal_Region" not in atlas_results

    # Result should have non-zero damage for overlapping region
    assert atlas_results["mni2mm_Frontal_Region"] > 0


def test_all_atlases_incompatible_shape_still_returns_result(tmp_path):
    """Test that analysis returns empty results when all atlases are incompatible."""
    # Create lesion
    lesion_shape = (64, 64, 64)
    lesion_array = np.zeros(lesion_shape, dtype=np.uint8)
    lesion_array[20:40, 20:40, 20:40] = 1
    lesion_img = nib.Nifti1Image(lesion_array, np.eye(4))
    lesion_data = LesionData(lesion_img=lesion_img)

    # Create atlas directory with only incompatible atlases
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Incompatible atlas 1
    atlas1_data = np.zeros((91, 109, 91), dtype=np.uint8)
    atlas1_data[40:50, 50:60, 40:50] = 1
    nib.save(nib.Nifti1Image(atlas1_data, np.eye(4)), atlas_dir / "atlas1.nii.gz")
    (atlas_dir / "atlas1_labels.txt").write_text("1 Region_A\n")

    # Incompatible atlas 2
    atlas2_data = np.zeros((182, 218, 182), dtype=np.uint8)
    atlas2_data[80:100, 100:120, 80:100] = 1
    nib.save(nib.Nifti1Image(atlas2_data, np.eye(4)), atlas_dir / "atlas2.nii.gz")
    (atlas_dir / "atlas2_labels.txt").write_text("1 Region_B\n")

    # Run analysis - should complete with warnings
    analysis = AtlasAggregation(atlas_dir=str(atlas_dir))

    # Should raise ValueError when ALL atlases are incompatible
    with pytest.raises(ValueError, match="No compatible atlases"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analysis.run(lesion_data)

            # Should warn about both incompatible atlases
            assert len(w) >= 2


def test_4d_atlas_spatial_compatibility(tmp_path):
    """Test spatial compatibility checking for 4D probabilistic atlases."""
    # Create lesion
    lesion_shape = (64, 64, 64)
    lesion_array = np.zeros(lesion_shape, dtype=np.uint8)
    lesion_array[20:40, 20:40, 20:40] = 1
    lesion_img = nib.Nifti1Image(lesion_array, np.eye(4))
    lesion_data = LesionData(lesion_img=lesion_img)

    # Create atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Compatible 4D atlas (64x64x64x3)
    atlas1_data = np.zeros(lesion_shape + (3,), dtype=np.float32)
    atlas1_data[10:30, 10:30, 10:30, 0] = 0.8  # Region 1
    atlas1_data[30:50, 30:50, 30:50, 1] = 0.9  # Region 2
    atlas1_data[40:60, 40:60, 40:60, 2] = 0.7  # Region 3
    nib.save(nib.Nifti1Image(atlas1_data, np.eye(4)), atlas_dir / "prob4d.nii.gz")
    (atlas_dir / "prob4d_labels.txt").write_text("Region_1\nRegion_2\nRegion_3\n")

    # Incompatible 4D atlas (97x115x97x2)
    atlas2_data = np.zeros((97, 115, 97, 2), dtype=np.float32)
    atlas2_data[10:30, 10:30, 10:30, 0] = 0.8
    nib.save(nib.Nifti1Image(atlas2_data, np.eye(4)), atlas_dir / "prob4d_incompatible.nii.gz")
    (atlas_dir / "prob4d_incompatible_labels.txt").write_text("Region_X\nRegion_Y\n")

    # Run analysis
    analysis = RegionalDamage(atlas_dir=str(atlas_dir), threshold=0.5)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = analysis.run(lesion_data)

        # Should warn about incompatible 4D atlas
        assert any("incompatible" in str(warning.message).lower() for warning in w)

    # Should have results from compatible 4D atlas only
    # Note: RegionalDamage stores results under its own class name "RegionalDamage"
    atlas_results = result.results.get("RegionalDamage", {})
    assert "prob4d_Region_1" in atlas_results
    assert "prob4d_Region_2" in atlas_results
    assert "prob4d_Region_3" in atlas_results
    assert "prob4d_incompatible_Region_X" not in atlas_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
