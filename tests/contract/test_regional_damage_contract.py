"""
Contract tests for RegionalDamage analysis class.

Tests the interface and behavior requirements for lesion-atlas overlap
quantification following the BaseAnalysis contract.
"""

import pytest


def test_regional_damage_import():
    """Test that RegionalDamage can be imported."""
    from lacuna.analysis.regional_damage import RegionalDamage

    assert RegionalDamage is not None


def test_regional_damage_inherits_base_analysis():
    """Test that RegionalDamage inherits from BaseAnalysis."""
    from lacuna.analysis.base import BaseAnalysis
    from lacuna.analysis.regional_damage import RegionalDamage

    assert issubclass(RegionalDamage, BaseAnalysis)


def test_regional_damage_can_instantiate():
    """Test that RegionalDamage can be instantiated."""
    from lacuna.analysis.regional_damage import RegionalDamage

    analysis = RegionalDamage()
    assert analysis is not None
    # Uses atlas registry, no atlas_dir parameter


def test_regional_damage_has_run_method():
    """Test that RegionalDamage has the run() method from BaseAnalysis."""
    from lacuna.analysis.regional_damage import RegionalDamage

    analysis = RegionalDamage()
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_regional_damage_validates_atlas_available(synthetic_mask_img):
    """Test that RegionalDamage validates requested atlas exists."""
    from lacuna import MaskData
    from lacuna.analysis.regional_damage import RegionalDamage

    # Request a nonexistent atlas
    analysis = RegionalDamage(parcel_names=["NonExistentAtlas123"])
    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Should raise error if requested atlas doesn't exist
    with pytest.raises(ValueError, match="atlas|not found|NonExistentAtlas123"):
        analysis.run(mask_data)


def test_regional_damage_uses_atlas_registry(tmp_path):
    """Test that RegionalDamage uses the atlas registry."""
    import nibabel as nib
    import numpy as np

    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.parcellations.registry import (
        list_parcellations,
        register_parcellations_from_directory,
    )

    # Create mock atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Create mock atlas files
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test_atlas.nii.gz")
    (atlas_dir / "test_atlas_labels.txt").write_text("1 Region1\n")

    # Register atlas
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    # Should be able to instantiate and find atlases
    RegionalDamage()
    assert len(list_parcellations()) > 0


def test_regional_damage_requires_binary_mask(synthetic_mask_img, tmp_path):
    """Test that RegionalDamage requires binary lesion mask."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    # Create non-binary lesion
    data = synthetic_mask_img.get_fdata()
    data = data.astype(float) * 0.5

    non_binary_img = nib.Nifti1Image(data, synthetic_mask_img.affine)

    # MaskData now validates binary mask in __init__
    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        MaskData(mask_img=non_binary_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})


def test_regional_damage_returns_mask_data(synthetic_mask_img, tmp_path):
    """Test that run() returns a MaskData object with namespaced results."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Create simple 3D integer atlas
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1  # Region 1
    atlas_data[30:50, 30:50, 30:50] = 2  # Region 2
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    nib.save(atlas_img, atlas_dir / "test_atlas.nii.gz")

    # Create label file
    (atlas_dir / "test_atlas_labels.txt").write_text("1 Region1\n2 Region2\n")
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = RegionalDamage()
    result = analysis.run(mask_data)

    # Should return MaskData
    assert isinstance(result, MaskData)

    # Should have namespaced results
    assert "RegionalDamage" in result.results


def test_regional_damage_result_structure(synthetic_mask_img, tmp_path):
    """Test that results contain expected ROI damage percentages."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    nib.save(atlas_img, atlas_dir / "test_atlas.nii.gz")
    (atlas_dir / "test_atlas_labels.txt").write_text("1 TestRegion\n")
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = RegionalDamage()
    result = analysis.run(mask_data)

    # Results are returned as dict with atlas name as key (BIDS format)
    atlas_results = result.results["RegionalDamage"]
    assert "atlas-test_atlas_desc-MaskImg" in atlas_results

    # Get the ParcelData for this atlas
    roi_result = atlas_results["atlas-test_atlas_desc-MaskImg"]
    results_dict = roi_result.get_data()

    # Should contain ROI-level damage percentages
    # Format: {"test_atlas_TestRegion": 15.3, ...}
    assert len(results_dict) > 0
    assert isinstance(results_dict, dict)
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))
        assert 0 <= value <= 100  # Percentage


def test_regional_damage_handles_3d_and_4d_atlases(synthetic_mask_img, tmp_path):
    """Test that RegionalDamage can handle both 3D and 4D atlases."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Create 3D atlas
    atlas_3d = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_3d[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_3d, np.eye(4)), atlas_dir / "atlas_3d.nii.gz")
    (atlas_dir / "atlas_3d_labels.txt").write_text("1 Region1\n")

    # Create 4D probabilistic atlas
    atlas_4d = np.zeros((64, 64, 64, 2), dtype=np.float32)
    atlas_4d[20:40, 20:40, 20:40, 0] = 0.8  # Region 1
    atlas_4d[30:50, 30:50, 30:50, 1] = 0.7  # Region 2
    nib.save(nib.Nifti1Image(atlas_4d, np.eye(4)), atlas_dir / "atlas_4d.nii.gz")
    (atlas_dir / "atlas_4d_labels.txt").write_text("Region1\nRegion2\n")

    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = RegionalDamage()
    result = analysis.run(mask_data)

    # Results are returned as dict with one entry per atlas (BIDS format)
    atlas_results = result.results["RegionalDamage"]
    assert "atlas-atlas_3d_desc-MaskImg" in atlas_results
    assert "atlas_4d_from_mask_img" in atlas_results

    # Each atlas should have its own ParcelData
    results_3d = atlas_results["atlas_3d_from_mask_img"].get_data()
    results_4d = atlas_results["atlas_4d_from_mask_img"].get_data()
    assert len(results_3d) > 0
    assert len(results_4d) > 0


def test_regional_damage_preserves_input_immutability(synthetic_mask_img, tmp_path):
    """Test that run() does not modify the input MaskData."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_results = mask_data.results.copy()

    analysis = RegionalDamage()
    result = analysis.run(mask_data)

    # Input should not be modified
    assert mask_data.results == original_results
    assert "RegionalDamage" not in mask_data.results

    # Result should be different object
    assert result is not mask_data


def test_regional_damage_adds_provenance(synthetic_mask_img, tmp_path):
    """Test that run() adds provenance record."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_prov_len = len(mask_data.provenance)

    analysis = RegionalDamage()
    result = analysis.run(mask_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "RegionalDamage" in latest_prov["function"]
