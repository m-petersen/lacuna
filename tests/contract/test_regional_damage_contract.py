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


def test_regional_damage_validates_atlas_available(synthetic_lesion_img):
    """Test that RegionalDamage validates requested atlas exists."""
    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage

    # Request a nonexistent atlas
    analysis = RegionalDamage(atlas_names=["NonExistentAtlas123"])
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    # Should raise error if requested atlas doesn't exist
    with pytest.raises(ValueError, match="atlas|not found|NonExistentAtlas123"):
        analysis.run(lesion_data)


def test_regional_damage_uses_atlas_registry(tmp_path):
    """Test that RegionalDamage uses the atlas registry."""
    import nibabel as nib
    import numpy as np
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory, list_atlases

    # Create mock atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Create mock atlas files
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test_atlas.nii.gz")
    (atlas_dir / "test_atlas_labels.txt").write_text("1 Region1\n")

    # Register atlas
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    # Should be able to instantiate and find atlases
    analysis = RegionalDamage()
    assert len(list_atlases()) > 0


def test_regional_damage_requires_binary_mask(synthetic_lesion_img, tmp_path):
    """Test that RegionalDamage requires binary lesion mask."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    # Create non-binary lesion
    data = synthetic_lesion_img.get_fdata()
    data = data.astype(float) * 0.5

    non_binary_img = nib.Nifti1Image(data, synthetic_lesion_img.affine)
    
    # LesionData now validates binary mask in __init__
    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        lesion_data = LesionData(lesion_img=non_binary_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})


def test_regional_damage_returns_lesion_data(synthetic_lesion_img, tmp_path):
    """Test that run() returns a LesionData object with namespaced results."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory

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
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = RegionalDamage()
    result = analysis.run(lesion_data)

    # Should return LesionData
    assert isinstance(result, LesionData)

    # Should have namespaced results
    assert "RegionalDamage" in result.results


def test_regional_damage_result_structure(synthetic_lesion_img, tmp_path):
    """Test that results contain expected ROI damage percentages."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    nib.save(atlas_img, atlas_dir / "test_atlas.nii.gz")
    (atlas_dir / "test_atlas_labels.txt").write_text("1 TestRegion\n")
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = RegionalDamage()
    result = analysis.run(lesion_data)

    # Results are returned as dict with atlas name as key
    atlas_results = result.results["RegionalDamage"]
    assert "test_atlas" in atlas_results
    
    # Get the ROIResult for this atlas
    roi_result = atlas_results["test_atlas"]
    results_dict = roi_result.get_data()

    # Should contain ROI-level damage percentages
    # Format: {"test_atlas_TestRegion": 15.3, ...}
    assert len(results_dict) > 0
    assert isinstance(results_dict, dict)
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))
        assert 0 <= value <= 100  # Percentage


def test_regional_damage_handles_3d_and_4d_atlases(synthetic_lesion_img, tmp_path):
    """Test that RegionalDamage can handle both 3D and 4D atlases."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory

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
    
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = RegionalDamage()
    result = analysis.run(lesion_data)

    # Results are returned as dict with one entry per atlas
    atlas_results = result.results["RegionalDamage"]
    assert "atlas_3d" in atlas_results
    assert "atlas_4d" in atlas_results
    
    # Each atlas should have its own ROIResult
    results_3d = atlas_results["atlas_3d"].get_data()
    results_4d = atlas_results["atlas_4d"].get_data()
    assert len(results_3d) > 0
    assert len(results_4d) > 0


def test_regional_damage_preserves_input_immutability(synthetic_lesion_img, tmp_path):
    """Test that run() does not modify the input LesionData."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
    original_results = lesion_data.results.copy()

    analysis = RegionalDamage()
    result = analysis.run(lesion_data)

    # Input should not be modified
    assert lesion_data.results == original_results
    assert "RegionalDamage" not in lesion_data.results

    # Result should be different object
    assert result is not lesion_data


def test_regional_damage_adds_provenance(synthetic_lesion_img, tmp_path):
    """Test that run() adds provenance record."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.regional_damage import RegionalDamage
    from lacuna.assets.atlases.registry import register_atlases_from_directory

    # Create mock atlas and register it
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")
    register_atlases_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
    original_prov_len = len(lesion_data.provenance)

    analysis = RegionalDamage()
    result = analysis.run(lesion_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "RegionalDamage" in latest_prov["function"]
