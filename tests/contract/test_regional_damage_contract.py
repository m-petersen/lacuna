"""
Contract tests for RegionalDamage analysis class.

Tests the interface and behavior requirements for lesion-atlas overlap
quantification following the BaseAnalysis contract.
"""

import pytest


def test_regional_damage_import():
    """Test that RegionalDamage can be imported."""
    from ldk.analysis.regional_damage import RegionalDamage

    assert RegionalDamage is not None


def test_regional_damage_inherits_base_analysis():
    """Test that RegionalDamage inherits from BaseAnalysis."""
    from ldk.analysis.base import BaseAnalysis
    from ldk.analysis.regional_damage import RegionalDamage

    assert issubclass(RegionalDamage, BaseAnalysis)


def test_regional_damage_can_instantiate():
    """Test that RegionalDamage can be instantiated with atlas directory."""
    from ldk.analysis.regional_damage import RegionalDamage

    analysis = RegionalDamage(atlas_dir="/path/to/atlases")
    assert analysis is not None
    assert analysis.atlas_dir == "/path/to/atlases"


def test_regional_damage_has_run_method():
    """Test that RegionalDamage has the run() method from BaseAnalysis."""
    from ldk.analysis.regional_damage import RegionalDamage

    analysis = RegionalDamage(atlas_dir="/path/to/atlases")
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_regional_damage_validates_atlas_directory(synthetic_lesion_img):
    """Test that RegionalDamage validates atlas directory exists."""
    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

    # Should raise error if atlas directory doesn't exist
    analysis = RegionalDamage(atlas_dir="/nonexistent/path")
    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    with pytest.raises((ValueError, FileNotFoundError), match="atlas"):
        analysis.run(lesion_data)


def test_regional_damage_discovers_atlases(tmp_path):
    """Test that RegionalDamage can discover atlas files in directory."""
    from ldk.analysis.regional_damage import RegionalDamage

    # Create mock atlas directory
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Create mock atlas files
    (atlas_dir / "schaefer100.nii.gz").touch()
    (atlas_dir / "schaefer100_labels.txt").write_text("Region1\nRegion2\n")

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))

    # Should discover the atlas
    assert hasattr(analysis, "atlases") or hasattr(analysis, "_discover_atlases")


def test_regional_damage_requires_binary_mask(synthetic_lesion_img, tmp_path):
    """Test that RegionalDamage requires binary lesion mask."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    # Create non-binary lesion
    data = synthetic_lesion_img.get_fdata()
    data = data.astype(float) * 0.5

    non_binary_img = nib.Nifti1Image(data, synthetic_lesion_img.affine)
    lesion_data = LesionData(lesion_img=non_binary_img)

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))

    # Should raise error for non-binary mask
    with pytest.raises(ValueError, match="binary"):
        analysis.run(lesion_data)


def test_regional_damage_returns_lesion_data(synthetic_lesion_img, tmp_path):
    """Test that run() returns a LesionData object with namespaced results."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

    # Create mock atlas
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

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    # Should return LesionData
    assert isinstance(result, LesionData)

    # Should have namespaced results
    assert "RegionalDamage" in result.results


def test_regional_damage_result_structure(synthetic_lesion_img, tmp_path):
    """Test that results contain expected ROI damage percentages."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
    nib.save(atlas_img, atlas_dir / "test_atlas.nii.gz")
    (atlas_dir / "test_atlas_labels.txt").write_text("1 TestRegion\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    results_dict = result.results["RegionalDamage"]

    # Should contain ROI-level damage percentages
    # Format: {"test_atlas_TestRegion": 15.3, ...}
    assert len(results_dict) > 0
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))
        assert 0 <= value <= 100  # Percentage


def test_regional_damage_handles_3d_and_4d_atlases(synthetic_lesion_img, tmp_path):
    """Test that RegionalDamage can handle both 3D and 4D atlases."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

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

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    # Should have results for both atlases
    results_dict = result.results["RegionalDamage"]
    assert any("atlas_3d" in key for key in results_dict.keys())
    assert any("atlas_4d" in key for key in results_dict.keys())


def test_regional_damage_preserves_input_immutability(synthetic_lesion_img, tmp_path):
    """Test that run() does not modify the input LesionData."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)
    original_results = lesion_data.results.copy()

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))
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

    from ldk import LesionData
    from ldk.analysis.regional_damage import RegionalDamage

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)
    original_prov_len = len(lesion_data.provenance)

    analysis = RegionalDamage(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "RegionalDamage" in latest_prov["function"]
