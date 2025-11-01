"""
Contract tests for AtlasAggregation analysis class.

Tests the interface and behavior requirements for composable ROI-level
aggregation of voxel-level maps following the BaseAnalysis contract.
"""

import pytest


def test_atlas_aggregation_import():
    """Test that AtlasAggregation can be imported."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    assert AtlasAggregation is not None


def test_atlas_aggregation_inherits_base_analysis():
    """Test that AtlasAggregation inherits from BaseAnalysis."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation
    from ldk.analysis.base import BaseAnalysis

    assert issubclass(AtlasAggregation, BaseAnalysis)


def test_atlas_aggregation_can_instantiate():
    """Test that AtlasAggregation can be instantiated with required parameters."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    analysis = AtlasAggregation(
        atlas_dir="/path/to/atlases", source="lesion_img", aggregation="mean"
    )
    assert analysis is not None


def test_atlas_aggregation_has_run_method():
    """Test that AtlasAggregation has the run() method from BaseAnalysis."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    analysis = AtlasAggregation(atlas_dir="/path/to/atlases")
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_atlas_aggregation_accepts_source_parameter():
    """Test that AtlasAggregation accepts source parameter to specify data source."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Can aggregate lesion directly
    analysis1 = AtlasAggregation(atlas_dir="/path/to/atlases", source="lesion_img")
    assert analysis1.source == "lesion_img"

    # Can aggregate from previous analysis result
    analysis2 = AtlasAggregation(
        atlas_dir="/path/to/atlases",
        source="FunctionalNetworkMapping.network_map",
    )
    assert analysis2.source == "FunctionalNetworkMapping.network_map"


def test_atlas_aggregation_accepts_aggregation_methods():
    """Test that AtlasAggregation accepts different aggregation methods."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    for method in ["mean", "sum", "percent", "volume"]:
        analysis = AtlasAggregation(
            atlas_dir="/path/to/atlases", source="lesion_img", aggregation=method
        )
        assert analysis.aggregation == method


def test_atlas_aggregation_validates_aggregation_method():
    """Test that invalid aggregation method raises ValueError."""
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    with pytest.raises(ValueError, match="aggregation"):
        AtlasAggregation(
            atlas_dir="/path/to/atlases",
            source="lesion_img",
            aggregation="invalid_method",
        )


def test_atlas_aggregation_validates_atlas_directory(synthetic_lesion_img):
    """Test that AtlasAggregation validates atlas directory exists."""
    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    analysis = AtlasAggregation(atlas_dir="/nonexistent/path")
    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    with pytest.raises((ValueError, FileNotFoundError), match="atlas"):
        analysis.run(lesion_data)


def test_atlas_aggregation_validates_source_exists(synthetic_lesion_img, tmp_path):
    """Test that AtlasAggregation validates source data exists."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    # Try to aggregate from non-existent analysis result
    analysis = AtlasAggregation(
        atlas_dir=str(atlas_dir),
        source="NonExistentAnalysis.network_map",
    )

    with pytest.raises(ValueError, match="source"):
        analysis.run(lesion_data)


def test_atlas_aggregation_can_chain_with_other_analyses(synthetic_lesion_img, tmp_path):
    """Test that AtlasAggregation can access results from previous analyses."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    # Create lesion data with mock analysis results
    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    # Add mock network map from previous analysis
    network_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_lesion_img.affine)
    lesion_data._results["MockAnalysis"] = {"network_map": network_map}

    # Should be able to aggregate the network map
    analysis = AtlasAggregation(
        atlas_dir=str(atlas_dir),
        source="MockAnalysis.network_map",
        aggregation="mean",
    )

    result = analysis.run(lesion_data)

    # Should have aggregated results
    assert "AtlasAggregation" in result.results


def test_atlas_aggregation_returns_lesion_data(synthetic_lesion_img, tmp_path):
    """Test that run() returns a LesionData object with namespaced results."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = AtlasAggregation(atlas_dir=str(atlas_dir), source="lesion_img")
    result = analysis.run(lesion_data)

    # Should return LesionData
    assert isinstance(result, LesionData)

    # Should have namespaced results
    assert "AtlasAggregation" in result.results


def test_atlas_aggregation_result_structure(synthetic_lesion_img, tmp_path):
    """Test that results contain ROI-level aggregated values."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 TestRegion\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = AtlasAggregation(atlas_dir=str(atlas_dir), source="lesion_img", aggregation="mean")
    result = analysis.run(lesion_data)

    results_dict = result.results["AtlasAggregation"]

    # Should contain ROI-level values
    # Format: {"test_TestRegion": 0.523, ...}
    assert len(results_dict) > 0
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))


def test_atlas_aggregation_handles_multiple_atlases(synthetic_lesion_img, tmp_path):
    """Test that AtlasAggregation can process multiple atlases in directory."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Create two atlases
    for atlas_name in ["atlas1", "atlas2"]:
        atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
        atlas_data[20:40, 20:40, 20:40] = 1
        nib.save(
            nib.Nifti1Image(atlas_data, np.eye(4)),
            atlas_dir / f"{atlas_name}.nii.gz",
        )
        (atlas_dir / f"{atlas_name}_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)

    analysis = AtlasAggregation(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    results_dict = result.results["AtlasAggregation"]

    # Should have results from both atlases
    assert any("atlas1" in key for key in results_dict.keys())
    assert any("atlas2" in key for key in results_dict.keys())


def test_atlas_aggregation_preserves_input_immutability(synthetic_lesion_img, tmp_path):
    """Test that run() does not modify the input LesionData."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)
    original_results = lesion_data.results.copy()

    analysis = AtlasAggregation(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    # Input should not be modified
    assert lesion_data.results == original_results
    assert "AtlasAggregation" not in lesion_data.results

    # Result should be different object
    assert result is not lesion_data


def test_atlas_aggregation_adds_provenance(synthetic_lesion_img, tmp_path):
    """Test that run() adds provenance record."""
    import nibabel as nib
    import numpy as np

    from ldk import LesionData
    from ldk.analysis.atlas_aggregation import AtlasAggregation

    # Create mock atlas
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()
    atlas_data = np.zeros((64, 64, 64), dtype=np.uint8)
    atlas_data[20:40, 20:40, 20:40] = 1
    nib.save(nib.Nifti1Image(atlas_data, np.eye(4)), atlas_dir / "test.nii.gz")
    (atlas_dir / "test_labels.txt").write_text("1 Region1\n")

    lesion_data = LesionData(lesion_img=synthetic_lesion_img)
    original_prov_len = len(lesion_data.provenance)

    analysis = AtlasAggregation(atlas_dir=str(atlas_dir))
    result = analysis.run(lesion_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "AtlasAggregation" in latest_prov["function"]
