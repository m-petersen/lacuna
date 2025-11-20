"""
Contract tests for AtlasAggregation analysis class.

Tests the interface and behavior requirements for composable ROI-level
aggregation of voxel-level maps following the BaseAnalysis contract.
"""

import pytest


def test_atlas_aggregation_import():
    """Test that AtlasAggregation can be imported."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    assert AtlasAggregation is not None


def test_atlas_aggregation_inherits_base_analysis():
    """Test that AtlasAggregation inherits from BaseAnalysis."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation
    from lacuna.analysis.base import BaseAnalysis

    assert issubclass(AtlasAggregation, BaseAnalysis)


def test_atlas_aggregation_can_instantiate():
    """Test that AtlasAggregation can be instantiated with required parameters."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    analysis = AtlasAggregation(source="lesion_img", aggregation="mean")
    assert analysis is not None


def test_atlas_aggregation_has_run_method():
    """Test that AtlasAggregation has the run() method from BaseAnalysis."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    analysis = AtlasAggregation()
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_atlas_aggregation_accepts_source_parameter():
    """Test that AtlasAggregation accepts source parameter to specify data source."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    # Can aggregate lesion directly
    analysis1 = AtlasAggregation(source="lesion_img")
    assert analysis1.source == "lesion_img"

    # Can aggregate from previous analysis result
    analysis2 = AtlasAggregation(
        source="FunctionalNetworkMapping.network_map",
    )
    assert analysis2.source == "FunctionalNetworkMapping.network_map"


def test_atlas_aggregation_accepts_aggregation_methods():
    """Test that AtlasAggregation accepts different aggregation methods."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    for method in ["mean", "sum", "percent", "volume"]:
        analysis = AtlasAggregation(source="lesion_img", aggregation=method)
        assert analysis.aggregation == method


def test_atlas_aggregation_validates_aggregation_method():
    """Test that invalid aggregation method raises ValueError."""
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    with pytest.raises(ValueError, match="aggregation"):
        AtlasAggregation(
            source="lesion_img",
            aggregation="invalid_method",
        )


def test_atlas_aggregation_validates_atlas_directory(synthetic_lesion_img):
    """Test that AtlasAggregation validates atlases are available in registry."""
    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation
    from lacuna.assets.atlases.registry import ATLAS_REGISTRY

    # Save current registry state
    saved_registry = ATLAS_REGISTRY.copy()
    
    try:
        # Clear registry
        ATLAS_REGISTRY.clear()
        
        analysis = AtlasAggregation()
        lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

        with pytest.raises(ValueError, match="atlas|registry"):
            analysis.run(lesion_data)
    finally:
        # Restore registry
        ATLAS_REGISTRY.clear()
        ATLAS_REGISTRY.update(saved_registry)


def test_atlas_aggregation_validates_source_exists(synthetic_lesion_img, tmp_path):
    """Test that AtlasAggregation validates source data exists."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation
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

    # Try to aggregate from non-existent analysis result
    analysis = AtlasAggregation(
        source="NonExistentAnalysis.network_map",
    )

    with pytest.raises(ValueError, match="source"):
        analysis.run(lesion_data)


def test_atlas_aggregation_can_chain_with_other_analyses(synthetic_lesion_img):
    """Test that AtlasAggregation can access results from previous analyses."""
    import nibabel as nib
    import numpy as np

    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    # Create lesion data with mock analysis results
    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    # Add mock network map from previous analysis
    network_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_lesion_img.affine)
    lesion_data._results["MockAnalysis"] = {"network_map": network_map}

    # Should be able to aggregate the network map using bundled atlas
    analysis = AtlasAggregation(
        source="MockAnalysis.network_map",
        aggregation="mean",
        atlas_names=["Schaefer2018_100Parcels7Networks"],
    )

    result = analysis.run(lesion_data)

    # Should have aggregated results
    assert "AtlasAggregation" in result.results


def test_atlas_aggregation_returns_lesion_data(synthetic_lesion_img):
    """Test that run() returns a LesionData object with namespaced results."""
    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = AtlasAggregation(
        source="lesion_img",
        atlas_names=["Schaefer2018_100Parcels7Networks"],
    )
    result = analysis.run(lesion_data)

    # Should return LesionData
    assert isinstance(result, LesionData)

    # Should have namespaced results
    assert "AtlasAggregation" in result.results


def test_atlas_aggregation_result_structure(synthetic_lesion_img):
    """Test that results contain ROI-level aggregated values."""
    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    analysis = AtlasAggregation(
        source="lesion_img",
        aggregation="mean",
        atlas_names=["Schaefer2018_100Parcels7Networks"],
    )
    result = analysis.run(lesion_data)

    # Results are returned as dict with atlas name as key
    atlas_results = result.results["AtlasAggregation"]
    assert "Schaefer2018_100Parcels7Networks" in atlas_results
    
    # Get the ROIResult for this atlas
    roi_result = atlas_results["Schaefer2018_100Parcels7Networks"]
    results_dict = roi_result.get_data()

    # Should contain ROI-level values
    # Format: {"Schaefer2018_100Parcels7Networks_7Networks_LH_Vis_1": 0.523, ...}
    assert len(results_dict) > 0
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))


def test_atlas_aggregation_handles_multiple_atlases(synthetic_lesion_img):
    """Test that AtlasAggregation can process multiple atlases from registry."""
    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    # Use two bundled atlases
    analysis = AtlasAggregation(
        atlas_names=["Schaefer2018_100Parcels7Networks", "Schaefer2018_200Parcels7Networks"],
    )
    result = analysis.run(lesion_data)

    # Results are returned as dict with one entry per atlas
    atlas_results = result.results["AtlasAggregation"]
    assert "Schaefer2018_100Parcels7Networks" in atlas_results
    assert "Schaefer2018_200Parcels7Networks" in atlas_results

    # Each atlas should have its own ROIResult with region data
    roi_100 = atlas_results["Schaefer2018_100Parcels7Networks"].get_data()
    roi_200 = atlas_results["Schaefer2018_200Parcels7Networks"].get_data()
    assert len(roi_100) > 0
    assert len(roi_200) > 0


def test_atlas_aggregation_preserves_input_immutability(synthetic_lesion_img):
    """Test that run() does not modify the input LesionData."""
    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
    original_results = lesion_data.results.copy()

    analysis = AtlasAggregation(atlas_names=["Schaefer2018_100Parcels7Networks"])
    result = analysis.run(lesion_data)

    # Input should not be modified
    assert lesion_data.results == original_results
    assert "AtlasAggregation" not in lesion_data.results

    # Result should be different object
    assert result is not lesion_data


def test_atlas_aggregation_adds_provenance(synthetic_lesion_img):
    """Test that run() adds provenance record."""
    from lacuna import LesionData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    lesion_data = LesionData(lesion_img=synthetic_lesion_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
    original_prov_len = len(lesion_data.provenance)

    analysis = AtlasAggregation(atlas_names=["Schaefer2018_100Parcels7Networks"])
    result = analysis.run(lesion_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "AtlasAggregation" in latest_prov["function"]
