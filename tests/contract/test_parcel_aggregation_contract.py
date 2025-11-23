"""
Contract tests for ParcelAggregation analysis class.

Tests the interface and behavior requirements for composable ROI-level
aggregation of voxel-level maps following the BaseAnalysis contract.
"""

import pytest


def test_atlas_aggregation_import():
    """Test that ParcelAggregation can be imported."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    assert ParcelAggregation is not None


def test_atlas_aggregation_inherits_base_analysis():
    """Test that ParcelAggregation inherits from BaseAnalysis."""
    from lacuna.analysis.base import BaseAnalysis
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    assert issubclass(ParcelAggregation, BaseAnalysis)


def test_atlas_aggregation_can_instantiate():
    """Test that ParcelAggregation can be instantiated with required parameters."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    analysis = ParcelAggregation(source="mask_img", aggregation="mean")
    assert analysis is not None


def test_atlas_aggregation_has_run_method():
    """Test that ParcelAggregation has the run() method from BaseAnalysis."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    analysis = ParcelAggregation()
    assert hasattr(analysis, "run")
    assert callable(analysis.run)


def test_atlas_aggregation_accepts_source_parameter():
    """Test that ParcelAggregation accepts source parameter to specify data source."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    # Can aggregate lesion directly
    analysis1 = ParcelAggregation(source="mask_img")
    assert analysis1.source == "mask_img"

    # Can aggregate from previous analysis result
    analysis2 = ParcelAggregation(
        source="FunctionalNetworkMapping.network_map",
    )
    assert analysis2.source == "FunctionalNetworkMapping.network_map"


def test_atlas_aggregation_accepts_aggregation_methods():
    """Test that ParcelAggregation accepts different aggregation methods."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    for method in ["mean", "sum", "percent", "volume"]:
        analysis = ParcelAggregation(source="mask_img", aggregation=method)
        assert analysis.aggregation == method


def test_atlas_aggregation_validates_aggregation_method():
    """Test that invalid aggregation method raises ValueError."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    with pytest.raises(ValueError, match="aggregation"):
        ParcelAggregation(
            source="mask_img",
            aggregation="invalid_method",
        )


def test_atlas_aggregation_validates_atlas_directory(synthetic_mask_img):
    """Test that ParcelAggregation validates atlases are available in registry."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation
    from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

    # Save current registry state
    saved_registry = PARCELLATION_REGISTRY.copy()

    try:
        # Clear registry
        PARCELLATION_REGISTRY.clear()

        analysis = ParcelAggregation()
        mask_data = MaskData(
            mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
        )

        with pytest.raises(ValueError, match="atlas|registry"):
            analysis.run(mask_data)
    finally:
        # Restore registry
        PARCELLATION_REGISTRY.clear()
        PARCELLATION_REGISTRY.update(saved_registry)


def test_atlas_aggregation_validates_source_exists(synthetic_mask_img, tmp_path):
    """Test that ParcelAggregation validates source data exists."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation
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

    # Try to aggregate from non-existent analysis result
    analysis = ParcelAggregation(
        source="NonExistentAnalysis.network_map",
    )

    with pytest.raises(ValueError, match="source"):
        analysis.run(mask_data)


def test_atlas_aggregation_can_chain_with_other_analyses(synthetic_mask_img):
    """Test that ParcelAggregation can access results from previous analyses."""
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    # Create lesion data with mock analysis results
    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock network map from previous analysis
    network_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_mask_img.affine)
    mask_data._results["MockAnalysis"] = {"network_map": network_map}

    # Should be able to aggregate the network map using bundled atlas
    analysis = ParcelAggregation(
        source="MockAnalysis.network_map",
        aggregation="mean",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    result = analysis.run(mask_data)

    # Should have aggregated results
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_returns_mask_data(synthetic_mask_img):
    """Test that run() returns a MaskData object with namespaced results."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = ParcelAggregation(
        source="mask_img",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )
    result = analysis.run(mask_data)

    # Should return MaskData
    assert isinstance(result, MaskData)

    # Should have namespaced results
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_result_structure(synthetic_mask_img):
    """Test that results contain ROI-level aggregated values."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )
    result = analysis.run(mask_data)

    # Results are returned as dict with BIDS-style keys "atlas-{name}_desc-{Source}" (PascalCase)
    atlas_results = result.results["ParcelAggregation"]
    assert "atlas-Schaefer100_desc-MaskImg" in atlas_results

    # Get the ParcelData for this atlas using descriptive key
    roi_result = atlas_results["atlas-Schaefer100_desc-MaskImg"]
    results_dict = roi_result.get_data()

    # Should contain ROI-level values
    # Format: {"Schaefer2018_100Parcels7Networks_7Networks_LH_Vis_1": 0.523, ...}
    assert len(results_dict) > 0
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))


def test_atlas_aggregation_handles_multiple_atlases(synthetic_mask_img):
    """Test that ParcelAggregation can process multiple atlases from registry."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Use two bundled atlases
    analysis = ParcelAggregation(
        parcel_names=["Schaefer2018_100Parcels7Networks", "Schaefer2018_200Parcels7Networks"],
    )
    result = analysis.run(mask_data)

    # Results are returned as dict with BIDS-style keys per atlas
    atlas_results = result.results["ParcelAggregation"]
    assert "atlas-Schaefer100_desc-MaskImg" in atlas_results
    assert "atlas-Schaefer200_desc-MaskImg" in atlas_results

    # Each atlas should have its own ParcelData with region data
    roi_100 = atlas_results["atlas-Schaefer100_desc-MaskImg"].get_data()
    roi_200 = atlas_results["atlas-Schaefer200_desc-MaskImg"].get_data()
    assert len(roi_100) > 0
    assert len(roi_200) > 0


def test_atlas_aggregation_preserves_input_immutability(synthetic_mask_img):
    """Test that run() does not modify the input MaskData."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_results = mask_data.results.copy()

    analysis = ParcelAggregation(parcel_names=["Schaefer2018_100Parcels7Networks"])
    result = analysis.run(mask_data)

    # Input should not be modified
    assert mask_data.results == original_results
    assert "ParcelAggregation" not in mask_data.results

    # Result should be different object
    assert result is not mask_data


def test_atlas_aggregation_adds_provenance(synthetic_mask_img):
    """Test that run() adds provenance record."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_prov_len = len(mask_data.provenance)

    analysis = ParcelAggregation(parcel_names=["Schaefer2018_100Parcels7Networks"])
    result = analysis.run(mask_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "ParcelAggregation" in latest_prov["function"]


# ============================================================================
# User Story 4: Enhanced Analysis Flexibility
# ============================================================================


def test_atlas_aggregation_cross_analysis_source_syntax(synthetic_mask_img):
    """Test that ParcelAggregation accepts 'AnalysisName.result_key' syntax.

    Contract: T053 - Cross-analysis source syntax
    """
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    # Create lesion data with mock analysis results
    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock result from previous analysis
    network_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_mask_img.affine)
    mask_data._results["StructuralNetworkMapping"] = {"disconnection_map": network_map}

    # Should accept "Analysis.key" syntax
    analysis = ParcelAggregation(
        source="StructuralNetworkMapping.disconnection_map",
        aggregation="mean",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    # Should successfully run without error
    result = analysis.run(mask_data)
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_threshold_accepts_any_float(synthetic_mask_img):
    """Test that threshold accepts any float value (not restricted to 0.0-1.0).

    Contract: T054 - Flexible thresholds
    """
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Should accept negative values (e.g., z-score thresholds)
    analysis_negative = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        threshold=-2.5,
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )
    assert analysis_negative.threshold == -2.5
    result = analysis_negative.run(mask_data)
    assert result is not None

    # Should accept zero
    analysis_zero = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        threshold=0.0,
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )
    assert analysis_zero.threshold == 0.0

    # Should accept values > 1.0
    analysis_high = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        threshold=5.0,
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )
    assert analysis_high.threshold == 5.0


def test_atlas_aggregation_result_keys_include_source_context(synthetic_mask_img):
    """Test that result keys include source context for traceability.

    Contract: T055 - Result key source context
    """
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock disconnection map
    disconnection_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_mask_img.affine)
    mask_data._results["StructuralNetworkMapping"] = {"disconnection_map": disconnection_map}

    # Run aggregation on disconnection map
    analysis = ParcelAggregation(
        source="StructuralNetworkMapping.disconnection_map",
        aggregation="mean",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    result = analysis.run(mask_data)
    atlas_results = result.results["ParcelAggregation"]

    # Result key should include source context in BIDS format
    # Should be "atlas-Schaefer100_desc-DisconnectionMap" (PascalCase per BIDS)
    result_keys = list(atlas_results.keys())
    assert len(result_keys) > 0

    # At least one key should reference the source (PascalCase format)
    has_source_context = any("DisconnectionMap" in key or "disconnection" in key.lower() for key in result_keys)
    assert has_source_context, f"Expected source context in keys, got: {result_keys}"
