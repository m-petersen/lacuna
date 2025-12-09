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

    # Results are returned as dict with BIDS-style keys
    # Format: "parc-{parcellation}_source-{SourceClass}_desc-{key}"
    atlas_results = result.results["ParcelAggregation"]
    expected_key = "parc-Schaefer2018_100Parcels7Networks_source-MaskData_desc-mask_img"
    assert expected_key in atlas_results

    # Get the ParcelData for this atlas using descriptive key
    roi_result = atlas_results[expected_key]
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
    # Format: "parc-{parcellation}_source-MaskData_desc-mask_img"
    atlas_results = result.results["ParcelAggregation"]
    assert "parc-Schaefer2018_100Parcels7Networks_source-MaskData_desc-mask_img" in atlas_results
    assert "parc-Schaefer2018_200Parcels7Networks_source-MaskData_desc-mask_img" in atlas_results

    # Each atlas should have its own ParcelData with region data
    roi_100 = atlas_results[
        "parc-Schaefer2018_100Parcels7Networks_source-MaskData_desc-mask_img"
    ].get_data()
    roi_200 = atlas_results[
        "parc-Schaefer2018_200Parcels7Networks_source-MaskData_desc-mask_img"
    ].get_data()
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
    # Should be "atlas-Schaefer100_desc-disconnection_map" (snake_case)
    result_keys = list(atlas_results.keys())
    assert len(result_keys) > 0

    # At least one key should reference the source (snake_case format)
    has_source_context = any(
        "disconnection_map" in key or "disconnection" in key.lower() for key in result_keys
    )
    assert has_source_context, f"Expected source context in keys, got: {result_keys}"


def test_multi_source_aggregation_contract(synthetic_mask_img):
    """
    Contract: Multi-source aggregation produces separate BIDS-style keys.

    Contract: T012 - Multi-source ParcelAggregation
    """
    import nibabel as nib
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation
    from lacuna.core.keys import parse_result_key

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock FNM result
    correlation_map = nib.Nifti1Image(
        np.random.randn(64, 64, 64).astype(np.float32), synthetic_mask_img.affine
    )
    mask_data._results["FunctionalNetworkMapping"] = {"correlation_map": correlation_map}

    # Run multi-source aggregation
    analysis = ParcelAggregation(
        source=["MaskData.mask_img", "FunctionalNetworkMapping.correlation_map"],
        parcel_names=["Schaefer2018_100Parcels7Networks"],
        aggregation="mean",
    )

    result = analysis.run(mask_data)
    parcel_results = result.results["ParcelAggregation"]

    # Should have results from both sources
    result_keys = list(parcel_results.keys())
    assert (
        len(result_keys) >= 2
    ), f"Expected at least 2 results, got {len(result_keys)}: {result_keys}"

    # Keys should be BIDS-style with source differentiation
    mask_keys = [k for k in result_keys if "MaskData" in k or "mask_img" in k]
    fnm_keys = [k for k in result_keys if "FunctionalNetworkMapping" in k or "correlation_map" in k]

    assert len(mask_keys) >= 1, f"Expected MaskData key, got keys: {result_keys}"
    assert len(fnm_keys) >= 1, f"Expected FunctionalNetworkMapping key, got keys: {result_keys}"

    # Keys should be parseable
    for key in result_keys:
        if "parc-" in key:  # Only parse BIDS-style keys
            parsed = parse_result_key(key)
            assert "parc" in parsed, f"Key {key} should have 'parc' component"
            assert "source" in parsed, f"Key {key} should have 'source' component"
            assert "desc" in parsed, f"Key {key} should have 'desc' component"


def test_multi_source_aggregation_dict_format(synthetic_mask_img):
    """Test that ParcelAggregation accepts dictionary format for source specification."""
    import numpy as np

    from lacuna import MaskData
    from lacuna.analysis import ParcelAggregation
    from lacuna.core import VoxelMap

    # Create mask data
    mask_data = MaskData(mask_img=synthetic_mask_img, space="MNI152NLin6Asym", resolution=2.0)

    # Create a VoxelMap result to aggregate
    correlation_data = np.random.randn(*synthetic_mask_img.shape).astype(np.float32)
    import nibabel as nib

    correlation_img = nib.Nifti1Image(correlation_data, synthetic_mask_img.affine)
    correlation_map = VoxelMap(
        name="correlation_map", data=correlation_img, space="MNI152NLin6Asym", resolution=2.0
    )
    z_map = VoxelMap(name="z_map", data=correlation_img, space="MNI152NLin6Asym", resolution=2.0)

    mask_data._results["FunctionalNetworkMapping"] = {
        "correlation_map": correlation_map,
        "z_map": z_map,
    }

    # NEW: Dictionary format for sources
    analysis = ParcelAggregation(
        source={"MaskData": "mask_img", "FunctionalNetworkMapping": ["correlation_map", "z_map"]},
        parcel_names=["Schaefer2018_100Parcels7Networks"],
        aggregation="mean",
    )

    # Verify sources were normalized correctly
    assert "MaskData.mask_img" in analysis.sources
    assert "FunctionalNetworkMapping.correlation_map" in analysis.sources
    assert "FunctionalNetworkMapping.z_map" in analysis.sources

    # Run and verify results
    result = analysis.run(mask_data)
    parcel_results = result.results["ParcelAggregation"]

    # Should have results from all three sources
    result_keys = list(parcel_results.keys())
    assert len(result_keys) >= 3, f"Expected at least 3 results, got {len(result_keys)}"


def test_source_dict_format_validation():
    """Test that dictionary source format validates input correctly."""
    import pytest

    from lacuna.analysis import ParcelAggregation

    # Empty dict should raise
    with pytest.raises(ValueError, match="empty"):
        ParcelAggregation(source={})

    # Non-string namespace should raise
    with pytest.raises(TypeError, match="namespace"):
        ParcelAggregation(source={123: "key"})

    # Non-string value should raise
    with pytest.raises(TypeError, match="str or list"):
        ParcelAggregation(source={"Namespace": 123})

    # Empty list value should raise
    with pytest.raises(ValueError, match="empty"):
        ParcelAggregation(source={"Namespace": []})

    # Non-string item in list should raise
    with pytest.raises(TypeError, match="key"):
        ParcelAggregation(source={"Namespace": ["valid", 123]})

    # Valid single key dict should work
    agg = ParcelAggregation(source={"FunctionalNetworkMapping": "correlation_map"})
    assert agg.sources == ["FunctionalNetworkMapping.correlation_map"]

    # Valid list keys dict should work
    agg = ParcelAggregation(source={"FunctionalNetworkMapping": ["correlation_map", "z_map"]})
    assert agg.sources == [
        "FunctionalNetworkMapping.correlation_map",
        "FunctionalNetworkMapping.z_map",
    ]
