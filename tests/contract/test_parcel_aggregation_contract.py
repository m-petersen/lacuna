"""
Contract tests for ParcelAggregation analysis class.

Tests the interface and behavior requirements for composable ROI-level
aggregation of voxel-level maps following the BaseAnalysis contract.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.parcellations.registry import (
    register_parcellations_from_directory,
    unregister_parcellation,
)


@pytest.fixture
def local_test_atlas(tmp_path):
    """Create and register a local test atlas for ParcelAggregation tests.

    This avoids TemplateFlow downloads for CI.
    """
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Use same dimensions as synthetic_mask_img (91, 109, 91) or (64, 64, 64)
    # Use 64x64x64 to match synthetic_mask_img in conftest
    shape = (64, 64, 64)
    affine = np.eye(4)

    # Create atlas with 5 regions
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[10:20, 20:40, 20:40] = 1
    atlas_data[20:30, 20:40, 20:40] = 2
    atlas_data[30:40, 20:40, 20:40] = 3
    atlas_data[40:50, 20:40, 20:40] = 4
    atlas_data[20:40, 40:50, 20:40] = 5

    atlas_img = nib.Nifti1Image(atlas_data, affine)
    atlas_path = atlas_dir / "test_parc_contract_atlas.nii.gz"
    nib.save(atlas_img, atlas_path)

    # Create labels file
    labels_path = atlas_dir / "test_parc_contract_atlas_labels.txt"
    labels_path.write_text("1 Region_A\\n2 Region_B\\n3 Region_C\\n4 Region_D\\n5 Region_E\\n")

    # Register the atlas
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    yield "test_parc_contract_atlas"

    # Cleanup
    try:
        unregister_parcellation("test_parc_contract_atlas")
    except KeyError:
        pass


@pytest.fixture
def local_test_atlas_pair(tmp_path):
    """Create two local test atlases for multi-atlas tests.

    This avoids TemplateFlow downloads for CI.
    """
    atlas_dir = tmp_path / "atlases_pair"
    atlas_dir.mkdir()

    shape = (64, 64, 64)
    affine = np.eye(4)

    # Atlas 1: 5 regions
    atlas_data_1 = np.zeros(shape, dtype=np.int16)
    atlas_data_1[10:20, 20:40, 20:40] = 1
    atlas_data_1[20:30, 20:40, 20:40] = 2
    atlas_data_1[30:40, 20:40, 20:40] = 3
    atlas_data_1[40:50, 20:40, 20:40] = 4
    atlas_data_1[20:40, 40:50, 20:40] = 5
    atlas_img_1 = nib.Nifti1Image(atlas_data_1, affine)
    nib.save(atlas_img_1, atlas_dir / "test_atlas_100.nii.gz")
    (atlas_dir / "test_atlas_100_labels.txt").write_text(
        "1 Atlas100_A\\n2 Atlas100_B\\n3 Atlas100_C\\n4 Atlas100_D\\n5 Atlas100_E\\n"
    )

    # Atlas 2: 3 regions (different parcellation)
    atlas_data_2 = np.zeros(shape, dtype=np.int16)
    atlas_data_2[10:25, 20:40, 20:40] = 1
    atlas_data_2[25:40, 20:40, 20:40] = 2
    atlas_data_2[40:55, 20:40, 20:40] = 3
    atlas_img_2 = nib.Nifti1Image(atlas_data_2, affine)
    nib.save(atlas_img_2, atlas_dir / "test_atlas_200.nii.gz")
    (atlas_dir / "test_atlas_200_labels.txt").write_text(
        "1 Atlas200_X\\n2 Atlas200_Y\\n3 Atlas200_Z\\n"
    )

    # Register both atlases
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    yield ("test_atlas_100", "test_atlas_200")

    # Cleanup
    for name in ["test_atlas_100", "test_atlas_200"]:
        try:
            unregister_parcellation(name)
        except KeyError:
            pass


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

    analysis = ParcelAggregation(source="maskimg", aggregation="mean")
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
    analysis1 = ParcelAggregation(source="maskimg")
    assert analysis1.source == "maskimg"

    # Can aggregate from previous analysis result
    analysis2 = ParcelAggregation(
        source="FunctionalNetworkMapping.network_map",
    )
    assert analysis2.source == "FunctionalNetworkMapping.network_map"


def test_atlas_aggregation_accepts_aggregation_methods():
    """Test that ParcelAggregation accepts different aggregation methods."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    for method in ["mean", "sum", "percent", "volume"]:
        analysis = ParcelAggregation(source="maskimg", aggregation=method)
        assert analysis.aggregation == method


def test_atlas_aggregation_validates_aggregation_method():
    """Test that invalid aggregation method raises ValueError."""
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    with pytest.raises(ValueError, match="aggregation"):
        ParcelAggregation(
            source="maskimg",
            aggregation="invalid_method",
        )


def test_atlas_aggregation_validates_atlas_directory(synthetic_mask_img):
    """Test that ParcelAggregation validates atlases are available in registry."""
    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation
    from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

    # Save current registry state
    saved_registry = PARCELLATION_REGISTRY.copy()

    try:
        # Clear registry
        PARCELLATION_REGISTRY.clear()

        analysis = ParcelAggregation()
        mask_data = SubjectData(
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

    from lacuna import SubjectData
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

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Try to aggregate from non-existent analysis result
    analysis = ParcelAggregation(
        source="NonExistentAnalysis.network_map",
    )

    with pytest.raises(ValueError, match="source"):
        analysis.run(mask_data)


def test_atlas_aggregation_can_chain_with_other_analyses(synthetic_mask_img, local_test_atlas):
    """Test that ParcelAggregation can access results from previous analyses."""
    import nibabel as nib
    import numpy as np

    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    # Create lesion data with mock analysis results
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock network map from previous analysis
    network_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_mask_img.affine)
    mask_data._results["MockAnalysis"] = {"network_map": network_map}

    # Should be able to aggregate the network map using local test atlas
    analysis = ParcelAggregation(
        source="MockAnalysis.network_map",
        aggregation="mean",
        parcel_names=[local_test_atlas],
    )

    result = analysis.run(mask_data)

    # Should have aggregated results
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_returns_mask_data(synthetic_mask_img, local_test_atlas):
    """Test that run() returns a SubjectData object with namespaced results."""
    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = ParcelAggregation(
        source="maskimg",
        parcel_names=[local_test_atlas],
    )
    result = analysis.run(mask_data)

    # Should return SubjectData
    assert isinstance(result, SubjectData)

    # Should have namespaced results
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_result_structure(synthetic_mask_img, local_test_atlas):
    """Test that results contain ROI-level aggregated values."""
    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    analysis = ParcelAggregation(
        source="maskimg",
        aggregation="mean",
        parcel_names=[local_test_atlas],
    )
    result = analysis.run(mask_data)

    # Results are returned as dict with BIDS-style keys
    # Format: "atlas-{parcellation}_source-InputMask"
    atlas_results = result.results["ParcelAggregation"]
    expected_key = f"atlas-{local_test_atlas}_source-InputMask"
    assert expected_key in atlas_results

    # Get the ParcelData for this atlas using descriptive key
    roi_result = atlas_results[expected_key]
    results_dict = roi_result.get_data()

    # Should contain ROI-level values
    assert len(results_dict) > 0
    for key, value in results_dict.items():
        assert isinstance(key, str)
        assert isinstance(value, (int, float))


def test_atlas_aggregation_handles_multiple_atlases(synthetic_mask_img, local_test_atlas_pair):
    """Test that ParcelAggregation can process multiple atlases from registry."""
    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    atlas_100, atlas_200 = local_test_atlas_pair

    # Use two local test atlases
    analysis = ParcelAggregation(
        parcel_names=[atlas_100, atlas_200],
    )
    result = analysis.run(mask_data)

    # Results are returned as dict with BIDS-style keys per atlas
    # Format: "atlas-{parcellation}_source-InputMask"
    atlas_results = result.results["ParcelAggregation"]
    assert f"atlas-{atlas_100}_source-InputMask" in atlas_results
    assert f"atlas-{atlas_200}_source-InputMask" in atlas_results

    # Each atlas should have its own ParcelData with region data
    roi_100 = atlas_results[f"atlas-{atlas_100}_source-InputMask"].get_data()
    roi_200 = atlas_results[f"atlas-{atlas_200}_source-InputMask"].get_data()
    assert len(roi_100) > 0
    assert len(roi_200) > 0


def test_atlas_aggregation_preserves_input_immutability(synthetic_mask_img, local_test_atlas):
    """Test that run() does not modify the input SubjectData."""
    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_results = mask_data.results.copy()

    analysis = ParcelAggregation(parcel_names=[local_test_atlas])
    result = analysis.run(mask_data)

    # Input should not be modified
    assert mask_data.results == original_results
    assert "ParcelAggregation" not in mask_data.results

    # Result should be different object
    assert result is not mask_data


def test_atlas_aggregation_adds_provenance(synthetic_mask_img, local_test_atlas):
    """Test that run() adds provenance record."""
    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    original_prov_len = len(mask_data.provenance)

    analysis = ParcelAggregation(parcel_names=[local_test_atlas])
    result = analysis.run(mask_data)

    # Should have added provenance
    assert len(result.provenance) == original_prov_len + 1

    # Latest provenance should reference the analysis
    latest_prov = result.provenance[-1]
    assert "ParcelAggregation" in latest_prov["function"]


# ============================================================================
# User Story 4: Enhanced Analysis Flexibility
# ============================================================================


def test_atlas_aggregation_cross_analysis_source_syntax(synthetic_mask_img, local_test_atlas):
    """Test that ParcelAggregation accepts 'AnalysisName.result_key' syntax.

    Contract: T053 - Cross-analysis source syntax
    """
    import nibabel as nib
    import numpy as np

    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    # Create lesion data with mock analysis results
    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock result from previous analysis
    network_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_mask_img.affine)
    mask_data._results["StructuralNetworkMapping"] = {"disconnection_map": network_map}

    # Should accept "Analysis.key" syntax
    analysis = ParcelAggregation(
        source="StructuralNetworkMapping.disconnection_map",
        aggregation="mean",
        parcel_names=[local_test_atlas],
    )

    # Should successfully run without error
    result = analysis.run(mask_data)
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_result_keys_include_source_context(synthetic_mask_img, local_test_atlas):
    """Test that result keys include source context for traceability.

    Contract: T055 - Result key source context
    """
    import nibabel as nib
    import numpy as np

    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock disconnection map
    disconnection_map = nib.Nifti1Image(np.random.randn(64, 64, 64), synthetic_mask_img.affine)
    mask_data._results["StructuralNetworkMapping"] = {"disconnection_map": disconnection_map}

    # Run aggregation on disconnection map
    analysis = ParcelAggregation(
        source="StructuralNetworkMapping.disconnection_map",
        aggregation="mean",
        parcel_names=[local_test_atlas],
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


def test_multi_source_aggregation_contract(synthetic_mask_img, local_test_atlas):
    """
    Contract: Multi-source aggregation produces separate BIDS-style keys.

    Contract: T012 - Multi-source ParcelAggregation
    """
    import nibabel as nib
    import numpy as np

    from lacuna import SubjectData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation
    from lacuna.core.keys import parse_result_key

    mask_data = SubjectData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Add mock FNM result
    correlation_map = nib.Nifti1Image(
        np.random.randn(64, 64, 64).astype(np.float32), synthetic_mask_img.affine
    )
    mask_data._results["FunctionalNetworkMapping"] = {"rmap": correlation_map}

    # Run multi-source aggregation
    analysis = ParcelAggregation(
        source=["SubjectData.maskimg", "FunctionalNetworkMapping.rmap"],
        parcel_names=[local_test_atlas],
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
    mask_keys = [k for k in result_keys if "InputMask" in k or "maskimg" in k]
    fnm_keys = [k for k in result_keys if "FunctionalNetworkMapping" in k or "rmap" in k]

    assert len(mask_keys) >= 1, f"Expected InputMask key, got keys: {result_keys}"
    assert len(fnm_keys) >= 1, f"Expected FunctionalNetworkMapping key, got keys: {result_keys}"

    # Keys should be parseable
    for key in result_keys:
        if "atlas-" in key:  # Only parse BIDS-style keys
            parsed = parse_result_key(key)
            assert "atlas" in parsed, f"Key {key} should have 'atlas' component"
            assert "source" in parsed, f"Key {key} should have 'source' component"


def test_multi_source_aggregation_dict_format(synthetic_mask_img, local_test_atlas):
    """Test that ParcelAggregation accepts dictionary format for source specification."""
    import numpy as np

    from lacuna import SubjectData
    from lacuna.analysis import ParcelAggregation
    from lacuna.core import VoxelMap

    # Create mask data
    mask_data = SubjectData(mask_img=synthetic_mask_img, space="MNI152NLin6Asym", resolution=2.0)

    # Create a VoxelMap result to aggregate
    correlation_data = np.random.randn(*synthetic_mask_img.shape).astype(np.float32)
    import nibabel as nib

    correlation_img = nib.Nifti1Image(correlation_data, synthetic_mask_img.affine)
    correlation_map = VoxelMap(
        name="rmap", data=correlation_img, space="MNI152NLin6Asym", resolution=2.0
    )
    z_map = VoxelMap(name="zmap", data=correlation_img, space="MNI152NLin6Asym", resolution=2.0)

    mask_data._results["FunctionalNetworkMapping"] = {
        "rmap": correlation_map,
        "zmap": z_map,
    }

    # NEW: Dictionary format for sources
    analysis = ParcelAggregation(
        source={"SubjectData": "maskimg", "FunctionalNetworkMapping": ["rmap", "zmap"]},
        parcel_names=[local_test_atlas],
        aggregation="mean",
    )

    # Verify sources were normalized correctly
    assert "SubjectData.maskimg" in analysis.sources
    assert "FunctionalNetworkMapping.rmap" in analysis.sources
    assert "FunctionalNetworkMapping.zmap" in analysis.sources

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
    agg = ParcelAggregation(source={"FunctionalNetworkMapping": "rmap"})
    assert agg.sources == ["FunctionalNetworkMapping.rmap"]

    # Valid list keys dict should work
    agg = ParcelAggregation(source={"FunctionalNetworkMapping": ["rmap", "zmap"]})
    assert agg.sources == [
        "FunctionalNetworkMapping.rmap",
        "FunctionalNetworkMapping.zmap",
    ]
