"""
Integration tests for end-to-end result workflow.

T021: Tests the complete workflow from running analysis to accessing results
via both dictionary and attribute syntax.
"""

import pytest


@pytest.mark.integration
def test_end_to_end_result_access_workflow(synthetic_mask_img):
    """Test complete workflow: run analysis → access via dict → access via attribute."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation

    # Create mask data
    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Run atlas aggregation (which should generate per-atlas results)
    # Use parcellation names that are registered in the system
    analysis = ParcelAggregation(
        parcel_names=["Schaefer2018_100Parcels7Networks", "TianSubcortex_3TS1"]
    )
    result = analysis.run(mask_data)

    # Dictionary access: results['ParcelAggregation']
    assert "ParcelAggregation" in result.results
    atlas_results = result.results["ParcelAggregation"]
    assert isinstance(atlas_results, dict)

    # Attribute access: result.ParcelAggregation (should return dict)
    attr_results = result.ParcelAggregation
    assert attr_results == atlas_results

    # Verify parcel results are present
    # Result keys use BIDS-style format
    assert len(atlas_results) > 0


@pytest.mark.integration
def test_multiple_analyses_result_access(synthetic_mask_img):
    """Test accessing results from multiple sequential analyses."""
    from lacuna import MaskData
    from lacuna.analysis.parcel_aggregation import ParcelAggregation
    from lacuna.analysis.regional_damage import RegionalDamage

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Run two different analyses using registered parcellation
    result = ParcelAggregation(parcel_names=["TianSubcortex_3TS1"]).run(mask_data)
    result = RegionalDamage(parcel_names=["TianSubcortex_3TS1"]).run(result)

    # Both analyses should have results
    assert "ParcelAggregation" in result.results
    assert "RegionalDamage" in result.results

    # Attribute access for both
    atlas_agg_results = result.ParcelAggregation
    regional_damage_results = result.RegionalDamage

    assert isinstance(atlas_agg_results, dict)
    assert isinstance(regional_damage_results, dict)

    # Results should be non-empty
    assert len(atlas_agg_results) > 0
    assert len(regional_damage_results) > 0


@pytest.mark.integration
def test_result_attribute_error_helpful_message(synthetic_mask_img):
    """Test that accessing non-existent analysis gives helpful error."""
    from lacuna import MaskData

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Try to access analysis that hasn't been run
    with pytest.raises(AttributeError) as exc_info:
        _ = mask_data.NonExistentAnalysis

    error_msg = str(exc_info.value)
    assert "NonExistentAnalysis" in error_msg
    assert "results" in error_msg.lower()

    # Should suggest available analyses or indicate none have been run
    assert len(mask_data.results) == 0  # No analyses run yet
