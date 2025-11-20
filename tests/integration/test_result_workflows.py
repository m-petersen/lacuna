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
    from lacuna.analysis.atlas_aggregation import AtlasAggregation

    # Create mask data
    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Run atlas aggregation (which should generate per-atlas results)
    analysis = AtlasAggregation(atlases=["DKT", "Schaefer2018_100Parcels_7Networks"])
    result = analysis.run(mask_data)

    # Dictionary access: results['AtlasAggregation']['atlas_DKT']
    assert "AtlasAggregation" in result.results
    atlas_results = result.results["AtlasAggregation"]
    assert isinstance(atlas_results, dict)
    assert "atlas_DKT" in atlas_results

    # Attribute access: result.AtlasAggregation (should return dict)
    attr_results = result.AtlasAggregation
    assert attr_results is atlas_results

    # Access individual atlas result
    dkt_result = atlas_results["atlas_DKT"]
    from lacuna.core.output import AtlasAggregationResult

    assert isinstance(dkt_result, AtlasAggregationResult)
    assert dkt_result.name == "DKT"
    assert isinstance(dkt_result.data, dict)


@pytest.mark.integration
def test_multiple_analyses_result_access(synthetic_mask_img):
    """Test accessing results from multiple sequential analyses."""
    from lacuna import MaskData
    from lacuna.analysis.atlas_aggregation import AtlasAggregation
    from lacuna.analysis.regional_damage import RegionalDamage

    mask_data = MaskData(
        mask_img=synthetic_mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )

    # Run two different analyses
    result = AtlasAggregation(atlases=["DKT"]).run(mask_data)
    result = RegionalDamage(atlases=["DKT"]).run(result)

    # Both analyses should have results
    assert "AtlasAggregation" in result.results
    assert "RegionalDamage" in result.results

    # Attribute access for both
    atlas_agg_results = result.AtlasAggregation
    regional_damage_results = result.RegionalDamage

    assert isinstance(atlas_agg_results, dict)
    assert isinstance(regional_damage_results, dict)

    # Each should have per-atlas results
    assert "atlas_DKT" in atlas_agg_results
    assert "atlas_DKT" in regional_damage_results


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
