"""
Contract tests for result object types.

T022: Tests that ParcelData properly handles region labels from atlas metadata.
"""

import pytest


@pytest.mark.contract
def test_atlas_aggregation_result_with_region_labels():
    """Test that ParcelData stores and exposes region labels."""
    from lacuna.core.data_types import ParcelData

    # Create result with region labels (from atlas metadata)
    region_data = {
        "Left-Thalamus": 0.45,
        "Right-Thalamus": 0.38,
        "Left-Caudate": 0.12,
        "Right-Caudate": 0.09,
    }

    result = ParcelData(
        name="DKT",
        data=region_data,
        region_labels=["Left-Thalamus", "Right-Thalamus", "Left-Caudate", "Right-Caudate"],
    )

    # Verify labels are stored
    assert hasattr(result, "region_labels")
    assert result.region_labels is not None
    assert len(result.region_labels) == 4
    assert "Left-Thalamus" in result.region_labels

    # Verify data uses region names as keys
    assert "Left-Thalamus" in result.data
    assert result.data["Left-Thalamus"] == 0.45


@pytest.mark.contract
def test_atlas_aggregation_result_without_region_labels():
    """Test that ParcelData works without region labels (backward compat)."""
    from lacuna.core.data_types import ParcelData

    # Create result with numeric keys (old style)
    result = ParcelData(
        name="TestAtlas",
        data={0: 0.5, 1: 0.3, 2: 0.7},
    )

    # Should work fine without labels
    assert result.name == "TestAtlas"
    assert len(result.data) == 3
    assert result.data[0] == 0.5


@pytest.mark.contract
def test_atlas_aggregation_result_labels_match_data():
    """Test that region_labels count matches data entries."""
    from lacuna.core.data_types import ParcelData

    region_data = {"region1": 0.1, "region2": 0.2, "region3": 0.3}
    labels = ["region1", "region2", "region3"]

    result = ParcelData(
        name="Atlas",
        data=region_data,
        region_labels=labels,
    )

    # Labels should match data keys
    assert len(result.region_labels) == len(result.data)
    for label in result.region_labels:
        assert label in result.data
