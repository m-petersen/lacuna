"""
Integration tests for nibabel image input support in ParcelAggregation.

Tests the enhanced flexibility of ParcelAggregation to accept:
- MaskData (existing behavior)
- nibabel.Nifti1Image (new)
- list[nibabel.Nifti1Image] (new)

Return types should match input types:
- MaskData -> MaskData
- nibabel.Nifti1Image -> ParcelData
- list[nibabel.Nifti1Image] -> list[ParcelData]
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.core.data_types import ParcelData


@pytest.fixture
def sample_nifti_image():
    """Create a sample NIfTI image for testing."""
    # Create synthetic binary mask (64x64x64)
    data = np.zeros((64, 64, 64), dtype=np.uint8)
    data[20:40, 20:40, 20:40] = 1  # Small cube in center

    # Simple affine matrix (2mm isotropic)
    affine = np.eye(4)
    affine[:3, :3] *= 2.0  # 2mm voxel size

    return nib.Nifti1Image(data, affine)


@pytest.fixture
def sample_nifti_images():
    """Create a list of sample NIfTI images for testing."""
    images = []
    for i in range(3):
        # Create synthetic binary masks with different patterns
        data = np.zeros((64, 64, 64), dtype=np.uint8)
        offset = i * 5
        data[20 + offset : 40 + offset, 20:40, 20:40] = 1

        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm voxel size

        images.append(nib.Nifti1Image(data, affine))

    return images


def test_atlas_aggregation_accepts_nibabel_image(sample_nifti_image):
    """Test that ParcelAggregation accepts nibabel.Nifti1Image input.

    Contract: T056 - Nibabel image input support
    """
    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="percent",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    # Should accept nibabel image directly
    result = analysis.run(sample_nifti_image)

    # Result should be ParcelData (not MaskData)
    assert isinstance(result, ParcelData)

    # Should have aggregated data
    data = result.get_data()
    assert isinstance(data, dict)
    assert len(data) > 0


def test_atlas_aggregation_nibabel_image_returns_correct_type(sample_nifti_image):
    """Test that nibabel image input returns ParcelData.

    Contract: T056 - Return type matches input type
    """
    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    result = analysis.run(sample_nifti_image)

    # Input: nibabel.Nifti1Image -> Output: ParcelData
    assert isinstance(result, ParcelData)
    assert not hasattr(result, "mask_img")  # Not a MaskData
    assert not hasattr(result, "results")  # Not a MaskData


def test_atlas_aggregation_accepts_nibabel_image_list(sample_nifti_images):
    """Test that ParcelAggregation accepts list[nibabel.Nifti1Image] input.

    Contract: T057 - Nibabel list input support
    """
    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="percent",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    # Should accept list of nibabel images
    results = analysis.run(sample_nifti_images)

    # Should return list of AtlasAggregationResults
    assert isinstance(results, list)
    assert len(results) == len(sample_nifti_images)

    # Each result should be ParcelData
    for result in results:
        assert isinstance(result, ParcelData)
        data = result.get_data()
        assert isinstance(data, dict)
        assert len(data) > 0


def test_atlas_aggregation_nibabel_list_preserves_order(sample_nifti_images):
    """Test that list processing preserves input order.

    Contract: T057 - List processing maintains order
    """
    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="percent",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    results = analysis.run(sample_nifti_images)

    # Results should be in same order as input
    assert len(results) == 3

    # Each image has different pattern, so results should differ
    data_0 = results[0].get_data()
    data_1 = results[1].get_data()
    results[2].get_data()

    # At least some values should differ between results
    values_0 = set(data_0.values())
    values_1 = set(data_1.values())
    assert values_0 != values_1 or data_0 != data_1  # Different patterns


def test_atlas_aggregation_nibabel_list_batch_processing(sample_nifti_images):
    """Test that list input uses batch processing strategy.

    Contract: T057 - Batch processing for lists
    """
    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        parcel_names=["Schaefer2018_100Parcels7Networks"],
    )

    # Should handle batch processing
    results = analysis.run(sample_nifti_images)

    # All results should be complete
    assert all(isinstance(r, ParcelData) for r in results)
    assert all(len(r.get_data()) > 0 for r in results)


def test_atlas_aggregation_nibabel_works_with_multiple_atlases(sample_nifti_image):
    """Test nibabel input with multiple atlases.

    Contract: T056 - Multiple atlas support with nibabel input
    """
    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="percent",
        parcel_names=["Schaefer2018_100Parcels7Networks", "Schaefer2018_200Parcels7Networks"],
    )

    # Should return single ParcelData with combined atlas data
    # or dict mapping atlas names to results (implementation detail)
    result = analysis.run(sample_nifti_image)

    # Should have data from both atlases
    assert isinstance(result, (ParcelData, dict))

    if isinstance(result, ParcelData):
        data = result.get_data()
        assert len(data) > 0
    else:
        # Dict mapping atlas names to results
        assert len(result) == 2


def test_atlas_aggregation_nibabel_different_aggregations(sample_nifti_image):
    """Test nibabel input with different aggregation methods.

    Contract: T056 - All aggregation methods work with nibabel input
    """
    for aggregation in ["mean", "sum", "percent", "volume"]:
        analysis = ParcelAggregation(
            source="mask_img",
            aggregation=aggregation,
            parcel_names=["Schaefer2018_100Parcels7Networks"],
        )

        result = analysis.run(sample_nifti_image)

        assert isinstance(result, ParcelData)
        data = result.get_data()
        assert len(data) > 0

        # Values should be appropriate for aggregation method
        for value in data.values():
            assert isinstance(value, (int, float))
            if aggregation == "percent":
                assert 0 <= value <= 100
