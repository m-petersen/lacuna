"""Unit tests for ParcelAggregation return type matching.

Tests that ParcelAggregation returns appropriate types based on input:
- MaskData -> MaskData (with results attached)
- nibabel.Nifti1Image -> ParcelData
- list[nibabel.Nifti1Image] -> list[ParcelData]
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.assets.parcellations.registry import (
    register_parcellations_from_directory,
    unregister_parcellation,
)
from lacuna.core import MaskData
from lacuna.core.data_types import ParcelData


@pytest.fixture
def local_test_atlas(tmp_path):
    """Create and register a local test atlas matching sample data dimensions.

    This avoids TemplateFlow downloads for CI.
    """
    atlas_dir = tmp_path / "atlases"
    atlas_dir.mkdir()

    # Use same dimensions as test data (64, 64, 64)
    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[:3, :3] *= 2.0  # 2mm resolution

    # Create atlas with 5 regions covering test area
    atlas_data = np.zeros(shape, dtype=np.int16)
    atlas_data[15:25, 20:40, 20:40] = 1
    atlas_data[25:35, 20:40, 20:40] = 2
    atlas_data[35:45, 20:40, 20:40] = 3
    atlas_data[20:40, 15:25, 20:40] = 4
    atlas_data[20:40, 35:45, 20:40] = 5

    atlas_img = nib.Nifti1Image(atlas_data, affine)
    atlas_path = atlas_dir / "test_return_types_atlas.nii.gz"
    nib.save(atlas_img, atlas_path)

    # Create labels file
    labels_path = atlas_dir / "test_return_types_atlas_labels.txt"
    labels_path.write_text("1 Region_A\n2 Region_B\n3 Region_C\n4 Region_D\n5 Region_E\n")

    # Register the atlas
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

    yield "test_return_types_atlas"

    # Cleanup
    try:
        unregister_parcellation("test_return_types_atlas")
    except KeyError:
        pass


def test_atlas_aggregation_maskdata_returns_maskdata(local_test_atlas):
    """Test that MaskData input returns MaskData output.

    Contract: T058 - Return type matches input (MaskData)
    """
    # Create synthetic mask
    data = np.zeros((64, 64, 64), dtype=np.uint8)
    data[20:40, 20:40, 20:40] = 1
    affine = np.eye(4)
    affine[:3, :3] *= 2.0
    mask_img = nib.Nifti1Image(data, affine)

    mask_data = MaskData(
        mask_img=mask_img,
        metadata={
            "space": "MNI152NLin6Asym",
            "resolution": 2.0,
        },
    )

    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="percent",
        parcel_names=[local_test_atlas],
    )

    result = analysis.run(mask_data)

    # Input: MaskData -> Output: MaskData
    assert isinstance(result, MaskData)
    assert hasattr(result, "mask_img")
    assert hasattr(result, "results")
    assert "ParcelAggregation" in result.results


def test_atlas_aggregation_nibabel_returns_aggregation_result(local_test_atlas):
    """Test that nibabel.Nifti1Image input returns ParcelData.

    Contract: T058 - Return type matches input (nibabel)
    """
    # Create synthetic nibabel image
    data = np.zeros((64, 64, 64), dtype=np.uint8)
    data[20:40, 20:40, 20:40] = 1
    affine = np.eye(4)
    affine[:3, :3] *= 2.0
    nifti_img = nib.Nifti1Image(data, affine)

    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        parcel_names=[local_test_atlas],
    )

    result = analysis.run(nifti_img)

    # Input: nibabel.Nifti1Image -> Output: ParcelData
    assert isinstance(result, ParcelData)
    assert not isinstance(result, MaskData)


def test_atlas_aggregation_nibabel_list_returns_list(local_test_atlas):
    """Test that list[nibabel.Nifti1Image] input returns list[ParcelData].

    Contract: T058 - Return type matches input (list)
    """
    # Create list of synthetic nibabel images
    images = []
    for i in range(3):
        data = np.zeros((64, 64, 64), dtype=np.uint8)
        offset = i * 5
        data[20 + offset : 40 + offset, 20:40, 20:40] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        images.append(nib.Nifti1Image(data, affine))

    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="percent",
        parcel_names=[local_test_atlas],
    )

    results = analysis.run(images)

    # Input: list[nibabel.Nifti1Image] -> Output: list[ParcelData]
    assert isinstance(results, list)
    assert len(results) == 3
    for result in results:
        assert isinstance(result, ParcelData)
        assert not isinstance(result, MaskData)


def test_atlas_aggregation_return_types_mutually_exclusive(local_test_atlas):
    """Test that return types are distinct for each input type.

    Contract: T058 - Return types are unambiguous
    """
    # Create test data
    data = np.zeros((64, 64, 64), dtype=np.uint8)
    data[20:40, 20:40, 20:40] = 1
    affine = np.eye(4)
    affine[:3, :3] *= 2.0

    # Test with MaskData
    mask_data = MaskData(
        mask_img=nib.Nifti1Image(data, affine),
        metadata={"space": "MNI152NLin6Asym", "resolution": 2.0},
    )

    # Test with nibabel
    nifti_img = nib.Nifti1Image(data, affine)

    # Test with list
    nifti_list = [nib.Nifti1Image(data, affine)]

    analysis = ParcelAggregation(
        source="mask_img",
        aggregation="mean",
        parcel_names=[local_test_atlas],
    )

    result_maskdata = analysis.run(mask_data)
    result_nibabel = analysis.run(nifti_img)
    result_list = analysis.run(nifti_list)

    # All return types should be different
    assert type(result_maskdata) is not type(result_nibabel)
    assert type(result_nibabel) is not type(result_list)
    assert type(result_maskdata) is not type(result_list)

    # Specific type checks
    assert isinstance(result_maskdata, MaskData)
    assert isinstance(result_nibabel, ParcelData)
    assert isinstance(result_list, list)
