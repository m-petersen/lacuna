"""Regression tests for graceful handling of nilearn's "No label left after
resampling" error (raised by nilearn >= 0.14).

When the atlas, resampled to the data grid, retains no regions — i.e. the data
grid overlaps none of the atlas regions — aggregation must report zero for every
region rather than propagating nilearn's ValueError. These tests force the
condition with a mock, so they validate the behavior on any nilearn version
(older nilearn tolerates the empty resample and never raises).
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis import parcel_aggregation as pa
from lacuna.analysis.parcel_aggregation import ParcelAggregation


def _atlas_and_source():
    """A small 2-region atlas and a source on the same grid."""
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    atlas = np.zeros((10, 10, 10), dtype=np.int16)
    atlas[2:5, 2:5, 2:5] = 1
    atlas[5:8, 5:8, 5:8] = 2
    atlas_img = nib.Nifti1Image(atlas, affine)
    source = np.zeros((10, 10, 10), dtype=np.float32)
    source[3, 3, 3] = 1.0
    source_img = nib.Nifti1Image(source, affine)
    labels = {1: "RegionA", 2: "RegionB"}
    return source_img, atlas_img, labels


def test_no_label_left_returns_zero_for_all_regions(monkeypatch):
    """nilearn raising 'No label left after resampling' -> zeros, no crash."""

    def _raise_no_label(self, *args, **kwargs):
        raise ValueError("No label left after resampling the labels image.")

    monkeypatch.setattr(pa.NiftiLabelsMasker, "fit_transform", _raise_no_label)

    agg = ParcelAggregation(aggregation="percent")
    source_img, atlas_img, labels = _atlas_and_source()

    result = agg._aggregate_3d_atlas(
        source_img, atlas_img, labels, voxel_volume_mm3=8.0
    )

    assert result == {"RegionA": 0.0, "RegionB": 0.0}


def test_unrelated_valueerror_still_propagates(monkeypatch):
    """A different ValueError must NOT be swallowed by the zero fallback."""

    def _raise_other(self, *args, **kwargs):
        raise ValueError("some unrelated masker failure")

    monkeypatch.setattr(pa.NiftiLabelsMasker, "fit_transform", _raise_other)

    agg = ParcelAggregation(aggregation="percent")
    source_img, atlas_img, labels = _atlas_and_source()

    with pytest.raises(ValueError, match="unrelated masker failure"):
        agg._aggregate_3d_atlas(source_img, atlas_img, labels, voxel_volume_mm3=8.0)
