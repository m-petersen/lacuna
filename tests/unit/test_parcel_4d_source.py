"""Regression: _aggregate_3d_atlas must handle 3D / 4D-singleton sources and
reject a genuinely multi-volume 4D source with a clear error, instead of
crashing with `float(<array>)` (TypeError)."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.utils.logging import ConsoleLogger


def _pa():
    obj = ParcelAggregation.__new__(ParcelAggregation)
    obj.aggregation = "mean"
    obj.verbose = False
    obj.logger = ConsoleLogger(verbose=False)
    return obj


def _atlas():
    arr = np.zeros((6, 6, 6), np.int32)
    arr[1:3, 1:3, 1:3] = 1
    arr[3:5, 3:5, 3:5] = 2
    return nib.Nifti1Image(arr, np.eye(4)), {1: "A", 2: "B"}


def _source(shape):
    return nib.Nifti1Image(
        np.random.default_rng(0).standard_normal(shape).astype(np.float32), np.eye(4)
    )


def test_3d_source_ok():
    atlas, labels = _atlas()
    res = _pa()._aggregate_3d_atlas(_source((6, 6, 6)), atlas, labels, 1.0)
    assert set(res) == {"A", "B"}
    assert all(isinstance(v, float) for v in res.values())


def test_4d_singleton_source_ok():
    atlas, labels = _atlas()
    res = _pa()._aggregate_3d_atlas(_source((6, 6, 6, 1)), atlas, labels, 1.0)
    assert set(res) == {"A", "B"}


def test_4d_multivolume_source_rejected():
    atlas, labels = _atlas()
    with pytest.raises(ValueError, match="single 3D map"):
        _pa()._aggregate_3d_atlas(_source((6, 6, 6, 3)), atlas, labels, 1.0)
