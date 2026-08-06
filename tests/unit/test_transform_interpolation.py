"""Regression: string interpolation must work through transform_mask_data.

transform_mask_data documents that `interpolation` may be a string
('nearest' / 'linear' / 'cubic'). Previously the provenance step did
`select_interpolation(...).value`, and select_interpolation returned a passed
string unchanged, so `"nearest".value` raised AttributeError *after* the
transform had already run. select_interpolation now normalizes strings to the
InterpolationMethod enum.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
from lacuna.core.subject_data import SubjectData
from lacuna.spatial.transform import (
    InterpolationMethod,
    TransformationStrategy,
    transform_mask_data,
)


def test_select_interpolation_normalizes_string():
    s = TransformationStrategy()
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    for name, expected in [
        ("nearest", InterpolationMethod.NEAREST),
        ("linear", InterpolationMethod.LINEAR),
        ("cubic", InterpolationMethod.CUBIC),
        ("NEAREST", InterpolationMethod.NEAREST),  # case-insensitive
    ]:
        got = s.select_interpolation(img, name)
        assert got is expected
        assert got.value == expected.value  # the .value access that used to crash


def test_select_interpolation_rejects_bad_string():
    s = TransformationStrategy()
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="Invalid interpolation"):
        s.select_interpolation(img, "bogus")


def test_select_interpolation_passthrough_enum_and_autodetect():
    s = TransformationStrategy()
    binary = nib.Nifti1Image(np.array([[[0, 1]]], np.uint8), np.eye(4))
    assert s.select_interpolation(binary, InterpolationMethod.CUBIC) is InterpolationMethod.CUBIC
    assert s.select_interpolation(binary, None) is InterpolationMethod.NEAREST  # binary -> nearest


def test_transform_mask_data_accepts_string_interpolation():
    """The documented string form must survive a real transform (this is the
    exact call that raised AttributeError before the fix)."""
    aff1 = REFERENCE_AFFINES[("MNI152NLin6Asym", 1)]
    data = np.zeros((182, 218, 182), np.uint8)
    data[90:96, 108:114, 90:96] = 1  # a small blob near center
    sd = SubjectData(mask_img=nib.Nifti1Image(data, aff1), space="MNI152NLin6Asym", resolution=1)
    target = CoordinateSpace("MNI152NLin6Asym", 2, REFERENCE_AFFINES[("MNI152NLin6Asym", 2)])

    out = transform_mask_data(sd, target, interpolation="nearest")

    assert out.mask_img.shape == (91, 109, 91)  # resampled to the 2mm grid
    assert out.provenance[-1]["interpolation"] == "nearest"  # recorded correctly
