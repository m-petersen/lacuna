"""Regression: get_image_space(declared_space=...) must not discard the
header-detected resolution. Previously `declared_resolution or 2` forced a 1mm
image (declared only by space) to 2mm with the wrong reference affine."""

import nibabel as nib
import numpy as np

from lacuna.core.spaces import REFERENCE_AFFINES, get_image_space


def _img(space, res):
    return nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), REFERENCE_AFFINES[(space, res)])


def test_declared_space_keeps_detected_1mm_resolution():
    img = _img("MNI152NLin6Asym", 1)  # genuine 1mm affine
    cs = get_image_space(img, declared_space="MNI152NLin6Asym")  # no declared_resolution
    assert cs.resolution == 1  # was 2 before the fix
    np.testing.assert_array_equal(cs.reference_affine, REFERENCE_AFFINES[("MNI152NLin6Asym", 1)])


def test_explicit_declared_resolution_wins():
    img = _img("MNI152NLin6Asym", 1)
    cs = get_image_space(img, declared_space="MNI152NLin6Asym", declared_resolution=1)
    assert cs.resolution == 1


def test_no_header_detection_falls_back_to_2mm():
    # An affine that matches no reference grid -> header detection fails.
    img = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.diag([3.0, 3.0, 3.0, 1.0]))
    cs = get_image_space(img, declared_space="MNI152NLin6Asym")
    assert cs.resolution == 2  # last-resort default when nothing is detected


def test_detection_only_unaffected():
    img = _img("MNI152NLin6Asym", 1)
    cs = get_image_space(img)  # no declaration at all
    assert cs.identifier == "MNI152NLin6Asym" and cs.resolution == 1
