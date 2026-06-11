"""Regression tests for several medium-severity fixes found in review:
- resolution detection collapsing 0.5mm to 0 (round/int truncation)
- apply_resampling landing on a non-canonical grid for resolution changes
- SubjectData attribute access leaking a mutable internal reference
"""

import nibabel as nib
import numpy as np

from lacuna.core.spaces import (
    REFERENCE_AFFINES,
    REFERENCE_SHAPES,
    CoordinateSpace,
    detect_space_from_filename,
)
from lacuna.core.subject_data import SubjectData

# --- resolution detection preserves 0.5mm -----------------------------------

def test_detect_resolution_from_image_preserves_half_mm():
    img = nib.Nifti1Image(np.zeros((10, 10, 10), np.uint8), np.diag([0.5, 0.5, 0.5, 1.0]))
    assert SubjectData._detect_resolution_from_image(img) == 0.5


def test_detect_resolution_from_image_integers():
    for r in (1.0, 2.0):
        img = nib.Nifti1Image(np.zeros((5, 5, 5), np.uint8), np.diag([r, r, r, 1.0]))
        assert SubjectData._detect_resolution_from_image(img) == r


def test_detect_space_from_filename_preserves_half_mm():
    assert detect_space_from_filename(
        "sub-01_space-MNI152NLin2009bAsym_res-0.5_mask.nii.gz"
    ) == ("MNI152NLin2009bAsym", 0.5)


# --- apply_resampling lands on the canonical grid ----------------------------

def test_apply_resampling_uses_canonical_grid():
    from lacuna.spatial.transform import TransformationStrategy

    src_key = ("MNI152NLin2009cAsym", 1)
    tgt_key = ("MNI152NLin2009cAsym", 2)
    img = nib.Nifti1Image(
        np.zeros(REFERENCE_SHAPES[src_key], np.float32), REFERENCE_AFFINES[src_key]
    )
    target = CoordinateSpace("MNI152NLin2009cAsym", 2, REFERENCE_AFFINES[tgt_key])

    out = TransformationStrategy().apply_resampling(img, target)
    assert out.shape[:3] == REFERENCE_SHAPES[tgt_key]
    np.testing.assert_allclose(out.affine, REFERENCE_AFFINES[tgt_key], atol=1e-6)


# --- attribute access does not leak a mutable internal reference -------------

def test_result_attribute_access_returns_copy():
    img = nib.Nifti1Image(
        (np.arange(np.prod(REFERENCE_SHAPES[("MNI152NLin6Asym", 2)])) % 2)
        .reshape(REFERENCE_SHAPES[("MNI152NLin6Asym", 2)])
        .astype(np.uint8),
        REFERENCE_AFFINES[("MNI152NLin6Asym", 2)],
    )
    sd = SubjectData(mask_img=img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})
    sd = sd.add_result("MyAnalysis", {"value": 1})

    leaked = sd.MyAnalysis  # attribute access
    leaked["value"] = 999  # mutate the returned object

    # Internal state must be unchanged (deep copy returned, not a live ref)
    assert sd.results["MyAnalysis"]["value"] == 1
    assert sd.MyAnalysis["value"] == 1
