"""Regression: AFNM must reject a lesion whose space differs from the atlas
space. It resamples the atlas to the lesion by affine only (no nonlinear warp
between MNI variants), so a mismatch silently corrupts m @ C."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from lacuna.analysis import AcceleratedFunctionalNetworkMapping
from lacuna.assets.parcellations import list_parcellations
from lacuna.core.subject_data import SubjectData

AFF2MM = np.diag([2.0, 2.0, 2.0, 1.0])


def _tiny_matrix(tmp_path):
    p = tmp_path / "m.tsv"
    pd.DataFrame(np.eye(2), index=["A", "B"], columns=["A", "B"]).to_csv(p, sep="\t")
    return p


def _lesion(space):
    img = nib.Nifti1Image(np.array([[[0, 1]]], np.uint8), AFF2MM)
    return SubjectData(mask_img=img, space=space, resolution=2)


def _atlas_and_spaces():
    parc = list_parcellations()[0]
    atlas_space = parc.space
    other = "MNI152NLin2009cAsym" if atlas_space != "MNI152NLin2009cAsym" else "MNI152NLin6Asym"
    return parc.name, atlas_space, other


def test_mismatched_lesion_space_rejected(tmp_path):
    atlas, atlas_space, wrong = _atlas_and_spaces()
    afnm = AcceleratedFunctionalNetworkMapping(
        matrix_path=_tiny_matrix(tmp_path), parcel_names=[atlas], verbose=False
    )
    with pytest.raises(ValueError, match="does not match the atlas space"):
        afnm._validate_inputs(_lesion(wrong))


def test_matching_lesion_space_ok(tmp_path):
    atlas, atlas_space, _ = _atlas_and_spaces()
    afnm = AcceleratedFunctionalNetworkMapping(
        matrix_path=_tiny_matrix(tmp_path), parcel_names=[atlas], verbose=False
    )
    afnm._validate_inputs(_lesion(atlas_space))  # must not raise
