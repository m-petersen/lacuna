"""Regression test: _find_brain_mask must not return a subject's 4D functional
series (named '*_finalmask.nii.gz') as the brain mask."""

import nibabel as nib
import numpy as np

from lacuna.io.fetch import _find_brain_mask


def test_find_brain_mask_skips_functional_finalmask(tmp_path):
    # GSP1000 subject functional file matches the '*mask*' glob but is 4D.
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10, 5), np.float32), np.eye(4)),
        tmp_path / "sub-01_bld001_rest_finalmask.nii.gz",
    )
    # The actual 3D brain mask.
    nib.save(
        nib.Nifti1Image(np.ones((10, 10, 10), np.uint8), np.eye(4)),
        tmp_path / "brain_mask.nii.gz",
    )

    result = _find_brain_mask(tmp_path)
    assert result.name == "brain_mask.nii.gz"
    assert nib.load(result).ndim == 3
