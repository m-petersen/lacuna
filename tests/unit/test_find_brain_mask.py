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


def test_find_brain_mask_derives_from_functional_data(tmp_path):
    """With no 3D mask present, the mask is derived from the functional series:
    the common non-zero support across subjects (GSP1000 is skull-stripped)."""
    affine = np.diag([2, 2, 2, 1]).astype(float)
    shape = (10, 10, 10)

    # A shared brain region all subjects cover, plus a per-subject stray voxel
    # that must be excluded by the intersection.
    brain = np.zeros(shape, bool)
    brain[3:7, 3:7, 3:7] = True

    rng = np.random.default_rng(0)
    for s in range(3):
        data = np.zeros((*shape, 5), np.float32)
        data[brain] = rng.standard_normal((int(brain.sum()), 5)).astype(np.float32) + 1.0
        data[s, s, s, :] = 1.0  # stray voxel unique to this subject
        func_dir = tmp_path / f"sub-{s:02d}" / "func"
        func_dir.mkdir(parents=True)
        nib.save(
            nib.Nifti1Image(data, affine),
            func_dir / f"sub-{s:02d}_bld001_rest_skip4_finalmask.nii.gz",
        )

    result = _find_brain_mask(tmp_path)
    derived = nib.load(result)
    assert derived.ndim == 3
    np.testing.assert_array_equal(derived.get_fdata().astype(bool), brain)
    np.testing.assert_allclose(derived.affine, affine)
