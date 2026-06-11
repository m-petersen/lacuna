"""Regression test: gsp1000_to_hdf5 must reject subjects whose grid differs
from the brain mask (otherwise it silently extracts the wrong voxels)."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.io import gsp1000_to_hdf5


def _make_subject(gsp_dir, name, shape4d, affine):
    func = gsp_dir / name / "func"
    func.mkdir(parents=True)
    img = nib.Nifti1Image(np.random.randn(*shape4d).astype(np.float32), affine)
    path = func / f"{name}_bld001_rest_skip4_stc_mc_finalmask.nii.gz"
    nib.save(img, path)


def test_gsp1000_rejects_grid_mismatch(tmp_path):
    affine = np.eye(4)
    gsp_dir = tmp_path / "gsp"
    # Subject is 8x8x8 but the mask will be 10x10x10 -> mismatch.
    _make_subject(gsp_dir, "sub-01", (8, 8, 8, 5), affine)
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((10, 10, 10), np.uint8), affine), mask_path)

    with pytest.raises(ValueError, match="share the mask grid|spatial shape"):
        gsp1000_to_hdf5(gsp_dir, mask_path, tmp_path / "out", subjects_per_chunk=5)


def test_gsp1000_accepts_matching_grid(tmp_path):
    affine = np.eye(4)
    gsp_dir = tmp_path / "gsp"
    _make_subject(gsp_dir, "sub-01", (10, 10, 10, 5), affine)
    mask = np.zeros((10, 10, 10), np.uint8)
    mask[3:7, 3:7, 3:7] = 1
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    out = gsp1000_to_hdf5(gsp_dir, mask_path, tmp_path / "out", subjects_per_chunk=5)
    assert len(out) == 1 and out[0].exists()
