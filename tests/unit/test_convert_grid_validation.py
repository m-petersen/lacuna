"""Regression test: gsp1000_to_hdf5 must reject subjects whose grid differs
from the brain mask (otherwise it silently extracts the wrong voxels)."""

import h5py
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


def test_gsp1000_accepts_radiological_subject_without_lr_flip(tmp_path):
    """A subject stored in FSL's radiological orientation (x-flipped vs. the
    templateflow RAS+ mask) must be accepted and extract the anatomically
    correct voxels — not the left-right mirrored ones."""
    nx = ny = nz = 10
    nt = 4
    ras_affine = np.array([[2.0, 0, 0, -10], [0, 2.0, 0, -10], [0, 0, 2.0, -10], [0, 0, 0, 1]])

    # RAS+ content where each voxel's value encodes its (i, j, k) index, so a
    # left-right flip would be detectable in the extracted value.
    ii, jj, kk = np.indices((nx, ny, nz))
    voxel_code = (ii * 100 + jj * 10 + kk).astype(np.float32)
    data_ras = np.repeat(voxel_code[..., None], nt, axis=3)

    # Store the SAME world content radiologically: flip the data along x and
    # negate the x-axis of the affine (origin shifts so world coords match).
    data_rad = data_ras[::-1, :, :, :].copy()
    rad_affine = ras_affine.copy()
    rad_affine[0, 0] = -2.0
    rad_affine[0, 3] = 2.0 * (nx - 1) - 10.0

    gsp_dir = tmp_path / "gsp"
    func = gsp_dir / "sub-01" / "func"
    func.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(data_rad, rad_affine),
        func / "sub-01_bld001_rest_skip4_stc_mc_finalmask.nii.gz",
    )

    # Mask (RAS+) selects a single, x-asymmetric in-brain voxel.
    mask = np.zeros((nx, ny, nz), np.uint8)
    mask[1, 5, 5] = 1
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, ras_affine), mask_path)

    out = gsp1000_to_hdf5(gsp_dir, mask_path, tmp_path / "out", subjects_per_chunk=5)
    assert len(out) == 1 and out[0].exists()

    # The extracted timeseries must equal the RAS+ value at (1, 5, 5) = 155,
    # not the mirrored voxel (8, 5, 5) = 855.
    with h5py.File(out[0], "r") as hf:
        ts = hf["timeseries"][0]  # (n_timepoints, n_voxels)
    assert ts.shape == (nt, 1)
    np.testing.assert_allclose(ts[:, 0], 155.0)
