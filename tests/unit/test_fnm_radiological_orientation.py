"""Regression tests for L/R orientation handling in functional network mapping.

Background
----------
The GSP1000 connectome is stored in canonical RAS+ (see ``io/convert.py``, which
runs ``nib.as_closest_canonical``). A lesion mask supplied in the *radiological*
convention (e.g. FSL's MNI152, ``srow_x < 0``) shares that grid only up to an
axis flip. The connectome-matching step previously gated resampling on shape
equality alone, so a radiological lesion with the same shape as the connectome
was matched to connectome columns by *raw voxel index* and silently L/R-mirrored.

These tests lock in:
1. ``_get_mask_voxel_indices`` matches by world coordinate, so a radiological
   lesion and its neurological twin select the *same* connectome columns.
2. ``reorient_to_affine_orientation`` restores the caller's storage convention
   losslessly (anatomy/world coordinates preserved).
"""

import h5py
import nibabel as nib
import numpy as np
import pytest

from lacuna import SubjectData
from lacuna.analysis import FunctionalNetworkMapping
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)
from lacuna.core.validation import reorient_to_affine_orientation

# A RAS+ grid with a non-symmetric origin so an L/R flip is detectable.
SHAPE = (10, 10, 10)
RAS_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -10.0],
        [0.0, 2.0, 0.0, -12.0],
        [0.0, 0.0, 2.0, -14.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
# The single "on" voxel of the test lesion, in RAS+ grid coordinates.
ON_VOXEL = (2, 3, 4)
# Column index into a full C-ordered mask over SHAPE for ON_VOXEL.
EXPECTED_COLUMN = ON_VOXEL[0] * 100 + ON_VOXEL[1] * 10 + ON_VOXEL[2]
# The mirror voxel an index-based (buggy) match would have selected.
MIRROR_COLUMN = (SHAPE[0] - 1 - ON_VOXEL[0]) * 100 + ON_VOXEL[1] * 10 + ON_VOXEL[2]


@pytest.fixture
def fnm(tmp_path):
    """A FunctionalNetworkMapping whose mask grid is a full RAS+ SHAPE grid."""
    connectome_path = tmp_path / "connectome.h5"
    with h5py.File(connectome_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(3, 20, np.prod(SHAPE)))
        f.create_dataset("mask_indices", data=np.zeros((3, np.prod(SHAPE)), dtype=int))
        f.create_dataset("mask_affine", data=RAS_AFFINE)
        f.attrs["mask_shape"] = SHAPE

    register_functional_connectome(
        name="test_radiological",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=connectome_path,
        n_subjects=3,
        description="Test",
    )
    analysis = FunctionalNetworkMapping(
        connectome_name="test_radiological", method="boes", verbose=False
    )
    # Full brain mask: every voxel is in-mask, C-ordered, so the flat column of
    # grid voxel (i, j, k) is i*100 + j*10 + k.
    full_mask = np.ones(SHAPE, dtype=bool)
    analysis._mask_info = {
        "mask_shape": SHAPE,
        "mask_indices": np.where(full_mask),
        "mask_affine": RAS_AFFINE,
    }
    try:
        yield analysis
    finally:
        unregister_functional_connectome("test_radiological")


def _neurological_lesion():
    """SubjectData with a single voxel at ON_VOXEL on the RAS+ grid."""
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[ON_VOXEL] = 1
    img = nib.Nifti1Image(data, RAS_AFFINE)
    return SubjectData(mask_img=img, space="MNI152NLin6Asym", resolution=2.0)


def _radiological_lesion():
    """Same anatomy as ``_neurological_lesion`` but stored radiologically.

    ``as_reoriented`` with an x-axis flip reverses the data along axis 0 and
    negates the affine's x column, so the world location of the "on" voxel is
    unchanged — only the storage convention differs.
    """
    neuro = _neurological_lesion().mask_img
    x_flip = np.array([[0, -1], [1, 1], [2, 1]])
    radio = neuro.as_reoriented(x_flip)
    assert radio.affine[0, 0] < 0, "expected radiological (negative x) storage"
    return SubjectData(mask_img=radio, space="MNI152NLin6Asym", resolution=2.0)


def test_radiological_and_neurological_select_same_column(fnm):
    """A radiological lesion must map to the SAME connectome column as its twin."""
    cols_neuro, _ = fnm._get_mask_voxel_indices(_neurological_lesion())
    cols_radio, _ = fnm._get_mask_voxel_indices(_radiological_lesion())

    assert list(cols_neuro) == [EXPECTED_COLUMN]
    # The core assertion: matched by world coordinate, not raw index, so no flip.
    assert sorted(cols_radio) == [EXPECTED_COLUMN]
    assert MIRROR_COLUMN not in set(cols_radio)


def test_radiological_connectome_matches_lesion_by_world(fnm):
    """The connectome grid itself may be non-RAS+ (e.g. a radiological
    mask_affine). The lesion must still be matched to the connectome voxel at
    the SAME world location, not the same array index."""
    # Radiological connectome grid over the same field of view as RAS_AFFINE
    # (x voxel 0 -> +8, decreasing), full brain mask, C-ordered.
    rad_affine = np.array(
        [
            [-2.0, 0, 0, 8.0],
            [0, 2.0, 0, RAS_AFFINE[1, 3]],
            [0, 0, 2.0, RAS_AFFINE[2, 3]],
            [0, 0, 0, 1],
        ]
    )
    fnm._mask_info = {
        "mask_shape": SHAPE,
        "mask_indices": np.where(np.ones(SHAPE, dtype=bool)),
        "mask_affine": rad_affine,
    }

    lesion = _neurological_lesion()  # RAS+ lesion, single voxel at ON_VOXEL
    lesion_world = nib.affines.apply_affine(RAS_AFFINE, ON_VOXEL)

    cols, _ = fnm._get_mask_voxel_indices(lesion)
    assert len(cols) == 1
    ijk = tuple(int(c[cols[0]]) for c in fnm._mask_info["mask_indices"])
    selected_world = nib.affines.apply_affine(rad_affine, ijk)

    np.testing.assert_allclose(selected_world, lesion_world, atol=1e-6)


def test_reorient_to_affine_orientation_restores_convention():
    """RAS+ output flipped to a radiological target adopts that storage order."""
    data = np.zeros(SHAPE, dtype=np.float32)
    data[ON_VOXEL] = 5.0
    ras_img = nib.Nifti1Image(data, RAS_AFFINE)

    radiological_target = RAS_AFFINE.copy()
    radiological_target[0, 0] = -2.0
    radiological_target[0, 3] = -RAS_AFFINE[0, 3]  # keep same field of view

    out = reorient_to_affine_orientation(ras_img, radiological_target)

    # Storage convention now radiological...
    assert out.affine[0, 0] < 0
    # ...but the anatomy is preserved: the hot voxel is at the same world point.
    src_world = nib.affines.apply_affine(ras_img.affine, ON_VOXEL)
    out_vox = np.array(np.where(out.get_fdata() == 5.0)).ravel()
    out_world = nib.affines.apply_affine(out.affine, out_vox)
    np.testing.assert_allclose(out_world, src_world, atol=1e-6)


def test_reorient_is_noop_when_orientation_matches():
    """Reorienting a RAS+ image to a RAS+ target returns the input unchanged."""
    img = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), RAS_AFFINE)
    assert reorient_to_affine_orientation(img, RAS_AFFINE) is img


# --------------------------------------------------------------------------- #
# End-to-end coverage: a full analysis run must (a) produce an anatomically
# identical map for radiological and neurological inputs, and (b) return the
# map in the caller's storage orientation.
# --------------------------------------------------------------------------- #

E2E_SHAPE = (8, 8, 8)
E2E_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, -8.0],
        [0.0, 2.0, 0.0, -8.0],
        [0.0, 0.0, 2.0, -8.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


@pytest.fixture
def e2e_connectome():
    """A small but fully consistent RAS+ connectome for real analysis runs."""
    import tempfile
    from pathlib import Path

    rng = np.random.default_rng(0)
    brain = np.ones(E2E_SHAPE, dtype=bool)
    mask_indices = np.where(brain)
    n_vox = mask_indices[0].size
    timeseries = rng.standard_normal((4, 30, n_vox)).astype(np.float32)

    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "e2e.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries", data=timeseries)
        f.create_dataset("mask_indices", data=np.vstack(mask_indices).T)
        f.create_dataset("mask_affine", data=E2E_AFFINE)
        f.attrs["mask_shape"] = E2E_SHAPE

    register_functional_connectome(
        name="test_e2e_orientation",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=path,
        n_subjects=4,
        description="Test",
    )
    try:
        yield
    finally:
        unregister_functional_connectome("test_e2e_orientation")


def _block_lesion(radiological: bool):
    """A multi-voxel lesion on the -x side; radiological twin has identical anatomy."""
    data = np.zeros(E2E_SHAPE, dtype=np.uint8)
    data[1:3, 3:5, 3:5] = 1
    img = nib.Nifti1Image(data, E2E_AFFINE)
    if radiological:
        img = img.as_reoriented(np.array([[0, -1], [1, 1], [2, 1]]))
    return SubjectData(mask_img=img, space="MNI152NLin6Asym", resolution=2.0)


def _rmap(result):
    return result.results["FunctionalNetworkMapping"]["rmap"].data


def test_end_to_end_radiological_matches_neurological(e2e_connectome):
    """A full run: radiological and neurological inputs give the same anatomy,
    and each output is returned in its own input's storage orientation."""
    neuro = FunctionalNetworkMapping(
        connectome_name="test_e2e_orientation", method="boes", verbose=False
    ).run(_block_lesion(radiological=False))
    radio = FunctionalNetworkMapping(
        connectome_name="test_e2e_orientation", method="boes", verbose=False
    ).run(_block_lesion(radiological=True))

    rmap_neuro, rmap_radio = _rmap(neuro), _rmap(radio)

    # Output orientation follows the input convention.
    assert rmap_neuro.affine[0, 0] > 0, "neurological input -> neurological output"
    assert rmap_radio.affine[0, 0] < 0, "radiological input -> radiological output"

    # Anatomically identical once both are brought to a common orientation:
    # this is what fails if the radiological seed is L/R-mirrored.
    a = nib.as_closest_canonical(rmap_neuro).get_fdata()
    b = nib.as_closest_canonical(rmap_radio).get_fdata()
    np.testing.assert_allclose(a, b, atol=1e-5)
    # Sanity: the map is not trivially symmetric (which would mask a flip).
    assert not np.allclose(a, a[::-1], atol=1e-5)


def test_batch_path_radiological_matches_neurological(e2e_connectome):
    """The vectorized run_batch path (which bypasses BaseAnalysis.run) must
    also avoid the flip and restore the input orientation."""
    fnm = FunctionalNetworkMapping(
        connectome_name="test_e2e_orientation", method="boes", verbose=False
    )
    (neuro,) = fnm.run_batch([_block_lesion(radiological=False)])
    (radio,) = fnm.run_batch([_block_lesion(radiological=True)])

    rmap_neuro, rmap_radio = _rmap(neuro), _rmap(radio)
    assert rmap_neuro.affine[0, 0] > 0
    assert rmap_radio.affine[0, 0] < 0

    a = nib.as_closest_canonical(rmap_neuro).get_fdata()
    b = nib.as_closest_canonical(rmap_radio).get_fdata()
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_batch_matches_single_subject_path(e2e_connectome):
    """Batch and single-subject paths must agree for the same radiological input."""
    single = FunctionalNetworkMapping(
        connectome_name="test_e2e_orientation", method="boes", verbose=False
    ).run(_block_lesion(radiological=True))
    (batched,) = FunctionalNetworkMapping(
        connectome_name="test_e2e_orientation", method="boes", verbose=False
    ).run_batch([_block_lesion(radiological=True)])

    a = nib.as_closest_canonical(_rmap(single)).get_fdata()
    b = nib.as_closest_canonical(_rmap(batched)).get_fdata()
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_base_run_canonicalizes_input_and_restores_orientation():
    """The shared BaseAnalysis.run() canonicalizes any input to RAS+ before
    analysis and restores the input orientation on VoxelMap outputs. This is the
    mechanism every pipeline inherits, tested here on a minimal subclass."""
    from lacuna.analysis.base import BaseAnalysis
    from lacuna.core.data_types import VoxelMap

    seen = {}

    class _Echo(BaseAnalysis):
        TARGET_SPACE = None  # adaptive: no space transform

        def _validate_inputs(self, mask_data):
            pass

        def _run_analysis(self, mask_data):
            # Record the orientation the analysis actually computes on.
            seen["analysis_affine"] = mask_data.mask_img.affine.copy()
            return {
                "map": VoxelMap(
                    name="map",
                    data=mask_data.mask_img,
                    space=mask_data.space or "MNI152NLin6Asym",
                    resolution=mask_data.resolution or 2.0,
                )
            }

    data = np.zeros(SHAPE, dtype=np.uint8)
    data[ON_VOXEL] = 1
    radio_img = nib.Nifti1Image(data, RAS_AFFINE).as_reoriented(np.array([[0, -1], [1, 1], [2, 1]]))
    assert radio_img.affine[0, 0] < 0
    sd = SubjectData(mask_img=radio_img, space="MNI152NLin6Asym", resolution=2.0)

    out = _Echo(verbose=False).run(sd)

    # Analysis computed on a canonicalized RAS+ mask...
    assert seen["analysis_affine"][0, 0] > 0
    # ...but the output was restored to the caller's radiological orientation.
    assert out.results["_Echo"]["map"].data.affine[0, 0] < 0
