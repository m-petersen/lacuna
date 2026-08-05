"""Regression test: vectorized batch mode must transform inputs to the
connectome space, exactly like the single-subject path.

Previously ``run_batch`` validated the *raw* inputs against the connectome
space and raised ``ValidationError`` for anything not already in that space,
while ``run()`` transformed first and succeeded — an inconsistent, breaking
divergence. ``run_batch`` now runs the same ``_prepare_input`` step (canonicalize
to RAS+, then transform to ``TARGET_SPACE``) before validating.
"""

import tempfile
from pathlib import Path

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

SHAPE = (8, 8, 8)
AFFINE = np.array([[2.0, 0, 0, -8], [0, 2.0, 0, -8], [0, 0, 2.0, -8], [0, 0, 0, 1]])


@pytest.fixture
def connectome_nlin6():
    """A small connectome registered in MNI152NLin6Asym."""
    rng = np.random.default_rng(0)
    mask_indices = np.where(np.ones(SHAPE, dtype=bool))
    n_vox = mask_indices[0].size
    ts = rng.standard_normal((4, 25, n_vox)).astype(np.float32)

    tmpdir = Path(tempfile.mkdtemp())
    path = tmpdir / "c.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries", data=ts)
        f.create_dataset("mask_indices", data=np.vstack(mask_indices).T)
        f.create_dataset("mask_affine", data=AFFINE)
        f.attrs["mask_shape"] = SHAPE

    register_functional_connectome(
        name="test_batch_space",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=path,
        n_subjects=4,
        description="Test",
    )
    try:
        yield
    finally:
        unregister_functional_connectome("test_batch_space")


def _lesion(space):
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[1:3, 3:5, 3:5] = 1
    return SubjectData(mask_img=nib.Nifti1Image(data, AFFINE), space=space, resolution=2.0)


def test_run_batch_transforms_foreign_space_input(connectome_nlin6, monkeypatch):
    """A batch input in a different space than the connectome must be
    transformed (not rejected). We simulate the warp so the test stays fast and
    asset-free; the point under test is that batch mode calls the transform step
    before validating, which it previously skipped."""
    fnm = FunctionalNetworkMapping(connectome_name="test_batch_space", method="boes", verbose=False)

    seen_spaces = []

    def fake_ensure(mask_data):
        seen_spaces.append(mask_data.space)
        # Simulate a successful warp onto the connectome grid/space.
        return SubjectData(
            mask_img=nib.Nifti1Image(mask_data.mask_img.get_fdata().astype(np.uint8), AFFINE),
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata=mask_data.metadata,
        )

    monkeypatch.setattr(fnm, "_ensure_target_space", fake_ensure)

    # Input is in 2009c; connectome is NLin6. Previously this raised
    # ValidationError before any transform was attempted.
    (result,) = fnm.run_batch([_lesion("MNI152NLin2009cAsym")])

    # The transform step was invoked with the foreign-space input...
    assert "MNI152NLin2009cAsym" in seen_spaces
    # ...and a real result was produced instead of a ValidationError.
    assert "FunctionalNetworkMapping" in result.results
    assert "rmap" in result.results["FunctionalNetworkMapping"]


def test_run_batch_same_space_still_works(connectome_nlin6):
    """Sanity: an input already in the connectome space still processes."""
    fnm = FunctionalNetworkMapping(connectome_name="test_batch_space", method="boes", verbose=False)
    (result,) = fnm.run_batch([_lesion("MNI152NLin6Asym")])
    assert "rmap" in result.results["FunctionalNetworkMapping"]
