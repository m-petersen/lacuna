"""Regression: empty/non-overlapping FNM masks must be returned in the input
space, like non-empty results — not left on the connectome grid (which would
give a mixed-grid batch)."""

import os
import tempfile

import h5py
import nibabel as nib
import numpy as np

from lacuna.analysis import FunctionalNetworkMapping
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)
from lacuna.core.spaces import REFERENCE_AFFINES, REFERENCE_SHAPES
from lacuna.core.subject_data import SubjectData

NAME = "empty_input_space_test"


def _fnm_2mm():
    aff2 = REFERENCE_AFFINES[("MNI152NLin6Asym", 2)]
    shp2 = REFERENCE_SHAPES[("MNI152NLin6Asym", 2)]
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.h5")
    with h5py.File(p, "w") as f:
        f.create_dataset("timeseries", data=np.random.randn(3, 20, 3).astype(np.float32))
        f.create_dataset("mask_indices", data=np.array([[45, 46, 47]] * 3).T)
        f.create_dataset("mask_affine", data=aff2)
        f.attrs["mask_shape"] = shp2
    register_functional_connectome(
        name=NAME, space="MNI152NLin6Asym", resolution=2.0, data_path=p, n_subjects=3, description="t"
    )
    fnm = FunctionalNetworkMapping(connectome_name=NAME, method="boes", verbose=False)
    fnm._load_mask_info()
    return fnm, aff2, shp2


def test_empty_results_returned_in_input_space():
    fnm, aff2, shp2 = _fnm_2mm()
    try:
        # An empty mask prepared onto the connectome grid, but originating at 1mm.
        empty = SubjectData(
            mask_img=nib.Nifti1Image(np.zeros(shp2, np.uint8), aff2),
            space="MNI152NLin6Asym",
            resolution=2,
            metadata={
                "_original_input_space": "MNI152NLin6Asym",
                "_original_input_resolution": 1,
            },
        )
        # raw build is on the connectome grid...
        assert fnm._build_empty_mask_results()["rmap"].data.shape == shp2  # (91,109,91)
        # ...but the returned empty results are transformed to the 1mm input grid.
        res = fnm._empty_results_in_input_space(empty)
        assert res["rmap"].data.shape == REFERENCE_SHAPES[("MNI152NLin6Asym", 1)]  # (182,218,182)
        assert not np.any(np.asarray(res["rmap"].data.dataobj))  # still all-zero
    finally:
        unregister_functional_connectome(NAME)
