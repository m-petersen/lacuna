"""Regression: SNM's mask_streamline_count must be the true streamline count
(from the filtered tractogram), not the TDI voxel-sum (~count x mean length)."""

import nibabel as nib
import numpy as np
from nibabel.streamlines import Tractogram

from lacuna.analysis.structural_network_mapping import _streamline_count


def test_streamline_count_is_a_count(tmp_path):
    # 7 streamlines of varying length
    streams = [
        np.random.default_rng(i).standard_normal((5 + 3 * i, 3)).astype(np.float32)
        for i in range(7)
    ]
    p = tmp_path / "mask_streamlines.tck"
    nib.streamlines.save(Tractogram(streams, affine_to_rasmm=np.eye(4)), str(p))

    assert _streamline_count(p) == 7
    # It must NOT be a length-weighted total (the essence of the old TDI-sum bug).
    total_points = sum(len(s) for s in streams)
    assert _streamline_count(p) != total_points


def test_streamline_count_empty(tmp_path):
    p = tmp_path / "empty.tck"
    nib.streamlines.save(Tractogram([], affine_to_rasmm=np.eye(4)), str(p))
    assert _streamline_count(p) == 0
