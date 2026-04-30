"""Unit tests for lacuna.assets.connectomes.functional_io."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from lacuna.assets.connectomes.functional_io import iter_subject_timeseries


def _write_synthetic_batch(
    path: Path,
    *,
    n_subjects: int,
    n_timepoints: int,
    mask_shape: tuple[int, int, int] = (5, 5, 5),
) -> None:
    """Write one HDF5 batch matching the gsp1000_to_hdf5 schema.

    Each subject's timeseries is filled with a deterministic recognisable
    constant so callers can verify per-subject content.
    """
    flat_mask = np.zeros(np.prod(mask_shape), dtype=bool)
    flat_mask[: int(np.prod(mask_shape) // 2)] = True
    n_voxels = int(flat_mask.sum())
    indices_3d = np.array(
        np.unravel_index(np.where(flat_mask)[0], mask_shape)
    )  # (3, n_voxels)

    with h5py.File(path, "w") as hf:
        ts = np.empty((n_subjects, n_timepoints, n_voxels), dtype=np.float32)
        for s in range(n_subjects):
            ts[s] = float(s + 1)  # subject 0 → 1.0, subject 1 → 2.0, ...
        hf.create_dataset("timeseries", data=ts)
        hf.create_dataset("mask_indices", data=indices_3d)
        hf.create_dataset("mask_affine", data=np.eye(4))
        hf.attrs["n_subjects"] = n_subjects
        hf.attrs["n_timepoints"] = n_timepoints
        hf.attrs["n_voxels"] = n_voxels
        hf.attrs["mask_shape"] = mask_shape


def test_iter_yields_one_timeseries_per_subject_across_batches(tmp_path):
    _write_synthetic_batch(tmp_path / "batch_0001.h5", n_subjects=3, n_timepoints=8)
    _write_synthetic_batch(tmp_path / "batch_0002.h5", n_subjects=2, n_timepoints=8)

    yielded = list(iter_subject_timeseries(tmp_path))

    assert len(yielded) == 5
    # Order is batch-then-row, alphabetical batch order
    expected_constants = [1.0, 2.0, 3.0, 1.0, 2.0]  # batch1 has 3, batch2 has 2
    for (sid, ts), constant in zip(yielded, expected_constants):
        assert isinstance(sid, str)
        assert ts.shape == (8, ts.shape[1])  # (n_timepoints, n_voxels)
        assert np.allclose(ts, constant)


def test_iter_subject_ids_are_unique_and_deterministic(tmp_path):
    _write_synthetic_batch(tmp_path / "batch_0001.h5", n_subjects=3, n_timepoints=8)
    _write_synthetic_batch(tmp_path / "batch_0002.h5", n_subjects=3, n_timepoints=8)

    ids_run1 = [sid for sid, _ in iter_subject_timeseries(tmp_path)]
    ids_run2 = [sid for sid, _ in iter_subject_timeseries(tmp_path)]

    assert len(set(ids_run1)) == len(ids_run1) == 6
    assert ids_run1 == ids_run2


def test_iter_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_subject_timeseries(tmp_path))
