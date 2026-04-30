"""Shared low-level helpers for voxelwise functional connectome HDF5 files.

Factored out of :mod:`lacuna.analysis.functional_network_mapping` so the same
reader can be reused by :mod:`lacuna.prepare.parcellate`.

Each HDF5 file is expected to contain:
    - ``timeseries`` : (n_subjects, n_timepoints, n_voxels) float array
    - ``mask_indices`` : (3, n_voxels) or (n_voxels, 3) brain-mask coordinates
    - ``mask_affine`` : (4, 4) affine transformation matrix
    - attr ``mask_shape`` : (nx, ny, nz)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


def list_connectome_batch_files(path: Path) -> list[Path]:
    """Return sorted list of HDF5 batch files under ``path``.

    Parameters
    ----------
    path : Path
        Either a single ``.h5`` file or a directory containing ``*.h5`` files.

    Returns
    -------
    list[Path]
        Sorted list of HDF5 files. At least one file is guaranteed or
        ``FileNotFoundError`` is raised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Connectome path not found: {path}")
    if path.is_dir():
        h5_files = sorted(path.glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No HDF5 files found in directory: {path}")
        return h5_files
    return [path]


def read_mask_info(h5_path: Path) -> dict:
    """Read mask geometry from an HDF5 connectome file.

    Returns a dict with:
        - ``mask_indices``: tuple of three ``int64`` 1-D arrays (ix, iy, iz)
        - ``mask_affine``: (4, 4) ndarray
        - ``mask_shape``: tuple of three ints
    """
    with h5py.File(h5_path, "r") as hf:
        mi = hf["mask_indices"][:]
        if mi.shape[0] == 3:
            mask_indices = tuple(mi[i, :].astype(np.int64) for i in range(3))
        else:
            mask_indices = tuple(mi[:, i].astype(np.int64) for i in range(3))
        mask_affine = hf["mask_affine"][:]
        mask_shape = tuple(int(x) for x in hf.attrs["mask_shape"])
    return {
        "mask_indices": mask_indices,
        "mask_affine": mask_affine,
        "mask_shape": mask_shape,
    }


def iter_subject_timeseries(path: Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(subject_id, timeseries)`` per subject across all batches.

    Each ``timeseries`` is shape ``(n_timepoints, n_voxels)`` in
    connectome-mask space. Reads are per-subject so the caller pays at
    most one HDF5 chunk per yield (the GSP1000 schema chunks at
    ``(1, n_timepoints, n_voxels)`` precisely so this access pattern is
    cheap). Use this for whole-brain operations such as ACE prepare. For
    lesion-masked reads (which fancy-index only the lesion slice from
    disk), use
    :meth:`FunctionalNetworkMapping._iter_batch_lesion_timeseries`.

    Subject IDs are synthesised from the batch filename and row index
    (``f"{batch_stem}-{row_idx:04d}"``) — the GSP1000 HDF5 schema does
    not currently store explicit per-subject IDs. The ID is stable
    given the schema (sorted batch order + row order is deterministic)
    and is informational only; downstream consumers that pair subjects
    across passes must rely on iteration order, not ID.
    """
    for batch_file in list_connectome_batch_files(path):
        stem = batch_file.stem
        with h5py.File(batch_file, "r") as hf:
            ts = hf["timeseries"]
            n_subjects = ts.shape[0]
            for row_idx in range(n_subjects):
                yield f"{stem}-{row_idx:04d}", ts[row_idx]
