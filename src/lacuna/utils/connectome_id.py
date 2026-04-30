"""Functional connectome structural fingerprinting for cache validation.

Used by the FNTF prepare/run pair to detect when an ACE cache built
against one connectome is reused with a different connectome.

A *structural* fingerprint (sorted batch shapes + mask geometry) is
preferred over content hashing: connectome HDF5s can be 100+ GB and
full hashing would dominate analysis runtime, while structural identity
catches every realistic mismatch (different connectome, different
subject count, different voxel grid).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py

from lacuna.assets.connectomes.functional_io import (
    list_connectome_batch_files,
    read_mask_info,
)

_FINGERPRINT_SCHEMA = 1


def compute_functional_connectome_fingerprint(
    connectome_path: str | Path,
) -> dict[str, Any]:
    """Compute a structural identity for a functional connectome.

    Captures sorted batch filenames + their ``timeseries`` shapes plus
    the mask geometry (shape, affine, indices). No timeseries bytes are
    hashed — purely structural metadata.

    Parameters
    ----------
    connectome_path : Path
        Either a single HDF5 file or a directory of batch files.

    Returns
    -------
    dict
        ``{"schema": ..., "batches": [{name, shape}, ...],
            "mask_shape": [...], "mask_affine_sha256": ...,
            "mask_indices_sha256": ..., "digest": ...}``
    """
    files = list_connectome_batch_files(Path(connectome_path))

    batch_entries: list[dict[str, Any]] = []
    for f in files:
        with h5py.File(f, "r") as hf:
            shape = tuple(int(x) for x in hf["timeseries"].shape)
        batch_entries.append({"name": f.name, "shape": list(shape)})

    mask = read_mask_info(files[0])
    mask_affine_bytes = mask["mask_affine"].astype("float64").tobytes()
    mi = mask["mask_indices"]
    mask_indices_bytes = b"".join(arr.astype("int64").tobytes() for arr in mi)

    affine_hash = hashlib.sha256(mask_affine_bytes).hexdigest()
    indices_hash = hashlib.sha256(mask_indices_bytes).hexdigest()

    h = hashlib.sha256()
    for entry in batch_entries:
        h.update(entry["name"].encode())
        h.update(repr(entry["shape"]).encode())
    h.update(repr(mask["mask_shape"]).encode())
    h.update(affine_hash.encode())
    h.update(indices_hash.encode())

    return {
        "schema": _FINGERPRINT_SCHEMA,
        "batches": batch_entries,
        "mask_shape": list(mask["mask_shape"]),
        "mask_affine_sha256": affine_hash,
        "mask_indices_sha256": indices_hash,
        "digest": h.hexdigest(),
    }


def fingerprints_match(
    expected: dict[str, Any], actual: dict[str, Any]
) -> bool:
    """Return True iff the two fingerprints describe the same connectome."""
    return expected.get("digest") == actual.get("digest")


def total_subjects(fingerprint: dict[str, Any]) -> int:
    """Sum of the first dimension across all batches in the fingerprint."""
    return sum(int(b["shape"][0]) for b in fingerprint.get("batches", []))


def n_timepoints(fingerprint: dict[str, Any]) -> int:
    """Second dimension shared by all batches; raises if batches disagree."""
    batches = fingerprint.get("batches", [])
    if not batches:
        raise ValueError("Fingerprint has no batches")
    nts = {int(b["shape"][1]) for b in batches}
    if len(nts) != 1:
        raise ValueError(
            f"Inconsistent n_timepoints across batches: {sorted(nts)}"
        )
    return next(iter(nts))
