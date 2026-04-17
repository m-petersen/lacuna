"""Unit tests for lacuna.prepare.parcellate (functional branch)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from lacuna.prepare.parcellate import (
    ResolvedParcellation,
    parcellate_functional,
)


def _write_connectome(
    path: Path,
    timeseries: np.ndarray,
    mask_indices: np.ndarray,
    mask_affine: np.ndarray,
    mask_shape: tuple[int, int, int],
) -> None:
    with h5py.File(path, "w") as hf:
        hf.create_dataset("timeseries", data=timeseries)
        hf.create_dataset("mask_indices", data=mask_indices)
        hf.create_dataset("mask_affine", data=mask_affine)
        hf.attrs["mask_shape"] = np.asarray(mask_shape, dtype=np.int32)


def _make_atlas(shape: tuple[int, int, int], assignments: dict[int, list[tuple[int, int, int]]]):
    data = np.zeros(shape, dtype=np.int16)
    for region_id, voxels in assignments.items():
        for x, y, z in voxels:
            data[x, y, z] = region_id
    return nib.Nifti1Image(data, np.eye(4))


def test_parcellate_functional_two_regions_identity_correlation(tmp_path):
    """With identical timeseries in a region, parcel correlation = 1 between its voxels.

    Construct a tiny volume with 6 brain voxels mapped to 2 regions (3 each).
    Region 1 voxels all share one timeseries; region 2 voxels share another.
    The two regions' timeseries are orthogonal in the first subject and
    anti-correlated in the second. Fisher-z average should give a predictable
    off-diagonal value.
    """
    mask_shape = (4, 4, 4)
    mask_affine = np.eye(4)

    # 6 brain voxels: 3 in region 1, 3 in region 2
    coords_r1 = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    coords_r2 = [(1, 0, 0), (1, 0, 1), (1, 0, 2)]
    all_coords = coords_r1 + coords_r2
    mask_indices = np.array(all_coords).T  # (3, 6)

    atlas = _make_atlas(mask_shape, {1: coords_r1, 2: coords_r2})

    # Two subjects, 20 timepoints, 6 voxels
    rng = np.random.default_rng(0)
    n_tp = 50

    def build_subject(ts_r1: np.ndarray, ts_r2: np.ndarray) -> np.ndarray:
        out = np.zeros((n_tp, 6), dtype=np.float32)
        out[:, :3] = ts_r1[:, None]
        out[:, 3:] = ts_r2[:, None]
        return out

    # Subject 1: orthogonal signals (sine vs. cosine)
    t = np.linspace(0, 2 * np.pi, n_tp, endpoint=False)
    s1_r1 = np.sin(t).astype(np.float32)
    s1_r2 = np.cos(t).astype(np.float32)
    # Subject 2: negatively correlated
    s2_r1 = rng.standard_normal(n_tp).astype(np.float32)
    s2_r2 = -s2_r1

    ts = np.stack([build_subject(s1_r1, s1_r2), build_subject(s2_r1, s2_r2)], axis=0)

    connectome_dir = tmp_path / "conn"
    connectome_dir.mkdir()
    _write_connectome(connectome_dir / "batch0.h5", ts, mask_indices, mask_affine, mask_shape)

    parcellation = ResolvedParcellation(
        short_name="synthetic2",
        image=atlas,
        labels={1: "R1", 2: "R2"},
        space="MNI152NLin6Asym",
        source="custom",
    )

    out_dir = tmp_path / "out"
    written = parcellate_functional(
        connectome_path=connectome_dir,
        parcellations=[parcellation],
        output_dir=out_dir,
    )

    # Two outputs per parcellation: r matrix and z matrix
    assert len(written) == 2
    tsv_r = [p for p in written if "fcgroupz" not in p.name][0]
    tsv_z = [p for p in written if "fcgroupz" in p.name][0]
    assert tsv_r.exists()
    assert tsv_z.exists()

    df = pd.read_csv(tsv_r, sep="\t", index_col=0)
    assert list(df.columns) == ["R1", "R2"]
    assert list(df.index) == ["R1", "R2"]
    assert df.shape == (2, 2)

    # Diagonal must be 1
    np.testing.assert_allclose(np.diag(df.values), [1.0, 1.0], atol=1e-5)
    # Matrix symmetric
    np.testing.assert_allclose(df.values, df.values.T, atol=1e-5)

    # Compute expected off-diagonal by hand: per-subject corr, Fisher-z mean
    r1 = np.corrcoef(s1_r1, s1_r2)[0, 1]
    r2 = np.corrcoef(s2_r1, s2_r2)[0, 1]
    expected = np.tanh(
        (
            np.arctanh(np.clip(r1, -0.999999, 0.999999))
            + np.arctanh(np.clip(r2, -0.999999, 0.999999))
        )
        / 2
    )
    np.testing.assert_allclose(df.values[0, 1], expected, atol=1e-4)

    # z matrix: diagonal should be 0, off-diagonal = mean Fisher z
    # Looser tolerance than the r-matrix check because float32 arithmetic in
    # _per_subject_correlation diverges from float64 np.corrcoef, and arctanh
    # amplifies small differences at high |r| values.
    df_z = pd.read_csv(tsv_z, sep="\t", index_col=0)
    np.testing.assert_allclose(np.diag(df_z.values), [0.0, 0.0], atol=1e-5)
    expected_z = (
        np.arctanh(np.clip(r1, -0.999999, 0.999999)) + np.arctanh(np.clip(r2, -0.999999, 0.999999))
    ) / 2
    np.testing.assert_allclose(df_z.values[0, 1], expected_z, atol=5e-3)

    # Sidecar present and has expected metadata
    sidecar = json.loads(tsv_r.with_suffix(".json").read_text())
    assert sidecar["MatrixType"] == "functional"
    assert sidecar["ValueType"] == "pearson_r"
    assert sidecar["Shape"] == [2, 2]
    assert sidecar["RegionLabels"] == ["R1", "R2"]
    assert sidecar["Metadata"]["modality"] == "functional"
    assert sidecar["Metadata"]["n_subjects"] == 2

    sidecar_z = json.loads(tsv_z.with_suffix(".json").read_text())
    assert sidecar_z["ValueType"] == "fisher_z"


def test_parcellate_functional_multiple_batches_accumulate(tmp_path):
    """Splitting one dataset across two HDF5 batches must give the same matrix."""
    mask_shape = (4, 4, 4)
    mask_affine = np.eye(4)
    coords_r1 = [(0, 0, 0), (0, 0, 1)]
    coords_r2 = [(1, 0, 0), (1, 0, 1)]
    mask_indices = np.array(coords_r1 + coords_r2).T

    atlas = _make_atlas(mask_shape, {1: coords_r1, 2: coords_r2})

    rng = np.random.default_rng(123)
    n_tp = 40
    n_sub = 4
    ts = rng.standard_normal((n_sub, n_tp, 4)).astype(np.float32)

    # Single-file version
    d_single = tmp_path / "single"
    d_single.mkdir()
    _write_connectome(d_single / "all.h5", ts, mask_indices, mask_affine, mask_shape)

    # Two-batch version
    d_batched = tmp_path / "batched"
    d_batched.mkdir()
    _write_connectome(d_batched / "a.h5", ts[:2], mask_indices, mask_affine, mask_shape)
    _write_connectome(d_batched / "b.h5", ts[2:], mask_indices, mask_affine, mask_shape)

    parc = ResolvedParcellation(
        short_name="tiny",
        image=atlas,
        labels={1: "A", 2: "B"},
        space=None,
        source="custom",
    )

    out_single = tmp_path / "out_single"
    out_batched = tmp_path / "out_batched"
    parcellate_functional(d_single, [parc], out_single)
    parcellate_functional(d_batched, [parc], out_batched)

    r_glob = "*fcgroupr_connmatrix.tsv"
    m1 = pd.read_csv(next(out_single.glob(r_glob)), sep="\t", index_col=0).values
    m2 = pd.read_csv(next(out_batched.glob(r_glob)), sep="\t", index_col=0).values
    np.testing.assert_allclose(m1, m2, atol=1e-6)

    z_glob = "*fcgroupz_connmatrix.tsv"
    z1 = pd.read_csv(next(out_single.glob(z_glob)), sep="\t", index_col=0).values
    z2 = pd.read_csv(next(out_batched.glob(z_glob)), sep="\t", index_col=0).values
    np.testing.assert_allclose(z1, z2, atol=1e-6)


def test_parcellate_functional_refuses_overwrite(tmp_path):
    mask_shape = (2, 2, 2)
    mask_affine = np.eye(4)
    coords = [(0, 0, 0), (1, 0, 0)]
    mask_indices = np.array(coords).T
    atlas = _make_atlas(mask_shape, {1: [coords[0]], 2: [coords[1]]})

    ts = np.random.default_rng(0).standard_normal((2, 10, 2)).astype(np.float32)
    d = tmp_path / "c"
    d.mkdir()
    _write_connectome(d / "a.h5", ts, mask_indices, mask_affine, mask_shape)

    parc = ResolvedParcellation(
        short_name="mini",
        image=atlas,
        labels={1: "A", 2: "B"},
        space=None,
        source="custom",
    )
    out = tmp_path / "out"
    parcellate_functional(d, [parc], out)

    with pytest.raises(FileExistsError):
        parcellate_functional(d, [parc], out)

    parcellate_functional(d, [parc], out, overwrite=True)
