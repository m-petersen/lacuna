"""Regression tests for FunctionalNetworkMapping run() vs run_batch() parity.

These cover three bugs found in review:
1. The Fisher-z infinity clamp differed between the non-batch path (+/-10) and
   the vectorized streaming path (+/-3.0), so identical input produced different
   z/r maps depending on execution mode.
2. The streaming path computed variance via the unstable E[z^2]-E[z]^2 formula
   instead of the two-pass np.std(ddof=1) used by the non-batch path.
3. Undefined correlations (zero-variance connectome voxels -> x/0) were clamped
   to a *perfect* correlation (r=+/-1) instead of 0 (no correlation).
"""

import h5py
import nibabel as nib
import numpy as np

from lacuna import SubjectData
from lacuna.analysis import FunctionalNetworkMapping
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)

AFFINE = np.array(
    [[-2.0, 0.0, 0.0, 90.0], [0.0, 2.0, 0.0, -126.0], [0.0, 0.0, 2.0, -72.0], [0.0, 0.0, 0.0, 1.0]]
)
MASK_SHAPE = (91, 109, 91)
N_VOX = 200


def _mask_indices():
    # 200 voxels at brain coords within 0-9 (overlaps the lesion below)
    return np.array(
        [np.repeat(range(10), 20), np.tile(np.repeat(range(10), 2), 10), np.tile(range(10), 20)]
    )


def _write_connectome(path, timeseries):
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries", data=timeseries.astype(np.float32))
        f.create_dataset("mask_indices", data=_mask_indices())
        f.create_dataset("mask_affine", data=AFFINE)
        f.attrs["mask_shape"] = MASK_SHAPE


def _lesion(tmp_path):
    data = np.zeros(MASK_SHAPE, dtype=np.uint8)
    data[2:7, 2:7, 2:7] = 1
    p = tmp_path / "lesion.nii.gz"
    nib.save(nib.Nifti1Image(data, AFFINE), p)
    return SubjectData.from_nifti(str(p), metadata={"space": "MNI152NLin6Asym", "resolution": 2})


def _maps(result):
    res = result.results["FunctionalNetworkMapping"]
    return {k: np.asarray(res[k].data.dataobj) for k in ("rmap", "zmap", "tmap") if k in res}


def test_run_and_run_batch_agree_on_random_data(tmp_path):
    """General parity guard on non-degenerate data (means + t-maps must match)."""
    rng = np.random.default_rng(2)
    ts = rng.standard_normal((6, 90, N_VOX)).astype(np.float32)
    conn = tmp_path / "c.h5"
    _write_connectome(conn, ts)

    register_functional_connectome(
        name="parity_rand", space="MNI152NLin6Asym", resolution=2.0,
        data_path=conn, n_subjects=6, description="test",
    )
    try:
        a = FunctionalNetworkMapping(connectome_name="parity_rand", method="boes", verbose=False)
        single = _maps(a.run(_lesion(tmp_path)))
        batched = _maps(a.run_batch([_lesion(tmp_path)])[0])
        for k in single:
            np.testing.assert_allclose(single[k], batched[k], atol=1e-4, err_msg=k)
    finally:
        unregister_functional_connectome("parity_rand")


def test_run_and_run_batch_agree_in_clamp_band(tmp_path):
    """Parity when correlations are strong enough that Fisher-z lands in the
    (3, 10) band. The old code clamped the streaming path to +/-3 while the
    non-batch path used +/-10, so run() and run_batch() disagreed here."""
    rng = np.random.default_rng(7)
    # Strong shared signal -> r ~0.99-0.998 -> z up to ~3.4 (straddles old clamp=3)
    base = rng.standard_normal((6, 90, 1)).astype(np.float32)
    ts = (base + 0.07 * rng.standard_normal((6, 90, N_VOX))).astype(np.float32)
    conn = tmp_path / "c.h5"
    _write_connectome(conn, ts)

    register_functional_connectome(
        name="parity_band", space="MNI152NLin6Asym", resolution=2.0,
        data_path=conn, n_subjects=6, description="test",
    )
    try:
        a = FunctionalNetworkMapping(connectome_name="parity_band", method="boes", verbose=False)
        single = _maps(a.run(_lesion(tmp_path)))
        batched = _maps(a.run_batch([_lesion(tmp_path)])[0])
        # rmap/zmap must agree; the old +/-3 vs +/-10 clamp would break zmap here.
        for k in ("rmap", "zmap"):
            np.testing.assert_allclose(single[k], batched[k], atol=1e-3, err_msg=k)
        # The clamp band is genuinely exercised (z exceeds the old 3.0 cap).
        assert np.nanmax(np.abs(single["zmap"])) > 3.0


    finally:
        unregister_functional_connectome("parity_band")


def test_streaming_variance_is_stable():
    """The streaming moment-combine must match two-pass np.std(ddof=1), even in
    the catastrophic-cancellation regime that broke the old sum_z2 formula."""
    from lacuna.analysis.functional_network_mapping import _combine_moments

    rng = np.random.default_rng(4)
    n_vox = 30
    # Large offset + tiny spread: this is where E[z^2]-E[z]^2 loses precision.
    blocks = [rng.normal(5.0, 0.01, size=(rng.integers(3, 12), n_vox)) for _ in range(8)]
    allz = np.vstack(blocks)

    mean = np.zeros(n_vox)
    M2 = np.zeros(n_vox)
    n = 0
    for b in blocks:
        mb = b.mean(axis=0)
        M2b = np.sum((b - mb) ** 2, axis=0)
        mean, M2, n = _combine_moments(mean, M2, n, mb, M2b, b.shape[0])

    np.testing.assert_allclose(mean, allz.mean(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(M2 / (n - 1)), allz.std(axis=0, ddof=1), rtol=0, atol=1e-12)

    # The OLD unstable formula would have produced a visibly different std here.
    old_std = np.sqrt(
        np.maximum((np.sum(allz**2, axis=0) / n) - allz.mean(axis=0) ** 2, 0) * n / (n - 1)
    )
    assert np.max(np.abs(old_std - allz.std(axis=0, ddof=1))) > 1e-13


def test_undefined_correlation_maps_to_zero(tmp_path):
    """A zero-variance connectome voxel yields r=0 (undefined), not r=+/-1."""
    rng = np.random.default_rng(3)
    conn = tmp_path / "c.h5"
    _write_connectome(conn, rng.standard_normal((2, 50, N_VOX)))
    register_functional_connectome(
        name="undef_corr", space="MNI152NLin6Asym", resolution=2.0,
        data_path=conn, n_subjects=2, description="test",
    )
    try:
        a = FunctionalNetworkMapping(connectome_name="undef_corr", method="boes", verbose=False)
        n_time, n_vox = 50, 4
        ts = rng.standard_normal((n_time, n_vox)).astype(np.float32)
        ts[:, 2] = 1.0  # constant -> zero variance -> undefined correlation
        seed = ts[:, 0][np.newaxis, :]  # (n_subjects=1, n_timepoints)
        r = a._compute_correlation_maps_batch(seed, ts[np.newaxis, :, :])
        # shape (1, n_vox); the constant voxel must be 0, not 1
        assert r[0, 2] == 0.0, f"undefined correlation should be 0, got {r[0, 2]}"
        assert abs(r[0, 0] - 1.0) < 1e-4  # seed vs itself is a true r=1
    finally:
        unregister_functional_connectome("undef_corr")
