"""Regression test: trk_to_tck must stream streamlines lazily (low memory),
not eagerly load the whole tractogram into RAM.

The eager nibabel path needs ~32 GB for dTOR985 (~11M streamlines); lazy
loading keeps peak memory to a single streamline plus I/O buffers.
"""

import nibabel as nib
import numpy as np
import pytest
from nibabel.streamlines import TckFile, Tractogram, TrkFile

from lacuna.io.convert import trk_to_tck


def _make_trk(path):
    rng = np.random.default_rng(0)
    streams = [
        rng.standard_normal((rng.integers(4, 12), 3)).astype(np.float32) * 20 for _ in range(120)
    ]
    affine = np.array([[2, 0, 0, -90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]], float)
    nib.streamlines.save(Tractogram(streamlines=streams, affine_to_rasmm=affine), str(path))
    return path


def test_streaming_output_matches_eager_conversion(tmp_path):
    """The streaming conversion must be identical to the eager one."""
    trk = _make_trk(tmp_path / "in.trk")

    # Eager reference (the previous implementation)
    TckFile(tractogram=TrkFile.load(str(trk)).tractogram).save(str(tmp_path / "eager.tck"))
    eager = nib.streamlines.load(str(tmp_path / "eager.tck")).streamlines

    out = trk_to_tck(trk, tmp_path / "lazy.tck", overwrite=True)
    lazy = nib.streamlines.load(str(out)).streamlines

    assert len(lazy) == len(eager) == 120
    for a, b in zip(lazy, eager, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-4)


# A spread of source-tractogram orientations, all with a non-identity
# voxel_to_rasmm so each exercises the double-affine bug. The conversion must be
# agnostic to the source's storage convention (nibabel always yields RAS+ world
# points), so world coordinates must be preserved for every one of these.
_SOURCE_AFFINES = {
    "radiological_x": np.array(
        [[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]], float
    ),
    "neurological_x": np.array(
        [[2, 0, 0, -90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]], float
    ),
    "flip_y_LAS": np.array([[2, 0, 0, -90], [0, -2, 0, 126], [0, 0, 2, -72], [0, 0, 0, 1]], float),
    "oblique_rot": np.array(
        [[1.7, -1.0, 0, 10], [1.0, 1.7, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]], float
    ),
}


@pytest.mark.parametrize("name", list(_SOURCE_AFFINES))
def test_trk_to_tck_preserves_world_coordinates(tmp_path, name):
    """Regression: trk_to_tck must not re-apply the source affine — for ANY
    source orientation.

    A real .trk carries a non-identity ``voxel_to_rasmm`` header (e.g. dTOR985
    in MNI space). nibabel's streamline iterator already yields RAS+ world (mm)
    coordinates regardless of the file's storage convention, so the converted
    .tck must have *identical* world coordinates. The previous code passed the
    source affine as ``affine_to_rasmm``, applying it a second time on save and
    mislocating every streamline by ~10 cm.

    NB: a Tractogram saved with only ``affine_to_rasmm=`` (as the streaming test
    above does) round-trips through an *identity* header and cannot expose this
    bug — the header must carry a non-identity ``voxel_to_rasmm``.
    """
    aff = _SOURCE_AFFINES[name]
    world = np.array([[10, 20, 30], [12, 22, 32], [14, 24, 34]], np.float32)
    trk = tmp_path / f"{name}.trk"
    nib.streamlines.save(
        Tractogram(streamlines=[world], affine_to_rasmm=np.eye(4)),
        str(trk),
        header={"voxel_to_rasmm": aff, "dimensions": [100, 100, 100], "voxel_sizes": [2, 2, 2]},
    )
    # Guard against a vacuous test: the .trk must actually carry a non-identity affine.
    assert not np.allclose(nib.streamlines.load(str(trk)).affine, np.eye(4))

    out = trk_to_tck(trk, tmp_path / "out.tck", overwrite=True)
    got = nib.streamlines.load(str(out)).streamlines[0]

    np.testing.assert_allclose(got, world, atol=1e-3)
    # And explicitly reject the double-applied (previously-buggy) coordinates.
    double_applied = nib.affines.apply_affine(aff, world)
    assert not np.allclose(got, double_applied, atol=1e-3)


def test_trk_to_tck_uses_lazy_loading(tmp_path, monkeypatch):
    """Every tractogram load must pass lazy_load=True (no eager materialization)."""
    trk = _make_trk(tmp_path / "in.trk")

    lazy_flags = []
    real_load = nib.streamlines.load

    def spy(fileobj, *args, lazy_load=False, **kwargs):
        lazy_flags.append(lazy_load)
        return real_load(fileobj, *args, lazy_load=lazy_load, **kwargs)

    monkeypatch.setattr(nib.streamlines, "load", spy)
    trk_to_tck(trk, tmp_path / "out.tck", overwrite=True)

    assert lazy_flags, "nibabel.streamlines.load was never called"
    assert all(lazy_flags), "trk_to_tck must load with lazy_load=True (it loaded eagerly)"
