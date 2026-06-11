"""Regression test: trk_to_tck must stream streamlines lazily (low memory),
not eagerly load the whole tractogram into RAM.

The eager nibabel path needs ~32 GB for dTOR985 (~11M streamlines); lazy
loading keeps peak memory to a single streamline plus I/O buffers.
"""

import nibabel as nib
import numpy as np
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
