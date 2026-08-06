"""Regression: the MRtrix wrappers must not raise FileExistsError when called
with the default (auto-generated) output path.

Previously they created the temp output with make_temp_file (which creates the
file on disk), so the very next 'if output_path.exists() and not force' guard
always fired. They now use a not-yet-created file inside a fresh temp dir.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

import lacuna.utils.mrtrix as mrtrix


def _fake_run(captured):
    def run(cmd, **kwargs):
        captured["cmd"] = cmd
        # tckedit <in> <out> ...  — simulate MRtrix writing the output file.
        out = cmd[2]
        Path(out).write_bytes(b"out")

    return run


def test_default_output_does_not_raise_file_exists(tmp_path, monkeypatch):
    trk = tmp_path / "in.tck"
    trk.write_bytes(b"streamlines")  # only existence is checked before MRtrix runs
    mask = nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4))

    captured = {}
    monkeypatch.setattr(mrtrix, "run_mrtrix_command", _fake_run(captured))

    # output_path=None, force=False — the default form that used to always raise.
    out = mrtrix.filter_tractogram_by_mask(trk, mask)

    out = Path(out)
    assert out.exists()
    assert out.suffix == ".tck"
    # A freshly-generated temp output needs no MRtrix -force flag.
    assert "-force" not in captured["cmd"]


def test_explicit_existing_output_still_guarded(tmp_path, monkeypatch):
    """The exists-guard must still protect a user-supplied path that exists."""
    trk = tmp_path / "in.tck"
    trk.write_bytes(b"streamlines")
    mask = nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4))
    existing = tmp_path / "already.tck"
    existing.write_bytes(b"present")

    monkeypatch.setattr(mrtrix, "run_mrtrix_command", _fake_run({}))

    import pytest

    with pytest.raises(FileExistsError):
        mrtrix.filter_tractogram_by_mask(trk, mask, output_path=existing)  # force defaults False

    # ...but force=True overwrites it (and adds -force to the command).
    captured = {}
    monkeypatch.setattr(mrtrix, "run_mrtrix_command", _fake_run(captured))
    out = mrtrix.filter_tractogram_by_mask(trk, mask, output_path=existing, force=True)
    assert Path(out) == existing
    assert "-force" in captured["cmd"]
