"""GSP1000 connectome-registration wiring — runs without any download.

The fetch_* flows are only exercised end-to-end by slow, network-dependent
integration tests. This covers the registration step in isolation: success
wiring, graceful degradation on failure (a registration error must not crash
the fetch, only warn), and the register=False short-circuit.
"""

from types import SimpleNamespace

import lacuna.assets.connectomes as conn_mod
from lacuna.io.fetch import _register_gsp1000


def _source():
    return SimpleNamespace(space="MNI152NLin6Asym", n_subjects=1000, description="GSP1000")


def test_register_gsp1000_success(tmp_path, monkeypatch):
    captured = {}

    def fake_register(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(conn_mod, "register_functional_connectome", fake_register)

    warns: list[str] = []
    ok = _register_gsp1000(True, "gsp1000", _source(), tmp_path, None, warns)

    assert ok is True
    assert warns == []
    assert captured["name"] == "gsp1000"
    assert captured["space"] == "MNI152NLin6Asym"
    assert captured["data_path"] == tmp_path


def test_register_gsp1000_failure_is_graceful(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("registry write failed")

    monkeypatch.setattr(conn_mod, "register_functional_connectome", boom)

    warns: list[str] = []
    ok = _register_gsp1000(True, "gsp1000", _source(), tmp_path, None, warns)

    assert ok is False
    assert any("Registration failed" in w for w in warns)


def test_register_gsp1000_skipped_when_register_false(tmp_path):
    warns: list[str] = []
    assert _register_gsp1000(False, "gsp1000", _source(), tmp_path, None, warns) is False
    assert warns == []
