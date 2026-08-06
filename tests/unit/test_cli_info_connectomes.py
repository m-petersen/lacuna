"""`lacuna info connectomes` lists the fetchable catalogue (path-based CLI, no
persistent registry), marking any already downloaded and using correct sizes."""

from lacuna.cli.main import _show_connectomes_info


def test_info_marks_downloaded_and_lists_fetchable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LACUNA_DATA_DIR", str(tmp_path))
    conn = tmp_path / "connectomes"
    proc = conn / "gsp1000" / "processed"
    proc.mkdir(parents=True)
    (proc / "gsp1000_chunk_000.h5").touch()
    (conn / "hcp1065").mkdir(parents=True)
    (conn / "hcp1065" / "hcp1065.tck").touch()

    rc = _show_connectomes_info()
    out = capsys.readouterr().out

    assert rc == 0
    assert "Fetchable connectomes" in out
    assert "[downloaded]" in out  # downloaded ones are marked in the catalogue
    # the removed section, old framing, and wrong size must all be gone
    assert "Downloaded (ready to use)" not in out
    assert "--connectome-path" not in out
    assert "Registered Connectomes" not in out
    assert "200GB" not in out and "200 GB" not in out


def test_info_no_downloaded_tag_when_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LACUNA_DATA_DIR", str(tmp_path))

    rc = _show_connectomes_info()
    out = capsys.readouterr().out

    assert rc == 0
    assert "Fetchable connectomes" in out
    assert "gsp1000" in out
    assert "[downloaded]" not in out
    assert "Downloaded (ready to use)" not in out
