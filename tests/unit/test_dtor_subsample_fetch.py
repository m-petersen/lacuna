"""Tests for the dTOR985 subsample fetch (10pct / 25pct from OSF)."""

import pytest

from lacuna.core.exceptions import DownloadError
from lacuna.io import fetch_dtor985_subsample, list_fetchable_connectomes
from lacuna.io.downloaders import CONNECTOME_SOURCES


def test_subsamples_registered_as_fetchable():
    names = [s.name for s in list_fetchable_connectomes()]
    assert "dtor985_10pct" in names
    assert "dtor985_25pct" in names


@pytest.mark.parametrize("name", ["dtor985_10pct", "dtor985_25pct"])
def test_subsample_source_metadata(name):
    s = CONNECTOME_SOURCES[name]
    assert s.type == "structural"
    assert s.source_type == "osf"
    assert s.space == "MNI152NLin2009bAsym"
    assert s.download_url.startswith("https://osf.io/")
    assert s.download_url.endswith("/download")


def test_unknown_variant_raises(tmp_path):
    with pytest.raises(DownloadError, match="Unknown dTOR985 variant"):
        fetch_dtor985_subsample("99pct", tmp_path)


def test_uses_existing_tck_without_downloading(tmp_path):
    # Pre-place the expected .tck so the fetch skips the (network) download.
    (tmp_path / "dtor985_10pct.tck").write_bytes(b"mrtrix tracks\nEND\n")

    result = fetch_dtor985_subsample("10pct", tmp_path, register=False, force=False)

    assert result.success
    assert result.connectome_name == "dtor985_10pct"
    assert result.output_files[0].name == "dtor985_10pct.tck"
    assert result.download_time_seconds == 0.0  # nothing was downloaded


def test_cli_dispatch_routes_subsamples(monkeypatch, tmp_path):
    """`lacuna fetch dtor985_10pct` must route to the subsample handler."""
    import argparse

    from lacuna.cli import fetch_cmd

    called = {}

    def fake(variant, output_dir, **kw):
        called["variant"] = variant
        from lacuna.io.downloaders.base import FetchResult

        return FetchResult(
            success=True, connectome_name=f"dtor985_{variant}", output_dir=output_dir
        )

    monkeypatch.setattr("lacuna.io.fetch_dtor985_subsample", fake)
    args = argparse.Namespace(connectome="dtor985_25pct", output_dir=tmp_path, force=False)
    rc = fetch_cmd.handle_fetch_command(args)
    assert rc == 0
    assert called["variant"] == "25pct"
