"""fetch_gsp1000 must reuse already-extracted raw data instead of re-downloading.

Users frequently already have the extracted sub-*/func/*finalmask tree (prior
run, manual copy). fetch previously always drove a tarball download, ignoring the
extracted data. It now detects the extracted subjects (same glob the converter
uses) and skips the download + extraction entirely.
"""

import nibabel as nib
import numpy as np

from lacuna.io import fetch as fetch_mod
from lacuna.io.convert import GSP1000_FUNC_GLOB
from lacuna.io.fetch import fetch_gsp1000

AFF = np.diag([2.0, 2.0, 2.0, 1.0])
SHAPE = (4, 4, 4)
T = 5


def _make_extracted_raw(raw_dir, subject_ids):
    for sid in subject_ids:
        d = raw_dir / f"sub-{sid}" / "func"
        d.mkdir(parents=True)
        data = np.random.default_rng(int(sid)).standard_normal((*SHAPE, T)).astype(np.float32)
        nib.save(nib.Nifti1Image(data, AFF), d / f"sub-{sid}_bld001_rest_x_finalmask.nii.gz")


def test_fetch_reuses_extracted_raw_and_skips_download(tmp_path, monkeypatch):
    out = tmp_path / "gsp"
    raw = out / "raw"
    _make_extracted_raw(raw, ["0001", "0002", "0003"])

    # Brain mask on the same grid (so the converter's affine check passes).
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), AFF), mask_path)
    monkeypatch.setattr(fetch_mod, "_find_brain_mask", lambda raw_dir: mask_path)

    # Any attempt to download must fail the test.
    from lacuna.io.downloaders.dataverse import DataverseDownloader

    def _boom(self, *a, **k):
        raise AssertionError("download must be skipped when extracted raw exists")

    monkeypatch.setattr(DataverseDownloader, "download", _boom)

    result = fetch_gsp1000(
        output_dir=out, api_key="unused", batches=1, register=False, verbose=True
    )

    assert result.success
    assert len(result.output_files) >= 1
    assert result.download_time_seconds == 0.0
    # A leftover partial tarball must not trigger a re-download either.
    assert any("skipping download" in w for w in result.warnings)


def test_fetch_leftover_tar_tmp_does_not_force_redownload(tmp_path, monkeypatch):
    out = tmp_path / "gsp"
    raw = out / "raw"
    _make_extracted_raw(raw, ["0001", "0002"])
    (raw / "GSP1000_v2_00.tar.tmp").write_bytes(b"partial")  # interrupted prior download

    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), AFF), mask_path)
    monkeypatch.setattr(fetch_mod, "_find_brain_mask", lambda raw_dir: mask_path)

    from lacuna.io.downloaders.dataverse import DataverseDownloader

    monkeypatch.setattr(
        DataverseDownloader,
        "download",
        lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("should not download")),
    )

    result = fetch_gsp1000(output_dir=out, api_key="unused", batches=1, register=False)
    assert result.success
    assert any("incomplete download" in w for w in result.warnings)


def test_no_raw_present_still_downloads(tmp_path, monkeypatch):
    """Guard: when there is NO extracted raw, the download path is still taken."""
    out = tmp_path / "gsp"
    (out / "raw").mkdir(parents=True)  # empty raw -> no subjects

    called = {"download": False}

    from lacuna.io.downloaders.dataverse import DataverseDownloader

    def _fake_download(self, *a, **k):
        called["download"] = True
        raise RuntimeError("stop after confirming download was attempted")

    monkeypatch.setattr(DataverseDownloader, "download", _fake_download)

    # No extracted raw -> reuse path is skipped -> download attempted.
    assert sorted((out / "raw").glob(GSP1000_FUNC_GLOB)) == []
    try:
        fetch_gsp1000(output_dir=out, api_key="unused", batches=1, register=False)
    except Exception:
        pass  # we only care that the download path was entered
    assert called["download"] is True
