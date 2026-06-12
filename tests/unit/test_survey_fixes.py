"""Regression tests for the pre-release survey fixes."""

import hashlib

import pandas as pd


def test_dataverse_make_hasher_selects_algorithm():
    """Dataverse checksum verification must honour the reported algorithm,
    not always assume MD5 (newer Dataverse reports SHA-1/SHA-256)."""
    from lacuna.io.downloaders.dataverse import _make_hasher

    assert _make_hasher("MD5").name == "md5"
    assert _make_hasher("SHA-1").name == "sha1"
    assert _make_hasher("SHA-256").name == "sha256"
    assert _make_hasher(None).name == "md5"  # default
    assert _make_hasher("nonsense").name == "md5"  # safe fallback

    # And it actually produces the right digest.
    payload = b"hello world"
    h = _make_hasher("SHA-256")
    h.update(payload)
    assert h.hexdigest() == hashlib.sha256(payload).hexdigest()


def test_normalize_session_set_accepts_both_forms():
    """Multi-session selection must match labels with or without 'ses-'."""
    from lacuna.cli.main import _normalize_session_set

    s = _normalize_session_set(["01", "ses-02"])
    assert {"01", "ses-01", "02", "ses-02"} <= s


def test_is_output_empty_treats_afnm_as_tsv(tmp_path):
    """`check --check-content` must detect all-zero AFNM parcel TSVs (AFNM,
    like FocalDamage, writes parcelstats.tsv — not a NIfTI)."""
    from lacuna.cli.main import _is_output_empty

    zero = tmp_path / "sub-01_method-afnm_atlas-x_desc-afnmap_parcelstats.tsv"
    pd.DataFrame({"region-1": [0.0], "region-2": [0.0]}).to_csv(zero, sep="\t", index=False)
    nonzero = tmp_path / "sub-02_method-afnm_atlas-x_desc-afnmap_parcelstats.tsv"
    pd.DataFrame({"region-1": [0.0], "region-2": [0.3]}).to_csv(nonzero, sep="\t", index=False)

    assert bool(_is_output_empty(zero, "afnm")) is True
    assert bool(_is_output_empty(nonzero, "afnm")) is False


def test_aggregate_3d_atlas_correct_labels_when_region_dropped():
    """When nilearn drops an empty region during resampling, values must map to
    the SURVIVING labels (recovered from masker.region_names_), not shift
    positionally onto the wrong parcel. Region 2 here sits outside the source
    FOV, so it is dropped after resampling atlas->data."""
    import nibabel as nib
    import numpy as np

    from lacuna.analysis import ParcelAggregation

    atlas_arr = np.zeros((12, 4, 4), np.int16)
    atlas_arr[1:3] = 1  # region 1 — inside source FOV
    atlas_arr[4:6] = 3  # region 3 — inside source FOV
    atlas_arr[9:11] = 2  # region 2 — OUTSIDE source FOV -> dropped on resample
    atlas = nib.Nifti1Image(atlas_arr, np.eye(4))

    src_arr = np.zeros((6, 4, 4), np.float32)  # source covers x 0..5 only
    src_arr[1:3] = 10.0  # over region 1
    src_arr[4:6] = 30.0  # over region 3
    src = nib.Nifti1Image(src_arr, np.eye(4))

    labels = {1: "label-1", 2: "label-2", 3: "label-3"}
    out = ParcelAggregation(aggregation="mean")._aggregate_3d_atlas(
        src, atlas, labels, voxel_volume_mm3=1.0
    )

    assert out["label-1"] == 10.0
    assert out["label-3"] == 30.0
    assert "label-2" not in out  # dropped region must not steal another's value


def test_afnm_requires_parcellation_upfront(tmp_path):
    """`run afnm` without --parcel-atlases must fail fast at config validation
    with a clear message, not deep inside the analysis constructor."""
    import pytest

    from lacuna.cli.main import RunConfig

    cfg = RunConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        analysis="afnm",
        analysis_options={},
    )
    with pytest.raises(ValueError, match="afnm requires a parcellation"):
        cfg.validate()
