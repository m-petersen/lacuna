"""Regression test: _find_brain_mask must not return a subject's 4D functional
series (named '*_finalmask.nii.gz') as the brain mask."""

import nibabel as nib
import numpy as np

from lacuna.io.fetch import _find_brain_mask


def test_find_brain_mask_skips_functional_finalmask(tmp_path):
    # GSP1000 subject functional file matches the '*mask*' glob but is 4D.
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10, 5), np.float32), np.eye(4)),
        tmp_path / "sub-01_bld001_rest_finalmask.nii.gz",
    )
    # The actual 3D brain mask.
    nib.save(
        nib.Nifti1Image(np.ones((10, 10, 10), np.uint8), np.eye(4)),
        tmp_path / "brain_mask.nii.gz",
    )

    result = _find_brain_mask(tmp_path)
    assert result.name == "brain_mask.nii.gz"
    assert nib.load(result).ndim == 3


def test_find_brain_mask_falls_back_to_osf_asset(tmp_path, monkeypatch):
    """With no mask in the download, _find_brain_mask fetches the canonical
    MNI152NLin6Asym 2mm brain mask via the OSF asset loader (no templateflow)."""
    import lacuna.assets.masks as masks_mod

    raw_dir = tmp_path / "raw"  # empty: no mask in the download
    raw_dir.mkdir()
    sentinel = tmp_path / "cached_brain_mask.nii.gz"
    calls = {}

    def fake_loader(space, resolution, **kwargs):
        calls["args"] = (space, resolution)
        return sentinel

    monkeypatch.setattr(masks_mod, "load_brain_mask", fake_loader)

    result = _find_brain_mask(raw_dir)
    assert result == sentinel
    assert calls["args"] == ("MNI152NLin6Asym", 2.0)


def test_brain_mask_registry_has_four_validated_entries():
    """All four space/resolution masks are registered with URL + sha256."""
    from lacuna.assets.masks import BRAIN_MASK_REGISTRY, mask_name

    expected = [
        ("MNI152NLin6Asym", 1.0),
        ("MNI152NLin6Asym", 2.0),
        ("MNI152NLin2009cAsym", 1.0),
        ("MNI152NLin2009cAsym", 2.0),
    ]
    for space, res in expected:
        meta = BRAIN_MASK_REGISTRY.get(mask_name(space, res))
        assert meta.space == space and meta.resolution == res
        assert meta.url.startswith("https://osf.io/")
        assert len(meta.sha256) == 64


def test_validate_mask_rejects_non_binary(tmp_path):
    """The on-load validator rejects a non-binary mask."""
    import pytest

    from lacuna.assets.masks.loader import _validate_mask
    from lacuna.core.spaces import REFERENCE_AFFINES, REFERENCE_SHAPES

    key = ("MNI152NLin6Asym", 2.0)
    bad = np.zeros(REFERENCE_SHAPES[key], np.int16)
    bad[10:20, 10:20, 10:20] = 7  # not 0/1
    p = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(bad, REFERENCE_AFFINES[key]), p)
    with pytest.raises(ValueError, match="not binary"):
        _validate_mask(p, "MNI152NLin6Asym", 2.0)
