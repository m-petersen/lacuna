import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.envelope import (
    AssetEnvelope,
    AssetType,
    ENVELOPE_FILENAME,
    ENVELOPE_SCHEMA_VERSION,
    IdentityRef,
    fingerprint,
    read_envelope,
    RequiresEntry,
    write_envelope,
)


def test_envelope_round_trips_to_dict():
    env = AssetEnvelope(
        asset_type=AssetType.SNTF_CACHE,
        identity=IdentityRef(
            kind="sha256_first_mib+size",
            fields={"sha256_first_mib": "abc", "size_bytes": 100, "n_streamlines": 985},
        ),
        requires=[
            RequiresEntry(
                role="tractogram",
                asset_type=AssetType.STRUCTURAL_CONNECTOME,
                identity=IdentityRef(
                    kind="sha256_first_mib+size",
                    fields={"sha256_first_mib": "def", "size_bytes": 1000},
                ),
            ),
        ],
        provenance={"lacuna_version": "0.1.0", "command": "lacuna prepare sntf"},
        data={"targets": ["D1", "D2"]},
    )
    blob = env.to_dict()
    assert blob["lacuna_schema_version"] == ENVELOPE_SCHEMA_VERSION
    assert blob["asset_type"] == "sntf_cache"
    assert blob["identity"]["fields"]["n_streamlines"] == 985
    restored = AssetEnvelope.from_dict(blob)
    assert restored == env


def test_from_dict_falls_back_to_current_schema_version_when_missing():
    blob = {
        "asset_type": "ntatlas",
        "identity": {"kind": "sha256_concat", "fields": {"sha256": "abc"}},
    }
    env = AssetEnvelope.from_dict(blob)
    assert env.lacuna_schema_version == ENVELOPE_SCHEMA_VERSION
    assert env.requires == []
    assert env.provenance == {}
    assert env.data == {}


def test_envelope_with_empty_requires_round_trips():
    env = AssetEnvelope(
        asset_type=AssetType.NTATLAS,
        identity=IdentityRef(kind="sha256_concat", fields={"sha256": "x"}),
    )
    assert AssetEnvelope.from_dict(env.to_dict()) == env


def _minimal_envelope():
    return AssetEnvelope(
        asset_type=AssetType.NTATLAS,
        identity=IdentityRef(kind="content_hash", fields={"sha256": "x"}),
    )


def test_write_envelope_creates_canonical_filename(tmp_path):
    env = _minimal_envelope()
    write_envelope(env, tmp_path)
    assert (tmp_path / ENVELOPE_FILENAME).exists()


def test_read_envelope_round_trip(tmp_path):
    env = _minimal_envelope()
    write_envelope(env, tmp_path)
    assert read_envelope(tmp_path) == env


def test_read_envelope_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=ENVELOPE_FILENAME):
        read_envelope(tmp_path)


def test_read_write_envelope_round_trips_populated_envelope(tmp_path):
    env = AssetEnvelope(
        asset_type=AssetType.SNTF_CACHE,
        identity=IdentityRef(
            kind="sha256_concat",
            fields={"sha256": "abc", "n_streamlines": 985},
        ),
        requires=[
            RequiresEntry(
                role="tractogram",
                asset_type=AssetType.STRUCTURAL_CONNECTOME,
                identity=IdentityRef(
                    kind="sha256_first_mib+size",
                    fields={"sha256_first_mib": "def", "size_bytes": 1000},
                ),
                path_hint="/data/tractogram.tck",
            ),
        ],
        provenance={"command": "lacuna prepare sntf"},
        data={"targets": ["D1", "5HT1a"]},
    )
    write_envelope(env, tmp_path)
    assert read_envelope(tmp_path) == env


def test_fingerprint_tractogram_matches_helper(tmp_path):
    from lacuna.utils.tractogram_id import compute_tractogram_fingerprint
    tck = tmp_path / "t.tck"
    streams = nib.streamlines.ArraySequence(
        [np.zeros((2, 3), dtype=np.float32) for _ in range(5)]
    )
    nib.streamlines.save(
        nib.streamlines.Tractogram(streams, affine_to_rasmm=np.eye(4)),
        str(tck),
    )
    ref = compute_tractogram_fingerprint(tck)
    out = fingerprint(tck, AssetType.STRUCTURAL_CONNECTOME)
    assert out.fields["sha256_first_mib"] == ref["sha256_first_mib"]
    assert out.fields["size_bytes"] == ref["size_bytes"]


def test_fingerprint_ntatlas_is_deterministic(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    affine = np.eye(4)
    for name in ["A", "B"]:
        nib.save(
            nib.Nifti1Image(np.full((2, 2, 2), ord(name), dtype=np.float32), affine),
            str(maps / f"{name}.nii.gz"),
        )
    f1 = fingerprint(tmp_path, AssetType.NTATLAS)
    f2 = fingerprint(tmp_path, AssetType.NTATLAS)
    assert f1 == f2
    assert f1.fields["n_targets"] == 2
    # Changing one map changes the fingerprint
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine),
        str(maps / "A.nii.gz"),
    )
    f3 = fingerprint(tmp_path, AssetType.NTATLAS)
    assert f3 != f1
