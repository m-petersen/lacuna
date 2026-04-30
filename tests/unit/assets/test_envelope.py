from lacuna.assets.envelope import (
    AssetEnvelope, AssetType, IdentityRef, RequiresEntry, ENVELOPE_SCHEMA_VERSION,
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
