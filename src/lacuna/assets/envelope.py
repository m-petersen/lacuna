"""Shared metadata envelope for lacuna assets.

Every asset directory (NT atlas, SNTF cache, ACE cache, …) carries a
``lacuna_asset.json`` describing its identity, the assets it depends on,
provenance, and a per-type payload. This is the single contract a
loader needs to verify "do my inputs still match what this cache was
built from?" without reaching into asset-specific code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from lacuna.utils.tractogram_id import compute_tractogram_fingerprint

ENVELOPE_FILENAME = "lacuna_asset.json"
ENVELOPE_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB


class AssetType(str, Enum):
    """Kinds of lacuna-managed assets that carry an envelope."""

    NTATLAS = "ntatlas"
    STRUCTURAL_CONNECTOME = "structural_connectome"
    FUNCTIONAL_CONNECTOME = "functional_connectome"
    SNTF_CACHE = "sntf_cache"
    ACE_CACHE = "ace_cache"


@dataclass
class IdentityRef:
    """Content fingerprint of an asset (hash + size + type-specific fields)."""

    kind: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "fields": dict(self.fields)}

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "IdentityRef":
        return cls(kind=blob["kind"], fields=dict(blob.get("fields", {})))


@dataclass
class RequiresEntry:
    """A typed dependency on another asset, pinned by its identity."""

    role: str
    asset_type: AssetType
    identity: IdentityRef
    path_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "asset_type": self.asset_type.value,
            "identity": self.identity.to_dict(),
            "path_hint": self.path_hint,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "RequiresEntry":
        return cls(
            role=blob["role"],
            asset_type=AssetType(blob["asset_type"]),
            identity=IdentityRef.from_dict(blob["identity"]),
            path_hint=blob.get("path_hint"),
        )


@dataclass
class AssetEnvelope:
    """Shared metadata for a lacuna asset directory: identity + dependencies + provenance + per-type payload."""

    asset_type: AssetType
    identity: IdentityRef
    requires: list[RequiresEntry] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    lacuna_schema_version: int = ENVELOPE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "lacuna_schema_version": self.lacuna_schema_version,
            "asset_type": self.asset_type.value,
            "identity": self.identity.to_dict(),
            "requires": [r.to_dict() for r in self.requires],
            "provenance": dict(self.provenance),
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "AssetEnvelope":
        return cls(
            lacuna_schema_version=blob.get("lacuna_schema_version", ENVELOPE_SCHEMA_VERSION),
            asset_type=AssetType(blob["asset_type"]),
            identity=IdentityRef.from_dict(blob["identity"]),
            requires=[RequiresEntry.from_dict(r) for r in blob.get("requires", [])],
            provenance=dict(blob.get("provenance", {})),
            data=dict(blob.get("data", {})),
        )


def write_envelope(env: AssetEnvelope, asset_root: Path | str) -> Path:
    """Write the envelope to ``<asset_root>/lacuna_asset.json``.

    Creates ``asset_root`` (and missing parents) if it does not exist.
    """
    asset_root = Path(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    path = asset_root / ENVELOPE_FILENAME
    path.write_text(json.dumps(env.to_dict(), indent=2))
    return path


def read_envelope(asset_root: Path | str) -> AssetEnvelope:
    """Read the envelope from ``<asset_root>/lacuna_asset.json``."""
    asset_root = Path(asset_root)
    path = asset_root / ENVELOPE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {ENVELOPE_FILENAME} at {asset_root}. "
            "This asset directory is either missing, corrupted, or written by an "
            "older lacuna version."
        )
    return AssetEnvelope.from_dict(json.loads(path.read_text()))


def fingerprint(path: Path | str, asset_type: AssetType) -> IdentityRef:
    """Compute the canonical identity for an asset on disk.

    Dispatches by ``asset_type``. Each concrete fingerprint is a content hash
    plus a few type-specific fields used for at-a-glance debugging
    (e.g. ``n_streamlines`` for tractograms, ``n_targets`` for atlases).
    """
    path = Path(path)
    if asset_type == AssetType.STRUCTURAL_CONNECTOME:
        fp = compute_tractogram_fingerprint(path)
        return IdentityRef(
            kind="sha256_first_mib+size",
            fields={
                "sha256_first_mib": fp["sha256_first_mib"],
                "size_bytes": fp["size_bytes"],
            },
        )
    if asset_type == AssetType.NTATLAS:
        return _fingerprint_ntatlas(path)
    if asset_type == AssetType.SNTF_CACHE:
        return _fingerprint_sntf_cache(path)
    raise NotImplementedError(f"No fingerprint for asset_type={asset_type.value}")


def _hash_file_into(h: "hashlib._Hash", path: Path) -> None:
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)


def _fingerprint_ntatlas(asset_root: Path) -> IdentityRef:
    maps_dir = asset_root / "maps"
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"NT atlas missing maps/ directory: {asset_root}")
    map_files = sorted(maps_dir.glob("*.nii.gz"))
    h = hashlib.sha256()
    for p in map_files:
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        _hash_file_into(h, p)
    return IdentityRef(
        kind="sha256_concat",
        fields={"sha256": h.hexdigest(), "n_targets": len(map_files)},
    )


def _fingerprint_sntf_cache(asset_root: Path) -> IdentityRef:
    # Files are guaranteed present by the SNTF prepare path; missing-file
    # diagnostics live in the prepare/load sites, not here.
    h = hashlib.sha256()
    for name in ("start_weights.npy", "end_weights.npy"):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        _hash_file_into(h, asset_root / name)
    return IdentityRef(
        kind="sha256_concat",
        fields={"sha256": h.hexdigest()},
    )
