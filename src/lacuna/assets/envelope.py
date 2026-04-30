"""Shared metadata envelope for lacuna assets.

Every asset directory (NT atlas, SNTF cache, ACE cache, …) carries a
``lacuna_asset.json`` describing its identity, the assets it depends on,
provenance, and a per-type payload. This is the single contract a
loader needs to verify "do my inputs still match what this cache was
built from?" without reaching into asset-specific code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

ENVELOPE_FILENAME = "lacuna_asset.json"
ENVELOPE_SCHEMA_VERSION = 1


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
    """Write the envelope to ``<asset_root>/lacuna_asset.json``."""
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
