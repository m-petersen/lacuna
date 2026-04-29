"""Bundled NT atlas collection metadata pinned to a NiSpace-data commit.

Loads `collection.json` which lists the curated representative PET map
per neurotransmitter system at a specific NiSpace-data commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_COLLECTION_PATH = Path(__file__).parent / "collection.json"


def load_collection() -> dict[str, Any]:
    """Load the bundled NT atlas collection."""
    with _COLLECTION_PATH.open() as f:
        return json.load(f)


def map_rel_path(map_id: str) -> str:
    """Repo-relative path for a map's MNI152NLin6Asym .nii.gz."""
    coll = load_collection()
    return coll["map_path_template"].format(map_id=map_id)


def map_url(map_id: str) -> str:
    """Raw GitHub URL for a map's MNI152NLin6Asym .nii.gz."""
    coll = load_collection()
    return coll["url_template"].format(commit=coll["nispace_commit"], path=map_rel_path(map_id))


def hashes_url() -> str:
    """Raw GitHub URL for `file_hashes.json` at the pinned commit."""
    coll = load_collection()
    return coll["url_template"].format(commit=coll["nispace_commit"], path=coll["hashes_path"])


def all_map_ids() -> list[str]:
    """Flat list of all representative map IDs across systems."""
    return [mid for ids in load_collection()["systems"].values() for mid in ids]


def systems() -> dict[str, list[str]]:
    """System -> list of map IDs."""
    return load_collection()["systems"]


def parse_target(map_id: str) -> str:
    """Extract the target name (e.g. '5HT1a') from a map ID."""
    if not map_id.startswith("target-"):
        raise ValueError(f"Map ID does not start with 'target-': {map_id}")
    return map_id.split("_", 1)[0][len("target-"):]
