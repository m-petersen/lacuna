"""Tractogram content fingerprinting for cache validation.

Used by the SNTF prepare/run pair to detect when a precomputed cache
is reused with a tractogram other than the one it was built for.
A cheap content-based fingerprint (size + sha256 of the first 1 MiB,
which always covers the .tck text header) is enough to flag every
realistic mismatch without rehashing multi-GB tractograms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_FINGERPRINT_SCHEMA = 1
_FINGERPRINT_BYTES = 1 << 20  # 1 MiB


def compute_tractogram_fingerprint(path: str | Path) -> dict[str, Any]:
    """Compute a content-based identity for a tractogram file."""
    path = Path(path).resolve()
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(_FINGERPRINT_BYTES))
    return {
        "schema": _FINGERPRINT_SCHEMA,
        "path": str(path),
        "size_bytes": int(size),
        "sha256_first_mib": h.hexdigest(),
    }
