"""BIDS-style entity helpers for user-facing lacuna cache directories.

These helpers shape the *paths users see*, not the contents of cache
bundles. Files inside an asset directory keep their plain names
(``start_weights.npy``, ``maps/D1.nii.gz``); only the directory itself
gets the ``key-value_key-value`` form.
"""

from __future__ import annotations

import re

_VALID_VALUE = re.compile(r"^[A-Za-z0-9]+$")


def cache_dir_name(prefix: str, **entities: str) -> str:
    """Build ``<prefix>/<entity1>-<value1>_<entity2>-<value2>``.

    Entities are sorted by key so the same set of entities always
    produces the same path. Values must be alphanumeric — underscores
    and dashes are reserved by BIDS as separators.
    """
    if not entities:
        raise ValueError("cache_dir_name requires at least one entity")
    parts = []
    for key, value in sorted(entities.items()):
        if not _VALID_VALUE.match(str(value)):
            raise ValueError(
                f"entity value {value!r} for key {key!r} must be alphanumeric "
                "(no underscores or dashes — those are BIDS separators)"
            )
        parts.append(f"{key}-{value}")
    return f"{prefix}/{'_'.join(parts)}"
