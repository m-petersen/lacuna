"""NT target grouping, presets, and filename parsers.

Provides constants and utilities for working with neurotransmitter receptor/
transporter targets in the lacuna atlas engine.
"""

from __future__ import annotations

import re
from typing import Union

# ---------------------------------------------------------------------------
# Target groups
# ---------------------------------------------------------------------------

NT_TARGET_GROUPS: dict[str, list[str]] = {
    "serotonergic": ["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"],
    "dopaminergic": ["D1", "D23", "DAT", "FDOPA"],
    "cholinergic": ["VAChT", "M1", "A4B2"],
    "noradrenergic": ["NET"],
    "gabaergic": ["GABAa", "GABAa5"],
    "cannabinoid": ["CB1"],
    "opioid": ["MOR", "KOR"],
    "histaminergic": ["H3"],
    "glutamatergic": ["mGluR5", "NMDA"],
    "vesicular": ["VMAT2"],
}

ALL_TARGETS: list[str] = sorted(
    {t for targets in NT_TARGET_GROUPS.values() for t in targets}
)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

NT_PRESETS: dict[str, list[str]] = {
    "all": ALL_TARGETS,
    "dopaminergic": NT_TARGET_GROUPS["dopaminergic"],
    "serotonergic": NT_TARGET_GROUPS["serotonergic"],
    "cholinergic": NT_TARGET_GROUPS["cholinergic"],
    "monoaminergic": (
        NT_TARGET_GROUPS["serotonergic"]
        + NT_TARGET_GROUPS["dopaminergic"]
        + NT_TARGET_GROUPS["noradrenergic"]
    ),
}

# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

_TARGET_RE = re.compile(r"(?:^|_)target-([A-Za-z0-9]+)(?:_|$|\.)")
_PUB_RE = re.compile(r"(?:^|_)pub-([A-Za-z0-9]+)(?:_|$|\.)")


def parse_target_from_filename(filename: str) -> str:
    """Extract the target identifier from a BIDS-style filename.

    Looks for a ``target-{X}`` key-value pair in *filename*.

    Parameters
    ----------
    filename : str
        Filename or basename, e.g. ``"target-5HT1a_space-MNI.nii.gz"``.

    Returns
    -------
    str
        The target string (e.g. ``"5HT1a"``).

    Raises
    ------
    ValueError
        If no ``target-{X}`` pattern is found.
    """
    match = _TARGET_RE.search(filename)
    if match is None:
        raise ValueError(
            f"No 'target-{{X}}' pattern found in filename: {filename!r}"
        )
    return match.group(1)


def parse_publication_from_filename(filename: str) -> str:
    """Extract the publication key from a BIDS-style filename.

    Looks for a ``pub-{KEY}`` key-value pair in *filename*.

    Parameters
    ----------
    filename : str
        Filename or basename, e.g. ``"pub-beliveau2017_target-5HT1a.nii.gz"``.

    Returns
    -------
    str
        The publication key (e.g. ``"beliveau2017"``).

    Raises
    ------
    ValueError
        If no ``pub-{KEY}`` pattern is found.
    """
    match = _PUB_RE.search(filename)
    if match is None:
        raise ValueError(
            f"No 'pub-{{KEY}}' pattern found in filename: {filename!r}"
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def resolve_targets(
    targets: Union[str, list[str]],
    available: list[str],
) -> list[str]:
    """Resolve a preset name or explicit list against *available* targets.

    Parameters
    ----------
    targets : str | list[str]
        Either a preset name (e.g. ``"dopaminergic"``) or an explicit list of
        target strings.
    available : list[str]
        Targets that are actually present in the atlas.

    Returns
    -------
    list[str]
        Resolved and validated target list.

    Raises
    ------
    ValueError
        If a string *targets* is not a known preset name, or if any requested
        target is not in *available*.
    """
    if isinstance(targets, str):
        if targets == "all":
            return list(available)
        if targets not in NT_PRESETS:
            raise ValueError(
                f"unknown preset: {targets!r}. "
                f"Known presets: {sorted(NT_PRESETS)}"
            )
        requested = NT_PRESETS[targets]
    else:
        requested = list(targets)

    available_set = set(available)
    missing = [t for t in requested if t not in available_set]
    if missing:
        raise ValueError(
            f"Requested targets not available in atlas: {missing}"
        )
    return requested


