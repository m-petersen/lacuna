"""
Fuzzy string matching utilities for improved error messages.

This module provides helpers for suggesting similar strings when users
make typos or mistakes in key/attribute names. Used to enhance error
messages with "Did you mean?" suggestions.

Examples
--------
>>> from lacuna.utils.suggestions import suggest_similar
>>> available = ["correlationmap", "zscoremap", "damagescore"]
>>> suggestions = suggest_similar("correlation", available, max_suggestions=2)
>>> suggestions
['correlation_map']
"""

from __future__ import annotations

from difflib import SequenceMatcher


def suggest_similar(
    query: str,
    candidates: list[str],
    max_suggestions: int = 3,
    min_similarity: float = 0.4,
) -> list[str]:
    """
    Find candidates most similar to the query string.

    Uses difflib.SequenceMatcher for similarity scoring. Results are sorted
    by similarity (most similar first) and filtered by minimum threshold.

    Parameters
    ----------
    query : str
        The string to find matches for (e.g., user's typo).
    candidates : list[str]
        Available options to suggest from.
    max_suggestions : int, default=3
        Maximum number of suggestions to return.
    min_similarity : float, default=0.4
        Minimum similarity ratio (0.0 to 1.0) to include a suggestion.
        Higher values require closer matches.

    Returns
    -------
    list[str]
        Up to `max_suggestions` similar candidates, sorted by similarity.
        Empty list if no candidates meet the minimum similarity threshold.

    Examples
    --------
    >>> available = ["correlationmap", "zscoremap", "damagescore"]
    >>> suggest_similar("correltion_map", available)
    ['correlation_map']

    >>> suggest_similar("score", available)
    ['z_score_map', 'damage_score']

    >>> suggest_similar("xyz", available, min_similarity=0.5)
    []  # No close matches

    >>> # Case-insensitive matching
    >>> suggest_similar("Correlation_Map", available)
    ['correlation_map']
    """
    if not candidates:
        return []

    # Compute similarity for each candidate
    query_lower = query.lower()
    scored = []

    for candidate in candidates:
        # Use case-insensitive comparison for scoring
        ratio = SequenceMatcher(None, query_lower, candidate.lower()).ratio()
        if ratio >= min_similarity:
            scored.append((ratio, candidate))

    # Sort by similarity (descending), then alphabetically for ties
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Return top suggestions
    return [candidate for _, candidate in scored[:max_suggestions]]


def format_suggestions(suggestions: list[str]) -> str:
    """
    Format a list of suggestions for inclusion in an error message.

    Parameters
    ----------
    suggestions : list[str]
        List of suggested strings.

    Returns
    -------
    str
        Formatted string for error message, or empty string if no suggestions.

    Examples
    --------
    >>> format_suggestions(["correlationmap"])
    "Did you mean 'correlation_map'?"

    >>> format_suggestions(["correlationmap", "zscoremap"])
    "Did you mean one of: 'correlation_map', 'z_score_map'?"

    >>> format_suggestions([])
    ''
    """
    if not suggestions:
        return ""

    if len(suggestions) == 1:
        return f"Did you mean '{suggestions[0]}'?"

    quoted = [f"'{s}'" for s in suggestions]
    return f"Did you mean one of: {', '.join(quoted)}?"
