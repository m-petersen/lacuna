"""Unit tests for lacuna.utils.suggestions module.

Tests fuzzy matching utilities for error message suggestions.
"""

import pytest

from lacuna.utils.suggestions import format_suggestions, suggest_similar


class TestSuggestSimilar:
    """Tests for suggest_similar function."""

    def test_exact_match(self):
        """Exact match should be first suggestion."""
        candidates = ["correlation_map", "z_score_map", "damage_score"]
        result = suggest_similar("correlation_map", candidates)
        assert result[0] == "correlation_map"

    def test_close_typo(self):
        """Close typo should find the correct suggestion first."""
        candidates = ["correlation_map", "z_score_map", "damage_score"]
        result = suggest_similar("correltion_map", candidates)
        assert result[0] == "correlation_map"

    def test_partial_match(self):
        """Partial match should find relevant suggestions."""
        candidates = ["correlation_map", "z_score_map", "damage_score"]
        result = suggest_similar("score", candidates)
        assert "z_score_map" in result
        assert "damage_score" in result

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        candidates = ["correlation_map", "z_score_map"]
        result = suggest_similar("CORRELATION_MAP", candidates)
        assert result[0] == "correlation_map"

    def test_max_suggestions(self):
        """Should respect max_suggestions limit."""
        candidates = ["aaa", "aab", "aac", "aad", "aae"]
        result = suggest_similar("aaa", candidates, max_suggestions=2)
        assert len(result) <= 2

    def test_min_similarity(self):
        """Should respect min_similarity threshold."""
        candidates = ["correlation_map", "z_score_map"]
        # Very high threshold - nothing matches
        result = suggest_similar("xyz", candidates, min_similarity=0.9)
        assert result == []

    def test_empty_candidates(self):
        """Empty candidate list returns empty result."""
        result = suggest_similar("anything", [])
        assert result == []

    def test_empty_query(self):
        """Empty query still returns results based on similarity."""
        candidates = ["a", "ab", "abc"]
        result = suggest_similar("", candidates)
        # Empty string has low similarity to everything
        assert isinstance(result, list)

    def test_sorted_by_similarity(self):
        """Results should be sorted by similarity."""
        candidates = ["apple", "application", "app"]
        result = suggest_similar("app", candidates)
        # "app" is exact match, should be first
        assert result[0] == "app"

    def test_multiple_good_matches(self):
        """Multiple similar candidates should all appear."""
        candidates = ["test_one", "test_two", "test_three", "unrelated"]
        result = suggest_similar("test", candidates)
        assert "test_one" in result
        assert "test_two" in result
        assert "test_three" in result


class TestFormatSuggestions:
    """Tests for format_suggestions function."""

    def test_single_suggestion(self):
        """Single suggestion formats correctly."""
        result = format_suggestions(["correlation_map"])
        assert result == "Did you mean 'correlation_map'?"

    def test_multiple_suggestions(self):
        """Multiple suggestions format correctly."""
        result = format_suggestions(["correlation_map", "z_score_map"])
        assert result == "Did you mean one of: 'correlation_map', 'z_score_map'?"

    def test_empty_suggestions(self):
        """Empty suggestions return empty string."""
        result = format_suggestions([])
        assert result == ""

    def test_three_suggestions(self):
        """Three suggestions format correctly."""
        result = format_suggestions(["a", "b", "c"])
        assert result == "Did you mean one of: 'a', 'b', 'c'?"
