"""Unit tests for analysis auto-discovery registry."""

from __future__ import annotations

import pytest

from lacuna.analysis.base import BaseAnalysis
from lacuna.analysis.registry import AnalysisRegistry, get_analysis, list_analyses


class TestAnalysisRegistry:
    """Tests for AnalysisRegistry class."""

    def setup_method(self):
        """Clear cache before each test."""
        AnalysisRegistry.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        AnalysisRegistry.clear_cache()

    def test_discover_returns_dict(self):
        """Test that discover() returns a dictionary."""
        result = AnalysisRegistry.discover()
        assert isinstance(result, dict)

    def test_discover_finds_known_analyses(self):
        """Test that discover() finds the known analysis classes."""
        result = AnalysisRegistry.discover()

        # Should find the existing analyses
        expected_analyses = [
            "FunctionalNetworkMapping",
            "ParcelAggregation",
            "RegionalDamage",
            "StructuralNetworkMapping",
        ]

        for name in expected_analyses:
            assert name in result, f"Expected to find {name} in discovered analyses"
            assert issubclass(result[name], BaseAnalysis)

    def test_discover_caches_results(self):
        """Test that discover() caches results."""
        result1 = AnalysisRegistry.discover()
        result2 = AnalysisRegistry.discover()

        # Should be the same object (cached)
        assert result1 is result2

    def test_list_analyses_returns_sorted_tuples(self):
        """Test that list_analyses() returns sorted (name, class) tuples."""
        result = AnalysisRegistry.list_analyses()

        assert isinstance(result, list)
        assert len(result) > 0

        # Check it's a list of tuples
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            name, cls = item
            assert isinstance(name, str)
            assert issubclass(cls, BaseAnalysis)

        # Check it's sorted
        names = [name for name, _ in result]
        assert names == sorted(names)

    def test_get_returns_correct_class(self):
        """Test that get() returns the correct class."""
        cls = AnalysisRegistry.get("FunctionalNetworkMapping")

        assert cls.__name__ == "FunctionalNetworkMapping"
        assert issubclass(cls, BaseAnalysis)

    def test_get_raises_keyerror_for_unknown(self):
        """Test that get() raises KeyError for unknown analysis."""
        with pytest.raises(KeyError) as exc_info:
            AnalysisRegistry.get("NonExistentAnalysis")

        assert "NonExistentAnalysis" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_clear_cache_works(self):
        """Test that clear_cache() clears the cache."""
        _ = AnalysisRegistry.discover()
        assert AnalysisRegistry._discovered is not None

        AnalysisRegistry.clear_cache()
        assert AnalysisRegistry._discovered is None


class TestModuleLevelFunctions:
    """Tests for module-level list_analyses() and get_analysis() functions."""

    def setup_method(self):
        """Clear cache before each test."""
        AnalysisRegistry.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        AnalysisRegistry.clear_cache()

    def test_list_analyses_function(self):
        """Test the module-level list_analyses() function."""
        result = list_analyses()

        assert isinstance(result, list)
        assert len(result) >= 4  # At least our known analyses

        names = [name for name, _ in result]
        assert "FunctionalNetworkMapping" in names
        assert "ParcelAggregation" in names

    def test_get_analysis_function(self):
        """Test the module-level get_analysis() function."""
        cls = get_analysis("RegionalDamage")

        assert cls.__name__ == "RegionalDamage"
        assert issubclass(cls, BaseAnalysis)

    def test_get_analysis_raises_keyerror(self):
        """Test that get_analysis() raises KeyError for unknown analysis."""
        with pytest.raises(KeyError):
            get_analysis("InvalidAnalysisName")


class TestDiscoveryFiltering:
    """Tests for analysis discovery filtering logic."""

    def setup_method(self):
        """Clear cache before each test."""
        AnalysisRegistry.clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        AnalysisRegistry.clear_cache()

    def test_base_analysis_not_discovered(self):
        """Test that BaseAnalysis itself is not discovered."""
        result = AnalysisRegistry.discover()

        assert "BaseAnalysis" not in result

    def test_private_classes_not_discovered(self):
        """Test that private classes (starting with _) are not discovered."""
        result = AnalysisRegistry.discover()

        for name in result.keys():
            assert not name.startswith("_"), f"Private class {name} should not be discovered"

    def test_discovered_classes_are_concrete(self):
        """Test that all discovered classes are concrete (not abstract)."""
        result = AnalysisRegistry.discover()

        for name, cls in result.items():
            # Check it's not abstract (has no unimplemented abstract methods)
            abstract_methods = getattr(cls, "__abstractmethods__", set())
            assert len(abstract_methods) == 0, f"{name} has abstract methods: {abstract_methods}"
