"""Tests verifying the RegionalDamage → LocalDamage rename."""

from lacuna.analysis import LocalDamage
from lacuna.analysis.base import BaseAnalysis


class TestLocalDamageRename:
    def test_local_damage_importable(self):
        assert LocalDamage is not None

    def test_is_base_analysis_subclass(self):
        assert issubclass(LocalDamage, BaseAnalysis)

    def test_regional_damage_removed(self):
        """RegionalDamage should no longer be importable from analysis."""
        import lacuna.analysis as analysis_module

        assert not hasattr(analysis_module, "RegionalDamage")
