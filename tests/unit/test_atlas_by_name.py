"""Tests for atlas-by-name functionality in analysis modules."""

import pytest
from lacuna.analysis import AtlasAggregation, RegionalDamage


class TestAtlasAggregationAtlasParameter:
    """Test AtlasAggregation with atlas_names parameter."""

    def test_single_atlas_name(self):
        """Test creating AtlasAggregation with single atlas name."""
        analysis = AtlasAggregation(
            atlas_names=["Schaefer400"],
            source="lesion_img",
            aggregation="percent"
        )
        
        assert analysis.atlas_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating AtlasAggregation with multiple atlas names."""
        analysis = AtlasAggregation(
            atlas_names=["Schaefer400", "TianS2"],
            source="lesion_img",
            aggregation="percent"
        )
        
        assert analysis.atlas_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting atlas_names parameter leaves it as None."""
        analysis = AtlasAggregation(
            source="lesion_img",
            aggregation="percent"
        )
        
        assert analysis.atlas_names is None


class TestRegionalDamageAtlasParameter:
    """Test RegionalDamage with atlas_names parameter."""

    def test_single_atlas_name(self):
        """Test creating RegionalDamage with single atlas name."""
        analysis = RegionalDamage(
            atlas_names=["Schaefer400"]
        )
        
        assert analysis.atlas_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating RegionalDamage with multiple atlas names."""
        analysis = RegionalDamage(
            atlas_names=["Schaefer400", "TianS2"]
        )
        
        assert analysis.atlas_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting atlas_names uses all bundled atlases."""
        analysis = RegionalDamage()
        
        assert analysis.atlas_names is None  # Will load all bundled atlases at validation
