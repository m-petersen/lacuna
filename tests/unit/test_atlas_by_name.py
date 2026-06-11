"""Tests for atlas-by-name functionality in analysis modules."""

from lacuna.analysis import FocalDamage, ParcelAggregation


class TestParcelAggregationAtlasParameter:
    """Test ParcelAggregation with parcel_names parameter."""

    def test_single_atlas_name(self):
        """Test creating ParcelAggregation with single atlas name."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer400"], source="maskimg", aggregation="percent"
        )

        assert analysis.parcel_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating ParcelAggregation with multiple atlas names."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer400", "TianS2"], source="maskimg", aggregation="percent"
        )

        assert analysis.parcel_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting parcel_names parameter leaves it as None."""
        analysis = ParcelAggregation(source="maskimg", aggregation="percent")

        assert analysis.parcel_names is None


class TestFocalDamageAtlasParameter:
    """Test FocalDamage with parcel_names parameter."""

    def test_single_atlas_name(self):
        """Test creating FocalDamage with single atlas name."""
        analysis = FocalDamage(parcel_names=["Schaefer400"])

        assert analysis.parcel_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating FocalDamage with multiple atlas names."""
        analysis = FocalDamage(parcel_names=["Schaefer400", "TianS2"])

        assert analysis.parcel_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting parcel_names uses all bundled atlases."""
        analysis = FocalDamage()

        assert analysis.parcel_names is None  # Will load all bundled atlases at validation
