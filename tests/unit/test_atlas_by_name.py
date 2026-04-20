"""Tests for atlas-by-name functionality in analysis modules."""

from lacuna.analysis import ParcelAggregation, LocalDamage


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


class TestRegionalDamageAtlasParameter:
    """Test LocalDamage with parcel_names parameter."""

    def test_single_atlas_name(self):
        """Test creating LocalDamage with single atlas name."""
        analysis = LocalDamage(parcel_names=["Schaefer400"])

        assert analysis.parcel_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating LocalDamage with multiple atlas names."""
        analysis = LocalDamage(parcel_names=["Schaefer400", "TianS2"])

        assert analysis.parcel_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting parcel_names uses all bundled atlases."""
        analysis = LocalDamage()

        assert analysis.parcel_names is None  # Will load all bundled atlases at validation
