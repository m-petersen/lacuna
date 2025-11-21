"""Tests for atlas-by-name functionality in analysis modules."""

from lacuna.analysis import ParcelAggregation, RegionalDamage


class TestAtlasAggregationAtlasParameter:
    """Test ParcelAggregation with parcel_names parameter."""

    def test_single_atlas_name(self):
        """Test creating ParcelAggregation with single atlas name."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer400"], source="mask_img", aggregation="percent"
        )

        assert analysis.parcel_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating ParcelAggregation with multiple atlas names."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer400", "TianS2"], source="mask_img", aggregation="percent"
        )

        assert analysis.parcel_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting parcel_names parameter leaves it as None."""
        analysis = ParcelAggregation(source="mask_img", aggregation="percent")

        assert analysis.parcel_names is None


class TestRegionalDamageAtlasParameter:
    """Test RegionalDamage with parcel_names parameter."""

    def test_single_atlas_name(self):
        """Test creating RegionalDamage with single atlas name."""
        analysis = RegionalDamage(parcel_names=["Schaefer400"])

        assert analysis.parcel_names == ["Schaefer400"]

    def test_multiple_atlas_names(self):
        """Test creating RegionalDamage with multiple atlas names."""
        analysis = RegionalDamage(parcel_names=["Schaefer400", "TianS2"])

        assert analysis.parcel_names == ["Schaefer400", "TianS2"]

    def test_no_atlas_defaults_to_none(self):
        """Test that omitting parcel_names uses all bundled atlases."""
        analysis = RegionalDamage()

        assert analysis.parcel_names is None  # Will load all bundled atlases at validation
