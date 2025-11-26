"""Unit tests for lacuna.core.keys module.

Tests the BIDS-style result key building and parsing utilities.
"""

import pytest

from lacuna.core.keys import (
    SOURCE_ABBREVIATIONS,
    build_result_key,
    get_source_abbreviation,
    parse_result_key,
)


class TestBuildResultKey:
    """Tests for build_result_key function."""

    def test_simple_key(self):
        """Build a simple result key."""
        key = build_result_key("Schaefer100", "FunctionalNetworkMapping", "correlation_map")
        assert key == "parc-Schaefer100_source-FunctionalNetworkMapping_desc-correlation_map"

    def test_mask_source(self):
        """Build key with MaskData source."""
        key = build_result_key("AAL", "MaskData", "mask_img")
        assert key == "parc-AAL_source-MaskData_desc-mask_img"

    def test_parc_with_underscore(self):
        """Build key when parcellation name contains underscore."""
        key = build_result_key("Tian_S4", "StructuralNetworkMapping", "mean_connectivity")
        assert key == "parc-Tian_S4_source-StructuralNetworkMapping_desc-mean_connectivity"

    def test_desc_with_underscore(self):
        """Build key when description contains underscore."""
        key = build_result_key("Schaefer100", "RegionalDamage", "damage_score")
        assert key == "parc-Schaefer100_source-RegionalDamage_desc-damage_score"

    def test_all_source_names(self):
        """Build keys with all standard source names."""
        sources = [
            "MaskData",
            "FunctionalNetworkMapping",
            "StructuralNetworkMapping",
            "RegionalDamage",
            "ParcelAggregation",
        ]
        for source in sources:
            key = build_result_key("TestAtlas", source, "test_result")
            assert f"_source-{source}_" in key


class TestParseResultKey:
    """Tests for parse_result_key function."""

    def test_parse_simple_key(self):
        """Parse a simple result key."""
        result = parse_result_key(
            "parc-Schaefer100_source-FunctionalNetworkMapping_desc-correlation_map"
        )
        assert result == {
            "parc": "Schaefer100",
            "source": "FunctionalNetworkMapping",
            "desc": "correlation_map",
        }

    def test_parse_parc_with_underscore(self):
        """Parse key when parcellation name contains underscore."""
        result = parse_result_key("parc-Tian_S4_source-MaskData_desc-mask_img")
        assert result == {
            "parc": "Tian_S4",
            "source": "MaskData",
            "desc": "mask_img",
        }

    def test_parse_desc_with_underscore(self):
        """Parse key when description contains underscore."""
        result = parse_result_key("parc-AAL_source-RegionalDamage_desc-damage_score")
        assert result == {
            "parc": "AAL",
            "source": "RegionalDamage",
            "desc": "damage_score",
        }

    def test_parse_multiple_underscores(self):
        """Parse key with multiple underscores in components."""
        key = build_result_key(
            "Custom_Atlas_V2", "FunctionalNetworkMapping", "some_long_desc_name"
        )
        result = parse_result_key(key)
        assert result == {
            "parc": "Custom_Atlas_V2",
            "source": "FunctionalNetworkMapping",
            "desc": "some_long_desc_name",
        }

    def test_parse_empty_key_raises(self):
        """Parsing empty key raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_result_key("")

    def test_parse_invalid_format_raises(self):
        """Parsing invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid result key format"):
            parse_result_key("invalid_key_format")

    def test_roundtrip(self):
        """Build then parse should return original components."""
        parc = "Schaefer200"
        source = "FunctionalNetworkMapping"
        desc = "correlation_map"

        key = build_result_key(parc, source, desc)
        parsed = parse_result_key(key)

        assert parsed["parc"] == parc
        assert parsed["source"] == source
        assert parsed["desc"] == desc


class TestSourceAbbreviations:
    """Tests for SOURCE_ABBREVIATIONS mapping (now identity mapping)."""

    def test_maskdata_maps_to_self(self):
        """MaskData maps to 'MaskData'."""
        assert SOURCE_ABBREVIATIONS["MaskData"] == "MaskData"

    def test_fnm_maps_to_self(self):
        """FunctionalNetworkMapping maps to itself."""
        assert SOURCE_ABBREVIATIONS["FunctionalNetworkMapping"] == "FunctionalNetworkMapping"

    def test_snm_maps_to_self(self):
        """StructuralNetworkMapping maps to itself."""
        assert SOURCE_ABBREVIATIONS["StructuralNetworkMapping"] == "StructuralNetworkMapping"

    def test_rd_maps_to_self(self):
        """RegionalDamage maps to itself."""
        assert SOURCE_ABBREVIATIONS["RegionalDamage"] == "RegionalDamage"

    def test_pa_maps_to_self(self):
        """ParcelAggregation maps to itself."""
        assert SOURCE_ABBREVIATIONS["ParcelAggregation"] == "ParcelAggregation"

    def test_all_values_unique(self):
        """All values should be unique."""
        values = list(SOURCE_ABBREVIATIONS.values())
        assert len(values) == len(set(values))


class TestGetSourceAbbreviation:
    """Tests for get_source_abbreviation function (validation/passthrough)."""

    def test_known_class_returns_itself(self):
        """Known class returns itself."""
        assert get_source_abbreviation("FunctionalNetworkMapping") == "FunctionalNetworkMapping"
        assert get_source_abbreviation("MaskData") == "MaskData"

    def test_unknown_class_raises(self):
        """Unknown class raises KeyError."""
        with pytest.raises(KeyError, match="Unknown analysis class"):
            get_source_abbreviation("UnknownClass")

    def test_error_message_lists_known(self):
        """Error message lists known classes."""
        with pytest.raises(KeyError, match="FunctionalNetworkMapping"):
            get_source_abbreviation("BadClass")
