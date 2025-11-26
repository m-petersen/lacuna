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
        key = build_result_key("Schaefer100", "fnm", "correlation_map")
        assert key == "parc-Schaefer100_source-fnm_desc-correlation_map"

    def test_mask_source(self):
        """Build key with mask source."""
        key = build_result_key("AAL", "mask", "mask_img")
        assert key == "parc-AAL_source-mask_desc-mask_img"

    def test_parc_with_underscore(self):
        """Build key when parcellation name contains underscore."""
        key = build_result_key("Tian_S4", "snm", "mean_connectivity")
        assert key == "parc-Tian_S4_source-snm_desc-mean_connectivity"

    def test_desc_with_underscore(self):
        """Build key when description contains underscore."""
        key = build_result_key("Schaefer100", "rd", "damage_score")
        assert key == "parc-Schaefer100_source-rd_desc-damage_score"

    def test_all_source_abbreviations(self):
        """Build keys with all standard source abbreviations."""
        for source in ["mask", "fnm", "snm", "rd", "pa"]:
            key = build_result_key("TestAtlas", source, "test_result")
            assert f"_source-{source}_" in key


class TestParseResultKey:
    """Tests for parse_result_key function."""

    def test_parse_simple_key(self):
        """Parse a simple result key."""
        result = parse_result_key("parc-Schaefer100_source-fnm_desc-correlation_map")
        assert result == {
            "parc": "Schaefer100",
            "source": "fnm",
            "desc": "correlation_map",
        }

    def test_parse_parc_with_underscore(self):
        """Parse key when parcellation name contains underscore."""
        result = parse_result_key("parc-Tian_S4_source-mask_desc-mask_img")
        assert result == {
            "parc": "Tian_S4",
            "source": "mask",
            "desc": "mask_img",
        }

    def test_parse_desc_with_underscore(self):
        """Parse key when description contains underscore."""
        result = parse_result_key("parc-AAL_source-rd_desc-damage_score")
        assert result == {
            "parc": "AAL",
            "source": "rd",
            "desc": "damage_score",
        }

    def test_parse_multiple_underscores(self):
        """Parse key with multiple underscores in components."""
        key = build_result_key("Custom_Atlas_V2", "fnm", "some_long_desc_name")
        result = parse_result_key(key)
        assert result == {
            "parc": "Custom_Atlas_V2",
            "source": "fnm",
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
        source = "fnm"
        desc = "correlation_map"

        key = build_result_key(parc, source, desc)
        parsed = parse_result_key(key)

        assert parsed["parc"] == parc
        assert parsed["source"] == source
        assert parsed["desc"] == desc


class TestSourceAbbreviations:
    """Tests for SOURCE_ABBREVIATIONS mapping."""

    def test_maskdata_abbreviation(self):
        """MaskData abbreviates to 'mask'."""
        assert SOURCE_ABBREVIATIONS["MaskData"] == "mask"

    def test_fnm_abbreviation(self):
        """FunctionalNetworkMapping abbreviates to 'fnm'."""
        assert SOURCE_ABBREVIATIONS["FunctionalNetworkMapping"] == "fnm"

    def test_snm_abbreviation(self):
        """StructuralNetworkMapping abbreviates to 'snm'."""
        assert SOURCE_ABBREVIATIONS["StructuralNetworkMapping"] == "snm"

    def test_rd_abbreviation(self):
        """RegionalDamage abbreviates to 'rd'."""
        assert SOURCE_ABBREVIATIONS["RegionalDamage"] == "rd"

    def test_pa_abbreviation(self):
        """ParcelAggregation abbreviates to 'pa'."""
        assert SOURCE_ABBREVIATIONS["ParcelAggregation"] == "pa"

    def test_all_abbreviations_unique(self):
        """All abbreviations should be unique."""
        values = list(SOURCE_ABBREVIATIONS.values())
        assert len(values) == len(set(values))


class TestGetSourceAbbreviation:
    """Tests for get_source_abbreviation function."""

    def test_known_class(self):
        """Get abbreviation for known class."""
        assert get_source_abbreviation("FunctionalNetworkMapping") == "fnm"
        assert get_source_abbreviation("MaskData") == "mask"

    def test_unknown_class_raises(self):
        """Unknown class raises KeyError."""
        with pytest.raises(KeyError, match="Unknown analysis class"):
            get_source_abbreviation("UnknownClass")

    def test_error_message_lists_known(self):
        """Error message lists known classes."""
        with pytest.raises(KeyError, match="FunctionalNetworkMapping"):
            get_source_abbreviation("BadClass")
