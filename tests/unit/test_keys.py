"""Unit tests for lacuna.core.keys module.

Tests the BIDS-style result key building and parsing utilities.
"""

import re

import pytest

from lacuna.core.keys import (
    SOURCE_ABBREVIATIONS,
    build_result_key,
    format_bids_export_filename,
    get_source_abbreviation,
    parse_result_key,
    split_atlas_name,
    to_bids_label,
)


class TestBuildResultKey:
    """Tests for build_result_key function."""

    def test_simple_key(self):
        """Build a simple result key."""
        key = build_result_key("Schaefer100", "FunctionalNetworkMapping", "rmap")
        assert key == "atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap"

    def test_mask_source_omits_desc(self):
        """Build key with SubjectData source omits desc (InputMask is the data)."""
        key = build_result_key("TianSubcortex_3TS1", "SubjectData", "maskimg")
        # SubjectData maps to InputMask and desc is automatically omitted
        assert key == "atlas-TianSubcortex_3TS1_source-InputMask"

    def test_mask_source_no_desc_provided(self):
        """Build key with SubjectData source without desc."""
        key = build_result_key("TianSubcortex_3TS1", "SubjectData")
        assert key == "atlas-TianSubcortex_3TS1_source-InputMask"

    def test_parc_with_underscore(self):
        """Build key when parcellation name contains underscore."""
        key = build_result_key("Tian_S4", "StructuralNetworkMapping", "mean_connectivity")
        assert key == "atlas-Tian_S4_source-StructuralNetworkMapping_desc-mean_connectivity"

    def test_desc_with_underscore(self):
        """Build key when description contains underscore."""
        key = build_result_key("Schaefer100", "RegionalDamage", "damagescore")
        assert key == "atlas-Schaefer100_source-RegionalDamage_desc-damagescore"

    def test_all_source_names(self):
        """Build keys with all standard source names."""
        sources = [
            ("SubjectData", "InputMask"),  # Maps to InputMask
            ("FunctionalNetworkMapping", "FunctionalNetworkMapping"),
            ("StructuralNetworkMapping", "StructuralNetworkMapping"),
            ("RegionalDamage", "RegionalDamage"),
            ("ParcelAggregation", "ParcelAggregation"),
        ]
        for source, expected_source in sources:
            key = build_result_key("TestAtlas", source, "test_result")
            if expected_source == "InputMask":
                # InputMask omits desc
                assert key == f"atlas-TestAtlas_source-{expected_source}"
            else:
                assert f"_source-{expected_source}_" in key


class TestParseResultKey:
    """Tests for parse_result_key function."""

    def test_parse_simple_key(self):
        """Parse a simple result key."""
        result = parse_result_key(
            "atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap"
        )
        assert result == {
            "atlas": "Schaefer100",
            "source": "FunctionalNetworkMapping",
            "desc": "rmap",
        }

    def test_parse_atlas_with_underscore(self):
        """Parse key when atlas name contains underscore."""
        result = parse_result_key("atlas-Tian_S4_source-InputMask")
        assert result == {
            "atlas": "Tian_S4",
            "source": "InputMask",
        }

    def test_parse_desc_with_underscore(self):
        """Parse key when description contains underscore."""
        result = parse_result_key("atlas-HCP1065_source-RegionalDamage_desc-damagescore")
        assert result == {
            "atlas": "HCP1065",
            "source": "RegionalDamage",
            "desc": "damagescore",
        }

    def test_parse_multiple_underscores(self):
        """Parse key with multiple underscores in components."""
        key = build_result_key("Custom_Atlas_V2", "FunctionalNetworkMapping", "some_long_desc_name")
        result = parse_result_key(key)
        assert result == {
            "atlas": "Custom_Atlas_V2",
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
        atlas = "Schaefer200"
        source = "FunctionalNetworkMapping"
        desc = "rmap"

        key = build_result_key(atlas, source, desc)
        parsed = parse_result_key(key)

        assert parsed["atlas"] == atlas
        assert parsed["source"] == source
        assert parsed["desc"] == desc

    def test_roundtrip_inputmask_no_desc(self):
        """Build then parse InputMask should work without desc."""
        atlas = "Schaefer200"
        source = "SubjectData"  # Maps to InputMask

        key = build_result_key(atlas, source, "maskimg")
        parsed = parse_result_key(key)

        assert parsed["atlas"] == atlas
        assert parsed["source"] == "InputMask"
        assert "desc" not in parsed  # No desc for InputMask


class TestSourceAbbreviations:
    """Tests for SOURCE_ABBREVIATIONS mapping."""

    def test_subjectdata_maps_to_inputmask(self):
        """SubjectData maps to 'InputMask'."""
        assert SOURCE_ABBREVIATIONS["SubjectData"] == "InputMask"

    def test_inputmask_maps_to_inputmask(self):
        """InputMask maps to 'InputMask'."""
        assert SOURCE_ABBREVIATIONS["InputMask"] == "InputMask"

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


class TestGetSourceAbbreviation:
    """Tests for get_source_abbreviation function."""

    def test_known_class_returns_abbreviation(self):
        """Known class returns appropriate abbreviation."""
        assert get_source_abbreviation("FunctionalNetworkMapping") == "FunctionalNetworkMapping"
        assert get_source_abbreviation("SubjectData") == "InputMask"

    def test_unknown_class_raises(self):
        """Unknown class raises KeyError."""
        with pytest.raises(KeyError, match="Unknown analysis class"):
            get_source_abbreviation("UnknownClass")

    def test_error_message_lists_known(self):
        """Error message lists known classes."""
        with pytest.raises(KeyError, match="FunctionalNetworkMapping"):
            get_source_abbreviation("BadClass")


class TestToBidsLabel:
    """Tests for to_bids_label function."""

    def test_removes_underscores_and_lowercases(self):
        """Underscores are removed and value is lowercased."""
        assert to_bids_label("correlation_map") == "correlationmap"

    def test_multiple_underscores(self):
        """Multiple underscores are all removed."""
        assert to_bids_label("some_long_desc_name") == "somelongdescname"

    def test_no_underscore_lowercased(self):
        """Values without underscores are just lowercased."""
        assert to_bids_label("HCP1065") == "hcp1065"

    def test_empty_string(self):
        """Empty string is unchanged."""
        assert to_bids_label("") == ""

    def test_already_lowercase(self):
        """Already lowercase values are unchanged."""
        assert to_bids_label("schaefer100") == "schaefer100"


class TestSplitAtlasName:
    """Tests for split_atlas_name function."""

    def test_schaefer_parcellation(self):
        """Schaefer2018_100Parcels7Networks splits correctly."""
        atlas, desc = split_atlas_name("Schaefer2018_100Parcels7Networks")
        assert atlas == "schaefer2018"
        assert desc == "100parcels7networks"

    def test_schaefer_1000_parcels(self):
        """Schaefer2018_1000Parcels7Networks splits correctly."""
        atlas, desc = split_atlas_name("Schaefer2018_1000Parcels7Networks")
        assert atlas == "schaefer2018"
        assert desc == "1000parcels7networks"

    def test_tian_subcortex(self):
        """TianSubcortex_3TS1 splits correctly."""
        atlas, desc = split_atlas_name("TianSubcortex_3TS1")
        assert atlas == "tiansubcortex"
        assert desc == "3ts1"

    def test_hcp_thresholded(self):
        """HCP1065_thr0p1 splits correctly."""
        atlas, desc = split_atlas_name("HCP1065_thr0p1")
        assert atlas == "hcp1065"
        assert desc == "thr0p1"

    def test_simple_name_no_underscore(self):
        """HCP1065 returns None for description (no underscore)."""
        atlas, desc = split_atlas_name("HCP1065")
        assert atlas == "hcp1065"
        assert desc is None

    def test_multiple_underscores_combines_rest(self):
        """Multiple underscores: rest after first is combined."""
        atlas, desc = split_atlas_name("Custom_Atlas_V2")
        assert atlas == "custom"
        assert desc == "atlasv2"  # Remaining underscores removed


class TestFormatBidsExportFilename:
    """Tests for format_bids_export_filename function."""

    def test_simple_fnm_key_no_desc_prefix(self):
        """FNM simple keys use fnm prefix without desc- entity."""
        result = format_bids_export_filename("rmap", "map")
        assert result == "fnmrmap"  # No desc- prefix, no _stat suffix

    def test_simple_fnm_key_with_underscore_converted(self):
        """FNM keys with underscores are converted to lowercase with fnm prefix."""
        result = format_bids_export_filename("correlation_map", "map")
        assert result == "fnmcorrelationmap"  # underscore removed, fnm prefix, no _stat suffix

    def test_simple_snm_key_no_desc_prefix(self):
        """SNM simple keys use snm prefix without desc- entity."""
        result = format_bids_export_filename("disconnectionmap", "map")
        assert result == "snmdisconnectionmap"  # No desc- prefix, no _stat suffix

    def test_simple_key_desc_prefix_for_unknown(self):
        """Unknown simple keys use desc- prefix."""
        result = format_bids_export_filename("customoutput", "map")
        assert result == "desc-customoutput"  # desc- prefix for unknown, no _stat suffix

    def test_bids_key_with_redundant_desc_omitted(self):
        """Redundant desc (when it matches source) is omitted."""
        result = format_bids_export_filename("atlas-Schaefer100_source-InputMask", "values")
        # InputMask source has no desc to omit
        # Uses atlas- and parcelstats instead of values
        # Schaefer100 has no underscore, so no separate desc from atlas split
        assert result == "atlas-schaefer100_source-inputmask_parcelstats"

    def test_bids_key_with_nonredundant_desc_included(self):
        """Non-redundant desc (different info than source) is included."""
        result = format_bids_export_filename(
            "atlas-Schaefer100_source-RegionalDamage_desc-damagescore", "values"
        )
        # damagescore doesn't map to regionaldamage, so include desc
        assert result == "atlas-schaefer100_source-regionaldamage_desc-damagescore_parcelstats"

    def test_parcellation_with_underscore_splits_to_desc(self):
        """Parcellation names with underscores split into atlas and desc entities."""
        result = format_bids_export_filename(
            "atlas-Schaefer2018_100Parcels7Networks_source-InputMask", "values"
        )
        # Schaefer2018_100Parcels7Networks splits into:
        # atlas-schaefer2018_desc-100parcels7networks
        assert "atlas-schaefer2018" in result
        assert "desc-100parcels7networks" in result
        # No underscores in values
        assert "2018_100" not in result

    def test_hcp_atlas_splits_correctly(self):
        """HCP1065_thr0p1 splits into atlas-hcp1065_desc-thr0p1."""
        result = format_bids_export_filename("atlas-HCP1065_thr0p1_source-InputMask", "values")
        assert "atlas-hcp1065" in result
        assert "desc-thr0p1" in result

    def test_tian_atlas_splits_correctly(self):
        """TianSubcortex_3TS1 splits into atlas-tiansubcortex_desc-3ts1."""
        result = format_bids_export_filename("atlas-TianSubcortex_3TS1_source-InputMask", "values")
        assert "atlas-tiansubcortex" in result
        assert "desc-3ts1" in result

    def test_fnm_correlation_map_redundant(self):
        """FNM correlation_map desc is redundant with fnm source."""
        result = format_bids_export_filename(
            "atlas-HCP1065_source-FunctionalNetworkMapping_desc-rmap", "values"
        )
        assert result == "atlas-hcp1065_source-fnm_parcelstats"
        assert "rmap" not in result

    def test_suffix_mapping_values_to_parcelstats(self):
        """Internal 'values' suffix maps to BIDS 'parcelstats'."""
        result = format_bids_export_filename("test", "values")
        assert result.endswith("_parcelstats")

    def test_suffix_mapping_map_to_empty(self):
        """Internal 'map' suffix maps to no suffix for VoxelMaps."""
        result = format_bids_export_filename("test", "map")
        # VoxelMaps don't have _stat suffix
        assert not result.endswith("_stat")
        assert result == "desc-test"

    def test_suffix_mapping_connmatrix_unchanged(self):
        """Connmatrix suffix is unchanged (valid BIDS derivative)."""
        result = format_bids_export_filename("test", "connmatrix")
        assert result.endswith("_connmatrix")

    def test_underscores_only_separate_key_value_pairs(self):
        """Verify underscores only appear between BIDS key-value pairs."""
        test_cases = [
            ("atlas-Schaefer2018_100Parcels7Networks_source-InputMask", "values", "parcelstats"),
            ("atlas-HCP1065_source-RegionalDamage_desc-damage_score", "values", "parcelstats"),
        ]

        # BIDS entity pattern: key-value where key is lowercase letters, value is alphanumeric
        bids_entity_pattern = re.compile(r"^[a-z]+-[a-z0-9]+$")

        for result_key, suffix, expected_suffix in test_cases:
            filename = format_bids_export_filename(result_key, suffix)

            # Split by underscore
            parts = filename.split("_")

            # All parts except the last (suffix) should be key-value pairs
            for part in parts[:-1]:
                assert bids_entity_pattern.match(part), (
                    f"Part '{part}' in '{filename}' is not a valid BIDS key-value pair. "
                    "Values must be lowercase alphanumeric, underscores only separate pairs."
                )

            # Last part should be the BIDS suffix
            assert parts[-1] == expected_suffix, f"Last part should be suffix '{expected_suffix}'"

    def test_source_abbreviations_applied(self):
        """Source names are converted to short abbreviations."""
        result = format_bids_export_filename(
            "atlas-Test_source-FunctionalNetworkMapping_desc-test", "values"
        )
        assert "source-fnm" in result
        assert "FunctionalNetworkMapping" not in result

    def test_inputmask_source_abbreviation(self):
        """InputMask source maps to inputmask abbreviation."""
        result = format_bids_export_filename("atlas-Test_source-InputMask", "values")
        assert "source-inputmask" in result

    def test_legacy_parc_input_uses_atlas_output(self):
        """Legacy parc- input is accepted and converted to atlas- output."""
        result = format_bids_export_filename("parc-Schaefer100_source-InputMask", "values")
        assert "atlas-" in result
        assert "parc-" not in result
