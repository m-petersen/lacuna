"""Unit tests for lacuna.core.keys module.

Tests the BIDS-style result key building and parsing utilities.
"""

import pytest

from lacuna.core.keys import (
    SOURCE_ABBREVIATIONS,
    BidsFilename,
    build_result_key,
    get_source_abbreviation,
    parse_result_key,
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
        key = build_result_key("tian2020parcels16", "SubjectData", "maskimg")
        # SubjectData maps to InputMask and desc is automatically omitted
        assert key == "atlas-tian2020parcels16_source-InputMask"

    def test_mask_source_no_desc_provided(self):
        """Build key with SubjectData source without desc."""
        key = build_result_key("tian2020parcels16", "SubjectData")
        assert key == "atlas-tian2020parcels16_source-InputMask"

    def test_parc_with_underscore(self):
        """Build key when parcellation name contains underscore."""
        key = build_result_key("Tian_S4", "StructuralNetworkMapping", "mean_connectivity")
        assert key == "atlas-Tian_S4_source-StructuralNetworkMapping_desc-mean_connectivity"

    def test_desc_with_underscore(self):
        """Build key when description contains underscore."""
        key = build_result_key("Schaefer100", "FocalDamage", "damagescore")
        assert key == "atlas-Schaefer100_source-FocalDamage_desc-damagescore"

    def test_all_source_names(self):
        """Build keys with all standard source names."""
        sources = [
            ("SubjectData", "InputMask"),  # Maps to InputMask
            ("FunctionalNetworkMapping", "FunctionalNetworkMapping"),
            ("StructuralNetworkMapping", "StructuralNetworkMapping"),
            ("FocalDamage", "FocalDamage"),
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
        result = parse_result_key("atlas-Schaefer100_source-FunctionalNetworkMapping_desc-rmap")
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
        result = parse_result_key("atlas-TianSubcortex_source-FocalDamage_desc-damagescore")
        assert result == {
            "atlas": "TianSubcortex",
            "source": "FocalDamage",
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

    def test_fd_maps_to_self(self):
        """FocalDamage maps to itself."""
        assert SOURCE_ABBREVIATIONS["FocalDamage"] == "FocalDamage"

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
        assert to_bids_label("Schaefer2018") == "schaefer2018"

    def test_empty_string(self):
        """Empty string is unchanged."""
        assert to_bids_label("") == ""

    def test_already_lowercase(self):
        """Already lowercase values are unchanged."""
        assert to_bids_label("schaefer100") == "schaefer100"


class TestBidsFilenameStr:
    """Tests for BidsFilename.__str__() entity ordering and omission."""

    def test_entity_ordering(self):
        """Entities are ordered: method > space > atlas > desc > suffix."""
        bf = BidsFilename(
            method="snm",
            space="MNI152NLin6Asym",
            atlas="schaefer2018parcels100networks7",
            desc="disconnectionpct",
            suffix="connmatrix",
        )
        result = str(bf)
        assert result == (
            "method-snm_space-MNI152NLin6Asym"
            "_atlas-schaefer2018parcels100networks7"
            "_desc-disconnectionpct_connmatrix"
        )

    def test_omit_none_fields(self):
        """None fields are omitted from output."""
        bf = BidsFilename(
            method="fnm",
            space="MNI152NLin6Asym",
            desc="rmap",
            suffix="",
        )
        result = str(bf)
        assert result == "method-fnm_space-MNI152NLin6Asym_desc-rmap"
        assert "atlas-" not in result

    def test_snm_voxelmap_disconnectionpct(self):
        """SNM voxelmap percentage output."""
        bf = BidsFilename(
            method="snm",
            space="MNI152NLin6Asym",
            desc="disconnectionpct",
        )
        result = str(bf)
        assert result == "method-snm_space-MNI152NLin6Asym_desc-disconnectionpct"

    def test_fnm_rmap(self):
        """FNM r-map output."""
        bf = BidsFilename(
            method="fnm",
            space="MNI152NLin6Asym",
            desc="rmap",
        )
        result = str(bf)
        assert result == "method-fnm_space-MNI152NLin6Asym_desc-rmap"

    def test_fd_parcelstats(self):
        """RD parcelstats output."""
        bf = BidsFilename(
            method="fd",
            atlas="schaefer2018parcels100networks7",
            desc="damagepct",
            suffix="parcelstats",
        )
        result = str(bf)
        assert (
            result == "method-fd_atlas-schaefer2018parcels100networks7_desc-damagepct_parcelstats"
        )

    def test_input_mask_no_method(self):
        """Input mask has no method entity."""
        bf = BidsFilename(
            space="MNI152NLin6Asym",
            suffix="mask",
        )
        result = str(bf)
        assert result == "space-MNI152NLin6Asym_mask"
        assert "method-" not in result

    def test_snm_connmatrix(self):
        """SNM connectivity matrix with atlas."""
        bf = BidsFilename(
            method="snm",
            atlas="schaefer2018parcels100networks7",
            desc="disconnectionpct",
            suffix="connmatrix",
        )
        result = str(bf)
        assert result == (
            "method-snm_atlas-schaefer2018parcels100networks7" "_desc-disconnectionpct_connmatrix"
        )

    def test_fnm_summarystatistics(self):
        """FNM summary statistics output."""
        bf = BidsFilename(
            method="fnm",
            desc="summarystatistics",
            suffix="stats",
        )
        result = str(bf)
        assert result == "method-fnm_desc-summarystatistics_stats"

    def test_suffix_only(self):
        """Suffix only (edge case)."""
        bf = BidsFilename(suffix="mask")
        result = str(bf)
        assert result == "mask"

    def test_empty_suffix_not_appended(self):
        """Empty string suffix is not appended."""
        bf = BidsFilename(method="fnm", desc="rmap", suffix="")
        result = str(bf)
        assert result == "method-fnm_desc-rmap"
        assert not result.endswith("_")


class TestBidsFilenameFromResultKey:
    """Tests for BidsFilename.from_result_key() classmethod."""

    def test_simple_fnm_key(self):
        """Simple key like 'rmap' resolves method from namespace."""
        bf = BidsFilename.from_result_key(
            "rmap", suffix="map", namespace="FunctionalNetworkMapping"
        )
        assert bf.method == "fnm"
        assert bf.desc == "rmap"

    def test_simple_snm_key(self):
        """Simple key 'disconnection_pct' resolves to snm."""
        bf = BidsFilename.from_result_key(
            "disconnection_pct", suffix="map", namespace="StructuralNetworkMapping"
        )
        assert bf.method == "snm"
        assert bf.desc == "disconnectionpct"

    def test_bids_key_with_atlas(self):
        """BIDS key with atlas entity merges into BidsFilename."""
        key = "atlas-schaefer2018parcels100networks7_source-StructuralNetworkMapping_desc-disconnectivity_percent"
        bf = BidsFilename.from_result_key(key, suffix="connmatrix")
        assert bf.method == "snm"
        assert bf.atlas == "schaefer2018parcels100networks7"
        assert bf.desc == "disconnectionpct"
        assert bf.suffix == "connmatrix"

    def test_bids_key_fd(self):
        """BIDS key with FocalDamage source."""
        key = "atlas-schaefer2018parcels100networks7_source-FocalDamage_desc-damagepct"
        bf = BidsFilename.from_result_key(key, suffix="values")
        assert bf.method == "fd"
        assert bf.atlas == "schaefer2018parcels100networks7"
        assert bf.desc == "damagepct"
        assert bf.suffix == "parcelstats"

    def test_namespace_provides_method(self):
        """Namespace is used to derive method when source is absent."""
        bf = BidsFilename.from_result_key(
            "summarystatistics",
            suffix="metrics",
            namespace="StructuralNetworkMapping",
        )
        assert bf.method == "snm"
        assert bf.desc == "summarystatistics"
        assert bf.suffix == "stats"

    def test_inputmask_key_no_method(self):
        """InputMask source produces no method entity."""
        key = "atlas-schaefer2018parcels100networks7_source-InputMask"
        bf = BidsFilename.from_result_key(key, suffix="values")
        assert bf.method is None
        assert bf.atlas == "schaefer2018parcels100networks7"
        assert bf.suffix == "parcelstats"

    def test_desc_override_disconnectivity_percent(self):
        """disconnectivity_percent is overridden to disconnectionpct."""
        key = "atlas-X_source-StructuralNetworkMapping_desc-disconnectivity_percent"
        bf = BidsFilename.from_result_key(key, suffix="connmatrix")
        assert bf.desc == "disconnectionpct"

    def test_desc_override_roi_disconnection(self):
        """roi_disconnection is overridden to disconnectionpct."""
        key = "atlas-X_source-StructuralNetworkMapping_desc-roi_disconnection"
        bf = BidsFilename.from_result_key(key, suffix="values")
        assert bf.desc == "disconnectionpct"

    def test_suffix_mapping_values_to_parcelstats(self):
        """Internal 'values' suffix maps to 'parcelstats'."""
        bf = BidsFilename.from_result_key(
            "rmap", suffix="values", namespace="FunctionalNetworkMapping"
        )
        assert bf.suffix == "parcelstats"

    def test_suffix_mapping_metrics_to_stats(self):
        """Internal 'metrics' suffix maps to 'stats'."""
        bf = BidsFilename.from_result_key(
            "summarystatistics", suffix="metrics", namespace="FunctionalNetworkMapping"
        )
        assert bf.suffix == "stats"

    def test_suffix_mapping_map_to_empty(self):
        """Internal 'map' suffix maps to empty string."""
        bf = BidsFilename.from_result_key(
            "rmap", suffix="map", namespace="FunctionalNetworkMapping"
        )
        assert bf.suffix == ""
