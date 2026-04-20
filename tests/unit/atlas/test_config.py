"""Tests for lacuna.atlas.config."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from lacuna.atlas.config import (
    ALL_TARGETS,
    NT_PRESETS,
    NT_TARGET_GROUPS,
    parse_map_selection,
    parse_publication_from_filename,
    parse_target_from_filename,
    resolve_targets,
)


# ---------------------------------------------------------------------------
# NT_TARGET_GROUPS and ALL_TARGETS
# ---------------------------------------------------------------------------


class TestNtTargetGroups:
    def test_expected_systems_present(self):
        expected = {
            "serotonergic",
            "dopaminergic",
            "cholinergic",
            "noradrenergic",
            "gabaergic",
            "cannabinoid",
            "opioid",
            "histaminergic",
            "glutamatergic",
            "vesicular",
        }
        assert set(NT_TARGET_GROUPS.keys()) == expected

    def test_serotonergic_targets(self):
        assert NT_TARGET_GROUPS["serotonergic"] == [
            "5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT"
        ]

    def test_dopaminergic_targets(self):
        assert NT_TARGET_GROUPS["dopaminergic"] == ["D1", "D23", "DAT", "FDOPA"]

    def test_cholinergic_targets(self):
        assert NT_TARGET_GROUPS["cholinergic"] == ["VAChT", "M1", "A4B2"]

    def test_noradrenergic_targets(self):
        assert NT_TARGET_GROUPS["noradrenergic"] == ["NET"]

    def test_gabaergic_targets(self):
        assert NT_TARGET_GROUPS["gabaergic"] == ["GABAa", "GABAa5"]

    def test_cannabinoid_targets(self):
        assert NT_TARGET_GROUPS["cannabinoid"] == ["CB1"]

    def test_opioid_targets(self):
        assert NT_TARGET_GROUPS["opioid"] == ["MOR", "KOR"]

    def test_histaminergic_targets(self):
        assert NT_TARGET_GROUPS["histaminergic"] == ["H3"]

    def test_glutamatergic_targets(self):
        assert NT_TARGET_GROUPS["glutamatergic"] == ["mGluR5", "NMDA"]

    def test_vesicular_targets(self):
        assert NT_TARGET_GROUPS["vesicular"] == ["VMAT2"]


class TestAllTargets:
    def test_is_sorted(self):
        assert ALL_TARGETS == sorted(ALL_TARGETS)

    def test_contains_all_group_targets(self):
        all_from_groups = {t for targets in NT_TARGET_GROUPS.values() for t in targets}
        assert set(ALL_TARGETS) == all_from_groups

    def test_no_duplicates(self):
        assert len(ALL_TARGETS) == len(set(ALL_TARGETS))

    def test_is_list(self):
        assert isinstance(ALL_TARGETS, list)


# ---------------------------------------------------------------------------
# NT_PRESETS
# ---------------------------------------------------------------------------


class TestNtPresets:
    def test_all_preset_equals_all_targets(self):
        assert NT_PRESETS["all"] == ALL_TARGETS

    def test_dopaminergic_preset(self):
        assert NT_PRESETS["dopaminergic"] == NT_TARGET_GROUPS["dopaminergic"]

    def test_serotonergic_preset(self):
        assert NT_PRESETS["serotonergic"] == NT_TARGET_GROUPS["serotonergic"]

    def test_cholinergic_preset(self):
        assert NT_PRESETS["cholinergic"] == NT_TARGET_GROUPS["cholinergic"]

    def test_monoaminergic_preset(self):
        expected = (
            NT_TARGET_GROUPS["serotonergic"]
            + NT_TARGET_GROUPS["dopaminergic"]
            + NT_TARGET_GROUPS["noradrenergic"]
        )
        assert NT_PRESETS["monoaminergic"] == expected

    def test_known_presets_present(self):
        assert {"all", "dopaminergic", "serotonergic", "cholinergic", "monoaminergic"} <= set(
            NT_PRESETS.keys()
        )


# ---------------------------------------------------------------------------
# parse_target_from_filename
# ---------------------------------------------------------------------------


class TestParseTargetFromFilename:
    def test_simple_target(self):
        assert parse_target_from_filename("target-5HT1a_space-MNI.nii.gz") == "5HT1a"

    def test_target_with_underscores_after(self):
        assert parse_target_from_filename("pub-beliveau2017_target-DAT_res-1mm.nii") == "DAT"

    def test_target_at_start(self):
        assert parse_target_from_filename("target-CB1_pub-hillmer2012.nii.gz") == "CB1"

    def test_raises_if_no_target(self):
        with pytest.raises(ValueError, match="target"):
            parse_target_from_filename("pub-beliveau2017_space-MNI.nii.gz")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            parse_target_from_filename("")

    def test_target_alphanumeric(self):
        assert parse_target_from_filename("target-GABAa5_res-2mm.nii") == "GABAa5"

    def test_target_uppercase_lowercase_mixed(self):
        assert parse_target_from_filename("target-mGluR5_res-2mm.nii") == "mGluR5"


# ---------------------------------------------------------------------------
# parse_publication_from_filename
# ---------------------------------------------------------------------------


class TestParsePublicationFromFilename:
    def test_simple_pub(self):
        assert (
            parse_publication_from_filename("pub-beliveau2017_target-5HT1a.nii.gz")
            == "beliveau2017"
        )

    def test_pub_at_end(self):
        assert (
            parse_publication_from_filename("target-DAT_pub-mccluskey2024") == "mccluskey2024"
        )

    def test_raises_if_no_pub(self):
        with pytest.raises(ValueError, match="pub"):
            parse_publication_from_filename("target-5HT1a_space-MNI.nii.gz")

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError):
            parse_publication_from_filename("")

    def test_pub_with_numbers_and_letters(self):
        assert (
            parse_publication_from_filename("pub-savli2012_target-5HT1b.nii") == "savli2012"
        )


# ---------------------------------------------------------------------------
# resolve_targets
# ---------------------------------------------------------------------------


class TestResolveTargets:
    AVAILABLE = ["5HT1a", "5HT1b", "DAT", "D1", "NET", "CB1"]

    def test_explicit_list_all_available(self):
        result = resolve_targets(["5HT1a", "DAT"], self.AVAILABLE)
        assert result == ["5HT1a", "DAT"]

    def test_explicit_list_preserves_order(self):
        result = resolve_targets(["DAT", "5HT1a"], self.AVAILABLE)
        assert result == ["DAT", "5HT1a"]

    def test_raises_if_target_not_available(self):
        with pytest.raises(ValueError, match="VMAT2"):
            resolve_targets(["VMAT2"], self.AVAILABLE)

    def test_preset_dopaminergic(self):
        available = ["D1", "D23", "DAT", "FDOPA", "5HT1a"]
        result = resolve_targets("dopaminergic", available)
        assert set(result) == {"D1", "D23", "DAT", "FDOPA"}

    def test_preset_raises_if_target_not_available(self):
        # D23 and FDOPA not in available — should raise
        with pytest.raises(ValueError):
            resolve_targets("dopaminergic", ["D1", "DAT"])

    def test_preset_all_returns_available(self):
        result = resolve_targets("all", self.AVAILABLE)
        assert result == self.AVAILABLE

    def test_preset_all_does_not_raise_for_subset(self):
        # "all" preset should return whatever is available, no validation
        small = ["DAT", "CB1"]
        result = resolve_targets("all", small)
        assert result == small

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown preset"):
            resolve_targets("nonexistent_preset", self.AVAILABLE)

    def test_empty_explicit_list(self):
        result = resolve_targets([], self.AVAILABLE)
        assert result == []


# ---------------------------------------------------------------------------
# parse_map_selection
# ---------------------------------------------------------------------------


class TestParseMapSelection:
    def _write_yaml(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        tmp.write(content)
        tmp.flush()
        return Path(tmp.name)

    def test_returns_none_for_none_path(self):
        assert parse_map_selection(None) is None

    def test_single_string_wrapped_in_list(self):
        path = self._write_yaml(
            "targets:\n  5HT1a: beliveau2017\n"
        )
        result = parse_map_selection(path)
        assert result["5HT1a"] == ["beliveau2017"]

    def test_list_stays_list(self):
        path = self._write_yaml(
            "targets:\n  5HT1b:\n    - savli2012\n    - gallezot2010\n"
        )
        result = parse_map_selection(path)
        assert result["5HT1b"] == ["savli2012", "gallezot2010"]

    def test_all_literal(self):
        path = self._write_yaml(
            "targets:\n  D1: all\n"
        )
        result = parse_map_selection(path)
        assert result["D1"] == "all"

    def test_exclude_literal(self):
        path = self._write_yaml(
            "targets:\n  DAT: exclude\n"
        )
        result = parse_map_selection(path)
        assert result["DAT"] == "exclude"

    def test_mixed_entries(self):
        yaml_str = (
            "targets:\n"
            "  5HT1a: beliveau2017\n"
            "  5HT1b:\n"
            "    - savli2012\n"
            "    - gallezot2010\n"
            "  D1: all\n"
            "  DAT: exclude\n"
        )
        path = self._write_yaml(yaml_str)
        result = parse_map_selection(path)
        assert result["5HT1a"] == ["beliveau2017"]
        assert result["5HT1b"] == ["savli2012", "gallezot2010"]
        assert result["D1"] == "all"
        assert result["DAT"] == "exclude"

    def test_raises_if_no_targets_key(self):
        path = self._write_yaml("maps:\n  5HT1a: beliveau2017\n")
        with pytest.raises(ValueError, match="targets"):
            parse_map_selection(path)

    def test_returns_dict(self):
        path = self._write_yaml("targets:\n  NET: aston2009\n")
        result = parse_map_selection(path)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Target conflict detection
# ---------------------------------------------------------------------------


class TestTargetConflictDetection:
    def test_run_target_excluded_at_prepare_time(self):
        """Requesting a target at run time that was excluded during prepare."""
        available = ["D1", "5HT1a"]  # DAT was excluded
        with pytest.raises(ValueError, match="not available.*DAT"):
            resolve_targets(["D1", "DAT"], available)

    def test_helpful_error_message_lists_missing(self):
        """Error message should list the missing targets."""
        available = ["D1", "5HT1a"]
        with pytest.raises(ValueError, match="GABA"):
            resolve_targets(["GABA"], available)

    def test_multiple_missing_targets(self):
        """All missing targets should appear in error."""
        available = ["D1"]
        with pytest.raises(ValueError, match="not available"):
            resolve_targets(["DAT", "NET"], available)
