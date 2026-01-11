"""Tests for group-level parcelstats aggregation.

Tests the aggregate_parcelstats function which combines subject-level
parcelstats TSV files into group-level DataFrames.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from lacuna.io.bids import (
    BidsError,
    _extract_output_type,
    _parse_bids_filename,
    aggregate_parcelstats,
)


class TestParseBidsFilename:
    """Tests for _parse_bids_filename helper."""

    def test_parse_simple_filename(self):
        """Test parsing a simple BIDS filename."""
        result = _parse_bids_filename("sub-001_ses-01_label-lesion_parcelstats.tsv")
        assert result["sub"] == "001"
        assert result["ses"] == "01"
        assert result["label"] == "lesion"

    def test_parse_complex_filename(self):
        """Test parsing a complex BIDS filename with atlas entities."""
        filename = "sub-CAS001_ses-01_label-acuteinfarct_atlas-schaefer2018_desc-100parcels7networks_source-fnm_desc-rmap_parcelstats.tsv"
        result = _parse_bids_filename(filename)
        assert result["sub"] == "CAS001"
        assert result["ses"] == "01"
        assert result["label"] == "acuteinfarct"
        assert result["atlas"] == "schaefer2018"
        assert result["source"] == "fnm"

    def test_parse_no_session(self):
        """Test parsing filename without session."""
        result = _parse_bids_filename("sub-001_label-lesion_parcelstats.tsv")
        assert result["sub"] == "001"
        assert "ses" not in result
        assert result["label"] == "lesion"


class TestExtractOutputType:
    """Tests for _extract_output_type helper."""

    def test_extract_output_type_removes_subject_entities(self):
        """Test that subject-specific entities are removed."""
        filename = "sub-001_ses-01_label-lesion_atlas-schaefer_parcelstats.tsv"
        result = _extract_output_type(filename)
        assert "sub-" not in result
        assert "ses-" not in result
        assert "label-" not in result
        assert "atlas-schaefer" in result
        assert "parcelstats" in result

    def test_extract_output_type_consistent_across_subjects(self):
        """Test that output type is consistent across different subjects."""
        file1 = "sub-001_ses-01_label-lesion_atlas-schaefer_parcelstats.tsv"
        file2 = "sub-002_ses-02_label-wmh_atlas-schaefer_parcelstats.tsv"

        result1 = _extract_output_type(file1)
        result2 = _extract_output_type(file2)

        assert result1 == result2


class TestAggregateParcelstats:
    """Tests for aggregate_parcelstats function."""

    @pytest.fixture
    def sample_derivatives(self, tmp_path):
        """Create a sample derivatives directory with parcelstats files."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        # Create subject directories with parcelstats files
        for sub_id in ["001", "002", "003"]:
            sub_dir = derivatives_dir / f"sub-{sub_id}" / "ses-01" / "anat"
            sub_dir.mkdir(parents=True)

            # Create parcelstats TSV for each subject
            tsv_path = (
                sub_dir
                / f"sub-{sub_id}_ses-01_label-lesion_atlas-schaefer_desc-100parcels_source-fnm_desc-rmap_parcelstats.tsv"
            )
            df = pd.DataFrame(
                {
                    "region": ["Region_A", "Region_B", "Region_C"],
                    "value": [0.1 * int(sub_id), 0.2 * int(sub_id), 0.3 * int(sub_id)],
                }
            )
            df.to_csv(tsv_path, sep="\t", index=False)

        return derivatives_dir

    def test_aggregate_parcelstats_creates_group_file(self, sample_derivatives):
        """Test that aggregation creates a group-level TSV file."""
        result = aggregate_parcelstats(sample_derivatives)

        assert len(result) == 1
        group_file = list(result.values())[0]
        assert group_file.exists()
        assert group_file.name.startswith("group_")
        assert group_file.suffix == ".tsv"

    def test_aggregate_parcelstats_correct_structure(self, sample_derivatives):
        """Test that the aggregated file has correct structure."""
        result = aggregate_parcelstats(sample_derivatives)
        group_file = list(result.values())[0]

        df = pd.read_csv(group_file, sep="\t")

        # Should have 3 subjects
        assert len(df) == 3

        # Should have participant_id column
        assert "participant_id" in df.columns

        # Should have region columns
        assert "Region_A" in df.columns
        assert "Region_B" in df.columns
        assert "Region_C" in df.columns

    def test_aggregate_parcelstats_correct_values(self, sample_derivatives):
        """Test that aggregated values are correct."""
        result = aggregate_parcelstats(sample_derivatives)
        group_file = list(result.values())[0]

        df = pd.read_csv(group_file, sep="\t")
        df = df.sort_values("participant_id")

        # Check values for subject 001
        # participant_id may be read as int (1) or string ("001") depending on pandas
        sub_001 = df[df["participant_id"].astype(str).isin(["001", "1"])].iloc[0]
        assert sub_001["Region_A"] == pytest.approx(0.1)
        assert sub_001["Region_B"] == pytest.approx(0.2)

    def test_aggregate_parcelstats_creates_sidecar(self, sample_derivatives):
        """Test that a JSON sidecar is created."""
        result = aggregate_parcelstats(sample_derivatives)
        group_file = list(result.values())[0]
        sidecar_path = group_file.with_suffix(".json")

        assert sidecar_path.exists()

        with open(sidecar_path) as f:
            sidecar = json.load(f)

        assert "Description" in sidecar
        assert "NumberOfSubjects" in sidecar
        assert sidecar["NumberOfSubjects"] == 3

    def test_aggregate_parcelstats_empty_directory_raises(self, tmp_path):
        """Test that empty directory raises BidsError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(BidsError, match="No parcelstats files found"):
            aggregate_parcelstats(empty_dir)

    def test_aggregate_parcelstats_nonexistent_dir_raises(self, tmp_path):
        """Test that nonexistent directory raises BidsError."""
        with pytest.raises(BidsError, match="Derivatives directory not found"):
            aggregate_parcelstats(tmp_path / "nonexistent")

    def test_aggregate_parcelstats_overwrite(self, sample_derivatives):
        """Test that overwrite works correctly."""
        # First aggregation
        result1 = aggregate_parcelstats(sample_derivatives)
        group_file = list(result1.values())[0]
        original_mtime = group_file.stat().st_mtime

        # Second aggregation without overwrite should skip
        result2 = aggregate_parcelstats(sample_derivatives, overwrite=False)
        assert group_file.stat().st_mtime == original_mtime

        # Third aggregation with overwrite should update
        result3 = aggregate_parcelstats(sample_derivatives, overwrite=True)
        assert group_file.stat().st_mtime >= original_mtime

    def test_aggregate_parcelstats_multiple_output_types(self, tmp_path):
        """Test aggregation with multiple output types."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        # Create subject with multiple output types
        sub_dir = derivatives_dir / "sub-001" / "ses-01" / "anat"
        sub_dir.mkdir(parents=True)

        # Create two different parcelstats types
        for output_type in ["fnm_desc-rmap", "snm_desc-disconnectionmap"]:
            tsv_path = sub_dir / f"sub-001_ses-01_label-lesion_source-{output_type}_parcelstats.tsv"
            df = pd.DataFrame({"region": ["A", "B"], "value": [0.1, 0.2]})
            df.to_csv(tsv_path, sep="\t", index=False)

        result = aggregate_parcelstats(derivatives_dir)

        # Should create two group files
        assert len(result) == 2


class TestAggregateParcelstatsIntegration:
    """Integration tests for aggregate_parcelstats with CLI-like structure."""

    @pytest.fixture
    def realistic_derivatives(self, tmp_path):
        """Create realistic BIDS derivatives structure."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        # Create dataset_description.json
        with open(derivatives_dir / "dataset_description.json", "w") as f:
            json.dump({"Name": "lacuna", "BIDSVersion": "1.9.0"}, f)

        # Create multiple subjects with realistic filenames
        subjects = [
            ("CAS001", "01", "acuteinfarct"),
            ("CAS002", "01", "acuteinfarct"),
            ("CAS003", "02", "chronicinfarct"),
        ]

        for sub_id, ses_id, label in subjects:
            sub_dir = derivatives_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "anat"
            sub_dir.mkdir(parents=True)

            # FNM parcelstats
            fnm_tsv = (
                sub_dir
                / f"sub-{sub_id}_ses-{ses_id}_label-{label}_atlas-schaefer2018_desc-100parcels7networks_source-fnm_desc-rmap_parcelstats.tsv"
            )
            df = pd.DataFrame(
                {
                    "region": [f"Region_{i}" for i in range(10)],
                    "value": [0.1 * (i + 1) for i in range(10)],
                }
            )
            df.to_csv(fnm_tsv, sep="\t", index=False)

            # SNM parcelstats
            snm_tsv = (
                sub_dir
                / f"sub-{sub_id}_ses-{ses_id}_label-{label}_atlas-schaefer2018_desc-100parcels7networks_source-snm_desc-disconnectionmap_parcelstats.tsv"
            )
            df = pd.DataFrame(
                {
                    "region": [f"Region_{i}" for i in range(10)],
                    "value": [1.0 * (i + 1) for i in range(10)],
                }
            )
            df.to_csv(snm_tsv, sep="\t", index=False)

        return derivatives_dir

    def test_aggregate_realistic_structure(self, realistic_derivatives):
        """Test aggregation with realistic BIDS structure."""
        result = aggregate_parcelstats(realistic_derivatives)

        # Should create two group files (FNM and SNM)
        assert len(result) == 2

        # Each file should have 3 subjects
        for output_type, group_file in result.items():
            df = pd.read_csv(group_file, sep="\t")
            assert len(df) == 3
            assert "participant_id" in df.columns
            assert "session_id" in df.columns
            assert "label" in df.columns

    def test_aggregate_to_custom_output_dir(self, realistic_derivatives, tmp_path):
        """Test aggregation to a custom output directory."""
        output_dir = tmp_path / "group_results"

        result = aggregate_parcelstats(
            realistic_derivatives, output_dir=output_dir
        )

        # Files should be in custom output directory
        for group_file in result.values():
            assert group_file.parent == output_dir
