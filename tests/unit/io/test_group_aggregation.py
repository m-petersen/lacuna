"""Tests for group-level parcelstats aggregation.

Tests the aggregate_parcelstats function which combines subject-level
parcelstats TSV files into group-level DataFrames.
"""

import json

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
        filename = "sub-CAS001_ses-01_label-acuteinfarct_method-fnm_atlas-schaefer2018parcels100networks7_desc-rmap_parcelstats.tsv"
        result = _parse_bids_filename(filename)
        assert result["sub"] == "CAS001"
        assert result["ses"] == "01"
        assert result["label"] == "acuteinfarct"
        assert result["atlas"] == "schaefer2018parcels100networks7"
        assert result["method"] == "fnm"

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
        assert "ses-01" in result
        assert "label-lesion" in result
        assert "atlas-schaefer" in result
        assert "parcelstats" in result

    def test_extract_output_type_consistent_across_subjects(self):
        """Test that output type is consistent across different subjects with same ses/label."""
        file1 = "sub-001_ses-01_label-lesion_atlas-schaefer_parcelstats.tsv"
        file2 = "sub-002_ses-01_label-lesion_atlas-schaefer_parcelstats.tsv"

        result1 = _extract_output_type(file1)
        result2 = _extract_output_type(file2)

        assert result1 == result2

    def test_extract_output_type_differs_for_different_sessions(self):
        """Test that different sessions produce different output types."""
        file1 = "sub-001_ses-01_label-lesion_atlas-schaefer_parcelstats.tsv"
        file2 = "sub-001_ses-02_label-lesion_atlas-schaefer_parcelstats.tsv"

        assert _extract_output_type(file1) != _extract_output_type(file2)

    def test_extract_output_type_differs_for_different_labels(self):
        """Test that different labels produce different output types."""
        file1 = "sub-001_ses-01_label-lesion_atlas-schaefer_parcelstats.tsv"
        file2 = "sub-001_ses-01_label-wmh_atlas-schaefer_parcelstats.tsv"

        assert _extract_output_type(file1) != _extract_output_type(file2)


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
                / f"sub-{sub_id}_ses-01_label-lesion_method-fnm_atlas-schaefer100parcels_desc-rmap_parcelstats.tsv"
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
        result = aggregate_parcelstats(sample_derivatives, progress=False)

        assert len(result) == 1
        group_file = list(result.values())[0]
        assert group_file.exists()
        assert group_file.name.startswith("group_")
        assert group_file.suffix == ".tsv"

    def test_aggregate_parcelstats_correct_structure(self, sample_derivatives):
        """Test that the aggregated file has correct structure."""
        result = aggregate_parcelstats(sample_derivatives, progress=False)
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
        result = aggregate_parcelstats(sample_derivatives, progress=False)
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
        result = aggregate_parcelstats(sample_derivatives, progress=False)
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
        result1 = aggregate_parcelstats(sample_derivatives, progress=False)
        group_file = list(result1.values())[0]
        original_mtime = group_file.stat().st_mtime

        # Second aggregation without overwrite should skip
        aggregate_parcelstats(sample_derivatives, overwrite=False, progress=False)
        assert group_file.stat().st_mtime == original_mtime

        # Third aggregation with overwrite should update
        aggregate_parcelstats(sample_derivatives, overwrite=True, progress=False)
        assert group_file.stat().st_mtime >= original_mtime

    def test_aggregate_parcelstats_multiple_output_types(self, tmp_path):
        """Test aggregation with multiple output types."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        # Create subject with multiple output types
        sub_dir = derivatives_dir / "sub-001" / "ses-01" / "anat"
        sub_dir.mkdir(parents=True)

        # Create two different parcelstats types
        for output_type in ["method-fnm_desc-rmap", "method-snm_desc-disconnectionpct"]:
            tsv_path = sub_dir / f"sub-001_ses-01_label-lesion_{output_type}_parcelstats.tsv"
            df = pd.DataFrame({"region": ["A", "B"], "value": [0.1, 0.2]})
            df.to_csv(tsv_path, sep="\t", index=False)

        result = aggregate_parcelstats(derivatives_dir, progress=False)

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
                / f"sub-{sub_id}_ses-{ses_id}_label-{label}_method-fnm_atlas-schaefer2018parcels100networks7_desc-rmap_parcelstats.tsv"
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
                / f"sub-{sub_id}_ses-{ses_id}_label-{label}_method-snm_atlas-schaefer2018parcels100networks7_desc-disconnectionpct_parcelstats.tsv"
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
        result = aggregate_parcelstats(realistic_derivatives, progress=False)

        # Should create 4 group files:
        # ses-01_label-acuteinfarct x FNM + SNM = 2
        # ses-02_label-chronicinfarct x FNM + SNM = 2
        assert len(result) == 4

        # The ses-01/acuteinfarct groups should have 2 subjects (CAS001, CAS002)
        acute_groups = {
            k: v for k, v in result.items() if "ses-01" in k and "label-acuteinfarct" in k
        }
        assert len(acute_groups) == 2
        for _output_type, group_file in acute_groups.items():
            df = pd.read_csv(group_file, sep="\t")
            assert len(df) == 2
            assert "participant_id" in df.columns
            assert "session_id" in df.columns
            assert "label" in df.columns

        # The ses-02/chronicinfarct groups should have 1 subject (CAS003)
        chronic_groups = {
            k: v for k, v in result.items() if "ses-02" in k and "label-chronicinfarct" in k
        }
        assert len(chronic_groups) == 2
        for _output_type, group_file in chronic_groups.items():
            df = pd.read_csv(group_file, sep="\t")
            assert len(df) == 1

    def test_aggregate_to_custom_output_dir(self, realistic_derivatives, tmp_path):
        """Test aggregation to a custom output directory."""
        output_dir = tmp_path / "group_results"

        result = aggregate_parcelstats(
            realistic_derivatives,
            output_dir=output_dir,
            progress=False,
        )

        # Files should be in custom output directory
        for group_file in result.values():
            assert group_file.parent == output_dir


class TestAggregateGroupingBehavior:
    """Tests for per-output-type grouping behavior."""

    def test_same_output_across_subjects_produces_one_group(self, tmp_path):
        """Test that identical ses/label/method/atlas across subjects yields one group."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        for sub_id in ["001", "002", "003"]:
            sub_dir = derivatives_dir / f"sub-{sub_id}" / "ses-01" / "anat"
            sub_dir.mkdir(parents=True)

            tsv_path = (
                sub_dir
                / f"sub-{sub_id}_ses-01_label-lesion_method-rd_atlas-jhu_desc-damagepct_parcelstats.tsv"
            )
            df = pd.DataFrame({"region": ["A", "B"], "value": [0.1, 0.2]})
            df.to_csv(tsv_path, sep="\t", index=False)

        result = aggregate_parcelstats(derivatives_dir, progress=False)

        assert len(result) == 1
        group_file = list(result.values())[0]
        df = pd.read_csv(group_file, sep="\t")
        assert len(df) == 3

    def test_different_labels_produce_separate_groups(self, tmp_path):
        """Test that different labels produce separate group files."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        for sub_id, label in [("001", "lesion"), ("002", "lesion"), ("003", "wmh")]:
            sub_dir = derivatives_dir / f"sub-{sub_id}" / "ses-01" / "anat"
            sub_dir.mkdir(parents=True)

            tsv_path = (
                sub_dir / f"sub-{sub_id}_ses-01_label-{label}_method-rd_atlas-jhu_parcelstats.tsv"
            )
            df = pd.DataFrame({"region": ["A"], "value": [0.5]})
            df.to_csv(tsv_path, sep="\t", index=False)

        result = aggregate_parcelstats(derivatives_dir, progress=False)

        assert len(result) == 2
        # lesion group has 2, wmh group has 1
        counts = sorted(pd.read_csv(f, sep="\t").shape[0] for f in result.values())
        assert counts == [1, 2]

    def test_different_sessions_produce_separate_groups(self, tmp_path):
        """Test that different sessions produce separate group files."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        for sub_id, ses_id in [("001", "01"), ("002", "01"), ("003", "02")]:
            sub_dir = derivatives_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "anat"
            sub_dir.mkdir(parents=True)

            tsv_path = (
                sub_dir
                / f"sub-{sub_id}_ses-{ses_id}_label-lesion_method-rd_atlas-jhu_parcelstats.tsv"
            )
            df = pd.DataFrame({"region": ["A"], "value": [0.5]})
            df.to_csv(tsv_path, sep="\t", index=False)

        result = aggregate_parcelstats(derivatives_dir, progress=False)

        assert len(result) == 2
        counts = sorted(pd.read_csv(f, sep="\t").shape[0] for f in result.values())
        assert counts == [1, 2]

    def test_group_filename_includes_ses_and_label(self, tmp_path):
        """Test that group filenames include session and label entities."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        sub_dir = derivatives_dir / "sub-001" / "ses-01" / "anat"
        sub_dir.mkdir(parents=True)

        tsv_path = sub_dir / "sub-001_ses-01_label-acuteinfarct_method-rd_atlas-jhu_parcelstats.tsv"
        df = pd.DataFrame({"region": ["A"], "value": [0.5]})
        df.to_csv(tsv_path, sep="\t", index=False)

        result = aggregate_parcelstats(derivatives_dir, progress=False)

        group_file = list(result.values())[0]
        assert "ses-01" in group_file.name
        assert "label-acuteinfarct" in group_file.name
        assert "sub-" not in group_file.name

    def test_progress_false_suppresses_tqdm(self, tmp_path, capsys):
        """Test that progress=False suppresses tqdm output."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        sub_dir = derivatives_dir / "sub-001" / "ses-01" / "anat"
        sub_dir.mkdir(parents=True)
        tsv_path = sub_dir / "sub-001_ses-01_label-lesion_method-rd_parcelstats.tsv"
        df = pd.DataFrame({"region": ["A"], "value": [0.1]})
        df.to_csv(tsv_path, sep="\t", index=False)

        aggregate_parcelstats(derivatives_dir, progress=False)
        captured = capsys.readouterr()
        assert "Collecting" not in captured.err
        assert "Reading" not in captured.err


class TestAggregateAnalysisFilter:
    """Tests for the analysis_filter parameter of aggregate_parcelstats."""

    @pytest.fixture
    def mixed_derivatives(self, tmp_path):
        """Create derivatives with FNM, SNM, and AFNM parcelstats."""
        derivatives_dir = tmp_path / "lacuna"
        derivatives_dir.mkdir()

        for sub_id in ["001", "002"]:
            sub_dir = derivatives_dir / f"sub-{sub_id}" / "ses-01" / "anat"
            sub_dir.mkdir(parents=True)
            df = pd.DataFrame({"region": ["A", "B"], "value": [0.1, 0.2]})

            for fname in [
                f"sub-{sub_id}_ses-01_label-lesion_method-fnm_atlas-schaefer_desc-rmap_parcelstats.tsv",
                f"sub-{sub_id}_ses-01_label-lesion_method-snm_atlas-schaefer_desc-disconnectionpct_parcelstats.tsv",
                f"sub-{sub_id}_ses-01_label-lesion_method-afnm_atlas-schaefer_desc-zmap_parcelstats.tsv",
                f"sub-{sub_id}_ses-01_label-lesion_method-afnm_atlas-schaefer_desc-afnmstatistics_parcelstats.tsv",
            ]:
                (sub_dir / fname).write_text(df.to_csv(sep="\t", index=False))

        return derivatives_dir

    def test_filter_afnm_returns_only_afnm(self, mixed_derivatives):
        """Filtering by 'afnm' returns only AFNM parcelstats files."""
        result = aggregate_parcelstats(
            mixed_derivatives, analysis_filter="afnm", progress=False
        )
        for key in result:
            assert "afnm" in key.lower()

    def test_filter_fnm_excludes_afnm(self, mixed_derivatives):
        """Filtering by 'fnm' does not return AFNM files."""
        result = aggregate_parcelstats(
            mixed_derivatives, analysis_filter="fnm", progress=False
        )
        for key in result:
            assert "afnm" not in key.lower()

    def test_filter_afnm_case_insensitive(self, mixed_derivatives):
        """The filter is case-insensitive."""
        result_lower = aggregate_parcelstats(
            mixed_derivatives, analysis_filter="afnm", progress=False
        )
        result_upper = aggregate_parcelstats(
            mixed_derivatives, analysis_filter="AFNM", progress=False, overwrite=True
        )
        assert set(result_lower.keys()) == set(result_upper.keys())

    def test_filter_no_match_raises(self, mixed_derivatives):
        """An analysis filter with no matches raises BidsError."""
        with pytest.raises(BidsError, match="No parcelstats files found"):
            aggregate_parcelstats(
                mixed_derivatives, analysis_filter="nonexistent", progress=False
            )
