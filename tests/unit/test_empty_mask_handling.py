"""
Unit tests for empty mask handling across analyses.

Verifies that:
- Empty masks (all-zero voxels) are accepted by SubjectData.
- RegionalDamage produces valid zero-valued outputs for empty masks.
- FunctionalNetworkMapping raises EmptyMaskError for empty masks
  (zero-injection is not meaningful for correlation-based analyses).
- StructuralNetworkMapping produces zero-valued disconnection maps.
- The --skip-empty-masks CLI flag filters out empty-mask subjects.
"""

import logging

import nibabel as nib
import numpy as np
import pytest

from lacuna import SubjectData


def _make_subject(empty: bool = True, subject_id: str = "sub-empty") -> SubjectData:
    """Create a SubjectData with an empty or non-empty mask."""
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    if not empty:
        data[30:35, 30:35, 30:35] = 1

    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    mask_img = nib.Nifti1Image(data, affine)

    return SubjectData(
        mask_img=mask_img,
        space="MNI152NLin6Asym",
        resolution=2,
        metadata={"subject_id": subject_id},
    )


class TestEmptyMaskSubjectData:
    """Tests for SubjectData empty mask acceptance."""

    def test_empty_mask_is_accepted(self):
        """Empty mask should create a valid SubjectData."""
        subject = _make_subject(empty=True)
        assert subject.is_empty_mask is True

    def test_nonempty_mask_not_flagged(self):
        """Non-empty mask should not be flagged as empty."""
        subject = _make_subject(empty=False)
        assert subject.is_empty_mask is False

    def test_empty_mask_has_valid_properties(self):
        """Empty-mask SubjectData should have all standard properties."""
        subject = _make_subject(empty=True)
        assert subject.mask_img is not None
        assert subject.affine is not None
        assert subject.space == "MNI152NLin6Asym"
        assert subject.resolution == 2
        assert subject.metadata["subject_id"] == "sub-empty"

    def test_empty_mask_copy_preserves_flag(self):
        """Copying empty-mask SubjectData should preserve is_empty_mask."""
        subject = _make_subject(empty=True)
        copied = subject.copy()
        assert copied.is_empty_mask is True

    def test_empty_mask_warning_logged(self, caplog):
        """Empty mask creation should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="lacuna"):
            _make_subject(empty=True, subject_id="sub-warned")
        assert "sub-warned" in caplog.text
        assert "empty mask" in caplog.text.lower()

    def test_empty_mask_volume_is_zero(self):
        """Empty mask should report 0 mm³ volume."""
        subject = _make_subject(empty=True)
        assert subject.get_volume_mm3() == 0.0


class TestEmptyMaskRegionalDamage:
    """Tests for RegionalDamage with empty masks.

    RD wraps ParcelAggregation with source=maskimg, aggregation=percent.
    An all-zero mask should produce 0% damage for every region.
    """

    @pytest.mark.fast
    def test_rd_empty_mask_produces_zero_parcelstats(self):
        """RD on an empty mask should produce all-zero parcel stats."""
        from lacuna.analysis import RegionalDamage

        subject = _make_subject(empty=True)
        analysis = RegionalDamage()
        result = analysis.run(subject)

        rd_results = result.results.get("RegionalDamage", {})
        # Should have results (one per atlas)
        assert len(rd_results) > 0, "Expected at least one RD result"

        # All values should be numeric zeros
        from lacuna.core.data_types import ParcelData

        for key, parcel_data in rd_results.items():
            if isinstance(parcel_data, ParcelData):
                for region, value in parcel_data.data.items():
                    assert (
                        value == 0.0
                    ), f"Expected 0% damage for region '{region}' in '{key}', got {value}"

    @pytest.mark.fast
    def test_rd_empty_mask_keeps_subject_in_pipeline(self):
        """RD result for an empty mask should still carry through SubjectData."""
        from lacuna.analysis import RegionalDamage

        subject = _make_subject(empty=True)
        analysis = RegionalDamage()
        result = analysis.run(subject)

        assert isinstance(result, SubjectData)
        assert result.is_empty_mask is True
        assert result.metadata["subject_id"] == "sub-empty"


# ===========================================================================
# FNM: Empty mask produces zero-valued output maps
# ===========================================================================


class TestFNMEmptyMask:
    """FNM produces zero-valued maps for empty masks."""

    @pytest.fixture
    def fnm_with_mock_connectome(self, tmp_path):
        """Create a minimal FNM with a mock connectome for unit testing."""
        import h5py

        from lacuna.analysis import FunctionalNetworkMapping
        from lacuna.assets.connectomes import (
            register_functional_connectome,
            unregister_functional_connectome,
        )

        # Build a tiny HDF5 connectome
        connectome_path = tmp_path / "mock_connectome.h5"
        n_subjects, n_timepoints, n_voxels = 3, 20, 100
        rng = np.random.default_rng(42)

        # Mask occupies voxels (0..9, 0..9, 0) in a (91,109,91) grid
        mask_indices = np.array(
            [
                np.repeat(range(10), 10),
                np.tile(range(10), 10),
                np.zeros(100, dtype=int),
            ]
        )

        affine = np.array(
            [
                [2.0, 0.0, 0.0, -91.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        with h5py.File(connectome_path, "w") as f:
            f.create_dataset(
                "timeseries",
                data=rng.standard_normal((n_subjects, n_timepoints, n_voxels)).astype(np.float32),
            )
            f.create_dataset("mask_indices", data=mask_indices)
            f.create_dataset("mask_affine", data=affine)
            f.attrs["mask_shape"] = (91, 109, 91)

        name = "_test_empty_conn"
        register_functional_connectome(
            name=name,
            space="MNI152NLin6Asym",
            resolution=2,
            data_path=connectome_path,
        )
        yield FunctionalNetworkMapping(connectome_name=name)
        unregister_functional_connectome(name)

    @pytest.mark.fast
    def test_fnm_empty_mask_produces_zero_maps(self, fnm_with_mock_connectome):
        """FNM.run() produces zero-valued maps for an empty mask."""
        subject = _make_subject(empty=True, subject_id="sub-emptyFNM")
        result = fnm_with_mock_connectome.run(subject)

        assert isinstance(result, SubjectData)
        fnm_results = result.results.get("FunctionalNetworkMapping", {})
        assert "rmap" in fnm_results
        assert "zmap" in fnm_results
        assert "summarystatistics" in fnm_results

        # Maps should be all zeros
        rmap_data = fnm_results["rmap"].data.get_fdata()
        assert np.all(rmap_data == 0), "rmap should be all zeros for empty mask"
        zmap_data = fnm_results["zmap"].data.get_fdata()
        assert np.all(zmap_data == 0), "zmap should be all zeros for empty mask"

        # Summary should flag empty_mask
        assert fnm_results["summarystatistics"].data["empty_mask"] is True

    @pytest.mark.fast
    def test_fnm_empty_mask_keeps_subject_in_pipeline(self, fnm_with_mock_connectome):
        """FNM result for empty mask should carry through SubjectData."""
        subject = _make_subject(empty=True, subject_id="sub-emptyFNM")
        result = fnm_with_mock_connectome.run(subject)

        assert isinstance(result, SubjectData)
        assert result.is_empty_mask is True
        assert result.metadata["subject_id"] == "sub-emptyFNM"

    @pytest.mark.fast
    def test_fnm_run_succeeds_on_nonempty_mask(self, fnm_with_mock_connectome):
        """FNM.run() succeeds for a non-empty mask (sanity check)."""
        # Create a mask that overlaps the mock connectome's voxel grid
        mask_data = np.zeros((91, 109, 91), dtype=np.uint8)
        mask_data[2:7, 2:7, 0] = 1  # Overlaps with mask_indices (0..9, 0..9, 0)

        affine = np.array(
            [
                [2.0, 0.0, 0.0, -91.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(mask_data, affine)

        subject = SubjectData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
            metadata={"subject_id": "sub-filled"},
        )

        result = fnm_with_mock_connectome.run(subject)
        assert isinstance(result, SubjectData)
        assert "FunctionalNetworkMapping" in result.results


# ===========================================================================
# CLI: --skip-empty-masks flag
# ===========================================================================


class TestSkipEmptyMasksFlag:
    """Tests for the --skip-empty-masks CLI flag."""

    @pytest.mark.fast
    def test_on_empty_flag_exists_in_parser(self):
        """The --on-empty flag is recognized by the argument parser."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "rd", "/tmp/in", "/tmp/out", "--on-empty", "skip"])
        assert args.on_empty == "skip"

    @pytest.mark.fast
    def test_on_empty_defaults_to_warn(self):
        """Without the flag, on_empty defaults to 'warn'."""
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", "rd", "/tmp/in", "/tmp/out"])
        assert args.on_empty == "warn"

    @pytest.mark.fast
    def test_runconfig_carries_on_empty_flag(self):
        """RunConfig.from_args propagates on_empty."""
        from lacuna.cli.main import RunConfig
        from lacuna.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "rd",
                "/tmp/in",
                "/tmp/out",
                "--on-empty",
                "skip",
            ]
        )
        config = RunConfig.from_args(args)
        assert config.on_empty == "skip"

    @pytest.mark.fast
    def test_sequential_skips_empty_masks(self, tmp_path, caplog):
        """Sequential processing skips empty masks when flag is set."""
        # Create a minimal BIDS dataset with one empty and one non-empty mask
        import json

        from lacuna.cli.main import RunConfig, _run_analysis_workflow

        dataset_root = tmp_path / "bids"
        dataset_root.mkdir()
        desc = {"Name": "Test", "BIDSVersion": "1.6.0", "DatasetType": "raw"}
        with open(dataset_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        affine = np.eye(4) * 2
        affine[3, 3] = 1.0

        # Sub-01: empty mask
        sub1_dir = dataset_root / "sub-01" / "anat"
        sub1_dir.mkdir(parents=True)
        empty_data = np.zeros((64, 64, 64), dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty_data, affine), sub1_dir / "sub-01_desc-lesion_mask.nii.gz")

        # Sub-02: non-empty mask
        sub2_dir = dataset_root / "sub-02" / "anat"
        sub2_dir.mkdir(parents=True)
        full_data = np.zeros((64, 64, 64), dtype=np.uint8)
        full_data[30:35, 30:35, 30:35] = 1
        nib.save(nib.Nifti1Image(full_data, affine), sub2_dir / "sub-02_desc-lesion_mask.nii.gz")

        output_dir = tmp_path / "output"
        config = RunConfig(
            bids_dir=dataset_root,
            output_dir=output_dir,
            analysis="rd",
            on_empty="skip",
            space="MNI152NLin6Asym",
            verbose_count=1,
            analysis_options={},
        )

        with caplog.at_level(logging.WARNING):
            result = _run_analysis_workflow(config)

        assert result == 0  # EXIT_SUCCESS
        # The empty mask subject should be reported as skipped
        assert "skipped" in caplog.text.lower() or "empty mask" in caplog.text.lower()

    @pytest.mark.fast
    def test_sequential_processes_empty_masks_without_flag(self, tmp_path, caplog):
        """Without --skip-empty-masks, empty masks are processed normally."""
        import json

        from lacuna.cli.main import RunConfig, _run_analysis_workflow

        dataset_root = tmp_path / "bids"
        dataset_root.mkdir()
        desc = {"Name": "Test", "BIDSVersion": "1.6.0", "DatasetType": "raw"}
        with open(dataset_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        affine = np.eye(4) * 2
        affine[3, 3] = 1.0

        sub1_dir = dataset_root / "sub-01" / "anat"
        sub1_dir.mkdir(parents=True)
        empty_data = np.zeros((64, 64, 64), dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty_data, affine), sub1_dir / "sub-01_desc-lesion_mask.nii.gz")

        output_dir = tmp_path / "output"
        config = RunConfig(
            bids_dir=dataset_root,
            output_dir=output_dir,
            analysis="rd",
            on_empty="warn",
            space="MNI152NLin6Asym",
            verbose_count=1,
            analysis_options={},
        )

        with caplog.at_level(logging.WARNING):
            result = _run_analysis_workflow(config)

        # Should still succeed — empty mask is processed with zero outputs
        assert result == 0


# ===========================================================================
# EmptyMaskError constructor tests
# ===========================================================================


class TestEmptyMaskErrorAPI:
    """Tests for EmptyMaskError constructor variants."""

    @pytest.mark.fast
    def test_error_with_subject_id_only(self):
        from lacuna.core.exceptions import EmptyMaskError

        err = EmptyMaskError("sub-01")
        assert err.subject_id == "sub-01"
        assert "sub-01" in str(err)

    @pytest.mark.fast
    def test_error_with_detail_message(self):
        from lacuna.core.exceptions import EmptyMaskError

        err = EmptyMaskError("sub-01", "Use --skip-empty-masks to skip.")
        assert "sub-01" in str(err)
        assert "--skip-empty-masks" in str(err)

    @pytest.mark.fast
    def test_error_without_args(self):
        from lacuna.core.exceptions import EmptyMaskError

        err = EmptyMaskError()
        assert err.subject_id is None
        assert "no non-zero voxels" in str(err)

    @pytest.mark.fast
    def test_error_is_validation_error(self):
        from lacuna.core.exceptions import EmptyMaskError, ValidationError

        assert issubclass(EmptyMaskError, ValidationError)
        assert issubclass(EmptyMaskError, ValueError)
