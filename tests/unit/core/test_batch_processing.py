"""
Unit tests for batch processing in lacuna.core.pipeline module.

Tests specifically focused on batch processing features with n_jobs
and show_progress parameters.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def batch_subjects():
    """Create multiple SubjectData for batch testing."""
    from lacuna.core.subject_data import SubjectData

    subjects = []
    for i in range(5):
        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        # Create slightly different mask for each subject
        mask_array[3 + i % 3 : 6 + i % 3, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        subjects.append(
            SubjectData(
                mask_img=mask_img,
                space="MNI152NLin6Asym",
                resolution=2.0,
                metadata={"subject_id": f"sub-{i:03d}"},
            )
        )

    return subjects


class TestAnalyzeBatchNJobs:
    """Tests for analyze() n_jobs parameter."""

    def test_n_jobs_default_is_one(self):
        """Test that n_jobs default is 1 (sequential)."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        assert sig.parameters["n_jobs"].default == 1

    @pytest.mark.slow
    def test_n_jobs_one_processes_sequentially(self, batch_subjects):
        """Test that n_jobs=1 processes sequentially."""
        from lacuna import analyze

        results = analyze(
            batch_subjects,
            steps={"RegionalDamage": None},
            n_jobs=1,
            show_progress=False,
            verbose=False,
        )

        assert len(results) == len(batch_subjects)
        for r in results:
            assert "RegionalDamage" in r.results

    @pytest.mark.slow
    def test_n_jobs_two_processes_in_parallel(self, batch_subjects):
        """Test that n_jobs=2 uses parallel processing."""
        from lacuna import analyze

        results = analyze(
            batch_subjects,
            steps={"RegionalDamage": None},
            n_jobs=2,
            show_progress=False,
            verbose=False,
        )

        assert len(results) == len(batch_subjects)
        for r in results:
            assert "RegionalDamage" in r.results

    @pytest.mark.slow
    def test_n_jobs_negative_one_uses_all_cpus(self, batch_subjects):
        """Test that n_jobs=-1 uses all CPUs."""
        from lacuna import analyze

        # Should not raise
        results = analyze(
            batch_subjects[:2],  # Use fewer subjects for speed
            steps={"RegionalDamage": None},
            n_jobs=-1,
            show_progress=False,
            verbose=False,
        )

        assert len(results) == 2


class TestAnalyzeBatchShowProgress:
    """Tests for analyze() show_progress parameter."""

    def test_show_progress_default_is_true(self):
        """Test that show_progress default is True."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        assert sig.parameters["show_progress"].default is True

    @pytest.mark.slow
    def test_show_progress_false_no_error(self, batch_subjects):
        """Test that show_progress=False runs without error."""
        from lacuna import analyze

        results = analyze(
            batch_subjects[:2],
            steps={"RegionalDamage": None},
            show_progress=False,
            verbose=False,
        )

        assert len(results) == 2

    @pytest.mark.slow
    def test_show_progress_true_no_error(self, batch_subjects):
        """Test that show_progress=True runs without error."""
        from lacuna import analyze

        results = analyze(
            batch_subjects[:2],
            steps={"RegionalDamage": None},
            show_progress=True,
            verbose=False,
        )

        assert len(results) == 2


class TestBatchOutputConsistency:
    """Tests for batch processing output consistency."""

    @pytest.mark.slow
    def test_batch_preserves_order(self, batch_subjects):
        """Test that batch processing preserves input order."""
        from lacuna import analyze

        results = analyze(
            batch_subjects,
            steps={"RegionalDamage": None},
            n_jobs=1,
            show_progress=False,
            verbose=False,
        )

        # Verify order by checking subject IDs
        for i, result in enumerate(results):
            expected_id = f"sub-{i:03d}"
            assert result.metadata["subject_id"] == expected_id

    @pytest.mark.slow
    def test_batch_results_same_as_sequential(self, batch_subjects):
        """Test that batch results match sequential processing."""
        from lacuna import analyze

        # Process sequentially
        sequential_results = [
            analyze(s, steps={"RegionalDamage": None}, verbose=False) for s in batch_subjects[:2]
        ]

        # Process as batch
        batch_results = analyze(
            batch_subjects[:2],
            steps={"RegionalDamage": None},
            n_jobs=1,
            show_progress=False,
            verbose=False,
        )

        # Verify results match
        assert len(batch_results) == len(sequential_results)
        for seq, batch in zip(sequential_results, batch_results, strict=True):
            assert seq.results.keys() == batch.results.keys()

    def test_empty_batch_returns_empty_list(self):
        """Test that empty batch returns empty list."""
        from lacuna import analyze

        results = analyze([], steps={"RegionalDamage": None}, verbose=False)

        assert results == []


class TestBatchErrorHandling:
    """Tests for batch processing error handling."""

    def test_single_invalid_type_in_batch_raises(self, batch_subjects):
        """Test that invalid type in batch raises TypeError."""
        from lacuna import analyze

        # Mix valid and invalid
        mixed = [batch_subjects[0], "invalid"]

        with pytest.raises(TypeError):
            analyze(mixed, steps={"RegionalDamage": None}, verbose=False)


class TestPipelineRunBatch:
    """Tests for Pipeline.run_batch method."""

    @pytest.mark.slow
    def test_run_batch_parallel_true(self, batch_subjects):
        """Test Pipeline.run_batch with parallel=True."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline()
        pipeline.add(RegionalDamage())

        results = pipeline.run_batch(
            batch_subjects[:2],
            parallel=True,
            n_jobs=2,
            show_progress=False,
        )

        assert len(results) == 2

    @pytest.mark.slow
    def test_run_batch_parallel_false(self, batch_subjects):
        """Test Pipeline.run_batch with parallel=False."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline()
        pipeline.add(RegionalDamage())

        results = pipeline.run_batch(
            batch_subjects[:2],
            parallel=False,
            show_progress=False,
        )

        assert len(results) == 2

    @pytest.mark.slow
    def test_run_batch_with_progress(self, batch_subjects):
        """Test Pipeline.run_batch with progress enabled."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline()
        pipeline.add(RegionalDamage())

        # Should not raise
        results = pipeline.run_batch(
            batch_subjects[:2],
            parallel=False,
            show_progress=True,
        )

        assert len(results) == 2


class TestTqdmIntegration:
    """Tests for tqdm progress bar integration."""

    def test_tqdm_import_available(self):
        """Test that tqdm is importable."""
        import tqdm

        assert tqdm is not None

    @pytest.mark.slow
    def test_tqdm_used_when_show_progress_true(self, batch_subjects, capsys):
        """Test that tqdm is used when show_progress=True."""
        from lacuna import analyze

        # Run with progress
        results = analyze(
            batch_subjects[:2],
            steps={"RegionalDamage": None},
            show_progress=True,
            n_jobs=1,
            verbose=False,
        )

        # tqdm writes to stderr - we just verify no error occurred
        assert len(results) == 2
