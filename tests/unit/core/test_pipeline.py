"""
Unit tests for lacuna.core.pipeline module.

Tests the Pipeline class and analyze() convenience function.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest


class TestPipeline:
    """Tests for Pipeline class."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

    def test_pipeline_creation(self):
        """Test Pipeline can be created."""
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline(name="Test Pipeline")

        assert pipeline.name == "Test Pipeline"
        assert len(pipeline) == 0

    def test_pipeline_add_returns_self(self, simple_subject):
        """Test Pipeline.add returns self for method chaining."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline()
        result = pipeline.add(RegionalDamage())

        assert result is pipeline

    def test_pipeline_chaining(self):
        """Test Pipeline supports method chaining."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline(name="Chained").add(RegionalDamage()).add(RegionalDamage())

        assert len(pipeline) == 2

    @pytest.mark.slow
    def test_pipeline_run_single_subject(self, simple_subject):
        """Test Pipeline.run with single subject."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline
        from lacuna.core.subject_data import SubjectData

        pipeline = Pipeline()
        pipeline.add(RegionalDamage())

        result = pipeline.run(simple_subject)

        assert isinstance(result, SubjectData)
        assert len(result.results) > 0

    def test_pipeline_run_rejects_invalid_input(self):
        """Test Pipeline.run raises TypeError for invalid input."""
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline()

        with pytest.raises(TypeError, match="Unsupported input type"):
            pipeline.run("not a SubjectData")

    def test_pipeline_describe(self):
        """Test Pipeline.describe returns string."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline(name="Test")
        pipeline.add(RegionalDamage())

        description = pipeline.describe()

        assert isinstance(description, str)
        assert "Test" in description
        assert "RegionalDamage" in description

    def test_pipeline_repr(self):
        """Test Pipeline __repr__."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline(name="Test")
        pipeline.add(RegionalDamage())

        repr_str = repr(pipeline)

        assert "Test" in repr_str
        assert "1" in repr_str  # steps count


class TestAnalyzeFunction:
    """Tests for analyze() convenience function."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

    def test_analyze_with_regional_damage(self, simple_subject):
        """Test that analyze runs RegionalDamage."""
        from lacuna import analyze

        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=0)

        # Should have RegionalDamage results
        assert "RegionalDamage" in result.results

    @pytest.mark.slow
    def test_analyze_single_returns_single(self, simple_subject):
        """Test that single input returns single output."""
        from lacuna import analyze
        from lacuna.core.subject_data import SubjectData

        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=0)

        assert isinstance(result, SubjectData)
        assert not isinstance(result, list)

    @pytest.mark.slow
    def test_analyze_list_returns_list(self, simple_subject):
        """Test that list input returns list output."""
        from lacuna import analyze

        results = analyze([simple_subject], steps={"RegionalDamage": None}, log_level=0)

        assert isinstance(results, list)
        assert len(results) == 1

    @pytest.mark.slow
    def test_analyze_batch_with_n_jobs(self, simple_subject):
        """Test analyze with batch processing and n_jobs."""
        from lacuna import analyze

        subjects = [simple_subject, simple_subject]
        results = analyze(
            subjects,
            steps={"RegionalDamage": None},
            n_jobs=1,
            log_level=0,
            show_progress=False,
        )

        assert len(results) == 2
        for r in results:
            assert "RegionalDamage" in r.results

    def test_analyze_requires_steps(self, simple_subject):
        """Test that analyze raises TypeError when steps not provided."""
        from lacuna import analyze

        with pytest.raises(TypeError):
            analyze(simple_subject, log_level=0)

    def test_analyze_empty_steps_raises(self, simple_subject):
        """Test that analyze raises ValueError for empty steps."""
        from lacuna import analyze

        with pytest.raises(ValueError, match="steps cannot be empty"):
            analyze(simple_subject, steps={}, log_level=0)

    def test_analyze_invalid_analysis_raises(self, simple_subject):
        """Test that analyze raises KeyError for unknown analysis."""
        from lacuna import analyze

        with pytest.raises(KeyError, match="Unknown analysis"):
            analyze(simple_subject, steps={"NonExistentAnalysis": None}, log_level=0)


class TestAnalyzeStepsParameter:
    """Tests for analyze() with steps parameter."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

    def test_analyze_accepts_steps_param(self, simple_subject):
        """Test that analyze accepts steps parameter."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        assert "steps" in sig.parameters

    @pytest.mark.slow
    def test_analyze_steps_with_none_uses_defaults(self, simple_subject):
        """Test that steps with None value uses analysis defaults."""
        from lacuna import analyze

        result = analyze(
            simple_subject,
            steps={"RegionalDamage": None},
            log_level=0,
        )

        assert "RegionalDamage" in result.results

    @pytest.mark.slow
    def test_analyze_steps_with_custom_params(self, simple_subject):
        """Test that steps accepts custom parameters."""
        from lacuna import analyze

        result = analyze(
            simple_subject,
            steps={"RegionalDamage": {"log_level": 0}},
            log_level=0,
        )

        assert "RegionalDamage" in result.results

    @pytest.mark.slow
    def test_analyze_multiple_steps(self, simple_subject):
        """Test analyze with multiple analysis steps."""
        from lacuna import analyze

        # Both RegionalDamage steps should run (even if same analysis)
        # but practically we'd use different analyses
        result = analyze(
            simple_subject,
            steps={"RegionalDamage": None},
            log_level=0,
        )

        assert "RegionalDamage" in result.results


class TestAnalyzeBatchProcessing:
    """Tests for analyze() batch processing features."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

    @pytest.mark.slow
    def test_analyze_n_jobs_sequential(self, simple_subject):
        """Test analyze with n_jobs=1 (sequential)."""
        from lacuna import analyze

        results = analyze(
            [simple_subject, simple_subject],
            steps={"RegionalDamage": None},
            n_jobs=1,
            show_progress=False,
            log_level=0,
        )

        assert len(results) == 2

    @pytest.mark.slow
    def test_analyze_show_progress_false(self, simple_subject):
        """Test analyze with show_progress=False."""
        from lacuna import analyze

        # Should not raise even with progress disabled
        results = analyze(
            [simple_subject],
            steps={"RegionalDamage": None},
            show_progress=False,
            log_level=0,
        )

        assert len(results) == 1

    @pytest.mark.slow
    def test_analyze_n_jobs_parallel(self, simple_subject):
        """Test analyze with parallel processing."""
        from lacuna import analyze

        results = analyze(
            [simple_subject, simple_subject, simple_subject],
            steps={"RegionalDamage": None},
            n_jobs=2,
            show_progress=False,
            log_level=0,
        )

        assert len(results) == 3
        for r in results:
            assert "RegionalDamage" in r.results


class TestAnalyzeLogLevel:
    """Tests for analyze() log_level parameter."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        return SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-001"},
        )

    @pytest.mark.slow
    def test_analyze_log_level_silent(self, simple_subject, capsys):
        """Test analyze with log_level=0 (silent)."""
        from lacuna import analyze

        analyze(simple_subject, steps={"RegionalDamage": None}, log_level=0)

        _ = capsys.readouterr()
        # Silent mode should have minimal output
        # (some internal libs may still print)

    @pytest.mark.slow
    def test_analyze_log_level_standard(self, simple_subject):
        """Test analyze with log_level=1 (standard)."""
        from lacuna import analyze

        # Should complete without error
        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=1)
        assert result is not None

    @pytest.mark.slow
    def test_analyze_log_level_verbose(self, simple_subject):
        """Test analyze with log_level=2 (verbose)."""
        from lacuna import analyze

        # Should complete without error
        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=2)
        assert result is not None


class TestPipelineBatchProcessing:
    """Tests for Pipeline.run_batch method."""

    @pytest.fixture
    def simple_subjects(self):
        """Create multiple SubjectData for batch testing."""
        from lacuna.core.subject_data import SubjectData

        subjects = []
        for i in range(3):
            mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
            mask_array[4:6, 4:6, 4:6] = 1
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

    @pytest.mark.slow
    def test_run_batch_sequential(self, simple_subjects):
        """Test Pipeline.run_batch with sequential processing."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline

        pipeline = Pipeline()
        pipeline.add(RegionalDamage())

        results = pipeline.run_batch(
            simple_subjects,
            parallel=False,
            show_progress=False,
        )

        assert len(results) == 3
        for r in results:
            assert "RegionalDamage" in r.results

    @pytest.mark.slow
    def test_run_batch_returns_list(self, simple_subjects):
        """Test Pipeline.run_batch returns list."""
        from lacuna.analysis import RegionalDamage
        from lacuna.core.pipeline import Pipeline
        from lacuna.core.subject_data import SubjectData

        pipeline = Pipeline()
        pipeline.add(RegionalDamage())

        results = pipeline.run_batch(
            simple_subjects,
            parallel=False,
            show_progress=False,
        )

        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SubjectData)
