"""
Contract tests for the analyze() API.

These tests verify that the analyze() function follows the API contract
defined in contracts/api.md.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestAnalyzeSignature:
    """Tests for analyze() function signature."""

    def test_analyze_exists_in_module(self):
        """Test that analyze is importable from lacuna."""
        from lacuna import analyze

        assert callable(analyze)

    def test_analyze_accepts_subject_data(self):
        """Test that analyze accepts a SubjectData object."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        params = sig.parameters

        assert "data" in params
        # data should be the first positional parameter

    def test_analyze_has_steps_param(self):
        """Test that analyze has steps keyword parameter (required)."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        params = sig.parameters

        assert "steps" in params
        # steps is keyword-only and required (no default)
        assert params["steps"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_analyze_has_log_level_param(self):
        """Test that analyze has log_level keyword parameter."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        params = sig.parameters

        assert "log_level" in params


class TestAnalyzeContractSteps:
    """Tests for analyze() steps parameter per new API."""

    def test_analyze_has_n_jobs_param(self):
        """Test that analyze has n_jobs keyword parameter."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        params = sig.parameters

        assert "n_jobs" in params
        assert params["n_jobs"].default == 1

    def test_analyze_has_show_progress_param(self):
        """Test that analyze has show_progress keyword parameter."""
        import inspect

        from lacuna import analyze

        sig = inspect.signature(analyze)
        params = sig.parameters

        assert "show_progress" in params
        assert params["show_progress"].default is True

    @pytest.mark.slow
    def test_analyze_steps_accepts_dict(self):
        """Test that steps accepts a dictionary."""
        import nibabel as nib

        from lacuna import analyze
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        subject = SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
        )

        # Should accept dict
        result = analyze(subject, steps={"RegionalDamage": None}, log_level=0)
        assert result is not None

    @pytest.mark.slow
    def test_analyze_steps_none_values_use_defaults(self):
        """Test that None values in steps use analysis defaults."""
        import nibabel as nib

        from lacuna import analyze
        from lacuna.core.subject_data import SubjectData

        mask_array = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_array[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(mask_array, np.eye(4))

        subject = SubjectData(
            mask_img=mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
        )

        result = analyze(subject, steps={"RegionalDamage": None}, log_level=0)
        assert "RegionalDamage" in result.results


class TestAnalyzeBasicBehavior:
    """Tests for analyze() basic behavior."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        import nibabel as nib

        from lacuna.core.subject_data import SubjectData

        # Create a simple mask
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
    def test_analyze_returns_subject_data(self, simple_subject):
        """Test that analyze returns SubjectData."""
        from lacuna import analyze
        from lacuna.core.subject_data import SubjectData

        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=0)

        assert isinstance(result, SubjectData)

    @pytest.mark.slow
    def test_analyze_adds_results(self, simple_subject):
        """Test that analyze adds results to SubjectData."""
        from lacuna import analyze

        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=0)

        # Should have at least RegionalDamage results
        assert result.results is not None
        assert len(result.results) > 0

    @pytest.mark.slow
    def test_analyze_accepts_list_input(self, simple_subject):
        """Test that analyze accepts list of SubjectData."""
        from lacuna import analyze
        from lacuna.core.subject_data import SubjectData

        results = analyze([simple_subject], steps={"RegionalDamage": None}, log_level=0)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SubjectData)


class TestAnalyzeReturnType:
    """Tests for analyze() return type consistency."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        import nibabel as nib

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
    def test_single_input_returns_single(self, simple_subject):
        """Test that single SubjectData input returns single SubjectData."""
        from lacuna import analyze
        from lacuna.core.subject_data import SubjectData

        result = analyze(simple_subject, steps={"RegionalDamage": None}, log_level=0)

        assert isinstance(result, SubjectData)
        assert not isinstance(result, list)

    @pytest.mark.slow
    def test_list_input_returns_list(self, simple_subject):
        """Test that list input returns list."""
        from lacuna import analyze

        results = analyze(
            [simple_subject, simple_subject],
            steps={"RegionalDamage": None},
            log_level=0,
        )

        assert isinstance(results, list)
        assert len(results) == 2


class TestAnalyzeValidation:
    """Tests for analyze() validation behavior."""

    @pytest.fixture
    def simple_subject(self):
        """Create a simple SubjectData for testing."""
        import nibabel as nib

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

    def test_analyze_empty_steps_raises(self, simple_subject):
        """Test that empty steps raises ValueError."""
        from lacuna import analyze

        with pytest.raises(ValueError, match="steps cannot be empty"):
            analyze(simple_subject, steps={}, log_level=0)

    def test_analyze_unknown_analysis_raises(self, simple_subject):
        """Test that unknown analysis name raises KeyError."""
        from lacuna import analyze

        with pytest.raises(KeyError, match="Unknown analysis"):
            analyze(simple_subject, steps={"FakeAnalysis": None}, log_level=0)

    def test_analyze_missing_steps_raises(self, simple_subject):
        """Test that missing steps parameter raises TypeError."""
        from lacuna import analyze

        with pytest.raises(TypeError):
            analyze(simple_subject, log_level=0)
