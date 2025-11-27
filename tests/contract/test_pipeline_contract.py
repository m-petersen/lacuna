"""Contract tests for Pipeline API."""

import numpy as np
import nibabel as nib
import pytest

from lacuna import Pipeline, MaskData
from lacuna.analysis import RegionalDamage


class TestPipelineContract:
    """Contract tests for Pipeline class."""

    @pytest.fixture
    def sample_mask_data(self):
        """Create a minimal MaskData for testing."""
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)
        return MaskData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
            metadata={"subject_id": "sub-001"},
        )

    def test_pipeline_add_returns_self(self):
        """Contract: add() returns self for method chaining."""
        pipeline = Pipeline()
        result = pipeline.add(RegionalDamage())
        assert result is pipeline

    def test_pipeline_fluent_chaining(self):
        """Contract: Multiple add() calls can be chained."""
        pipeline = (
            Pipeline(name="Test Pipeline")
            .add(RegionalDamage())
            .add(RegionalDamage())  # Same analysis twice for testing
        )
        assert len(pipeline) == 2

    def test_pipeline_run_returns_mask_data(self, sample_mask_data):
        """Contract: run() returns MaskData with results."""
        pipeline = Pipeline().add(RegionalDamage())
        result = pipeline.run(sample_mask_data)
        assert isinstance(result, MaskData)
        assert "RegionalDamage" in result.results

    def test_pipeline_describe_returns_string(self):
        """Contract: describe() returns human-readable string."""
        pipeline = Pipeline(name="My Analysis").add(RegionalDamage())
        description = pipeline.describe()
        assert isinstance(description, str)
        assert "My Analysis" in description
        assert "RegionalDamage" in description

    def test_pipeline_len_returns_step_count(self):
        """Contract: len(pipeline) returns number of steps."""
        pipeline = Pipeline()
        assert len(pipeline) == 0
        pipeline.add(RegionalDamage())
        assert len(pipeline) == 1
        pipeline.add(RegionalDamage())
        assert len(pipeline) == 2


class TestPipelineConditionalContract:
    """Contract tests for conditional pipeline steps."""

    @pytest.fixture
    def sample_mask_data(self):
        """Create a minimal MaskData for testing."""
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)
        return MaskData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
            metadata={"subject_id": "sub-001"},
        )

    def test_conditional_step_skipped_when_false(self, sample_mask_data):
        """Contract: Step is skipped when condition returns False."""
        # Condition that always returns False
        pipeline = Pipeline().add(
            RegionalDamage(), condition=lambda x: False
        )
        result = pipeline.run(sample_mask_data)
        # RegionalDamage should not have run
        assert "RegionalDamage" not in result.results

    def test_conditional_step_runs_when_true(self, sample_mask_data):
        """Contract: Step runs when condition returns True."""
        # Condition that always returns True
        pipeline = Pipeline().add(
            RegionalDamage(), condition=lambda x: True
        )
        result = pipeline.run(sample_mask_data)
        assert "RegionalDamage" in result.results


class TestAnalyzeFunctionContract:
    """Contract tests for analyze() convenience function."""

    @pytest.fixture
    def sample_mask_data(self):
        """Create a minimal MaskData for testing."""
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)
        return MaskData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
            metadata={"subject_id": "sub-001"},
        )

    def test_analyze_single_returns_mask_data(self, sample_mask_data):
        """Contract: analyze(single) returns MaskData."""
        from lacuna import analyze

        result = analyze(sample_mask_data)
        assert isinstance(result, MaskData)

    def test_analyze_list_returns_list(self, sample_mask_data):
        """Contract: analyze(list) returns list of MaskData."""
        from lacuna import analyze

        results = analyze([sample_mask_data, sample_mask_data])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, MaskData) for r in results)
