"""
Contract tests for the Pipeline API.

These tests verify that the Pipeline class follows the API contract
defined in contracts/api.md.
"""

from __future__ import annotations

import inspect


class TestPipelineSignature:
    """Tests for Pipeline class signature."""

    def test_pipeline_exists_in_module(self):
        """Test that Pipeline is importable from lacuna."""
        from lacuna import Pipeline

        assert Pipeline is not None

    def test_pipeline_has_name_parameter(self):
        """Test that Pipeline accepts name parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.__init__)
        params = sig.parameters

        assert "name" in params
        assert params["name"].default is None

    def test_pipeline_has_description_parameter(self):
        """Test that Pipeline accepts description parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.__init__)
        params = sig.parameters

        assert "description" in params
        assert params["description"].default is None


class TestPipelineAddMethod:
    """Tests for Pipeline.add method signature."""

    def test_add_method_exists(self):
        """Test that Pipeline has add method."""
        from lacuna import Pipeline

        pipeline = Pipeline()
        assert hasattr(pipeline, "add")
        assert callable(pipeline.add)

    def test_add_accepts_analysis(self):
        """Test that add accepts analysis parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.add)
        params = sig.parameters

        assert "analysis" in params

    def test_add_accepts_name(self):
        """Test that add accepts optional name parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.add)
        params = sig.parameters

        assert "name" in params
        assert params["name"].default is None

    def test_add_returns_self(self):
        """Test that add returns self for chaining."""
        from lacuna import Pipeline
        from lacuna.analysis import RegionalDamage

        pipeline = Pipeline()
        result = pipeline.add(RegionalDamage())

        assert result is pipeline


class TestPipelineRunMethod:
    """Tests for Pipeline.run method signature."""

    def test_run_method_exists(self):
        """Test that Pipeline has run method."""
        from lacuna import Pipeline

        pipeline = Pipeline()
        assert hasattr(pipeline, "run")
        assert callable(pipeline.run)

    def test_run_accepts_data(self):
        """Test that run accepts data parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.run)
        params = sig.parameters

        assert "data" in params

    def test_run_accepts_log_level(self):
        """Test that run accepts log_level parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.run)
        params = sig.parameters

        assert "log_level" in params
        assert params["log_level"].default == 1


class TestPipelineRunBatchMethod:
    """Tests for Pipeline.run_batch method signature."""

    def test_run_batch_method_exists(self):
        """Test that Pipeline has run_batch method."""
        from lacuna import Pipeline

        pipeline = Pipeline()
        assert hasattr(pipeline, "run_batch")
        assert callable(pipeline.run_batch)

    def test_run_batch_accepts_data_list(self):
        """Test that run_batch accepts data_list parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.run_batch)
        params = sig.parameters

        assert "data_list" in params

    def test_run_batch_accepts_n_jobs(self):
        """Test that run_batch accepts n_jobs parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.run_batch)
        params = sig.parameters

        assert "n_jobs" in params
        assert params["n_jobs"].default == -1

    def test_run_batch_accepts_show_progress(self):
        """Test that run_batch accepts show_progress parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.run_batch)
        params = sig.parameters

        assert "show_progress" in params
        assert params["show_progress"].default is True

    def test_run_batch_accepts_parallel(self):
        """Test that run_batch accepts parallel parameter."""
        from lacuna import Pipeline

        sig = inspect.signature(Pipeline.run_batch)
        params = sig.parameters

        assert "parallel" in params
        assert params["parallel"].default is True


class TestPipelineDescribeMethod:
    """Tests for Pipeline.describe method."""

    def test_describe_method_exists(self):
        """Test that Pipeline has describe method."""
        from lacuna import Pipeline

        pipeline = Pipeline()
        assert hasattr(pipeline, "describe")
        assert callable(pipeline.describe)

    def test_describe_returns_string(self):
        """Test that describe returns a string."""
        from lacuna import Pipeline

        pipeline = Pipeline(name="Test")
        result = pipeline.describe()

        assert isinstance(result, str)
        assert "Test" in result


class TestPipelineProtocol:
    """Tests for Pipeline protocol (len, repr)."""

    def test_pipeline_has_len(self):
        """Test that Pipeline supports len()."""
        from lacuna import Pipeline
        from lacuna.analysis import RegionalDamage

        pipeline = Pipeline()
        assert len(pipeline) == 0

        pipeline.add(RegionalDamage())
        assert len(pipeline) == 1

    def test_pipeline_has_repr(self):
        """Test that Pipeline has repr."""
        from lacuna import Pipeline

        pipeline = Pipeline(name="Test")
        repr_str = repr(pipeline)

        assert isinstance(repr_str, str)
        assert "Test" in repr_str
